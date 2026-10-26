import atexit
import errno
import json
from pathlib import Path
import posixpath
import queue
import subprocess
import threading
import time

import paramiko

from acquire_buildability_protection import utc_now


_pools = {}
_clients = set()
_lock = threading.Lock()


def close_connections():
    with _lock:
        clients = list(_clients)
        _clients.clear()
    for client in clients:
        client.close()


atexit.register(close_connections)


def _pool(server):
    with _lock:
        if server not in _pools:
            _pools[server] = queue.LifoQueue(maxsize=8)
        return _pools[server]


def _discard(client):
    if client is not None:
        with _lock:
            _clients.discard(client)
        client.close()


def _connect(server):
    config = paramiko.SSHConfig()
    with (Path.home() / '.ssh/config').open(encoding='utf-8') as stream:
        config.parse(stream)
    settings = config.lookup(server)
    if settings.get('proxycommand') or settings.get('proxyjump'):
        raise ValueError('本机中转的持久连接要求使用已登记的直连SSH地址')
    client = paramiko.SSHClient()
    client.load_host_keys(str(Path.home() / '.ssh/known_hosts'))
    connected = False
    try:
        client.connect(settings['hostname'], port=int(settings.get('port', 22)), username=settings['user'],
                       key_filename=settings.get('identityfile'), timeout=20, banner_timeout=20, auth_timeout=20)
        client.get_transport().set_keepalive(20)
        connected = True
    finally:
        if not connected:
            client.close()
    with _lock:
        _clients.add(client)
    return client


def _borrow(server):
    pool = _pool(server)
    while True:
        try:
            client = pool.get_nowait()
        except queue.Empty:
            return _connect(server)
        transport = client.get_transport()
        if transport is not None and transport.is_active() and transport.is_authenticated():
            return client
        _discard(client)


def _release(server, client):
    try:
        _pool(server).put_nowait(client)
    except queue.Full:
        _discard(client)


def _network_failure(error):
    if isinstance(error, (paramiko.AuthenticationException, paramiko.BadHostKeyException)):
        return False
    if isinstance(error, (EOFError, ConnectionError, TimeoutError, paramiko.ssh_exception.NoValidConnectionsError)):
        return True
    if isinstance(error, OSError) and error.errno in {errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED,
                                                    errno.ETIMEDOUT, errno.ECONNREFUSED, 10053, 10054, 10060, 10061}:
        return True
    return isinstance(error, paramiko.SSHException) and any(text in str(error).lower() for text in
           ('ssh protocol banner', 'ssh session not active', 'socket is closed', 'no existing session',
            'server connection dropped', 'eof during negotiation', 'connection reset', 'connection closed'))


def _run(server, command, files=(), remote_directory=None):
    for attempt in range(1, 9):
        client = None
        try:
            client = _borrow(server)
            if files:
                with client.open_sftp() as sftp:
                    for path in files:
                        path = Path(path)
                        sftp.put(str(path), posixpath.join(remote_directory, path.name), confirm=True)
            stdin, stdout, stderr = client.exec_command(command, timeout=900)
            stdin.close()
            output, error_text = stdout.read().decode('utf-8'), stderr.read().decode('utf-8')
            status = stdout.channel.recv_exit_status()
            if status < 0:
                raise EOFError('SSH连接关闭，远端退出状态尚未确认')
        except (EOFError, paramiko.SSHException, OSError) as error:
            _discard(client)
            if not _network_failure(error) or attempt == 8:
                raise
            print(json.dumps({'event': 'ssh_reconnect', 'attempt': attempt, 'error_type': type(error).__name__, 'utc': utc_now()}), flush=True)
            time.sleep(min(2 ** attempt, 30))
            continue
        _release(server, client)
        if status:
            raise RuntimeError(f'服务器操作失败，退出码{status}：{error_text[-1500:]}')
        return subprocess.CompletedProcess(['ssh', server, command], 0, output, error_text)


def remote_command(server, command):
    return _run(server, command)


def relay_files(server, files, remote_directory, command):
    # 连接中断时重传整个事务，服务器继续逐对象验证原始字节。
    return _run(server, command, files, remote_directory)
