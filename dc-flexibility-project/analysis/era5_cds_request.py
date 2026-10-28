from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import random
import re
import socket
import ssl

import requests

from acquire_buildability_protection import utc_now, write_json


MAX_RESUBMISSIONS = 5
QUEUE_WAIT_INITIAL_SECONDS = 300
QUEUE_WAIT_MAX_SECONDS = 3600
QUEUE_WAIT_LIMIT_SECONDS = 72 * 3600
NETWORK_WAIT_INITIAL_SECONDS = 60
NETWORK_WAIT_MAX_SECONDS = 30 * 60
NETWORK_WAIT_LIMIT_SECONDS = 24 * 3600
DATASET = 'reanalysis-era5-single-levels'
QUEUE_FULL_REASON = ('Maximum number queued requests for this dataset has been reached. '
                     'Please configure your scripts to request data sequentially')


def queue_full_reason(reason):
    return isinstance(reason, dict) and reason.get('traceback') == QUEUE_FULL_REASON


def queue_delay(attempt):
    base = min(QUEUE_WAIT_INITIAL_SECONDS * 2 ** min(max(attempt - 1, 0), 4), QUEUE_WAIT_MAX_SECONDS)
    return min(base * random.uniform(0.9, 1.1), QUEUE_WAIT_MAX_SECONDS)


def network_error_kind(error):
    if isinstance(error, requests.HTTPError):
        response = error.response
        if response is not None and (response.status_code == 429 or 500 <= response.status_code < 600):
            return f'http_{response.status_code}'
        return None
    if isinstance(error, (ssl.SSLError, requests.exceptions.SSLError)):
        return 'ssl'
    if isinstance(error, (socket.gaierror, requests.ConnectionError)):
        return 'dns' if isinstance(error, socket.gaierror) or any(text in str(error).lower() for text in (
            'name resolution', 'name or service not known', 'getaddrinfo failed')) else 'connection'
    if isinstance(error, (requests.exceptions.ChunkedEncodingError, requests.Timeout)):
        return type(error).__name__
    return None


def network_delay(attempt):
    base = min(NETWORK_WAIT_INITIAL_SECONDS * 2 ** min(max(attempt - 1, 0), 5), NETWORK_WAIT_MAX_SECONDS)
    return min(base * random.uniform(0.9, 1.1), NETWORK_WAIT_MAX_SECONDS)


NETWORK_ERRORS = (requests.RequestException, socket.gaierror, ssl.SSLError)


def dataset_jobs(client, page_size=100, status=None):
    jobs = {}
    page = client.get_jobs(limit=page_size, **({'status': status} if status else {}))
    seen = set()
    while page is not None:
        reply = page.json
        url = next(link['href'] for link in reply['links'] if link['rel'] == 'self')
        if url in seen:
            raise RuntimeError('CDS账号队列分页重复，停止提交')
        seen.add(url)
        for job in reply['jobs']:
            if job['processID'] == DATASET:
                jobs[job['jobID']] = job
        page = page.next
    return jobs


def active_dataset_jobs(client, page_size=100):
    return {request_id: job['status'] for request_id, job in dataset_jobs(
        client, page_size, ['accepted', 'running']).items() if job['status'] in ('accepted', 'running')}


class CDSRequestGate:
    def __init__(self, root, client):
        self.directory = Path(root) / '.era5-cds-request-gate'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.client = client

    @contextmanager
    def locked(self):
        # 每次检查队列、提交和保存编号使用同一把跨进程锁。
        with (self.directory / 'submission.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def snapshot(self):
        config_path = self.directory / 'config.json'
        if not config_path.exists():
            write_json(config_path, {'max_active_requests': int(os.environ.get('ERA5_CDS_MAX_ACTIVE_REQUESTS', '2'))})
        limit = json.loads(config_path.read_text())['max_active_requests']
        if type(limit) is not int or limit < 1:
            raise ValueError('ERA5全局并发名额必须为正整数')
        active = active_dataset_jobs(self.client)
        # 新编号即使尚未出现在列表中，也必须计入名额。
        for path in self.directory.glob('submitted-*.json'):
            token = json.loads(path.read_text())
            request_id = token['request_id']
            if request_id in active:
                continue
            status = saved_job_status(self.client, request_id)
            if status in ('accepted', 'running'):
                active[request_id] = status
            elif status in ('successful', 'rejected', 'failed', 'dismissed'):
                path.unlink()
            else:
                raise RuntimeError(f'CDS名额状态异常：{request_id}/{status}')
        result = {'checked_utc': utc_now(), 'max_active_requests': limit,
                  'active_requests': active, 'active_count': len(active)}
        write_json(self.directory / 'status.json', result)
        return result

    def remember(self, request_id, request_path):
        write_json(self.directory / f'submitted-{request_id}.json',
                   {'request_id': request_id, 'request_path': str(request_path), 'submitted_utc': utc_now()})


def dismissed_reason(error):
    response = error.response
    if response is None or response.status_code != 404:
        return None
    if 'json' not in response.headers.get('Content-Type', '').lower():
        return None
    reason = response.json()
    message = '\n'.join(str(reason.get(key, '')) for key in ('title', 'detail', 'traceback')).lower()
    if re.search(r'\bjob not found\b|\bjob (?:[0-9a-f-]+ )?dismissed\b', message):
        return reason
    return None


def saved_job_status(client, request_id):
    try:
        return client.get_remote(request_id).status
    except requests.HTTPError as error:
        if dismissed_reason(error) is None:
            raise
        return 'dismissed'


class PersistentCDSRequest:
    def __init__(self, client, dataset, request, path, update, stopped, known_id=None, *, gate):
        self.client = client
        self.dataset = dataset
        self.request = request
        self.path = path
        self.update = update
        self.stopped = stopped
        self.gate = gate
        if dataset != DATASET:
            raise ValueError('全局名额闸门仅用于ERA5单层补取')
        self.remote = None
        if path.exists():
            self.saved = json.loads(path.read_text())
            if self.saved['dataset'] != dataset or self.saved['request'] != request:
                raise ValueError('CDS持久请求与当前合格区域请求不同')
        else:
            self.saved = {'dataset': dataset, 'request': request,
                          'request_id': known_id, 'created_utc': utc_now(), 'retry_count': 0, 'history': [],
                          'pending_submission': not bool(known_id)}
            write_json(path, self.saved)
        self.migrate_queue_budget()

    def wait_for_network(self):
        next_retry = self.saved.get('network_next_retry_utc')
        while next_retry:
            remaining = (datetime.fromisoformat(next_retry) - datetime.now(timezone.utc)).total_seconds()
            if remaining <= 0:
                break
            if self.stopped.wait(min(remaining, 30)):
                raise RuntimeError('其他补取请求失败，停止网络等待；现有CDS编号已保留')

    def network_wait(self, operation, error, kind=None):
        used = self.saved.get('network_wait_seconds', 0.0)
        if used >= NETWORK_WAIT_LIMIT_SECONDS:
            self.saved['network_wait_exhausted'] = True
            write_json(self.path, self.saved)
            self.notify('network_wait_exhausted')
            print(json.dumps({'event': 'cds_network_wait_exhausted', 'request_path': str(self.path),
                              'request_id': self.saved['request_id'], 'operation': operation,
                              'network_wait_seconds': used, 'utc': utc_now()}), flush=True)
            raise RuntimeError(f'CDS网络故障累计等待达到24小时：{self.path}') from error
        attempt = self.saved.get('network_consecutive_failures', 0) + 1
        delay = min(network_delay(attempt), NETWORK_WAIT_LIMIT_SECONDS - used)
        event = {'event': 'cds_network_wait', 'utc': utc_now(), 'operation': operation,
                 'request_id': self.saved['request_id'], 'error_type': type(error).__name__,
                 'error_kind': kind or network_error_kind(error), 'attempt': attempt, 'delay_seconds': delay,
                 'next_retry_utc': (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat(),
                 'network_wait_seconds': used + delay}
        # 先持久保存等待额度和截止时间，进程重启后继续剩余等待。
        self.saved.update(network_wait_seconds=used + delay, network_consecutive_failures=attempt,
                          network_next_retry_utc=event['next_retry_utc'], network_wait_exhausted=False)
        self.saved.setdefault('network_wait_events', []).append(event)
        write_json(self.path, self.saved)
        self.notify('waiting_for_network')
        print(json.dumps({'request_path': str(self.path), **event}), flush=True)
        self.wait_for_network()

    def network_recovered(self):
        if self.saved.get('network_next_retry_utc'):
            self.saved.pop('network_next_retry_utc', None)
            self.saved['network_consecutive_failures'] = 0
            write_json(self.path, self.saved)

    def network_call(self, operation, call, recovered=True):
        self.wait_for_network()
        while True:
            if self.stopped.is_set():
                raise RuntimeError('其他补取请求失败；现有CDS请求编号已保留')
            try:
                result = call()
            except NETWORK_ERRORS as error:
                if network_error_kind(error) is None:
                    raise
                self.network_wait(operation, error)
            else:
                if recovered:
                    self.network_recovered()
                return result

    def migrate_queue_budget(self):
        if self.saved.get('retry_budget_version') == 2:
            return
        history = self.saved.get('history', [])
        queue_entries = [entry for entry in history if queue_full_reason(entry['reason'])]
        pending_queue = bool(self.saved.get('pending_resubmission') and history and queue_full_reason(history[-1]['reason']))
        charged_queue_retries = len(queue_entries) - int(pending_queue)
        self.saved['legacy_retry_count'] = self.saved.get('retry_count', 0)
        self.saved['retry_count'] = max(0, self.saved.get('retry_count', 0) - charged_queue_retries)
        self.saved['retry_budget_version'] = 2
        self.saved.setdefault('queue_wait_attempts', len(queue_entries))
        self.saved.setdefault('queue_wait_seconds', 0.0)
        if pending_queue:
            self.saved.update(retry_exhausted=False, pending_retry_kind='queue')
            self.begin_queue_wait(queue_entries[0]['retired_utc'])
        write_json(self.path, self.saved)

    def begin_queue_wait(self, started_utc=None):
        if not self.saved.get('queue_wait_started_utc'):
            started = datetime.fromisoformat(started_utc) if started_utc else datetime.now(timezone.utc)
            remaining = max(0, QUEUE_WAIT_LIMIT_SECONDS - self.saved.get('queue_wait_seconds', 0.0))
            self.saved.update(queue_wait_started_utc=started.isoformat(),
                              queue_wait_deadline_utc=(started + timedelta(seconds=remaining)).isoformat())

    def end_queue_wait(self):
        started = self.saved.pop('queue_wait_started_utc', None)
        if started:
            self.saved['queue_wait_seconds'] = self.saved.get('queue_wait_seconds', 0.0) + max(
                0, (datetime.now(timezone.utc) - datetime.fromisoformat(started)).total_seconds())
        self.saved.pop('queue_wait_deadline_utc', None)
        self.saved.pop('queue_wait_exhausted', None)

    def check_queue_deadline(self):
        deadline = self.saved.get('queue_wait_deadline_utc')
        if deadline and datetime.now(timezone.utc) >= datetime.fromisoformat(deadline):
            self.saved['queue_wait_exhausted'] = True
            write_json(self.path, self.saved)
            self.notify('queue_wait_exhausted')
            raise RuntimeError(f'CDS队列名额累计等待达到72小时：{self.path}')

    def schedule_queue_wait(self, **details):
        self.begin_queue_wait()
        self.check_queue_deadline()
        attempt = self.saved.get('queue_wait_attempts', 0) + 1
        self.saved.update(queue_wait_attempts=attempt,
                          next_retry_utc=(datetime.now(timezone.utc) + timedelta(seconds=queue_delay(attempt))).isoformat())
        write_json(self.path, self.saved)
        self.notify('waiting_for_queue_slot', **details)
        print(json.dumps({'event': 'cds_waiting_for_queue_slot', 'request_path': str(self.path),
                          'next_retry_utc': self.saved['next_retry_utc'],
                          'queue_wait_deadline_utc': self.saved['queue_wait_deadline_utc'], **details}), flush=True)

    def wait_until_retry(self):
        next_retry = self.saved.get('next_retry_utc')
        if not next_retry:
            self.check_queue_deadline()
            return
        remaining = max(0, (datetime.fromisoformat(next_retry) -
                            datetime.now(timezone.utc)).total_seconds())
        # 小段等待可响应其他请求失败，并在72小时期限到达时立即结束。
        while remaining > 0:
            self.check_queue_deadline()
            deadline = self.saved.get('queue_wait_deadline_utc')
            step = min(remaining, 30)
            if deadline:
                step = min(step, max(0, (datetime.fromisoformat(deadline) - datetime.now(timezone.utc)).total_seconds()))
            if self.stopped.wait(step):
                raise RuntimeError('其他补取请求失败，停止等待；旧编号已保留')
            remaining = max(0, (datetime.fromisoformat(next_retry) -
                                datetime.now(timezone.utc)).total_seconds())
        self.check_queue_deadline()

    def notify(self, status, **details):
        details.setdefault('next_retry_utc', self.saved.get('next_retry_utc'))
        self.update(status, self.saved['request_id'], history=list(self.saved.get('history', [])),
                    retry_count=self.saved.get('retry_count', 0), retry_limit=MAX_RESUBMISSIONS,
                    queue_wait_attempts=self.saved.get('queue_wait_attempts', 0),
                    queue_wait_seconds=self.saved.get('queue_wait_seconds', 0.0),
                    queue_wait_deadline_utc=self.saved.get('queue_wait_deadline_utc'),
                    network_wait_seconds=self.saved.get('network_wait_seconds', 0.0),
                    network_wait_limit_seconds=NETWORK_WAIT_LIMIT_SECONDS,
                    network_next_retry_utc=self.saved.get('network_next_retry_utc'),
                    network_wait_exhausted=self.saved.get('network_wait_exhausted', False),
                    network_wait_events=list(self.saved.get('network_wait_events', [])), **details)

    def recover_submission(self):
        attempt = self.saved['submission_uncertain']
        # POST返回丢失时只读取账号队列寻找新编号，避免重复提交同一份数据。
        while True:
            jobs = self.network_call('recover_submission_list', lambda: dataset_jobs(self.client), recovered=False)
            candidates = []
            for request_id in jobs.keys() - set(attempt['known_ids']):
                def read_reply():
                    return self.client.get_remote(request_id).json
                try:
                    reply = self.network_call('recover_submission_status', read_reply, recovered=False)
                except requests.HTTPError as error:
                    if dismissed_reason(error) is None:
                        raise
                    continue
                if reply['metadata']['request']['ids'] == self.request and reply['processID'] == self.dataset:
                    candidates.append(request_id)
            if len(candidates) > 1:
                raise RuntimeError(f'CDS提交结果存在多个匹配编号，需要核查：{candidates}')
            if candidates:
                return self.network_call('recover_submission_remote', lambda: self.client.get_remote(candidates[0]))
            self.network_wait('recover_submission', RuntimeError('CDS尚未返回唯一提交编号'),
                              kind='submission_outcome_unknown')

    def submit_once(self):
        if self.saved.get('submission_uncertain'):
            return self.recover_submission()
        process = self.network_call('get_process', lambda: self.client.get_process(self.dataset))
        jobs = self.network_call('before_submission_list', lambda: dataset_jobs(self.client))
        self.saved['submission_uncertain'] = {'started_utc': utc_now(), 'known_ids': list(jobs)}
        write_json(self.path, self.saved)
        try:
            return process.submit(self.request)
        except NETWORK_ERRORS as error:
            response = error.response if isinstance(error, requests.HTTPError) else None
            queue_rejection = (response is not None and response.status_code in (400, 429)
                               and 'json' in response.headers.get('Content-Type', '').lower()
                               and queue_full_reason(response.json()))
            if queue_rejection or network_error_kind(error) is None:
                self.saved.pop('submission_uncertain')
                write_json(self.path, self.saved)
                raise
            self.network_wait('submit', error)
            return self.recover_submission()

    def download(self, temporary):
        temporary = Path(temporary)
        def download_from_start():
            temporary.unlink(missing_ok=True)
            results = self.remote.get_results()
            # 直接检查HTTP状态，完整重下；SDK内部下载器会吞掉部分HTTP错误。
            with requests.get(results.location, stream=True, timeout=self.client.timeout,
                              verify=self.client.verify) as response:
                response.raise_for_status()
                with temporary.open('wb') as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        stream.write(chunk)
                length = response.headers.get('Content-Length')
                if (length is not None and response.headers.get('Content-Encoding', 'identity') == 'identity'
                        and temporary.stat().st_size < int(length)):
                    raise requests.exceptions.ChunkedEncodingError('CDS下载连接提前结束，未收到HTTP声明的全部字节')
            if temporary.stat().st_size != results.content_length:
                raise ValueError(f'CDS原文下载大小不符：{temporary.stat().st_size}/{results.content_length}')
            return str(temporary)
        try:
            return self.network_call('download', download_from_start)
        finally:
            # 只有成功完成下载的文件才交由既有SHA与科学校验流程处理。
            if self.saved.get('network_next_retry_utc') or self.stopped.is_set():
                temporary.unlink(missing_ok=True)

    def retire(self, status, reason):
        history = self.saved.setdefault('history', [])
        retry_count = self.saved.get('retry_count', 0)
        entry = {'request_id': self.saved['request_id'], 'status': status, 'reason': reason,
                 'retired_utc': utc_now(), 'retry_count': retry_count}
        history.append(entry)
        queue_full = queue_full_reason(reason)
        exhausted = not queue_full and retry_count >= MAX_RESUBMISSIONS
        delay = min(60 * 2 ** retry_count, 900)
        self.saved.update(pending_resubmission=True, retry_exhausted=exhausted,
                          pending_retry_kind='queue' if queue_full else 'error',
                          next_retry_utc=(datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat())
        if queue_full:
            self.schedule_queue_wait()
        archived = self.path.with_name(self.path.stem + f'.retired-{self.saved["request_id"]}.json')
        write_json(archived, self.saved)
        write_json(self.path, self.saved)
        self.notify('retry_exhausted' if exhausted else (
            'waiting_for_queue_slot' if status in ('rejected', 'failed') else 'waiting_to_resubmit'),
            next_retry_utc=self.saved['next_retry_utc'])
        print(json.dumps({'event': 'cds_request_retired', 'request_path': str(self.path), **entry,
                          'retry_exhausted': exhausted, 'next_retry_utc': self.saved['next_retry_utc']}), flush=True)

    def resubmit(self):
        retry_error = bool(self.saved.get('pending_resubmission') and self.saved.get('pending_retry_kind') != 'queue')
        if retry_error and (self.saved.get('retry_exhausted') or self.saved.get('retry_count', 0) >= MAX_RESUBMISSIONS):
            self.notify('retry_exhausted')
            raise RuntimeError(f'CDS重新提交达到上限：{self.path}/{MAX_RESUBMISSIONS}')
        self.notify('waiting_to_resubmit' if retry_error else 'waiting_for_queue_slot')
        while True:
            self.wait_until_retry()
            if self.stopped.is_set():
                raise RuntimeError('其他补取请求失败，停止提交；旧编号已保留')
            with self.gate.locked():
                self.check_queue_deadline()
                # 已发出的POST可能已经占用最后一个名额，先恢复该次提交结果。
                uncertain = bool(self.saved.get('submission_uncertain'))
                snapshot = None if uncertain else self.network_call('submission_gate', self.gate.snapshot)
                if uncertain or snapshot['active_count'] < snapshot['max_active_requests']:
                    try:
                        self.remote = self.submit_once()
                    except requests.HTTPError as error:
                        response = error.response
                        if (response is None or response.status_code not in (400, 429)
                                or 'json' not in response.headers.get('Content-Type', '').lower()
                                or not queue_full_reason(response.json())):
                            raise
                        self.schedule_queue_wait(reason=response.json())
                        continue
                    if retry_error:
                        self.saved['retry_count'] = self.saved.get('retry_count', 0) + 1
                    self.saved.update(request_id=self.remote.request_id, created_utc=utc_now(),
                                      pending_resubmission=False, pending_submission=False, retry_exhausted=False)
                    self.saved.pop('submission_uncertain', None)
                    self.end_queue_wait()
                    self.saved.pop('next_retry_utc', None)
                    self.saved.pop('pending_retry_kind', None)
                    write_json(self.path, self.saved)
                    self.gate.remember(self.remote.request_id, self.path)
                    break
            self.schedule_queue_wait(global_gate=snapshot)
        self.notify('submitted')
        print(json.dumps({'event': 'cds_request_resubmitted', 'request_path': str(self.path),
                          'request_id': self.remote.request_id, 'retry_count': self.saved['retry_count'],
                          'utc': utc_now()}), flush=True)

    def poll(self):
        while True:
            if self.stopped.is_set():
                raise RuntimeError('其他补取请求失败；现有CDS请求编号已保留')
            if self.saved.get('pending_resubmission') or self.saved.get('pending_submission'):
                self.resubmit()
            try:
                if self.remote is None:
                    self.remote = self.network_call('get_remote', lambda: self.client.get_remote(self.saved['request_id']))
                reply = self.network_call('poll', lambda: self.remote.json)
            except requests.HTTPError as error:
                reason = dismissed_reason(error)
                if reason is None:
                    raise
                self.retire('dismissed', reason)
                continue
            if reply['processID'] != self.dataset or reply['metadata']['request']['ids'] != self.request:
                raise ValueError('CDS服务端请求内容与本研究区域不同')
            status = reply['status']
            self.notify(status)
            if status in ('rejected', 'failed'):
                def failed_details():
                    response = self.remote.session.get(self.remote.url + '/results', headers=self.remote.headers, timeout=60)
                    if response.status_code != 400:
                        response.raise_for_status()
                    return response
                response = self.network_call('failed_results', failed_details)
                if response.status_code != 400:
                    response.raise_for_status()
                    raise RuntimeError(f'CDS失败详情返回异常：{response.status_code}')
                reason = response.json()
                if not queue_full_reason(reason):
                    raise RuntimeError(f'CDS定向请求失败：{self.remote.request_id}/{status}/{reason}')
                self.retire(status, reason)
                continue
            if status not in ('accepted', 'running', 'successful'):
                raise RuntimeError(f'CDS定向请求失败：{self.remote.request_id}/{status}')
            return status
