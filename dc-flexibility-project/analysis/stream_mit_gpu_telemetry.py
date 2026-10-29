import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import subprocess
import tarfile
import threading
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from mit_gpu_common import (BASE, PREFIX, SCHEMA_VERSION, DiskBudget, RangeDownloadError, atomic_json, database, download,
                            get_retry, identity, load_jobs, process_lock, session, summarize, utc)
from mit_gpu_priority import configure, priority_progress


def log(event, **values):
    print(json.dumps({'at': utc(), 'event': event, **values}, ensure_ascii=False) + '\n', end='', flush=True)


def prepare(args):
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    (root / 'cache').mkdir(exist_ok=True)
    db = database(root)
    if db.execute("SELECT value FROM settings WHERE name='inventory_complete'").fetchone():
        print('对象清单已完整，保留既有随机顺序', flush=True)
        return
    objects, token, pages = [], None, 0
    ns = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    while True:
        params = {'list-type': 2, 'prefix': PREFIX + 'gpu/', 'max-keys': 1000}
        if token:
            params['continuation-token'] = token
        response = get_retry(BASE, args.proxy, params=params)
        listing = ET.fromstring(response.content)
        for item in listing.findall('s:Contents', ns):
            key = item.findtext('s:Key', namespaces=ns)
            if not key.endswith('.csv'):
                raise ValueError('GPU 前缀出现非 CSV 对象: ' + key)
            objects.append({'key': key, 'etag': item.findtext('s:ETag', namespaces=ns).strip('"'),
                            'bytes': int(item.findtext('s:Size', namespaces=ns))})
        pages += 1
        print(json.dumps({'listing_pages': pages, 'objects': len(objects)}), flush=True)
        if listing.findtext('s:IsTruncated', namespaces=ns) == 'false':
            break
        token = listing.findtext('s:NextContinuationToken', namespaces=ns)
        if not token:
            raise ValueError('截断清单缺少 continuation token')
    random.Random(args.seed).shuffle(objects)
    with db:
        db.executemany('INSERT INTO objects(seq,key,etag,bytes) VALUES(?,?,?,?)',
                       [(i, o['key'], o['etag'], o['bytes']) for i, o in enumerate(objects)])
        db.executemany('INSERT INTO settings VALUES(?,?)', [
            ('seed', str(args.seed)), ('inventory_complete', utc()), ('schema_version', str(SCHEMA_VERSION))])
    inventory = {'objects': len(objects), 'bytes': sum(o['bytes'] for o in objects),
                 'max_object_bytes': max(o['bytes'] for o in objects), 'seed': args.seed,
                 'prefix': PREFIX + 'gpu/', 'listed_at': utc(), 'pages': pages}
    atomic_json(root / 'inventory.json', inventory)
    print(json.dumps(inventory), flush=True)


def benchmark(args):
    root = args.root
    db = database(root)
    # 各档使用相同对象和相同字节范围；Range 读不写磁盘，不保留原始数据。
    objects = [dict(r) for r in db.execute('SELECT * FROM objects WHERE bytes>=? ORDER BY seq LIMIT 32',
                                           (8 * 1024 * 1024,))]
    if len(objects) < 32:
        raise ValueError('吞吐试验至少需要 32 个真实对象')
    results = []
    for concurrency in (4, 8, 16):
        started = time.monotonic()

        def fetch(obj):
            begin = time.monotonic()
            size = 0
            error = None
            try:
                with session(args.proxy).get(BASE + obj['key'], timeout=(10, 30), stream=True, headers={
                        'Range': 'bytes=0-2097151', 'If-Match': '"' + obj['etag'] + '"'}) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes 0-2097151/{obj["bytes"]}':
                        raise ValueError('吞吐试验 Range 边界错误')
                    for block in response.iter_content(256 * 1024):
                        size += len(block)
                        if time.monotonic() - begin > 90:
                            raise TimeoutError('吞吐请求超过 90 秒')
                if size != 2 * 1024 * 1024:
                    raise ValueError('吞吐试验字节数不符')
            except (requests.RequestException, ValueError, TimeoutError) as exc:
                error = str(exc)
            log('benchmark_request', concurrency=concurrency, seq=obj['seq'], bytes=size,
                seconds=time.monotonic() - begin, error=error)
            return {'bytes': size, 'seconds': time.monotonic() - begin, 'error': error}

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            records = list(pool.map(fetch, objects))
        seconds = time.monotonic() - started
        record = {'concurrency': concurrency, 'objects': len(records), 'seconds': seconds,
                  'MiB_s': sum(r['bytes'] for r in records) / seconds / 1048576,
                  'range_bytes': 2097152, 'max_request_seconds': max(r['seconds'] for r in records),
                  'errors': sum(r['error'] is not None for r in records)}
        results.append(record)
        atomic_json(root / 'benchmark.json', {'trials': results, 'measured_at': utc()})
        print(json.dumps(record), flush=True)
    best = max(r['MiB_s'] for r in results if not r['errors'])
    selected = max(r['concurrency'] for r in results if r['MiB_s'] >= best * .9 and not r['errors'])
    atomic_json(root / 'benchmark.json', {'trials': results, 'selected_concurrency': selected,
                                        'rule': '最高并发档，吞吐达到最佳档 90%，所有 Range 成功', 'measured_at': utc()})


def audit(args):
    from collections import Counter
    root = args.root
    acquisition = json.loads((root / 'reference/acquisition.json').read_text(encoding='utf-8'))
    references = []
    for record in acquisition['files']:
        if record['file'] not in ('README.md', 'LICENSE', 'slurm-log.csv', 'tres-mapping.txt'):
            continue
        path = root / 'reference' / record['file']
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != record['sha256'] or path.stat().st_size != record['bytes'] or not record['etag_verified']:
            raise ValueError('Slurm/说明输入与既有经 ETag 核验的源文件不符: ' + str(path))
        references.append({k: record[k] for k in ('file', 'bytes', 'etag', 'sha256')})
    if len(references) != 4:
        raise ValueError('参考输入记录不完整')
    jobs = load_jobs(root)
    db = database(root)
    counts, job_ids = Counter(), set()
    multipart = mismatches = 0
    sequence = hashlib.sha256()
    for obj in db.execute('SELECT * FROM objects ORDER BY seq'):
        job_id, node, job, state = identity(obj['key'], jobs)
        counts[state] += 1
        job_ids.add(job_id)
        sequence.update(f'{obj["seq"]}\t{obj["key"]}\t{obj["etag"]}\t{obj["bytes"]}\n'.encode())
        if '-' in obj['etag']:
            multipart += 1
            mismatches += int((obj['bytes'] + 8388607) // 8388608 != int(obj['etag'].split('-')[-1]))
    report = {'audited_at': utc(), 'references': references, 'slurm_rows': len(jobs),
              'filename_mapping': '<id_job>-<node>.csv; node=r<digits>-n<digits>; id_job 精确连接，node 属于 nodelist',
              'mapping_counts': dict(counts), 'unique_filename_job_ids': len(job_ids),
              'multipart_objects': multipart, 'multipart_8MiB_part_count_mismatches': mismatches,
              'random_sequence_sha256': sequence.hexdigest(), 'seed': args.seed}
    if mismatches:
        raise ValueError('对象 multipart 分段与预期 8 MiB 不一致')
    atomic_json(root / 'input-audit.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def commit_batch(root, db, records):
    if not records:
        return
    records = sorted(records, key=lambda record: record[0]['seq'])
    name = f'{records[0][0]["seq"]:09d}-{records[-1][0]["seq"]:09d}'
    staging = root / 'batches' / (name + '.partial')
    target = staging.with_name(name)
    staging.mkdir(parents=True)
    successes = [(obj, tables, attempts) for obj, tables, attempts, reason in records if tables is not None]
    if successes:
        for kind in ('hourly', 'joint_hist', 'util_hist', 'files'):
            frame = pd.concat([tables[kind] for _, tables, _ in successes], ignore_index=True)
            frame.to_parquet(staging / (kind + '.parquet'), index=False, compression='zstd')
            # 重新读取持久化数据，检查文件完整、样本计数和文件数。
            check = pd.read_parquet(staging / (kind + '.parquet'))
            if len(check) != len(frame):
                raise ValueError('Parquet 写入后行数不符')
            if kind == 'files' and len(check) != len(successes):
                raise ValueError('批次文件数不符')
    manifest = {'schema_version': SCHEMA_VERSION, 'created_at': utc(), 'objects': [
        {'seq': obj['seq'], 'etag': obj['etag'], 'bytes': obj['bytes'], 'attempts': attempts,
         'status': 'done' if tables is not None else 'failed', 'reason': reason}
        for obj, tables, attempts, reason in records]}
    atomic_json(staging / 'manifest.json', manifest)
    os.replace(staging, target)
    with db:
        for row in manifest['objects']:
            db.execute('UPDATE objects SET status=?,attempts=attempts+?,reason=?,batch=?,finished=? WHERE seq=? AND etag=? AND bytes=?',
                       (row['status'], row['attempts'], row['reason'], name, utc(), row['seq'], row['etag'], row['bytes']))
    log('batch_committed', batch=name, done=len(successes), failed=len(records) - len(successes))


def reconcile(root, db):
    # 目录原子提交成功而 SQLite 尚未提交时，从不可变批次清单恢复状态。
    for folder in sorted((root / 'batches').glob('*')):
        if folder.name.endswith('.partial'):
            if folder.resolve().parent != (root / 'batches').resolve():
                raise ValueError('待清理批次超出预期目录')
            shutil.rmtree(folder)
            continue
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['schema_version'] != SCHEMA_VERSION:
            raise ValueError('批次 schema 与当前汇总版本不一致: ' + folder.name)
        with db:
            for row in manifest['objects']:
                current = db.execute('SELECT * FROM objects WHERE seq=?', (row['seq'],)).fetchone()
                if current is None or (current['etag'], current['bytes']) != (row['etag'], row['bytes']):
                    raise ValueError('已持久化批次与对象清单不符')
                if current['status'] == 'pending':
                    db.execute('UPDATE objects SET status=?,attempts=attempts+?,reason=?,batch=?,finished=? WHERE seq=?',
                               (row['status'], row['attempts'], row['reason'], folder.name, utc(), row['seq']))
    for path in (root / 'cache').glob('*.part'):
        if not path.stem.isdigit() or not db.execute("SELECT 1 FROM objects WHERE seq=? AND status='pending'", (int(path.stem),)).fetchone():
            path.unlink()


def synchronize(args, force=False):
    try:
        handle = process_lock(args.root / 'sync.lock')
    except OSError:
        log('sync_already_running')
        return
    previous_path = args.root / 'sync.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else {}
    try:
        transfer_summary(args, previous)
    except (subprocess.SubprocessError, OSError, ValueError, tarfile.TarError, sqlite3.Error) as error:
        details = str(error)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            details += ': ' + error.stderr.decode('utf-8', errors='replace')
        atomic_json(previous_path, {**previous, 'last_attempt': utc(), 'error': details})
        log('sync_failed', reason=details)
    finally:
        (args.root / 'sync-work/delta.tar').unlink(missing_ok=True)
        handle.close()


def transfer_summary(args, previous):
    root = args.root
    status_path = root / 'sync.json'
    synced = set(previous.get('batches', []))
    work = root / 'sync-work'
    work.mkdir(exist_ok=True)
    (work / 'delta.tar').unlink(missing_ok=True)
    source = sqlite3.connect(root / 'state.sqlite')
    snapshot = sqlite3.connect(work / 'state.sqlite')
    source.backup(snapshot)
    batches = {r[0] for r in snapshot.execute('SELECT DISTINCT batch FROM objects WHERE batch IS NOT NULL')}
    snapshot.close()
    source.close()
    new = sorted(batches - synced)
    archive = work / 'delta.tar'
    inputs = []
    for name in new:
        inputs.extend((p, 'batches/' + name + '/' + p.name) for p in (root / 'batches' / name).iterdir() if p.is_file())
    for name in ('progress.json', 'inventory.json', 'benchmark.json', 'input-audit.json', 'clock-boundary-audit.json', 'pilot-validation.json', 'run.json'):
        if (root / name).exists():
            inputs.append((root / name, name))
    inputs.append((work / 'state.sqlite', 'state.sqlite'))
    if (root / 'aggregates').exists():
        inputs.extend((p, str(p.relative_to(root)).replace('\\', '/')) for p in (root / 'aggregates').rglob('*')
                      if p.is_file() and not p.name.endswith('.tmp'))
    if (root / 'analysis/priority-month').exists():
        inputs.extend((p, str(p.relative_to(root)).replace('\\', '/'))
                      for p in (root / 'analysis/priority-month').rglob('*')
                      if p.is_file() and not p.name.endswith('.tmp'))
    # 下载缓存预留 19 GB，同步暂存最多 1 GB；超限报错并留待后续同步。
    if sum(p.stat().st_size + 2048 for p, _ in inputs) > 900_000_000 or (work / 'state.sqlite').stat().st_size > 90_000_000:
        raise OSError('同步暂存将超过 1 GB 上限')
    with tarfile.open(archive, 'w') as tar:
        for path, name in inputs:
            tar.add(path, arcname=name)
    try:
        # 服务器只接收派生汇总，保留至少 50 GB 给现有计算任务。
        free = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.sync_host,
            "df -B1 --output=avail /data3 | tail -1"], check=True, capture_output=True, text=True, timeout=45)
        if int(free.stdout.strip()) - archive.stat().st_size * 2 < 50_000_000_000:
            raise OSError('服务器余量不足 50 GB，跳过同步')
        subprocess.run(['ssh', '-o', 'BatchMode=yes', args.sync_host,
                        f'mkdir -p {args.sync_dir}/.incoming && chmod 700 {args.sync_dir}'], check=True, capture_output=True, timeout=45)
        helper = Path(__file__).with_name('commit_mit_gpu_summary.py')
        subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', str(archive), str(helper),
                        f'{args.sync_host}:{args.sync_dir}/.incoming/'],
                       check=True, capture_output=True, timeout=600)
        # 先在私有暂存目录解包，再逐批原子改名；服务器的批次只在完整后可见。
        command = f'python3 {args.sync_dir}/.incoming/commit_mit_gpu_summary.py --root {args.sync_dir}'
        subprocess.run(['ssh', '-o', 'BatchMode=yes', args.sync_host, command],
                       check=True, capture_output=True, timeout=180)
        atomic_json(status_path, {'last_success': utc(), 'batches': sorted(batches), 'error': None})
        log('sync_done', new_batches=len(new), bytes=archive.stat().st_size)
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        atomic_json(status_path, {**previous, 'last_attempt': utc(), 'error': str(error)})
        log('sync_failed', reason=str(error))
    finally:
        archive.unlink(missing_ok=True)


def run(args):
    root = args.root
    (root / 'cache').mkdir(exist_ok=True)
    (root / 'batches').mkdir(exist_ok=True)
    lock = process_lock(root / 'process.lock')
    db = database(root)
    if not db.execute("SELECT value FROM settings WHERE name='inventory_complete'").fetchone():
        raise ValueError('必须先完整列举全部对象并固化随机顺序')
    version = db.execute("SELECT value FROM settings WHERE name='schema_version'").fetchone()
    if version is None or int(version[0]) != SCHEMA_VERSION:
        raise ValueError('已有汇总 schema 与当前代码不一致，需要显式迁移私有运行目录')
    reconcile(root, db)
    if (root / 'STOP').exists():
        raise ValueError('STOP 文件存在；确认续跑时删除该文件')
    jobs = load_jobs(root)
    if args.priority_window:
        configure(db, jobs, *args.priority_window, timezone=args.priority_timezone)
    window_state = priority_progress(db)
    if window_state and window_state['failed']:
        raise ValueError('优先窗口存在失败对象，须处理失败后继续剩余随机队列')
    budget = DiskBudget(root / 'cache')
    selected = [dict(r) for r in db.execute("SELECT * FROM objects WHERE status='pending' ORDER BY priority DESC,seq")]
    if args.limit:
        selected = selected[:args.limit]
    initial_bytes = db.execute("SELECT COALESCE(SUM(bytes),0) FROM objects WHERE status='done'").fetchone()[0]
    started, last_progress, last_sync = time.monotonic(), 0, time.monotonic()
    sync_count = 0
    sync_thread = None
    computed = {'files': 0, 'bytes': 0}
    computed_lock = threading.Lock()
    atomic_json(root / 'run.json', {'pid': os.getpid(), 'started': utc(), 'concurrency': args.concurrency,
                                  'limit': args.limit, 'root': str(root), 'schema_version': SCHEMA_VERSION,
                                  'clock_policy': 'new_york_wall_time_inferred_from_pilot',
                                  'script_path': str(Path(__file__).resolve()),
                                  'priority_window': window_state,
                                  'args': vars(args) | {'root': str(root)}})

    def progress(state='running'):
        nonlocal last_progress
        counts = {r['status']: (r['n'], r['b']) for r in db.execute(
            'SELECT status,COUNT(*) n,COALESCE(SUM(bytes),0) b FROM objects GROUP BY status')}
        done, done_bytes = counts.get('done', (0, 0))
        elapsed = time.monotonic() - started
        with computed_lock:
            computed_bytes, computed_files = computed['bytes'], computed['files']
        rate = computed_bytes / elapsed if elapsed else 0
        pending_bytes = counts.get('pending', (0, 0))[1]
        first_pending = db.execute("SELECT MIN(seq) FROM objects WHERE status='pending'").fetchone()[0]
        prefix_count = first_pending if first_pending is not None else sum(v[0] for v in counts.values())
        buffered_bytes = max(0, initial_bytes + computed_bytes - done_bytes)
        synced = json.loads((root / 'sync.json').read_text(encoding='utf-8')) if (root / 'sync.json').exists() else {}
        cache_bytes = 0
        for path in (root / 'cache').glob('*.part'):
            try:
                cache_bytes += path.stat().st_size
            except FileNotFoundError:
                # 工作线程可在扫描期间完成计算并删除该缓存。
                continue
        atomic_json(root / 'progress.json', {'updated_at': utc(), 'pid': os.getpid(), 'state': state,
            'completed': done, 'total': sum(v[0] for v in counts.values()), 'completed_bytes': done_bytes,
            'total_bytes': sum(v[1] for v in counts.values()), 'failed': counts.get('failed', (0, 0))[0],
            'pending': counts.get('pending', (0, 0))[0], 'bytes_per_second': rate,
            'MiB_per_second': rate / 1048576, 'eta_seconds': max(0, pending_bytes - buffered_bytes) / rate if rate > 0 else None,
            'computed_this_run_files': computed_files, 'computed_this_run_bytes': computed_bytes,
            'computed_uncommitted_bytes': buffered_bytes,
            'random_prefix_objects': prefix_count,
            'priority_window': priority_progress(db, rate),
            'active_objects': [{'seq': obj['seq'], 'priority': obj['priority']} for obj in queue.values()],
            'elapsed_seconds': elapsed, 'cache_reserved_bytes': budget.reserved,
            'cache_actual_bytes': cache_bytes,
            'cache_limit_bytes': budget.limit, 'disk_free_bytes': shutil.disk_usage(root).free,
            'last_sync': synced.get('last_success'), 'sync_error': synced.get('error'),
            'concurrency': args.concurrency, 'seed': int(db.execute("SELECT value FROM settings WHERE name='seed'").fetchone()[0])})
        last_progress = time.monotonic()

    def worker(obj):
        path = root / 'cache' / f'{obj["seq"]:09d}.part'
        try:
            for attempt in range(1, args.retries + 1):
                try:
                    checksum = download(obj, path, args.proxy, retries=args.retries, range_pool=range_pool)
                    tables = summarize(path, obj, jobs)
                    for key, value in checksum.items():
                        tables['files'][key] = value
                    with computed_lock:
                        computed['files'] += 1
                        computed['bytes'] += obj['bytes']
                    log('object_computed', seq=obj['seq'], bytes=obj['bytes'], attempt=attempt, priority=obj['priority'])
                    return obj, tables, attempt, None
                except RangeDownloadError as error:
                    log('object_failed', seq=obj['seq'], reason=str(error))
                    return obj, None, attempt, str(error)
                except (requests.RequestException, OSError, ValueError, pd.errors.ParserError) as error:
                    log('object_retry', seq=obj['seq'], attempt=attempt, reason=str(error))
                    if attempt == args.retries:
                        return obj, None, attempt, f'{type(error).__name__}: {error}'
                    time.sleep(min(120, 2 ** attempt))
        finally:
            path.unlink(missing_ok=True)
            budget.release(obj['bytes'])

    # 优先组内保留原随机序号；窗口全部提交后才开始其余对象。
    cursor, queue, records = 0, {}, []
    last_commit = time.monotonic()
    failure = None
    successful = False
    progress()
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as range_pool, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            while cursor < len(selected) or queue:
                while cursor < len(selected) and len(queue) < args.concurrency and not (root / 'STOP').exists():
                    obj = selected[cursor]
                    if obj['priority'] == 0 and window_state:
                        if any(active['priority'] for active in queue.values()):
                            break
                        if records:
                            commit_batch(root, db, records)
                            sync_count += len(records)
                            records = []
                            last_commit = time.monotonic()
                        current_window = priority_progress(db)
                        if current_window['failed'] or not current_window['complete']:
                            raise RuntimeError('优先窗口尚未全部成功，保留剩余随机队列')
                        window_state = None
                        log('priority_window_complete', **current_window)
                    if not budget.acquire(obj['bytes'], f'{obj["seq"]:09d}.part'):
                        break
                    queue[pool.submit(worker, obj)] = obj
                    cursor += 1
                if not queue:
                    if cursor < len(selected) and not (root / 'STOP').exists():
                        raise OSError('已有待续传缓存占用容量，下一对象无法获得完整预留空间')
                    break
                finished, _ = wait(queue, timeout=5, return_when=FIRST_COMPLETED)
                for future in sorted(finished, key=lambda item: queue[item]['seq']):
                    queue.pop(future)
                    records.append(future.result())
                if records and (len(records) >= args.batch_size or time.monotonic() - last_commit >= 30):
                    commit_batch(root, db, records)
                    sync_count += len(records)
                    records = []
                    last_commit = time.monotonic()
                if time.monotonic() - last_progress >= 15:
                    progress()
                if (time.monotonic() - last_sync >= 1800 or sync_count >= 2000) and (sync_thread is None or not sync_thread.is_alive()):
                    sync_thread = threading.Thread(target=synchronize, args=(args,), daemon=True)
                    sync_thread.start()
                    last_sync, sync_count = time.monotonic(), 0
        commit_batch(root, db, records)
        successful = True
    except (OSError, ValueError, RuntimeError) as error:
        failure = error
        log('run_failed', reason=str(error))
    finally:
        progress('failed' if not successful else ('stopped' if (root / 'STOP').exists() else 'finished'))
        if sync_thread is not None:
            sync_thread.join(timeout=5)
        if sync_thread is None or not sync_thread.is_alive():
            synchronize(args, force=True)
        progress('failed' if not successful else ('stopped' if (root / 'STOP').exists() else 'finished'))
        db.close()
        lock.close()
    if failure:
        raise failure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'audit', 'benchmark', 'run', 'sync'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--proxy', default='http://127.0.0.1:7897')
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--concurrency', type=int, default=16, choices=range(1, 17))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--retries', type=int, default=5)
    parser.add_argument('--priority-window', nargs=2, metavar=('START', 'END'))
    parser.add_argument('--priority-timezone', default='UTC')
    parser.add_argument('--sync-host', default='pku-test')
    parser.add_argument('--sync-dir', default='/data3/dataset/ljt/mit-supercloud-gpu-summary-20261028')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'benchmark':
        benchmark(args)
    elif args.command == 'audit':
        audit(args)
    elif args.command == 'sync':
        synchronize(args, force=True)
    else:
        run(args)


if __name__ == '__main__':
    main()
