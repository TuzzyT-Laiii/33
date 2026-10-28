import argparse
import ast
from datetime import datetime, timedelta, timezone
import fcntl
import json
import multiprocessing
from pathlib import Path
import shutil
import threading

from ecmwf.datastores import Client

from acquire_buildability_protection import utc_now, write_json
from era5_cds_request import (CDSRequestGate, MAX_RESUBMISSIONS, PersistentCDSRequest,
                              QUEUE_WAIT_LIMIT_SECONDS, active_dataset_jobs, queue_delay, queue_full_reason)
from prepare_era5_wind_repair import cds_read_session


def check_locked(path, result):
    with path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            result.put(True)
        else:
            result.put(False)


def verify(root, output, source):
    temporary = output.parent / (output.stem + '-state')
    temporary.mkdir(parents=True, exist_ok=False)
    live_path = root / 'data/raw/era5-eligible-cds-repairs/lat-160-lon-101/wind-2023.request.json'
    saved = json.loads(live_path.read_text())
    key = json.loads((root / '.codex-temp/buildability-20261025/cds-credential.json').read_text())['key']
    client = Client(url='https://cds.climate.copernicus.eu/api', key=key, cleanup=False,
                    progress=False, timeout=60, maximum_tries=3, retry_after=2, session=cds_read_session())
    gate = CDSRequestGate(temporary, client)
    path = temporary / 'wind-2023.request.json'
    shutil.copy2(live_path, path)
    events = []

    def update(status, request_id, **details):
        events.append({'status': status, 'request_id': request_id, **details})

    def load():
        return PersistentCDSRequest(client, saved['dataset'], saved['request'], path,
                                    update, threading.Event(), gate=gate)

    job = load()
    history = saved['history']
    queue_history = [entry for entry in history if queue_full_reason(entry['reason'])]
    assert queue_history and history == job.saved['history']
    expected = saved.get('retry_count', 0) if saved.get('retry_budget_version') == 2 else (
        saved['retry_count'] - len(queue_history) + int(saved.get('pending_resubmission', False)))
    assert job.saved['retry_count'] == expected
    assert not job.saved['retry_exhausted'] and job.saved['pending_retry_kind'] == 'queue'
    before_restart = dict(job.saved)
    assert load().saved == before_restart
    reason = queue_history[-1]['reason']
    assert queue_full_reason(reason)
    remote = client.get_remote(saved['request_id'])
    if remote.status in ('rejected', 'failed'):
        response = remote.session.get(remote.url + '/results', headers=remote.headers, timeout=60)
        assert response.status_code == 400 and queue_full_reason(response.json())

    # 边界验证使用真实回执中的错误原因，状态文件仅写入独立临时目录。
    job.saved['retry_count'] = MAX_RESUBMISSIONS
    job.retire('rejected', reason)
    assert job.saved['retry_count'] == MAX_RESUBMISSIONS and not job.saved['retry_exhausted']
    assert job.saved['history'][:len(history)] == history
    assert load().saved == job.saved
    observed_delays = {}
    for attempt in (1, 2, 3, 4, 5, 1000000):
        delays = [queue_delay(attempt) for _ in range(100)]
        base = min(300 * 2 ** min(attempt - 1, 4), 3600)
        assert min(delays) >= base * 0.9 and max(delays) <= min(base * 1.1, 3600)
        observed_delays[str(attempt)] = [min(delays), max(delays)]
    job.saved['queue_wait_deadline_utc'] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    write_json(path, job.saved)
    try:
        load().check_queue_deadline()
    except RuntimeError as error:
        assert '72小时' in str(error)
    else:
        raise AssertionError('超过期限未失败')
    assert json.loads(path.read_text())['queue_wait_exhausted']
    dismissed = next(entry for entry in history if entry['status'] == 'dismissed')
    job = load()
    job.retire('dismissed', dismissed['reason'])
    assert job.saved['retry_exhausted']
    try:
        job.resubmit()
    except RuntimeError as error:
        assert '达到上限' in str(error)
    else:
        raise AssertionError('普通重提超过预算未失败')

    active = active_dataset_jobs(client, page_size=1)
    with gate.locked():
        for request_id in active:
            gate.remember(request_id, live_path)
        gate.remember(saved['request_id'], live_path)
        snapshot = gate.snapshot()
        assert not (gate.directory / f'submitted-{saved["request_id"]}.json').exists()
        context = multiprocessing.get_context('spawn')
        result = context.Queue()
        process = context.Process(target=check_locked, args=(gate.directory / 'submission.lock', result))
        process.start()
        assert result.get(timeout=30)
        process.join(timeout=30)
        assert process.exitcode == 0
    assert snapshot['max_active_requests'] == 2
    assert active == snapshot['active_requests'], '检查期间CDS队列发生变化，请重新运行验证'
    with (gate.directory / 'submission.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    current = (source / 'prepare_era5_wind_repair.py').read_text()
    original = (root / 'analysis/prepare_era5_wind_repair.py').read_text()
    assert current[current.index('    records = [record for'):] == original[original.index('    records = [record for'):]
    old_request = next(node for node in ast.walk(ast.parse(original)) if isinstance(node, ast.FunctionDef) and node.name == 'request_for')
    new_request = next(node for node in ast.walk(ast.parse(current)) if isinstance(node, ast.FunctionDef) and node.name == 'request_for')
    assert ast.dump(old_request) == ast.dump(new_request)
    report = {'checked_utc': utc_now(), 'status': 'passed', 'legacy_retry_count': saved.get('legacy_retry_count', saved['retry_count']),
              'migrated_retry_count': expected, 'preserved_history_entries': len(history),
              'queue_rejection_does_not_consume_retry_budget': True, 'ordinary_retry_limit_enforced': True,
              'restart_preserves_state': True, 'queue_wait_limit_seconds': QUEUE_WAIT_LIMIT_SECONDS,
              'deadline_enforced': True, 'delay_ranges_seconds': observed_delays,
              'real_cds_paginated_active_jobs': active, 'global_gate': snapshot,
              'active_tokens_not_double_counted_and_terminal_token_released': True,
              'cross_process_lock_exclusion_and_release': True, 'scientific_validation_and_request_unchanged': True,
              'new_cds_submissions': 0, 'events': events}
    write_json(output, report)
    print(json.dumps({key: value for key, value in report.items() if key != 'events'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(args.root, args.output, Path(__file__).parent)
