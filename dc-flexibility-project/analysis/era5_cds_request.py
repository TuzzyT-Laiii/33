from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import random
import re

import requests

from acquire_buildability_protection import utc_now, write_json


MAX_RESUBMISSIONS = 5
QUEUE_WAIT_INITIAL_SECONDS = 300
QUEUE_WAIT_MAX_SECONDS = 3600
QUEUE_WAIT_LIMIT_SECONDS = 72 * 3600
DATASET = 'reanalysis-era5-single-levels'
QUEUE_FULL_REASON = ('Maximum number queued requests for this dataset has been reached. '
                     'Please configure your scripts to request data sequentially')


def queue_full_reason(reason):
    return isinstance(reason, dict) and reason.get('traceback') == QUEUE_FULL_REASON


def queue_delay(attempt):
    base = min(QUEUE_WAIT_INITIAL_SECONDS * 2 ** min(max(attempt - 1, 0), 4), QUEUE_WAIT_MAX_SECONDS)
    return min(base * random.uniform(0.9, 1.1), QUEUE_WAIT_MAX_SECONDS)


def active_dataset_jobs(client, page_size=100):
    jobs = {}
    page = client.get_jobs(limit=page_size, status=['accepted', 'running'])
    seen = set()
    while page is not None:
        reply = page.json
        url = next(link['href'] for link in reply['links'] if link['rel'] == 'self')
        if url in seen:
            raise RuntimeError('CDS账号队列分页重复，停止提交')
        seen.add(url)
        for job in reply['jobs']:
            if job['processID'] == DATASET and job['status'] in ('accepted', 'running'):
                jobs[job['jobID']] = job['status']
        page = page.next
    return jobs


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
        remaining = max(0, (datetime.fromisoformat(self.saved.get('next_retry_utc', utc_now())) -
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
            remaining = max(0, (datetime.fromisoformat(self.saved.get('next_retry_utc', utc_now())) -
                                datetime.now(timezone.utc)).total_seconds())
        self.check_queue_deadline()

    def notify(self, status, **details):
        details.setdefault('next_retry_utc', self.saved.get('next_retry_utc'))
        self.update(status, self.saved['request_id'], history=list(self.saved.get('history', [])),
                    retry_count=self.saved.get('retry_count', 0), retry_limit=MAX_RESUBMISSIONS,
                    queue_wait_attempts=self.saved.get('queue_wait_attempts', 0),
                    queue_wait_seconds=self.saved.get('queue_wait_seconds', 0.0),
                    queue_wait_deadline_utc=self.saved.get('queue_wait_deadline_utc'), **details)

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
                snapshot = self.gate.snapshot()
                if snapshot['active_count'] < snapshot['max_active_requests']:
                    if retry_error:
                        self.saved['retry_count'] = self.saved.get('retry_count', 0) + 1
                    write_json(self.path, self.saved)
                    try:
                        self.remote = self.client.submit(self.dataset, self.request)
                    except requests.HTTPError as error:
                        response = error.response
                        if (response is None or response.status_code not in (400, 429)
                                or 'json' not in response.headers.get('Content-Type', '').lower()
                                or not queue_full_reason(response.json())):
                            raise
                        if retry_error:
                            self.saved['retry_count'] -= 1
                        self.schedule_queue_wait(reason=response.json())
                        continue
                    self.saved.update(request_id=self.remote.request_id, created_utc=utc_now(),
                                      pending_resubmission=False, pending_submission=False, retry_exhausted=False)
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
                    self.remote = self.client.get_remote(self.saved['request_id'])
                reply = self.remote.json
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
                response = self.remote.session.get(self.remote.url + '/results', headers=self.remote.headers, timeout=60)
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
