from datetime import datetime, timedelta, timezone
import json
import re

import requests

from acquire_buildability_protection import utc_now, write_json


MAX_RESUBMISSIONS = 5
QUEUE_FULL_REASON = ('Maximum number queued requests for this dataset has been reached. '
                     'Please configure your scripts to request data sequentially')


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
    def __init__(self, client, dataset, request, path, update, stopped, known_id=None):
        self.client = client
        self.dataset = dataset
        self.request = request
        self.path = path
        self.update = update
        self.stopped = stopped
        self.remote = None
        if path.exists():
            self.saved = json.loads(path.read_text())
            if self.saved['dataset'] != dataset or self.saved['request'] != request:
                raise ValueError('CDS持久请求与当前合格区域请求不同')
        else:
            if not known_id:
                self.remote = client.submit(dataset, request)
            self.saved = {'dataset': dataset, 'request': request,
                          'request_id': known_id if known_id else self.remote.request_id,
                          'created_utc': utc_now(), 'retry_count': 0, 'history': []}
            write_json(path, self.saved)

    def notify(self, status, **details):
        details.setdefault('next_retry_utc', self.saved.get('next_retry_utc'))
        self.update(status, self.saved['request_id'], history=list(self.saved.get('history', [])),
                    retry_count=self.saved.get('retry_count', 0), retry_limit=MAX_RESUBMISSIONS, **details)

    def retire(self, status, reason):
        history = self.saved.setdefault('history', [])
        retry_count = self.saved.get('retry_count', 0)
        entry = {'request_id': self.saved['request_id'], 'status': status, 'reason': reason,
                 'retired_utc': utc_now(), 'retry_count': retry_count}
        history.append(entry)
        exhausted = retry_count >= MAX_RESUBMISSIONS
        delay = min(60 * 2 ** retry_count, 900)
        self.saved.update(pending_resubmission=True, retry_exhausted=exhausted,
                          next_retry_utc=(datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat())
        archived = self.path.with_name(self.path.stem + f'.retired-{self.saved["request_id"]}.json')
        write_json(archived, self.saved)
        write_json(self.path, self.saved)
        self.notify('retry_exhausted' if exhausted else (
            'waiting_for_queue_slot' if status in ('rejected', 'failed') else 'waiting_to_resubmit'),
            next_retry_utc=self.saved['next_retry_utc'])
        print(json.dumps({'event': 'cds_request_retired', 'request_path': str(self.path), **entry,
                          'retry_exhausted': exhausted, 'next_retry_utc': self.saved['next_retry_utc']}), flush=True)

    def resubmit(self):
        if self.saved.get('retry_exhausted') or self.saved.get('retry_count', 0) >= MAX_RESUBMISSIONS:
            self.notify('retry_exhausted')
            raise RuntimeError(f'CDS重新提交达到上限：{self.path}/{MAX_RESUBMISSIONS}')
        self.notify('waiting_for_queue_slot' if self.saved['history'][-1]['status'] in ('rejected', 'failed')
                    else 'waiting_to_resubmit', next_retry_utc=self.saved['next_retry_utc'])
        remaining = (datetime.fromisoformat(self.saved['next_retry_utc']) - datetime.now(timezone.utc)).total_seconds()
        if self.stopped.wait(max(0, remaining)) or self.stopped.is_set():
            raise RuntimeError('其他补取请求失败，停止重新提交；旧编号已保留')
        # 提交前保存预算，避免重启进程重置次数。
        self.saved['retry_count'] = self.saved.get('retry_count', 0) + 1
        write_json(self.path, self.saved)
        self.remote = self.client.submit(self.dataset, self.request)
        self.saved.update(request_id=self.remote.request_id, created_utc=utc_now(), pending_resubmission=False)
        self.saved.pop('next_retry_utc', None)
        write_json(self.path, self.saved)
        self.notify('submitted')
        print(json.dumps({'event': 'cds_request_resubmitted', 'request_path': str(self.path),
                          'request_id': self.remote.request_id, 'retry_count': self.saved['retry_count'],
                          'utc': utc_now()}), flush=True)

    def poll(self):
        while True:
            if self.stopped.is_set():
                raise RuntimeError('其他补取请求失败；现有CDS请求编号已保留')
            if self.saved.get('pending_resubmission'):
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
                if reason.get('traceback') != QUEUE_FULL_REASON:
                    raise RuntimeError(f'CDS定向请求失败：{self.remote.request_id}/{status}/{reason}')
                self.retire(status, reason)
                continue
            if status not in ('accepted', 'running', 'successful'):
                raise RuntimeError(f'CDS定向请求失败：{self.remote.request_id}/{status}')
            return status
