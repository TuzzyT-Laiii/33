import argparse
import ast
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import ssl
import threading
import unittest
from unittest.mock import Mock, PropertyMock, patch

import requests
from ecmwf.datastores.processing import Remote

from acquire_buildability_protection import write_json
import era5_cds_request as cds


class Clock(datetime):
    value = datetime.now(timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.value if tz else cls.value.replace(tzinfo=None)


class AdvancingEvent:
    def __init__(self):
        self.steps = []
        self.stopped = False

    def is_set(self):
        return self.stopped

    def wait(self, seconds):
        self.steps.append(seconds)
        Clock.value += timedelta(seconds=seconds)
        return self.stopped


def http_error(status, body=None):
    response = requests.Response()
    response.status_code = status
    response.url = 'https://example.invalid/jobs/retained-id'
    response.headers['Content-Type'] = 'application/json'
    response._content = json.dumps(body or {'detail': 'verification'}).encode()
    return requests.HTTPError(f'HTTP {status}', response=response)


class NetworkRetryChecks(unittest.TestCase):
    directory = None

    def setUp(self):
        Clock.value = datetime.now(timezone.utc)
        self.clock = patch.object(cds, 'datetime', Clock)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.path = self.directory / self._testMethodName / 'wind.request.json'
        self.path.parent.mkdir(parents=True)
        self.events = []
        self.stopped = AdvancingEvent()
        self.client = Mock(timeout=5, verify=True)
        self.gate = cds.CDSRequestGate(self.path.parent, self.client)
        self.request = {'year': ['2024'], 'variable': ['100m_u_component_of_wind']}
        self.job = self.load()

    def load(self):
        return cds.PersistentCDSRequest(self.client, cds.DATASET, self.request, self.path,
                                        lambda status, request_id, **details: self.events.append(
                                            {'status': status, 'request_id': request_id, **details}),
                                        self.stopped, 'retained-id', gate=self.gate)

    def reply(self, status='accepted'):
        return {'processID': cds.DATASET, 'metadata': {'request': {'ids': self.request}}, 'status': status}

    def page(self, jobs):
        page = Mock()
        page.json = {'links': [{'rel': 'self', 'href': 'https://example.invalid/jobs'}],
                     'jobs': [{'jobID': request_id, 'status': 'accepted', 'processID': cds.DATASET}
                              for request_id in jobs]}
        page.next = None
        return page

    def test_exception_classification(self):
        transient = [requests.ConnectionError('reset'), requests.exceptions.SSLError('EOF'),
                     requests.Timeout(), requests.ConnectTimeout(), requests.ReadTimeout(),
                     requests.exceptions.ChunkedEncodingError(), socket.gaierror(socket.EAI_AGAIN, 'dns'),
                     ssl.SSLEOFError(), requests.ConnectionError('Temporary failure in name resolution')]
        for error in transient + [http_error(code) for code in (429, 500, 501, 502, 503, 504, 599)]:
            self.assertIsNotNone(cds.network_error_kind(error), repr(error))
        for error in [http_error(code) for code in (400, 401, 403, 404, 422)] + [
                ValueError(), requests.exceptions.InvalidURL(), OSError('disk failure')]:
            self.assertIsNone(cds.network_error_kind(error), repr(error))

    def test_exponential_backoff_jitter_and_cap(self):
        for attempt in (1, 2, 3, 4, 5, 6, 7, 1000000):
            base = min(60 * 2 ** min(attempt - 1, 5), 1800)
            delays = [cds.network_delay(attempt) for _ in range(100)]
            self.assertGreaterEqual(min(delays), base * 0.9)
            self.assertLessEqual(max(delays), min(base * 1.1, 1800))
        with patch.object(cds.random, 'uniform', return_value=1):
            self.assertEqual([cds.network_delay(n) for n in range(1, 8)], [60, 120, 240, 480, 960, 1800, 1800])

    def test_retry_retains_id_budget_and_logs_all_events(self):
        errors = [requests.exceptions.SSLError('EOF'), socket.gaierror(socket.EAI_AGAIN, 'dns'),
                  requests.Timeout(), requests.exceptions.ChunkedEncodingError(), http_error(503), http_error(429)]
        operation = Mock(side_effect=errors + ['ready'])
        original = dict(self.job.saved)
        self.assertEqual(self.job.network_call('poll', operation), 'ready')
        self.assertEqual(self.job.saved['request_id'], original['request_id'])
        self.assertEqual(self.job.saved['retry_count'], original['retry_count'])
        self.assertEqual(self.job.saved['history'], original['history'])
        self.assertEqual(len(self.job.saved['network_wait_events']), len(errors))
        self.assertEqual(sum(event['status'] == 'waiting_for_network' for event in self.events), len(errors))
        self.assertTrue(all(0 < step <= 30 for step in self.stopped.steps))
        self.assertEqual(self.load().saved, self.job.saved)
        self.assertNotIn('network_next_retry_utc', self.job.saved)
        self.client.submit.assert_not_called()

    def test_fatal_errors_do_not_wait(self):
        for error in [http_error(code) for code in (400, 401, 403, 404, 422)] + [ValueError('invalid data')]:
            with self.assertRaises(type(error)):
                self.job.network_call('poll', Mock(side_effect=error))
        self.assertNotIn('network_wait_events', self.job.saved)

    def test_restart_resumes_remaining_wait(self):
        self.stopped.stopped = True
        with self.assertRaises(RuntimeError):
            self.job.network_wait('poll', requests.Timeout())
        used = self.job.saved['network_wait_seconds']
        pending = self.job.saved['network_next_retry_utc']
        Clock.value = datetime.fromisoformat(pending) - timedelta(seconds=17)
        self.stopped.stopped = False
        self.stopped.steps.clear()
        restarted = self.load()
        self.assertEqual(restarted.network_call('poll', lambda: 'ready'), 'ready')
        self.assertAlmostEqual(sum(self.stopped.steps), 17, places=5)
        self.assertEqual(restarted.saved['network_wait_seconds'], used)

    def test_24_hour_budget_is_cumulative_and_persistent(self):
        self.job.saved['network_wait_seconds'] = cds.NETWORK_WAIT_LIMIT_SECONDS - 7
        write_json(self.path, self.job.saved)
        failing = Mock(side_effect=requests.Timeout())
        with self.assertRaisesRegex(RuntimeError, '24小时'):
            self.load().network_call('poll', failing)
        saved = self.load().saved
        self.assertEqual(saved['network_wait_seconds'], 86400)
        self.assertEqual(saved['network_wait_events'][-1]['delay_seconds'], 7)
        self.assertTrue(saved['network_wait_exhausted'])
        self.assertEqual(saved['request_id'], 'retained-id')
        self.assertEqual(saved['retry_count'], 0)
        with self.assertRaisesRegex(RuntimeError, '24小时'):
            self.load().network_call('poll', failing)

    def test_poll_and_get_remote_network_failures(self):
        remote = Mock()
        type(remote).json = PropertyMock(side_effect=[http_error(502), self.reply()])
        self.client.get_remote.side_effect = [requests.exceptions.SSLError('EOF'), remote]
        self.assertEqual(self.job.poll(), 'accepted')
        self.assertEqual([event['operation'] for event in self.job.saved['network_wait_events']], ['get_remote', 'poll'])
        self.assertEqual(self.job.saved['request_id'], 'retained-id')

    def test_failed_results_network_retry_preserves_fatal_reason(self):
        remote = Mock(url='https://example.invalid/jobs/retained-id', headers={})
        remote.json = self.reply('failed')
        reason = http_error(400, {'traceback': 'invalid parameter'}).response
        remote.session.get.side_effect = [http_error(503), reason]
        self.job.remote = remote
        with self.assertRaisesRegex(RuntimeError, 'invalid parameter'):
            self.job.poll()
        self.assertEqual(self.job.saved['network_wait_events'][-1]['operation'], 'failed_results')
        self.assertEqual(self.job.saved['history'], [])

    def test_gate_pagination_and_token_lookup_network_failures(self):
        self.client.get_jobs.side_effect = [requests.ConnectionError('dns'), self.page([]), self.page([])]
        self.gate.remember('retained-id', self.path)
        remote = Mock(status='running')
        self.client.get_remote.side_effect = [requests.Timeout(), remote]
        snapshot = self.job.network_call('submission_gate', self.gate.snapshot)
        self.assertEqual(snapshot['active_requests'], {'retained-id': 'running'})
        self.assertEqual(len(self.job.saved['network_wait_events']), 2)

    def test_submission_response_loss_recovers_without_second_post(self):
        self.job.saved.update(pending_submission=True, request_id=None)
        self.client.get_jobs.side_effect = [self.page([]), self.page([]), self.page(['recovered-id'])]
        process = Mock()
        process.submit.side_effect = requests.ReadTimeout('POST response lost')
        self.client.get_process.return_value = process
        remote = Mock(request_id='recovered-id')
        remote.json = self.reply()
        self.client.get_remote.return_value = remote
        self.job.resubmit()
        process.submit.assert_called_once_with(self.request)
        self.assertEqual(self.job.saved['request_id'], 'recovered-id')
        self.assertEqual(self.job.saved['retry_count'], 0)
        self.assertFalse(self.job.saved['pending_submission'])
        self.assertNotIn('submission_uncertain', self.job.saved)
        self.assertTrue((self.gate.directory / 'submitted-recovered-id.json').exists())

    def test_uncertain_submission_restart_bypasses_full_gate(self):
        self.job.saved.update(pending_submission=True, submission_uncertain={'known_ids': [], 'started_utc': cds.utc_now()})
        write_json(self.path, self.job.saved)
        self.client.get_jobs.return_value = self.page(['recovered-id', 'other-id'])
        remote = Mock(request_id='recovered-id')
        remote.json = self.reply()
        other = Mock()
        other.json = self.reply() | {'metadata': {'request': {'ids': {'year': ['2025']}}}}
        self.client.get_remote.side_effect = lambda request_id: remote if request_id == 'recovered-id' else other
        restarted = self.load()
        restarted.resubmit()
        self.assertEqual(restarted.saved['request_id'], 'recovered-id')
        self.client.get_process.assert_not_called()

    def test_submission_5xx_and_429_never_blindly_repeat_post(self):
        for code in (429, 500, 503):
            self.path = self.path.with_name(f'submit-{code}.json')
            self.job = self.load()
            self.client.get_jobs.side_effect = [self.page([]), self.page(['recovered-id'])]
            remote = Mock(request_id='recovered-id')
            remote.json = self.reply()
            self.client.get_remote.return_value = remote
            process = Mock()
            process.submit.side_effect = http_error(code)
            self.client.get_process.return_value = process
            self.assertEqual(self.job.submit_once().request_id, 'recovered-id')
            process.submit.assert_called_once()

    def test_submission_preflight_waits_before_single_post(self):
        process = Mock()
        remote = Mock(request_id='accepted-id')
        process.submit.return_value = remote
        self.client.get_process.side_effect = [requests.ConnectionError('reset'), process]
        self.client.get_jobs.side_effect = [http_error(503), self.page([])]
        self.assertIs(self.job.submit_once(), remote)
        process.submit.assert_called_once()
        self.assertEqual(self.job.saved['retry_count'], 0)
        self.assertEqual([event['operation'] for event in self.job.saved['network_wait_events']],
                         ['get_process', 'before_submission_list'])

    def test_unknown_submission_outcome_obeys_24_hour_limit(self):
        self.job.saved['network_wait_seconds'] = cds.NETWORK_WAIT_LIMIT_SECONDS - 1
        self.client.get_jobs.return_value = self.page([])
        process = Mock()
        process.submit.side_effect = requests.ReadTimeout('POST response lost')
        self.client.get_process.return_value = process
        with self.assertRaisesRegex(RuntimeError, '24小时'):
            self.job.submit_once()
        self.assertIn('submission_uncertain', self.load().saved)
        with self.assertRaisesRegex(RuntimeError, '24小时'):
            self.load().submit_once()
        process.submit.assert_called_once()

    def test_pagination_next_page_failure_retries_whole_read(self):
        first = self.page(['retained-id'])
        second = self.page(['second-id'])
        second.json['links'][0]['href'] += '?page=2'
        type(first).next = PropertyMock(side_effect=[requests.Timeout(), second])
        self.client.get_jobs.return_value = first
        jobs = self.job.network_call('list_jobs', lambda: cds.dataset_jobs(self.client))
        self.assertEqual(set(jobs), {'retained-id', 'second-id'})
        self.assertEqual(self.job.saved['network_wait_events'][0]['operation'], 'list_jobs')

    def test_explicit_queue_full_429_retains_queue_policy(self):
        self.client.get_jobs.return_value = self.page([])
        error = http_error(429, {'traceback': cds.QUEUE_FULL_REASON})
        self.client.get_process.return_value.submit.side_effect = error
        with self.assertRaises(requests.HTTPError):
            self.job.submit_once()
        self.assertNotIn('submission_uncertain', self.job.saved)
        self.assertNotIn('network_wait_events', self.job.saved)
        self.job.saved['retry_count'] = cds.MAX_RESUBMISSIONS
        self.job.retire('rejected', error.response.json())
        self.assertFalse(self.job.saved['retry_exhausted'])
        self.assertEqual(self.job.saved['retry_count'], cds.MAX_RESUBMISSIONS)

    def test_dismissed_and_ordinary_retry_budget(self):
        error = http_error(404, {'title': 'job not found', 'detail': 'job retained-id dismissed'})
        self.assertEqual(cds.dismissed_reason(error), error.response.json())
        self.job.saved['retry_count'] = cds.MAX_RESUBMISSIONS
        self.job.retire('dismissed', error.response.json())
        with self.assertRaisesRegex(RuntimeError, '达到上限'):
            self.job.resubmit()
        self.assertNotIn('network_wait_events', self.job.saved)

    def test_real_http_stream_failure_and_sdk_results_download(self):
        content = b'ERA5 network verification\n' * 6000
        request = self.request
        counts = {'status': 0, 'asset': 0, 'range_headers': []}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_GET(self):
                if self.path == '/asset':
                    counts['asset'] += 1
                    counts['range_headers'].append(self.headers.get('Range'))
                    self.send_response(200)
                    self.send_header('Content-Length', str(len(content)))
                    self.end_headers()
                    if counts['asset'] == 1:
                        self.wfile.write(content[:4096])
                        self.wfile.flush()
                        self.close_connection = True
                        return
                    self.wfile.write(content)
                    return
                if self.path.startswith('/jobs/retained-id') and not self.path.startswith('/jobs/retained-id/results'):
                    counts['status'] += 1
                    if counts['status'] == 1:
                        self.send_response(503)
                        self.send_header('Content-Length', '0')
                        self.end_headers()
                        return
                    reply = {'processID': cds.DATASET, 'status': 'successful',
                             'metadata': {'request': {'ids': request}},
                             'links': [{'rel': 'results', 'href': f'http://127.0.0.1:{self.server.server_port}/jobs/retained-id/results'}]}
                else:
                    reply = {'asset': {'value': {'href': '/asset', 'file:size': len(content),
                                                'type': 'application/octet-stream'}}}
                body = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        session = requests.Session()
        session.trust_env = False
        self.addCleanup(session.close)
        self.job.remote = Remote(url=f'http://127.0.0.1:{server.server_port}/jobs/retained-id',
                                 headers={}, session=session, retry_options={'maximum_tries': 1},
                                 request_options={'timeout': 5}, download_options={}, sleep_max=1,
                                 cleanup=False, log_callback=None)
        target = self.path.with_suffix('.nc.part')
        self.job.download(target)
        self.assertEqual(hashlib.sha256(target.read_bytes()).digest(), hashlib.sha256(content).digest())
        self.assertEqual(counts['asset'], 2)
        self.assertEqual(counts['range_headers'], [None, None])
        self.assertEqual([event['error_kind'] for event in self.job.saved['network_wait_events']],
                         ['http_503', 'ChunkedEncodingError'])
        self.assertEqual(self.job.saved['request_id'], 'retained-id')

    def test_download_size_integrity_failure_is_fatal(self):
        results = Mock(location='https://example.invalid/asset', content_length=2)
        self.job.remote = Mock()
        self.job.remote.get_results.return_value = results
        response = Mock(headers={})
        response.iter_content.return_value = [b'one']
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with patch.object(cds.requests, 'get', return_value=response):
            with self.assertRaisesRegex(ValueError, '下载大小不符'):
                self.job.download(self.path.with_suffix('.nc.part'))
        self.assertNotIn('network_wait_events', self.job.saved)


def verify_scientific_code(baseline):
    current = Path(__file__).with_name('prepare_era5_wind_repair.py').read_text(encoding='utf-8')
    original = baseline.read_text(encoding='utf-8')
    assert current[current.index('    records = [record for'):] == original[original.index('    records = [record for'):]
    for function in ('request_for',):
        old = next(node for node in ast.walk(ast.parse(original)) if isinstance(node, ast.FunctionDef) and node.name == function)
        new = next(node for node in ast.walk(ast.parse(current)) if isinstance(node, ast.FunctionDef) and node.name == function)
        assert ast.dump(old) == ast.dump(new)
    old = original[original.index("            record = {'dataset'"):original.index("        update(period, 'downloaded'")]
    assert old in current
    return True


def verify_live_state_compatibility(root, directory):
    checked = []
    for block in ('lat-091-lon-353', 'lat-144-lon-101', 'lat-160-lon-101', 'lat-168-lon-278'):
        paths = sorted((root / 'data/raw/era5-eligible-cds-repairs' / block).glob('*.request.json'))
        assert paths, f'缺少真实持久状态：{block}'
        for path in paths:
            before = json.loads(path.read_text())
            target = directory / 'compatibility' / block / path.name
            write_json(target, before)
            client = Mock()
            job = cds.PersistentCDSRequest(client, before['dataset'], before['request'], target,
                                           lambda *args, **kwargs: None, threading.Event(),
                                           gate=cds.CDSRequestGate(target.parent, client))
            for field in ('request_id', 'dataset', 'request', 'history', 'pending_submission', 'pending_resubmission'):
                assert job.saved.get(field) == before.get(field), (path, field)
            client.get_remote.assert_not_called()
            client.submit.assert_not_called()
            checked.append({'block': block, 'path': path.name, 'request_id': before.get('request_id'),
                            'history_entries_preserved': len(before.get('history', []))})
    return checked


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--state-root', type=Path)
    args = parser.parse_args()
    NetworkRetryChecks.directory = args.output.parent / (args.output.stem + '-state')
    NetworkRetryChecks.directory.mkdir(parents=True, exist_ok=False)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NetworkRetryChecks)
    with args.output.with_suffix('.events.jsonl').open('w', encoding='utf-8') as stream, redirect_stdout(stream):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'tests_run': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors),
              'status': 'passed' if result.wasSuccessful() else 'failed',
              'scientific_code_unchanged': verify_scientific_code(args.baseline) if args.baseline else None,
              'live_state_compatibility': verify_live_state_compatibility(
                  args.state_root, NetworkRetryChecks.directory) if args.state_root else None,
              'external_cds_calls': 0, 'new_cds_submissions': 0,
              'scope': '离线异常与时间模拟；真实本机HTTP连接及ECMWF Remote验证503、断流和完整重下；未访问真实CDS'}
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
