import argparse
import ast
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
from io import StringIO
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt
import numpy as np
import pandas as pd


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def number(cell, expected, exact=False):
    value = str(cell).strip().replace(',', '').removesuffix('%')
    if pd.isna(expected):
        assert value == '缺失', (cell, expected)
        return
    actual = Decimal(value)
    tolerance = 0 if exact else float(Decimal(10) ** actual.as_tuple().exponent) / 2 + 1e-7
    assert abs(float(actual) - float(expected)) <= tolerance, (cell, expected, tolerance)


def sequence(cell, expected):
    pieces = str(cell).split('/')
    assert len(pieces) == len(expected), (cell, expected)
    for piece, value in zip(pieces, expected):
        number(piece, value)


def quantiles(row, prefix, suffixes=('p50', 'p90', 'p99')):
    return [row[prefix + '_' + suffix] for suffix in suffixes]


def tables(path):
    parser = MarkdownIt().enable('table')
    text = path.read_text(encoding='utf-8')
    return pd.read_html(StringIO(parser.render(text)), keep_default_na=False)


def find_table(items, column):
    chosen = [table for table in items if column in table.columns]
    assert len(chosen) == 1, (column, len(chosen))
    return chosen[0]


def validate_readme(root, jobs, asi, server_audit):
    items = tables(root / 'README.md')
    acme = jobs[jobs.dataset.eq('Acme_Kalos') & jobs.resource_pool.eq('gpu_hours')].set_index('class')
    table = find_table(items, '真实标签')
    assert len(table) == len(acme)
    for cells in table.itertuples(index=False, name=None):
        row = acme.loc[cells[0].split('：')[0]]
        for cell, value, exact in zip(cells[1:5], [row['rows'], row.row_fraction_of_resource_pool * 100,
                  row.resource_hours, row.work_fraction_of_resource_pool * 100], [True, False, False, False]):
            number(cell, value, exact)
        sequence(cells[5], quantiles(row, 'runtime_s'))
        sequence(cells[6], quantiles(row, 'wait_s'))
        sequence(cells[7], quantiles(row, 'units', ('p50', 'p90', 'max')))
    table = find_table(items, '原始类型')
    assert len(table) == len(asi)
    for cells in table.itertuples(index=False, name=None):
        row = asi.loc[cells[0].split('：')[0]]
        for cell, value, exact in zip(cells[1:6], [row.pod_hour_observations, row.observation_fraction * 100,
                row.distinct_pods, row.used_gpu_hours, row.gpu_hour_fraction * 100], [True, False, True, False, False]):
            number(cell, value, exact)
        sequence(cells[6], quantiles(row, 'gpu_request_per_pod_server_hour'))
        sequence(cells[7], quantiles(row, 'first_observed_pod_schedule_delay_s'))
    lisa = jobs[jobs.dataset.eq('Lisa')].set_index('class')
    table = find_table(items, 'CPU分区原始标签')
    seen = set()
    for cells in table.itertuples(index=False, name=None):
        for offset in [0, 3]:
            if cells[offset] == '—':
                continue
            row = lisa.loc[cells[offset]]
            number(cells[offset + 1], row['rows'], exact=True)
            number(cells[offset + 2], row.work_fraction_of_resource_pool * 100)
            seen.add(cells[offset])
    assert seen == set(lisa.index)
    mit = jobs[jobs.dataset.eq('MIT_Supercloud')]
    table = find_table(items, '原始 job_type')
    assert len(table) == mit['class'].nunique()
    for cells in table.itertuples(index=False, name=None):
        group = mit[mit['class'].eq(cells[0])].set_index('resource_pool')
        gpu, cpu = group.loc['gpu_hours'], group.loc['independent_slurm_cpu_hours']
        number(cells[1], gpu['rows'], exact=True)
        number(cells[2], gpu.work_fraction_of_resource_pool * 100)
        number(cells[3], cpu['rows'], exact=True)
        number(cells[4], cpu.work_fraction_of_resource_pool * 100)
        sequence(cells[5], quantiles(gpu, 'runtime_s'))
        sequence(cells[6], quantiles(gpu, 'wait_s', ('p50', 'p90')))
    repeats = pd.DataFrame(server_audit['details']['Acme_Kalos']['repeated_submission_candidates']).set_index('class')
    table = find_table(items, '弱关联后续作业数')
    for cells in table.itertuples(index=False, name=None):
        row = repeats.loc[cells[0]]
        for cell, key in zip(cells[1:], ['candidate_successor_rows', 'successor_completed_rows', 'candidate_gap_s_p50']):
            number(cell, row[key], exact=True)
    lp = pd.read_csv(root / 'asi-priorities.csv').query('priority == "LP"').set_index('class')
    table = find_table(items, '本类型LP归属GPU小时')
    for cells in table.itertuples(index=False, name=None):
        row = lp.loc[cells[0]]
        number(cells[1], row.used_gpu_hours)
        number(cells[2], row.fraction_class_gpu_hours * 100)
        number(cells[3], row.fraction_all_gpu_hours * 100)
    table = find_table(items, '每段总开销H（假设）')
    for cells in table.itertuples(index=False, name=None):
        overhead = float(cells[0].removesuffix('分钟'))
        number(cells[1].removesuffix('分钟'), overhead * 19, exact=True)
        number(cells[2].removesuffix('分钟'), overhead * 20, exact=True)
        number(cells[3].removesuffix('小时'), np.ceil(overhead * 20 / 60), exact=True)
    return 7


def validate_statistics(root, jobs, asi, training):
    items = tables(root / 'statistics.md')
    cursor = 0
    for _, group in jobs.groupby(['dataset', 'resource_pool']):
        counts, scales = items[cursor:cursor + 2]
        cursor += 2
        assert len(counts) == len(scales) == len(group)
        for cells, (_, row) in zip(counts.itertuples(index=False, name=None), group.iterrows()):
            assert cells[0] == row['class']
            for cell, value, exact in zip(cells[1:5], [row['rows'], row.row_fraction_of_resource_pool * 100,
                    row.resource_hours, row.work_fraction_of_resource_pool * 100], [True, False, False, False]):
                number(cell, value, exact)
            sequence(cells[5], quantiles(row, 'runtime_s'))
            sequence(cells[6], quantiles(row, 'wait_s'))
        for cells, (_, row) in zip(scales.itertuples(index=False, name=None), group.iterrows()):
            assert cells[0] == row['class']
            sequence(cells[1], quantiles(row, 'units', ('p50', 'p90', 'p99', 'max')))
            sequence(cells[2], quantiles(row, 'nodes', ('p50', 'p90', 'p99', 'max')))
            sequence(cells[3], [row.valid_runtime_rows, row.invalid_runtime_rows])
            for cell, key in zip(cells[4:], ['runtime_under60_row_fraction', 'runtime_under60_work_fraction', 'runtime_ge1h_work_fraction']):
                number(cell, row[key] * 100)
    for cells in items[cursor].itertuples(index=False, name=None):
        row = asi.loc[cells[0]]
        for cell, key, factor in zip(cells[1:6], ['pod_hour_observations', 'observation_fraction',
                'distinct_pods', 'used_gpu_hours', 'gpu_hour_fraction'], [1, 100, 1, 1, 100]):
            number(cell, row[key] * factor, exact=key in ['pod_hour_observations', 'distinct_pods'])
        sequence(cells[6], quantiles(row, 'gpu_request_per_pod_server_hour', ('p50', 'p90', 'p99', 'max')))
        sequence(cells[7], quantiles(row, 'first_observed_pod_schedule_delay_s'))
    assert len(items) == cursor + 2
    for cells in items[-1].itertuples(index=False, name=None):
        row = training.loc[cells[0]]
        number(cells[1], row.pod_hour_observations, exact=True)
        number(cells[2], row.used_gpu_hours)
        number(cells[3], row.gpu_hour_fraction_within_training * 100)
    return len(items)


def verify(root):
    analysis = Path(__file__).resolve().parent
    expected = {f'{prefix}-{kind}.csv' for prefix in ['server', 'mit']
                for kind in ['classes', 'states', 'scales', 'durations']}
    expected.update('asi-' + name + '.csv' for name in ['classes', 'priorities', 'states', 'presence',
                    'workload-scales', 'inventory', 'hourly-type-priority', 'model-type-priority', 'training-model-types'])
    expected.update({'candidate-bounds.csv', 'checkpoint-sensitivity.csv', 'hourly-candidate-bounds.csv',
        'literature-evidence.csv', 'model-constraints.csv', 'server-audit.json', 'mit-audit.json',
        'asi-training-model-audit.json', 'tables-manifest.json', 'validation.json', 'README.md', 'statistics.md',
        'report-validation.json', 'delivery-manifest.json', 'source-readback-validation.json', 'literature-link-check.json'})
    generated = {'report-validation.json', 'delivery-manifest.json'}
    files = sorted(path for path in root.iterdir() if path.is_file())
    names = {path.name for path in files}
    assert expected - generated <= names <= expected, (expected - generated - names, names - expected)
    assert files and all(path.stat().st_size > 0 for path in files)
    for path in files:
        text = path.read_text(encoding='utf-8')
        assert '\ufffd' not in text and '\x00' not in text, path
        if path.suffix == '.json':
            json.loads(text)
        elif path.suffix == '.csv':
            frame = pd.read_csv(path)
            assert len(frame) > 0 and not frame.columns.str.startswith('Unnamed:').any(), path
    jobs = pd.concat([pd.read_csv(root / f'{prefix}-classes.csv') for prefix in ['server', 'mit']], ignore_index=True)
    asi = pd.read_csv(root / 'asi-classes.csv').set_index('class')
    training = pd.read_csv(root / 'asi-training-model-types.csv').set_index('model_type_public')
    domains = pd.read_csv(root / 'asi-model-type-priority.csv')
    for key, group in domains.groupby('job_type_public'):
        number(group.pod_hour_observations.sum(), asi.loc[key, 'pod_hour_observations'], exact=True)
        np.testing.assert_allclose(group.used_gpu_hours.sum(), asi.loc[key, 'used_gpu_hours'], rtol=1e-11)
    source_training = domains[domains.job_type_public.eq('training')].groupby('model_type_public')[['pod_hour_observations', 'used_gpu_hours']].sum()
    np.testing.assert_allclose(source_training.sort_index(), training[source_training.columns].sort_index(), rtol=1e-11)
    np.testing.assert_allclose(training.gpu_hour_fraction_within_training, training.used_gpu_hours / training.used_gpu_hours.sum())
    assert set(training.index) == {'genai', 'rec', 'unknown'}
    server_audit = json.loads((root / 'server-audit.json').read_text(encoding='utf-8'))
    for name in ['server-audit.json', 'mit-audit.json']:
        audit = json.loads((root / name).read_text(encoding='utf-8'))
        assert audit['analysis_sha256'] == sha256(analysis / 'audit_job_flexibility_constraints.py')
        assert audit['input_metadata_unchanged'] is True
    model_audit = json.loads((root / 'asi-training-model-audit.json').read_text(encoding='utf-8'))
    assert model_audit['analysis_sha256'] == sha256(analysis / 'audit_asi_training_subtypes.py')
    recorded = {entry['path']: entry for entry in server_audit['inputs']}
    assert len(model_audit['source_metadata']) == 24
    for entry in model_audit['source_metadata']:
        assert all(entry[key] == recorded[entry['path']][key] for key in ['bytes', 'mtime_ns'])
    manifest = json.loads((root / 'tables-manifest.json').read_text(encoding='utf-8'))
    assert manifest['generator_sha256'] == sha256(analysis / 'build_job_flexibility_tables.py')
    assert all(sha256(root / name) == digest for name, digest in manifest['inputs'].items())
    access = json.loads((root / 'literature-link-check.json').read_text(encoding='utf-8'))
    literature = pd.read_csv(root / 'literature-evidence.csv')
    assert {entry['url'] for entry in access['checks']} == set(literature.url)
    assert all(entry.get('status') == 200 and entry.get('bytes', 0) > 0 for entry in access['checks'])
    readme_tables = validate_readme(root, jobs, asi, server_audit)
    statistics_tables = validate_statistics(root, jobs, asi, training)
    local_links = []
    parser = MarkdownIt().enable('table')
    for name in ['README.md', 'statistics.md']:
        for token in parser.parse((root / name).read_text(encoding='utf-8')):
            for child in token.children or []:
                if child.type != 'link_open':
                    continue
                target = urlsplit(child.attrGet('href'))
                if not target.scheme and target.path:
                    destination = root / unquote(target.path)
                    if target.path not in {'report-validation.json', 'delivery-manifest.json'}:
                        assert destination.is_file(), target.path
                    local_links.append(destination)
    # 公开数据只允许类别及统计量，不发布逐作业键、状态轨迹或时间戳。
    forbidden = {'job_id', 'id_job', 'user', 'id_user', 'pod_id', 'workload_id', 'server_id',
                 'nodelist', 'start_time', 'end_time', 'time_submit', 'time_start', 'time_end', 'tres_req', 'tres_alloc'}
    for path in root.glob('*.csv'):
        assert not forbidden.intersection(pd.read_csv(path, nrows=0).columns), path
    constraints = pd.read_csv(root / 'model-constraints.csv').set_index('constraint_id')
    candidates = pd.read_csv(root / 'candidate-bounds.csv').set_index('screen')
    np.testing.assert_allclose(float(constraints.loc['asi_lp_combined', 'value']), candidates.loc['LP_training_and_offline', 'candidate_ceiling_fraction'])
    for label in ['training', 'offline_inference']:
        key = 'asi_lp_training' if label == 'training' else 'asi_lp_offline'
        np.testing.assert_allclose(float(constraints.loc[key, 'value']), candidates.loc['LP_within_' + label, 'candidate_ceiling_fraction'])
    sensitivity = pd.read_csv(root / 'checkpoint-sensitivity.csv')
    np.testing.assert_allclose(sensitivity.min_useful_segment_min,
        (1 - sensitivity.max_overhead_fraction) / sensitivity.max_overhead_fraction * sensitivity.overhead_min)
    scripts = [analysis / name for name in ['audit_job_flexibility_constraints.py', 'audit_asi_training_subtypes.py',
               'build_job_flexibility_tables.py', 'verify_job_flexibility_report.py']]
    for path in scripts:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    checks = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
        'nonempty_utf8_json_csv': 'passed', 'expected_delivery_files': len(expected),
        'readme_numeric_tables_checked': readme_tables,
        'statistics_numeric_tables_checked': statistics_tables, 'asi_domain_conservation': 'passed',
        'source_metadata_and_analysis_hashes': 'passed', 'generator_input_hashes': 'passed',
        'local_links_checked': len(local_links), 'public_csv_fields': 'passed', 'constraint_values': 'passed',
        'literature_urls_accessible': len(access['checks']),
        'python_syntax': 'passed', 'scope': '本次报告和真实聚合；不运行能源优化或后台采集'}
    write_json(root / 'report-validation.json', checks)
    delivery = {'created_at_utc': checks['checked_at_utc'], 'base_commit': 'e7b03cb8d0476bbc8f489fac599b9b6fcd460b8b',
        'files': {path.name: {'bytes': path.stat().st_size, 'sha256': sha256(path)}
                  for path in sorted(root.iterdir()) if path.is_file() and path.name != 'delivery-manifest.json'},
        'analysis_files': {path.name: sha256(path) for path in scripts + [analysis / 'requirements-job-flexibility-constraints.txt']}}
    write_json(root / 'delivery-manifest.json', delivery)
    assert all(path.is_file() for path in local_links)
    assert {path.name for path in root.iterdir() if path.is_file()} == expected
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    verify(args.results)
