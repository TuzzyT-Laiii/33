import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pyarrow.parquet as pq

from audit_research_gpu_demand import tres_count
from compute_load.cooling import CoolingModel
from compute_load.model import sha256
from typical_power_models import reference_power


RUN = 'scenario-typical-day-by-task-20261031'
FORBIDDEN = {'id', 'id_job', 'job_id', 'user', 'id_user', 'pod_id', 'workload_id', 'server_id', 'nodelist', 'task_id'}


def direct_overlap(frame, start_column, end_column, units, start):
    left = pd.DatetimeIndex(frame[start_column]).as_unit('ns').asi8 / 1e9
    right = pd.DatetimeIndex(frame[end_column]).as_unit('ns').asi8 / 1e9
    hours = start.value / 1e9 + np.arange(24) * 3600
    # 独立使用逐作业交集秒数逐小时求和，核验事件积分的输出。
    return np.array([np.sum(np.maximum(0, np.minimum(right, hour + 3600) - np.maximum(left, hour)) * units) / 3600 for hour in hours])


def verify_intervals(args, summary, baseline, checks):
    local = args.project / 'data/processed' / RUN
    for use in ('HPC', 'SAI', 'TRN'):
        selection = summary['selections'][use]
        start = pd.Timestamp(selection['source_date'], tz='UTC') if use == 'TRN' else pd.Timestamp(selection['source_date'])
        expected = pd.read_parquet(local / (use + '-hourly-resource-fractions.parquet')).iloc[selection['day_index'] * 24:selection['day_index'] * 24 + 24]
        direct = {column: np.zeros(24) for column in expected.columns}
        if use == 'HPC':
            path = args.input_project / 'data/processed/task-trace-audit-20261026/lisa/lisa-tasks.parquet'
            columns = ['start_date', 'end_date', 'nodetypes', 'numcores', 'hardware_group']
            for batch in pq.ParquetFile(path).iter_batches(batch_size=32768, columns=columns, use_threads=False):
                frame = batch.to_pandas()
                frame = frame[frame.hardware_group.eq('cpu_partition')]
                for label, group in frame.groupby('nodetypes'):
                    category = 'Jupyter' if 'jupyter' in label else ('课程分区' if 'course' in label else 'CPU 科研分区')
                    direct['cpu:' + category] += direct_overlap(group, 'start_date', 'end_date', group.numcores, start)
            denominator = baseline['sources']['lisa_cpu']['reference_resource_count']
            for key in direct:
                direct[key] /= denominator
        elif use == 'SAI':
            path = args.mit_root / 'reference/slurm-log.csv'
            columns = ['time_start', 'time_end', 'time_suspended', 'tres_req', 'tres_alloc', 'job_type']
            names = {'LLSUB:BATCH': '批处理', 'LLMAPREDUCE:MAP': 'MapReduce', 'LLSUB:INTERACTIVE': '交互式', 'OTHER': 'OTHER'}
            for frame in pd.read_csv(path, usecols=columns, chunksize=20000, dtype={'tres_req': str, 'tres_alloc': str}):
                requested = sum(tres_count(frame.tres_req, key).fillna(0) for key in (1001, 1002))
                frame['gpu'] = sum(tres_count(frame.tres_alloc, key).fillna(0) for key in (1001, 1002))
                frame['cpu'] = tres_count(frame.tres_alloc, 1).fillna(0)
                selected = requested.gt(0) & frame.gpu.gt(0) & frame.time_start.gt(0) & frame.time_end.ge(frame.time_start) & frame.time_suspended.eq(0)
                frame = frame[selected].copy()
                frame['start'] = pd.to_datetime(frame.time_start, unit='s')
                frame['end'] = pd.to_datetime(frame.time_end, unit='s')
                for label, group in frame.groupby('job_type'):
                    for resource in ('cpu', 'gpu'):
                        direct[resource + ':' + names[label]] += direct_overlap(group, 'start', 'end', group[resource], start)
            for key in direct:
                direct[key] /= baseline['sources']['mit_' + key.split(':')[0]]['normalization_denominator']
        else:
            path = args.input_project / 'data/raw/task-trace-audit-20261026/trace_kalos.csv'
            frame = pd.read_csv(path)
            for column in ('start_time', 'end_time', 'submit_time'):
                frame[column] = pd.to_datetime(frame[column], utc=True, format='mixed')
            frame = frame[frame.end_time.ge(frame.start_time) & frame.start_time.ge(frame.submit_time)]
            names = {'Pretrain': '预训练', 'SFT': 'SFT', 'Eval': '评估', 'Debug': '调试', 'Other': '其他', 'CPU': '其他'}
            for resource in ('cpu', 'gpu'):
                selected = frame[frame.gpu_num.eq(0) if resource == 'cpu' else frame.gpu_num.gt(0)]
                for label, group in selected.groupby('type'):
                    direct[resource + ':' + names[label]] += direct_overlap(group, 'start_time', 'end_time', group[resource + '_num'], start)
                for key in direct:
                    if key.startswith(resource + ':'):
                        direct[key] /= baseline['sources']['acme_' + resource]['reference_resource_count']
            cpu = sum(value for key, value in direct.items() if key.startswith('cpu:'))
            gpu = sum(value for key, value in direct.items() if key.startswith('gpu:'))
            for key in list(direct):
                if key.startswith('cpu:'):
                    gpu_key = key.replace('cpu:', 'gpu:')
                    direct[key] = np.where(.25 * gpu >= cpu, .25 * direct[gpu_key], direct[key])
        for column in expected:
            np.testing.assert_allclose(direct[column], expected[column], rtol=1e-7, atol=2e-7)
        checks[use + '_direct_job_hour_overlap'] = True


def verify(args):
    output = args.project / 'results' / RUN
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    baseline = json.loads((args.project / 'results/typical-load-curves-20261029-v2/manifest.json').read_text(encoding='utf-8'))
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    checks = {}
    for entry in manifest['input_files']:
        if sha256(Path(entry['file'])) != entry['sha256']:
            raise AssertionError('输入在处理后发生变化：' + entry['file'])
    checks['input_hashes_unchanged'] = True
    verify_intervals(args, summary, baseline, checks)
    configs = {row['configuration_id']: row for row in json.loads((args.project / 'results/server-count-scenarios-20261028-v2/server-configurations.json').read_text(encoding='utf-8'))}
    old_keys = {'A': 'C20', 'B': 'C21', 'C': 'C22', 'D': 'C28'}
    for row in summary['scenarios']:
        scenario = row['scenario']
        hourly = pd.read_csv(output / (scenario + '-hourly.csv'))
        tasks = pd.read_csv(output / (scenario + '-by-task.csv'))
        assert not (set(tasks.columns) | set(hourly.columns)) & FORBIDDEN
        assert hourly.hour.tolist() == list(range(24))
        assert tasks.groupby(['hour', 'task_type']).size().eq(1).all()
        for column in ('it_mw', 'facility_mw'):
            grouped = tasks.groupby('hour')[column].sum()
            np.testing.assert_allclose(grouped, hourly[column], atol=1e-8)
            assert np.isfinite(tasks[column]).all() and tasks[column].ge(-1e-8).all()
        np.testing.assert_allclose(hourly.facility_mw.sum(), row['daily_facility_mwh'])
        np.testing.assert_allclose([hourly.facility_mw.max(), hourly.facility_mw.min()], [row['peak_mw'], row['minimum_mw']])
        assert .5 < row['daily_to_annual_mean_ratio'] < 2
        with Image.open(output / (scenario + '.png')) as img:
            assert min(img.info['dpi']) > 299
        assert (output / (scenario + '.svg')).stat().st_size > 10000
        checks[scenario + '_csv_energy_components_dpi'] = True
        use = scenario.split('-')[1]
        source = next(entry for entry in manifest['input_files'] if entry['role'] == scenario + '_v2_curve') if use != 'CLD' else next(entry for entry in manifest['input_files'] if entry['role'] == 'A-HPC_v2_curve')
        annual = pd.read_parquet(source['file'])
        weather = annual.iloc[summary['selections'][use]['day_index'] * 24:summary['selections'][use]['day_index'] * 24 + 24]
        config = configs[old_keys[scenario.split('-')[0]]]
        units = int(10e6 // reference_power(config))
        for scale in (1, 100, 500):
            scaled_units = int(scale * 1e6 // reference_power(config))
            factor = scaled_units / units
            scaled = CoolingModel().calculate(weather, hourly.it_mw.to_numpy() * factor,
                hourly.installed_reference_it_mw.iloc[0] * factor, full_year=False)
            np.testing.assert_allclose(scaled.facility_mw, hourly.facility_mw * factor, rtol=1e-9)
            np.testing.assert_allclose(scaled.pue, hourly.pue, rtol=1e-9)
        checks[scenario + '_scale_shape_and_pue'] = True
    for use in summary['selections']:
        table = pd.read_csv(output / (use + '-day-selection.csv'))
        distance = table.absolute_distance_mwh.to_numpy()
        candidates = table[np.isclose(distance, distance.min(), atol=1e-9, rtol=0)]
        chosen = candidates.sort_values(['rmse_mw', 'day_index']).iloc[0]
        assert int(chosen.day_index) == summary['selections'][use]['day_index']
        checks[use + '_selection_ranking'] = True
    for use in ('SAI', 'TRN', 'INF'):
        rows = [row for row in summary['scenarios'] if row['scenario'].endswith('-' + use)]
        assert len(rows) == 3
        assert len({row['source_date'] for row in rows}) == 1
        assert all(row['composition'] == rows[0]['composition'] for row in rows)
    checks['three_specifications_share_day_and_composition'] = True
    result = {'status': 'passed', 'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'checks': checks,
        'scope': '真实作业逐小时交集独立核验；CSV守恒、选日排序、输入哈希、尺度与PUE、300dpi'}
    (output / 'independent-validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'checks_passed': len(checks)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--input-project', type=Path, required=True)
    parser.add_argument('--mit-root', type=Path, required=True)
    verify(parser.parse_args())
