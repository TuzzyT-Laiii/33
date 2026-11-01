import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compute_load.model import sha256
from task_trace_inputs import interval_resource_hours
from summarize_seren_trn_power import statistics


RUN = 'trn-seren-20261101'
TYPES = ('Pretrain', 'MLLM', 'Other', 'SFT', 'Eval', 'Debug')
INVENTORY = 2288


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def quality(frame, coverage=.8):
    return (frame.physical_device_coverage.ge(coverage) & frame.overlap_samples.ge(228)
            & frame.sample_matched_allocated_gpus.gt(0) & frame.inventory_excess_samples.eq(0))


def prepare(args):
    output = args.project / 'results' / RUN
    output.mkdir(parents=True, exist_ok=True)
    evidence = args.project / 'results/trn-power-model-20261101/job-power-evidence/seren'
    old_audit = json.loads((evidence / 'seren-audit.json').read_text('utf-8'))
    digest = sha256(args.jobs)
    if digest != old_audit['job_sha256']:
        raise ValueError('Seren作业表与#36核验版本不同')
    jobs = pd.read_csv(args.jobs, usecols=['job_id', 'gpu_num', 'cpu_num', 'type', 'start_time', 'end_time', 'duration'])
    if jobs.job_id.isna().any() or jobs.job_id.duplicated().any():
        raise ValueError('作业标识缺失或重复')
    start, end = [pd.to_datetime(jobs[name], utc=True, format='mixed') for name in ('start_time', 'end_time')]
    if start.isna().any() or end.isna().any() or end.lt(start).any() or (jobs[['gpu_num', 'cpu_num']] < 0).any().any():
        raise ValueError('作业时间或分配资源无效')
    np.testing.assert_array_equal((end - start).dt.total_seconds(), jobs.duration)
    edges = pd.date_range(start.min().floor('h'), end.max().ceil('h'), freq='h')
    stamps, stops = start.astype('int64').to_numpy(), end.astype('int64').to_numpy()
    gpu_jobs = jobs.gpu_num.gt(0).to_numpy()
    source = pd.DataFrame({'source_clock': edges[:-1], 'source_hour_index': np.arange(len(edges) - 1)})

    def integral(mask, column):
        return interval_resource_hours(stamps[mask], stops[mask], jobs.loc[mask, column].to_numpy(float), edges.asi8)

    source['allocated_gpu_hours'] = integral(gpu_jobs, 'gpu_num')
    source['host_cpu_core_hours'] = integral(gpu_jobs, 'cpu_num')
    source['independent_cpu_core_hours'] = integral(~gpu_jobs, 'cpu_num')
    for kind in TYPES:
        selected = gpu_jobs & jobs.type.eq(kind).to_numpy()
        source['gpu_hours_' + kind] = integral(selected, 'gpu_num')
        source['host_cpu_core_hours_' + kind] = integral(selected, 'cpu_num')
    np.testing.assert_allclose(source.filter(like='gpu_hours_').sum(axis=1), source.allocated_gpu_hours, atol=1e-6)
    np.testing.assert_allclose(source.filter(like='host_cpu_core_hours_').sum(axis=1), source.host_cpu_core_hours, atol=1e-6)
    np.testing.assert_allclose(source.allocated_gpu_hours.sum(), np.sum(jobs.loc[gpu_jobs, 'gpu_num'] * jobs.loc[gpu_jobs, 'duration'] / 3600), rtol=1e-12)
    event_time = np.r_[stamps[gpu_jobs], stops[gpu_jobs]]
    changes = np.r_[jobs.loc[gpu_jobs, 'gpu_num'], -jobs.loc[gpu_jobs, 'gpu_num']]
    # 同时刻先合并开始和结束，避免把零时长状态计作并发峰值。
    peak = float(pd.Series(changes).groupby(event_time).sum().sort_index().cumsum().max())
    source['allocated_gpu_fraction'] = source.allocated_gpu_hours / peak
    source['u_cpu_proxy'] = .25 * source.allocated_gpu_fraction
    source['hour_of_source_day'] = source.source_clock.dt.hour
    source['hour_of_source_week'] = source.source_clock.dt.dayofweek * 24 + source.hour_of_source_day
    source['source_partial_boundary_hour'] = (source.source_clock.lt(start.min()) | source.source_clock.add(pd.Timedelta(hours=1)).gt(end.max()))
    monitor_path = evidence / 'seren-hourly.csv'
    expected = json.loads((evidence.parent / 'artifact-manifest.json').read_text('utf-8'))['artifact_sha256']['seren/seren-hourly.csv']
    if sha256(monitor_path) != expected:
        raise ValueError('Seren监测小时汇总哈希不符')
    monitor = pd.read_csv(monitor_path, parse_dates=['time_utc'])
    base = source.merge(monitor.drop(columns=['allocated_gpu_count', *['allocated_' + x for x in TYPES]]),
                        left_on='source_clock', right_on='time_utc', how='left', validate='one_to_one').drop(columns='time_utc')
    base['power_quality_80'] = quality(base)
    ratio = base['inverse_ratio_idle_60.000']
    if not ratio[base.power_quality_80].ge(0).all():
        raise ValueError('合格小时反推功率为负')
    base['power_ratio_above_tdp'] = ratio.gt(1)
    base['source_dataset'] = 'Acme Seren'
    base.to_csv(output / 'source-hourly.csv', index=False)
    profiles = []
    central = base.loc[base.power_quality_80]
    sample_weights = central.sample_matched_allocated_gpus * central.overlap_samples / 240
    for grouping, count in [('hour_of_source_day', 24), ('hour_of_source_week', 168)]:
        for position, group in central.groupby(grouping):
            w = group.sample_matched_allocated_gpus * group.overlap_samples / 240
            profiles.append({'grouping': grouping, 'position': int(position),
                             **statistics(group['inverse_ratio_idle_60.000'].to_numpy(), w.to_numpy())})
        if central[grouping].nunique() != count:
            raise ValueError('合格窗口未覆盖全部日或周分组')
    pd.DataFrame(profiles).to_csv(output / 'power-profiles.csv', index=False)
    composition = []
    for window, mask in [('full_jobs', np.ones(len(base), bool)), ('qualified_overlap', base.power_quality_80.to_numpy())]:
        selected = base.loc[mask]
        for kind in TYPES:
            composition.append({'window': window, 'type': kind, 'gpu_hours': selected['gpu_hours_' + kind].sum(),
                'gpu_share': selected['gpu_hours_' + kind].sum() / selected.allocated_gpu_hours.sum(),
                'host_cpu_core_hours': selected['host_cpu_core_hours_' + kind].sum(),
                'host_cpu_share': selected['host_cpu_core_hours_' + kind].sum() / selected.host_cpu_core_hours.sum()})
    pd.DataFrame(composition).to_csv(output / 'task-composition.csv', index=False)
    central_clock = pd.DatetimeIndex(central.source_clock)
    full_grid = pd.date_range(central_clock[0], central_clock[-1], freq='h')
    missing = full_grid.difference(central_clock)
    pd.DataFrame({'source_clock': missing, 'reason': 'not_qualified_power_hour_no_fill_in_body'}).to_csv(output / 'body-excluded-hours.csv', index=False)
    group_id = pd.Series(np.r_[True, np.diff(central_clock.asi8) != 3600 * 10**9]).cumsum().to_numpy()
    segments = central.assign(segment=group_id).groupby('segment').agg(start=('source_clock', 'min'), end_inclusive=('source_clock', 'max'), hours=('source_clock', 'size'))
    segments.to_csv(output / 'body-segments.csv')
    power_stats = statistics(central['inverse_ratio_idle_60.000'].to_numpy(), sample_weights.to_numpy())
    save_json(output / 'source-audit.json', {'utc': datetime.now(timezone.utc).isoformat(), 'input_commit': args.input_commit,
        'job_sha256': digest, 'monitor_aggregate_sha256': sha256(monitor_path), 'rows': len(jobs), 'gpu_jobs': int(gpu_jobs.sum()),
        'independent_cpu_jobs': int((~gpu_jobs).sum()), 'raw_job_start': start.min().isoformat(), 'raw_job_end': end.max().isoformat(),
        'full_source_hours': len(source), 'full_source_start': edges[0].isoformat(), 'full_source_end_exclusive': edges[-1].isoformat(),
        'concurrent_gpu_peak_normalizer': peak, 'physical_gpu_inventory': INVENTORY, 'source_gpu': 'A100 SXM 80GB', 'source_tdp_w': 400,
        'target_gpu': {'B': 'H100 PCIe 350W', 'C': 'H100 PCIe 350W', 'D': 'H100 SXM 700W'},
        'cpu_model': 'Same Kalos convention: u_cpu_proxy=0.25*allocated_gpu_fraction; measured host allocations exported as diagnostics; independent CPU jobs excluded from TRN power.',
        'normalization': 'GPU-hours divided by full-trace instantaneous concurrent GPU peak, as in the Kalos source; not physical fleet fraction.',
        'calendar_overlap_start': old_audit['monitoring_start_utc'], 'calendar_overlap_end': old_audit['monitoring_end_utc'],
        'qualified_hours': len(central), 'qualified_start': central_clock[0].isoformat(), 'qualified_end_inclusive': central_clock[-1].isoformat(),
        'interior_excluded_hours': len(missing), 'contiguous_segments': len(segments), 'longest_segment_hours': int(segments.hours.max()),
        'qualified_sample_equivalent_hours': float(central.overlap_samples.sum() / 240), 'central_power_statistics': power_stats,
        'full_gpu_hours': float(source.allocated_gpu_hours.sum()), 'qualified_gpu_hours': float(central.allocated_gpu_hours.sum()),
        'host_cpu_core_hours': float(source.host_cpu_core_hours.sum()), 'independent_cpu_core_hours': float(source.independent_cpu_core_hours.sum()),
        'inverse_assumptions': 'All 2288 physical GPUs online; missing reporters represented by reporting mean; subtract 60W per unallocated A100. No per-job/device power map.',
        'source_raw_files': old_audit['monitoring_sha256'], 'raw_files_copied_to_local_or_Git': False})
    print(json.dumps({'full_hours': len(base), 'qualified_hours': len(central), 'peak_normalizer': peak, 'power': power_stats}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--input-commit', default='6de2e4cde4dbf66f055c0802364c96e511945bbe')
    prepare(parser.parse_args())
