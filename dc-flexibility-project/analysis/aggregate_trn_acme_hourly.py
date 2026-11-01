import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compute_load.model import sha256
from task_trace_inputs import interval_resource_hours


INVENTORY, TDP = 2416, 400.


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def job_events(path):
    jobs = pd.read_csv(path)
    if jobs.job_id.isna().any() or jobs.job_id.duplicated().any():
        raise ValueError('作业标识缺失或重复')
    starts = pd.to_datetime(jobs.start_time, utc=True, format='mixed')
    ends = pd.to_datetime(jobs.end_time, utc=True, format='mixed')
    submitted = pd.to_datetime(jobs.submit_time, utc=True, format='mixed')
    censored = ends.isna() & jobs.state.eq('RUNNING') & starts.notna() & jobs.duration.gt(0)
    snapshot = submitted[censored] + pd.to_timedelta(jobs.loc[censored, 'duration'], unit='s')
    if len(snapshot) != 3 or (snapshot.max() - snapshot.min()).total_seconds() > 1:
        raise ValueError('运行中作业的共同导出时刻依据不符')
    # duration 包含排队；submit_time + duration 给出共同导出时刻，结束时间仍为右删失。
    ends.loc[censored] = snapshot
    valid = jobs.gpu_num.gt(0) & starts.notna() & ends.notna() & ends.ge(starts)
    selected = jobs.loc[valid].copy()
    start, end = starts[valid].astype('int64').to_numpy(), ends[valid].astype('int64').to_numpy()
    units = selected.gpu_num.to_numpy(float)
    event_time = np.r_[start, end]
    order = np.argsort(event_time, kind='stable')
    event_time = event_time[order]
    cumulative = np.r_[0., np.cumsum(np.r_[units, -units][order])]
    def allocation(stamps):
        return cumulative[np.searchsorted(event_time, stamps, side='right')]
    begin = starts.min().floor('h')
    stop = ends.max().ceil('h')
    edges = pd.date_range(begin, stop, freq='h')
    hourly = pd.DataFrame({'time_utc': edges[:-1],
        'allocated_gpu_count_time_mean': interval_resource_hours(start, end, units, edges.asi8)})
    pretrain = selected.type.eq('Pretrain').to_numpy()
    hourly['pretrain_gpu_count_time_mean'] = interval_resource_hours(start[pretrain], end[pretrain], units[pretrain], edges.asi8)
    actual_end = ~censored[valid].to_numpy()
    for name, stamps, weights in [('starts', start, units), ('ends', end[actual_end], units[actual_end])]:
        labels = pd.to_datetime(stamps, utc=True).floor('h')
        group = pd.DataFrame({'time_utc': labels, 'gpu': weights}).groupby('time_utc')
        hourly[name + '_jobs'] = hourly.time_utc.map(group.size()).fillna(0).astype(int)
        hourly[name + '_gpu'] = hourly.time_utc.map(group.gpu.sum()).fillna(0.)
    return allocation, hourly, {'rows': len(jobs), 'gpu_jobs': int(jobs.gpu_num.gt(0).sum()),
        'valid_gpu_intervals': int(valid.sum()), 'invalid_or_missing_gpu_intervals': int((jobs.gpu_num.gt(0) & ~valid).sum()),
        'right_censored_running_gpu_jobs': int((censored & jobs.gpu_num.gt(0)).sum()),
        'right_censor_snapshot_min_utc': snapshot.min().isoformat(), 'right_censor_snapshot_max_utc': snapshot.max().isoformat(),
        'right_censor_rule': 'RUNNING with missing end: submit_time + duration; three inferred export instants agree within one second; no extension beyond that instant.',
        'start_utc': starts.min().isoformat(), 'end_utc': ends.max().isoformat(),
        'job_sha256': sha256(path), 'maximum_hourly_allocation': float(hourly.allocated_gpu_count_time_mean.max())}


def aggregate(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    allocation, jobs, audit = job_events(args.raw / 'github/data/job_trace/trace_kalos.csv')
    jobs.to_csv(output / 'kalos-job-hourly.csv', index=False)
    paths = [args.raw / 'hf/data/utilization/kalos' / f'{name}.csv' for name in ('POWER_USAGE', 'GPU_UTIL')]
    headers = [pd.read_csv(path, nrows=0).columns.tolist() for path in paths]
    if set(headers[0]) != set(headers[1]) or headers[0][0] != 'Time':
        raise ValueError('Kalos 两份监测的设备总体不一致')
    devices = headers[0][1:]
    readers = [pd.read_csv(path, chunksize=args.chunk_rows, dtype={key: np.float32 for key in devices}) for path in paths]
    tables = defaultdict(lambda: np.zeros(10, dtype=np.float64))
    device_count = np.zeros(len(devices), dtype=np.int64)
    panel_sum = np.zeros((24, len(devices)), dtype=np.float64)
    panel_count = np.zeros((24, len(devices)), dtype=np.int64)
    row_count = 0
    zero_hist = np.zeros(2001, dtype=np.int64)
    zero_count = 0
    zero_sum = 0.
    previous = None
    first = None
    job_stop = pd.Timestamp(audit['end_utc']).value
    job_start = pd.Timestamp(audit['start_utc']).value
    for power, util in zip(*readers, strict=True):
        times = pd.to_datetime(power.pop('Time'), utc=True)
        if not times.equals(pd.to_datetime(util.pop('Time'), utc=True)):
            raise ValueError('Kalos 两份监测的时间行不一致')
        stamps = times.astype('int64').to_numpy()
        if first is None:
            first = times.iloc[0]
        if np.any(np.diff(np.r_[previous, stamps] if previous is not None else stamps) <= 0):
            raise ValueError('监测时间重复或倒序')
        previous = stamps[-1]
        p, u = power[devices].to_numpy(), util[devices].to_numpy()
        if np.any(np.isfinite(p) & ((p < 0) | (p > 2000))) or np.any(np.isfinite(u) & ((u < 0) | (u > 100))):
            raise ValueError('监测值超出声明的筛选范围')
        pc, uc = np.isfinite(p).sum(axis=1), np.isfinite(u).sum(axis=1)
        device_count += np.isfinite(p).sum(axis=0)
        row_count += len(p)
        ps, us = np.nansum(p, axis=1, dtype=np.float64), np.nansum(u, axis=1, dtype=np.float64)
        if np.any(pc == 0) or np.any(uc == 0):
            raise ValueError('出现全设备缺测的监测时间行')
        alloc = allocation(stamps)
        overlap = (stamps >= job_start) & (stamps < job_stop)
        zero = np.isfinite(p) & (u == 0) & overlap[:, None]
        values = p[zero].astype(float)
        zero_count += len(values)
        zero_sum += values.sum()
        zero_hist += np.bincount(np.floor(values).astype(int), minlength=2001)
        labels = pd.DatetimeIndex(times).floor('h')
        for hour in labels.unique():
            selected = labels == hour
            matched = selected & overlap
            # 缺测设备采用同一采样时刻有效设备均值估算；物理库存来自论文，在线状态未知。
            tables[hour] += [selected.sum(), ps[selected].sum(), pc[selected].sum(), us[selected].sum(), uc[selected].sum(),
                np.sum(ps[selected] / pc[selected] * INVENTORY), alloc[matched].sum(), matched.sum(),
                np.sum(ps[matched] / pc[matched] * INVENTORY),
                np.sum(np.isfinite(p[selected]) & np.isfinite(u[selected]))]
        for hour in np.unique(pd.DatetimeIndex(times).hour):
            selected = pd.DatetimeIndex(times).hour == hour
            panel_sum[hour] += np.nansum(p[selected], axis=0, dtype=np.float64)
            panel_count[hour] += np.isfinite(p[selected]).sum(axis=0)
    rows = []
    for hour, values in sorted(tables.items()):
        n, ps, pc, us, uc, estimated, alloc, matched, matched_power, paired = values
        rows.append({'time_utc': hour, 'hour_utc': hour.hour, 'weekday_utc': hour.dayofweek,
            'samples': int(n), 'sample_time_coverage': n / 240, 'power_valid_device_samples': int(pc),
            'util_valid_device_samples': int(uc), 'paired_device_samples': int(paired),
            'mean_gpu_power_w': ps / pc, 'mean_gpu_util_pct': us / uc,
            'observed_total_gpu_power_w': ps / n, 'mean_reporting_gpu_count': pc / n,
            'device_coverage_physical': pc / (n * INVENTORY), 'device_coverage_released_columns': pc / (n * len(devices)),
            'estimated_total_gpu_power_w': estimated / n, 'overlap_samples': int(matched),
            'allocated_gpu_count_at_samples': alloc / matched if matched else np.nan,
            'overlap_estimated_total_power_w': matched_power / matched if matched else np.nan})
    hourly = pd.DataFrame(rows).merge(jobs, on='time_utc', how='left', validate='one_to_one')
    idle_proxy = zero_sum / zero_count
    for idle in (50., 60., 70., idle_proxy):
        label = f'{idle:.3f}'
        n_alloc = hourly.allocated_gpu_count_at_samples.where(hourly.allocated_gpu_count_at_samples.gt(0))
        hourly['allocated_power_w_idle_' + label] = (hourly.overlap_estimated_total_power_w - (INVENTORY - n_alloc) * idle) / n_alloc
    hourly.to_csv(output / 'kalos-hourly.csv', index=False)
    panel_rows = []
    for cutoff in (.95, .99, .999):
        selected = device_count >= cutoff * row_count
        if not selected.any():
            raise ValueError('固定设备样本为空')
        for hour in range(24):
            # 先求每个设备的同小时均值，再对固定设备等权平均，避免报送数量变化的权重偏差。
            means = panel_sum[hour, selected] / panel_count[hour, selected]
            panel_rows.append({'hour_utc': hour, 'device_coverage_threshold': cutoff, 'devices': int(selected.sum()),
                'mean_power_w': float(means.mean()), 'device_samples': int(panel_count[hour, selected].sum())})
    pd.DataFrame(panel_rows).to_csv(output / 'kalos-fixed-panel-diurnal.csv', index=False)
    pd.DataFrame({'power_w_bin_lower': np.arange(2001), 'samples': zero_hist}).query('samples > 0').to_csv(output / 'kalos-zero-util-power-histogram.csv', index=False)
    audit.update(utc=datetime.now(timezone.utc).isoformat(), physical_gpu_inventory=INVENTORY, released_gpu_columns=len(devices),
        gpu_model='NVIDIA A100 SXM 80GB', tdp_w=TDP, chunk_rows=args.chunk_rows,
        monitoring_start_utc=first.isoformat(), monitoring_end_utc=times.iloc[-1].isoformat(),
        overlapping_hour_bins=int(hourly.overlap_samples.gt(0).sum()), overlap_sample_hours=float(hourly.overlap_samples.sum() / 240),
        complete_overlapping_hours=int((hourly.overlap_samples == 240).sum()), zero_util_power_mean_w=idle_proxy,
        zero_util_samples=zero_count, zero_util_power_median_bin_midpoint_w=float(np.searchsorted(zero_hist.cumsum(), zero_count / 2) + .5),
        fixed_panel_devices_by_threshold={str(cutoff): int((device_count >= cutoff * row_count).sum()) for cutoff in (.95, .99, .999)},
        zero_util_semantics='Zero GPU_UTIL and valid POWER_USAGE at identical device/time within job overlap; allocation status unknown without device mapping.',
        monitoring_sha256={path.name: sha256(path) for path in paths},
        missing_devices_assumption='Each timestamp: observed mean power multiplied by 2416 physical GPUs; missing GPUs assumed exchangeable. Reporting count is not an online inventory.',
        time_basis='All timestamps converted to UTC; allocations evaluated at exact telemetry instants with half-open [start,end) intervals; hourly means do not hold across gaps.')
    dump(output / 'kalos-hourly-audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    print(hourly.loc[hourly.overlap_samples.gt(0), ['time_utc', 'samples', 'allocated_gpu_count_at_samples', 'mean_gpu_power_w', 'mean_reporting_gpu_count', 'allocated_power_w_idle_60.000']].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk-rows', type=int, default=512)
    aggregate(parser.parse_args())
