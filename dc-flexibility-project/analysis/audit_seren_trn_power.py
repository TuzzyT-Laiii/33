import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compute_load.model import sha256
from task_trace_inputs import interval_resource_hours


INVENTORY, TDP = 2288, 400.


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_jobs(raw, output):
    path = raw / 'github/data/job_trace/trace_seren.csv'
    fields = pd.read_csv(path, nrows=0).columns.tolist()
    frame = pd.read_csv(path, usecols=['job_id', 'gpu_num', 'type', 'start_time', 'end_time', 'duration', 'state'])
    if frame.job_id.isna().any() or frame.job_id.duplicated().any():
        raise ValueError('Seren作业标识缺失或重复')
    start, end = [pd.to_datetime(frame[name], utc=True, format='mixed') for name in ('start_time', 'end_time')]
    valid = frame.gpu_num.gt(0) & start.notna() & end.notna() & end.ge(start)
    if (frame.gpu_num.gt(0) & ~valid).any():
        raise ValueError('Seren GPU作业存在无有效执行区间的记录，须先审阅')
    begin, stop = start.min().floor('h'), end.max().ceil('h')
    edges = pd.date_range(begin, stop, freq='h')
    table = pd.DataFrame({'time_utc': edges[:-1]})
    selected = frame.loc[valid]
    starts, ends = start[valid].astype('int64').to_numpy(), end[valid].astype('int64').to_numpy()
    units = selected.gpu_num.to_numpy(float)
    table['allocated_gpu_count'] = interval_resource_hours(starts, ends, units, edges.asi8)
    for kind in sorted(selected.type.unique()):
        mask = selected.type.eq(kind).to_numpy()
        table['allocated_' + kind] = interval_resource_hours(starts[mask], ends[mask], units[mask], edges.asi8)
    np.testing.assert_allclose(table.filter(like='allocated_').iloc[:, 1:].sum(axis=1), table.allocated_gpu_count, atol=1e-7)
    table.to_csv(output / 'seren-job-hourly.csv', index=False)
    events, order = np.r_[starts, ends], np.argsort(np.r_[starts, ends], kind='stable')
    events, cumulative = events[order], np.r_[0., np.cumsum(np.r_[units, -units][order])]
    def allocated(stamps):
        return cumulative[np.searchsorted(events, stamps, side='right')]
    audit = {'utc': datetime.now(timezone.utc).isoformat(), 'rows': len(frame), 'gpu_jobs': int(valid.sum()),
        'fields': fields, 'type_counts': frame.type.value_counts().astype(int).to_dict(),
        'start_utc': start.min().isoformat(), 'end_utc': end.max().isoformat(),
        'job_sha256': sha256(path), 'node_device_mapping': False,
        'mapping_note': 'node_num is a count; no node list or GPU/device identifier in the released job schema',
        'missing_ends': int(end.isna().sum()), 'duration_mismatch_rows': int((end - start).dt.total_seconds().ne(frame.duration).sum()),
        'maximum_hourly_allocated_gpus': float(table.allocated_gpu_count.max()),
        'hourly_inventory_excess': int(table.allocated_gpu_count.gt(INVENTORY).sum())}
    write_json(output / 'seren-job-audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    return table, allocated, audit


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs, allocation, audit = read_jobs(args.raw, output)
    if args.jobs_only:
        return
    paths = [args.raw / 'hf/data/utilization/seren' / (metric + '.csv') for metric in ('POWER_USAGE', 'GPU_UTIL')]
    devices = pd.read_csv(paths[0], nrows=0).columns.tolist()[1:]
    if set(devices) != set(pd.read_csv(paths[1], nrows=0).columns[1:]):
        raise ValueError('Seren监测设备集合不同')
    readers = [pd.read_csv(path, chunksize=args.chunk_rows, dtype={name: np.float32 for name in devices}) for path in paths]
    accum = defaultdict(lambda: np.zeros(13, dtype=float))
    first = previous = None
    zeros = np.zeros(2001, dtype=np.int64)
    zero_sum, rows = 0., 0
    for power, util in zip(*readers, strict=True):
        time = pd.to_datetime(power.pop('Time'), utc=True)
        if not time.equals(pd.to_datetime(util.pop('Time'), utc=True)):
            raise ValueError('Seren功率与利用率时间行不同')
        stamps = time.astype('int64').to_numpy()
        if first is None:
            first = time.iloc[0]
        if np.any(np.diff(np.r_[previous, stamps] if previous is not None else stamps) <= 0):
            raise ValueError('Seren时间重复或倒序')
        previous = stamps[-1]
        p, u = power[devices].to_numpy(), util[devices].to_numpy()
        if np.any(np.isfinite(p) & ((p < 0) | (p > 2000))) or np.any(np.isfinite(u) & ((u < 0) | (u > 100))):
            raise ValueError('Seren监测超出声明的数值筛选范围')
        pc, uc = np.isfinite(p).sum(axis=1), np.isfinite(u).sum(axis=1)
        ps, us = np.nansum(p, axis=1, dtype=np.float64), np.nansum(u, axis=1, dtype=np.float64)
        if np.any(pc == 0):
            raise ValueError('存在全部设备缺少功率的时间行')
        alloc = allocation(stamps)
        overlap = (stamps >= pd.Timestamp(audit['start_utc']).value) & (stamps < pd.Timestamp(audit['end_utc']).value)
        selected_zero = np.isfinite(p) & (u == 0) & overlap[:, None]
        values = p[selected_zero].astype(float)
        zeros += np.bincount(np.floor(values).astype(int), minlength=2001)
        zero_sum += values.sum()
        labels = pd.DatetimeIndex(time).floor('h')
        for hour in labels.unique():
            mask = labels == hour
            matched = mask & overlap
            accum[hour] += [mask.sum(), ps[mask].sum(), pc[mask].sum(), us[mask].sum(), uc[mask].sum(),
                np.sum(ps[mask] / pc[mask] * INVENTORY), matched.sum(), alloc[matched].sum(),
                np.sum(ps[matched] / pc[matched] * INVENTORY), np.sum(pc[matched]),
                np.sum(np.isfinite(p[mask]) & np.isfinite(u[mask])), np.sum(alloc[matched] > INVENTORY),
                np.sum((pc[matched] / INVENTORY >= .8) & (alloc[matched] > 0) & (alloc[matched] <= INVENTORY))]
        rows += len(time)
        if rows % (args.chunk_rows * 64) == 0:
            print('Seren rows', rows, flush=True)
    result = []
    for hour, a in sorted(accum.items()):
        n, ps, pc, us, uc, estimated, matched, alloc, overlap_power, overlap_pc, paired, excess, qualified = a
        result.append({'time_utc': hour, 'hour_utc': hour.hour, 'weekday_utc': hour.dayofweek,
            'samples': int(n), 'sample_time_coverage': n / 240, 'mean_gpu_power_w': ps / pc,
            'mean_gpu_util_pct': us / uc if uc else np.nan, 'power_device_samples': int(pc), 'util_device_samples': int(uc),
            'paired_device_samples': int(paired), 'observed_total_power_w': ps / n, 'estimated_total_power_w': estimated / n,
            'mean_reporting_gpus': pc / n, 'physical_device_coverage': pc / n / INVENTORY,
            'released_device_coverage': pc / n / len(devices), 'overlap_samples': int(matched),
            'sample_matched_allocated_gpus': alloc / matched if matched else np.nan,
            'overlap_estimated_power_w': overlap_power / matched if matched else np.nan,
            'inventory_excess_samples': int(excess), 'coverage_eligible_samples': int(qualified)})
    hourly = pd.DataFrame(result).merge(jobs, on='time_utc', how='left', validate='one_to_one')
    idle_proxy = zero_sum / zeros.sum()
    for idle in (50., 60., 70., idle_proxy):
        denominator = hourly.sample_matched_allocated_gpus.where(hourly.sample_matched_allocated_gpus.gt(0))
        ratio = (hourly.overlap_estimated_power_w - (INVENTORY - denominator) * idle) / denominator / TDP
        hourly[f'inverse_ratio_idle_{idle:.3f}'] = ratio
        hourly[f'eligible_idle_{idle:.3f}'] = ratio.between(0, 1) & hourly.physical_device_coverage.ge(.8) & hourly.overlap_samples.ge(228) & hourly.inventory_excess_samples.eq(0)
    hourly.to_csv(output / 'seren-hourly.csv', index=False)
    pd.DataFrame({'power_w_bin_lower': np.arange(2001), 'samples': zeros}).query('samples > 0').to_csv(output / 'seren-zero-util-power.csv', index=False)
    audit.update(monitoring_start_utc=first.isoformat(), monitoring_end_utc=time.iloc[-1].isoformat(),
        monitoring_rows=rows, released_gpu_columns=len(devices), physical_gpu_inventory=INVENTORY,
        gpu_model='NVIDIA A100 SXM 80GB', tdp_w=TDP, zero_util_proxy_mean_w=float(idle_proxy),
        zero_util_proxy_samples=int(zeros.sum()), zero_util_proxy_status='allocation status unknown without device mapping',
        monitoring_sha256={p.name: sha256(p) for p in paths}, chunk_rows=args.chunk_rows,
        extrapolation_assumption='At each timestamp multiply reporting GPU mean by 2288; missing GPUs assumed representative. Reporting status does not establish online status.',
        type_attribution='Only type-level allocated GPU counts are observable; no per-type power attribution is identified.')
    write_json(output / 'seren-audit.json', audit)
    print('Seren aggregation complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk-rows', type=int, default=1024)
    parser.add_argument('--jobs-only', action='store_true')
    main(parser.parse_args())
