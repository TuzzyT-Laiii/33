import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from decimal import Decimal
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from mit_gpu_common import METRICS, atomic_json, database, download, identity, load_jobs, utc


def independent(path, obj, jobs):
    _, _, job, _ = identity(obj['key'], jobs)
    hourly = defaultdict(Counter)
    joint, util, previous, cadence = Counter(), Counter(), {}, Counter()
    row_count = 0
    shift = 0
    if job and job['start'] > 0 and job['end'] >= job['start']:
        zone = ZoneInfo('America/New_York')
        start_offset = datetime.fromtimestamp(job['start'], timezone.utc).astimezone(zone).utcoffset()
        end_offset = datetime.fromtimestamp(job['end'], timezone.utc).astimezone(zone).utcoffset()
        if start_offset == end_offset:
            shift = -int(start_offset.total_seconds()) * 1_000_000

    def location(t, gpu):
        hour = t // 3_600_000_000 * 3_600_000_000
        within = bool(job and job['start'] * 1_000_000 <= t < job['end'] * 1_000_000)
        month = datetime.fromtimestamp(hour / 1_000_000, timezone.utc).strftime('%Y-%m')
        return (gpu, hour, within), month, within

    def weights(gpu, t, metrics, duration):
        while duration > 0:
            key, month, within = location(t, gpu)
            amount = min(duration, key[1] + 3_600_000_000 - t)
            hourly[key]['duration_us'] += amount
            for metric, value in metrics.items():
                if value is not None:
                    hourly[key][metric + '_valid_us'] += amount
                    hourly[key][metric + '_time_sum_us'] += value * amount
                    hourly[key][metric + '_time_sumsq_us'] += value * value * amount
            u, power = metrics['utilization_gpu_pct'], metrics['power_draw_W']
            if u is not None:
                util[(month, within, u, 'duration_us')] += amount
                if power is not None:
                    joint[(month, within, 0 if u == 0 else math.ceil(u / 5), math.floor(power / 5), 'duration_us')] += amount
            t += amount
            duration -= amount

    with path.open(newline='', encoding='utf-8') as stream:
        for raw in csv.DictReader(stream):
            row_count += 1
            if not raw['timestamp'] or not raw['gpu_index']:
                continue
            t = int(Decimal(raw['timestamp']) * 1_000_000) + shift
            gpu = int(raw['gpu_index'])
            metrics = {}
            for metric in METRICS:
                try:
                    value = float(raw[metric])
                except (ValueError, TypeError):
                    value = None
                if value is not None and (not math.isfinite(value) or
                        (metric.startswith('utilization') and not 0 <= value <= 100) or
                        (metric == 'power_draw_W' and value < 0) or
                        (metric.startswith('temperature') and value < -273.15)):
                    value = None
                metrics[metric] = value
            key, month, within = location(t, gpu)
            values = hourly[key]
            values['sample_count'] += 1
            for metric, value in metrics.items():
                if value is not None:
                    values[metric + '_count'] += 1
                    values[metric + '_sum'] += value
                    values[metric + '_sumsq'] += value * value
                    values[metric + '_max'] = max(values.get(metric + '_max', -math.inf), value)
            u, power = metrics['utilization_gpu_pct'], metrics['power_draw_W']
            if u is not None:
                util[(month, within, u, 'sample_count')] += 1
                if power is not None:
                    joint[(month, within, 0 if u == 0 else math.ceil(u / 5), math.floor(power / 5), 'sample_count')] += 1
            if gpu in previous:
                prev_t, prev_metrics = previous[gpu]
                gap = t - prev_t
                if gap > 0:
                    cadence[gap // 1000] += 1
                duration = min(1_000_000, max(0, gap))
                prev_within = location(prev_t, gpu)[2]
                if prev_within:
                    duration = min(duration, int(job['end'] * 1_000_000) - prev_t)
                weights(gpu, prev_t, prev_metrics, duration)
            previous[gpu] = (t, metrics)
    return row_count, hourly, joint, util, cadence


def verify(args):
    db = database(args.root)
    done = [dict(r) for r in db.execute("SELECT * FROM objects WHERE status='done' ORDER BY seq")]
    if args.seqs:
        requested = set(args.seqs)
        selected = [r for r in done if r['seq'] in requested]
        if len(selected) != len(requested):
            raise ValueError('指定核验对象包含尚未完成的序号')
    else:
        # 覆盖首尾、较大对象以及多 GPU/跨小时文件；均取自真实试点。
        indices = sorted(set([0, len(done) // 2, len(done) - 1])) if done else []
        selected = [done[i] for i in indices]
        selected += sorted([r for r in done if r not in selected], key=lambda r: r['bytes'])[:max(0, args.files-len(selected))]
    if not selected:
        raise ValueError('没有已完成对象可供核验')
    checked_files = checked_rows = 0
    for batch in sorted({r['batch'] for r in done}):
        folder = args.root / 'batches' / batch
        files = pd.read_parquet(folder / 'files.parquet')
        hourly = pd.read_parquet(folder / 'hourly.parquet')
        util_hist = pd.read_parquet(folder / 'util_hist.parquet')
        joint_hist = pd.read_parquet(folder / 'joint_hist.parquet')
        if int(hourly.sample_count.sum() + files.invalid_identity_rows.sum()) != int(files.rows.sum()):
            raise ValueError('全批次原始行数与小时样本计数不守恒')
        if int(util_hist.sample_count.sum()) != int(hourly.utilization_gpu_pct_count.sum()):
            raise ValueError('全批次利用率直方图样本数不守恒')
        if int(joint_hist.sample_count.sum()) != int(files.joint_valid_samples.sum()):
            raise ValueError('全批次联合直方图样本数不守恒')
        if int(hourly.duration_us.sum()) != int(files.duration_us.sum()):
            raise ValueError('全批次有效微秒不守恒')
        if int(util_hist.duration_us.sum()) != int(hourly.utilization_gpu_pct_valid_us.sum()):
            raise ValueError('全批次利用率直方图时长不守恒')
        np.testing.assert_allclose((util_hist.util_value * util_hist.sample_count).sum(),
            hourly.utilization_gpu_pct_sum.sum(), rtol=2e-10, atol=1e-5)
        np.testing.assert_allclose((util_hist.util_value * util_hist.duration_us).sum(),
            hourly.utilization_gpu_pct_time_sum_us.sum(), rtol=2e-10, atol=1e-5)
        checked_files += len(files)
        checked_rows += int(files.rows.sum())
    jobs = load_jobs(args.root)
    results = []
    for obj in selected:
        path = args.root / 'cache' / f'verify-{obj["seq"]}.part'
        try:
            download(obj, path, args.proxy)
            rows, hourly, joint, util, cadence = independent(path, obj, jobs)
            folder = args.root / 'batches' / obj['batch']
            checks = 0
            table = pd.read_parquet(folder / 'hourly.parquet')
            table = table.loc[table.seq.eq(obj['seq'])]
            if len(hourly) != len(table):
                raise ValueError('独立重算的 GPU×小时分组数不同')
            for row in table.to_dict(orient='records'):
                key = (row['gpu_index'], row['hour_us'], row['within_allocation'])
                direct = hourly[key]
                for column in ['sample_count', 'duration_us', *[m+s for m in METRICS for s in
                        ['_count', '_sum', '_sumsq', '_max', '_valid_us', '_time_sum_us', '_time_sumsq_us']]]:
                    actual = row[column]
                    expected = direct.get(column, math.nan if column.endswith('_max') else 0)
                    if column.endswith('_count') or column in ('sample_count', 'duration_us') or column.endswith('_valid_us'):
                        if actual != expected:
                            raise ValueError(f'{obj["seq"]} {key} {column}: {actual} != {expected}')
                    else:
                        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-5, equal_nan=True)
                    checks += 1
            for name, direct, dimensions in [('joint_hist', joint, ['month', 'within_allocation', 'util_bin', 'power_bin']),
                                               ('util_hist', util, ['month', 'within_allocation', 'util_value'])]:
                table = pd.read_parquet(folder / (name + '.parquet'))
                table = table.loc[table.seq.eq(obj['seq'])]
                actual = Counter()
                for row in table.to_dict(orient='records'):
                    for value in ['sample_count', 'duration_us']:
                        actual[tuple(row[c] for c in dimensions) + (value,)] += row[value]
                if actual != direct:
                    raise ValueError(f'{obj["seq"]}: {name} 直方图不守恒')
                checks += len(actual)
            files = pd.read_parquet(folder / 'files.parquet')
            info = files.loc[files.seq.eq(obj['seq'])].iloc[0]
            if rows != info.rows or dict(cadence) != {int(k): v for k,v in json.loads(info.cadence_ms_counts).items()}:
                raise ValueError('原始行数或采样间隔计数不同')
            result = {'seq': obj['seq'], 'bytes': obj['bytes'], 'rows': rows,
                      'checks': checks, 'hourly_groups': len(hourly), 'passed': True}
            results.append(result)
            print(json.dumps(result), flush=True)
        finally:
            path.unlink(missing_ok=True)
    report = {'verified_at': utc(), 'independent_parser': 'Python csv + Decimal 时间戳；不调用生产汇总函数',
              'pilot_completed_files': len(done), 'verified_files': results,
              'batch_invariants': {'files': checked_files, 'rows': checked_rows, 'all_passed': True},
              'all_passed': True, 'float_rtol': 2e-10, 'float_atol': 1e-5}
    atomic_json(args.root / 'pilot-validation.json', report)
    db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--proxy', default='http://127.0.0.1:7897')
    parser.add_argument('--files', type=int, default=5)
    parser.add_argument('--seqs', type=int, nargs='+')
    args = parser.parse_args()
    verify(args)
