import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from mit_gpu_common import SCHEMA_VERSION, atomic_json, utc


def quantile(values, weights, probability):
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    cumulative = np.cumsum(weights)
    if not len(cumulative) or cumulative[-1] <= 0:
        return None
    return float(values[min(np.searchsorted(cumulative, cumulative[-1] * probability, side='left'), len(values) - 1)])


def atomic_csv(path, frame):
    partial = path.with_suffix('.csv.tmp')
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def aggregate(root, output, max_prefix=None):
    output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect((root / 'state.sqlite').resolve().as_uri() + '?mode=ro', uri=True, timeout=60)
    # 在同一读事务中固定批次与随机前缀，避免异步完成的较小文件造成样本选择偏差。
    db.execute('BEGIN')
    first_pending = db.execute("SELECT MIN(seq) FROM objects WHERE status='pending'").fetchone()[0]
    prefix = first_pending if first_pending is not None else db.execute('SELECT COUNT(*) FROM objects').fetchone()[0]
    if max_prefix is not None:
        if max_prefix <= 0 or max_prefix > prefix:
            raise ValueError('max_prefix 必须为已完成连续随机前缀范围内的正整数')
        prefix = max_prefix
    all_done = db.execute("SELECT COUNT(*) FROM objects WHERE status='done'").fetchone()[0]
    prefix_failed = db.execute("SELECT COUNT(*) FROM objects WHERE status='failed' AND seq<?", (prefix,)).fetchone()[0]
    batches = [r[0] for r in db.execute("SELECT DISTINCT batch FROM objects WHERE status='done' AND seq<? ORDER BY batch", (prefix,))]
    db.close()
    util, joint = Counter(), Counter()
    quality, mapping, cadence, clocks = Counter(), Counter(), Counter(), Counter()
    file_count = rows = source_bytes = anomalies = 0
    file_keys = set()
    covered_us = 0
    scheduled_node_hours = 0
    for name in batches:
        folder = root / 'batches' / name
        files = pd.read_parquet(folder / 'files.parquet')
        files = files.loc[files.seq < prefix]
        if set(files.schema_version) != {SCHEMA_VERSION}:
            raise ValueError('批次 schema 与当前聚合版本不一致')
        if any(seq in file_keys for seq in files.seq):
            raise ValueError('汇总快照出现重复对象')
        file_keys.update(files.seq.tolist())
        file_count += len(files)
        rows += int(files.rows.sum())
        source_bytes += int(files.bytes.sum())
        anomalies += int(files.parse_anomalies.sum())
        quality.update(files.eligibility.value_counts().to_dict())
        mapping.update(files.mapping_status.value_counts().to_dict())
        clocks.update(files.clock_status.value_counts().to_dict())
        eligible_files = files.loc[files.eligibility.eq('eligible')]
        scheduled_node_hours += float(((eligible_files.job_end - eligible_files.job_start) *
            eligible_files.allocated_gpus / eligible_files.job_nodes).sum() / 3600)
        for record in files.cadence_ms_counts:
            cadence.update({int(k): v for k, v in json.loads(record).items()})
        for kind, accumulator, extra in [('util_hist', util, ['util_value']),
                                          ('joint_hist', joint, ['util_bin', 'power_bin'])]:
            table = pd.read_parquet(folder / (kind + '.parquet'))
            table = table.loc[table.seq.lt(prefix) & table.eligibility.eq('eligible') & table.within_allocation]
            columns = ['month', 'allocated_gpus', 'gpu_type', *extra, 'sample_count', 'duration_us']
            for row in table[columns].itertuples(index=False, name=None):
                accumulator[row[:-2] + ('count',)] += int(row[-2])
                accumulator[row[:-2] + ('us',)] += int(row[-1])
        hourly = pd.read_parquet(folder / 'hourly.parquet', columns=[
            'seq', 'eligibility', 'within_allocation', 'duration_us', 'utilization_gpu_pct_time_sum_us'])
        hourly = hourly.loc[hourly.seq.lt(prefix) & hourly.eligibility.eq('eligible') & hourly.within_allocation]
        covered_us += int(hourly.duration_us.sum())

    utilization_rows = []
    if not util:
        raise ValueError('尚无能够确认分配归属的有效利用率观测')
    records = [(*key[:-1], value, util[key[:-1] + ('count',)]) for key, value in util.items() if key[-1] == 'us']
    util_frame = pd.DataFrame(records, columns=['month', 'allocated_gpus', 'gpu_type', 'utilization_pct', 'weight_us', 'sample_count'])
    for grouping in [[], ['month'], ['allocated_gpus'], ['gpu_type'], ['month', 'allocated_gpus', 'gpu_type']]:
        groups = [((), util_frame)] if not grouping else util_frame.groupby(grouping, sort=True)
        for group, frame in groups:
            values = group if isinstance(group, tuple) else (group,)
            record = {'grouping': '+'.join(grouping) or 'global', 'month': 'all',
                      'allocated_gpus': 'all', 'gpu_type': 'all'}
            record.update(dict(zip(grouping, values)))
            total = int(frame.weight_us.sum())
            if not total:
                continue
            record.update({'observed_allocated_gpu_hours': total / 3_600_000_000,
                'mean_utilization_pct': float(np.average(frame.utilization_pct, weights=frame.weight_us)),
                'idle_time_share': float(frame.loc[frame.utilization_pct.eq(0), 'weight_us'].sum() / total),
                **{f'p{int(p * 100):02d}_pct': quantile(frame.utilization_pct, frame.weight_us, p)
                   for p in (.1, .25, .5, .75, .9, .95, .99)}})
            samples = int(frame.sample_count.sum())
            record.update({'sample_count': samples,
                'sample_mean_utilization_pct': float(np.average(frame.utilization_pct, weights=frame.sample_count)) if samples else None,
                **{f'sample_p{int(p*100):02d}_pct': quantile(frame.utilization_pct, frame.sample_count, p)
                   for p in (.1, .25, .5, .75, .9, .95, .99)}})
            utilization_rows.append(record)
    distribution = pd.DataFrame(utilization_rows)
    joint_frame = pd.DataFrame([(*key[:-1], value, joint[key[:-1] + ('count',)])
                                for key, value in joint.items() if key[-1] == 'us'],
        columns=['month', 'allocated_gpus', 'gpu_type', 'util_bin', 'power_bin', 'weight_us', 'sample_count'])
    curves = []
    for grouping in [['util_bin'], ['gpu_type', 'util_bin']]:
        for group, frame in joint_frame.groupby(grouping, sort=True):
            values = group if isinstance(group, tuple) else (group,)
            record = {'gpu_type': 'all'} | dict(zip(grouping, values))
            total = int(frame.weight_us.sum())
            if not total:
                continue
            index = int(record['util_bin'])
            record.update({'utilization_interval_pct': '0' if index == 0 else f'({(index-1)*5},{index*5}]',
                           'observed_gpu_hours': total / 3_600_000_000, 'sample_count': int(frame.sample_count.sum())})
            for probability, label in [(.1, 'p10'), (.5, 'p50'), (.9, 'p90')]:
                bin_id = quantile(frame.power_bin, frame.weight_us, probability)
                record[label + '_W'] = bin_id * 5 + 2.5
                record[label + '_lower_W'] = bin_id * 5
                record[label + '_upper_exclusive_W'] = bin_id * 5 + 5
                sample_bin = quantile(frame.power_bin, frame.sample_count, probability)
                record['sample_' + label + '_W'] = sample_bin * 5 + 2.5 if sample_bin is not None else None
                record['sample_' + label + '_lower_W'] = sample_bin * 5 if sample_bin is not None else None
                record['sample_' + label + '_upper_exclusive_W'] = sample_bin * 5 + 5 if sample_bin is not None else None
            curves.append(record)
    curve = pd.DataFrame(curves)
    atomic_csv(output / 'utilization-distribution.csv', distribution)
    atomic_csv(output / 'utilization-power-curve.csv', curve)
    atomic_csv(output / 'utilization-time-histogram.csv', util_frame)
    atomic_csv(output / 'utilization-power-time-histogram.csv', joint_frame)
    cadence_values, cadence_weights = list(cadence), list(cadence.values())
    result = {'created_at': utc(), 'snapshot_batches': batches, 'completed_files': file_count,
        'random_prefix_objects': prefix, 'prefix_failed_objects': prefix_failed,
        'completed_files_waiting_for_prefix': all_done - file_count,
        'source_bytes': source_bytes, 'rows': rows, 'parse_anomalies': anomalies,
        'mapping_counts': dict(mapping), 'eligibility_counts': dict(quality),
        'clock_counts': dict(clocks),
        'cadence_p10_ms': quantile(cadence_values, cadence_weights, .1),
        'cadence_median_ms': quantile(cadence_values, cadence_weights, .5),
        'cadence_p90_ms': quantile(cadence_values, cadence_weights, .9),
        'observed_allocated_gpu_hours_all_metrics': covered_us / 3_600_000_000,
        'sampled_files_scheduled_node_gpu_hours': scheduled_node_hours,
        'sampled_files_telemetry_time_coverage': covered_us / 3_600_000_000 / scheduled_node_hours if scheduled_node_hours else None,
        'utilization': distribution.loc[distribution.grouping.eq('global')].to_dict(orient='records')[0],
        'power_curve_all_gpu_types': curve.loc[curve.gpu_type.eq('all')].to_dict(orient='records'),
        'scope': '作业文件简单随机顺序前缀；时间加权的已确认分配 GPU 观测，仅代表有监测覆盖的区间',
        'duration_rule': '每张 GPU 向前保持至下一样本，上限 1 s，末尾不外推；分配结束和小时边界截断',
        'power_quantiles': '5 W 固定直方图的分位数所在箱；代表值取箱中心，另附完整箱边界',
        'allocation_limit': '不能用缺失监测的分配小时补零；未证明全体已分配 GPU 监测完整性',
        'timezone': 'GPU 源时钟按 America/New_York 季节偏移推断转换至 Slurm 参考时钟，再按 UTC 坐标分月；跨 DST 作业单列'}
    atomic_json(output / 'summary.json', result)
    print(json.dumps({'completed_files': file_count, 'rows': rows, 'utilization': result['utilization'],
                      'summary_path': str(output / 'summary.json')}, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-prefix', type=int)
    args = parser.parse_args()
    aggregate(args.root, args.output or args.root / 'aggregates/current', args.max_prefix)
