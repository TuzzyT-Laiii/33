import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate_mit_gpu_telemetry import quantile
from compute_load.model import sha256


PROBABILITIES = (.1, .25, .5, .75, .9, .95, .99)


def describe(values, weights):
    values, weights = np.asarray(values), np.asarray(weights)
    if not len(values) or not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('统计输入必须为有限观测及正时长权重')
    return {'mean': float(np.average(values, weights=weights)),
            **{f'p{int(p * 100):02d}': quantile(values, weights, p) for p in PROBABILITIES}}


def summarize(root, snapshot, output, tdp_w, model):
    if not np.isfinite(tdp_w) or tdp_w <= 0:
        raise ValueError('源GPU的官方TDP必须为正有限数')
    summary = json.loads((snapshot / 'summary.json').read_text(encoding='utf-8'))
    prefix = summary['random_prefix_objects']
    hours, util, joint, files = [], [], [], []
    for batch in summary['snapshot_batches']:
        folder = root / 'batches' / batch
        file_table = pd.read_parquet(folder / 'files.parquet')
        files.append(file_table.loc[file_table.seq.lt(prefix), ['seq', 'eligibility', 'gpu_type']])
        for name, collection in [('hourly', hours), ('util_hist', util), ('joint_hist', joint)]:
            table = pd.read_parquet(folder / (name + '.parquet'))
            mask = table.seq.lt(prefix) & table.eligibility.eq('eligible') & table.within_allocation & table.duration_us.gt(0)
            collection.append(table.loc[mask])
    hour = pd.concat(hours, ignore_index=True)
    util = pd.concat(util, ignore_index=True)
    joint = pd.concat(joint, ignore_index=True)
    files = pd.concat(files, ignore_index=True)
    if len(files) != summary['completed_files'] or files.seq.duplicated().any() or summary['prefix_failed_objects']:
        raise ValueError('固定前缀的文件数或失败状态不满足完整快照要求')
    if hour.duplicated(['seq', 'gpu_index', 'hour_us']).any():
        raise ValueError('GPU 小时重复，不能重复计权')
    hour = hour.loc[hour.power_draw_W_valid_us.gt(0)].copy()
    hour['mean_w'] = hour.power_draw_W_time_sum_us / hour.power_draw_W_valid_us
    hour['sd_w'] = np.sqrt(np.maximum(0, hour.power_draw_W_time_sumsq_us / hour.power_draw_W_valid_us - hour.mean_w ** 2))
    rows, detail = [], {}
    for limit in sorted({500, 1000, prefix}):
        if limit > prefix:
            continue
        for gpu_type in sorted(hour.gpu_type.unique()):
            h = hour.loc[hour.seq.lt(limit) & hour.gpu_type.eq(gpu_type)]
            u = util.loc[util.seq.lt(limit) & util.gpu_type.eq(gpu_type)]
            j = joint.loc[joint.seq.lt(limit) & joint.gpu_type.eq(gpu_type)]
            if h.empty or u.empty or j.empty:
                continue
            weights = h.power_draw_W_valid_us.to_numpy(dtype=float)
            hourly = describe(h.mean_w, weights)
            fluctuation = describe(h.sd_w, weights)
            instantaneous = describe(j.power_bin * 5 + 2.5, j.duration_us)
            idle = j.loc[j.util_bin.eq(0) & j.duration_us.gt(0)]
            idle_power = describe(idle.power_bin * 5 + 2.5, idle.duration_us)
            utilization = describe(u.util_value, u.duration_us)
            ratio = {key: value / tdp_w for key, value in hourly.items()} if gpu_type == 'gpu:volta' else {}
            record = {'prefix_files': limit, 'gpu_type': gpu_type,
                'eligible_files': int(files.loc[files.seq.lt(limit) & files.eligibility.eq('eligible') & files.gpu_type.eq(gpu_type)].shape[0]),
                'observed_power_gpu_hours': float(weights.sum() / 3_600_000_000),
                'gpu_hour_segments': len(h), 'mean_utilization_pct': utilization['mean'],
                'idle_time_share': float(u.loc[u.util_value.eq(0), 'duration_us'].sum() / u.duration_us.sum()),
                **{key + '_w': value for key, value in hourly.items()},
                **{key + '_tdp_ratio': value for key, value in ratio.items()},
                'within_hour_sd_mean_w': fluctuation['mean'], 'within_hour_sd_p50_w': fluctuation['p50'],
                'within_hour_sd_p90_w': fluctuation['p90']}
            rows.append(record)
            detail[f'{limit}:{gpu_type}'] = {'hour_mean_power_w': hourly, 'hour_mean_power_over_tdp': ratio,
                'within_hour_time_weighted_sd_w': fluctuation, 'instantaneous_power_w_5w_bin_midpoint': instantaneous,
                'zero_utilization_power_w_5w_bin_midpoint': idle_power,
                'utilization_pct': utilization, 'statistics': record}
    table = pd.DataFrame(rows)
    v100 = table.loc[table.gpu_type.eq('gpu:volta')].sort_values('prefix_files').copy()
    columns = ['mean_tdp_ratio', 'p10_tdp_ratio', 'p50_tdp_ratio', 'p90_tdp_ratio', 'idle_time_share']
    changes = []
    for index in range(1, len(v100)):
        before, after = v100.iloc[index - 1], v100.iloc[index]
        changes.append({'from_prefix': int(before.prefix_files), 'to_prefix': int(after.prefix_files),
            'absolute_changes': {c: float(after[c] - before[c]) for c in columns},
            'relative_changes': {c: float(after[c] / before[c] - 1) for c in columns}})
    stable = all(abs(value) < .05 for change in changes for value in change['relative_changes'].values())
    v = detail[f'{prefix}:gpu:volta']
    result = {'created_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'snapshot_directory': str(snapshot), 'snapshot_summary_sha256': sha256(snapshot / 'summary.json'),
        'completed_files': summary['completed_files'], 'random_prefix_objects': prefix,
        'all_types_telemetry_gpu_hours': summary['observed_allocated_gpu_hours_all_metrics'],
        'all_types_telemetry_time_coverage': summary['sampled_files_telemetry_time_coverage'],
        'eligibility_counts': summary['eligibility_counts'], 'mapping_counts': summary['mapping_counts'],
        'clock_counts': summary['clock_counts'],
        'rows': summary['rows'], 'source_bytes': summary['source_bytes'],
        'source_gpu_model': model, 'source_gpu_tdp_w': tdp_w,
        'transfer_gpu_type': 'gpu:volta', 'transfer_distribution': v['hour_mean_power_over_tdp'],
        'transfer_power_w': v['hour_mean_power_w'], 'within_hour_sd_w': v['within_hour_time_weighted_sd_w'],
        'idle_time_share': v['statistics']['idle_time_share'],
        'zero_utilization_power_w': v['zero_utilization_power_w_5w_bin_midpoint'],
        'stability_changes': changes, 'within_5pct_for_all_nested_comparisons': stable,
        'stability_interpretation': '5%为诊断阈值，不构成总体收敛或代表性证明；文件简单随机采样，GPU时间权重可能受长作业影响',
        'definition': '每个作业文件、GPU、UTC小时内，有效功率的时间积分/有效覆盖时间；分位数按有效GPU小时计权，边界不完整小时按覆盖时长计权',
        'instantaneous_definition': '原秒级/100ms观测的5W箱中心，仅用于辅助分布及零利用率条件功率，误差至多2.5W',
        'within_hour_definition': 'sqrt(时间加权二阶矩减均值平方)；部分小时也保留，描述监测覆盖区间波动',
        'missingness': '间隙上限1s，末端不外推；缺失功率与未覆盖分配时间均不补零',
        'license': 'CC-BY-NC-ND-4.0；公开结果仅聚合分位数、均值、计数及图件',
        'details': detail}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'mit-power-ratio-summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    table.to_csv(output / 'mit-power-ratio-stability.csv', index=False, encoding='utf-8-sig')
    print(json.dumps({key: result[key] for key in ['completed_files', 'source_gpu_model', 'source_gpu_tdp_w',
        'transfer_distribution', 'idle_time_share', 'within_hour_sd_w', 'stability_changes']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tdp-w', type=float, required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    summarize(args.root, args.snapshot, args.output, args.tdp_w, args.model)
