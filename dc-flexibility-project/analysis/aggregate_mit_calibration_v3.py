import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from compute_load.model import sha256


def weighted_quantiles(values, weights):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return {f'p{q}': float(values[order][np.searchsorted(cumulative, cumulative[-1] * q / 100)])
            for q in (10, 25, 50, 75, 90, 95, 99)}


def aggregate(source, output, allow_partial=False):
    summary_path = source / 'summary.json'
    summary_hash = sha256(summary_path)
    summary = json.loads(summary_path.read_text('utf-8'))
    verification = json.loads((source / 'verification.json').read_text('utf-8'))
    if verification['completed_objects'] != summary['completed_objects'] or verification['input_analysis_created_at'] != summary['created_at']:
        raise ValueError('MIT 月度分析与独立核验的快照不一致')
    if (pd.Timestamp(summary['window_start_utc']) != pd.Timestamp('2021-06-01', tz='UTC')
            or pd.Timestamp(summary['window_end_utc']) != pd.Timestamp('2021-07-01', tz='UTC')
            or summary['tdp_W'] != 250 or summary['failed_objects'] != 0):
        raise ValueError('MIT 月度窗口、归一化参考或失败数量与本次约定不符')
    if summary['partial_data'] and not allow_partial:
        raise ValueError('六月正式窗口尚未完成')
    if not summary['partial_data'] and summary['completed_objects'] != summary['total_objects']:
        raise ValueError('MIT 完成标识与对象数量不一致')
    device_path = source / 'device-hours.parquet'
    device_hash = sha256(device_path)
    data = pd.read_parquet(device_path)
    good = data.loc[~data.ambiguous & data.power_valid_us.gt(0)].copy()
    end = pd.Timestamp(summary['calibration_end_utc']).value // 1000
    calibration = good.loc[good.hour_us.lt(end)]
    weights = calibration.power_valid_us.to_numpy(float)
    values = calibration.mean_power_W.to_numpy(float)
    mean = float(np.average(values, weights=weights))
    if not np.isclose(mean, summary['allocated_power_W_per_gpu'], rtol=1e-9):
        raise ValueError('独立设备小时积分与月度标定均值不一致')
    distribution = weighted_quantiles(values, weights)
    full_weights = good.power_valid_us.to_numpy(float)
    full_values = good.mean_power_W.to_numpy(float)
    output.mkdir(parents=True, exist_ok=True)
    result = {**summary, 'source_summary_sha256': summary_hash, 'source_device_hours_sha256': device_hash,
              'aggregation_utc': datetime.now(timezone.utc).isoformat(),
              'calibration_device_hour_power_W': distribution,
              'calibration_device_hour_power_to_tdp': {key: value / summary['tdp_W'] for key, value in distribution.items()},
              'whole_month_device_hour_mean_W': float(np.average(full_values, weights=full_weights)),
              'whole_month_weighted_device_hour_mean_below_50W_share': float(np.sum(full_weights[full_values < 50]) / full_weights.sum()),
              'distribution_unit': '合并且排除歧义的设备小时平均功率，以有效监测时长计权；分位数为描述性敏感性端点',
              'license': 'CC BY-NC-ND 4.0',
              'citation': 'Samsi et al., The MIT Supercloud Dataset, DOI:10.1109/HPEC49654.2021.9622850',
              'release_status': '阶段版' if summary['partial_data'] else '六月窗口完成版',
              'target_idle_assumption': '已分配且零利用率的V100空转代理/TDP按比例迁移至H100；未分配H100无匹配实测'}
    for name in ('hourly-cluster-power.csv', 'diurnal-shape.csv', 'weekly-shape.csv',
                 'diurnal-observed-support-shape.csv', 'weekly-observed-support-shape.csv', 'verification.json'):
        shutil.copy2(source / name, output / ('mit-' + name))
    bins = np.r_[np.arange(0, 301, 10), np.inf]
    histogram, _ = np.histogram(full_values, bins=bins, weights=full_weights / 3.6e9)
    pd.DataFrame({'lower_W': bins[:-1], 'upper_W': [str(x) for x in bins[1:]],
                  'valid_gpu_hours': histogram}).to_csv(output / 'mit-device-hour-power-distribution.csv', index=False)
    if sha256(summary_path) != summary_hash or sha256(device_path) != device_hash:
        raise RuntimeError('读取期间 MIT 源文件发生变化')
    (output / 'mit-calibration.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    result = aggregate(args.source, args.output, args.allow_partial)
    print(json.dumps({key: result[key] for key in ('release_status', 'completed_objects', 'allocated_power_W_per_gpu')}, ensure_ascii=False))
