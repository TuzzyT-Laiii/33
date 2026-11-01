import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate_trn_acme_hourly import INVENTORY, TDP
from compute_load.model import sha256


def main(args):
    output = args.output.resolve()
    hourly = pd.read_csv(output / 'kalos-hourly.csv', parse_dates=['time_utc'])
    audit = json.loads((output / 'kalos-hourly-audit.json').read_text('utf-8'))
    native_end = pd.Timestamp('2023-08-16', tz='UTC')
    hourly['in_native_window'] = hourly.time_utc.lt(native_end)
    hourly['util_device_coverage_physical'] = hourly.util_valid_device_samples / (hourly.samples * INVENTORY)
    hourly['paired_device_coverage_physical'] = hourly.paired_device_samples / (hourly.samples * INVENTORY)
    hourly['reporting_count_below_global_allocation'] = hourly.mean_reporting_gpu_count.lt(hourly.allocated_gpu_count_at_samples)
    hourly['coverage_eligible'] = hourly.device_coverage_physical.ge(.8) & hourly.overlap_samples.ge(228)
    records = []
    for idle in (50., 60., 70., audit['zero_util_power_mean_w']):
        column = f'allocated_power_w_idle_{idle:.3f}'
        ratio = hourly[column] / TDP
        eligible = hourly.overlap_samples.gt(0) & hourly.allocated_gpu_count_at_samples.gt(0)
        physical = ratio.between(0, 1) & eligible
        hourly[f'inverse_ratio_idle_{idle:.3f}'] = ratio
        hourly[f'inverse_physical_idle_{idle:.3f}'] = physical
        hourly[f'inverse_eligible_idle_{idle:.3f}'] = physical & hourly.coverage_eligible
        for scope, mask in [('job_overlap', eligible), ('native_overlap', eligible & hourly.in_native_window)]:
            part = ratio[mask]
            record = {'idle_w': idle, 'scope': scope, 'positive_allocation_hours': int(mask.sum()),
                'physical_range_hours': int((physical & mask).sum()),
                'coverage_and_range_eligible_hours': int((physical & mask & hourly.coverage_eligible).sum())}
            record.update({name: float(value) for name, value in part.quantile([0, .1, .5, .9, 1]).rename(index={0: 'minimum', .1: 'p10', .5: 'median', .9: 'p90', 1: 'maximum'}).items()})
            records.append(record)
    hourly.to_csv(output / 'kalos-hourly-screened.csv', index=False)
    pd.DataFrame(records).to_csv(output / 'inverse-sensitivity.csv', index=False)
    source = pd.read_parquet(args.native_source)
    clock = pd.DatetimeIndex(pd.to_datetime(source.source_clock, utc=True))
    weights = source.groupby(clock.hour).allocated_gpu_fraction.sum().reindex(range(24)).to_numpy()
    fixed = pd.read_csv(output / 'kalos-fixed-panel-diurnal.csv')
    profile = pd.DataFrame({'hour_utc': np.arange(24)})
    for cutoff, label in [(.99, 'central'), (.95, 'panel095'), (.999, 'panel0999')]:
        part = fixed.loc[fixed.device_coverage_threshold.eq(cutoff)].sort_values('hour_utc')
        if len(part) != 24 or part.mean_power_w.isna().any():
            raise ValueError('固定设备样本的日形状缺少小时')
        relative = part.mean_power_w.to_numpy() / np.average(part.mean_power_w, weights=weights)
        profile['shape_' + label] = .8 * relative
        profile['source_mean_power_w_' + label] = part.mean_power_w.to_numpy()
        if np.any(profile['shape_' + label] > 1):
            raise ValueError('形状迁移超过TDP，需要重新确定迁移幅度')
    # 中央日形状的幅度敏感性保留原生GPU小时加权均值0.80。
    for amplitude, label in [(0., 'zero_amplitude'), (.5, 'half_amplitude'), (1.5, 'onehalf_amplitude')]:
        profile['shape_' + label] = .8 + amplitude * (profile.shape_central - .8)
        if not profile['shape_' + label].between(0, 1).all():
            raise ValueError('幅度敏感性超出0至1')
    profile['source_kind'] = 'fixed-panel whole-Kalos GPU power shape; allocated-only shape unidentifiable'
    profile['transfer_status'] = 'engineering scenario; repeated hour-of-day pattern over native window, no weekly modifier'
    profile.to_csv(output / 'transfer-profile.csv', index=False)
    overlap = hourly.loc[hourly.overlap_samples.gt(0)]
    native = overlap.loc[overlap.in_native_window]
    weekly = hourly.loc[hourly.samples.ge(228)].groupby(['weekday_utc', 'hour_utc']).agg(
        mean_gpu_power_w=('mean_gpu_power_w', 'mean'), source_hours=('mean_gpu_power_w', 'size'),
        mean_device_coverage=('device_coverage_physical', 'mean')).reset_index()
    weekly.to_csv(output / 'kalos-observed-weekly.csv', index=False)
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'native_source_sha256': sha256(args.native_source),
        'job_overlap_hours': len(overlap), 'native_overlap_hour_bins': len(native),
        'native_overlap_sample_hours': float(native.overlap_samples.sum() / 240),
        'overlap_physical_device_coverage_min': float(overlap.device_coverage_physical.min()),
        'overlap_physical_device_coverage_max': float(overlap.device_coverage_physical.max()),
        'native_overlap_physical_device_coverage_min': float(native.device_coverage_physical.min()),
        'native_overlap_physical_device_coverage_max': float(native.device_coverage_physical.max()),
        'native_overlap_paired_device_coverage_min': float(native.paired_device_coverage_physical.min()),
        'native_overlap_paired_device_coverage_max': float(native.paired_device_coverage_physical.max()),
        'quality_coverage_threshold': .8, 'quality_sample_time_threshold': .95,
        'inverse_eligible_hours_at_idle60': int(hourly['inverse_eligible_idle_60.000'].sum()),
        'scheme_a': '0.80 times fixed-device-panel diurnal power shape; target allocation-weighted mean exactly 0.80; no weekly factor',
        'scheme_a_source': 'All released Kalos monitoring; devices valid in at least 99% of released time rows; equal device weight, sample mean within each UTC hour; gaps not filled',
        'scheme_b': 'not_identifiable: insufficient overlap coverage and non-exchangeable device reporting; no direct H100 allocation-ratio simulation',
        'direct_scheme_missing_values_filled': False, 'source_values_clipped_to_tdp': False,
        'source_shape_peak_valley': float(profile.shape_central.max() / profile.shape_central.min()),
        'target_ratio_min': float(profile.shape_central.min()), 'target_ratio_max': float(profile.shape_central.max()),
        'target_ratio_gpu_hour_mean': float(np.average(profile.shape_central, weights=weights)),
        'trough_hour_utc': int(profile.loc[profile.shape_central.idxmin(), 'hour_utc']),
        'peak_hour_utc': int(profile.loc[profile.shape_central.idxmax(), 'hour_utc']),
        'checkpoint_attribution': 'unknown: no checkpoint or device allocation event log; job starts/ends reported descriptively only',
        'weekly_attribution': 'unknown: short source window and changing reporting population; no weekday modulation applied',
        'input_sha256': {name: sha256(output / name) for name in ('kalos-hourly.csv', 'kalos-hourly-audit.json', 'kalos-fixed-panel-diurnal.csv')},
        'inverse_statistics': records}
    (output / 'transfer-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-source', type=Path, required=True)
    main(parser.parse_args())
