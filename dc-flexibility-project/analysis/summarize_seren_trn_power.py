import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW


def statistics(values, weights):
    selected = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not selected.any():
        return {'hours': 0}
    r, w = np.asarray(values)[selected], np.asarray(weights)[selected]
    q = DescrStatsW(r, weights=w).quantile([.1, .5, .9], return_pandas=False)
    return {'hours': len(r), 'gpu_hour_weighted_mean_ratio': float(np.average(r, weights=w)),
        'p10': float(q[0]), 'median': float(q[1]), 'p90': float(q[2]),
        'minimum': float(r.min()), 'maximum': float(r.max()), 'allocated_gpu_hours': float(w.sum()),
        'negative_hours': int((r < 0).sum()), 'above_tdp_hours': int((r > 1).sum())}


def main(args):
    output = args.output.resolve()
    audit = json.loads((output / 'seren-audit.json').read_text('utf-8'))
    frame = pd.read_csv(output / 'seren-hourly.csv', parse_dates=['time_utc'])
    frame['paired_physical_coverage'] = frame.paired_device_samples / frame.samples / 2288
    quality = frame.physical_device_coverage.ge(.8) & frame.overlap_samples.ge(228) & frame.sample_matched_allocated_gpus.gt(0) & frame.inventory_excess_samples.eq(0)
    weights = frame.sample_matched_allocated_gpus * frame.overlap_samples / 240
    frame['quality_eligible'] = quality
    records = []
    for column in frame.filter(like='inverse_ratio_idle_'):
        idle = column.removeprefix('inverse_ratio_idle_')
        for case, mask in [('coverage_qualified_all_ratios', quality), ('coverage_and_0_to_1', quality & frame[column].between(0, 1))]:
            records.append({'idle_w': float(idle), 'case': case, **statistics(frame.loc[mask, column].to_numpy(), weights[mask].to_numpy())})
    physical = frame.sample_matched_allocated_gpus
    for missing_w in (60., 400.):
        total = frame.observed_total_power_w + (2288 - frame.mean_reporting_gpus) * missing_w
        ratio = (total - (2288 - physical) * 60) / physical / 400
        records.append({'idle_w': 60., 'case': f'missing_gpu_power_{missing_w:.0f}W',
            **statistics(ratio[quality].to_numpy(), weights[quality].to_numpy())})
    pd.DataFrame(records).to_csv(output / 'seren-inverse-sensitivity.csv', index=False)
    ratios = frame['inverse_ratio_idle_60.000']
    type_rows, type_columns = [], [c for c in frame if c.startswith('allocated_') and c != 'allocated_gpu_count']
    for name in type_columns:
        share = frame[name] / frame.allocated_gpu_count
        kind = name.removeprefix('allocated_')
        for threshold in (.8, .9, 1.):
            mask = quality & share.ge(threshold - 1e-9)
            type_rows.append({'dominant_type': kind, 'minimum_gpu_allocation_share': threshold,
                'interpretation': 'mixed allocated power conditional on type share; not per-type power attribution',
                **statistics(ratios[mask].to_numpy(), weights[mask].to_numpy())})
    pd.DataFrame(type_rows).to_csv(output / 'seren-type-dominance.csv', index=False)
    days = []
    for case, mask in [('coverage_qualified_all_ratios', quality), ('coverage_and_0_to_1', quality & ratios.between(0, 1))]:
        for hour in range(24):
            selected = mask & frame.hour_utc.eq(hour)
            row = statistics(ratios[selected].to_numpy(), weights[selected].to_numpy())
            days.append({'case': case, 'hour_utc': hour, 'hour_beijing': (hour + 8) % 24, **row})
    day = pd.DataFrame(days)
    day.to_csv(output / 'seren-diurnal.csv', index=False)
    frame.to_csv(output / 'seren-hourly-screened.csv', index=False)
    primary = day.loc[day.case.eq('coverage_qualified_all_ratios')]
    central = next(row for row in records if row['idle_w'] == 60 and row['case'] == 'coverage_qualified_all_ratios')
    summary = {'calendar_overlap_start_utc': audit['monitoring_start_utc'], 'calendar_overlap_end_utc': audit['monitoring_end_utc'],
        'calendar_overlap_days': float((pd.Timestamp(audit['monitoring_end_utc']) - pd.Timestamp(audit['monitoring_start_utc'])).total_seconds() / 86400),
        'sampled_overlap_hours': float(frame.overlap_samples.sum() / 240), 'hour_bins': len(frame),
        'quality_eligible_hour_bins': int(quality.sum()), 'quality_sampled_hours': float(frame.loc[quality, 'overlap_samples'].sum() / 240),
        'physical_power_coverage_mean': float(frame.power_device_samples.sum() / frame.samples.sum() / 2288),
        'physical_power_coverage_min': float(frame.physical_device_coverage.min()),
        'physical_power_coverage_max': float(frame.physical_device_coverage.max()),
        'paired_coverage_mean': float(frame.paired_device_samples.sum() / frame.samples.sum() / 2288),
        'online_device_inventory': 'unknown: reporting status is not online status',
        'allocated_power_estimate_status': 'conditional cluster inference, not job-device matched measurement',
        'per_type_power_status': 'not_identified except pure-type intervals if any; dominance groups remain mixtures',
        'pretrain_gpu_hour_share_in_qualified_period': float((frame.loc[quality, 'allocated_Pretrain'] * frame.loc[quality, 'overlap_samples']).sum() / (frame.loc[quality, 'allocated_gpu_count'] * frame.loc[quality, 'overlap_samples']).sum()),
        'central_idle60': central,
        'diurnal_peak_valley': float(primary.gpu_hour_weighted_mean_ratio.max() / primary.gpu_hour_weighted_mean_ratio.min()),
        'diurnal_trough_hour_utc': int(primary.loc[primary.gpu_hour_weighted_mean_ratio.idxmin(), 'hour_utc']),
        'diurnal_peak_hour_utc': int(primary.loc[primary.gpu_hour_weighted_mean_ratio.idxmax(), 'hour_utc']),
        'zero_util_proxy_mean_w': audit['zero_util_proxy_mean_w']}
    (output / 'seren-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
