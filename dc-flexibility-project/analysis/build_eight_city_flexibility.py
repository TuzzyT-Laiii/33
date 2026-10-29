import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eight_city_physics import (DEMAND_DAYS, DELAYS, MULTIPLES, SHARES, flatten_periods,
    independent_hour_envelope, mean_range, read_profiles, replay)
from typical_power_models_v2 import power_v2
from eight_city_scenarios import PURPOSES, scenario_fields


RUN = 'eight-city-phase-a-20261029'
MODEL_VERSION = 'complete-period-objective-v2'


def build(args):
    project, baseline = args.project.resolve(), args.baseline.resolve()
    output, processed = project / 'results' / RUN, project / 'data/processed' / RUN / 'flexibility'
    output.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    profiles, sources = read_profiles(project, baseline)
    hours = 8760
    times = pd.date_range('2025-01-01', periods=hours, freq='h', tz='UTC')
    representative = {'Lisa': ('C20', 'Lisa'), 'MIT': ('C21', 'MIT'), 'Acme': ('C21', 'Acme'), 'ASI': ('C21', 'ASI')}
    rows, envelopes, checks = [], {}, {}
    for demand, key in representative.items():
        profile = profiles[key]
        cpu, gpu, it = replay(profile, hours)
        # 资源小时为服务代理；GPU需求采用分配GPU小时，CPU需求采用CPU分配小时。
        work = cpu if demand == 'Lisa' else gpu
        for share, delay, multiple in itertools.product(SHARES, DELAYS, MULTIPLES):
            label = f'{demand}-f{share:g}-D{delay}-r{multiple:g}'
            target = processed / (label + '.parquet')
            receipt_path = processed / (label + '.json')
            if target.exists() and receipt_path.exists():
                saved = json.loads(receipt_path.read_text(encoding='utf-8'))
                if saved.get('model_version') == MODEL_VERSION:
                    rows.append(saved)
                    continue
            removable = independent_hour_envelope(work, share, delay, multiple)
            day, day_check = flatten_periods(work, share, delay, multiple, 24)
            week, week_check = flatten_periods(work, share, delay, multiple, 168)
            row = {'model_version': MODEL_VERSION, 'demand': demand, 'year': 2025, 'share': share, 'max_extra_delay_h': delay,
                'capacity_multiple': multiple, 'hours': hours,
                'mean_work_fraction': float(work.mean()), 'max_independent_removable_resource_h': float(removable.max()),
                'mean_independent_removable_resource_h': float(removable.mean()),
                'baseline_day_range': mean_range(work, 24), 'flattened_day_range': mean_range(day, 24),
                'baseline_week_range': mean_range(work, 168), 'flattened_week_range': mean_range(week, 168),
                'day_range_reduction': mean_range(work, 24) - mean_range(day, 24),
                'week_range_reduction': mean_range(work, 168) - mean_range(week, 168),
                'day_max_cumulative_residual': day_check['max_cumulative_residual'],
                'week_max_cumulative_residual': week_check['max_cumulative_residual'],
                'max_capacity_residual': max(day_check['max_capacity_residual'], week_check['max_capacity_residual']),
                'service_model': 'divisible_preemptible_resource_hour_upper_bound',
                'original_deadlines_identified': False, 'phase': 'source_midnight_to_UTC_midnight'}
            frame = pd.DataFrame({'time_utc': times, 'baseline_resource_fraction': work,
                'independent_max_removable_fraction': removable,
                'day_flattened_fraction': day, 'week_flattened_fraction': week})
            frame.to_parquet(target, index=False, compression='zstd')
            receipt_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')
            rows.append(row)
            print(label, flush=True)
        print(json.dumps({'demand_complete': demand, 'utc': datetime.now(timezone.utc).isoformat()}), flush=True)
    summary = pd.DataFrame(rows)
    summary.insert(0, 'purpose', summary.demand.map(lambda value: PURPOSES[value][0]))
    summary.rename(columns={'demand': 'demand_source'}, inplace=True)
    summary.to_csv(output / 'flexibility-resource-summary.csv', index=False, encoding='utf-8-sig')
    electrical = []
    for (config, demand), profile in profiles.items():
        cpu, gpu, it = replay(profile, hours)
        work = cpu if demand == 'Lisa' else gpu
        for row in summary.loc[summary.demand_source.eq(demand)].itertuples():
            label = f'{demand}-f{row.share:g}-D{row.max_extra_delay_h}-r{row.capacity_multiple:g}'
            frame = pd.read_parquet(processed / (label + '.parquet'))
            removed = frame.independent_max_removable_fraction.to_numpy()
            remaining = np.divide(work-removed, work, out=np.ones(hours), where=work > 0)
            less = power_v2(profile['config'], cpu * remaining, gpu * remaining, profile['method'], profile['ratios'])
            difference = it - less['unit_w'] * profile['units'] / 1e6
            if difference.min() < -1e-8:
                raise ValueError('中央功率方法移除任务后IT功率增加')
            path = processed / f'{config}-{label}-electric.parquet'
            pd.DataFrame({'time_utc': times, 'independent_max_removable_it_mwh': difference}).to_parquet(path, index=False)
            electrical.append({**scenario_fields(config, demand), 'share': row.share,
                'max_extra_delay_h': row.max_extra_delay_h, 'capacity_multiple': row.capacity_multiple,
                'maximum_single_hour_removed_it_mwh': float(difference.max()),
                'mean_independent_single_hour_removed_it_mwh': float(difference.mean()),
                'interpretation': 'independent_hour_bounds_not_simultaneous; excludes_idle_and_cooling; GPU_host_proportional_proxy'})
    pd.DataFrame(electrical).to_csv(output / 'flexibility-electric-summary.csv', index=False, encoding='utf-8-sig')
    checks['scenario_rows_360'] = len(summary) == 360
    checks['electric_rows_1080'] = len(electrical) == 1080
    checks['zero_share_zero_removal'] = bool(summary.loc[summary.share.eq(0), 'max_independent_removable_resource_h'].max() < 1e-8)
    checks['zero_delay_zero_removal'] = bool(summary.loc[summary.max_extra_delay_h.eq(0), 'max_independent_removable_resource_h'].max() < 1e-8)
    checks['nonnegative_range_reductions'] = bool(summary[['day_range_reduction', 'week_range_reduction']].min().min() > -1e-6)
    checks['cumulative_service_constraints'] = bool(summary[['day_max_cumulative_residual', 'week_max_cumulative_residual', 'max_capacity_residual']].max().max() < 1e-6)
    (output / 'flexibility-validation.json').write_text(json.dumps({'checks': checks, 'all_passed': all(checks.values()),
        'sources': sources, 'year': 2025, 'hourly_directory': str(processed)}, indent=2, ensure_ascii=False), encoding='utf-8')
    if not all(checks.values()):
        raise ValueError('可移动负荷检查失败')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--baseline', type=Path, required=True)
    build(parser.parse_args())
