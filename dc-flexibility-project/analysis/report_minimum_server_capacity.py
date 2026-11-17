import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from flex_econ_copt import CITIES, SCENARIOS, crf, load_config
from joint_capacity_inputs import atomic_json
from minimum_server_capacity import DEADLINES


def interval(values):
    a = np.asarray(values, float)
    if len(a) != 8 or not np.isfinite(a).all():
        raise ValueError('城市汇总必须包含 8 个有效城市')
    return float(np.median(a)), float(a.min()), float(a.max())


def generate(args):
    manifest = json.loads((args.resources / 'resource-manifest.json').read_text('utf-8'))
    p, config = load_config(args.project)
    factor = crf(p['discount_rate'], p['compute_life_years']) + p['compute_fom_fraction']
    claude = pd.read_csv(args.claude_lower).set_index(['scenario', 'whole_h', 'split_h'])
    resource_lower = pd.read_csv(args.resource_lower).set_index(['scenario', 'whole_h', 'split_h'])
    first_part = pd.read_csv(args.first_part, dtype={'card_price': str})
    capacities, summary, comparisons, reserves, part1_comparisons, waiting, card_prices = [], [], [], [], [], [], []
    for scenario in SCENARIOS:
        fleet = manifest[scenario]
        annual_unit = config['cards']['server_price_usd'][scenario[0]]['value'] * factor
        for whole_h, split_h in DEADLINES:
            path = args.capacities / f'{scenario}-w{whole_h}-s{split_h}.json'
            result = json.loads(path.read_text('utf-8'))
            if result['gap_servers'] > 1 and not args.allow_open_bounds:
                raise ValueError(f'{scenario} {whole_h}/{split_h} 台数上下界差仍超过 1 台')
            capacity = {key: result[key] for key in ('scenario', 'whole_h', 'split_h', 'servers', 'lower_servers', 'gap_servers')}
            capacity.update(A_servers=fleet['baseline_units'],
                            capacity_proof_complete=result['gap_servers'] <= 1,
                            C_over_A=result['servers'] / fleet['baseline_units'],
                            resource_lp_lower_servers=float(resource_lower.loc[(scenario, whole_h, split_h), 'resource_lower_servers']),
                            resource_lp_lower_over_A=float(resource_lower.loc[(scenario, whole_h, split_h), 'resource_lower_over_A']),
                            claude_power_lower_ratio=float(claude.loc[(scenario, whole_h, split_h), 'C_min_over_K']),
                            seconds=result.get('seconds', result.get('solve_seconds', 0.)))
            capacities.append(capacity)
        main = capacities[-3]
        minimum = main['servers']
        price = config['cards']['server_price_usd'][scenario[0]]['value']
        for margin in (0.05, 0.10):
            reserved = int(np.ceil(minimum * (1 + margin)))
            reserves.append({'scenario': scenario, 'reserve_fraction': margin, 'minimum_servers': minimum,
                             'provisioned_servers': reserved, 'annual_equipment_saving_usd':
                                 (fleet['baseline_units'] - reserved) * annual_unit})
        for token in ('economic', 'alpha100'):
            city_rows = [json.loads((args.cities / f'{city}-{scenario}-{token}.json').read_text('utf-8')) for city in CITIES]
            if any(row['minimum_servers'] != minimum for row in city_rows):
                raise ValueError(f'{scenario} {token} 城市费用与最少台数不一致')
            baseline_fleet = fleet['baseline_units'] * annual_unit
            record = {**main, 'target': token, 'A_gpus': fleet['baseline_units'] * fleet['gpus_per_unit'],
                      'C_gpus': minimum * fleet['gpus_per_unit'],
                      'servers_avoided': fleet['baseline_units'] - minimum,
                      'gpus_avoided': (fleet['baseline_units'] - minimum) * fleet['gpus_per_unit'],
                      'reduction_pct': 100 * (1 - minimum / fleet['baseline_units']),
                      'purchase_avoided_usd': (fleet['baseline_units'] - minimum) * price,
                      'purchase_avoided_upper_usd': (fleet['baseline_units'] - main['lower_servers']) * price,
                      'equipment_annual_saving_usd': (fleet['baseline_units'] - minimum) * annual_unit}
            for field in ('energy_annual_saving_vs_A_usd', 'net_annual_saving_vs_A_usd',
                          'energy_annual_saving_vs_hourly_B_usd', 'net_annual_saving_vs_hourly_B_usd',
                          'hourly_vs_continuous_plan_total_difference_pct',
                          'net_annual_saving_vs_A_upper_usd',
                          'energy_annual_saving_vs_hourly_B_lower_usd',
                          'energy_annual_saving_vs_hourly_B_upper_usd', 'maximum_energy_lp_gap_usd'):
                for label, value in zip(('median', 'minimum', 'maximum'), interval([row[field] for row in city_rows])):
                    record[field + '_' + label] = value
            record['net_incremental_pct_of_fleet'] = 100 * record['net_annual_saving_vs_hourly_B_usd_median'] / baseline_fleet
            record['net_total_pct_of_fleet'] = 100 * record['net_annual_saving_vs_A_usd_median'] / baseline_fleet
            record['energy_method'] = ('feasible_upper_and_energy_relaxation_lower' if
                any(row.get('metrics_version') == 3 for row in city_rows) else 'continuous_hourly_resource_lp')
            record['annual_fleet_equipment_cost_usd'] = baseline_fleet
            for multiplier in (.5, 1., 2., 3.):
                saving = record['equipment_annual_saving_usd'] * multiplier
                card_prices.append({'scenario': scenario, 'target': token, 'price_multiplier': multiplier,
                    'minimum_servers': minimum, 'equipment_annual_saving_usd': saving,
                    'net_annual_saving_vs_A_usd': saving + record['energy_annual_saving_vs_A_usd_median'],
                    'scope': '最少容量策略下设备价格敏感性，供能沿用相同台数方案'})
            for row in city_rows:
                new = row['continuous_job_plan']
                original = row['original_C']
                old_total = original.get('total_annual_usd')
                old_gap = original.get('optimality_gap_annual_usd')
                valid_gap = old_gap is not None and np.isfinite(old_gap) and 0 <= old_gap <= (old_total or 0)
                original_task_peak = row.get('original_task_peak_mw', row['A']['peak_task_mw'])
                power_ratio = new['peak_task_power_mw'] / original_task_peak
                power_lower = record['claude_power_lower_ratio']
                if power_ratio < power_lower - .0005:
                    raise ValueError(f'{row["city"]} {scenario} 功率口径低于独立功率 LP 下界')
                comparisons.append({
                    'scenario': scenario, 'city': row['city'], 'target': token,
                    'minimum_servers': minimum, 'original_MIP_servers': original.get('servers'),
                    'original_MIP_status': original['status'],
                    'original_MIP_annual_gap_usd': old_gap if valid_gap else None,
                    'original_MIP_gap_valid': valid_gap,
                    'new_feasible_C_annual_usd': new['total_annual_usd'],
                    'original_MIP_C_annual_usd': old_total,
                    'total_cost_difference_pct': 100 * (new['total_annual_usd'] - old_total) / old_total if old_total else None,
                    'independent_capacity_plan_cost_difference_pct':
                        100 * (row.get('independent_plan_C_annual_usd', new['total_annual_usd']) - old_total) / old_total if old_total else None,
                    'energy_upper_source': row.get('energy_upper_source', 'verified_capacity_schedule'),
                    'new_C_annual_lower_usd': new['total_annual_lower_bound_usd'],
                    'new_C_annual_bracket_usd': new['optimality_gap_annual_usd'],
                    'task_power_peak_ratio': power_ratio, 'claude_power_lower_ratio': power_lower,
                    'mean_bottleneck_fraction': row.get('capacity_plan_metrics', new)['mean_bottleneck_fraction'],
                    'hours_above_95pct_capacity': row.get('capacity_plan_metrics', new)['hours_above_95pct_capacity'],
                })
                for stage, metrics in (('C', row.get('capacity_plan_metrics', new)),
                                       ('B', row['hourly_baseline_fleet'])):
                    waiting.append({'scenario': scenario, 'city': row['city'], 'target': token,
                        'stage': stage, **{key: value for key, value in metrics.items() if
                            key.startswith(('flex_energy_', 'mean_', 'hours_above_'))},
                        **(row.get('whole_job_waiting_C', {}) if stage == 'C' else {}),
                        'scope': '已核验容量可行调度' if stage == 'C' else 'B 能源模型；缺少逐作业调度时不推算等待'})
                target_rows = first_part.loc[first_part.city.eq(row['city']) & first_part.scenario.eq(scenario) &
                    (first_part.alpha.isna() if token == 'economic' else first_part.alpha.eq(1.))]
                flexible = target_rows.loc[target_rows.flex.eq('both') & target_rows.card_price.eq('forced0') &
                                           target_rows.whole_h.eq(12) & target_rows.split_h.eq(48)]
                fixed = target_rows.loc[target_rows.flex.eq('none')]
                if len(flexible) != 1 or len(fixed) != 1:
                    raise ValueError(f'第一部分对照不唯一 {scenario} {row["city"]} {token}')
                annual = 8760 / flexible.hours.iloc[0]
                first_energy = float(flexible.energy_cost_usd.iloc[0] * annual)
                first_A = float(fixed.energy_cost_usd.iloc[0] * annual)
                old_b = row['original_B']
                part1_comparisons.append({
                    'scenario': scenario, 'city': row['city'], 'target': token,
                    'first_part_B_energy_annual_usd': first_energy,
                    'first_part_A_energy_annual_usd': first_A,
                    'new_hourly_B_energy_annual_usd': row['hourly_baseline_fleet']['energy_annual_usd'],
                    'original_continuous_job_B_energy_annual_usd': old_b.get('energy_annual_usd'),
                    'original_continuous_job_B_status': old_b['status'],
                    'new_A_energy_annual_usd': row['A']['energy_annual_usd'],
                    'new_hourly_B_minus_first_part_energy_usd': row['hourly_baseline_fleet']['energy_annual_usd'] - first_energy,
                    'old_job_B_minus_first_part_energy_usd': old_b.get('energy_annual_usd', np.nan) - first_energy,
                    'first_part_energy_saving_usd': first_A - first_energy,
                    'new_hourly_B_energy_saving_usd': row['A']['energy_annual_usd'] - row['hourly_baseline_fleet']['energy_annual_usd'],
                    'original_job_B_energy_saving_usd': row['A']['energy_annual_usd'] - old_b.get('energy_annual_usd', np.nan),
                    'new_hourly_saving_minus_first_part_usd': row['A']['energy_annual_usd'] -
                        row['hourly_baseline_fleet']['energy_annual_usd'] - first_A + first_energy,
                    'original_job_saving_minus_first_part_usd': row['A']['energy_annual_usd'] -
                        old_b.get('energy_annual_usd', np.nan) - first_A + first_energy,
                    'scope': '第一部分原表与新资源容量模型存在待机台数及资源约束差异，费用差不单独归因于连续运行',
                })
            summary.append(record)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('capacity-sensitivity', capacities), ('part2-main-table', summary),
                       ('original-mip-comparison', comparisons), ('equipment-reserve-sensitivity', reserves),
                       ('first-part-comparison', part1_comparisons), ('waiting-and-utilization', waiting),
                       ('equipment-price-sensitivity', card_prices)):
        pd.DataFrame(rows).to_csv(args.output / f'{name}.csv', index=False, encoding='utf-8-sig')
    atomic_json(args.output / 'brief-summary.json',
                {'generated_at': datetime.now(timezone.utc).isoformat(), 'annual_factor': factor,
                 'capacity': capacities, 'main': summary, 'comparisons': comparisons, 'reserves': reserves})


def main():
    parser = argparse.ArgumentParser()
    for name in ('project', 'resources', 'capacities', 'cities', 'city-inputs', 'claude-lower', 'resource-lower', 'first-part', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--allow-open-bounds', action='store_true',
                        help='保留明确标注的容量上下界，未达到一台精度的情境不宣称最少台数已证明')
    generate(parser.parse_args())


if __name__ == '__main__':
    main()
