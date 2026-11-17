import argparse
import copy
import json
from pathlib import Path

from flex_econ_copt import CITIES, SCENARIOS
from joint_capacity_inputs import atomic_json, sha256


def collect(args):
    args.output.mkdir(parents=True, exist_ok=True)
    for scenario in args.scenario or SCENARIOS:
        for city in CITIES:
            for target in ('economic', 'alpha100'):
                name = f'{city}-{scenario}-{target}.json'
                available = [folder / name for folder in args.source if (folder / name).is_file()]
                if not available:
                    raise ValueError(f'缺少八城市结果 {name}')
                candidates = [(path, json.loads(path.read_text('utf-8'))) for path in available]
                bounded = [(path, row) for path, row in candidates if row.get('metrics_version') == 3]
                path, row = bounded[0] if bounded else candidates[0]
                row = copy.deepcopy(row)
                row.setdefault('input_sha256', row['A']['input_sha256'])
                row['source_result_sha256'] = sha256(path)
                row['capacity_plan_metrics'] = {key: value for key, value in row['continuous_job_plan'].items()
                    if key.startswith(('flex_energy_', 'mean_', 'hours_above_', 'peak_task_'))}
                row['independent_plan_C_annual_usd'] = row['continuous_job_plan']['total_annual_usd']
                row['energy_upper_source'] = 'verified_capacity_schedule'
                if bounded:
                    upper_candidates = []
                    old = row['original_C']
                    if (old.get('servers') == row['minimum_servers'] and
                            old['status'] in ('optimal', 'feasible_with_gap') and
                            old.get('input_sha256') == row['input_sha256'] and
                            max(old.get('audit', {'missing': 1.}).values()) <= 1e-4):
                        upper_candidates.append(('original_C', old))
                    for other_path, other in candidates:
                        plan = other['continuous_job_plan']
                        if (other['minimum_servers'] == row['minimum_servers'] and
                                other.get('input_sha256', other['A']['input_sha256']) == row['input_sha256'] and
                                plan.get('whole_schedule_fixed') is False and
                                plan.get('status') in ('optimal', 'feasible_with_gap') and
                                max(plan.get('audit', {'missing': 1.}).values()) <= 1e-4):
                            upper_candidates.append(('fixed_whole_plan_lp:' + sha256(other_path), plan))
                    current = row['continuous_job_plan']
                    for label, candidate in upper_candidates:
                        if candidate['total_annual_usd'] < current['total_annual_usd']:
                            current = dict(candidate)
                            current['peak_task_power_mw'] = candidate.get('peak_task_power_mw',
                                candidate['peak_it_mw'] - row['minimum_servers'] * args.idle[scenario])
                            row['energy_upper_source'] = label
                    lower = row['continuous_job_plan']['total_annual_lower_bound_usd']
                    current.update(total_annual_lower_bound_usd=lower,
                        optimality_gap_annual_usd=max(0., current['total_annual_usd'] - lower),
                        status='feasible_energy_bracket')
                    row['continuous_job_plan'] = current
                    relaxed = row['hourly_capacity_fixed']
                    row['hourly_capacity_fixed'] = {key: relaxed[key] for key in (
                        'servers', 'energy_annual_usd', 'equipment_annual_usd', 'equipment_purchase_usd',
                        'total_annual_usd')}
                    row['hourly_capacity_fixed']['status'] = 'energy_only_relaxation'
                    row['hourly_capacity_fixed']['scope'] = '能源费用下界，不提供资源调度或供能资产方案'
                    b = row['hourly_baseline_fleet']
                    b['source_status'] = b['status']
                    b['status'] = 'feasible_energy_bracket'
                    b['source_stage'] = b.get('stage')
                    b['stage'] = 'B'
                c, b, a = row['continuous_job_plan'], row['hourly_baseline_fleet'], row['A']
                for short, full in {'pv': 'pv_mw', 'wind': 'wind_mw', 'battery_power': 'battery_mw',
                                    'battery_energy': 'battery_mwh', 'grid_capacity': 'grid_cap_mw'}.items():
                    if short in c:
                        c[full] = c[short]
                    elif full in c:
                        c[short] = c[full]
                row.update(energy_annual_saving_vs_A_usd=a['energy_annual_usd'] - c['energy_annual_usd'],
                    net_annual_saving_vs_A_usd=a['total_annual_usd'] - c['total_annual_usd'],
                    net_annual_saving_vs_A_upper_usd=a['total_annual_usd'] - c['total_annual_lower_bound_usd'],
                    energy_annual_saving_vs_hourly_B_usd=b['energy_annual_usd'] - c['energy_annual_usd'],
                    net_annual_saving_vs_hourly_B_usd=b['total_annual_usd'] - c['total_annual_usd'],
                    energy_annual_saving_vs_hourly_B_lower_usd=b.get('total_annual_lower_bound_usd', b['total_annual_usd']) -
                        b['equipment_annual_usd'] - c['energy_annual_usd'],
                    energy_annual_saving_vs_hourly_B_upper_usd=b['energy_annual_usd'] -
                        c['total_annual_lower_bound_usd'] + c['equipment_annual_usd'],
                    net_saving_pct_of_A_fleet=100 * (a['total_annual_usd'] - c['total_annual_usd']) / a['equipment_annual_usd'],
                    maximum_energy_lp_gap_usd=c['optimality_gap_annual_usd'])
                row['energy_bound_width_pct_of_total'] = 100 * c['optimality_gap_annual_usd'] / c['total_annual_usd']
                if bounded:
                    row['hourly_vs_continuous_plan_total_difference_pct'] = row['energy_bound_width_pct_of_total']
                atomic_json(args.output / name, row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--resources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenario', choices=SCENARIOS, action='append')
    args = parser.parse_args()
    args.idle = {key: value['idle_mw_per_unit'] for key, value in
        json.loads((args.resources / 'resource-manifest.json').read_text('utf-8')).items()}
    collect(args)


if __name__ == '__main__':
    main()
