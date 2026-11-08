import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import pandas as pd

from flex_econ_copt import CITIES, SCENARIOS
from joint_capacity_inputs import atomic_json, sha256


FEASIBLE = {'optimal', 'feasible_with_gap'}


def cost_bound(case):
    total = case['total_annual_usd']
    lower = case['total_annual_lower_bound_usd']
    valid = case.get('solver_bound_available', True) and math.isfinite(lower) and 0 <= lower <= total + 10
    if valid:
        return min(lower, total), True
    # B 的机群台数固定。供能各项成本非负，设备费可作为可靠下界。
    return case['equipment_annual_usd'] if case['stage'] == 'B' else 0., False


def compare(a, b, c, name):
    if not a or not b or not c or any(case['status'] not in FEASIBLE for case in (a, b, c)):
        return {}
    if not a['servers'] == b['servers'] >= c['servers'] or len({case['input_sha256'] for case in (a, b, c)}) != 1:
        raise ValueError(f'{name} 的基准台数或配对输入不一致')
    equipment_saving = b['equipment_annual_usd'] - c['equipment_annual_usd']
    energy_saving = b['energy_annual_usd'] - c['energy_annual_usd']
    total_saving = b['total_annual_usd'] - c['total_annual_usd']
    if abs(equipment_saving + energy_saving - total_saving) > 1e-4:
        raise ValueError(f'{name} 的成本拆分不守恒')
    b_lower, b_valid = cost_bound(b)
    c_lower, c_valid = cost_bound(c)
    lower = max(0., b_lower - c['total_annual_usd'])
    upper = b['total_annual_usd'] - c_lower
    if a['total_annual_usd'] + 10 < b_lower or upper < -10:
        raise ValueError(f'{name} 的成本界违反嵌套情境关系')
    return {
        'servers_avoided': b['servers'] - c['servers'],
        'gpus_avoided': b['gpus'] - c['gpus'],
        'server_reduction_pct': (b['servers'] - c['servers']) / b['servers'] * 100,
        'purchase_avoided_usd': b['equipment_purchase_usd'] - c['equipment_purchase_usd'],
        'equipment_annual_saving_usd': equipment_saving,
        'energy_annual_saving_usd': energy_saving,
        'joint_annual_saving_usd': total_saving,
        'joint_annual_saving_lower_bound_usd': lower,
        'joint_annual_saving_upper_bound_usd': upper,
        'joint_saving_solver_bounds_available': b_valid and c_valid,
        'joint_saving_pct_of_fleet': total_saving / a['equipment_annual_usd'] * 100,
        'scheduling_annual_saving_usd': a['total_annual_usd'] - b['total_annual_usd'],
        'total_annual_saving_usd': a['total_annual_usd'] - c['total_annual_usd'],
        'covered_certified': b['servers'] > c['servers'] and lower > 0,
        'above_3pct_certified': b['servers'] > c['servers'] and lower > .03 * a['equipment_annual_usd'],
    }


def summarize(root):
    statuses = Counter()
    records = []
    for target in ('economic', 'alpha100'):
        for scenario in SCENARIOS:
            for city in CITIES:
                name = f'{city}-{scenario}-{target}'
                row = {'city': city, 'scenario': scenario, 'target': target}
                cases = []
                for stage in ('A', 'B', 'C'):
                    path = root / stage / f'{name}.json'
                    case = json.loads(path.read_text('utf-8')) if path.is_file() else None
                    status = case['status'] if case else 'pending'
                    statuses[(target, stage, status)] += 1
                    row[f'{stage}_status'] = status
                    if case:
                        row[f'{stage}_sha256'] = sha256(path)
                    if case and status in FEASIBLE:
                        for key in ('servers', 'gpus', 'equipment_purchase_usd', 'equipment_annual_usd',
                                    'energy_annual_usd', 'total_annual_usd'):
                            row[f'{stage}_{key}'] = case[key]
                    cases.append(case)
                row.update(compare(*cases, name))
                records.append(row)
    counts = [{'target': t, 'stage': s, 'status': p, 'cases': n} for (t, s, p), n in sorted(statuses.items())]
    return records, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records, counts = summarize(args.results)
    pd.DataFrame(records).to_csv(args.output / 'city-comparison.csv', index=False, encoding='utf-8-sig')
    atomic_json(args.output / 'city-comparison.json',
                {'generated_at': datetime.now(timezone.utc).isoformat(), 'counts': counts, 'cases': records})
    print(counts, flush=True)


if __name__ == '__main__':
    main()
