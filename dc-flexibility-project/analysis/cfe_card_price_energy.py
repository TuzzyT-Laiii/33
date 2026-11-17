import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from bounded_city_energy import energy_solve, feasible_dispatch
from flex_econ_copt import CITIES, SCENARIOS, load_config
from joint_capacity_inputs import atomic_json, sha256


LAYERS = ['rigid', 'unknown', 'whole', 'split']
RESERVES = (0, 5, 10)
CONTEXT = {}


def relaxed_energy(args, frame, scenario, city, target, alpha, baseline, idle):
    if scenario.endswith('-INF'):
        result = energy_solve(frame, CONTEXT['parameters'], city, alpha, baseline, idle)
        return result['energy_annual_usd'], result['max_residual'], 'new_energy_only_lp'

    name = f'{city}-{scenario}-{target}.json'
    prior = args.archived_energy_bounds / name
    receipt = json.loads((args.collected / name).read_text('utf-8'))
    if sha256(prior) != receipt['source_result_sha256']:
        raise ValueError(f'{name} 的供能下界存档与交付记录不一致')
    original = json.loads(prior.read_text('utf-8'))
    bound = original['hourly_baseline_fleet']
    if (original['metrics_version'] != 3 or bound['servers'] != baseline or
            original['input_sha256'] != sha256(args.city_inputs / f'{city}-{scenario}.parquet')):
        raise ValueError(f'{name} 的供能下界口径不一致')
    return bound['total_annual_lower_bound_usd'] - bound['equipment_annual_usd'], None, 'archived_energy_only_lp'


def compute_case(task):
    scenario, city = task
    args = CONTEXT['args']
    checkpoint = args.output.parent / 'cases' / f'{scenario}-{city}.json'
    if checkpoint.is_file():
        return json.loads(checkpoint.read_text('utf-8'))
    fleet, capacity, reference, dispatch = CONTEXT['inputs'][scenario]
    frame = pd.read_parquet(args.city_inputs / f'{city}-{scenario}.parquet')
    np.testing.assert_array_equal(frame.source_hour_index, reference.source_hour_index)
    np.testing.assert_allclose(frame[LAYERS], reference[LAYERS], rtol=0, atol=1e-10)
    baseline = fleet['baseline_units']
    idle = fleet['idle_mw_per_unit']
    rows = []
    for target, alpha in (('economic', None), ('alpha100', 1.)):
        relaxed, relaxed_residual, source = relaxed_energy(args, frame, scenario, city, target, alpha, baseline, idle)
        for reserve in RESERVES if dispatch is not None else (0,):
            servers = math.ceil(capacity['servers'] * (100 + reserve) / 100)
            row = dict(scenario=scenario, city=city, target=target, reserve_pct=reserve,
                       original_servers=baseline, capacity_min_servers=capacity['servers'],
                       provisioned_servers=servers,
                       energy_relaxed_original_usd=relaxed, relaxed_source=source,
                       relaxed_residual=relaxed_residual)
            if dispatch is not None:
                result = energy_solve(frame, CONTEXT['parameters'], city, alpha, servers, idle, dispatch)
                row.update(reserve_energy_usd=result['energy_annual_usd'],
                           reserve_energy_residual=result['max_residual'],
                           annual_incremental_idle_it_mwh=(servers - capacity['servers']) * idle * 8760,
                           reserve_pv_mw=result['pv_mw'], reserve_wind_mw=result['wind_mw'],
                           reserve_storage_mwh=result['battery_mwh'], reserve_grid_mwh=result['grid_mwh'])
            rows.append(row)
    atomic_json(checkpoint, rows)
    print(scenario, city, len(rows), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    for name in ('project', 'reference', 'resources', 'whole-jobs', 'city-inputs', 'capacities',
                 'archived-energy-bounds', 'collected', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    parameters, _ = load_config(args.project)
    manifest = json.loads((args.resources / 'resource-manifest.json').read_text('utf-8'))
    inputs = {}
    for scenario in SCENARIOS:
        fleet = manifest[scenario]
        if not fleet['gpus_per_unit']:
            continue
        capacity = json.loads((args.capacities / f'{scenario}-w12-s48.json').read_text('utf-8'))
        if capacity['gap_servers'] != 0:
            raise ValueError(f'{scenario} 的基准减配台数尚未证明最优')
        frame = pd.read_parquet(args.city_inputs / f'frankfurt-{scenario}.parquet')
        dispatch = None
        if scenario.startswith('D-'):
            verified, _, _, _ = feasible_dispatch(args, scenario, fleet, capacity, frame)
            dispatch = verified[:, 2]
        inputs[scenario] = fleet, capacity, frame, dispatch
    CONTEXT.update(args=args, parameters=parameters, inputs=inputs)
    (args.output.parent / 'cases').mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('fork')) as pool:
        rows = [row for group in pool.map(compute_case, [(scenario, city) for scenario in inputs for city in CITIES])
                for row in group]
    result = pd.DataFrame(rows).sort_values(['scenario', 'city', 'target', 'reserve_pct'])
    if len(result) != 240 or result.duplicated(['scenario', 'city', 'target', 'reserve_pct']).any():
        raise ValueError('供能敏感性结果的样本数或组合键错误')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, float_format='%.12g')


if __name__ == '__main__':
    main()
