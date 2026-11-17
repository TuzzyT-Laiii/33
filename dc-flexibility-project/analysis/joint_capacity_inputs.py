import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from flex_econ_copt import CITIES, SCENARIOS, crf, load_config, solve


CONFIGURATION = {'A': 'C20', 'B': 'C21', 'C': 'C22', 'D': 'C28'}


def sha256(path):
    with path.open('rb') as handle:
        return file_digest(handle, 'sha256').hexdigest()


def atomic_json(path, data):
    pending = path.with_suffix('.partial')
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    pending.replace(path)


def prepare(args):
    catalogue = {row['configuration_id']: row for row in json.loads(args.catalogue.read_text('utf-8'))}
    manifest = {}
    for scenario in SCENARIOS:
        key, use = scenario.split('-')
        config = catalogue[CONFIGURATION[key]]
        source_path = args.reference / f'{scenario}-10MW.parquet'
        template_path = args.city_inputs / f'frankfurt-{scenario}.parquet'
        source = pd.read_parquet(source_path)
        template = pd.read_parquet(template_path)
        np.testing.assert_array_equal(source.source_hour_index, template.source_hour_index)
        np.testing.assert_allclose(source.it_mw, template.it_mw, rtol=0, atol=1e-8)
        if source.unit_count.nunique() != 1 or template.base.nunique() != 1:
            raise ValueError(f'{scenario} 的参考台数和待机功率必须是常量')
        units = int(source.unit_count.iloc[0])
        cores = config['cpu_cores_per_unit']
        cards = config['gpu_devices_per_unit']
        resources = pd.DataFrame({
            'source_hour_index': source.source_hour_index,
            'cpu_core_hours': source.u_cpu_proxy * units * cores,
            'gpu_hours': source.allocated_gpu_fraction * units * cards,
        })
        if not np.isfinite(resources.to_numpy()).all() or (resources[['cpu_core_hours', 'gpu_hours']] < 0).any().any():
            raise ValueError(f'{scenario} 的资源需求无效')
        # 先在统一小时粒度上确定原时序的 CPU/GPU 容量。
        peak = max(float(source.u_cpu_proxy.max()), float(source.allocated_gpu_fraction.max()))
        baseline = int(np.ceil(units * peak - 1e-8))
        if not 0 < baseline <= units:
            raise ValueError(f'{scenario} 的基准台数超出参考设备范围')
        output_path = args.output / f'{scenario}.parquet'
        resources.to_parquet(output_path, index=False)
        manifest[scenario] = {
            'use': use, 'reference_units': units, 'baseline_units': baseline,
            'cores_per_unit': cores, 'gpus_per_unit': cards,
            'idle_mw_per_unit': float(template.base.iloc[0]) / units,
            'reference_it_mw_per_unit': float(template.cap_it_mw.iloc[0]) / units,
            'hours': len(resources), 'reference_sha256': sha256(source_path),
            'template_sha256': sha256(template_path), 'resources_sha256': sha256(output_path),
        }
        print(scenario, manifest[scenario], flush=True)
    atomic_json(args.output / 'resource-manifest.json', manifest)


def baseline_case(task):
    city, scenario, alpha, args, fleet = task
    path = args.city_inputs / f'{city}-{scenario}.parquet'
    frame = pd.read_parquet(path)
    resources = pd.read_parquet(args.resources / f'{scenario}.parquet')
    np.testing.assert_array_equal(frame.source_hour_index, resources.source_hour_index)
    parameters, config = load_config(args.project)
    result = solve_baseline(frame, parameters, config, city, scenario, alpha, fleet)
    result['input_sha256'] = sha256(path)
    name = f"{city}-{scenario}-{'economic' if alpha is None else 'alpha100'}"
    atomic_json(args.output / f'{name}.json', result)
    print(name, result['status'], flush=True)


def solve_baseline(frame, parameters, config, city, scenario, alpha, fleet):
    frame = frame.copy()
    baseline = fleet['baseline_units']
    frame.base = fleet['idle_mw_per_unit'] * baseline
    frame.it_mw = frame[['base', 'rigid', 'unknown', 'whole', 'split']].sum(axis=1)
    frame.facility_mw = frame.pue * frame.it_mw
    frame.attrs['scenario'] = scenario
    result, _ = solve(frame, parameters, city, alpha=alpha, design='opt', whole_h=0, split_h=0)
    result['stage'] = 'A'
    result['servers'] = baseline
    result['gpus'] = baseline * fleet['gpus_per_unit']
    if result['status'] == 'optimal':
        price = config['cards']['server_price_usd'][scenario[0]]['value']
        factor = crf(parameters['discount_rate'], parameters['compute_life_years']) + parameters['compute_fom_fraction']
        result['equipment_purchase_usd'] = baseline * price
        result['equipment_annual_usd'] = baseline * price * factor
        result['energy_annual_usd'] = result['total_cost_usd'] * 8760 / len(frame)
        result['total_annual_usd'] = result['equipment_annual_usd'] + result['energy_annual_usd']
    return result


def baseline(args):
    manifest = json.loads((args.resources / 'resource-manifest.json').read_text('utf-8'))
    tasks = [(city, scenario, alpha, args, manifest[scenario])
             for alpha in (None, 1.0) for city in CITIES for scenario in SCENARIOS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(baseline_case, tasks))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--reference', type=Path, required=True)
    prep.add_argument('--catalogue', type=Path, required=True)
    run = commands.add_parser('baseline')
    run.add_argument('--project', type=Path, required=True)
    run.add_argument('--resources', type=Path, required=True)
    run.add_argument('--workers', type=int, default=2)
    for command in (prep, run):
        command.add_argument('--city-inputs', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    {'prepare': prepare, 'baseline': baseline}[args.command](args)


if __name__ == '__main__':
    main()
