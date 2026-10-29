import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eight_city_onsite_solver import annual_costs, sha256, solve_onsite, value
from run_eight_city_flex_value import DEFAULT_CONFIG, PROJECT, load_input, write_json


def require_optimal(result):
    if result['summary']['status'] != 'optimal':
        raise AssertionError(result['summary'])
    return result['summary']['annual_cost_usd']


def equal_cost(a, b):
    left, right = require_optimal(a), require_optimal(b)
    np.testing.assert_allclose(left, right, rtol=2e-7, atol=.2)
    return left - right


def generated_test_frame(n=168, flat=False):
    t = np.arange(n)
    solar = np.maximum(0., np.sin((t % 24 - 6) * np.pi / 12)) * .8
    wind = np.clip(.25 + .2 * np.sin(t / 31) + .12 * np.cos(t / 7), 0, .7)
    return pd.DataFrame({'time_utc': pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC'),
        'it_mw': np.full(n, 10.) if flat else 5. + 2. * np.sin(t / 13) ** 2,
        'pue': 1.15 + .08 * np.sin(t / 20) ** 2,
        'pv_capacity_factor': solar, 'wind_capacity_factor': wind})


def structural_validation(config, legacy_path):
    checks, maximum_residual, objective_gaps = [], 0., []
    frame = generated_test_frame(72)
    for deadline in (0, 1, 6, 24, 168):
        for fraction in (.5, 1.):
            for storage, alpha in ((0, None), (4, .95), ('opt', 1.)):
                settings = dict(deadline=deadline, fraction=fraction, storage=storage, alpha=alpha, idle_mw=2.)
                backlog = solve_onsite(frame, config, **settings)
                explicit = solve_onsite(frame, config, explicit=True, **settings)
                gap = equal_cost(backlog, explicit)
                objective_gaps.append(abs(gap))
                maximum_residual = max(maximum_residual, max(backlog['summary']['diagnostics'].values()))
    checks.append({'check': 'backlog_vs_explicit_x_a_t', 'status': 'passed', 'comparisons': len(objective_gaps),
                   'maximum_objective_gap_usd': max(objective_gaps), 'hours': len(frame)})
    for storage in config['sweep']['S']:
        costs = []
        fixed = solve_onsite(frame, config, deadline=0, storage=storage, alpha=.95, idle_mw=2.)
        for deadline in config['sweep']['D']:
            result = solve_onsite(frame, config, deadline=deadline, storage=storage, alpha=.95, idle_mw=2.)
            costs.append(require_optimal(result))
        if max(np.diff(costs)) > .2:
            raise AssertionError(f'V(D)单调性失败：{storage}, {costs}')
        zero = solve_onsite(frame, config, deadline=168, fraction=0, storage=storage, alpha=.95, idle_mw=2.)
        gap = equal_cost(fixed, zero)
        checks.append({'check': 'monotonic_D_and_phi_zero', 'S': storage, 'status': 'passed',
                       'phi_zero_cost_gap_usd': gap, 'D_zero_value_usd': 0.})
    impossible = frame.copy()
    impossible.loc[0, ['pv_capacity_factor', 'wind_capacity_factor']] = 0
    result = solve_onsite(impossible, config, deadline=168, storage=0, alpha=1, idle_mw=2.)
    if result['summary']['status'] != 'infeasible' or result['summary']['infeasibility_reason'] != 'zero_renewable_hour_with_idle_load_no_storage':
        raise AssertionError('S0/α100不可行分类失败')
    checks.append({'check': 'S0_alpha100_infeasibility', 'status': 'passed',
                   'reason': result['summary']['infeasibility_reason']})
    flat = generated_test_frame(168, flat=True)
    flat['wind_capacity_factor'] = np.maximum(.05, flat.wind_capacity_factor)
    equal_cost(solve_onsite(flat, config, deadline=0, rmax=1, idle_mw=2.),
               solve_onsite(flat, config, deadline=168, rmax=1, idle_mw=2.))
    checks.append({'check': 'r1_full_utilization_zero_value', 'status': 'passed'})
    base = solve_onsite(frame, config, deadline=0, storage=0, alpha=.95, idle_mw=2.)
    assets = {key: base['summary'][key] for key in ('pv_mw', 'wind_mw', 'battery_power_mw', 'battery_energy_mwh')}
    first = solve_onsite(frame, config, deadline=24, idle_mw=2., fixed_assets=assets, physical=True)
    changed = copy.deepcopy(config)
    for key in ('pv_capex', 'wind_capex', 'compute_capex_usd_mw', 'grid_price_usd_mwh'):
        changed['parameters'][key]['value'] *= 2
    second = solve_onsite(frame, changed, deadline=24, idle_mw=2., fixed_assets=assets, physical=True)
    np.testing.assert_allclose(first['summary']['grid_mwh'], second['summary']['grid_mwh'], atol=1e-6)
    checks.append({'check': 'fixed_assets_physical_objective_independent_of_prices', 'status': 'passed',
                   'grid_mwh': first['summary']['grid_mwh']})
    if legacy_path is not None:
        spec = importlib.util.spec_from_file_location('legacy_grid_solver', legacy_path)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        unit_cost = annual_costs(config)
        old = {'reference_it_mw': 10, 'idle_fraction': .2, 'baseline_service_per_hour': 1,
               'annual_operating_multiplier': 8760 / len(flat), 'flexible_fraction': 1,
               'deadline_h': 24, 'capacity_multiple_max': 2, 'scenario_label': '168小时等价基准',
               'battery_charge_efficiency': np.sqrt(.85), 'battery_discharge_efficiency': np.sqrt(.85),
               'battery_hourly_loss': 0,
               'cost_basis': {'currency': 'USD', 'price_year': 2024, 'source': '统一参数配置', 'parameter_status': 'validation'},
               'costs': {}, 'capacity_limits': {}}
        keys = ['compute_module_usd_year', 'pv_usd_mw_year', 'wind_usd_mw_year', 'battery_power_usd_mw_year', 'battery_energy_usd_mwh_year']
        for key, amount in zip(keys, unit_cost):
            old['costs'][key] = {'value': float(amount), 'source': '统一参数配置', 'status': 'validation'}
        old['costs']['battery_throughput_usd_mwh'] = {'value': value(config, 'battery_throughput_usd_mwh'), 'source': '统一参数配置', 'status': 'validation'}
        for key, amount in [('pv_mw', 10000), ('wind_mw', 10000), ('battery_power_mw', 100), ('battery_energy_mwh', 2000), ('compute_capacity_multiple', 2)]:
            old['capacity_limits'][key] = {'value': amount, 'source': '统一参数配置', 'status': 'validation'}
        for storage, legacy_name in ((0, 'CD'), ('opt', 'CSD')):
            first = legacy.solve_grid(flat, old, legacy_name, storage == 'opt', True)
            second = solve_onsite(flat, config, deadline=24, storage=storage, alpha=1, idle_mw=2.)
            if first['summary']['status'] != 'optimal':
                raise AssertionError(first['summary'])
            left, right = first['summary']['annual_cost_usd'], require_optimal(second)
            np.testing.assert_allclose(left, right, atol=.3, rtol=2e-7)
            checks.append({'check': 'legacy_168h_solver', 'status': 'passed', 'S': storage,
                           'objective_gap_usd': left - right, 'legacy_sha256': sha256(legacy_path)})
    return {'status': 'passed', 'checks': checks, 'maximum_conservation_residual': maximum_residual,
            'generated_inputs': True, 'scope': '显式构造序列仅用于数学结构核验，不作为城市观测或收益结果'}


def annual_validation_group(task):
    config, directory, city, scenario, output, alpha, storage = task
    loaded = load_input(directory, city, scenario, config)
    if loaded is None:
        raise FileNotFoundError(f'真实全年输入未就绪：{city}/{scenario}')
    frame, idle, design, provenance = loaded
    base = dict(idle_mw=idle, design_mw=design, emission_factor=config['cities'][city]['emission_factor'])
    fixed = None
    costs = []
    for deadline in config['sweep']['D']:
        folder = output / 'annual-validation-cases' / city / scenario / f'a{alpha}-S{storage}-D{deadline}'
        target = folder / 'summary.json'
        if target.exists():
            summary = json.loads(target.read_text(encoding='utf-8'))
            valid = (summary.get('config_sha256') == config['_validation_config_sha256'] and
                     summary['input_provenance']['sha256'] == provenance['sha256'])
        else:
            valid = False
        if valid and summary['status'] == 'optimal':
            result = {'summary': summary}
        else:
            result = solve_onsite(frame, config, deadline=deadline, storage=storage, alpha=alpha, **base)
            summary = {**result['summary'], 'city': city, 'scenario': scenario,
                       'config_sha256': config['_validation_config_sha256'], 'input_provenance': provenance}
            write_json(target, summary)
            if not result['hourly'].empty:
                result['hourly'].to_parquet(folder / 'hourly.parquet', index=False, compression='zstd')
        if result['summary']['status'] not in ('optimal', 'infeasible'):
            raise AssertionError(result['summary'])
        costs.append(result['summary'].get('annual_cost_usd'))
        if deadline == 0:
            fixed = result
        print(json.dumps({'validation': scenario, 'alpha': alpha, 'S': storage, 'D': deadline,
                          'cost': costs[-1], 'seconds': summary['solve_seconds']}), flush=True)
    feasible_costs = [cost for cost in costs if cost is not None]
    first_feasible = next((index for index, cost in enumerate(costs) if cost is not None), len(costs))
    if any(cost is None for cost in costs[first_feasible:]):
        raise AssertionError('D增加后出现权限嵌套可行性异常')
    if len(feasible_costs) > 1 and max(np.diff(feasible_costs)) > max(.3, feasible_costs[0] * 2e-7):
        raise AssertionError(f'全年柔性成本单调性失败：{scenario}/{storage}/{alpha}/{costs}')
    zero = solve_onsite(frame, config, deadline=168, fraction=0, storage=storage, alpha=alpha, **base)
    if fixed['summary']['status'] != zero['summary']['status']:
        raise AssertionError('φ0与D0可行性不同')
    gap = equal_cost(fixed, zero) if fixed['summary']['status'] == 'optimal' else None
    checks = [{'check': 'annual_D_zero_phi_zero_monotonic_and_conservation', 'scenario': scenario,
               'alpha': alpha, 'S': storage, 'phi_zero_gap_usd': gap, 'status': 'passed',
               'infeasible_D_count': len(costs) - len(feasible_costs),
               'fixed_reference_status': fixed['summary']['status'],
               'D_zero_value_usd': 0. if fixed['summary']['status'] == 'optimal' else None}]
    if storage == 0 and alpha is None:
        impossible = solve_onsite(frame, config, deadline=168, storage=0, alpha=1, **base)
        if impossible['summary']['status'] not in ('optimal', 'infeasible'):
            raise AssertionError('α100%边界未求解完成')
        checks.append({'check': 'annual_S0_alpha100_classification', 'scenario': scenario, 'status': 'passed',
                       'solver_status': impossible['summary']['status'],
                       'reason': impossible['summary'].get('infeasibility_reason'),
                       'note': '若每小时至少一种资源可用，充足装机可能可行；只按实际求解分类'})
    return {'checks': checks, 'input': provenance, 'cases': len(costs)}


def annual_validation(config, directory, city, scenarios, output, workers):
    tasks = [(config, directory, city, scenario, output, alpha, storage)
             for scenario in scenarios for alpha in (None, .95) for storage in config['sweep']['S']]
    checks, inputs, count = [], {}, 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(annual_validation_group, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            checks.extend(result['checks'])
            inputs[result['input']['sha256']] = result['input']
            count += result['cases']
    return {'status': 'passed', 'annual_real_input_gate': True, 'city': city, 'scenarios': scenarios,
            'checks': checks, 'input_provenance': list(inputs.values()), 'case_count': count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--legacy-solver', type=Path)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--city', default='phoenix')
    parser.add_argument('--scenario', nargs='+', default=['A-HPC', 'D-TRN'])
    parser.add_argument('--annual', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'results/eight-city-flex-value-20261029')
    parser.add_argument('--details-dir', type=Path, default=PROJECT / 'data/processed/eight-city-flex-value-20261029')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    report = structural_validation(config, args.legacy_solver)
    report.update(utc=datetime.now(timezone.utc).isoformat(), config_sha256=sha256(args.config),
                  solver_sha256=sha256(Path(__file__).with_name('eight_city_onsite_solver.py')), annual_real_input_gate=False)
    if args.annual:
        config['_validation_config_sha256'] = sha256(args.config)
        report['annual'] = annual_validation(config, args.input_dir, args.city, args.scenario, args.details_dir, args.workers)
        report['annual_real_input_gate'] = True
    write_json(args.output_dir / 'validation.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
