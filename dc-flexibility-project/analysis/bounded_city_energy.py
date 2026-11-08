import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from capacity_plan import apply_whole_plan
from fixed_capacity_city_energy import delay_summary, whole_waiting
from flex_econ_copt import CITIES, SCENARIOS, crf, load_config, solve
from joint_capacity_copt import resource_layers
from joint_capacity_inputs import atomic_json, sha256


def feasible_dispatch(args, scenario, fleet, certificate, frame):
    hours, rigid, whole, split = resource_layers(args, scenario, fleet)
    whole = np.column_stack((whole, frame.whole))
    fixed = np.column_stack((rigid, frame.rigid + frame.unknown))
    hourly = fleet['use'] in ('CLD', 'INF')
    if not hourly:
        jobs = pd.read_parquet(args.whole_jobs / f'{scenario}.parquet')
        plan = pd.read_parquet(args.capacities / f'{scenario}-w12-s48.whole-plan.parquet')
        fixed += apply_whole_plan(whole, hours, jobs, plan)
    profiles = [(np.column_stack((split, frame.split)), 48)]
    if hourly:
        profiles.append((whole, 12))
    base = certificate.get('reused_resource_model', scenario)
    saved = args.capacities / f'{base}-w12-s48.npz'
    required = any(np.any(profile[:, :2] > 0) for profile, _ in profiles)
    assignment = np.load(saved)['assignment'] if required else np.empty(0)
    offset, delays, weights = 0, [], []
    for profile, deadline in profiles:
        origins = np.flatnonzero(np.any(profile[:, :2] > 0, axis=1))
        targets = np.searchsorted(hours, hours[origins, None] + np.arange(deadline + 1))
        valid = (targets < len(hours)) & (hours[targets.clip(max=len(hours) - 1)] ==
                                         hours[origins, None] + np.arange(deadline + 1))
        origin, shift = np.nonzero(valid)
        fractions = assignment[offset:offset + len(origin)]
        if len(fractions) != len(origin):
            raise ValueError('容量证书缺少完整的可拆分任务调度')
        np.testing.assert_allclose(np.bincount(origin, weights=fractions, minlength=len(origins)), 1., atol=1e-6)
        values = profile[origins[origin]] * fractions[:, None]
        np.add.at(fixed, targets[origin, shift], values)
        delays.append(shift)
        weights.append(values[:, 2])
        offset += len(origin)
    original = np.column_stack((rigid, frame.rigid + frame.unknown)) + whole + np.column_stack((split, frame.split))
    checks = {'resources': max(0., float(fixed[:, :2].max() - certificate['servers'])),
              'conservation': float(np.abs(fixed.sum(axis=0) - original.sum(axis=0)).max()),
              'nonnegative': max(0., -float(fixed.min()))}
    stored = np.load(args.capacities / f'{scenario}-w12-s48.dispatch.npz')['resource_dispatch']
    checks['capacity_dispatch_reconstruction'] = float(np.abs(fixed[:, :2] - stored).max())
    if max(checks.values()) > 1e-4:
        raise RuntimeError(f'完整容量调度复核失败 {scenario} {checks}')
    extra, job_waiting = whole_waiting(args, scenario, frame)
    if extra is not None:
        delays.append(extra[0])
        weights.append(extra[1])
    metrics = delay_summary(np.concatenate(delays), np.concatenate(weights), 'flex_energy')
    peak = fixed[:, :2].max(axis=1)
    metrics.update(mean_cpu_fraction=float(fixed[:, 0].mean() / certificate['servers']),
                   mean_gpu_fraction=float(fixed[:, 1].mean() / certificate['servers']),
                   mean_bottleneck_fraction=float(peak.mean() / certificate['servers']),
                   hours_above_95pct_capacity=int(np.count_nonzero(peak > .95 * certificate['servers'])),
                   peak_task_power_mw=float(fixed[:, 2].max()))
    return fixed, checks, metrics, job_waiting


def energy_solve(frame, parameters, city, alpha, count, idle, task=None):
    data = frame.copy()
    data['base'] = count * idle
    if task is not None:
        data['rigid'] = task
        data[['unknown', 'whole', 'split']] = 0.
    data['it_mw'] = data[['base', 'rigid', 'unknown', 'whole', 'split']].sum(axis=1)
    summary, _ = solve(data, parameters, city, alpha=alpha, design='opt', whole_h=12, split_h=48)
    if summary['status'] != 'optimal' or summary['max_residual'] > 1e-5:
        raise RuntimeError(f'{city} 供能子问题未通过状态和残差核验 {summary}')
    summary['energy_annual_usd'] = summary['total_cost_usd'] * 8760 / len(data)
    return summary


def group(task):
    args, scenario = task
    started = time.monotonic()
    fleet = json.loads((args.resources / 'resource-manifest.json').read_text('utf-8'))[scenario]
    import fcntl

    stem = args.capacities / f'{scenario}-w12-s48'
    with stem.with_suffix('.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        certificate = json.loads(stem.with_suffix('.json').read_text('utf-8'))
        certificate_hash = sha256(stem.with_suffix('.json'))
        if certificate['gap_servers'] > 1 and not args.allow_open_capacity:
            raise RuntimeError(f'{scenario} 尚未达到一台以内的容量证明精度')
        reference = pd.read_parquet(args.city_inputs / f'frankfurt-{scenario}.parquet')
        dispatch, checks, metrics, job_waiting = feasible_dispatch(args, scenario, fleet, certificate, reference)
    parameters, config = load_config(args.project)
    price = config['cards']['server_price_usd'][scenario[0]]['value']
    unit = price * (crf(parameters['discount_rate'], parameters['compute_life_years']) + parameters['compute_fom_fraction'])
    count = certificate['servers']
    for city in args.city or CITIES:
        frame_path = args.city_inputs / f'{city}-{scenario}.parquet'
        frame = pd.read_parquet(frame_path)
        np.testing.assert_allclose(frame[['rigid', 'unknown', 'whole', 'split']],
                                   reference[['rigid', 'unknown', 'whole', 'split']], rtol=0, atol=1e-10)
        for target in args.target or ('economic', 'alpha100'):
            path = args.output / f'{city}-{scenario}-{target}.json'
            if path.is_file():
                existing = json.loads(path.read_text('utf-8'))
                if (existing.get('capacity_certificate_sha256') ==
                        certificate_hash and
                        existing['input_sha256'] == sha256(frame_path)):
                    continue
            alpha = None if target == 'economic' else 1.
            name = f'{city}-{scenario}-{target}.json'
            old = {stage: json.loads((args.original_results / stage / name).read_text('utf-8')) for stage in ('A', 'B', 'C')}
            if old['A']['input_sha256'] != sha256(frame_path):
                raise ValueError('城市基准输入版本不一致')
            c_upper = energy_solve(frame, parameters, city, alpha, count, fleet['idle_mw_per_unit'], dispatch[:, 2])
            c_lower = energy_solve(frame, parameters, city, alpha, count, fleet['idle_mw_per_unit'])
            b_lower = energy_solve(frame, parameters, city, alpha, fleet['baseline_units'], fleet['idle_mw_per_unit'])
            feasible_b = [old['A']]
            if (old['B']['status'] in ('optimal', 'feasible_with_gap') and
                    old['B']['servers'] == fleet['baseline_units'] and
                    old['B'].get('input_sha256') == sha256(frame_path) and
                    max(old['B'].get('audit', {'missing': 1.}).values()) <= 1e-4):
                feasible_b.append(old['B'])
            b = min(feasible_b, key=lambda item: item['energy_annual_usd'])
            upper_cost, lower_cost = c_upper['energy_annual_usd'], c_lower['energy_annual_usd']
            if lower_cost > upper_cost + 1.:
                raise ValueError('能源松弛下界超过完整任务调度的可行上界')
            chosen = {**c_upper, **metrics, 'status': 'feasible_energy_bracket', 'servers': count,
                      'gpus': count * fleet['gpus_per_unit'], 'energy_annual_usd': upper_cost,
                      'equipment_purchase_usd': count * price, 'equipment_annual_usd': count * unit,
                      'total_annual_usd': upper_cost + count * unit,
                      'total_annual_lower_bound_usd': lower_cost + count * unit,
                      'optimality_gap_annual_usd': upper_cost - lower_cost, 'audit': checks}
            relaxed = {**chosen, 'energy_annual_usd': lower_cost,
                       'total_annual_usd': lower_cost + count * unit, 'status': 'energy_only_relaxation'}
            b = {**b, 'total_annual_lower_bound_usd': b_lower['energy_annual_usd'] + fleet['baseline_units'] * unit}
            row = {'scenario': scenario, 'city': city, 'target': target, 'minimum_servers': count,
                   'minimum_lower_servers': certificate['lower_servers'], 'input_sha256': sha256(frame_path),
                   'method': '已核验完整容量调度的供能上界与忽略资源及连续约束的能源下界',
                   'metrics_version': 3, 'continuous_job_plan': chosen, 'hourly_capacity_fixed': relaxed,
                   'C_energy_lower_problem': c_lower, 'B_energy_lower_problem': b_lower,
                   'capacity_certificate_sha256': certificate_hash,
                   'capacity_gap_servers': certificate['gap_servers'],
                   'capacity_proof_complete': certificate['gap_servers'] <= 1,
                   'hourly_baseline_fleet': b, 'A': old['A'], 'original_B': old['B'], 'original_C': old['C'],
                   'whole_job_waiting_C': job_waiting,
                   'original_task_peak_mw': float(reference[['rigid', 'unknown', 'whole', 'split']].sum(axis=1).max()),
                   'equipment_purchase_avoided_usd': (fleet['baseline_units'] - count) * price,
                   'equipment_annual_saving_usd': (fleet['baseline_units'] - count) * unit,
                   'energy_annual_saving_vs_A_usd': old['A']['energy_annual_usd'] - upper_cost,
                   'net_annual_saving_vs_A_usd': old['A']['total_annual_usd'] - chosen['total_annual_usd'],
                   'net_annual_saving_vs_A_upper_usd': old['A']['total_annual_usd'] - lower_cost - count * unit,
                   'energy_annual_saving_vs_hourly_B_usd': b['energy_annual_usd'] - upper_cost,
                   'energy_annual_saving_vs_hourly_B_lower_usd': b_lower['energy_annual_usd'] - upper_cost,
                   'energy_annual_saving_vs_hourly_B_upper_usd': b['energy_annual_usd'] - lower_cost,
                   'net_annual_saving_vs_hourly_B_usd': b['total_annual_usd'] - chosen['total_annual_usd'],
                   'maximum_energy_lp_gap_usd': upper_cost - lower_cost,
                   'hourly_vs_continuous_plan_total_difference_pct': 100 * (upper_cost - lower_cost) / chosen['total_annual_usd'],
                   'group_seconds': time.monotonic() - started}
            row['net_saving_pct_of_A_fleet'] = 100 * row['net_annual_saving_vs_A_usd'] / old['A']['equipment_annual_usd']
            atomic_json(path, row)
            print(city, scenario, target, count, upper_cost, lower_cost, flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('project', 'reference', 'resources', 'whole-jobs', 'city-inputs', 'capacities', 'original-results', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--scenario', choices=SCENARIOS, action='append', required=True)
    parser.add_argument('--city', choices=CITIES, action='append')
    parser.add_argument('--target', choices=('economic', 'alpha100'), action='append')
    parser.add_argument('--workers', type=int, choices=range(1, 4), default=1)
    parser.add_argument('--allow-open-capacity', action='store_true',
                        help='明确报告固定可行台数的条件费用，保留尚未闭合的容量区间')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [(args, scenario) for scenario in args.scenario]
    if args.workers == 1:
        for task in tasks:
            group(task)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(group, tasks))


if __name__ == '__main__':
    main()
