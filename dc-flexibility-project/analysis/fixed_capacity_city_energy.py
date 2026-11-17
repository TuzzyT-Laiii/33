import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from flex_econ_copt import CITIES, SCENARIOS, crf, load_config
from joint_capacity_copt import Schedule, make_schedule, solve_with_start
from joint_capacity_inputs import atomic_json, sha256
from resource_capacity_lower_bound import resource_layers
from capacity_plan import apply_whole_plan


def fluid_schedule(hours, fixed, profiles):
    blocks, slots, columns, values = [], [], [], []
    options = count = 0
    for profile, deadline in profiles:
        origins = np.flatnonzero(np.any(profile > 1e-12, axis=1))
        shifts = np.arange(deadline + 1)
        targets = np.searchsorted(hours, hours[origins, None] + shifts)
        valid = (targets < len(hours)) & (hours[targets.clip(max=len(hours) - 1)] == hours[origins, None] + shifts)
        origin, offset = np.nonzero(valid)
        blocks.append(count + origin)
        slots.append(targets[origin, offset])
        columns.append(options + np.arange(len(origin)))
        values.append(profile[origins[origin]])
        options += len(origin)
        count += len(origins)
    return Schedule(hours, fixed, fixed + sum(profile for profile, _ in profiles),
                    np.concatenate(blocks), np.concatenate(slots), np.concatenate(columns), np.concatenate(values),
                    np.zeros(options, dtype=bool), options, count, 0, np.ones(count))


def whole_dispatch(args, scenario, fleet, source, frame, whole_resources, rigid_resources):
    plan_path = args.capacities / f'{scenario}-w12-s48.whole-plan.parquet'
    if plan_path.is_file():
        jobs = pd.read_parquet(args.whole_jobs / f'{scenario}.parquet')
        plan = pd.read_parquet(plan_path)
        whole = np.column_stack((whole_resources, frame.whole))
        return apply_whole_plan(whole, source.source_hour_index.to_numpy(), jobs, plan)
    stored = args.capacities / f'{scenario}-w12-s48.dispatch.npz'
    if stored.is_file():
        return np.load(stored)['whole_dispatch']
    if fleet['use'] in ('CLD', 'INF'):
        return np.column_stack((whole_resources, frame.whole))
    schedule = make_schedule(args, scenario, fleet, whole_h=12, split_h=48)
    assignment = np.load(args.capacities / f'{scenario}-w12-s48.npz')['assignment']
    fixed_rigid = np.column_stack((rigid_resources, frame.rigid + frame.unknown))
    whole = schedule.fixed - fixed_rigid
    entries = schedule.integer[schedule.columns]
    np.add.at(whole, schedule.slots[entries],
              schedule.values[entries] * assignment[schedule.columns[entries], None])
    if whole.min() < -1e-6:
        raise ValueError(f'{scenario} 的固定整体作业功率或资源量为负')
    return whole.clip(min=0)


def delay_summary(delay, weight, prefix):
    order = np.argsort(delay)
    cumulative = np.cumsum(weight[order])
    total = float(weight.sum())
    if total <= 0:
        return {}
    return {prefix + '_delayed_fraction': float(weight[delay > 0].sum() / total),
            prefix + '_mean_delay_h': float(weight @ delay / total),
            prefix + '_p95_delay_h': float(delay[order[np.searchsorted(cumulative, .95 * total)]])}


def whole_waiting(args, scenario, reference):
    path = args.capacities / f'{scenario}-w12-s48.whole-plan.parquet'
    if not path.is_file():
        return None, {}
    plan = pd.read_parquet(path)
    jobs = pd.read_parquet(args.whole_jobs / f'{scenario}.parquet', columns=['job', 'job_count', 'mw'])
    counts = jobs.groupby('job').job_count.first()
    np.testing.assert_array_equal(plan.groupby('job')['count'].sum().reindex(counts.index), counts)
    energy = plan['count'].to_numpy() * plan.job.map(jobs.groupby('job').mw.sum()).to_numpy()
    unmatched = float(reference.whole.sum() - energy.sum())
    if unmatched < -1e-5:
        raise ValueError('整体作业用电超过原始可推迟整体工作量')
    waiting = (np.r_[plan.shift_h.to_numpy(), 0], np.r_[energy, max(0., unmatched)])
    metrics = delay_summary(plan.shift_h.to_numpy(), plan['count'].to_numpy(), 'eligible_whole_jobs')
    metrics['eligible_whole_job_count'] = int(counts.sum())
    return waiting, metrics


def schedule_metrics(schedule, assignment, capacity, extra_waiting=None):
    work = schedule.fixed.copy()
    np.add.at(work, schedule.slots, schedule.values * assignment[schedule.columns, None])
    peak_resource = work[:, :2].max(axis=1)
    entries = np.flatnonzero(schedule.values[:, 2] > 0)
    first_option = np.unique(schedule.blocks, return_index=True)[1]
    first_entry = np.searchsorted(schedule.columns, first_option)
    origin = schedule.hours[schedule.slots[first_entry]]
    delay = schedule.hours[schedule.slots[entries]] - origin[schedule.blocks[schedule.columns[entries]]]
    weight = schedule.values[entries, 2] * assignment[schedule.columns[entries]]
    if extra_waiting is not None:
        delay = np.r_[delay, extra_waiting[0]]
        weight = np.r_[weight, extra_waiting[1]]
    waiting = delay_summary(delay, weight, 'flex_energy')
    return {**waiting, 'mean_cpu_fraction': float(work[:, 0].mean() / capacity),
            'mean_gpu_fraction': float(work[:, 1].mean() / capacity),
            'mean_bottleneck_fraction': float(peak_resource.mean() / capacity),
            'hours_above_95pct_capacity': int(np.count_nonzero(peak_resource > .95 * capacity)),
            'peak_task_power_mw': float(work[:, 2].max())}


def compute_group(task):
    args, scenario = task
    started = time.monotonic()
    manifest = json.loads((args.resources / 'resource-manifest.json').read_text('utf-8'))
    fleet = manifest[scenario]
    capacity_result = json.loads((args.capacities / f'{scenario}-w12-s48.json').read_text('utf-8'))
    if capacity_result['gap_servers'] > 1:
        raise RuntimeError(f'{scenario} 最少台数尚未达到 1 台以内的证明精度')
    minimum = capacity_result['servers']
    source = pd.read_parquet(args.reference / f'{scenario}-10MW.parquet')
    reference = pd.read_parquet(args.city_inputs / f'frankfurt-{scenario}.parquet')
    hours, rigid_resources, whole_resources, split_resources = resource_layers(args, scenario, fleet)
    split = np.column_stack((split_resources, reference.split))
    original_whole = np.column_stack((whole_resources, reference.whole))
    rigid = np.column_stack((rigid_resources, reference.rigid + reference.unknown))
    fixed_whole = whole_dispatch(args, scenario, fleet, source, reference, whole_resources, rigid_resources)
    fixed_waiting, job_waiting = whole_waiting(args, scenario, reference)
    plan_path = args.capacities / f'{scenario}-w12-s48.whole-plan.parquet'
    plan_hash = sha256(plan_path) if plan_path.is_file() else 'hourly-flexible'
    coupled_hourly = fluid_schedule(hours, rigid, [(original_whole, 12), (split, 48)])
    feasible = (coupled_hourly if fleet['use'] in ('CLD', 'INF') else
                fluid_schedule(hours, rigid + fixed_whole, [(split, 48)]))
    parameters, config = load_config(args.project)
    factors = parameters['emission_factor']
    price = config['cards']['server_price_usd'][scenario[0]]['value']
    equipment_saving = (fleet['baseline_units'] - minimum) * price * (
        crf(parameters['discount_rate'], parameters['compute_life_years']) + parameters['compute_fom_fraction'])
    allowed_energy_gap = max(5000., equipment_saving * .0025)
    reduced_fleet = {**fleet, 'baseline_units': minimum}
    for city in args.city or CITIES:
        city_path = args.city_inputs / f'{city}-{scenario}.parquet'
        frame = pd.read_parquet(city_path)
        np.testing.assert_allclose(frame[['rigid', 'unknown', 'whole', 'split']],
                                   reference[['rigid', 'unknown', 'whole', 'split']], rtol=0, atol=1e-10)
        input_hash = sha256(city_path)
        parameters['emission_factor'] = factors[city]
        for alpha in ([None if target == 'economic' else 1. for target in args.target] if args.target else (None, 1.)):
            token = 'economic' if alpha is None else 'alpha100'
            destination = args.output / f'{city}-{scenario}-{token}.json'
            if destination.is_file():
                saved = json.loads(destination.read_text('utf-8'))
                if (saved['minimum_servers'] == minimum and saved.get('metrics_version') == 2 and
                        saved.get('whole_plan_sha256') == plan_hash and saved.get('input_sha256') == input_hash):
                    continue
            row = {'scenario': scenario, 'city': city, 'target': token, 'minimum_servers': minimum,
                   'minimum_lower_servers': capacity_result['lower_servers'], 'input_sha256': input_hash,
                   'metrics_version': 2, 'whole_plan_sha256': plan_hash,
                   'whole_job_waiting_C': job_waiting,
                   'original_task_peak_mw': float(reference[['rigid', 'unknown', 'whole', 'split']].sum(axis=1).max())}
            for name, schedule, selected_fleet in (
                    ('hourly_baseline_fleet', coupled_hourly, fleet),
                    ('hourly_capacity_fixed', coupled_hourly, reduced_fleet),
                    ('continuous_job_plan', feasible, reduced_fleet)):
                if name == 'continuous_job_plan' and feasible is coupled_hourly:
                    row[name] = dict(row['hourly_capacity_fixed'])
                    continue
                result, x = solve_with_start(frame, schedule, selected_fleet, parameters, price, alpha,
                                             'B', args.time_limit, None, relax_whole=True, lp_solver=args.solver)
                if (result['status'] not in ('optimal', 'feasible_with_gap') or
                        result['optimality_gap_annual_usd'] > allowed_energy_gap):
                    raise RuntimeError(f'{city} {scenario} {token} {name} 能源 LP 未满足状态或间隙要求 {result}')
                first = 6 + 6 * len(frame)
                metrics = schedule_metrics(schedule, x[first:], selected_fleet['baseline_units'],
                    fixed_waiting if name == 'continuous_job_plan' and feasible is not coupled_hourly else None)
                row[name] = {**result, **metrics}
            baseline = json.loads((args.original_results / 'A' / f'{city}-{scenario}-{token}.json').read_text('utf-8'))
            if baseline['input_sha256'] != input_hash:
                raise ValueError('原 A 组与本次能源复核的城市输入不一致')
            old_c = json.loads((args.original_results / 'C' / f'{city}-{scenario}-{token}.json').read_text('utf-8'))
            old_b = json.loads((args.original_results / 'B' / f'{city}-{scenario}-{token}.json').read_text('utf-8'))
            row['A'] = baseline
            row['original_C'] = old_c
            row['original_B'] = old_b
            chosen = row['continuous_job_plan']
            row['equipment_purchase_avoided_usd'] = baseline['equipment_purchase_usd'] - chosen['equipment_purchase_usd']
            row['equipment_annual_saving_usd'] = baseline['equipment_annual_usd'] - chosen['equipment_annual_usd']
            row['energy_annual_saving_vs_A_usd'] = baseline['energy_annual_usd'] - chosen['energy_annual_usd']
            row['net_annual_saving_vs_A_usd'] = baseline['total_annual_usd'] - chosen['total_annual_usd']
            row['net_saving_pct_of_A_fleet'] = 100 * row['net_annual_saving_vs_A_usd'] / baseline['equipment_annual_usd']
            b = row['hourly_baseline_fleet']
            row['energy_annual_saving_vs_hourly_B_usd'] = b['energy_annual_usd'] - chosen['energy_annual_usd']
            row['net_annual_saving_vs_hourly_B_usd'] = b['total_annual_usd'] - chosen['total_annual_usd']
            row['net_annual_saving_vs_A_upper_usd'] = baseline['total_annual_usd'] - chosen['total_annual_lower_bound_usd']
            row['energy_annual_saving_vs_hourly_B_lower_usd'] = (
                b['total_annual_lower_bound_usd'] - b['equipment_annual_usd'] - chosen['energy_annual_usd'])
            row['energy_annual_saving_vs_hourly_B_upper_usd'] = (
                b['energy_annual_usd'] - chosen['total_annual_lower_bound_usd'] + chosen['equipment_annual_usd'])
            row['maximum_energy_lp_gap_usd'] = max(row[name]['optimality_gap_annual_usd'] for name in (
                'hourly_baseline_fleet', 'hourly_capacity_fixed', 'continuous_job_plan'))
            row['hourly_vs_continuous_plan_total_difference_pct'] = (
                100 * (chosen['total_annual_usd'] - row['hourly_capacity_fixed']['total_annual_usd']) /
                chosen['total_annual_usd'])
            row['group_seconds'] = time.monotonic() - started
            atomic_json(destination, row)
            print(city, scenario, token, minimum, row['net_annual_saving_vs_A_usd'], flush=True)


def group(task):
    import fcntl

    args, scenario = task
    with (args.output / f'.{scenario}.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return compute_group(task)


def main():
    parser = argparse.ArgumentParser()
    for name in ('project', 'reference', 'resources', 'whole-jobs', 'city-inputs', 'capacities',
                 'original-results', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--scenario', choices=SCENARIOS, action='append')
    parser.add_argument('--city', choices=CITIES, action='append')
    parser.add_argument('--target', choices=('economic', 'alpha100'), action='append')
    parser.add_argument('--solver', choices=('copt', 'highs'), default='highs')
    parser.add_argument('--workers', type=int, default=2, choices=range(1, 5))
    parser.add_argument('--time-limit', type=int, default=300)
    args = parser.parse_args()
    args.hourly_whole = True
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [(args, scenario) for scenario in (args.scenario or SCENARIOS)]
    if args.workers == 1:
        for task in tasks:
            group(task)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(group, tasks))


if __name__ == '__main__':
    main()
