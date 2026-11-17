import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from joint_capacity_copt import make_resource_schedule, whole_choices
from joint_capacity_inputs import atomic_json, sha256
from minimum_server_capacity import DEADLINES, audit, capacity_matrix, resource_dispatch, solve_capacity
from highs_capacity_solver import capacity_lp, solve_capacity_highs
from capacity_plan import derive_family
from persistent_capacity_lp import PersistentCapacityLP


class WholePricing:
    def __init__(self, jobs, hours, delay):
        self.jobs = jobs.sort_values(['job', 'source_hour_index'])
        self.codes, self.labels = pd.factorize(self.jobs.job, sort=False)
        self.starts = np.r_[0, np.flatnonzero(np.diff(self.codes)) + 1]
        self.ends = np.r_[self.starts[1:] - 1, len(self.jobs) - 1]
        self.widths = self.ends - self.starts + 1
        raw_hours = self.jobs.source_hour_index.to_numpy(np.int64)
        self.hours = hours
        self.release = raw_hours[self.starts]
        self.finish = raw_hours[self.ends]
        self.valid = whole_choices(raw_hours, self.codes, hours, delay)
        self.movable = self.valid.sum(axis=1) > 1
        self.counts = self.jobs.job_count.to_numpy()[self.starts]
        values = self.jobs[['cpu_units', 'gpu_units']].to_numpy()
        self.leading = values[self.starts]
        self.trailing = values[self.ends] * (self.widths > 1)[:, None]
        self.interior = values[(self.starts + 1).clip(max=len(self.jobs) - 1)] * (self.widths > 2)[:, None]
        self.checked_prices = None
        middle = (np.arange(len(self.jobs)) > self.starts[self.codes]) & (np.arange(len(self.jobs)) < self.ends[self.codes])
        np.testing.assert_allclose(values[middle], self.interior[self.codes[middle]], rtol=0, atol=1e-10)

    def price(self, dual, selections):
        prefix = np.vstack((np.zeros((1, 2)), np.cumsum(dual, axis=0)))
        costs = np.full(self.valid.shape, np.inf)
        for shift in range(self.valid.shape[1]):
            left = np.searchsorted(self.hours, self.release + shift).clip(max=len(self.hours) - 1)
            right = np.searchsorted(self.hours, self.finish + shift).clip(max=len(self.hours) - 1)
            costs[:, shift] = -(dual[left] * self.leading + dual[right] * self.trailing +
                               (prefix[right] - prefix[(left + 1).clip(max=len(self.hours))]) * self.interior).sum(axis=1)
        costs[~self.valid] = np.inf
        if self.checked_prices is None:
            strength = np.where(self.valid, np.abs(costs), 0).max(axis=1)
            samples = np.unique(np.r_[np.argsort(strength)[-24:], np.argsort(self.widths)[-16:],
                                      np.flatnonzero(self.widths == 1)[:8], np.arange(min(8, len(self.labels)))])
            error, checked = 0., 0
            values = self.jobs[['cpu_units', 'gpu_units']].to_numpy()
            source_hours = self.jobs.source_hour_index.to_numpy()
            for group in samples:
                rows = slice(self.starts[group], self.ends[group] + 1)
                for shift in np.flatnonzero(self.valid[group]):
                    target = np.searchsorted(self.hours, source_hours[rows] + shift)
                    direct = -float((values[rows] * dual[target]).sum())
                    error = max(error, abs(direct - costs[group, shift]))
                    checked += 1
            if error > 1e-7:
                raise ValueError(f'完整作业价格与逐小时直接计算不一致 {error}')
            self.checked_prices = {'real_groups': len(samples), 'choices_checked': checked, 'maximum_error': error}
        best = costs.argmin(axis=1)
        minimum = costs[np.arange(len(best)), best]
        selected_minimum = np.where(selections, costs, np.inf).min(axis=1)
        improvements = selected_minimum - minimum
        return minimum, best, improvements


def restricted_lp(schedule, lower, baseline, limit, solver='copt'):
    if solver == 'highs':
        result, _ = capacity_lp(schedule, baseline, lower, limit)
        dual = np.minimum(result.ineqlin.marginals.reshape(2, len(schedule.hours)).T, 0)
        return result.x, dual
    import coptpy as cp

    n = len(schedule.hours)
    matrix = capacity_matrix(schedule)
    objective = np.r_[1., np.zeros(schedule.option_count)]
    lb = np.r_[lower, np.zeros(schedule.option_count)]
    ub = np.r_[baseline, np.full(schedule.option_count, cp.COPT.INFINITY)]
    lhs = np.r_[np.full(2 * n, -cp.COPT.INFINITY), schedule.block_sizes]
    rhs = np.r_[-schedule.fixed[:, :2].T.ravel(), schedule.block_sizes]
    env = cp.Envr()
    model = env.createModel('restricted_capacity_patterns')
    for key, value in {'Logging': 0, 'Threads': 1, 'LpMethod': 1, 'Crossover': 1,
                       'FeasTol': 1e-8, 'DualTol': 1e-8, 'TimeLimit': limit}.items():
        model.setParam(key, value)
    model.loadMatrix(objective, matrix, lhs, rhs, lb, ub)
    model.solve()
    if model.status != cp.COPT.OPTIMAL:
        raise RuntimeError(f'受限模式 LP 未达到最优 {model.status}')
    x = np.asarray(model.getValues())
    dual = np.minimum(np.asarray(model.getDuals())[:2 * n].reshape(2, n).T, 0)
    audit(schedule, x[1:], x[0], False)
    model.clear()
    env.close()
    return x, dual


def full_dual_bound(schedule, dual, whole_prices, pricing, resource_lower, baseline, fixed_override=None):
    fixed = schedule.fixed[:, :2] if fixed_override is None else fixed_override
    value = -float((fixed * dual).sum())
    value += float(whole_prices[pricing.movable] @ pricing.counts[pricing.movable])
    fluid_columns = np.flatnonzero(~schedule.integer)
    if len(fluid_columns):
        entries = np.flatnonzero(~schedule.integer[schedule.columns])
        prices = -(schedule.values[entries, :2] * dual[schedule.slots[entries]]).sum(axis=1)
        option_prices = np.zeros(schedule.option_count)
        np.add.at(option_prices, schedule.columns[entries], prices)
        starts = np.r_[0, np.flatnonzero(np.diff(schedule.blocks[fluid_columns])) + 1]
        minimum = np.minimum.reduceat(option_prices[fluid_columns], starts)
        value += float(minimum @ schedule.block_sizes[np.unique(schedule.blocks[fluid_columns])])
    coefficient = 1 + float(dual.sum())
    value += coefficient * (resource_lower if coefficient >= 0 else baseline)
    return value


def save_plan(args, scenario, deadline, pricing, choices, schedule, assignment, result):
    whole_h, split_h = DEADLINES[deadline]
    name = args.output / f'{scenario}-w{whole_h}-s{split_h}'
    eligible = choices & pricing.valid & pricing.movable[:, None]
    groups, shifts = np.nonzero(eligible)
    values = assignment[schedule.integer]
    if len(values) != len(groups):
        raise ValueError('受限模式解与作业选择顺序不一致')
    selected = values > .5
    moved = pd.DataFrame({'job': pricing.labels[groups[selected]], 'shift_h': shifts[selected],
                          'count': np.rint(values[selected]).astype(int)})
    unchanged = pd.DataFrame({'job': pricing.labels[~pricing.movable], 'shift_h': 0,
                              'count': pricing.counts[~pricing.movable]})
    plan = pd.concat([moved, unchanged], ignore_index=True)
    expected = pricing.jobs.groupby('job').job_count.first()
    np.testing.assert_array_equal(plan.groupby('job')['count'].sum().reindex(expected.index), expected)
    plan.to_parquet(name.with_suffix('.whole-plan.parquet'), index=False)
    fleet = json.loads((args.resources / 'resource-manifest.json').read_text('utf-8'))[scenario]
    np.savez_compressed(name.with_suffix('.dispatch.npz'), resource_dispatch=resource_dispatch(schedule, assignment))
    np.savez_compressed(name.with_suffix('.npz'), assignment=assignment, pattern_choices=choices)
    atomic_json(name.with_suffix('.json'), result)


def run_case(args, scenario, deadline):
    started = time.monotonic()
    whole_h, split_h = DEADLINES[deadline]
    fleet = json.loads((args.resources / 'resource-manifest.json').read_text('utf-8'))[scenario]
    source = pd.read_parquet(args.reference / f'{scenario}-10MW.parquet')
    jobs = pd.read_parquet(args.whole_jobs / f'{scenario}.parquet')
    pricing = WholePricing(jobs, source.source_hour_index.to_numpy(), whole_h)
    original_pricing, original_fixed = pricing, None
    if args.fine_whole_jobs is not None:
        fine_jobs = pd.read_parquet(args.fine_whole_jobs / f'{scenario}.parquet')
        original_pricing = WholePricing(fine_jobs, pricing.hours, whole_h)
        original_args = copy.copy(args)
        original_args.whole_jobs = args.fine_whole_jobs
        original_args.pattern_choices = np.zeros(original_pricing.valid.shape, dtype=bool)
        original_args.pattern_choices[:, 0] = True
        original_fixed = make_resource_schedule(original_args, scenario, fleet, whole_h=whole_h, split_h=split_h).fixed
    resource = json.loads((args.lower_bounds / f'{scenario}-w{whole_h}-s{split_h}.json').read_text('utf-8'))
    best_lower = resource_lower = resource['resource_lower_servers']
    choices = np.zeros(pricing.valid.shape, dtype=bool)
    choices[:, 0] = True
    original = source[['u_cpu_proxy', 'allocated_gpu_fraction']].to_numpy() * fleet['reference_units']
    critical_prefix = np.r_[0, np.cumsum(np.any(original > resource_lower, axis=1))]
    first_slot = np.searchsorted(pricing.hours, pricing.release)
    last_slot = np.searchsorted(pricing.hours, pricing.finish)
    critical_jobs = critical_prefix[last_slot + 1] > critical_prefix[first_slot]
    for shift in (whole_h // 2, whole_h):
        choices[:, shift] = pricing.valid[:, shift] & critical_jobs
    history = []
    name = args.output / f'{scenario}-w{whole_h}-s{split_h}'
    checkpoint = name.with_suffix('.checkpoint.npz')
    fingerprint = ''.join(sha256(path) for path in (args.resources / 'resource-manifest.json',
                           args.whole_jobs / f'{scenario}.parquet', args.reference / f'{scenario}-10MW.parquet'))
    first_iteration, previous_seconds = 0, 0.
    if checkpoint.is_file():
        saved = np.load(checkpoint)
        if str(saved['fingerprint']) != fingerprint:
            raise ValueError('容量检查点与当前输入不一致')
        choices = saved['choices']
        best_lower = float(saved['best_lower'])
        first_iteration = int(saved['iteration'])
        previous_seconds = float(saved['seconds'])
        history = json.loads(name.with_suffix('.progress.json').read_text('utf-8'))['history'][:first_iteration]
    engine = None
    for iteration in range(first_iteration, args.iterations):
        repair_failed = False
        args.pattern_choices = choices
        schedule = make_resource_schedule(args, scenario, fleet, whole_h=whole_h, split_h=split_h)
        if args.solver == 'highs-persistent':
            if engine is None:
                engine = PersistentCapacityLP(schedule, choices, pricing, resource_lower, fleet['baseline_units'])
            fractional, dual = engine.solve(schedule, choices, pricing, min(60, args.lp_limit))
        else:
            fractional, dual = restricted_lp(schedule, resource_lower, fleet['baseline_units'], args.lp_limit, args.solver)
        prices, shifts, improvements = pricing.price(dual, choices)
        original_prices = prices if original_pricing is pricing else original_pricing.price(dual, original_pricing.valid)[0]
        bound = full_dual_bound(schedule, dual, original_prices, original_pricing, resource_lower,
                                fleet['baseline_units'], fixed_override=original_fixed)
        if bound > fractional[0] + 1e-4:
            raise RuntimeError(f'全模式对偶下界超过受限模式可行值 {bound} {fractional[0]}')
        best_lower = max(best_lower, bound)
        record = {'iteration': iteration, 'global_lower': best_lower, 'restricted_lp': float(fractional[0]),
                  'options': schedule.option_count, 'seconds': previous_seconds + time.monotonic() - started}
        history.append(record)
        if iteration == 0:
            np.savez_compressed(name.with_suffix('.first-dual.npz'), dual=dual)
        atomic_json(name.with_suffix('.progress.json'), {'scenario': scenario, 'history': history})
        print(scenario, whole_h, split_h, iteration, best_lower, fractional[0], schedule.option_count, flush=True)
        # 对偶下界已检查全部启动时间。停止条件直接采用整机台数所需精度。
        if int(np.ceil(fractional[0] - 1e-5)) <= int(np.ceil(best_lower - 1e-5)) or iteration == args.iterations - 1:
            if args.relaxation_only:
                np.savez_compressed(name.with_suffix('.npz'), assignment=fractional[1:], pattern_choices=choices)
                atomic_json(name.with_suffix('.json'), {'scenario': scenario, 'whole_h': whole_h, 'split_h': split_h,
                    'status': 'relaxation_complete', 'relaxation_peak': float(fractional[0]),
                    'continuous_lower_servers': best_lower, 'pricing_history': history,
                    'original_whole_groups': len(original_pricing.labels), 'lp_cohorts': len(pricing.labels),
                    'direct_pricing_check': original_pricing.checked_prices,
                    'seconds': previous_seconds + time.monotonic() - started})
                return
            if args.solver in ('highs', 'highs-persistent'):
                result = solve_capacity_highs(schedule, fleet['baseline_units'], name.with_suffix('.repair'),
                                              args.time_limit, full_mip=False, fractional=fractional,
                                              certified_lower=best_lower)
            else:
                result = solve_capacity(schedule, fleet['baseline_units'], name.with_suffix('.repair'),
                                        args.time_limit, full_mip=False, lp_method=1)
            assignment = np.load(name.with_suffix('.npz'))['assignment']
            lower_integer = int(np.ceil(best_lower - 1e-5))
            result.update(scenario=scenario, whole_h=whole_h, split_h=split_h,
                          baseline_servers=fleet['baseline_units'], lower_servers=lower_integer,
                          gpus_per_server=fleet['gpus_per_unit'],
                          continuous_lower_servers=best_lower, gap_servers=result['servers'] - lower_integer,
                          method='受限模式列生成与全部模式对偶校验、整数修复',
                          source_manifest_sha256=sha256(args.resources / 'resource-manifest.json'),
                          whole_jobs_sha256=sha256(args.whole_jobs / f'{scenario}.parquet'),
                          seconds=previous_seconds + time.monotonic() - started, pricing_history=history,
                          direct_pricing_check=pricing.checked_prices)
            result['status'] = 'certified_within_one' if result['gap_servers'] <= 1 else 'feasible_with_gap'
            if result['gap_servers'] <= 1 or iteration == args.iterations - 1:
                save_plan(args, scenario, deadline, pricing, choices, schedule, assignment, result)
                print(scenario, whole_h, split_h, 'complete', lower_integer, result['servers'], flush=True)
                return
            repair_failed = True
        active_groups, active_shifts = np.nonzero(choices & pricing.valid & pricing.movable[:, None])
        active = fractional[1:][schedule.integer] > 1e-8
        following = np.zeros_like(choices)
        following[:, 0] = True
        following[active_groups[active], active_shifts[active]] = True
        following[np.arange(len(shifts)), shifts] = True
        # 保留相邻候选以便整数修复，同时避免累积未使用的整段模式。
        for offset in (-1, 1):
            neighbor = (shifts + offset).clip(0, whole_h)
            moving = np.flatnonzero(pricing.movable & (shifts > 0))
            following[moving, neighbor[moving]] |= pricing.valid[moving, neighbor[moving]]
        if repair_failed:
            fractional_group = np.unique(active_groups[np.abs(fractional[1:][schedule.integer] -
                                                   np.rint(fractional[1:][schedule.integer])) > 1e-6])
            following[fractional_group] = pricing.valid[fractional_group]
        if args.solver == 'highs-persistent':
            following |= choices
        if np.array_equal(following, choices) and args.solver != 'highs-persistent':
            raise RuntimeError('列生成未新增候选且尚未形成所需容量证书')
        choices = following
        pending = checkpoint.with_suffix('.partial.npz')
        np.savez_compressed(pending, choices=choices, best_lower=best_lower, iteration=iteration + 1,
                            seconds=previous_seconds + time.monotonic() - started, fingerprint=fingerprint)
        pending.replace(checkpoint)


def main():
    parser = argparse.ArgumentParser()
    for name in ('reference', 'resources', 'whole-jobs', 'lower-bounds', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--city-inputs', type=Path)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--deadline', type=int, choices=(0, 1, 2), action='append')
    parser.add_argument('--derive-family', action='store_true')
    parser.add_argument('--iterations', type=int, default=80)
    parser.add_argument('--lp-limit', type=int, default=600)
    parser.add_argument('--time-limit', type=int, default=300)
    parser.add_argument('--solver', choices=('copt', 'highs', 'highs-persistent'), default='highs-persistent')
    parser.add_argument('--fine-whole-jobs', type=Path)
    parser.add_argument('--relaxation-only', action='store_true')
    args = parser.parse_args()
    args.hourly_whole = True
    args.output.mkdir(parents=True, exist_ok=True)
    for deadline in args.deadline or (0, 1, 2):
        whole_h, split_h = DEADLINES[deadline]
        path = args.output / f'{args.scenario}-w{whole_h}-s{split_h}.json'
        completed = json.loads(path.read_text('utf-8')) if path.is_file() else {}
        if (not completed or (args.relaxation_only and completed.get('status') != 'relaxation_complete') or
                (not args.relaxation_only and completed.get('gap_servers', 99) > 1)):
            run_case(args, args.scenario, deadline)
        if args.derive_family:
            derive_family(args, args.scenario, deadline)


if __name__ == '__main__':
    main()
