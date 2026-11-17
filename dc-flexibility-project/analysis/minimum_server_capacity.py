import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from flex_econ_copt import SCENARIOS
from joint_capacity_copt import make_resource_schedule
from joint_capacity_inputs import atomic_json, sha256


DEADLINES = ((12, 48), (24, 96), (48, 168))


def capacity_matrix(schedule):
    n = len(schedule.hours)
    resources = schedule.values[:, :2]
    rows = np.r_[schedule.slots, n + schedule.slots, 2 * n + schedule.blocks,
                 np.arange(2 * n)]
    columns = np.r_[1 + schedule.columns, 1 + schedule.columns,
                    1 + np.arange(schedule.option_count), np.zeros(2 * n, dtype=int)]
    values = np.r_[resources[:, 0], resources[:, 1], np.ones(schedule.option_count),
                   -np.ones(2 * n)]
    matrix = sparse.csc_matrix((values, (rows, columns)), shape=(2 * n + schedule.block_count,
                                                                1 + schedule.option_count))
    matrix.eliminate_zeros()
    return matrix


def resource_dispatch(schedule, assignment):
    dispatch = schedule.fixed[:, :2].copy()
    np.add.at(dispatch, schedule.slots, schedule.values[:, :2] * assignment[schedule.columns, None])
    return dispatch


def original_assignment(schedule):
    result = np.zeros(schedule.option_count)
    result[np.unique(schedule.blocks, return_index=True)[1]] = schedule.block_sizes
    return result


def compulsory_resources(schedule):
    mandatory = schedule.fixed[:, :2].copy()
    excess = schedule.values[:, :2].copy()
    offsets = np.r_[0, np.cumsum(np.bincount(schedule.columns, minlength=schedule.option_count))]
    first = np.r_[0, np.flatnonzero(np.diff(schedule.blocks)) + 1]
    stop = np.r_[first[1:], schedule.option_count]
    for a, b in zip(first, stop):
        if not schedule.integer[a]:
            continue
        options = np.arange(a, b)
        start_slots = schedule.slots[offsets[options]]
        end_slots = schedule.slots[offsets[options + 1] - 1]
        left, right = start_slots.max(), end_slots.min()
        if left > right:
            continue
        common = np.arange(left, right + 1)
        entries = offsets[options, None] + common - start_slots[:, None]
        np.testing.assert_array_equal(schedule.slots[entries], np.broadcast_to(common, entries.shape))
        minimum = schedule.values[entries, :2].min(axis=0)
        mandatory[common] += schedule.block_sizes[schedule.blocks[a]] * minimum
        excess[entries] -= minimum
    peaks = np.zeros(schedule.option_count)
    whole = schedule.integer[schedule.columns]
    np.maximum.at(peaks, schedule.columns[whole], (mandatory[schedule.slots[whole]] + excess[whole]).max(axis=1))
    lower = float(mandatory.max())
    for a, b in zip(first, stop):
        if schedule.integer[a]:
            lower = max(lower, float(peaks[a:b].min()))
    return mandatory, excess, lower


def audit(schedule, assignment, servers, integral):
    dispatch = resource_dispatch(schedule, assignment)
    service = np.bincount(schedule.blocks, weights=assignment, minlength=schedule.block_count)
    binary = assignment[schedule.integer]
    errors = {
        'capacity_servers': max(0., float(dispatch.max() - servers)),
        'batch_conservation': float(np.max(np.abs(service - schedule.block_sizes), initial=0)),
        'work_conservation': float(np.max(np.abs(dispatch.sum(axis=0) - schedule.original[:, :2].sum(axis=0)))),
        'nonnegative': max(0., -float(assignment.min(initial=0))),
        'whole_integrality': float(np.max(np.abs(binary - np.rint(binary)), initial=0)) if integral else 0.,
    }
    if max(errors.values()) > 1e-4:
        raise RuntimeError(f'最少台数调度核验失败 {errors}')
    return errors


def round_whole(schedule, fractional):
    assignment = fractional.copy()
    fractional_block = np.zeros(schedule.block_count)
    np.maximum.at(fractional_block, schedule.blocks[schedule.integer],
                  np.abs(assignment[schedule.integer] - np.rint(assignment[schedule.integer])))
    already_integer = schedule.integer & (fractional_block[schedule.blocks] < 1e-7)
    assignment[already_integer] = np.rint(assignment[already_integer])
    dispatch = resource_dispatch(schedule, assignment)
    offsets = np.r_[0, np.cumsum(np.bincount(schedule.columns, minlength=schedule.option_count))]
    order = np.argsort(schedule.columns, kind='stable')
    first = np.r_[0, np.flatnonzero(np.diff(schedule.blocks)) + 1]
    stop = np.r_[first[1:], schedule.option_count]
    groups = [(a, b) for a, b in zip(first, stop) if schedule.integer[a] and
              fractional_block[schedule.blocks[a]] >= 1e-7]
    # 优先安排资源占用较大的作业组，保留每个组原有的连续执行区间。
    sizes = []
    for a, b in groups:
        entries = order[offsets[a]:offsets[a + 1]]
        sizes.append(float(schedule.values[entries, :2].max()))
    for k in np.argsort(sizes)[::-1]:
        a, b = groups[k]
        entries = order[offsets[a]:offsets[b]]
        np.add.at(dispatch, schedule.slots[entries],
                  -schedule.values[entries, :2] * assignment[schedule.columns[entries], None])
        floors = np.floor(assignment[a:b] + 1e-8)
        amount = int(schedule.block_sizes[schedule.blocks[a]])
        remaining = amount - int(floors.sum())
        if remaining < 0:
            raise ValueError('同类作业取整后超过其作业数量')
        assignment[a:b] = floors
        np.add.at(dispatch, schedule.slots[entries],
                  schedule.values[entries, :2] * assignment[schedule.columns[entries], None])
        for _ in range(remaining):
            base_peak = float(dispatch.max())
            chosen = a
            best = (math.inf, math.inf)
            for option in range(a, b):
                positions = order[offsets[option]:offsets[option + 1]]
                slots = schedule.slots[positions]
                values = schedule.values[positions, :2]
                proposed = dispatch[slots] + values
                score = (max(base_peak, float(proposed.max())),
                         float(np.sum(proposed * proposed - dispatch[slots] * dispatch[slots])))
                if score < best:
                    best, chosen = score, option
            assignment[chosen] += 1
            positions = order[offsets[chosen]:offsets[chosen + 1]]
            np.add.at(dispatch, schedule.slots[positions], schedule.values[positions, :2])
    return assignment, int(np.ceil(dispatch.max() - 1e-7))


def rebalance_whole(schedule, assignment, target, passes=4):
    assignment = assignment.copy()
    dispatch = resource_dispatch(schedule, assignment)
    sizes = np.bincount(schedule.columns, minlength=schedule.option_count)
    offsets = np.r_[0, np.cumsum(sizes)]
    first = np.r_[0, np.flatnonzero(np.diff(schedule.blocks)) + 1]
    stop = np.r_[first[1:], schedule.option_count]
    for _ in range(passes):
        overloaded = np.any(dispatch > target + 1e-7, axis=1)
        if not overloaded.any():
            break
        candidates = np.unique(schedule.columns[overloaded[schedule.slots] &
                              schedule.integer[schedule.columns] & (assignment[schedule.columns] > .5)])
        changes = 0
        for old in candidates:
            if assignment[old] < .5:
                continue
            a, b = offsets[old:old + 2]
            original_slots = schedule.slots[a:b]
            if not np.any(dispatch[original_slots] > target + 1e-7):
                continue
            profile = schedule.values[a:b, :2]
            block = schedule.blocks[old]
            choices = range(first[block], stop[block])
            best = None
            for new in choices:
                if old == new:
                    continue
                c, d = offsets[new:new + 2]
                slots, inverse = np.unique(np.r_[original_slots, schedule.slots[c:d]], return_inverse=True)
                delta = np.zeros((len(slots), 2))
                np.add.at(delta, inverse[:b - a], -profile)
                np.add.at(delta, inverse[b - a:], schedule.values[c:d, :2])
                positive, negative = delta > 1e-12, delta < -1e-12
                if not positive.any() or not negative.any():
                    continue
                room = (target - dispatch[slots])[positive] / delta[positive]
                available = min(int(round(assignment[old])), int(np.floor(room.min() + 1e-8)))
                if available < 1:
                    continue
                needed = max(1, int(np.ceil(np.maximum(dispatch[slots] - target, 0)[negative].max() /
                                            max((-delta[negative]).max(), 1e-12))))
                amount = min(available, needed)
                proposed = dispatch[slots] + amount * delta
                improvement = float(np.maximum(dispatch[slots] - target, 0).sum() -
                                    np.maximum(proposed - target, 0).sum())
                score = (improvement, -float((proposed * proposed).sum()))
                if improvement > 1e-8 and (best is None or score > best[0]):
                    best = (score, new, amount, slots, delta)
            if best is not None:
                _, new, amount, slots, delta = best
                assignment[old] -= amount
                assignment[new] += amount
                dispatch[slots] += amount * delta
                changes += 1
        if not changes:
            break
    return assignment, int(np.ceil(dispatch.max() - 1e-6))


def solve_capacity(schedule, baseline, output, time_limit, *, lower_only=False, full_mip=True, lp_method=2):
    import coptpy as cp

    started = time.monotonic()
    matrix = capacity_matrix(schedule)
    n = len(schedule.hours)
    objective = np.zeros(1 + schedule.option_count)
    objective[0] = 1.
    lower = np.zeros(len(objective))
    upper = np.ones(len(objective))
    upper[1:] = schedule.block_sizes[schedule.blocks]
    lower[0] = max(1., schedule.fixed[:, :2].max(), schedule.original[:, :2].mean(axis=0).max())
    upper[0] = baseline
    lhs = np.r_[np.full(2 * n, -cp.COPT.INFINITY), schedule.block_sizes]
    rhs = np.r_[-schedule.fixed[:, :2].T.ravel(), schedule.block_sizes]
    env = cp.Envr()
    model = env.createModel('minimum_server_capacity')
    for key, value in {'Logging': 0, 'Threads': 1, 'LpMethod': lp_method, 'Crossover': 1,
                       'FeasTol': 1e-8, 'DualTol': 1e-8, 'TimeLimit': time_limit,
                       'RelGap': 0., 'AbsGap': .999}.items():
        model.setParam(key, value)
    model.loadMatrix(objective, matrix, lhs, rhs, lower, upper)
    model.solve()
    if model.status != cp.COPT.OPTIMAL:
        raise RuntimeError(f'容量 LP 松弛未达到最优状态 {model.status}')
    fractional = np.asarray(model.getValues())
    lp_lower = float(fractional[0])
    lp_audit = audit(schedule, fractional[1:], lp_lower, False)
    lp_seconds = time.monotonic() - started
    if lower_only:
        model.clear()
        env.close()
        return {'status': 'relaxation_lower_bound', 'lower_servers': int(np.ceil(lp_lower - 1e-6)),
                'continuous_lower_servers': lp_lower, 'solve_seconds': lp_seconds, 'lp_audit': lp_audit}
    rounded, rounded_servers = round_whole(schedule, fractional[1:])
    rounded, rounded_servers = rebalance_whole(schedule, rounded, int(np.ceil(lp_lower - 1e-6)) + 1)
    original = original_assignment(schedule)
    if rounded_servers > baseline:
        rounded, rounded_servers = original, baseline
    audit(schedule, rounded, rounded_servers, True)
    integer_lower = int(np.ceil(lp_lower - 1e-6))
    atomic_json(output.with_suffix('.progress.json'),
                {'lp_lower': lp_lower, 'integer_lower': integer_lower,
                 'rounded_upper': rounded_servers, 'lp_seconds': lp_seconds})
    rounded_peak = float(resource_dispatch(schedule, rounded).max())
    best = np.r_[rounded_peak, rounded]
    mip_lower = integer_lower
    continuous_lower = lp_lower
    if rounded_servers - integer_lower > 1 and schedule.integer.any():
        model.clear()
        fixed_lower, fixed_upper = lower.copy(), upper.copy()
        whole_columns = 1 + np.flatnonzero(schedule.integer)
        fixed_lower[whole_columns] = fixed_upper[whole_columns] = rounded[schedule.integer]
        model.loadMatrix(objective, matrix, lhs, rhs, fixed_lower, fixed_upper)
        model.solve()
        if model.status == cp.COPT.OPTIMAL:
            best = np.asarray(model.getValues())
            rounded = best[1:]
            rounded_peak = float(resource_dispatch(schedule, rounded).max())
            rounded_servers = int(np.ceil(rounded_peak - 1e-6))
            audit(schedule, rounded, rounded_servers, True)
            rounded, rounded_servers = rebalance_whole(schedule, rounded, integer_lower + 1)
            rounded_peak = float(resource_dispatch(schedule, rounded).max())
            best = np.r_[rounded_peak, rounded]
    np.savez_compressed(output.with_suffix('.seed.npz'), assignment=best[1:])
    status = 'certified_within_one' if rounded_servers - integer_lower <= 1 else 'pending'
    if status == 'pending':
        model.clear()
        kinds = np.full(len(objective), cp.COPT.CONTINUOUS)
        kinds[1:][schedule.integer] = cp.COPT.INTEGER
        lower[0] = lp_lower
        upper[0] = rounded_peak
        # 固定 LP 中已为整数的整作业选择，仅修复剩余作业组，所得解仍是完整问题的可行上界。
        first = np.r_[0, np.flatnonzero(np.diff(schedule.blocks)) + 1]
        stop = np.r_[first[1:], schedule.option_count]
        fixed_groups = 0
        for a, b in zip(first, stop):
            if schedule.integer[a] and np.max(np.abs(fractional[1 + a:1 + b] -
                                                      np.rint(fractional[1 + a:1 + b]))) < 1e-6:
                lower[1 + a:1 + b] = upper[1 + a:1 + b] = np.rint(fractional[1 + a:1 + b])
                kinds[1 + a:1 + b] = cp.COPT.CONTINUOUS
                fixed_groups += 1
        model.loadMatrix(objective, matrix, lhs, rhs, lower, upper, kinds)
        model.setMipStart(model.getVars(), best.tolist())
        model.loadMipStart()
        model.solve()
        if model.getAttr(cp.COPT.Attr.HasMipSol):
            best = np.asarray(model.getValues())
            raw_lower = float(model.getAttr(cp.COPT.Attr.BestBnd))
            if fixed_groups == 0 and np.isfinite(raw_lower) and raw_lower >= 0:
                continuous_lower = max(lp_lower, raw_lower)
                mip_lower = max(integer_lower, int(np.ceil(raw_lower - 1e-6)))
        servers = int(np.ceil(float(resource_dispatch(schedule, best[1:]).max()) - 1e-6))
        status = 'certified_within_one' if servers - mip_lower <= 1 else 'feasible_with_gap'
    else:
        servers = rounded_servers
    model.clear()
    env.close()
    if servers - mip_lower > 1 and full_mip:
        bounds_lower = np.zeros(len(objective))
        bounds_upper = np.r_[float(resource_dispatch(schedule, best[1:]).max()),
                              schedule.block_sizes[schedule.blocks]]
        bounds_lower[0] = lp_lower
        integrality = np.r_[0, schedule.integer.astype(int)]
        result = milp(objective, integrality=integrality, bounds=Bounds(bounds_lower, bounds_upper),
                      constraints=LinearConstraint(matrix, lhs, rhs),
                      options={'time_limit': time_limit, 'node_limit': 10000,
                               'mip_rel_gap': .25 / max(servers, 1), 'presolve': True, 'threads': 1})
        if result.status == 2:
            raise RuntimeError('完整容量问题排除了已经核验的可行调度')
        if result.x is not None:
            candidate = np.asarray(result.x)
            candidate[1:][schedule.integer] = np.rint(candidate[1:][schedule.integer])
            candidate_servers = int(np.ceil(resource_dispatch(schedule, candidate[1:]).max() - 1e-6))
            audit(schedule, candidate[1:], candidate_servers, True)
            if candidate_servers <= servers:
                best, servers = candidate, candidate_servers
        dual = getattr(result, 'mip_dual_bound', None)
        if dual is not None and np.isfinite(dual):
            continuous_lower = max(continuous_lower, float(dual))
            mip_lower = max(mip_lower, int(np.ceil(continuous_lower - 1e-6)))
        status = 'certified_within_one' if servers - mip_lower <= 1 else 'feasible_with_gap'
    checks = audit(schedule, best[1:], servers, True)
    np.savez_compressed(output.with_suffix('.npz'), assignment=best[1:])
    power = (schedule.fixed[:, 2] + np.bincount(schedule.slots,
                weights=schedule.values[:, 2] * best[1:][schedule.columns], minlength=len(schedule.hours))
             if schedule.original.shape[1] > 2 else None)
    return {'status': status, 'servers': servers, 'lower_servers': mip_lower,
            'gap_servers': servers - mip_lower, 'grouped_lp_lower_servers': lp_lower,
            'continuous_lower_servers': continuous_lower,
            'continuous_upper_servers': float(resource_dispatch(schedule, best[1:]).max()),
            'lp_seconds': lp_seconds, 'solve_seconds': time.monotonic() - started,
            'options': schedule.option_count, 'whole_groups': schedule.whole_jobs,
            'task_power_peak_mw': float(power.max()) if power is not None else None,
            'task_power_peak_ratio': float(power.max() / schedule.original[:, 2].max()) if power is not None else None,
            'hours': len(schedule.hours), 'audit': checks, 'lp_audit': lp_audit}


def case(task):
    from highs_capacity_solver import solve_capacity_highs

    args, scenario, whole_h, split_h = task
    solver = solve_capacity_highs if args.solver == 'highs' else solve_capacity
    name = f'{scenario}-w{whole_h}-s{split_h}'
    output = args.output / f'{name}.json'
    manifest = json.loads((args.resources / 'resource-manifest.json').read_text('utf-8'))
    fleet = manifest[scenario]
    schedule = make_resource_schedule(args, scenario, fleet, whole_h=whole_h, split_h=split_h)
    base = 'B-' + fleet['use']
    base_path = args.output / f'{base}-w{whole_h}-s{split_h}.json'
    reusable = scenario[0] in ('C', 'D') and base_path.is_file()
    if reusable:
        base_result = json.loads(base_path.read_text('utf-8'))
        factor = fleet['reference_units'] / manifest[base]['reference_units']
        for path, base_file in (
                (args.reference / f'{scenario}-10MW.parquet', args.reference / f'{base}-10MW.parquet'),
                (args.whole_jobs / f'{scenario}.parquet', args.whole_jobs / f'{base}.parquet')):
            if path.is_file() != base_file.is_file():
                raise ValueError(f'{scenario} 与 {base} 的资源记录不同')
            if not path.is_file():
                continue
            actual, expected = pd.read_parquet(path), pd.read_parquet(base_file)
            np.testing.assert_array_equal(actual.source_hour_index, expected.source_hour_index)
            if 'cpu_units' in actual:
                np.testing.assert_array_equal(actual.job, expected.job)
                np.testing.assert_allclose(actual[['cpu_units', 'gpu_units']], expected[['cpu_units', 'gpu_units']] * factor,
                                           rtol=1e-12, atol=1e-8)
            else:
                np.testing.assert_allclose(actual[['u_cpu_proxy', 'allocated_gpu_fraction']],
                                           expected[['u_cpu_proxy', 'allocated_gpu_fraction']], rtol=0, atol=1e-12)
        assignment = np.load(base_path.with_suffix('.npz'))['assignment']
        lower_value = base_result['continuous_lower_servers'] * factor
        upper_value = float(resource_dispatch(schedule, assignment).max())
        servers = int(np.ceil(upper_value - 1e-6))
        lower_servers = int(np.ceil(lower_value - 1e-6))
        if servers - lower_servers <= 1:
            np.savez_compressed(output.with_suffix('.npz'), assignment=assignment)
            result = {**base_result, 'status': 'certified_within_one', 'servers': servers, 'lower_servers': lower_servers,
                      'gap_servers': servers - lower_servers, 'continuous_lower_servers': lower_value,
                      'grouped_lp_lower_servers': base_result['grouped_lp_lower_servers'] * factor,
                      'continuous_upper_servers': upper_value, 'reused_resource_model': base,
                      'resource_scale': factor, 'solve_seconds': 0.,
                      'audit': audit(schedule, assignment, servers, True)}
            result.update(task_power_peak_mw=None, task_power_peak_ratio=None)
        else:
            result = solver(schedule, fleet['baseline_units'], output, args.time_limit,
                                    full_mip=not args.lp_repair_only)
    else:
        result = solver(schedule, fleet['baseline_units'], output, args.time_limit,
                                full_mip=not args.lp_repair_only)
    result.update(scenario=scenario, whole_h=whole_h, split_h=split_h,
                  baseline_servers=fleet['baseline_units'], gpus_per_server=fleet['gpus_per_unit'],
                  hourly_whole=args.hourly_whole, source_manifest_sha256=sha256(args.resources / 'resource-manifest.json'))
    atomic_json(output, result)
    print(name, result['status'], result['lower_servers'], result['servers'], result['solve_seconds'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('reference', 'resources', 'whole-jobs', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--city-inputs', type=Path)
    parser.add_argument('--workers', type=int, default=2, choices=range(1, 5))
    parser.add_argument('--time-limit', type=int, default=900)
    parser.add_argument('--scenario', choices=SCENARIOS, action='append')
    parser.add_argument('--deadline', type=int, choices=(0, 1, 2), action='append')
    parser.add_argument('--hourly-whole', action='store_true')
    parser.add_argument('--lp-repair-only', action='store_true')
    parser.add_argument('--solver', choices=('copt', 'highs'), default='copt')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [(args, scenario, *DEADLINES[k]) for k in (args.deadline or (0, 1, 2))
             for scenario in (args.scenario or SCENARIOS)]
    if args.workers == 1:
        for task in tasks:
            case(task)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(case, tasks))


if __name__ == '__main__':
    main()
