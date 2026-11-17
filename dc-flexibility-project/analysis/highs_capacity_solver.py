import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from minimum_server_capacity import audit, capacity_matrix, original_assignment, rebalance_whole, resource_dispatch, round_whole


def fixed_fluid_lp(schedule, fixed_assignment, baseline, lower, time_limit):
    from scipy import sparse

    n = len(schedule.hours)
    fluid = np.flatnonzero(~schedule.integer)
    selected = np.flatnonzero(~schedule.integer[schedule.columns])
    mapping = np.full(schedule.option_count, -1)
    mapping[fluid] = np.arange(len(fluid))
    blocks, block_labels = np.unique(schedule.blocks[fluid], return_inverse=True)
    block_map = np.full(schedule.block_count, -1)
    block_map[blocks] = np.arange(len(blocks))
    amplitude = np.zeros(len(blocks))
    np.maximum.at(amplitude, block_map[schedule.blocks[schedule.columns[selected]]],
                  schedule.values[selected, :2].max(axis=1))
    amplitude[amplitude == 0] = 1.
    columns = 1 + mapping[schedule.columns[selected]]
    scaled = schedule.values[selected, :2] / amplitude[block_labels[mapping[schedule.columns[selected]]], None]
    rows = np.r_[schedule.slots[selected], n + schedule.slots[selected],
                 2 * n + block_labels, np.arange(2 * n)]
    cols = np.r_[columns, columns, 1 + np.arange(len(fluid)), np.zeros(2 * n, dtype=int)]
    values = np.r_[scaled[:, 0], scaled[:, 1], np.ones(len(fluid)), -np.ones(2 * n)]
    matrix = sparse.csc_matrix((values, (rows, cols)), shape=(2 * n + len(blocks), 1 + len(fluid)))
    matrix.eliminate_zeros()
    whole = fixed_assignment.copy()
    whole[fluid] = 0.
    fixed = resource_dispatch(schedule, whole)
    bounds = np.column_stack((np.zeros(1 + len(fluid)),
        np.r_[baseline, schedule.block_sizes[blocks[block_labels]] * amplitude[block_labels]]))
    bounds[0, 0] = lower
    result = linprog(np.r_[1., np.zeros(len(fluid))], A_ub=matrix[:2*n], b_ub=-fixed.T.ravel(),
        A_eq=matrix[2*n:], b_eq=schedule.block_sizes[blocks] * amplitude, bounds=bounds,
        method='highs-ipm', options={'time_limit': time_limit, 'threads': 1,
            'primal_feasibility_tolerance': 1e-8, 'dual_feasibility_tolerance': 1e-8})
    candidate = None
    if result.x is not None:
        candidate = whole.copy()
        candidate[fluid] = result.x[1:] / amplitude[block_labels]
        audit(schedule, candidate, result.x[0], True)
    return candidate, result.message


def capacity_lp(schedule, baseline, lower, time_limit, *, matrix=None, fixed=None):
    if matrix is None:
        matrix = capacity_matrix(schedule)
    n = len(schedule.hours)
    bounds = np.column_stack((np.zeros(1 + schedule.option_count), np.full(1 + schedule.option_count, np.inf)))
    bounds[0] = (lower, baseline)
    if fixed is not None:
        indices = 1 + np.flatnonzero(schedule.integer)
        bounds[indices, 0] = bounds[indices, 1] = fixed[schedule.integer]
    result = linprog(np.r_[1., np.zeros(schedule.option_count)],
                     A_ub=matrix[:2 * n], b_ub=-schedule.fixed[:, :2].T.ravel(),
                     A_eq=matrix[2 * n:], b_eq=schedule.block_sizes, bounds=bounds,
                     method='highs-ds', options={'time_limit': time_limit, 'threads': 1,
                        'primal_feasibility_tolerance': 1e-8, 'dual_feasibility_tolerance': 1e-8})
    if not result.success:
        raise RuntimeError(f'HiGHS 容量 LP 未达到最优 {result.message}')
    audit(schedule, result.x[1:], result.fun, False)
    return result, matrix


def solve_capacity_highs(schedule, baseline, output, time_limit, *, full_mip=True,
                         fractional=None, certified_lower=None):
    started = time.monotonic()
    lower = max(1., schedule.fixed[:, :2].max(), schedule.original[:, :2].mean(axis=0).max())
    matrix = None
    if fractional is None:
        relaxation, matrix = capacity_lp(schedule, baseline, lower, time_limit)
        relaxed = relaxation.x
        lp_lower = float(relaxation.fun)
    else:
        relaxed = np.asarray(fractional)
        lp_lower = float(certified_lower)
        audit(schedule, relaxed[1:], relaxed[0], False)
    lower_integer = int(np.ceil(lp_lower - 1e-5))
    assignment, servers = round_whole(schedule, relaxed[1:])
    print('integer rounding', lower_integer, servers, 'seconds', time.monotonic() - started, flush=True)
    assignment, servers = rebalance_whole(schedule, assignment, lower_integer + 1)
    print('integer rebalancing', lower_integer, servers, 'seconds', time.monotonic() - started, flush=True)
    if servers > baseline:
        assignment, servers = original_assignment(schedule), baseline
    audit(schedule, assignment, servers, True)
    if (~schedule.integer).any() and schedule.integer.any() and servers > lower_integer + 1:
        candidate, fluid_status = fixed_fluid_lp(schedule, assignment, baseline, lower, time_limit)
        print('fluid LP status', fluid_status, flush=True)
        if candidate is not None:
            candidate_servers = int(np.ceil(resource_dispatch(schedule, candidate).max() - 1e-5))
            if candidate_servers <= servers:
                assignment, servers = candidate, candidate_servers
        print('fluid reoptimization', lower_integer, servers, 'seconds', time.monotonic() - started, flush=True)
    if servers - lower_integer > 1:
        if matrix is None:
            matrix = capacity_matrix(schedule)
        n = len(schedule.hours)
        lb = np.zeros(1 + schedule.option_count)
        ub = np.r_[float(resource_dispatch(schedule, assignment).max()) + 1e-5,
                    schedule.block_sizes[schedule.blocks]]
        lb[0] = lp_lower
        if not full_mip:
            first = np.r_[0, np.flatnonzero(np.diff(schedule.blocks)) + 1]
            stop = np.r_[first[1:], schedule.option_count]
            for a, b in zip(first, stop):
                if schedule.integer[a] and np.max(np.abs(relaxed[1 + a:1 + b] -
                                                          np.rint(relaxed[1 + a:1 + b]))) < 1e-6:
                    lb[1 + a:1 + b] = ub[1 + a:1 + b] = np.rint(assignment[a:b])
        lhs = np.r_[np.full(2 * n, -np.inf), schedule.block_sizes]
        rhs = np.r_[-schedule.fixed[:, :2].T.ravel(), schedule.block_sizes]
        integer = milp(np.r_[1., np.zeros(schedule.option_count)],
                       integrality=np.r_[0, schedule.integer.astype(int)], bounds=Bounds(lb, ub),
                       constraints=LinearConstraint(matrix, lhs, rhs),
                       options={'time_limit': time_limit, 'threads': 1, 'mip_rel_gap': .25 / max(servers, 1)})
        if integer.status == 2:
            raise RuntimeError('容量整数修复排除了已核验的可行方案')
        if integer.x is not None:
            candidate = integer.x[1:]
            candidate[schedule.integer] = np.rint(candidate[schedule.integer])
            candidate_servers = int(np.ceil(resource_dispatch(schedule, candidate).max() - 1e-5))
            audit(schedule, candidate, candidate_servers, True)
            if candidate_servers <= servers:
                assignment, servers = candidate, candidate_servers
    peak = float(resource_dispatch(schedule, assignment).max())
    checks = audit(schedule, assignment, servers, True)
    np.savez_compressed(output.with_suffix('.npz'), assignment=assignment)
    return {'status': 'certified_within_one' if servers - lower_integer <= 1 else 'feasible_with_gap',
            'servers': servers, 'lower_servers': lower_integer, 'gap_servers': servers - lower_integer,
            'continuous_lower_servers': lp_lower, 'continuous_upper_servers': peak,
            'grouped_lp_lower_servers': lp_lower, 'solve_seconds': time.monotonic() - started,
            'options': schedule.option_count, 'whole_groups': schedule.whole_jobs,
            'hours': len(schedule.hours), 'audit': checks, 'lp_audit': audit(schedule, relaxed[1:], relaxed[0], False),
            'solver': 'HiGHS', 'task_power_peak_mw': None, 'task_power_peak_ratio': None}
