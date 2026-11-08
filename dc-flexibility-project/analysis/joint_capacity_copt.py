import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse

from flex_econ_copt import CITIES, SCENARIOS, crf, load_config
from joint_capacity_inputs import atomic_json, sha256


ASSETS = ('pv', 'wind', 'battery_power', 'battery_energy', 'grid_capacity')
OPERATIONS = ('direct', 'grid', 'charge', 'discharge', 'soc', 'curtail')
SPLIT_SAI = ('DNN 训练（含推断）', '长时单任务', '多节点大作业')
METRICS = ('cpu_units', 'gpu_units', 'mw')
MIP_RELATIVE_GAP = 1e-4


@dataclass
class Schedule:
    hours: np.ndarray
    fixed: np.ndarray
    original: np.ndarray
    blocks: np.ndarray
    slots: np.ndarray
    columns: np.ndarray
    values: np.ndarray
    integer: np.ndarray
    option_count: int
    block_count: int
    whole_jobs: int
    block_sizes: np.ndarray | None = None


def split_resources(reference, use, source):
    if use == 'TRN':
        gpu = source[['gpu_hours_Pretrain', 'gpu_hours_SFT', 'gpu_hours_MLLM']].sum(axis=1).to_numpy() / 2032
        return np.column_stack((.25 * gpu, gpu))
    if use == 'SAI':
        inputs = pd.read_parquet(reference / 'SAI-resources.parquet')
        for resource, field in (('cpu', 'u_cpu_proxy'), ('gpu', 'allocated_gpu_fraction')):
            np.testing.assert_allclose(inputs.filter(regex='^' + resource + ':').sum(axis=1), source[field], atol=1e-9, rtol=0)
        return np.column_stack([inputs[[resource + ':' + name for name in SPLIT_SAI]].sum(axis=1) for resource in ('cpu', 'gpu')])
    return np.zeros((len(source), 2))


def resource_layers(args, scenario, fleet):
    source = pd.read_parquet(args.reference / f'{scenario}-10MW.parquet')
    use = fleet['use']
    units = fleet['reference_units']
    original = units * source[['u_cpu_proxy', 'allocated_gpu_fraction']].to_numpy()
    split = units * split_resources(args.reference, use, source)
    if use == 'TRN':
        gpu = source.gpu_hours_Eval.to_numpy() * units / 2032
        whole = np.column_stack((.25 * gpu, gpu))
    else:
        resources = pd.read_parquet(args.reference / f'{use}-resources.parquet')
        labels = {'HPC': ['CPU 科研分区'], 'CLD': ['离线批处理'],
                  'SAI': ['批处理', 'MapReduce', '数组批量'], 'INF': ['离线推理']}[use]
        whole = units * np.column_stack([resources[[kind + ':' + label for label in labels]].sum(axis=1)
                                         for kind in ('cpu', 'gpu')])
    fixed = original - whole - split
    if fixed.min() < -1e-6:
        raise ValueError(f'{scenario} 资源分类出现负值')
    return source.source_hour_index.to_numpy(np.int64), fixed.clip(min=0), whole, split


def whole_choices(job_hours, codes, hours, delay):
    starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    valid = np.empty((len(starts), delay + 1), dtype=bool)
    for shift in range(delay + 1):
        targets = np.searchsorted(hours, job_hours + shift)
        present = (targets < len(hours)) & (hours[targets.clip(max=len(hours) - 1)] == job_hours + shift)
        valid[:, shift] = np.logical_and.reduceat(present, starts)
    return valid


def whole_options(path, hours, delay, fixed, first_option, first_block, selections=None, metrics=METRICS):
    jobs = pd.read_parquet(path).sort_values(['job', 'source_hour_index'])
    if jobs.empty:
        return None
    codes, _ = pd.factorize(jobs.job, sort=False)
    starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    widths = np.diff(np.r_[starts, len(jobs)])
    job_hours = jobs.source_hour_index.to_numpy(np.int64)
    values = jobs[list(metrics)].to_numpy(float)
    counts = jobs.job_count.to_numpy() if 'job_count' in jobs else np.ones(len(jobs), dtype=int)
    if (counts < 1).any() or np.any(counts != np.rint(counts)) or np.any(counts != counts[starts][codes]):
        raise ValueError('同类作业数量无效')
    adjacent = np.diff(codes) == 0
    if np.any(np.diff(job_hours)[adjacent] != 1) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('整作业时间或资源无效')
    valid = whole_choices(job_hours, codes, hours, delay)
    if not valid[:, 0].all():
        raise ValueError('整作业占用未观测时段')
    movable = valid.sum(axis=1) > 1
    selected = movable[codes]
    np.add.at(fixed, np.searchsorted(hours, job_hours[selected]), -values[selected] * counts[selected, None])
    if selections is not None:
        if selections.shape != valid.shape or not selections[:, 0].all():
            raise ValueError('受限启动时间必须包含原始时序，且与完整作业组形状一致')
        valid &= selections
    valid[~movable] = False
    groups, shifts = np.nonzero(valid)
    if not len(groups):
        return None
    entries = np.repeat(groups, widths[groups])
    offsets = np.arange(len(entries)) - np.repeat(np.cumsum(widths[groups]) - widths[groups], widths[groups])
    records = starts[entries] + offsets
    columns = np.repeat(np.arange(len(groups)), widths[groups])
    slots = np.searchsorted(hours, job_hours[records] + shifts[columns])
    blocks = np.cumsum(movable)[groups] - 1
    return (first_block + blocks, slots, first_option + columns, values[records],
            np.ones(len(groups), dtype=bool), counts[starts[movable]])


def make_resource_schedule(args, scenario, fleet, *, whole_h=12, split_h=48):
    hours, rigid, whole, split = resource_layers(args, scenario, fleet)
    original = rigid + whole + split
    hourly_whole = whole if fleet['use'] in ('CLD', 'INF') and args.hourly_whole else np.zeros_like(whole)
    fixed = original - split - hourly_whole
    blocks, slots, columns, values, integer, sizes = [], [], [], [], [], []
    option_count = block_count = whole_count = 0
    for profile, deadline in ((split, split_h), (hourly_whole, whole_h)):
        origins = np.flatnonzero(np.any(profile > 0, axis=1))
        targets = np.searchsorted(hours, hours[origins, None] + np.arange(deadline + 1))
        valid = (targets < len(hours)) & (hours[targets.clip(max=len(hours) - 1)] ==
                                         hours[origins, None] + np.arange(deadline + 1))
        origin, shift = np.nonzero(valid)
        blocks.append(block_count + origin)
        slots.append(targets[origin, shift])
        columns.append(option_count + np.arange(len(origin)))
        values.append(profile[origins[origin]])
        integer.append(np.zeros(len(origin), dtype=bool))
        sizes.extend(np.ones(len(origins)))
        option_count += len(origin)
        block_count += len(origins)
    path = args.whole_jobs / f'{scenario}.parquet'
    if path.is_file():
        packed = whole_options(path, hours, whole_h, fixed, option_count, block_count,
                               getattr(args, 'pattern_choices', None), metrics=('cpu_units', 'gpu_units'))
        if packed is not None:
            b, s, c, v, integral, counts = packed
            blocks.append(b)
            slots.append(s)
            columns.append(c)
            values.append(v)
            integer.append(integral)
            sizes.extend(counts)
            option_count += len(integral)
            block_count += len(counts)
            whole_count += len(counts)
    if fixed.min() < -1e-7:
        raise ValueError('逐作业资源量超过原始任务需求')
    return Schedule(hours, fixed.clip(min=0), original, np.concatenate(blocks), np.concatenate(slots),
                    np.concatenate(columns), np.concatenate(values), np.concatenate(integer),
                    option_count, block_count, whole_count, np.asarray(sizes))


def make_schedule(args, scenario, fleet, *, whole_h=12, split_h=48):
    use = fleet['use']
    source = pd.read_parquet(args.reference / f'{scenario}-10MW.parquet')
    frame = pd.read_parquet(args.city_inputs / f'frankfurt-{scenario}.parquet')
    hours = frame.source_hour_index.to_numpy(np.int64)
    np.testing.assert_array_equal(source.source_hour_index, hours)
    if np.any(np.diff(hours) <= 0):
        raise ValueError(f'{scenario} 原生时间轴无效')
    units = fleet['reference_units']
    all_work = np.column_stack((
        source.u_cpu_proxy * units, source.allocated_gpu_fraction * units,
        frame.rigid + frame.unknown + frame.whole + frame.split))
    split = np.column_stack((split_resources(args.reference, use, source) * units, frame.split))
    hourly_whole = np.zeros_like(split)
    if getattr(args, 'hourly_whole', False) and use in ('CLD', 'INF'):
        resources = pd.read_parquet(args.reference / f'{use}-resources.parquet')
        label = '离线批处理' if use == 'CLD' else '离线推理'
        hourly_whole = np.column_stack((resources['cpu:' + label] * units,
                                        resources['gpu:' + label] * units, frame.whole))
    fixed = all_work - split - hourly_whole
    if not np.isfinite(fixed).all() or not np.isfinite(split).all() or min(fixed.min(), split.min()) < -1e-7:
        raise ValueError(f'{scenario} 的固定或可拆分工作量无效')
    fixed = fixed.clip(min=0)
    split = split.clip(min=0)

    # 以来源小时为批次，按相同比例同时转移 CPU、GPU 和任务用电。
    blocks, slots, columns, values, integer, block_sizes = [], [], [], [], [], []
    option_count = block_count = 0
    for profile, deadline in ((split, split_h), (hourly_whole, whole_h)):
        origins = np.flatnonzero(np.any(profile > 0, axis=1))
        rows = np.searchsorted(hours, hours[origins, None] + np.arange(deadline + 1))
        valid = (rows < len(hours)) & (hours[rows.clip(max=len(hours) - 1)] == hours[origins, None] + np.arange(deadline + 1))
        release, offsets = np.nonzero(valid)
        destinations = rows[release, offsets]
        blocks.append(block_count + release)
        slots.append(destinations)
        columns.append(option_count + np.arange(len(destinations)))
        values.append(profile[origins[release]])
        integer.append(np.zeros(len(destinations), dtype=bool))
        block_sizes.extend(np.ones(len(origins)))
        option_count += len(destinations)
        block_count += len(origins)

    whole_count = 0
    path = args.whole_jobs / f'{scenario}.parquet'
    if path.is_file():
        packed = whole_options(path, hours, whole_h, fixed, option_count, block_count,
                               getattr(args, 'pattern_choices', None))
        if packed is not None:
            b, s, c, v, integral, sizes = packed
            blocks.append(b)
            slots.append(s)
            columns.append(c)
            values.append(v)
            integer.append(integral)
            block_sizes.extend(sizes)
            option_count += len(integral)
            block_count += len(sizes)
            whole_count += len(sizes)
    if fixed.min() < -1e-7:
        raise ValueError(f'{scenario} 的整作业资源超出原始需求')
    return Schedule(hours, fixed.clip(min=0), all_work, np.concatenate(blocks), np.concatenate(slots),
                    np.concatenate(columns), np.concatenate(values), np.concatenate(integer),
                    option_count, block_count, whole_count, np.asarray(block_sizes))


def solve_with_start(frame, workload, fleet, p, price, alpha, stage, time_limit, warm_start, *,
                     seed_only=False, relax_whole=False, fixed_whole_values=None, lp_solver='copt'):
    import coptpy as cp

    started = time.monotonic()
    n = len(frame)
    np.testing.assert_array_equal(frame.source_hour_index, workload.hours)
    pue = frame.pue.to_numpy()
    idx = np.arange(n)
    off = {key: 6 + i * n for i, key in enumerate(OPERATIONS)}
    first_option = 6 + n * len(OPERATIONS)
    count = first_option + workload.option_count
    lb = np.zeros(count)
    ub = np.full(count, cp.COPT.INFINITY)
    c = np.zeros(count)
    vtype = np.full(count, cp.COPT.CONTINUOUS)
    ub[:5] = [p['pv_max_mw'], p['wind_max_mw'], p['battery_power_max_mw'], p['battery_energy_max_mwh'], p['grid_max_mw']]
    generation_limit = frame.cf_pv.to_numpy() * ub[0] + frame.cf_wind.to_numpy() * ub[1]
    for key, limit in (('direct', generation_limit), ('curtail', generation_limit),
                       ('grid', ub[4]), ('charge', ub[2]), ('discharge', ub[2]), ('soc', ub[3])):
        ub[off[key] + idx] = limit
    ub[5] = fleet['baseline_units']
    lower_units = max(workload.fixed[:, :2].max(), workload.original[:, :2].mean(axis=0).max())
    lb[5] = ub[5] if stage == 'B' else max(1, np.ceil(lower_units - 1e-8))
    if stage == 'C':
        vtype[5] = cp.COPT.INTEGER
    sizes = workload.block_sizes if workload.block_sizes is not None else np.ones(workload.block_count)
    ub[first_option:] = sizes[workload.blocks]
    if not relax_whole:
        vtype[first_option:][workload.integer] = cp.COPT.INTEGER
    is_mip = stage == 'C' or (workload.integer.any() and not relax_whole and fixed_whole_values is None)
    first = np.unique(workload.blocks, return_index=True)[1]
    if seed_only:
        if warm_start is None:
            seed = np.zeros(workload.option_count)
            seed[first] = sizes
        else:
            seed = np.rint(warm_start[first_option:])
        whole = first_option + np.flatnonzero(workload.integer)
        lb[whole] = ub[whole] = seed[workload.integer]
        vtype[whole] = cp.COPT.CONTINUOUS
    if fixed_whole_values is not None:
        whole = first_option + np.flatnonzero(workload.integer)
        selected = np.rint(fixed_whole_values[workload.integer])
        if np.max(np.abs(selected - fixed_whole_values[workload.integer]), initial=0) > 1e-5:
            raise ValueError('固定的整作业选择含有非整数作业数量')
        lb[whole] = ub[whole] = selected
        vtype[whole] = cp.COPT.CONTINUOUS
    scale = n / 8760
    annual_equipment_per_unit = price * (crf(p['discount_rate'], p['compute_life_years']) + p['compute_fom_fraction'])
    c[:5] = [p['annual_usd_per_mw'][key] * scale for key in ASSETS]
    c[5] = annual_equipment_per_unit * scale
    c[off['grid'] + idx] = p['grid_price_usd_mwh'] + p['carbon_price_usd_t'] * p['emission_factor']
    c[off['discharge'] + idx] = p['battery_throughput_usd_mwh']
    rows, cols, vals, lhs, rhs = [], [], [], [], []
    row = 0

    def add(row_id, col_id, value):
        r, j, v = np.broadcast_arrays(row_id, col_id, value)
        rows.append(r.ravel())
        cols.append(j.ravel())
        vals.append(v.ravel())

    def bounds(lower, upper, size=n):
        nonlocal row
        lhs.append(np.broadcast_to(lower, size))
        rhs.append(np.broadcast_to(upper, size))
        row += size

    fixed = workload.fixed
    options = first_option + workload.columns
    slots = workload.slots
    powers = workload.values[:, 2]
    idle = fleet['idle_mw_per_unit']
    inf = cp.COPT.INFINITY

    add(row + idx, 0, frame.cf_pv)
    add(row + idx, 1, frame.cf_wind)
    for key in ('direct', 'charge', 'curtail'):
        add(row + idx, off[key] + idx, -1)
    bounds(0, 0)

    for key in ('direct', 'grid', 'discharge'):
        add(row + idx, off[key] + idx, 1)
    add(row + idx, 5, -pue * idle)
    add(row + slots, options, -powers * pue[slots])
    bounds(pue * fixed[:, 2], pue * fixed[:, 2])

    eta = np.sqrt(p['battery_roundtrip_efficiency'])
    add(row + idx, off['soc'] + idx, 1)
    add(row + idx, off['soc'] + (idx - 1) % n, -1)
    add(row + idx, off['charge'] + idx, -eta)
    add(row + idx, off['discharge'] + idx, 1 / eta)
    bounds(0, 0)

    for resource in range(2):
        resource_scale = max(1., fleet['baseline_units'])
        add(row + idx, 5, -1 / resource_scale)
        add(row + slots, options, workload.values[:, resource] / resource_scale)
        bounds(-inf, -fixed[:, resource] / resource_scale)

    add(row + idx, 5, idle)
    add(row + slots, options, powers)
    bounds(-inf, frame.cap_it_mw.to_numpy() - fixed[:, 2])

    for key, capacity in (('charge', 2), ('discharge', 2), ('soc', 3), ('grid', 4)):
        add(row + idx, off[key] + idx, 1)
        add(row + idx, capacity, -1)
        bounds(-inf, 0)

    batch_row_start = row
    add(row + workload.blocks, first_option + np.arange(workload.option_count), 1)
    bounds(sizes, sizes, workload.block_count)

    if alpha == 1:
        ub[off['grid'] + idx] = 0

    matrix = sparse.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(row, count))
    matrix.eliminate_zeros()
    left, right = np.concatenate(lhs), np.concatenate(rhs)
    variable_scale = np.ones(count)
    if not is_mip and relax_whole:
        amplitude = np.zeros(workload.block_count)
        resource_scale = max(1., fleet['baseline_units'])
        scale_entries = np.maximum(workload.values[:, 2], workload.values[:, :2].max(axis=1) / resource_scale)
        np.maximum.at(amplitude, workload.blocks[workload.columns], scale_entries)
        amplitude[amplitude == 0] = 1.
        variable_scale[first_option:] = amplitude[workload.blocks]
        matrix.data /= np.repeat(variable_scale, np.diff(matrix.indptr))
        batch_entries = matrix.indices >= batch_row_start
        matrix.data[batch_entries] *= amplitude[matrix.indices[batch_entries] - batch_row_start]
        left[batch_row_start:] *= amplitude
        right[batch_row_start:] *= amplitude
        lb *= variable_scale
        ub *= variable_scale
        c /= variable_scale
    if lp_solver == 'highs':
        if is_mip:
            raise ValueError('HiGHS 能源复核只接受明确放松后的连续模型')
        import highspy as hs

        solver = hs.Highs()
        for key, value in {'output_flag': True, 'threads': 1, 'solver': 'ipm',
                           'run_crossover': 'on', 'ipm_optimality_tolerance': 1e-10,
                           'time_limit': float(time_limit),
                           'primal_feasibility_tolerance': 1e-8,
                           'dual_feasibility_tolerance': 1e-8}.items():
            if solver.setOptionValue(key, value) == hs.HighsStatus.kError:
                raise RuntimeError(f'HiGHS 能源参数设置失败 {key}')
        lp = hs.HighsLp()
        lp.num_col_, lp.num_row_ = count, row
        lp.col_cost_, lp.col_lower_, lp.col_upper_ = c / 1000, lb, ub
        lp.row_lower_, lp.row_upper_ = left, right
        lp.a_matrix_.format_ = hs.MatrixFormat.kColwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = matrix.indptr, matrix.indices, matrix.data
        if solver.passModel(lp) == hs.HighsStatus.kError:
            raise RuntimeError('HiGHS 能源 LP 执行失败')
        if solver.run() == hs.HighsStatus.kError:
            raise RuntimeError('HiGHS 能源 LP 执行失败')
        solution = solver.getSolution()
        if not solution.value_valid:
            raise RuntimeError(f'HiGHS 能源 LP 没有可行解 {solver.modelStatusToString(solver.getModelStatus())}')
        lp_values = np.asarray(solution.col_value)
        status = cp.COPT.OPTIMAL if solver.getModelStatus() == hs.HighsModelStatus.kOptimal else 0
        lp_dual = np.asarray(solution.row_dual)
        lower_unbounded = left <= -cp.COPT.INFINITY
        upper_unbounded = right >= cp.COPT.INFINITY
        lp_dual[lower_unbounded] = np.minimum(lp_dual[lower_unbounded], 0)
        lp_dual[upper_unbounded] = np.maximum(lp_dual[upper_unbounded], 0)
        reduced = c / 1000 - matrix.T @ lp_dual
        lp_lower_window = float(lp_dual @ np.where(lp_dual >= 0, left, right) +
                                np.where(reduced >= 0, reduced * lb, reduced * ub).sum())
    else:
        env = cp.Envr()
        model = env.createModel('joint_capacity')
        for key, value in {'Logging': 0, 'Threads': 1, 'LpMethod': 2, 'Crossover': 1,
                           'FeasTol': 1e-8, 'DualTol': 1e-8, 'RelGap': MIP_RELATIVE_GAP, 'TimeLimit': time_limit}.items():
            model.setParam(key, value)
        model.loadMatrix(c / 1000, matrix, left, right, lb, ub, vtype)
        if is_mip:
            variables = model.getVars()
            if warm_start is not None:
                model.setMipStart(variables, warm_start.tolist())
            else:
                # 原时序是可行调度。先提供整数变量，供求解器补齐供能决策。
                initial = np.zeros(workload.option_count)
                initial[first] = sizes
                selected = np.flatnonzero(workload.integer)
                indices = np.r_[5, first_option + selected]
                model.setMipStart([variables[int(i)] for i in indices],
                                  np.r_[fleet['baseline_units'], initial[selected]].tolist())
            model.loadMipStart()
        model.solve()
        status = model.status
        if not is_mip and model.getAttr(cp.COPT.Attr.HasLpSol):
            dual = np.asarray(model.getDuals())
            dual[left <= -cp.COPT.INFINITY] = np.minimum(dual[left <= -cp.COPT.INFINITY], 0)
            dual[right >= cp.COPT.INFINITY] = np.maximum(dual[right >= cp.COPT.INFINITY], 0)
            reduced = c / 1000 - matrix.T @ dual
            lp_lower_window = float(dual @ np.where(dual >= 0, left, right) +
                                    np.where(reduced >= 0, reduced * lb, reduced * ub).sum())
    result = {'stage': stage, 'alpha': alpha, 'hours': n, 'status_code': int(status),
              'whole_jobs': workload.whole_jobs, 'schedule_options': workload.option_count,
              'whole_start_relaxation': relax_whole, 'whole_schedule_fixed': fixed_whole_values is not None,
              'solver': lp_solver,
              'requested_mip_relative_gap': MIP_RELATIVE_GAP if is_mip else None,
              'solve_seconds': time.monotonic() - started,
              'status': 'infeasible' if status == cp.COPT.INFEASIBLE else 'not_solved'}
    x = None
    if (status == cp.COPT.OPTIMAL or lp_solver == 'highs' or
            (is_mip and model.getAttr(cp.COPT.Attr.HasMipSol)) or
            (lp_solver == 'copt' and not is_mip and model.getAttr(cp.COPT.Attr.HasLpSol))):
        solver_x = lp_values if lp_solver == 'highs' else np.asarray(model.getValues(), dtype=float)
        x = solver_x / variable_scale
        servers = round(x[5])
        dispatch = fixed.copy()
        np.add.at(dispatch, slots, workload.values * x[options, None])
        operation = {key: x[off[key] + idx] for key in OPERATIONS}
        it = dispatch[:, 2] + servers * idle
        block_sum = np.bincount(workload.blocks, weights=x[first_option:], minlength=workload.block_count)
        binary = x[first_option:][workload.integer]
        errors = {
            'linear_rows': float(max(np.maximum(left - matrix @ solver_x, 0).max(initial=0),
                                     np.maximum(matrix @ solver_x - right, 0).max(initial=0))),
            'variable_bounds': float(max(np.maximum(lb - solver_x, 0).max(initial=0),
                                         np.maximum(solver_x - ub, 0).max(initial=0))),
            'resources': float(max(0, dispatch[:, :2].max() - servers)),
            'work_conservation': float(np.max(np.abs(dispatch.sum(axis=0) - workload.original.sum(axis=0)))),
            'batches': float(np.max(np.abs(block_sum - sizes), initial=0)),
            'integrality': float(max(abs(servers - x[5]), 0. if relax_whole else
                                     np.max(np.abs(binary - np.rint(binary)), initial=0))),
            'facility_balance': float(np.max(np.abs(pue * it - operation['direct'] - operation['grid'] - operation['discharge']))),
            'it_limit': float(max(0, np.max(it - frame.cap_it_mw.to_numpy()))),
        }
        if max(errors.values()) > 1e-4:
            raise RuntimeError(f'调度验收失败 {errors}')
        total = float(c @ solver_x) / scale
        equipment = servers * annual_equipment_per_unit
        lower_bound = (lp_lower_window * 1000 / scale if not is_mip else
                       float(model.getAttr(cp.COPT.Attr.BestBnd)) * 1000 / scale if is_mip else total)
        if lp_solver == 'highs' and np.all(c >= 0):
            lower_bound = max(lower_bound, lb[5] * annual_equipment_per_unit)
        solver_bound_available = np.isfinite(lower_bound) and 0 <= lower_bound <= total + 10
        if not solver_bound_available:
            lower_bound = lb[5] * annual_equipment_per_unit
        result.update(status='optimal' if status == cp.COPT.OPTIMAL else 'feasible_with_gap',
                      servers=servers, gpus=servers * fleet['gpus_per_unit'],
                      equipment_purchase_usd=servers * price, equipment_annual_usd=equipment,
                      energy_annual_usd=total - equipment, total_annual_usd=total,
                      total_annual_lower_bound_usd=min(lower_bound, total),
                      optimality_gap_annual_usd=max(0., total - lower_bound),
                      solver_bound_available=bool(solver_bound_available),
                      grid_mwh=float(operation['grid'].sum()), peak_it_mw=float(it.max()),
                      facility_mwh=float(pue @ it), **{key: float(x[i]) for i, key in enumerate(ASSETS)},
                      audit=errors)
    if lp_solver == 'copt':
        model.clear()
        env.close()
    return result, x


def city_group(task):
    args, scenario, fleet = task
    parameters, config = load_config(args.project)
    factors = parameters['emission_factor']
    price = config['cards']['server_price_usd'][scenario[0]]['value']
    schedule = make_schedule(args, scenario, fleet)
    for target in args.target or ('economic', 'alpha100'):
        alpha = None if target == 'economic' else 1.0
        for city in args.city or CITIES:
            parameters['emission_factor'] = factors[city]
            path = args.city_inputs / f'{city}-{scenario}.parquet'
            frame = pd.read_parquet(path)
            input_hash = sha256(path)
            name = f'{city}-{scenario}-{target}.json'
            start = None
            for stage in ('B', 'C'):
                if schedule.option_count == 0:
                    # 固定负荷下，基线台数已由两种资源的并发峰值决定。
                    result = json.loads((args.output / 'A' / name).read_text('utf-8'))
                    if result['input_sha256'] != input_hash or result['servers'] != fleet['baseline_units']:
                        raise ValueError(f'{name} 固定负荷基线的来源或设备台数不一致')
                    result.update(stage=stage, equivalent_to_fixed_baseline=True, whole_jobs=0, schedule_options=0)
                    if result['status'] == 'optimal':
                        result.update(total_annual_lower_bound_usd=result['total_annual_usd'], optimality_gap_annual_usd=0.)
                else:
                    if stage == 'C' and args.capacity_start and schedule.integer.any():
                        _, candidate = solve_with_start(frame, schedule, fleet, parameters, price, alpha,
                                                        stage, min(60, args.time_limit), start, seed_only=True)
                        if candidate is not None:
                            start = candidate
                    result, start = solve_with_start(frame, schedule, fleet, parameters, price, alpha, stage, args.time_limit, start)
                result.update(city=city, scenario=scenario, input_sha256=input_hash)
                atomic_json(args.output / stage / name, result)
                print(city, scenario, target, stage, result['status'], result.get('servers'), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('project', 'resources', 'reference', 'whole-jobs', 'city-inputs', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--time-limit', type=float, default=1800)
    parser.add_argument('--city', choices=CITIES, action='append')
    parser.add_argument('--scenario', choices=SCENARIOS, action='append')
    parser.add_argument('--target', choices=('economic', 'alpha100'), action='append')
    parser.add_argument('--capacity-start', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.resources / 'resource-manifest.json').read_text('utf-8'))
    for stage in ('B', 'C'):
        (args.output / stage).mkdir(parents=True, exist_ok=True)
    tasks = [(args, scenario, manifest[scenario]) for scenario in (args.scenario or SCENARIOS)]
    if args.workers == 1:
        for task in tasks:
            city_group(task)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(city_group, tasks))


if __name__ == '__main__':
    main()
