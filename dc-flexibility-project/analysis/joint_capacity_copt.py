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


def make_schedule(args, scenario, fleet):
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
    fixed = all_work - split
    if not np.isfinite(fixed).all() or not np.isfinite(split).all() or min(fixed.min(), split.min()) < -1e-7:
        raise ValueError(f'{scenario} 的固定或可拆分工作量无效')
    fixed = fixed.clip(min=0)
    split = split.clip(min=0)

    # 以来源小时为批次，按相同比例同时转移 CPU、GPU 和任务用电。
    origins = np.flatnonzero(np.any(split > 0, axis=1))
    rows = np.searchsorted(hours, hours[origins, None] + np.arange(49))
    valid = (rows < len(hours)) & (hours[rows.clip(max=len(hours) - 1)] == hours[origins, None] + np.arange(49))
    release, offsets = np.nonzero(valid)
    destinations = rows[release, offsets]
    option_count = len(destinations)
    block_count = len(origins)
    blocks = [release]
    slots = [destinations]
    columns = [np.arange(option_count)]
    values = [split[origins[release]]]
    integer = [np.zeros(option_count, dtype=bool)]

    whole_count = 0
    path = args.whole_jobs / f'{scenario}.parquet'
    if path.is_file():
        jobs = pd.read_parquet(path)
        index = pd.Index(hours)
        for _, job in jobs.groupby('job', sort=False):
            job_hours = job.source_hour_index.to_numpy(np.int64)
            contribution = job[list(METRICS)].to_numpy(float)
            if np.any(np.diff(job_hours) != 1) or not np.isfinite(contribution).all() or np.any(contribution < 0):
                raise ValueError(f'{scenario} 整作业时间或资源无效')
            original_rows = index.get_indexer(job_hours)
            if (original_rows < 0).any():
                raise ValueError(f'{scenario} 整作业占用未观测时段')
            targets = index.get_indexer((job_hours[None, :] + np.arange(13)[:, None]).ravel()).reshape(13, -1)
            targets = targets[(targets >= 0).all(axis=1)]
            if len(targets) <= 1:
                continue
            fixed[original_rows] -= contribution
            options = np.arange(option_count, option_count + len(targets))
            blocks.append(np.full(len(targets), block_count))
            slots.append(targets.ravel())
            columns.append(np.repeat(options, len(job_hours)))
            values.append(np.tile(contribution, (len(targets), 1)))
            integer.append(np.ones(len(targets), dtype=bool))
            option_count += len(targets)
            block_count += 1
            whole_count += 1
    if fixed.min() < -1e-7:
        raise ValueError(f'{scenario} 的整作业资源超出原始需求')
    return Schedule(hours, fixed.clip(min=0), all_work, np.concatenate(blocks), np.concatenate(slots),
                    np.concatenate(columns), np.concatenate(values), np.concatenate(integer),
                    option_count, block_count, whole_count)


def solve_with_start(frame, workload, fleet, p, price, alpha, stage, time_limit, warm_start, *, seed_only=False):
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
    ub[5] = fleet['baseline_units']
    lower_units = max(workload.fixed[:, :2].max(), workload.original[:, :2].mean(axis=0).max())
    lb[5] = ub[5] if stage == 'B' else max(1, np.ceil(lower_units - 1e-8))
    if stage == 'C':
        vtype[5] = cp.COPT.INTEGER
    ub[first_option:] = 1
    vtype[first_option:][workload.integer] = cp.COPT.BINARY
    is_mip = stage == 'C' or workload.integer.any()
    first = np.unique(workload.blocks, return_index=True)[1]
    if seed_only:
        if warm_start is None:
            seed = np.zeros(workload.option_count)
            seed[first] = 1
        else:
            seed = np.rint(warm_start[first_option:])
        whole = first_option + np.flatnonzero(workload.integer)
        lb[whole] = ub[whole] = seed[workload.integer]
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
        add(row + idx, 5, -1)
        add(row + slots, options, workload.values[:, resource])
        bounds(-inf, -fixed[:, resource])

    add(row + idx, 5, idle)
    add(row + slots, options, powers)
    bounds(-inf, frame.cap_it_mw.to_numpy() - fixed[:, 2])

    for key, capacity in (('charge', 2), ('discharge', 2), ('soc', 3), ('grid', 4)):
        add(row + idx, off[key] + idx, 1)
        add(row + idx, capacity, -1)
        bounds(-inf, 0)

    add(row + workload.blocks, first_option + np.arange(workload.option_count), 1)
    bounds(1, 1, workload.block_count)

    if alpha == 1:
        ub[off['grid'] + idx] = 0

    matrix = sparse.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(row, count))
    matrix.eliminate_zeros()
    env = cp.Envr()
    model = env.createModel('joint_capacity')
    for key, value in {'Logging': 0, 'Threads': 1, 'LpMethod': 2, 'Crossover': 1,
                       'FeasTol': 1e-8, 'DualTol': 1e-8, 'RelGap': MIP_RELATIVE_GAP, 'TimeLimit': time_limit}.items():
        model.setParam(key, value)
    model.loadMatrix(c / 1000, matrix, np.concatenate(lhs), np.concatenate(rhs), lb, ub, vtype)
    if is_mip:
        variables = model.getVars()
        if warm_start is not None:
            model.setMipStart(variables, warm_start.tolist())
        else:
            # 原时序是可行调度。先提供整数变量，供求解器补齐供能决策。
            initial = np.zeros(workload.option_count)
            initial[first] = 1
            selected = np.flatnonzero(workload.integer)
            indices = np.r_[5, first_option + selected]
            model.setMipStart([variables[int(i)] for i in indices],
                              np.r_[fleet['baseline_units'], initial[selected]].tolist())
        model.loadMipStart()
    model.solve()
    status = model.status
    result = {'stage': stage, 'alpha': alpha, 'hours': n, 'status_code': int(status),
              'whole_jobs': workload.whole_jobs, 'schedule_options': workload.option_count,
              'requested_mip_relative_gap': MIP_RELATIVE_GAP if is_mip else None,
              'solve_seconds': time.monotonic() - started,
              'status': 'infeasible' if status == cp.COPT.INFEASIBLE else 'not_solved'}
    x = None
    if status == cp.COPT.OPTIMAL or (is_mip and model.getAttr(cp.COPT.Attr.HasMipSol)):
        x = np.asarray(model.getValues(), dtype=float)
        servers = round(x[5])
        dispatch = fixed.copy()
        np.add.at(dispatch, slots, workload.values * x[options, None])
        operation = {key: x[off[key] + idx] for key in OPERATIONS}
        it = dispatch[:, 2] + servers * idle
        block_sum = np.bincount(workload.blocks, weights=x[first_option:], minlength=workload.block_count)
        binary = x[first_option:][workload.integer]
        errors = {
            'resources': float(max(0, dispatch[:, :2].max() - servers)),
            'work_conservation': float(np.max(np.abs(dispatch.sum(axis=0) - workload.original.sum(axis=0)))),
            'batches': float(np.max(np.abs(block_sum - 1), initial=0)),
            'integrality': float(max(abs(servers - x[5]), np.max(np.abs(binary - np.rint(binary)), initial=0))),
            'facility_balance': float(np.max(np.abs(pue * it - operation['direct'] - operation['grid'] - operation['discharge']))),
            'it_limit': float(max(0, np.max(it - frame.cap_it_mw.to_numpy()))),
        }
        if max(errors.values()) > 1e-4:
            raise RuntimeError(f'调度验收失败 {errors}')
        total = float(c @ x) / scale
        equipment = servers * annual_equipment_per_unit
        lower_bound = float(model.getAttr(cp.COPT.Attr.BestBnd)) * 1000 / scale if is_mip else total
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
