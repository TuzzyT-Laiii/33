import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse

import resource_inference_capacity as hourly

CONTEXT = {}


def prepare(args):
    hourly.prepare(args)
    manifest = json.loads((args.joint / 'inputs/resources/resource-manifest.json').read_text())
    source = pd.read_csv(args.prior)
    if len(source) != 162 or source.max_residual.max() > 1e-6:
        raise ValueError('已有 CPU/GPU 耦合服务结果不完整')
    CONTEXT.update(args=args, manifest=manifest, prior=source, schedule_cache={},
                   model_args=SimpleNamespace(reference=args.joint / 'inputs/reference',
                                              whole_jobs=args.capacity / 'inputs/exact-jobs',
                                              hourly_whole=True))


def one(task):
    import coptpy as cp
    from flex_econ_copt import backlog_limit
    from joint_capacity_copt import make_resource_schedule
    from minimum_server_capacity import round_whole, resource_dispatch
    from residual_flex_pilot import LP
    scenario, phase, share = task
    args = CONTEXT['args']
    target = args.output / f'{scenario}-{phase}-{share}.json'
    if target.exists():
        return json.loads(target.read_text())
    started = time.monotonic()
    print(json.dumps(dict(task=task, stage='build', utc=datetime.now(timezone.utc).isoformat())), flush=True)
    fleet = CONTEXT['manifest'][scenario]
    state = hourly.CONTEXT['scenarios'][scenario]
    cache = CONTEXT['schedule_cache']
    if scenario not in cache:
        cache.clear()
        schedule = make_resource_schedule(CONTEXT['model_args'], scenario, fleet)
        certificates = args.capacity / 'results/capacity'
        certificate = json.loads((certificates / f'{scenario}-w12-s48.json').read_text())
        source_scenario = certificate['reused_resource_model'] if 'reused_resource_model' in certificate else scenario
        if source_scenario != scenario:
            base_certificate = certificates / f'{source_scenario}-w12-s48.json'
            if sha256(base_certificate.read_bytes()).hexdigest() != certificate['base_capacity_sha256']:
                raise ValueError('派生机群的调度证书来源校验不一致')
        with np.load(certificates / f'{source_scenario}-w12-s48.npz') as saved:
            certified_assignment = saved['assignment']
        if certified_assignment.shape != (schedule.option_count,):
            raise ValueError('原完整作业方案与当前选项数量不一致')
        np.testing.assert_allclose(resource_dispatch(schedule, certified_assignment) / fleet['baseline_units'],
                                   state['certified'], rtol=0, atol=1e-7)
        cache[scenario] = (schedule, certified_assignment)
    schedule, certified_assignment = cache[scenario]
    np.testing.assert_array_equal(schedule.hours, state['hours'])
    n = len(schedule.hours)
    position = schedule.hours - schedule.hours[0] - phase
    mask = (position >= 0) & (position < len(hourly.CONTEXT['requests']))
    shape = np.zeros(n)
    shape[mask] = hourly.CONTEXT['requests'][position[mask].astype(int)]
    idle = float((1. - schedule.original[mask, 1] / fleet['baseline_units']).sum())
    demand = shape / shape.sum() * idle
    prior = CONTEXT['prior'].query('scenario == @scenario and phase_hours == @phase and offline_share == @share')
    if len(prior) != 3:
        raise ValueError('吞吐量敏感性配对不完整')
    best_lower, best_upper = prior.feasible_share_of_idle.iloc[0], prior.upper_share_of_idle.iloc[0]
    idx = np.arange(n)
    first = 1 + 2 * n
    lp = LP(n, 1)
    lp.count = first + schedule.option_count
    lower, upper = np.zeros(lp.count), np.full(lp.count, cp.COPT.INFINITY)
    upper[0] = best_upper + 1e-8
    upper[first:] = schedule.block_sizes[schedule.blocks]
    objective = np.zeros(lp.count)
    objective[0] = -1.
    execution, backlog = 1 + idx, 1 + n + idx
    online, offline = demand * (1 - share), demand * share
    lp.block('eq', [(idx, execution, 1.), (idx, backlog, 1.),
                    (idx[1:], backlog[:-1], -1.), (idx, 0, -offline)], np.zeros(n))
    lp.block('ub', [(idx, backlog, 1.),
                    (idx, 0, -backlog_limit(offline, 24, schedule.hours))], np.zeros(n))
    lp.block('eq', [(schedule.blocks, first + np.arange(schedule.option_count), 1.)], schedule.block_sizes)
    for k, coefficient in enumerate([state['host_ratio'], 1.]):
        lp.block('ub', [(schedule.slots, first + schedule.columns,
                         schedule.values[:, k] / fleet['baseline_units']),
                        (idx, execution, coefficient), (idx, 0, coefficient * online)],
                 1. - schedule.fixed[:, k] / fleet['baseline_units'])
    aeq, beq, aub, bub = lp.matrices()
    matrix = sparse.vstack([aeq, aub]).tocsc()
    left, right = np.r_[beq, np.full(len(bub), -cp.COPT.INFINITY)], np.r_[beq, bub]
    env = cp.Envr()
    model = env.createModel('independent-inference-whole-jobs')

    def configure(mip=False):
        # clear 会恢复默认参数，每次重建后都重新指定线程数与精度。
        for name, value in [('Logging', 1), ('Threads', 1), ('LpMethod', 2),
                             ('Crossover', int(mip)), ('FeasTol', 1e-9), ('DualTol', 1e-9),
                             ('IntTol', 1e-8), ('TimeLimit', args.mip_seconds if mip else 1800),
                             ('RelGap', 0.), ('AbsGap', 1e-7)]:
            model.setParam(name, value)

    configure()
    model.loadMatrix(objective, matrix, left, right, lower, upper)
    print(json.dumps(dict(task=task, stage='relaxation', variables=lp.count, rows=matrix.shape[0],
                          elapsed=time.monotonic() - started)), flush=True)
    model.solve()
    if model.status != cp.COPT.OPTIMAL:
        raise RuntimeError(f'完整作业模式 LP 未达到最优 {task} {model.status}')
    fractional = np.asarray(model.getValues())
    activity = matrix @ fractional
    lp_residual = max(float(np.maximum(left - activity, 0).max()),
                      float(np.maximum(activity - right, 0).max()),
                      float(np.maximum(lower - fractional, 0).max()),
                      float(np.maximum(fractional - upper, 0).max()))
    if lp_residual > 1e-6:
        raise RuntimeError(f'完整作业松弛约束残差超限 {task} {lp_residual}')
    relaxation_bound = -float(model.getAttr(cp.COPT.Attr.ObjBound))
    if not np.isfinite(relaxation_bound) or relaxation_bound < fractional[0] - 1e-6:
        raise RuntimeError(f'完整作业松弛目标界限无效 {task} {relaxation_bound}')
    best_upper = min(best_upper, max(float(fractional[0]), relaxation_bound))
    external = (fractional[execution] + fractional[0] * online)[:, None]
    occupied = fleet['baseline_units'] * external * np.array([state['host_ratio'], 1.])
    rounded, _ = round_whole(replace(schedule, fixed=schedule.fixed + occupied), fractional[first:])
    whole_columns = first + np.flatnonzero(schedule.integer)
    fixed_lower, fixed_upper = lower.copy(), upper.copy()
    fixed_lower[whole_columns] = fixed_upper[whole_columns] = rounded[schedule.integer]
    model.clear()
    configure()
    model.loadMatrix(objective, matrix, left, right, fixed_lower, fixed_upper)
    print(json.dumps(dict(task=task, stage='rounded', relaxation_upper=best_upper,
                          elapsed=time.monotonic() - started)), flush=True)
    model.solve()
    feasible = None
    if model.status == cp.COPT.OPTIMAL:
        feasible = np.asarray(model.getValues())
    elif model.status != cp.COPT.INFEASIBLE:
        raise RuntimeError(f'完整作业取整后的调度未求解 {task} {model.status}')
    rounded_value = float(feasible[0]) if feasible is not None else None
    fixed_lower[whole_columns] = fixed_upper[whole_columns] = certified_assignment[schedule.integer]
    model.clear()
    configure()
    model.loadMatrix(objective, matrix, left, right, fixed_lower, fixed_upper)
    print(json.dumps(dict(task=task, stage='certified', elapsed=time.monotonic() - started)), flush=True)
    model.solve()
    if model.status != cp.COPT.OPTIMAL:
        raise RuntimeError(f'原完整作业方案未能重新优化服务量 {task} {model.status}')
    certified_feasible = np.asarray(model.getValues())
    if certified_feasible[0] < best_lower - 1e-6:
        raise RuntimeError(f'完整作业模型未包含已有可行服务量 {task}')
    if feasible is None or certified_feasible[0] > feasible[0]:
        feasible = certified_feasible
    mip_status = None
    if feasible is None or best_upper - feasible[0] > 1e-7:
        model.clear()
        configure(mip=True)
        kinds = np.full(lp.count, cp.COPT.CONTINUOUS)
        kinds[whole_columns] = cp.COPT.INTEGER
        lower[0] = best_lower - 1e-8
        model.loadMatrix(objective, matrix, left, right, lower, upper, kinds)
        if feasible is not None and feasible[0] >= lower[0]:
            model.setMipStart(model.getVars(), feasible.tolist())
            model.loadMipStart()
        print(json.dumps(dict(task=task, stage='integer', rounded_value=rounded_value,
                              elapsed=time.monotonic() - started)), flush=True)
        model.solve()
        mip_status = int(model.status)
        if model.status == cp.COPT.INFEASIBLE:
            raise RuntimeError(f'完整模型排除了已核验的可行服务量 {task}')
        if model.status not in (cp.COPT.OPTIMAL, cp.COPT.TIMEOUT):
            raise RuntimeError(f'完整作业整数求解异常结束 {task} {model.status}')
        if model.getAttr(cp.COPT.Attr.HasMipSol):
            candidate = np.asarray(model.getValues())
            if feasible is None or candidate[0] > feasible[0]:
                feasible = candidate
            bound = -float(model.getAttr(cp.COPT.Attr.BestBnd))
            if np.isfinite(bound):
                best_upper = min(best_upper, bound)
    residual = 0.
    if feasible is not None:
        activity = matrix @ feasible
        residual = max(float(np.maximum(left - activity, 0).max()),
                       float(np.maximum(activity - right, 0).max()),
                       float(np.maximum(-feasible, 0).max()),
                       float(np.maximum(feasible - upper, 0).max()),
                       float(np.abs(feasible[whole_columns] - np.rint(feasible[whole_columns])).max()))
        if residual > 1e-6:
            raise RuntimeError(f'完整作业与服务调度残差超限 {task} {residual}')
        best_lower = max(best_lower, float(feasible[0]))
        np.savez_compressed(target.with_suffix('.npz'), assignment=feasible[first:],
                            external_offline=feasible[execution], external_scale=feasible[0])
    if best_lower > best_upper + 1e-6:
        raise RuntimeError('服务容量上下界颠倒')
    model.dispose()
    env.dispose()
    row = dict(scenario=scenario, phase_hours=phase, offline_share=share,
               feasible_share_of_idle=float(best_lower), upper_share_of_idle=float(max(best_lower, best_upper)),
               rounded_share=rounded_value, max_residual=max(residual, lp_residual), seconds=time.monotonic() - started,
               options=schedule.option_count, whole_groups=schedule.whole_jobs,
               capacity_proven=bool(best_upper - best_lower < 1e-6), mip_status=mip_status,
               solver_threads=1, solver_feasibility_tolerance=1e-9)
    target.write_text(json.dumps(row, allow_nan=False))
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    for name in ['project', 'joint', 'capacity', 'residual-code', 'requests', 'prior', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--mip-seconds', type=int, default=600)
    parser.add_argument('--scenarios', default='')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare(args)
    scenarios = args.scenarios.split(',') if args.scenarios else [s for s in hourly.CONTEXT['scenarios'] if not s.endswith('INF')]
    tasks = [(s, phase, share) for s in scenarios for phase in [0, 24, 168] for share in [0., .28]]
    if args.limit:
        tasks = tasks[:args.limit]
    remaining = iter(tasks)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        pending = {}

        def submit():
            task = next(remaining, None)
            if task is not None:
                pending[pool.submit(one, task)] = task

        for _ in range(args.workers):
            submit()
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                pending.pop(future)
                rows.append(future.result())
                submit()
    if args.limit or args.scenarios:
        return
    result = CONTEXT['prior'].copy()
    for row in rows:
        mask = ((result.scenario == row['scenario']) & (result.phase_hours == row['phase_hours']) &
                (result.offline_share == row['offline_share']))
        result.loc[mask, 'feasible_standard_requests'] *= row['feasible_share_of_idle'] / result.loc[mask, 'feasible_share_of_idle']
        result.loc[mask, 'feasible_trace_multiplier'] *= row['feasible_share_of_idle'] / result.loc[mask, 'feasible_share_of_idle']
        result.loc[mask, 'upper_standard_requests'] *= row['upper_share_of_idle'] / result.loc[mask, 'upper_share_of_idle']
        for key in ['feasible_share_of_idle', 'upper_share_of_idle', 'capacity_proven']:
            result.loc[mask, key] = row[key]
        result.loc[mask, 'max_residual'] = np.maximum(result.loc[mask, 'max_residual'], row['max_residual'])
    result.to_csv(args.output / 'gpu-service-capacity.csv', index=False)


if __name__ == '__main__':
    main()
