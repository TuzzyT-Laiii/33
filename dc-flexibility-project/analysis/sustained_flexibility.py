import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from flex_econ_copt import backlog_limit
from residual_flex_pilot import LP, solve_copt


def solve_event(frame, start, duration, solver='copt'):
    hours = frame.source_hour_index.to_numpy()
    finish = start + duration
    if finish > len(frame) or hours[finish - 1] - hours[start] != duration - 1:
        return None
    if hours[-1] < hours[finish - 1] + 48:
        return None
    stop = np.searchsorted(hours, hours[finish - 1] + 48, side='right')
    part = frame.iloc[start:stop]
    n = len(part)
    arrivals = [part.whole.to_numpy(float), part.split.to_numpy(float)]
    original = (part.rigid + part.unknown + part.whole + part.split).to_numpy(float)
    fixed = (part.rigid + part.unknown).to_numpy(float)
    capacity = float((frame.rigid + frame.unknown + frame.whole + frame.split).max())
    base = dict(start_hour=int(hours[start]), duration_h=duration, recovery_limit_h=48,
                baseline_task_peak_mw=capacity)
    upper_reduction = float((arrivals[0] + arrivals[1])[:duration].min())
    if upper_reduction <= 1e-10:
        return dict(base, reduction_mw=0., shifted_mwh=0., rebound_extra_mw=0.,
                    recovery_task_peak_mw=float(original[duration:].max()),
                    max_residual=0., status='zero_flexible_power')
    # 事件开始前按原时序执行；事件后最多 48 小时恢复，保持原有任务功率容量。
    lp = LP(n, 2)
    lp.count = 2 + 4 * n
    idx = np.arange(n)
    lb = np.zeros(lp.count)
    ub = np.full(lp.count, np.inf)
    ub[0] = upper_reduction
    xs = []
    for k, (arrival, deadline) in enumerate(zip(arrivals, [12, 48])):
        execution, backlog = 2 + 2 * k * n + idx, 2 + (2 * k + 1) * n + idx
        xs.append(execution)
        ub[backlog] = backlog_limit(arrival, deadline, part.source_hour_index)
        lp.block('eq', [(idx, execution, 1.), (idx, backlog, 1.),
                        (idx[1:], backlog[:-1], -1.)], arrival)
    lp.block('ub', [(idx, x, 1.) for x in xs], capacity - fixed)
    ev = np.arange(duration)
    lp.block('eq', [(ev, x[ev], 1.) for x in xs] + [(ev, 0, 1.)],
             (arrivals[0] + arrivals[1])[:duration])
    rec = np.arange(duration, n)
    lp.block('ub', [(rec - duration, x[rec], 1.) for x in xs] +
             [(rec - duration, 1, -1.)], (arrivals[0] + arrivals[1])[rec])
    aeq, beq, aub, bub = lp.matrices()
    objective = np.zeros(lp.count)
    objective[0] = -1.

    def optimize():
        if solver == 'copt':
            values, status = solve_copt(objective, aeq, beq, aub, bub, lb, ub)
            if values is None:
                raise RuntimeError(f'持续调节求解失败 {base} {status}')
            return values
        result = linprog(objective, A_eq=aeq, b_eq=beq, A_ub=aub, b_ub=bub,
                         bounds=np.column_stack([lb, ub]), method='highs',
                         options={'primal_feasibility_tolerance': 1e-9,
                                  'dual_feasibility_tolerance': 1e-9})
        if not result.success:
            raise RuntimeError(result.message)
        return result.x

    values = optimize()
    reduction = max(0., float(values[0]))
    # 在最大持续削减量下，再最小化补做时相对原时序增加的峰值。
    lb[0] = ub[0] = reduction
    objective[:] = 0.
    objective[1] = 1.
    values = optimize()
    residual = max(float(np.abs(aeq @ values - beq).max()),
                   float(np.maximum(aub @ values - bub, 0).max()),
                   float(np.maximum(lb - values, 0).max()),
                   float(np.maximum(values - ub, 0).max()))
    if residual > 1e-6:
        raise RuntimeError(f'持续调节约束残差超限 {residual}')
    scheduled = fixed + sum(values[x] for x in xs)
    return dict(base, reduction_mw=reduction, shifted_mwh=reduction * duration,
                rebound_extra_mw=float(values[1]),
                recovery_task_peak_mw=float(scheduled[duration:].max()),
                max_residual=residual, status='optimal')


def one(task):
    template, output, solver, stride = task
    frame = pd.read_parquet(template)
    rows = []
    started = time.monotonic()
    for duration in (1, 4, 12, 24):
        for start in range(0, len(frame), stride):
            row = solve_event(frame, start, duration, solver)
            if row is not None:
                rows.append(dict(scenario=template.stem, **row))
    result = pd.DataFrame(rows)
    result.to_parquet(output / f'{template.stem}.parquet', index=False)
    summary = dict(scenario=template.stem, events=len(result), seconds=time.monotonic() - started,
                   max_residual=float(result.max_residual.max()))
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--templates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=11)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--solver', choices=['copt', 'highs'], default='copt')
    parser.add_argument('--scenarios', default='')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted(args.templates.glob('*.parquet'))
    if args.scenarios:
        files = [p for p in files if p.stem in args.scenarios.split(',')]
    tasks = [(p, args.output, args.solver, args.stride) for p in files
             if not (args.output / p.name).exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(one, tasks))
    result = pd.concat([pd.read_parquet(args.output / p.name) for p in files], ignore_index=True)
    grouped = result.groupby(['scenario', 'duration_h'])
    summary = grouped.agg(events=('reduction_mw', 'size'),
                          reduction_min_mw=('reduction_mw', 'min'),
                          reduction_p10_mw=('reduction_mw', lambda x: x.quantile(.1)),
                          reduction_median_mw=('reduction_mw', 'median'),
                          reduction_max_mw=('reduction_mw', 'max'),
                          shifted_median_mwh=('shifted_mwh', 'median'),
                          rebound_median_mw=('rebound_extra_mw', 'median'),
                          recovery_task_peak_median_mw=('recovery_task_peak_mw', 'median'),
                          zero_share=('reduction_mw', lambda x: (x < 1e-8).mean()),
                          max_residual=('max_residual', 'max'))
    summary.to_csv(args.output / 'summary.csv')
    print(summary.to_string())


if __name__ == '__main__':
    main()
