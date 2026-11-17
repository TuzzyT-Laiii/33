import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

CONTEXT = {}
CLASSES = ('whole', 'split', 'off')
DEADLINES = (12, 48, 24)
SUPPLY = ('grid', 'direct', 'charge', 'discharge', 'soc', 'curtail')


def prepare(args):
    sys.path[:0] = [str(args.residual_code), str(args.project / 'analysis')]
    import residual_flex_global as g
    g.initialize(args.config, args.project, args.asi, 120., [.5, .75], args.max_table)
    city = pd.read_csv(args.inputs / 'city-grid-points.csv')
    meta = g.CONTEXT['meta']
    valid = [key for key in meta if (args.inputs / 'grids' / f'{key}.npz').exists()]
    latitude = np.array([meta[key]['lat'] for key in valid])
    longitude = np.array([meta[key]['lon'] for key in valid])
    selected = {}
    for row in city.itertuples():
        delta = (longitude - row.lon + 180) % 360 - 180
        distance = (latitude - row.lat) ** 2 + (delta * np.cos(np.deg2rad(row.lat))) ** 2
        selected[row.city] = valid[int(np.argmin(distance))]
    CONTEXT.update(g=g, selected=selected, output=args.output, horizon=args.horizon)


def forecast_indices(hours, current, count, perfect):
    future = np.arange(current, min(len(hours), current + count))
    if perfect:
        return future
    if current == 0:
        return np.zeros(len(future), dtype=int)
    target = hours[current] - 24 + (hours[future] - hours[current]) % 24
    previous = np.searchsorted(hours[:current], target, side='right') - 1
    previous = np.maximum(previous, 0)
    previous[0] = current
    if np.any(previous[1:] >= current):
        raise RuntimeError('预测使用了尚未发生的观测')
    return previous


def asap_path(times, fixed, arrivals, queues, capacity, final_hour):
    pending = [[list(item) for item in queue] for queue in queues]
    executed = np.zeros_like(arrivals)
    for h, now in enumerate(times):
        for k, deadline in enumerate(DEADLINES):
            pending[k].append([min(now + deadline, final_hour), float(arrivals[h, k]), False])
        available = max(0., capacity - fixed[h])
        candidates = sorted((item[0], k, j) for k in range(3) for j, item in enumerate(pending[k]) if item[1] > 1e-10)
        for _, k, j in candidates:
            amount = min(available, pending[k][j][1])
            pending[k][j][1] -= amount
            executed[h, k] += amount
            available -= amount
            if available < 1e-10:
                break
    return executed


def backlog_allowance(times, next_times, arrivals, queue, deadline):
    old = sum(item[1] for item in queue)
    cumulative = np.r_[0., np.cumsum(arrivals)]
    due_index = np.searchsorted(times, next_times - deadline, side='left')
    due_index = np.minimum(due_index, np.arange(1, len(times) + 1))
    old_due = np.array([sum(item[1] for item in queue if item[0] < hour) for hour in next_times])
    return old + cumulative[1:] - old_due - cumulative[due_index]


def step(frame, p, assets, queues, current, soc, peak, mode, perfect, env):
    import coptpy as cp
    from residual_flex_pilot import LP
    hours = frame.source_hour_index.to_numpy()
    future = np.arange(current, min(len(frame), current + CONTEXT['horizon']))
    predicted = forecast_indices(hours, current, CONTEXT['horizon'], perfect)
    times, n = hours[future], len(future)
    next_times = hours[np.minimum(future + 1, len(hours) - 1)].copy()
    next_times[future == len(hours) - 1] = hours[-1] + 1
    data = frame.iloc[predicted]
    arrivals = data[list(CLASSES)].to_numpy(float)
    fixed_task = (data.rigid + data.unknown + data.on).to_numpy(float)
    fixed_it = data.base.to_numpy() + fixed_task
    pue = data.pue.to_numpy(float)
    capacity = float(frame.attrs['capacity'])
    immediate = asap_path(times, fixed_task, arrivals, queues, capacity, hours[-1])
    terminal_backlog = (np.array([sum(item[1] for item in queue) for queue in queues]) +
                        arrivals.sum(axis=0) - immediate.sum(axis=0))
    if terminal_backlog.min() < -1e-7:
        raise RuntimeError('窗口末参考积压出现负值')
    terminal_backlog = np.maximum(terminal_backlog, 0.)
    lp = LP(n, 1)
    names = [prefix + key for prefix in ('x_', 'b_', 'late_') for key in CLASSES] + list(SUPPLY)
    lp.count = 1 + n * len(names)
    idx = np.arange(n)
    off = {key: 1 + i * n + idx for i, key in enumerate(names)}
    lower, upper = np.zeros(lp.count), np.full(lp.count, np.inf)
    lower[0], upper[0] = peak, p['grid_max_mw']
    objective = np.zeros(lp.count)
    for k, (key, deadline) in enumerate(zip(CLASSES, DEADLINES)):
        old = sum(item[1] for item in queues[k])
        rhs = arrivals[:, k].copy()
        rhs[0] += old
        lp.block('eq', [(idx, off['x_' + key], 1.), (idx, off['b_' + key], 1.),
                        (idx[1:], off['b_' + key][:-1], -1.)], rhs)
        allowed = backlog_allowance(times, next_times, arrivals[:, k], queues[k], deadline)
        if future[-1] == len(frame) - 1:
            allowed[-1] = 0.
        lp.block('ub', [(idx, off['b_' + key], 1.), (idx, off['late_' + key], -1.)], allowed)
        # 顺序执行已固定全部任务变量，窗口末积压由守恒式确定；协同模式需显式约束。
        if mode == 'joint':
            upper[off['b_' + key][-1]] = terminal_backlog[k]
        objective[off['late_' + key]] = 1.
    lp.block('ub', [(idx, off['x_' + key], 1.) for key in CLASSES], capacity - fixed_task)
    if mode == 'seq':
        for k, key in enumerate(CLASSES):
            lower[off['x_' + key]] = upper[off['x_' + key]] = immediate[:, k]
    generation = data.cf_pv.to_numpy() * assets['pv_mw'] + data.cf_wind.to_numpy() * assets['wind_mw']
    lp.block('eq', [(idx, off[key], 1.) for key in ('direct', 'charge', 'curtail')], generation)
    lp.block('eq', [(idx, off[key], 1.) for key in ('direct', 'grid', 'discharge')] +
             [(idx, off['x_' + key], -pue) for key in CLASSES], pue * fixed_it)
    eta = np.sqrt(p['battery_roundtrip_efficiency'])
    rhs = np.zeros(n)
    rhs[0] = soc
    lp.block('eq', [(idx, off['soc'], 1.), (idx[1:], off['soc'][:-1], -1.),
                    (idx, off['charge'], -eta), (idx, off['discharge'], 1 / eta)], rhs)
    upper[off['charge']] = upper[off['discharge']] = assets['battery_mw']
    upper[off['soc']] = assets['battery_mwh']
    lp.block('ub', [(idx, off['grid'], 1.), (idx, 0, -1.)], np.zeros(n))
    aeq, beq, aub, bub = lp.matrices()
    matrix = sparse.vstack([aeq, aub]).tocsc()
    left = np.r_[beq, np.full(len(bub), -cp.COPT.INFINITY)]
    right = np.r_[beq, bub]
    model = env.createModel('rolling-forecast')
    for key, value in [('Logging', 0), ('Threads', 1), ('LpMethod', 2), ('Crossover', 1),
                       ('FeasTol', 1e-9), ('DualTol', 1e-9), ('TimeLimit', 300)]:
        model.setParam(key, value)
    model.loadMatrix(objective, matrix, left, right, lower, np.minimum(upper, cp.COPT.INFINITY))
    model.solve()
    if model.status != cp.COPT.OPTIMAL:
        raise RuntimeError(f'滚动期限问题失败 {current} {mode} {model.status}')
    minimum_late = float(model.objval)
    variables = model.getVars()
    late_columns = np.flatnonzero(objective)
    if minimum_late < 1e-8:
        model.setInfo('UB', [variables[j] for j in late_columns], 0.)
    else:
        model.addConstr(cp.quicksum(variables[j] for j in late_columns) <= minimum_late + 1e-8)
    objective[:] = 0.
    # 电网接入可按原单价增配；实际费用按整个观测窗口实现的最高进口功率计算。
    objective[0] = p['annual_usd_per_mw']['grid_capacity'] * len(frame) / 8760
    objective[off['grid']] = p['grid_price_usd_mwh'] + p['carbon_price_usd_t'] * p['ef']
    objective[off['discharge']] = p['battery_throughput_usd_mwh']
    model.setInfo('Obj', variables, objective.tolist())
    model.solve()
    if model.status != cp.COPT.OPTIMAL:
        raise RuntimeError(f'滚动供能问题失败 {current} {mode} {model.status}')
    values = np.asarray(model.getValues(), float)
    residual = max(float(np.abs(aeq @ values - beq).max()),
                   float(np.maximum(aub @ values - bub, 0).max()),
                   float(np.maximum(lower - values, 0).max()),
                   float(np.maximum(values - upper, 0).max()))
    if residual > 2e-6:
        raise RuntimeError(f'滚动约束残差超限 {residual}')
    if mode == 'seq':
        terminal_residual = float(np.max(np.abs(
            [values[off['b_' + key][-1]] - terminal_backlog[k] for k, key in enumerate(CLASSES)])))
        if terminal_residual > 2e-6:
            raise RuntimeError(f'顺序调度的窗口末工作量不守恒 {terminal_residual}')
        residual = max(residual, terminal_residual)
    result = {key: float(values[column[0]]) for key, column in off.items()}
    result.update(max_residual=residual, forecast_index=predicted[1:].tolist(),
                  future_index=future[1:].tolist())
    del variables
    model.dispose()
    return result


def replay(frame, p, assets, mode, perfect, env):
    queues = [deque() for _ in CLASSES]
    soc, peak = 0., assets['grid_cap_mw']
    total_cost = total_grid = total_discharge = max_residual = 0.
    missed = np.zeros(3)
    executed_total = np.zeros(3)
    forecast_abs_error = np.zeros(3)
    forecast_sum = np.zeros(3)
    max_backlog = 0.
    for t, now in enumerate(frame.source_hour_index.to_numpy()):
        # 缺口期间没有执行时隙，已过期限的积压须在本小时补做之前计入逾期。
        for k, queue in enumerate(queues):
            for item in queue:
                if item[0] < now and not item[2]:
                    missed[k] += item[1]
                    item[2] = True
        action = step(frame, p, assets, queues, t, soc, peak, mode, perfect, env)
        for k, (key, deadline) in enumerate(zip(CLASSES, DEADLINES)):
            amount = float(frame.iloc[t][key])
            if amount > 0:
                queues[k].append([min(now + deadline, frame.source_hour_index.iloc[-1]), amount, False])
            execute = action['x_' + key]
            executed_total[k] += execute
            while execute > 1e-8 and queues[k]:
                amount = min(execute, queues[k][0][1])
                queues[k][0][1] -= amount
                execute -= amount
                if queues[k][0][1] < 1e-8:
                    queues[k].popleft()
            if execute > 1e-6:
                raise RuntimeError('实际执行量超出已到达任务')
            for item in queues[k]:
                if item[0] <= now and not item[2]:
                    missed[k] += item[1]
                    item[2] = True
        soc = action['soc']
        peak = max(peak, action['grid'])
        total_grid += action['grid']
        total_discharge += action['discharge']
        max_residual = max(max_residual, action['max_residual'])
        max_backlog = max(max_backlog, sum(item[1] for queue in queues for item in queue))
        if action['future_index']:
            actual = frame.iloc[action['future_index']]
            prediction = frame.iloc[action['forecast_index']]
            observed = np.column_stack([actual[['rigid', 'unknown', 'whole', 'split', 'on', 'off']].sum(axis=1),
                                        actual.cf_pv, actual.cf_wind])
            estimated = np.column_stack([prediction[['rigid', 'unknown', 'whole', 'split', 'on', 'off']].sum(axis=1),
                                         prediction.cf_pv, prediction.cf_wind])
            forecast_abs_error += np.abs(observed - estimated).sum(axis=0)
            forecast_sum += np.abs(observed).sum(axis=0)
    unserved = np.array([sum(item[1] for item in queue) for queue in queues])
    conservation = float(np.abs(executed_total + unserved - frame[list(CLASSES)].sum().to_numpy()).max())
    if conservation > 1e-5:
        raise RuntimeError(f'实际任务量不守恒 {conservation}')
    scale = len(frame) / 8760
    capital = sum(p['annual_usd_per_mw'][key] * assets[field] * scale for key, field in
                  [('pv', 'pv_mw'), ('wind', 'wind_mw'), ('battery_power', 'battery_mw'), ('battery_energy', 'battery_mwh')])
    total_cost = capital + peak * p['annual_usd_per_mw']['grid_capacity'] * scale + (
        p['grid_price_usd_mwh'] + p['carbon_price_usd_t'] * p['ef']) * total_grid + p['battery_throughput_usd_mwh'] * total_discharge
    return dict(mode=mode, forecast='perfect' if perfect else 'previous_day', cost_usd=total_cost,
                grid_mwh=total_grid, discharge_mwh=total_discharge, grid_peak_mw=peak,
                extra_grid_capacity_mw=max(0., peak - assets['grid_cap_mw']),
                deadline_missed_mwh=missed.tolist(), unserved_mwh=unserved.tolist(),
                executed_mwh=executed_total.tolist(), conservation_residual_mwh=conservation,
                max_backlog_mwh=max_backlog, max_residual=max_residual,
                forecast_wape=np.divide(forecast_abs_error, forecast_sum, out=np.zeros(3), where=forecast_sum > 0).tolist())


def one(task):
    import coptpy as cp
    from residual_flex_pilot import solve
    city, scenario, level = task
    path = CONTEXT['output'] / f'{city}-{scenario}-{level}.json'
    if path.exists():
        return json.loads(path.read_text())
    started = time.monotonic()
    g = CONTEXT['g']
    grid = CONTEXT['selected'][city]
    frame = g.build_frame(grid, scenario)
    frame, on_raw, off_raw, _, _ = g.align(frame, scenario)
    own_task = frame[['rigid', 'unknown', 'whole', 'split']].sum(axis=1).to_numpy()
    intensity = level * g.CONTEXT['max_intensity'][scenario]
    factor = intensity * (own_task.max() - own_task.mean()) / (on_raw.mean() + off_raw.mean())
    frame['on'], frame['off'] = factor * on_raw, factor * off_raw
    frame.attrs['capacity'] = float(own_task.max())
    p = {**g.CONTEXT['parameters'], 'emission_factor': {grid: g.CONTEXT['meta'][grid]['carbon_t_mwh']}}
    p['ef'] = p['emission_factor'][grid]
    first, schedule = solve(frame, p, grid, frame.on.to_numpy(), frame.off.to_numpy(), fee=120., mode='asap', solver='copt')
    if schedule is None:
        raise RuntimeError(f'顺序调度基准不可行 {task} {first}')
    cases = []
    env = cp.Envr()
    for mode in ('seq', 'joint'):
        assets, _ = solve(frame, p, grid, frame.on.to_numpy(), frame.off.to_numpy(), fee=120., mode='cost',
                          fixed_x=schedule if mode == 'seq' else None, solver='copt')
        if assets['status'] != 'optimal':
            raise RuntimeError(f'供能资产规划不可行 {task} {assets}')
        for perfect in (False, True):
            cases.append(replay(frame, p, assets, mode, perfect, env))
    env.dispose()
    e0 = float(np.dot(frame.pue, frame.base + own_task))
    actual = {row['mode']: row for row in cases if row['forecast'] == 'previous_day'}
    oracle = {row['mode']: row for row in cases if row['forecast'] == 'perfect'}
    complete = all(sum(item['unserved_mwh']) < 1e-5 for item in actual.values())
    deadlines_met = complete and all(sum(item['deadline_missed_mwh']) < 1e-5 for item in actual.values())
    row = dict(city=city, grid_id=grid, scenario=scenario, intensity_of_max=level, hours=len(frame),
               terminal_rule='backlog_by_class_no_greater_than_earliest_deadline_execution',
               E0_mwh=e0, flex_arrival_mwh=frame[list(CLASSES)].sum().tolist(),
               external_online_mwh=float(frame.on.sum()),
               service_complete=complete, all_deadlines_met=deadlines_met,
               realized_coordination_usd_mwh=((actual['seq']['cost_usd'] - actual['joint']['cost_usd']) / e0
                                             if complete else None),
               deadline_feasible_coordination_usd_mwh=((actual['seq']['cost_usd'] - actual['joint']['cost_usd']) / e0
                                                       if deadlines_met else None),
               perfect_rolling_coordination_usd_mwh=(oracle['seq']['cost_usd'] - oracle['joint']['cost_usd']) / e0,
               forecast_extra_joint_cost_usd_mwh=(actual['joint']['cost_usd'] - oracle['joint']['cost_usd']) / e0,
               cases=cases, seconds=time.monotonic() - started)
    path.write_text(json.dumps(row, allow_nan=False))
    print(city, scenario, level, row['realized_coordination_usd_mwh'], row['seconds'], flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    for name in ('project', 'config', 'asi', 'max-table', 'inputs', 'residual-code', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--horizon', type=int, default=49)
    parser.add_argument('--cities', default='')
    parser.add_argument('--scenarios', default='')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare(args)
    cities = args.cities.split(',') if args.cities else list(CONTEXT['selected'])
    scenarios = args.scenarios.split(',') if args.scenarios else list(CONTEXT['g'].CONTEXT['templates'])
    tasks = [(city, scenario, level) for city in cities for scenario in scenarios for level in (.5, .75)]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(one, tasks))
    rows = [json.loads((args.output / f'{city}-{scenario}-{level}.json').read_text()) for city, scenario, level in tasks]
    pd.DataFrame([{k: v for k, v in row.items() if k != 'cases'} for row in rows]).to_csv(args.output / 'summary.csv', index=False)


if __name__ == '__main__':
    main()
