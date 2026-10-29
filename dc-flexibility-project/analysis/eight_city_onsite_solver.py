import hashlib
import json
import time
import warnings
from collections import deque

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import OptimizeWarning, linprog


MODEL_VERSION = 'onsite-backlog-v1'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value(config, key):
    entry = config['parameters'][key]
    if not entry['source'] or not entry['status']:
        raise ValueError(f'参数缺少来源或情景标记：{key}')
    return entry['value']


def capital_recovery(rate, years):
    if rate < 0 or years <= 0:
        raise ValueError('折现率与寿命超出有效范围')
    return 1 / years if rate == 0 else rate / (1 - (1 + rate) ** (-years))


def annual_costs(config):
    p = lambda key: float(value(config, key))
    rate = p('discount_rate')
    costs = []
    for asset, unit_scale in (('pv', 1000), ('wind', 1000), ('battery_power', 1000), ('battery_energy', 1000)):
        life = p('battery_life_years') if asset.startswith('battery') else p(asset + '_life_years')
        om = p('battery_fom_fraction') if asset.startswith('battery') else p(asset + '_fom_fraction')
        factor = p('battery_price_year_factor') if asset.startswith('battery') else 1
        costs.append(p(asset + '_capex') * factor * unit_scale * (capital_recovery(rate, life) + om))
    compute = p('compute_capex_usd_mw') * (capital_recovery(rate, p('compute_life_years')) + p('compute_fom_fraction'))
    result = np.array([compute * p('reference_it_mw'), *costs])
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError('全部设备年化单价必须为正数')
    return result


def storage_bounds(frame, config, storage):
    if storage == 'opt':
        return [(0, value(config, 'battery_power_max_mw')), (0, value(config, 'battery_energy_max_mwh'))]
    if isinstance(storage, dict):
        power, energy = storage['power_mw'], storage['energy_mwh']
    else:
        hours = float(storage)
        if hours < 0:
            raise ValueError('储能时长必须非负')
        power = float(frame.it_mw.mean()) if hours > 0 else 0
        energy = power * hours
    if min(power, energy) < 0:
        raise ValueError('储能功率或电量为负数')
    return [(power, power), (energy, energy)]


def edf_metrics(arrivals, executed, deadline, tolerance=1e-5):
    pending = deque()
    delayed, waiting, maximum = 0., 0., 0
    for hour, (arrival, completed) in enumerate(zip(arrivals, executed)):
        if arrival > 0:
            pending.append([hour, float(arrival)])
        remaining = max(0., float(completed))
        while remaining > 1e-10 and pending:
            start, unfinished = pending[0]
            amount = min(remaining, unfinished)
            if amount > tolerance:
                maximum = max(maximum, hour - start)
            if hour > start:
                delayed += amount
            waiting += amount * (hour - start)
            remaining -= amount
            pending[0][1] -= amount
            if pending[0][1] < 1e-9:
                pending.popleft()
        if remaining > tolerance:
            raise ArithmeticError(f'EDF重建超过到达量：hour={hour}, amount={remaining}')
        if pending and hour - pending[0][0] >= deadline and pending[0][1] > tolerance:
            raise ArithmeticError(f'EDF重建发现逾期：hour={hour}, oldest={pending[0]}')
    residual = sum(item[1] for item in pending)
    if residual > tolerance or maximum > deadline:
        raise ArithmeticError(f'EDF重建期末遗留或超期：{residual}, {maximum}')
    return {'edf_delayed_dynamic_it_mwh': delayed, 'waiting_integral_mwh_h': waiting,
            'maximum_delay_h': maximum}


def check_frame(frame, idle_mw, design_mw):
    required = ['it_mw', 'pue', 'pv_capacity_factor', 'wind_capacity_factor']
    if len(frame) < 2 or not np.isfinite(frame[required].to_numpy(float)).all():
        raise ValueError('输入小时数不足或含非有限值')
    if frame.pue.lt(1).any() or frame.it_mw.lt(idle_mw - 1e-7).any() or frame.it_mw.gt(design_mw + 1e-6).any():
        raise ValueError('IT/PUE超出设备端点或物理边界')
    if ((frame[['pv_capacity_factor', 'wind_capacity_factor']] < 0) |
            (frame[['pv_capacity_factor', 'wind_capacity_factor']] > 1)).any().any():
        raise ValueError('风光容量因子超出[0,1]')


def solve_onsite(frame, config, *, deadline=0, fraction=1., rmax=2., storage=0,
                 alpha=None, carbon_price=100., emission_factor=.4, idle_mw=0.,
                 design_mw=10., fixed_assets=None, physical=False, explicit=False):
    started = time.monotonic()
    frame = frame.reset_index(drop=True).copy()
    check_frame(frame, idle_mw, design_mw)
    if int(deadline) != deadline or deadline < 0 or not 0 <= fraction <= 1 or rmax < 1:
        raise ValueError('D、φ或r无效')
    if alpha is not None and not 0 <= alpha <= 1:
        raise ValueError('α超出[0,1]')
    if min(carbon_price, emission_factor) < 0:
        raise ValueError('碳价或排放因子为负数')
    n = len(frame)
    p = lambda key: float(value(config, key))
    q = frame.it_mw.to_numpy(float) - idle_mw
    flexible_arrival, fixed = fraction * q, (1 - fraction) * q
    pue = frame.pue.to_numpy(float)
    dyn_cap = design_mw - idle_mw
    if dyn_cap <= 0:
        raise ValueError('动态功率端点必须大于零')
    costs = annual_costs(config)
    annual_multiplier = p('annual_hours') / n
    offsets = {key: 5 + k * n for k, key in enumerate(('service', 'backlog', 'direct', 'grid', 'charge', 'discharge', 'soc', 'curtail'))}
    count = 5 + 8 * n
    pairs = [(a, t) for a in range(n) for t in range(a, min(n, a + deadline + 1))] if explicit else []
    count += len(pairs)
    lower, upper = np.zeros(count), np.full(count, np.inf)
    lower[0], upper[0] = 1., rmax
    upper[1:3] = [p('pv_max_mw'), p('wind_max_mw')]
    for i, (lo, hi) in enumerate(storage_bounds(frame, config, storage), 3):
        lower[i], upper[i] = lo, hi
    if fixed_assets is not None:
        for i, key in enumerate(('pv_mw', 'wind_mw', 'battery_power_mw', 'battery_energy_mwh'), 1):
            lower[i] = upper[i] = fixed_assets[key]
    index = np.arange(n)
    cumulative = np.r_[0., np.cumsum(flexible_arrival)]
    # 累计到达减去截止时点以前必须完成的到达，约束每批任务的最迟执行时间。
    backlog_limit = cumulative[1:] - cumulative[np.maximum(index - deadline + 1, 0)]
    upper[offsets['backlog'] + index] = np.maximum(0., backlog_limit)
    upper[offsets['backlog'] + n - 1] = 0
    upper[offsets['grid'] + index] = p('grid_max_mw')
    objective = np.zeros(count)
    objective[:5] = costs
    grid_unit_cost = p('grid_price_usd_mwh') + carbon_price * emission_factor
    objective[offsets['grid'] + index] = annual_multiplier * grid_unit_cost
    objective[offsets['discharge'] + index] = annual_multiplier * p('battery_throughput_usd_mwh')
    if physical:
        objective[:] = 0
        objective[offsets['grid'] + index] = annual_multiplier
        objective[offsets['discharge'] + index] = annual_multiplier * 1e-7
        objective[0] = 1e-7
    eq_rows, eq_cols, eq_data = [], [], []
    ub_rows, ub_cols, ub_data = [], [], []

    def block(target, row, col, data):
        rows, cols, coeff = target
        shape = np.broadcast(row, col, data).shape
        rows.append(np.broadcast_to(row, shape).ravel())
        cols.append(np.broadcast_to(col, shape).ravel())
        coeff.append(np.broadcast_to(data, shape).ravel())

    eq, ub = (eq_rows, eq_cols, eq_data), (ub_rows, ub_cols, ub_data)
    # 明确分开可再生母线，电网不能给储能充电。
    block(eq, index, 1, frame.pv_capacity_factor.to_numpy(float))
    block(eq, index, 2, frame.wind_capacity_factor.to_numpy(float))
    for key in ('direct', 'charge', 'curtail'):
        block(eq, index, offsets[key] + index, -1)
    for key in ('direct', 'grid', 'discharge'):
        block(eq, n + index, offsets[key] + index, 1)
    block(eq, n + index, 0, -pue * idle_mw)
    block(eq, n + index, offsets['service'] + index, -pue)
    etac = etad = np.sqrt(p('battery_roundtrip_efficiency'))
    loss = p('battery_hourly_loss')
    if not 0 < etac <= 1 or not 0 <= loss < 1:
        raise ValueError('储能效率或自放电无效')
    block(eq, 2 * n + index, offsets['soc'] + index, 1)
    block(eq, 2 * n + index, offsets['soc'] + (index - 1) % n, -(1 - loss))
    block(eq, 2 * n + index, offsets['charge'] + index, -etac)
    block(eq, 2 * n + index, offsets['discharge'] + index, 1 / etad)
    block(eq, 3 * n + index, offsets['backlog'] + index, 1)
    block(eq, 3 * n + index[1:], offsets['backlog'] + index[:-1], -1)
    block(eq, 3 * n + index, offsets['service'] + index, 1)
    eq_rhs = np.r_[np.zeros(n), pue * fixed, np.zeros(n), flexible_arrival]
    eq_count = 4 * n
    if explicit:
        if n > 336:
            raise ValueError('逐批x(a,t)只允许用于小规模对照')
        a = np.array([item[0] for item in pairs])
        t = np.array([item[1] for item in pairs])
        cols = 5 + 8 * n + np.arange(len(pairs))
        block(eq, eq_count + a, cols, 1)
        block(eq, eq_count + n + t, cols, 1)
        block(eq, eq_count + n + index, offsets['service'] + index, -1)
        eq_rhs = np.r_[eq_rhs, flexible_arrival, np.zeros(n)]
        eq_count += 2 * n
        # 逐批形式独立限制截止时间，移除积压上界进行真正的等价性比较。
        upper[offsets['backlog'] + index] = np.inf
        upper[offsets['backlog'] + n - 1] = 0
    for number, (key, capacity, scale) in enumerate((('service', 0, dyn_cap), ('charge', 3, 1), ('discharge', 3, 1), ('soc', 4, 1))):
        block(ub, number * n + index, offsets[key] + index, 1)
        block(ub, number * n + index, capacity, -scale)
    ub_rhs = np.r_[-fixed, np.zeros(3 * n)]
    ub_count = 4 * n
    if alpha is not None:
        block(ub, ub_count, offsets['grid'] + index, 1)
        block(ub, ub_count, offsets['service'] + index, -(1 - alpha) * pue)
        block(ub, ub_count, 0, -(1 - alpha) * idle_mw * pue.sum())
        ub_rhs = np.r_[ub_rhs, (1 - alpha) * np.dot(pue, fixed)]
        ub_count += 1
    aeq = sparse.coo_matrix((np.concatenate(eq_data), (np.concatenate(eq_rows), np.concatenate(eq_cols))), shape=(eq_count, count)).tocsr()
    aub = sparse.coo_matrix((np.concatenate(ub_data), (np.concatenate(ub_rows), np.concatenate(ub_cols))), shape=(ub_count, count)).tocsr()
    summary = {'model_version': MODEL_VERSION, 'hours': n, 'D': deadline, 'phi': fraction, 'rmax': rmax,
               'storage': storage, 'alpha': alpha, 'carbon_price': carbon_price, 'emission_factor': emission_factor,
               'physical': physical, 'variables': count, 'matrix_nonzeros': aeq.nnz + aub.nnz,
               'baseline_it_mwh': float(frame.it_mw.sum()), 'idle_mw': idle_mw, 'design_mw': design_mw}
    options = {'threads': 1, 'primal_feasibility_tolerance': 1e-8, 'dual_feasibility_tolerance': 1e-8,
               'time_limit': p('solver_time_limit_seconds'), 'presolve': value(config, 'solver_presolve')}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected.*', category=OptimizeWarning)
        method = value(config, 'solver_method')
        if method not in ('highs', 'highs-ipm', 'highs-ds'):
            raise ValueError('必须使用HiGHS求解器')
        if alpha is not None:
            # 发电与储能容量增加不收紧可行域。固定到允许上限后，独立求最小α约束残差。
            certificate_lower = lower.copy()
            certificate_lower[1:5] = upper[1:5]
            alpha_row = aub.getrow(ub_count - 1).toarray().ravel()
            certificate = linprog(alpha_row, A_eq=aeq, b_eq=eq_rhs, A_ub=aub[:-1], b_ub=ub_rhs[:-1],
                bounds=np.column_stack([certificate_lower, upper]), method=method, options=options)
            summary.update(alpha_certificate_solver_status=int(certificate.status),
                           alpha_certificate_seconds=time.monotonic() - started)
            if not certificate.success:
                summary.update(status='not_solved', solver_status=int(certificate.status),
                    solver_message=certificate.message, solve_stage='alpha_feasibility',
                    solve_seconds=time.monotonic() - started)
                return {'summary': summary, 'hourly': pd.DataFrame()}
            minimum_residual = float(certificate.fun - ub_rhs[-1])
            summary['alpha_minimum_constraint_residual_mwh'] = minimum_residual
            if minimum_residual > p('alpha_certificate_tolerance_mwh'):
                reason = 'alpha_target_unreachable_with_given_storage_and_capacity_limits'
                if alpha == 1 and upper[3] == 0 and ((frame.pv_capacity_factor + frame.wind_capacity_factor) == 0).any() and idle_mw > 0:
                    reason = 'zero_renewable_hour_with_idle_load_no_storage'
                summary.update(status='infeasible', solver_status=int(certificate.status),
                    solver_message=certificate.message, solve_stage='alpha_feasibility',
                    infeasibility_reason=reason, solve_seconds=time.monotonic() - started)
                return {'summary': summary, 'hourly': pd.DataFrame()}
        result = linprog(objective / (1 if physical else 1000), A_eq=aeq, b_eq=eq_rhs, A_ub=aub,
                         b_ub=ub_rhs, bounds=np.column_stack([lower, upper]), method=method, options=options)
    summary.update(solve_seconds=time.monotonic() - started, solver_status=int(result.status), solver_message=result.message)
    if not result.success:
        reason = 'resource_or_capacity_limit'
        if alpha == 1 and upper[3] == 0 and ((frame.pv_capacity_factor + frame.wind_capacity_factor) == 0).any() and idle_mw > 0:
            reason = 'zero_renewable_hour_with_idle_load_no_storage'
        summary.update(status='infeasible' if result.status == 2 else 'not_solved',
                       infeasibility_reason=reason if result.status == 2 else None)
        return {'summary': summary, 'hourly': pd.DataFrame()}
    x = result.x
    dispatch = {key: x[offsets[key] + index] for key in offsets}
    r, pv, wind, power, energy = x[:5]
    executed = fixed + dispatch['service']
    it = idle_mw * r + executed
    facility = pue * it
    generation = pv * frame.pv_capacity_factor.to_numpy() + wind * frame.wind_capacity_factor.to_numpy()
    grid = dispatch['grid']
    residuals = {
        'power_mw': float(np.max(np.abs(generation + grid + dispatch['discharge'] - facility - dispatch['charge'] - dispatch['curtail']))),
        'renewable_bus_mw': float(np.max(np.abs(generation - dispatch['direct'] - dispatch['charge'] - dispatch['curtail']))),
        'soc_mwh': float(np.max(np.abs(dispatch['soc'] - (1 - loss) * np.roll(dispatch['soc'], 1) - etac * dispatch['charge'] + dispatch['discharge'] / etad))),
        'dynamic_it_energy_mwh': float(abs(q.sum() - executed.sum())),
        'backlog_deadline_mwh': float(max(0., np.max(dispatch['backlog'] - backlog_limit))),
        'end_backlog_mwh': float(abs(dispatch['backlog'][-1])),
        'simultaneous_charge_discharge_mw': float(np.minimum(dispatch['charge'], dispatch['discharge']).max()),
        'grid_charging_mw': float(np.maximum(dispatch['charge'] - generation, 0).max()),
        'capacity_violation_mw': float(max(0., np.max(executed - dyn_cap * r))),
        'matrix_equality': float(np.max(np.abs(aeq @ x - eq_rhs))),
        'matrix_inequality': float(max(0., np.max(aub @ x - ub_rhs))),
        'bound_violation': float(max(0., np.max(lower - x), np.max(x - upper))),
        'alpha_violation_mwh': float(max(0., grid.sum() - (1 - alpha) * facility.sum())) if alpha is not None else 0.,
    }
    if max(residuals.values()) > 2e-5:
        raise ArithmeticError(f'逐小时守恒或运行边界核验失败：{residuals}')
    edf = edf_metrics(flexible_arrival, dispatch['service'], deadline)
    if not np.isclose(edf['waiting_integral_mwh_h'], dispatch['backlog'].sum(), atol=1e-3, rtol=1e-7):
        raise ArithmeticError('EDF等待时间积分与积压量积分不一致')
    components = dict(zip(('compute', 'pv', 'wind', 'battery_power', 'battery_energy'), (costs * x[:5]).tolist()))
    components['grid_energy'] = float(grid.sum() * annual_multiplier * p('grid_price_usd_mwh'))
    components['carbon_tax'] = float(grid.sum() * annual_multiplier * carbon_price * emission_factor)
    components['battery_throughput'] = float(dispatch['discharge'].sum() * annual_multiplier * p('battery_throughput_usd_mwh'))
    annual_cost = sum(components.values())
    if not physical and not np.isclose(annual_cost, result.fun * 1000, atol=.05, rtol=1e-8):
        raise ArithmeticError('成本分项与求解器目标不一致')
    frame['dynamic_arrival_mwh'], frame['dynamic_executed_mwh'] = q, executed
    frame['executed_it_mw'], frame['executed_facility_mw'] = it, facility
    frame['renewable_generation_mw'] = generation
    frame['cfe_hourly'] = 1 - np.divide(grid, facility, out=np.zeros(n), where=facility > 1e-10)
    for key, series in dispatch.items():
        frame[key] = series
    moved = float(np.maximum(q - executed, 0).sum())
    summary.update(status='optimal', annual_cost_usd=annual_cost, cost_components_usd=components,
                   compute_capacity_multiple=float(r), pv_mw=float(pv), wind_mw=float(wind),
                   battery_power_mw=float(power), battery_energy_mwh=float(energy),
                   grid_mwh=float(grid.sum()), curtailment_mwh=float(dispatch['curtail'].sum()),
                   facility_mwh=float(facility.sum()), it_mwh=float(it.sum()),
                   extra_idle_it_mwh=float((r - 1) * idle_mw * n),
                   local_cfe_energy=1 - float(grid.sum() / facility.sum()),
                   local_cfe_hourly_mean=float(frame.cfe_hourly.mean()),
                   emissions_tco2e=float(grid.sum() * emission_factor), moved_dynamic_it_mwh=moved,
                   diagnostics=residuals, **edf)
    return {'summary': summary, 'hourly': frame}
