"""四类柔性的onsite风光储+市电经济优化（COPT）。

prepare：把第二章四类柔性小时层（flex-class-20261101，参考网格）与各城市同一源小时的
设施功率（城市气象下的PUE）、城市风光容量因子合并成求解输入。
solve：单个算例线性规划；run：全部城市×情境×α×设计×柔性组合并行求解。
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse


RUN = 'flex-econ-copt-20261102'
CITIES = ['lulea', 'frankfurt', 'ashburn', 'ulanqab', 'guiyang', 'phoenix', 'riyadh', 'singapore']
SCENARIOS = ['A-HPC', 'A-CLD', 'B-SAI', 'C-SAI', 'D-SAI', 'B-TRN', 'C-TRN', 'D-TRN', 'B-INF', 'C-INF', 'D-INF']
LAYERS = {'base': '基础与空载 IT', 'rigid': '刚性', 'unknown': '未知', 'whole': '可推迟·整体', 'split': '可推迟·可拆分'}
HOURLY = ['x_whole', 'x_split', 'b_whole', 'b_split', 'grid', 'charge', 'discharge', 'soc', 'direct', 'curtail']


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def crf(rate, years):
    return rate / (1 - (1 + rate) ** (-years))


def load_config(project):
    own = json.loads((project / 'analysis/flex-econ-copt-20261102.json').read_text('utf-8'))
    base = json.loads((project / 'analysis/eight-city-flex-value-20261029.json').read_text('utf-8'))
    p = {k: v['value'] for k, v in base['parameters'].items()}
    p.update({k: v['value'] for k, v in own['parameters'].items()})
    rate = p['discount_rate']
    battery = p['battery_price_year_factor']
    p['annual_usd_per_mw'] = {
        'pv': p['pv_capex'] * 1000 * (crf(rate, p['pv_life_years']) + p['pv_fom_fraction']),
        'wind': p['wind_capex'] * 1000 * (crf(rate, p['wind_life_years']) + p['wind_fom_fraction']),
        'battery_power': p['battery_power_capex'] * battery * 1000 * (crf(rate, p['battery_life_years']) + p['battery_fom_fraction']),
        'battery_energy': p['battery_energy_capex'] * battery * 1000 * (crf(rate, p['battery_life_years']) + p['battery_fom_fraction']),
        'grid_capacity': p['grid_capacity_usd_kw_year'] * 1000,
    }
    p['emission_factor'] = {k: v['emission_factor'] for k, v in base['cities'].items()}
    return p, own


# ---------------------------------------------------------------- prepare
def prepare(args):
    project, out = Path(args.project), Path(args.output)
    (out / 'inputs').mkdir(parents=True, exist_ok=True)
    flex_dir, source = Path(args.flex_hourly), Path(args.source_root)
    reps = pd.read_csv(flex_dir / 'representative-placements.csv').set_index('scenario')
    p, _ = load_config(project)
    year = p['weather_year']
    records = []
    for scenario in SCENARIOS:
        layers = pd.read_parquet(flex_dir / f'{scenario}.parquet')
        placement = reps.loc[scenario, 'placement']
        for city in CITIES:
            if scenario.endswith('TRN'):
                path = source / f'results/trn-seren-20261101/hourly/{city}/main/{scenario}-10MW-{year}.parquet'
            else:
                path = source / f'data/processed/typical-load-curves-20261101-v3-native-combined/{city}/{scenario}-10MW-{year}-{placement}.parquet'
            town = pd.read_parquet(path, columns=['time_utc', 'weather_time_utc', 'source_hour_index', 'it_mw', 'facility_mw', 'installed_reference_it_mw'])
            merged = layers.merge(town, on='source_hour_index', how='inner', suffixes=('_ref', ''))
            assert len(merged) == len(layers) == len(town), (scenario, city, len(layers), len(town))
            it_gap = float(np.max(np.abs(merged.it_mw - merged.it_mw_ref)))
            supply_path = Path(args.supply_root) / f'{city}-supply.parquet'
            supply = pd.read_parquet(supply_path, columns=['time_utc', 'pv_capacity_factor', 'wind_capacity_factor'])
            supply = supply.rename(columns={'time_utc': 'weather_time_utc'})
            merged['weather_time_utc'] = pd.to_datetime(merged.weather_time_utc, utc=True)
            supply['weather_time_utc'] = pd.to_datetime(supply.weather_time_utc, utc=True)
            merged = merged.merge(supply, on='weather_time_utc', how='left')
            assert merged[['pv_capacity_factor', 'wind_capacity_factor']].notna().all().all(), (scenario, city)
            frame = pd.DataFrame({
                'time_utc': merged.time_utc, 'source_hour_index': merged.source_hour_index,
                **{k: merged[v].to_numpy(float) for k, v in LAYERS.items()},
                'it_mw': merged.it_mw.to_numpy(float), 'facility_mw': merged.facility_mw.to_numpy(float),
                'pue': (merged.facility_mw / merged.it_mw).to_numpy(float),
                'cap_it_mw': merged.installed_reference_it_mw.to_numpy(float),
                'cf_pv': merged.pv_capacity_factor.clip(0, 1).to_numpy(float),
                'cf_wind': merged.wind_capacity_factor.clip(0, 1).to_numpy(float)})
            layer_gap = float(np.max(np.abs(frame[list(LAYERS)].sum(axis=1) - frame.it_mw)))
            target = out / 'inputs' / f'{city}-{scenario}.parquet'
            frame.to_parquet(target, index=False)
            records.append({'city': city, 'scenario': scenario, 'placement': placement, 'hours': len(frame),
                            'it_vs_reference_max_abs_mw': it_gap, 'layer_sum_vs_city_it_max_abs_mw': layer_gap,
                            'it_mean_mw': frame.it_mw.mean(), 'facility_mean_mw': frame.facility_mw.mean(),
                            'pue_mean': frame.facility_mw.sum() / frame.it_mw.sum(),
                            'cf_pv_mean': frame.cf_pv.mean(), 'cf_wind_mean': frame.cf_wind.mean(),
                            'cap_it_mw': frame.cap_it_mw.iloc[0],
                            'load_path': str(path.relative_to(source)), 'load_sha256': digest(path),
                            'supply_path': str(supply_path), 'supply_sha256': digest(supply_path),
                            'flex_sha256': digest(flex_dir / f'{scenario}.parquet'), 'input_sha256': digest(target)})
            print(city, scenario, len(frame), f'it_gap={it_gap:.2e}', f'layer_gap={layer_gap:.2e}', flush=True)
    pd.DataFrame(records).to_csv(out / 'inputs/input-index.csv', index=False, encoding='utf-8-sig')


# ---------------------------------------------------------------- solve
def backlog_limit(arrival, deadline, source_hours):
    # 行末只留下能在下一可观测小时执行且仍未到期的任务，末行必须清空。
    n = len(arrival)
    times = np.asarray(source_hours, dtype=float)
    if len(times) != n or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('原生经过小时轴必须与输入同长、有效且严格递增')
    if deadline < 0 or int(deadline) != deadline:
        raise ValueError('推迟时长必须为非负整数小时')
    cumulative = np.r_[0., np.cumsum(arrival)]
    limit = np.zeros(n)
    if deadline > 0 and n > 1:
        cutoff = times[1:] - deadline
        first_allowed = np.searchsorted(times, cutoff, side='left')
        first_allowed = np.minimum(first_allowed, np.arange(1, n))
        limit[:-1] = np.maximum(cumulative[1:n] - cumulative[first_allowed], 0.)
    return limit


def solve(frame, p, city, *, alpha, design, whole_h, split_h, keep_hourly=False, cards=None, fixed_assets=None):
    """cards=None：沿用整机铭牌余量（首版）。cards给出时：算力按调度前任务峰值K配置，
    超过K需加卡e（MW任务功率），e按年化卡价计费，加的卡全时待机耗电idle_per_mw×e。"""
    import coptpy as cp
    started = time.monotonic()
    n = len(frame)
    fixed = frame.base.to_numpy() + frame.rigid.to_numpy() + frame.unknown.to_numpy()
    arr_w, arr_s = frame.whole.to_numpy(), frame.split.to_numpy()
    pue, cfp, cfw = frame.pue.to_numpy(), frame.cf_pv.to_numpy(), frame.cf_wind.to_numpy()
    headroom = frame.cap_it_mw.to_numpy() - fixed
    scale = n / 8760.
    ef = p['emission_factor'][city]
    unit = p['annual_usd_per_mw']
    off = {key: 5 + k * n for k, key in enumerate(HOURLY)}
    count = 5 + len(HOURLY) * n
    task = frame.rigid.to_numpy() + frame.unknown.to_numpy() + arr_w + arr_s
    peak_task = float(task.max())
    if cards is not None:
        extra = count
        count += 1
        headroom = peak_task - frame.rigid.to_numpy() - frame.unknown.to_numpy()
        idle_per_mw = cards['idle_per_mw']
    idx = np.arange(n)
    lb, ub = np.zeros(count), np.full(count, cp.COPT.INFINITY)
    ub[:5] = [p['pv_max_mw'], p['wind_max_mw'], p['battery_power_max_mw'], p['battery_energy_max_mwh'], p['grid_max_mw']]
    ub[off['b_whole'] + idx] = backlog_limit(arr_w, whole_h, frame.source_hour_index)
    ub[off['b_split'] + idx] = backlog_limit(arr_s, split_h, frame.source_hour_index)
    if whole_h == 0:
        lb[off['x_whole'] + idx] = ub[off['x_whole'] + idx] = arr_w
    if split_h == 0:
        lb[off['x_split'] + idx] = ub[off['x_split'] + idx] = arr_s
    c = np.zeros(count)
    c[:5] = np.array([unit['pv'], unit['wind'], unit['battery_power'], unit['battery_energy'], unit['grid_capacity']]) * scale
    grid_unit = p['grid_price_usd_mwh'] + p['carbon_price_usd_t'] * ef
    c[off['grid'] + idx] = grid_unit
    c[off['discharge'] + idx] = p['battery_throughput_usd_mwh']
    if fixed_assets is not None:
        # 供给侧固定为给定装机（无柔性成本最优），只优化运行与加卡
        lb[:5] = ub[:5] = fixed_assets
    if cards is not None:
        ub[extra] = cards['extra_max_mw']
        c[extra] = cards['usd_per_mw_year'] * scale
    rows, cols, vals, lhs, rhs = [], [], [], [], []
    r0 = 0

    def add(r, cidx, v):
        shape = np.broadcast(r, cidx, v).shape
        rows.append(np.broadcast_to(r, shape).ravel())
        cols.append(np.broadcast_to(cidx, shape).ravel())
        vals.append(np.broadcast_to(np.asarray(v, float), shape).ravel())

    # 1 可再生母线：风光=直供+充电+弃电（市电不能给储能充电）
    r = r0 + idx
    add(r, 0, cfp); add(r, 1, cfw)
    for key in ('direct', 'charge', 'curtail'):
        add(r, off[key] + idx, -1.)
    lhs.append(np.zeros(n)); rhs.append(np.zeros(n)); r0 += n
    # 2 负荷平衡：直供+市电+放电=PUE×(固定+执行的两类柔性)
    r = r0 + idx
    for key in ('direct', 'grid', 'discharge'):
        add(r, off[key] + idx, 1.)
    add(r, off['x_whole'] + idx, -pue); add(r, off['x_split'] + idx, -pue)
    if cards is not None:
        add(r, extra, -pue * idle_per_mw)  # 新增卡待机功率
    lhs.append(pue * fixed); rhs.append(pue * fixed); r0 += n
    # 3 储能电量（循环边界）
    eta = np.sqrt(p['battery_roundtrip_efficiency'])
    r = r0 + idx
    add(r, off['soc'] + idx, 1.); add(r, off['soc'] + (idx - 1) % n, -1.)
    add(r, off['charge'] + idx, -eta); add(r, off['discharge'] + idx, 1 / eta)
    lhs.append(np.zeros(n)); rhs.append(np.zeros(n)); r0 += n
    # 4 两类积压：b_t=b_{t-1}+到达-执行，窗口末清零
    for key, arrival in (('whole', arr_w), ('split', arr_s)):
        r = r0 + idx
        add(r, off['b_' + key] + idx, 1.); add(r[1:], off['b_' + key] + idx[:-1], -1.)
        add(r, off['x_' + key] + idx, 1.)
        lhs.append(arrival); rhs.append(arrival); r0 += n
    # 5 IT装机上限
    r = r0 + idx
    add(r, off['x_whole'] + idx, 1.); add(r, off['x_split'] + idx, 1.)
    if cards is not None:
        add(r, extra, -1.)  # 任务功率 ≤ K + 新增卡
    lhs.append(np.full(n, -cp.COPT.INFINITY)); rhs.append(headroom); r0 += n
    # 6 储能功率/电量与市电接入容量
    for key, cap in (('charge', 2), ('discharge', 2), ('soc', 3), ('grid', 4)):
        r = r0 + idx
        add(r, off[key] + idx, 1.); add(r, cap, -1.)
        lhs.append(np.full(n, -cp.COPT.INFINITY)); rhs.append(np.zeros(n)); r0 += n
    base_facility = float(np.dot(pue, frame.it_mw.to_numpy()))
    # 7 窗口本地电量匹配比例：Σ市电 ≤ (1-α) Σ设施用电
    if alpha is not None:
        add(r0, off['grid'] + idx, 1.)
        add(r0, off['x_whole'] + idx, -(1 - alpha) * pue); add(r0, off['x_split'] + idx, -(1 - alpha) * pue)
        if cards is not None:
            add(r0, extra, -(1 - alpha) * idle_per_mw * pue.sum())
        lhs.append(np.array([-cp.COPT.INFINITY])); rhs.append(np.array([(1 - alpha) * float(np.dot(pue, fixed))])); r0 += 1
    # 8 风光按均值：窗口平均风光发电=无柔性基准平均设施负荷
    if design == 're_mean':
        add(r0, 0, cfp.sum()); add(r0, 1, cfw.sum())
        lhs.append(np.array([base_facility])); rhs.append(np.array([base_facility])); r0 += 1
    A = sparse.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r0, count))
    env = cp.Envr()
    model = env.createModel(RUN)
    model.setParam('Logging', 0)
    model.setParam('Threads', 1)
    model.setParam('LpMethod', 2)
    model.setParam('Crossover', 1)
    model.setParam('FeasTol', 1e-9)
    model.setParam('DualTol', 1e-9)
    model.setParam('TimeLimit', 1800)
    model.loadMatrix(c, A, np.concatenate(lhs), np.concatenate(rhs), lb, ub)
    model.solve()
    status = model.status
    summary = {'city': city, 'scenario': frame.attrs.get('scenario'), 'alpha': alpha, 'design': design,
               'whole_h': whole_h, 'split_h': split_h, 'hours': n, 'status_code': int(status),
               'deadline_clock': 'source_hour_index_next_observed_slot',
               'solver_feasibility_tolerance': 1e-9, 'solver_dual_tolerance': 1e-9, 'solver_crossover': 1,
               'unobserved_hours_have_execution_variables': False,
               'solve_seconds': time.monotonic() - started, 'baseline_facility_mwh': base_facility,
               'baseline_it_mwh': float(frame.it_mw.sum())}
    if status != cp.COPT.OPTIMAL:
        summary['status'] = 'infeasible' if status == cp.COPT.INFEASIBLE else 'not_solved'
        model.clear(); env.close()
        return summary, None
    x = np.asarray(model.getValues(), float)
    model.clear(); env.close()
    d = {key: x[off[key] + idx] for key in HOURLY}
    pv, wind, power, energy, gcap = x[:5]
    extra_mw = float(x[extra]) if cards is not None else 0.
    idle_extra = idle_per_mw * extra_mw if cards is not None else 0.
    executed = fixed + d['x_whole'] + d['x_split'] + idle_extra
    facility = pue * executed
    generation = pv * cfp + wind * cfw
    residual = {
        'balance': np.max(np.abs(d['direct'] + d['grid'] + d['discharge'] - facility)),
        'bus': np.max(np.abs(generation - d['direct'] - d['charge'] - d['curtail'])),
        'whole_energy': abs(d['x_whole'].sum() - arr_w.sum()), 'split_energy': abs(d['x_split'].sum() - arr_s.sum()),
        'whole_deadline': max(0., np.max(d['b_whole'] - backlog_limit(arr_w, whole_h, frame.source_hour_index))),
        'split_deadline': max(0., np.max(d['b_split'] - backlog_limit(arr_s, split_h, frame.source_hour_index))),
        'it_cap': (max(0., np.max(executed - frame.cap_it_mw.to_numpy())) if cards is None
                   else max(0., np.max(executed - idle_extra - frame.base.to_numpy() - peak_task - extra_mw))),
        'grid_cap': max(0., np.max(d['grid'] - gcap)),
        'alpha': max(0., d['grid'].sum() - (1 - alpha) * facility.sum()) if alpha is not None else 0.}
    cost = {'pv': unit['pv'] * scale * pv, 'wind': unit['wind'] * scale * wind,
            'battery_power': unit['battery_power'] * scale * power, 'battery_energy': unit['battery_energy'] * scale * energy,
            'grid_capacity': unit['grid_capacity'] * scale * gcap,
            'grid_energy': p['grid_price_usd_mwh'] * d['grid'].sum(), 'carbon': p['carbon_price_usd_t'] * ef * d['grid'].sum(),
            'throughput': p['battery_throughput_usd_mwh'] * d['discharge'].sum(),
            'cards': cards['usd_per_mw_year'] * scale * extra_mw if cards is not None else 0.}
    total = sum(cost.values())
    moved_w = float(np.maximum(arr_w - d['x_whole'], 0).sum())
    moved_s = float(np.maximum(arr_s - d['x_split'], 0).sum())
    summary.update(status='optimal', total_cost_usd=total, **{'cost_' + k: v for k, v in cost.items()},
                   pv_mw=pv, wind_mw=wind, battery_mw=power, battery_mwh=energy, grid_cap_mw=gcap,
                   grid_mwh=float(d['grid'].sum()), curtail_mwh=float(d['curtail'].sum()),
                   generation_mwh=float(generation.sum()), facility_mwh=float(facility.sum()),
                   cfe_hourly=1 - float(d['grid'].sum() / facility.sum()), emissions_t=float(d['grid'].sum() * ef),
                   facility_peak_mw=float(facility.max()), baseline_facility_peak_mw=float((pue * frame.it_mw).max()),
                   peak_task_mw=peak_task, extra_task_mw=extra_mw,
                   above_peak_mwh=float(np.maximum(executed - idle_extra - frame.base.to_numpy() - peak_task, 0).sum()),
                   discharge_mwh=float(d['discharge'].sum()), extra_idle_mwh=idle_extra * n,
                   energy_cost_usd=total - cost['cards'],
                   moved_whole_mwh=moved_w, moved_split_mwh=moved_s,
                   arrival_whole_mwh=float(arr_w.sum()), arrival_split_mwh=float(arr_s.sum()),
                   max_residual=float(max(residual.values())),
                   **{'residual_' + k: float(v) for k, v in residual.items()})
    hourly = None
    if keep_hourly:
        hourly = pd.DataFrame({'time_utc': frame.time_utc, 'fixed_it_mw': fixed, 'whole_arrival_mw': arr_w,
                               'split_arrival_mw': arr_s, 'whole_exec_mw': d['x_whole'], 'split_exec_mw': d['x_split'],
                               'whole_backlog_mwh': d['b_whole'], 'split_backlog_mwh': d['b_split'],
                               'facility_mw': facility, 'pv_mw': pv * cfp, 'wind_mw': wind * cfw,
                               'grid_mw': d['grid'], 'charge_mw': d['charge'], 'discharge_mw': d['discharge'],
                               'soc_mwh': d['soc'], 'curtail_mw': d['curtail']})
    return summary, hourly


def job(task):
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    project, out, city, scenario, alpha, design, flex, whole_h, split_h, keep, variant, overrides = task
    p, _ = load_config(Path(project))
    if overrides:
        p.update(overrides)
        p['annual_usd_per_mw']['grid_capacity'] = p['grid_capacity_usd_kw_year'] * 1000
    frame = pd.read_parquet(Path(out) / 'inputs' / f'{city}-{scenario}.parquet')
    frame.attrs['scenario'] = scenario
    try:
        summary, hourly = solve(frame, p, city, alpha=alpha, design=design, whole_h=whole_h, split_h=split_h, keep_hourly=keep)
    except Exception as error:  # 记录失败，不中断批次
        summary, hourly = {'city': city, 'scenario': scenario, 'alpha': alpha, 'design': design,
                           'status': 'error', 'error': repr(error)[:500]}, None
    summary.update(flex=flex, variant=variant)
    if hourly is not None:
        folder = Path(out) / 'hourly'
        folder.mkdir(exist_ok=True)
        hourly.to_parquet(folder / f'{city}-{scenario}-{design}-a{alpha}-{flex}-{variant}.parquet', index=False)
    return summary


def key(city, scenario, alpha, design, flex, variant):
    return city, scenario, None if alpha is None or pd.isna(alpha) else round(float(alpha), 4), design, flex, variant


def run(args):
    project, out = Path(args.project), Path(args.output)
    p, own = load_config(project)
    tasks = []
    for city in CITIES:
        for scenario in SCENARIOS:
            for design in own['designs']:
                for alpha in p['alpha']:
                    for flex, spec in own['flex_cases'].items():
                        keep = alpha in (None, 0.95, 1.0) and design == 'opt'
                        tasks.append((str(project), str(out), city, scenario, alpha, design, flex,
                                      spec['whole_h'], spec['split_h'], keep, 'main', None))
            for alpha in (None, 0.95, 1.0):
                # 推迟时长敏感性：两类同用一个时长
                for hours in (2, 4, 8, 24):
                    tasks.append((str(project), str(out), city, scenario, alpha, 'opt', f'uniform{hours}', hours, hours, False, 'main', None))
                # 市电接入容量费取0的敏感性
                for flex in ('none', 'both'):
                    spec = own['flex_cases'][flex]
                    tasks.append((str(project), str(out), city, scenario, alpha, 'opt', flex, spec['whole_h'], spec['split_h'],
                                  False, 'grid_fee0', {'grid_capacity_usd_kw_year': 0}))
    if args.limit:
        tasks = tasks[:args.limit]
    target = out / 'runs.csv'
    rows = pd.read_csv(target).to_dict('records') if target.exists() else []
    done = {key(r['city'], r['scenario'], r['alpha'], r['design'], r['flex'], r['variant']) for r in rows}
    tasks = [t for t in tasks if key(t[2], t[3], t[4], t[5], t[6], t[10]) not in done]
    print('tasks', len(tasks), flush=True)
    last = time.monotonic()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(job, t) for t in tasks]
        for k, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if time.monotonic() - last > 60 or k == len(futures):
                # 每次整表重写，保证各行列名对齐
                temporary = target.with_suffix('.tmp')
                pd.DataFrame(rows).to_csv(temporary, index=False, encoding='utf-8')
                temporary.replace(target)
                last = time.monotonic()
                (out / 'progress.json').write_text(json.dumps({'done': k, 'total': len(futures)}), 'utf-8')
                print('progress', k, len(futures), flush=True)


def card_spec(p, own, frame, scenario, multiplier):
    """把服务器价折成每MW任务功率容量的年化费用与待机功率。"""
    cards = own['cards']
    fleet = cards['fleet'][scenario]
    servers = fleet['unit_count'] * fleet['u_max']          # 无柔性时所需服务器台数
    peak = float((frame.rigid + frame.unknown + frame.whole + frame.split).max())
    price = cards['server_price_usd'][scenario[0]]['value']
    factor = crf(p['discount_rate'], 4) + 0.03
    idle = float(frame.base.iloc[0]) / fleet['unit_count'] * servers / peak
    return {'usd_per_mw_year': multiplier * price * servers / peak * factor, 'idle_per_mw': idle,
            'extra_max_mw': peak if multiplier is not None else 0., 'baseline_fleet_usd_year': price * servers * factor,
            'servers_per_mw': servers / peak}


def card_job(task):
    project, out, city, scenario, alpha, flex, whole_h, split_h, multiplier = task
    p, own = load_config(Path(project))
    frame = pd.read_parquet(Path(out) / 'inputs' / f'{city}-{scenario}.parquet')
    frame.attrs['scenario'] = scenario
    spec = card_spec(p, own, frame, scenario, 0. if multiplier == 'forced0' else multiplier)
    if multiplier == 'forced0':
        spec['extra_max_mw'] = 0.
    try:
        summary, _ = solve(frame, p, city, alpha=alpha, design='opt', whole_h=whole_h, split_h=split_h, cards=spec)
    except Exception as error:
        summary = {'city': city, 'scenario': scenario, 'alpha': alpha, 'status': 'error', 'error': repr(error)[:500]}
    summary.update(flex=flex, card_price=str(multiplier), card_usd_per_mw_year=spec['usd_per_mw_year'],
                   idle_per_mw=spec['idle_per_mw'], baseline_fleet_usd_year=spec['baseline_fleet_usd_year'],
                   servers_per_mw=spec['servers_per_mw'])
    return summary


def run_cards(args):
    project, out = Path(args.project), Path(args.output)
    p, own = load_config(project)
    tasks = []
    for city in CITIES:
        for scenario in SCENARIOS:
            for alpha in p['alpha']:
                tasks.append((str(project), str(out), city, scenario, alpha, 'none', 0, 0, 'forced0'))
                spec = own['flex_cases']['both']
                for multiplier in ['forced0', *own['cards']['price_multipliers']['value']]:
                    tasks.append((str(project), str(out), city, scenario, alpha, 'both', spec['whole_h'], spec['split_h'], multiplier))
    if args.limit:
        tasks = tasks[:args.limit]
    target = out / 'runs-cards.csv'
    rows = pd.read_csv(target).to_dict('records') if target.exists() else []
    done = {(r['city'], r['scenario'], None if pd.isna(r['alpha']) else round(float(r['alpha']), 4), r['flex'], str(r['card_price'])) for r in rows}
    tasks = [t for t in tasks if (t[2], t[3], None if t[4] is None else round(float(t[4]), 4), t[5], str(t[8])) not in done]
    print('tasks', len(tasks), flush=True)
    last = time.monotonic()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(card_job, t) for t in tasks]
        for k, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if time.monotonic() - last > 60 or k == len(futures):
                temporary = target.with_suffix('.tmp')
                pd.DataFrame(rows).to_csv(temporary, index=False, encoding='utf-8')
                temporary.replace(target)
                last = time.monotonic()
                (out / 'progress-cards.json').write_text(json.dumps({'done': k, 'total': len(futures)}), 'utf-8')
                print('progress', k, len(futures), flush=True)


def fixed_job(task):
    project, out, city, scenario, alpha, multiplier, assets = task
    p, own = load_config(Path(project))
    frame = pd.read_parquet(Path(out) / 'inputs' / f'{city}-{scenario}.parquet')
    frame.attrs['scenario'] = scenario
    noflex = multiplier == 'noflex'
    spec = card_spec(p, own, frame, scenario, 0. if noflex else multiplier)
    market = card_spec(p, own, frame, scenario, 1.0)['usd_per_mw_year']
    spec['extra_max_mw'] = 0. if noflex else 2 * spec['extra_max_mw']
    spec_whole, spec_split = (0, 0) if noflex else (own['flex_cases']['both']['whole_h'], own['flex_cases']['both']['split_h'])
    try:
        summary, _ = solve(frame, p, city, alpha=alpha, design='opt', whole_h=spec_whole, split_h=spec_split,
                           cards=spec, fixed_assets=assets)
    except Exception as error:
        summary = {'city': city, 'scenario': scenario, 'alpha': alpha, 'status': 'error', 'error': repr(error)[:500]}
    summary.update(flex='none' if noflex else 'both', supply='fixed_noflex_opt', card_price=str(multiplier), card_market_usd_per_mw_year=market,
                   idle_per_mw=spec['idle_per_mw'], baseline_fleet_usd_year=spec['baseline_fleet_usd_year'],
                   servers_per_mw=spec['servers_per_mw'])
    return summary


def run_fixed(args):
    project, out = Path(args.project), Path(args.output)
    p, own = load_config(project)
    base = pd.read_csv(out / 'runs-cards.csv')
    base = base[(base.flex == 'none') & (base.status == 'optimal')]
    tasks = []
    for row in base.itertuples():
        alpha = None if pd.isna(row.alpha) else float(row.alpha)
        assets = [row.pv_mw, row.wind_mw, row.battery_mw, row.battery_mwh, row.grid_cap_mw]
        for multiplier in ['noflex', 0.0, 1.0]:
            tasks.append((str(project), str(out), row.city, row.scenario, alpha, multiplier, assets))
    target = out / 'runs-fixed.csv'
    rows = []
    print('tasks', len(tasks), flush=True)
    last = time.monotonic()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fixed_job, t) for t in tasks]
        for k, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if time.monotonic() - last > 60 or k == len(futures):
                temporary = target.with_suffix('.tmp')
                pd.DataFrame(rows).to_csv(temporary, index=False, encoding='utf-8')
                temporary.replace(target)
                last = time.monotonic()
                (out / 'progress-fixed.json').write_text(json.dumps({'done': k, 'total': len(futures)}), 'utf-8')
                print('progress', k, len(futures), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run', 'run-cards', 'run-fixed'])
    parser.add_argument('--project', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--flex-hourly')
    parser.add_argument('--source-root')
    parser.add_argument('--supply-root')
    parser.add_argument('--workers', type=int, default=48)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'run-cards': run_cards, 'run-fixed': run_fixed}[args.stage](args)


if __name__ == '__main__':
    main()
