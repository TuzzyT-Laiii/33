"""第三节试算：真实推理需求进入既有机群后，算电协同还剩多少柔性。

机群与原任务沿用第一节（flex-econ-copt-20261102 的城市输入，容量 K = 调度前任务峰值，不加卡）。
外部推理需求取 ASI 推理占用曲线（D-INF 原生小时层）：刚性层 = 在线推理（当小时完成），
可推迟·整体层 = 离线推理（24 小时内交付）。在线/离线比例保持 ASI 观测值。
需求强度 I = 推理平均任务功率 ÷ 机群平均闲置任务功率（K − 原任务平均功率）。
需求形状：flat（各自取均值）、real（ASI 原曲线）、amp2（围绕均值把波动放大 2 倍，截断为非负后按均值重标）。

配对比较（同一需求、同一供能边界）：
  seq   顺序调度：先按“尽早执行”排算力（最小化积压总量，不看电价），再固定负荷优化风光储市电；
  joint 联合调度：原任务两类可推迟层与离线推理和供能一起优化。
剩余柔性价值 V = 成本(seq) − 成本(joint)；保留率 R = V(I) ÷ V(0)，V(0) 即第一节的调度节省。
求解器：scipy HiGHS（本机无 COPT），I=0 时与第一节 runs-cards 的 none/both 结果对照。
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flex_econ_copt import backlog_limit, load_config  # noqa: E402

RUN = 'residual-flex-pilot-20261114'
X = ['x_whole', 'x_split', 'x_off']
B = ['b_whole', 'b_split', 'b_off']
SUPPLY = ['grid', 'charge', 'discharge', 'soc', 'direct', 'curtail']
HOURLY = X + B + SUPPLY
CAPS = ['pv', 'wind', 'battery_power', 'battery_energy', 'grid_capacity']
INF_SHIFT = pd.Timedelta(days=14)  # 训练窗口末端超出 ASI 窗口 13 天；整周平移保持 UTC 时刻与星期对齐


def asi_inference(path):
    f = pd.read_parquet(path)
    cols = list(f.columns)
    # 列顺序：…, 基础与空载 IT, 刚性, 可推迟·整体, 可推迟·可拆分, 未知, …（中文列名在部分环境下编码异常，按位置取）
    on, off = f[cols[6]].to_numpy(float), f[cols[7]].to_numpy(float)
    assert abs(on.mean() - 0.977568) < 1e-4 and abs(off.mean() - 0.378696) < 1e-4, '列位置与 ASI D-INF 不符'
    return pd.DataFrame({'time_utc': pd.to_datetime(f.time_utc, utc=True), 'on': on, 'off': off})


def shaped(series, shape):
    m = series.mean()
    if shape == 'flat':
        return np.full_like(series, m)
    if shape == 'real':
        return series.copy()
    k = float(shape.removeprefix('amp'))
    s = np.clip(m + k * (series - m), 0, None)
    return s * m / s.mean()


def demand(frame, asi, intensity, shape):
    t = pd.to_datetime(frame.time_utc, utc=True) - INF_SHIFT
    a = asi.set_index('time_utc').reindex(t)
    assert a.notna().all().all(), 'ASI 平移后未覆盖训练窗口'
    on, off = a.on.to_numpy(), a.off.to_numpy()
    task = (frame.rigid + frame.unknown + frame.whole + frame.split).to_numpy()
    idle_mean = task.max() - task.mean()
    c = intensity * idle_mean / (on.mean() + off.mean())
    return c * shaped(on, shape), c * shaped(off, shape), idle_mean


class LP:
    def __init__(self, n, ncap):
        self.n, self.ncap = n, ncap
        self.off = {k: ncap + i * n for i, k in enumerate(HOURLY)}
        self.count = ncap + len(HOURLY) * n
        self.eq, self.ub = [], []  # (rows, cols, vals, rhs)
        self.neq = self.nub = 0

    def block(self, kind, terms, rhs):
        """terms: [(col_index_array_or_int, coef_array_or_scalar, row_offset_array)]；rhs 长度为块行数。"""
        rhs = np.atleast_1d(np.asarray(rhs, float))
        base = self.neq if kind == 'eq' else self.nub
        rows, cols, vals = [], [], []
        for r, c, v in terms:
            shape = np.broadcast(r, c, v).shape
            rows.append(base + np.broadcast_to(r, shape).ravel())
            cols.append(np.broadcast_to(c, shape).ravel())
            vals.append(np.broadcast_to(np.asarray(v, float), shape).ravel())
        (self.eq if kind == 'eq' else self.ub).append((np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), rhs))
        if kind == 'eq':
            self.neq += len(rhs)
        else:
            self.nub += len(rhs)

    def matrices(self):
        out = []
        for parts, m in ((self.eq, self.neq), (self.ub, self.nub)):
            if not parts:
                out += [None, None]
                continue
            A = sparse.csr_matrix((np.concatenate([p[2] for p in parts]),
                                   (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))),
                                  shape=(m, self.count))
            out += [A, np.concatenate([p[3] for p in parts])]
        return out


def solve(frame, p, city, on, off, *, fee, whole_h=12, split_h=48, off_h=24, mode='joint', fixed_x=None):
    """mode='asap'：只排算力，最小化积压；mode='cost'：给定（或不给定）算力执行，最小化供能成本。"""
    n = len(frame)
    idx = np.arange(n)
    hours = frame.source_hour_index
    fixed = (frame.base + frame.rigid + frame.unknown).to_numpy() + on
    arr = {'whole': frame.whole.to_numpy(), 'split': frame.split.to_numpy(), 'off': off}
    dl = {'whole': whole_h, 'split': split_h, 'off': off_h}
    pue, cfp, cfw = frame.pue.to_numpy(), frame.cf_pv.to_numpy(), frame.cf_wind.to_numpy()
    task0 = (frame.rigid + frame.unknown + frame.whole + frame.split).to_numpy()
    K = float(task0.max())
    headroom = K - frame.rigid.to_numpy() - frame.unknown.to_numpy() - on
    if headroom.min() < -1e-9:
        return {'status': 'online_exceeds_capacity', 'online_overflow_mw': float(-headroom.min())}, None
    scale = n / 8760.
    unit = dict(p['annual_usd_per_mw'])
    unit['grid_capacity'] = fee * 1000.
    ef = p['emission_factor'][city]
    lp = LP(n, len(CAPS))
    o = lp.off
    lb, ub = np.zeros(lp.count), np.full(lp.count, np.inf)
    ub[:5] = [p['pv_max_mw'], p['wind_max_mw'], p['battery_power_max_mw'], p['battery_energy_max_mwh'], p['grid_max_mw']]
    for k in arr:
        ub[o['b_' + k] + idx] = backlog_limit(arr[k], dl[k], hours)
    c = np.zeros(lp.count)
    # 算力：积压递推与容量
    for k in arr:
        lp.block('eq', [(idx, o['b_' + k] + idx, 1.), (idx[1:], o['b_' + k] + idx[:-1], -1.), (idx, o['x_' + k] + idx, 1.)], arr[k])
    lp.block('ub', [(idx, o[x] + idx, 1.) for x in X], headroom)
    if mode == 'asap':
        for k in arr:
            c[o['b_' + k] + idx] = 1.
        for key in SUPPLY:
            ub[o[key] + idx] = 0.
        ub[:5] = 0.
    else:
        if fixed_x is not None:
            for x in X:
                lb[o[x] + idx] = ub[o[x] + idx] = fixed_x[x]
        c[:5] = np.array([unit[k] for k in CAPS]) * scale
        c[o['grid'] + idx] = p['grid_price_usd_mwh'] + p['carbon_price_usd_t'] * ef
        c[o['discharge'] + idx] = p['battery_throughput_usd_mwh']
        lp.block('eq', [(idx, 0, cfp), (idx, 1, cfw)] + [(idx, o[k] + idx, -1.) for k in ('direct', 'charge', 'curtail')], np.zeros(n))
        lp.block('eq', [(idx, o[k] + idx, 1.) for k in ('direct', 'grid', 'discharge')] + [(idx, o[x] + idx, -pue) for x in X], pue * fixed)
        eta = np.sqrt(p['battery_roundtrip_efficiency'])
        lp.block('eq', [(idx, o['soc'] + idx, 1.), (idx, o['soc'] + (idx - 1) % n, -1.),
                        (idx, o['charge'] + idx, -eta), (idx, o['discharge'] + idx, 1 / eta)], np.zeros(n))
        for key, cap in (('charge', 2), ('discharge', 2), ('soc', 3), ('grid', 4)):
            lp.block('ub', [(idx, o[key] + idx, 1.), (idx, cap, -1.)], np.zeros(n))
    Aeq, beq, Aub, bub = lp.matrices()
    t0 = time.monotonic()
    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=np.c_[lb, ub], method='highs-ipm',
                  options={'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9, 'time_limit': 1800})
    secs = time.monotonic() - t0
    if res.status != 0:
        return {'status': f'highs_{res.status}', 'message': res.message, 'solve_seconds': secs}, None
    x = res.x
    d = {k: x[o[k] + idx] for k in HOURLY}
    out = {'status': 'optimal', 'solve_seconds': secs, 'K_mw': K}
    if mode == 'asap':
        return out, {k: d[k] for k in X}
    pv, wind, power, energy, gcap = x[:5]
    executed = fixed + sum(d[x_] for x_ in X)
    facility = pue * executed
    cost = {k: unit[k] * scale * v for k, v in zip(CAPS, x[:5])}
    cost.update(grid_energy=p['grid_price_usd_mwh'] * d['grid'].sum(), carbon=p['carbon_price_usd_t'] * ef * d['grid'].sum(),
                throughput=p['battery_throughput_usd_mwh'] * d['discharge'].sum())
    resid = max(np.max(np.abs(d['direct'] + d['grid'] + d['discharge'] - facility)),
                *(abs(d['x_' + k].sum() - arr[k].sum()) for k in arr),
                np.max(sum(d[x_] for x_ in X) - headroom))
    out.update(total_cost_usd=sum(cost.values()), **{'cost_' + k: v for k, v in cost.items()},
               pv_mw=pv, wind_mw=wind, battery_mw=power, battery_mwh=energy, grid_cap_mw=gcap,
               grid_mwh=float(d['grid'].sum()), emissions_t=float(d['grid'].sum() * ef),
               facility_mwh=float(facility.sum()), facility_peak_mw=float(facility.max()),
               cfe_hourly=1 - float(d['grid'].sum() / facility.sum()),
               moved_whole_mwh=float(np.maximum(arr['whole'] - d['x_whole'], 0).sum()),
               moved_split_mwh=float(np.maximum(arr['split'] - d['x_split'], 0).sum()),
               moved_off_mwh=float(np.maximum(arr['off'] - d['x_off'], 0).sum()),
               it_utilization=float((executed - frame.base.to_numpy()).sum() / (K * n)),
               max_residual=float(resid))
    return out, {k: d[k] for k in X + ['grid', 'discharge', 'charge']} | {'facility': facility}


def case(task):
    inputs, project, city, scenario, intensity, shape, fee, asi_path = task
    p, _ = load_config(Path(project))
    frame = pd.read_parquet(Path(inputs) / f'{city}-{scenario}.parquet')
    asi = asi_inference(asi_path)
    on, off, idle_mean = demand(frame, asi, intensity, shape)
    key = dict(city=city, scenario=scenario, intensity=intensity, shape=shape, grid_fee_usd_kw_year=fee,
               hours=len(frame), idle_mean_mw=idle_mean, inf_online_mwh=float(on.sum()), inf_offline_mwh=float(off.sum()),
               inf_peak_mw=float((on + off).max()), inf_cv=float((on + off).std() / max((on + off).mean(), 1e-12)))
    rows = []
    a, xs = solve(frame, p, city, on, off, fee=fee, mode='asap')
    if xs is None:
        rows.append(key | {'mode': 'seq', **a})
    else:
        s, _ = solve(frame, p, city, on, off, fee=fee, mode='cost', fixed_x=xs)
        rows.append(key | {'mode': 'seq', **s, 'asap_seconds': a['solve_seconds']})
    j, _ = solve(frame, p, city, on, off, fee=fee, mode='cost')
    rows.append(key | {'mode': 'joint', **j})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--asi', required=True)
    ap.add_argument('--project', default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--out', required=True)
    ap.add_argument('--cities', default='riyadh,frankfurt,singapore')
    ap.add_argument('--scenarios', default='D-TRN')
    ap.add_argument('--intensities', default='0,0.25,0.5,0.75,0.9')
    ap.add_argument('--shapes', default='flat,real,amp2')
    ap.add_argument('--fees', default='120,0')
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(args.inputs, args.project, c, s, float(i), sh, float(f), args.asi)
             for c in args.cities.split(',') for s in args.scenarios.split(',')
             for i in args.intensities.split(',') for sh in args.shapes.split(',') for f in args.fees.split(',')]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(case, t): t for t in tasks}
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows += r
            print(k, len(tasks), futs[fut][2:7], [x.get('status') for x in r], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'runs.csv', index=False, encoding='utf-8-sig')
    (out / 'run-config.json').write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), 'utf-8')


if __name__ == '__main__':
    main()
