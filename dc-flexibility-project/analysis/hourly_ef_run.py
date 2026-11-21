"""逐时电网碳强度敏感性（不接推理、碳价 100、2024 成本、规划边界）。
每个（网格，情境）：A 年均因子（同现有结果）；B 同一调度按逐时因子事后核算；C 逐时因子进入碳成本后重新优化。
逐时序列按小时缩放到该网格的 Ember 年均因子（只检验时段差异）；另存未缩放的原始均值以供对照。
用法：python hourly_ef_run.py --ci hourly-ci-2025.csv --assign assign.json --out runs.jsonl --workers 40"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import residual_flex_global as G  # noqa: E402

R = Path('/data3/dataset/ljt/global-flex-econ-20261103')
CI = {}


def init(ci_path):
    G.initialize(R / 'project/analysis/global-flex-econ-20261103.json', R / 'project',
                 str(HERE.parent / 'inputs/asi-D-INF-native-hourly.parquet'), 120., [0.], str(HERE / 'max-intensity.csv'))
    ci = pd.read_csv(ci_path, parse_dates=['time_utc'])
    CI.update({k: ci[k].to_numpy(float) for k in ci.columns if k != 'time_utc'})


def run(task):
    from residual_flex_pilot import solve
    grid, scenario, zone = task
    meta = G.CONTEXT['meta'][grid]
    frame = G.build_frame(grid, scenario)
    if frame is None:
        return [dict(grid_id=grid, scenario=scenario, status='input_no_land_points')]
    frame, on_raw, off_raw, _, _ = G.align(frame, scenario)
    n = len(frame)
    on, off = np.zeros(n), np.zeros(n)
    hour = frame.weather_hour_index.to_numpy(int) % 8760          # 风光与碳强度用同一气象年小时
    raw = CI[zone][hour]
    ef0 = float(meta['carbon_t_mwh'])
    efh = raw / raw.mean() * ef0
    p = dict(G.CONTEXT['parameters'])
    out = dict(grid_id=grid, scenario=scenario, zone=zone, iso3=meta['iso3'], ef_const=ef0, ci_raw_mean_g=float(raw.mean()),
               ci_cv=float(raw.std() / raw.mean()), hours=n)
    res = {}
    for tag, ef in (('const', ef0), ('hourly', efh)):
        p['emission_factor'] = {grid: ef}
        a, xs = solve(frame, p, grid, on, off, fee=120., mode='asap', solver='copt')
        s, hs = solve(frame, p, grid, on, off, fee=120., mode='cost', fixed_x=xs, solver='copt')
        j, hj = solve(frame, p, grid, on, off, fee=120., mode='cost', solver='copt')
        if s['status'] != 'optimal' or j['status'] != 'optimal':
            return [dict(out, status=f'{tag}:{s["status"]}/{j["status"]}')]
        res[tag] = (s, hs, j, hj)
    s, hs, j, hj = res['const']
    sh, hsh, jh, hjh = res['hourly']
    e0 = float(s['facility_mwh'])
    out.update(status='optimal', E0_window_mwh=e0,
               V_const=(s['total_cost_usd'] - j['total_cost_usd']) / e0, V_hourly=(sh['total_cost_usd'] - jh['total_cost_usd']) / e0,
               em_seq_const=float(np.dot(hs['grid'], np.full(n, ef0))), em_joint_const=float(np.dot(hj['grid'], np.full(n, ef0))),
               em_seq_post=float(np.dot(hs['grid'], efh)), em_joint_post=float(np.dot(hj['grid'], efh)),
               em_seq_opt=float(np.dot(hsh['grid'], efh)), em_joint_opt=float(np.dot(hjh['grid'], efh)),
               grid_seq=float(hs['grid'].sum()), grid_joint=float(hj['grid'].sum()),
               grid_seq_opt=float(hsh['grid'].sum()), grid_joint_opt=float(hjh['grid'].sum()),
               cap_seq=s['grid_cap_mw'], cap_joint=j['grid_cap_mw'], cap_seq_opt=sh['grid_cap_mw'], cap_joint_opt=jh['grid_cap_mw'],
               check_seq_em=s['emissions_t'] - float(np.dot(hs['grid'], np.full(n, ef0))))
    return [out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ci', required=True)
    ap.add_argument('--assign', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--workers', type=int, default=40)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    assign = json.loads(a.assign.read_text('utf-8'))
    init(a.ci)
    tasks = [(g, s, z) for g, z in assign.items() for s in G.CONTEXT['own']['scenarios']]
    done = set()
    if a.out.exists():
        done = {(r['grid_id'], r['scenario']) for r in map(json.loads, a.out.read_text().splitlines())}
    tasks = [t for t in tasks if t[:2] not in done][: a.limit or None]
    print('tasks', len(tasks), flush=True)
    with a.out.open('a') as sink, ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.ci,)) as ex:
        futs = {ex.submit(run, t): t for t in tasks}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                rows = f.result()
            except Exception:
                t = futs[f]
                rows = [dict(grid_id=t[0], scenario=t[1], status='error', error=traceback.format_exc()[-1500:])]
            for r in rows:
                sink.write(json.dumps(r, default=float) + '\n')
            sink.flush()
            if k % 100 == 0:
                print(k, flush=True)


if __name__ == '__main__':
    main()
