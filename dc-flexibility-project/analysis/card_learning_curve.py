"""第四节学习曲线（能加卡模型）：在卡价敏感性脚本上叠加 ATB 2025 供能成本路径。

原说明——卡价敏感性：平卡·电协同 / 加卡 / 自由配卡（含减卡）在同一调度模型下扫描卡价倍数 m 与增减服务器的待机比例。

模型沿用 #43 版 flex_econ_copt.solve（真实时钟截止期，K=调度前任务峰值）。本脚本只改两处：
  1. 服务器增量 extra 的下界可为负（自由配卡，允许少配到低于 K）；
  2. 增减服务器的待机功率乘以 idle_scale（1=全时待机，0=空闲可关机）。
模式：none（无柔性、extra=0，基准）、flat（柔性、extra=0）、add（柔性、0≤extra≤K）、free（柔性、−K≤extra≤K）。
flat/none 与 m、idle_scale 无关，每组只算一次。
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

R = Path('/data3/dataset/ljt/global-flex-econ-20261103')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'model'))            # 打过补丁的 flex_econ_copt 副本
sys.path.insert(1, str(R / 'project/analysis'))

import numpy as np
import pandas as pd

import flex_econ_copt as F

CITY_INPUTS = Path('/data3/dataset/ljt/flex-econ-20261102/out/inputs')
KEEP = ['status', 'total_cost_usd', 'energy_cost_usd', 'cost_cards', 'cost_grid_energy', 'cost_carbon', 'cost_grid_capacity',
        'cost_pv', 'cost_wind', 'cost_battery_power', 'cost_battery_energy', 'pv_mw', 'wind_mw', 'battery_mw', 'battery_mwh',
        'grid_cap_mw', 'grid_mwh', 'facility_mwh', 'cfe_hourly', 'emissions_t', 'peak_task_mw', 'extra_task_mw', 'extra_idle_mwh',
        'moved_whole_mwh', 'moved_split_mwh', 'hours', 'solve_seconds', 'max_residual']
CTX = {}


def cases(m_grid, idle_grid):
    out = [('none', None, None), ('flat', None, None)]
    for mode in ('add', 'free'):
        for m in m_grid:
            for s in idle_grid:
                out.append((mode, m, s))
    return out


def init(kind, cost_file=None):
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    p, own = F.load_config(R / 'project')
    costs = {'base-2024': None}
    if cost_file:
        costs.update(json.loads(Path(cost_file).read_text('utf-8'))['scenarios'])
    CTX.update(p=p, own=own, kind=kind, costs=costs)
    if kind == 'global':
        from global_flex_inputs import facility_frame
        CTX['facility_frame'] = facility_frame
        meta = pd.read_parquet(R / 'inputs/selected-grids.parquet').set_index('grid_id')
        CTX['carbon'] = meta.carbon_t_mwh.to_dict()
        CTX['templates'] = {}


def build(region, scenario):
    if CTX['kind'] == 'city':
        frame = pd.read_parquet(CITY_INPUTS / f'{region}-{scenario}.parquet')
        p = CTX['p']
    else:
        if scenario not in CTX['templates']:
            CTX['templates'][scenario] = pd.read_parquet(R / 'inputs/templates' / f'{scenario}.parquet')
        frame = CTX['templates'][scenario].copy()
        with np.load(R / 'inputs/grids' / f'{region}.npz') as data:
            weather = pd.DataFrame({k: data[k] for k in ('temperature_c', 'dew_point_c', 'pressure_pa')})
            index = frame.weather_hour_index.to_numpy(int)
            cooling = CTX['facility_frame'](frame, weather)
            frame['pue'], frame['facility_mw'] = cooling.pue.to_numpy(), cooling.facility_mw.to_numpy()
            frame['cf_pv'], frame['cf_wind'] = data['cf_pv'][index], data['cf_wind'][index]
        p = dict(CTX['p'])
        p['emission_factor'] = {region: CTX['carbon'][region]}
    frame.attrs['scenario'] = scenario
    return frame, p


def group(task):
    region, scenario, alpha, case_list = task
    frame, p0 = build(region, scenario)
    own = CTX['own']
    rows = []
    base_unit = dict(p0['annual_usd_per_mw'])
    for cost_name, mult in CTX['costs'].items():
        p = dict(p0)
        p['annual_usd_per_mw'] = {k: v * (mult or {}).get(k, 1.0) for k, v in base_unit.items()}
        rows += group_rows(region, scenario, alpha, case_list, frame, p, own, cost_name)
    return region, scenario, alpha, rows


def group_rows(region, scenario, alpha, case_list, frame, p, own, cost_name):
    both = own['flex_cases']['both']
    market = F.card_spec(p, own, frame, scenario, 1.0)
    rows = []
    for mode, m, s in case_list:
        spec = F.card_spec(p, own, frame, scenario, 0. if m is None else m)
        peak = spec['extra_max_mw']
        spec['idle_per_mw'] *= 1. if s is None else s
        if mode in ('none', 'flat'):
            spec['extra_max_mw'] = 0.
        spec['extra_min_mw'] = -peak if mode == 'free' else 0.
        wh, sh = (0, 0) if mode == 'none' else (both['whole_h'], both['split_h'])
        started = time.monotonic()
        try:
            summary, _ = F.solve(frame, p, region, alpha=alpha, design='opt', whole_h=wh, split_h=sh, cards=spec)
        except Exception as error:
            summary = {'status': 'error', 'error': repr(error)[:300]}
        row = {'region': region, 'scenario': scenario, 'alpha': alpha, 'cost_scenario': cost_name, 'mode': mode, 'm': m, 'idle_scale': s,
               'card_usd_per_mw_year_at_m1': market['usd_per_mw_year'], 'idle_per_mw_full': market['idle_per_mw'],
               'baseline_fleet_usd_year': spec['baseline_fleet_usd_year'], 'servers_per_mw': spec['servers_per_mw'],
               'wall_seconds': time.monotonic() - started}
        row.update({k: summary.get(k) for k in KEEP})
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=('city', 'global'), required=True)
    ap.add_argument('--regions', nargs='*')
    ap.add_argument('--regions-file', type=Path)
    ap.add_argument('--alphas', default='all')          # all 或 econ,1.0
    ap.add_argument('--m', default='0,0.01,0.03,0.1,0.3,0.5,1,1.5,2')
    ap.add_argument('--idle', default='1,0.5,0.1,0')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--cost-scenarios', help='learning-curves-atb2025.json；与 base-2024 一起逐组计算')
    args = ap.parse_args()
    p, _ = F.load_config(R / 'project')
    if args.alphas == 'all':
        alphas = list(p['alpha'])
    else:
        alphas = [None if a == 'econ' else float(a) for a in args.alphas.split(',')]
    regions = args.regions or (F.CITIES if args.kind == 'city' else None)
    if args.regions_file:
        regions = [x.strip() for x in args.regions_file.read_text().split() if x.strip()]
    case_list = cases([float(x) for x in args.m.split(',')], [float(x) for x in args.idle.split(',')])
    args.output.mkdir(parents=True, exist_ok=True)
    shards = args.output / 'shards'
    shards.mkdir(exist_ok=True)
    tag = lambda a: 'econ' if a is None else f'{a:g}'
    order = {'A-HPC': 0, 'C-SAI': 1, 'D-SAI': 2, 'B-SAI': 3}          # 慢的先发
    tasks = [(r, s, a, case_list) for s in sorted(F.SCENARIOS, key=lambda x: order.get(x, 9)) for r in regions for a in alphas
             if not (shards / f'{r}-{s}-{tag(a)}.jsonl').exists()]
    total = len(regions) * len(F.SCENARIOS) * len(alphas)
    print('groups', len(tasks), 'of', total, 'LP per group', len(case_list), flush=True)
    done, started = total - len(tasks), time.monotonic()
    with ProcessPoolExecutor(args.workers, initializer=init, initargs=(args.kind, args.cost_scenarios)) as pool:
        futures = [pool.submit(group, t) for t in tasks]
        for future in as_completed(futures):
            region, scenario, alpha, rows = future.result()
            tmp = shards / f'{region}-{scenario}-{tag(alpha)}.tmp'
            tmp.write_text('\n'.join(json.dumps(r, ensure_ascii=False, default=float) for r in rows) + '\n', 'utf-8')
            tmp.replace(shards / f'{region}-{scenario}-{tag(alpha)}.jsonl')
            done += 1
            (args.output / 'progress.json').write_text(json.dumps({'done': done, 'total': total,
                                                                   'elapsed_s': time.monotonic() - started}), 'utf-8')
    frames = [pd.read_json(f, lines=True) for f in sorted(shards.glob('*.jsonl'))]
    pd.concat(frames).to_parquet(args.output / 'runs.parquet', index=False)
    print('done', flush=True)


if __name__ == '__main__':
    main()
