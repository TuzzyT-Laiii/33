"""第四节加卡检验接入外部推理需求（补 PR #80 的局限“卡价与待机只在八城市、不含外部推理需求的加卡模型中检验”）。

模型：第三节联合调度（residual_flex_pilot.solve，mode='cost'，规划边界、经济最优、接入容量费 120、碳价 100、COPT），
加一个增减卡变量 e：任务容量 K + e，卡价与待机口径同 card_learning_curve.py / flex_econ_copt.card_spec
（每 MW 年化卡价 = m × 服务器价 × 台数 ÷ 峰值 × (CRF(4 年) + 3%)，增减卡待机 = s × 原机群每 MW 待机，全时计入）。
外部推理需求同第三节：ASI 在线 + 离线（24 h）推理，强度取各情境可承接上限的 0/25/50/75%，窗口对齐同 residual_flex_global.align。
供能成本：2024 与 ATB 2025 六条路径（learning-curves-atb2025.json）。

  --kind city    八城市 × 11 情境：flat（e=0）+ add（0 ≤ e ≤ K）/ free（−K ≤ e ≤ K）× m × s 全扫描；
  --kind global  全球 4,842 网格 × 11 情境：只算最有利于加卡的角点（默认 add、m=0.2、s=0）。
                 卡价越高、待机越大，加卡只会更少；学习曲线路径（m 0.22–1.92、s 0.60–1.0）都不比角点有利。
同一（区域，情境）的 flat 联合调度成本即第三节／第四节已有 joint 结果，global 默认不重算（--with-flat 可加）。
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import residual_flex_global as G  # noqa: E402

CITY_INPUTS = Path('/data3/dataset/ljt/flex-econ-20261102/out/inputs')
CARD = {}


def initialize(args):
    G.initialize(args['config'], args['project'], args['asi'], args['fee'], args['intensities'], args['max_table'],
                 None, None, args['cost_scenarios'])
    from flex_econ_copt import load_config
    p, own = load_config(Path(args['project']))
    CARD.update(p=p, own=own, kind=args['kind'], cases=args['cases'], city_inputs=args['city_inputs'])


def cases(kind, m_grid, s_grid, with_flat):
    out = [('flat', None, None)] if (kind == 'city' or with_flat) else []
    modes = ('add', 'free') if kind == 'city' else ('add',)
    return out + [(mode, m, s) for mode in modes for m in m_grid for s in s_grid]


def run_group(task):
    from flex_econ_copt import card_spec
    from residual_flex_pilot import solve
    region, scenario, cost_name = task
    C = G.CONTEXT
    if CARD['kind'] == 'city':
        frame = pd.read_parquet(Path(CARD['city_inputs']) / f'{region}-{scenario}.parquet')
        base = dict(region=region, scenario=scenario)
        p = dict(C['parameters'])
    else:
        meta = C['meta'][region]
        base = dict(region=region, scenario=scenario, lat=meta['lat'], lon=meta['lon'], iso3=meta['iso3'],
                    continent=meta['continent'], carbon_t_mwh=meta['carbon_t_mwh'])
        if meta['carbon_t_mwh'] is None:
            return [dict(base, cost_scenario=cost_name, status='input_country_or_carbon_unknown')]
        frame = G.build_frame(region, scenario)
        if frame is None:
            return [dict(base, cost_scenario=cost_name, status='input_no_land_points')]
        p = dict(C['parameters'])
        p['emission_factor'] = {region: meta['carbon_t_mwh']}
    frame.attrs['scenario'] = scenario
    market = card_spec(CARD['p'], CARD['own'], frame, scenario, 1.0)  # 用完整窗口峰值折算每 MW 卡价，与第四节加卡模型一致
    frame, on_raw, off_raw, dropped, shift = G.align(frame, scenario)
    task0 = (frame.rigid + frame.unknown + frame.whole + frame.split).to_numpy()
    K, idle_mean = float(task0.max()), float(task0.max() - task0.mean())
    total_mean = on_raw.mean() + off_raw.mean()
    cap = C['max_intensity'][scenario]
    mult = (C['cost_scenarios'] or {}).get(cost_name) or {}
    p['annual_usd_per_mw'] = {k: v * mult.get(k, 1.0) for k, v in C['parameters']['annual_usd_per_mw'].items()}
    base.update(cost_scenario=cost_name, carbon_price_usd_t=p['carbon_price_usd_t'], hours=len(frame), hours_dropped_outside_asi=dropped,
                K_mw=K, idle_mean_mw=idle_mean, max_intensity=cap, card_usd_per_mw_year_at_m1=market['usd_per_mw_year'],
                idle_per_mw_full=market['idle_per_mw'], baseline_fleet_usd_year=market['baseline_fleet_usd_year'])
    rows = []
    for level in C['intensities']:
        c = level * cap * idle_mean / total_mean
        on, off = c * on_raw, c * off_raw
        key = dict(base, intensity_of_max=level, intensity=level * cap, inf_online_mwh=float(on.sum()), inf_offline_mwh=float(off.sum()))
        for mode, m, s in CARD['cases']:
            cards = None if mode == 'flat' else {'usd_per_mw_year': m * market['usd_per_mw_year'], 'idle_per_mw': s * market['idle_per_mw'],
                                                 'min_mw': -K if mode == 'free' else 0., 'max_mw': K}
            out, _ = solve(frame, p, region, on, off, fee=C['fee'], mode='cost', solver='copt', cards=cards)
            out.setdefault('extra_task_mw', 0.)
            out.setdefault('cost_cards', 0.)
            rows.append(dict(key, mode=mode, m=m, idle_scale=s, **out))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=('city', 'global'), required=True)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--project', required=True, type=Path)
    ap.add_argument('--asi', required=True)
    ap.add_argument('--max-table', required=True)
    ap.add_argument('--cost-scenarios', required=True)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--intensities', default='0,0.25,0.5,0.75')
    ap.add_argument('--fee', type=float, default=120.)
    ap.add_argument('--m', default=None, help='city 默认 0.2,0.5,1,2；global 默认 0.2')
    ap.add_argument('--idle', default=None, help='city 默认 1,0.5,0；global 默认 0')
    ap.add_argument('--with-flat', action='store_true')
    ap.add_argument('--regions', default='', help='逗号分隔；缺省为全部城市或全部网格')
    ap.add_argument('--grid-ids', type=Path, help='JSON {"grid_ids": [...]}，用于试跑')
    ap.add_argument('--scenarios', default='')
    ap.add_argument('--city-inputs', default=str(CITY_INPUTS))
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--modes', default='', help='2026-11-21：覆盖 global 的模式，如 free')
    ap.add_argument('--costs', default='', help='2026-11-21：只算这些成本路径，如 base-2024')
    args = ap.parse_args()
    city = args.kind == 'city'
    m_grid = [float(v) for v in (args.m or ('0.2,0.5,1,2' if city else '0.2')).split(',')]
    s_grid = [float(v) for v in (args.idle or ('1,0.5,0' if city else '0')).split(',')]
    init = dict(config=args.config, project=args.project, asi=args.asi, fee=args.fee, max_table=args.max_table,
                intensities=[float(v) for v in args.intensities.split(',')], cost_scenarios=args.cost_scenarios,
                kind=args.kind, cases=cases(args.kind, m_grid, s_grid, args.with_flat), city_inputs=args.city_inputs)
    if args.modes:
        init['cases'] = [c for c in init['cases'] if c[0] == 'flat'] + [(md, m, s) for md in args.modes.split(',') for m in m_grid for s in s_grid]
    initialize(init)
    costs = ['base-2024'] + list(G.CONTEXT['cost_scenarios'])
    if args.costs:
        costs = [c for c in costs if c in args.costs.split(',')]
    G.CONTEXT['cost_scenarios'] = {'base-2024': None, **G.CONTEXT['cost_scenarios']}
    scenarios = args.scenarios.split(',') if args.scenarios else G.CONTEXT['own']['scenarios']
    if args.regions:
        regions = args.regions.split(',')
    elif args.grid_ids:
        regions = json.loads(args.grid_ids.read_text('utf-8'))['grid_ids']
    else:
        from flex_econ_copt import CITIES
        regions = list(CITIES) if city else list(G.CONTEXT['meta'])
    args.out.mkdir(parents=True, exist_ok=True)
    runs = args.out / 'runs.jsonl'
    done = set()
    if runs.exists():
        for line in runs.read_text('utf-8').splitlines():
            r = json.loads(line)
            done.add((r['region'], r['scenario'], r['cost_scenario']))
    slow = {'A-HPC': 0, 'C-SAI': 1, 'D-SAI': 2, 'B-SAI': 3}  # 慢的先发
    tasks = [(r, s, c) for s in sorted(scenarios, key=lambda x: slow.get(x, 9)) for r in regions for c in costs
             if (r, s, c) not in done]
    total = len(regions) * len(scenarios) * len(costs)
    (args.out / 'run-config.json').write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()}, 'cases': init['cases']},
                                                         ensure_ascii=False, indent=2), 'utf-8')
    print('groups', len(tasks), 'of', total, 'LP per group', len(init['cases']) * len(init['intensities']), flush=True)
    started, finished, errors = time.time(), len(done), 0
    # 子进程里 G.CONTEXT['cost_scenarios'] 也要带 base-2024
    with runs.open('a', encoding='utf-8') as sink, ProcessPoolExecutor(args.workers, initializer=worker_init, initargs=(init,)) as ex:
        queue, pending = iter(tasks), {}
        for task in queue:
            pending[ex.submit(run_group, task)] = task
            if len(pending) >= args.workers * 3:
                break
        while pending:
            complete, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in complete:
                task = pending.pop(fut)
                try:
                    rows = fut.result()
                except Exception:
                    errors += 1
                    rows = [dict(region=task[0], scenario=task[1], cost_scenario=task[2], status='error', error=traceback.format_exc()[-2000:])]
                for r in rows:
                    sink.write(json.dumps(r, ensure_ascii=False, default=float) + '\n')
                sink.flush()
                finished += 1
                nxt = next(queue, None)
                if nxt is not None:
                    pending[ex.submit(run_group, nxt)] = nxt
            elapsed = time.time() - started
            rate = (finished - len(done)) / max(elapsed, 1)
            (args.out / 'progress.json').write_text(json.dumps({
                'utc': pd.Timestamp.utcnow().isoformat(), 'finished': finished, 'total': total, 'errors': errors,
                'groups_per_hour': rate * 3600, 'eta_hours': (total - finished) / max(rate, 1e-9) / 3600}, indent=2), 'utf-8')


def worker_init(init):
    initialize(init)
    G.CONTEXT['cost_scenarios'] = {'base-2024': None, **G.CONTEXT['cost_scenarios']}


if __name__ == '__main__':
    main()
