"""第三节全球计算：在第一节全球抽样网格上重复 residual_flex_pilot 的配对比较。

每个（网格，情境）：按第一节全球输入构建逐小时帧（模板 + 网格气象/风光 + 冷却 PUE），叠加 ASI 推理需求（real 形状），
需求强度取各情境可承接上限（max_intensity.py，与网格无关）的 0/25%/50%/75%；供能为规划边界、经济最优（α 不设）、接入容量费 120 美元/kW·年；求解器 COPT。

ASI 曲线与情境窗口按 UTC 对齐：训练窗口整周平移 14 天；推理、云计算窗口原位；科研 AI 与 HPC 窗口长于 ASI，
只保留与 ASI 重叠的小时（不循环重放需求），I=0 在同一截断窗口上重算，保留率分子分母同窗口。
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CONTEXT = {}
SHIFT_DAYS = {'TRN': 14}


def setup_paths(project):
    sys.path.insert(0, str(Path(project) / 'analysis'))
    sys.path.insert(1, str(HERE))


def initialize(config, project, asi_path, fee, intensities, max_table=None, tiers=None):
    setup_paths(project)
    from flex_econ_copt import load_config
    from global_flex_inputs import settings
    from residual_flex_pilot import asi_inference
    own, paths = settings(config)
    parameters, _ = load_config(Path(project))
    run = paths['run_root']
    grids = pd.read_parquet(run / 'inputs/selected-grids.parquet')
    CONTEXT.update(own=own, run=run, parameters=parameters, fee=fee, intensities=intensities,
                   templates={s: pd.read_parquet(run / 'inputs/templates' / (s + '.parquet')) for s in own['scenarios']},
                   meta=grids.astype(object).where(grids.notna(), None).set_index('grid_id').to_dict('index'),
                   asi=asi_inference(asi_path),
                   tiers=tiers,
                   max_intensity=pd.read_csv(max_table).set_index('scenario').max_intensity.to_dict() if max_table else None)


def build_frame(grid, scenario):
    from global_flex_inputs import facility_frame
    frame = CONTEXT['templates'][scenario].copy()
    target = CONTEXT['run'] / 'inputs/grids' / (grid + '.npz')
    if not target.exists():
        return None
    with np.load(target) as data:
        weather = pd.DataFrame({name: data[name] for name in ('temperature_c', 'dew_point_c', 'pressure_pa')})
        index = frame.weather_hour_index.to_numpy(int)
        cooling = facility_frame(frame, weather)
        frame['pue'], frame['facility_mw'] = cooling.pue.to_numpy(), cooling.facility_mw.to_numpy()
        frame['cf_pv'], frame['cf_wind'] = data['cf_pv'][index], data['cf_wind'][index]
    return frame


def align(frame, scenario):
    """返回与 ASI 重叠的帧（必要时截断）及对应的在线、离线推理原始曲线。"""
    asi = CONTEXT['asi'].set_index('time_utc')
    shift = pd.Timedelta(days=SHIFT_DAYS.get(scenario.split('-')[1], 0))
    t = pd.to_datetime(frame.time_utc, utc=True) - shift
    covered = t.isin(asi.index).to_numpy()
    frame = frame.loc[covered].reset_index(drop=True)
    a = asi.reindex(t[covered])
    return frame, a.on.to_numpy(float), a.off.to_numpy(float), int((~covered).sum()), shift.days


def run_group(task):
    from residual_flex_pilot import solve
    grid, scenario = task
    meta = CONTEXT['meta'][grid]
    base = dict(grid_id=grid, scenario=scenario, lat=meta['lat'], lon=meta['lon'], iso3=meta['iso3'], country=meta['country'],
                continent=meta['continent'], carbon_t_mwh=meta['carbon_t_mwh'], grid_fee_usd_kw_year=CONTEXT['fee'], shape='real')
    if meta['carbon_t_mwh'] is None:
        return [dict(base, status='input_country_or_carbon_unknown')]
    frame = build_frame(grid, scenario)
    if frame is None:
        return [dict(base, status='input_no_land_points')]
    frame, on_raw, off_raw, dropped, shift = align(frame, scenario)
    p = dict(CONTEXT['parameters'])
    p['emission_factor'] = {grid: meta['carbon_t_mwh']}
    task0 = (frame.rigid + frame.unknown + frame.whole + frame.split).to_numpy()
    idle_mean = float(task0.max() - task0.mean())
    total_mean = on_raw.mean() + off_raw.mean()
    base.update(hours=len(frame), hours_dropped_outside_asi=dropped, asi_shift_days=shift, idle_mean_mw=idle_mean, K_mw=float(task0.max()))
    rows = []
    cap = CONTEXT['max_intensity'][scenario] if CONTEXT['max_intensity'] else None
    if CONTEXT.get('tiers'):
        # 第二节卖推理：只算不协同（顺序调度）。max = 卖到可承接上限；ideal = 每小时闲置全部作在线负荷售出
        for tier in CONTEXT['tiers']:
            if tier == 'ideal':
                on, off = np.clip(task0.max() - task0, 0, None), np.zeros(len(frame))
                intensity = float(on.mean() / idle_mean)
            else:
                intensity = cap * 0.999  # 上限表保留 4 位小数，四舍五入可能略超可行边界
                c = intensity * idle_mean / total_mean
                on, off = c * on_raw, c * off_raw
            key = dict(base, tier=tier, intensity=intensity, max_intensity=cap, inf_online_mwh=float(on.sum()), inf_offline_mwh=float(off.sum()))
            a, xs = solve(frame, p, grid, on, off, fee=CONTEXT['fee'], mode='asap', solver='copt')
            if xs is None:
                seq = dict(a)
            else:
                seq, _ = solve(frame, p, grid, on, off, fee=CONTEXT['fee'], mode='cost', fixed_x=xs, solver='copt')
            rows.append(dict(key, mode='seq', **seq))
        return rows
    for level in CONTEXT['intensities']:
        intensity = level * cap if cap is not None else level
        c = intensity * idle_mean / total_mean
        on, off = c * on_raw, c * off_raw
        key = dict(base, intensity=intensity, intensity_of_max=level if cap is not None else None, max_intensity=cap,
                   inf_online_mwh=float(on.sum()), inf_offline_mwh=float(off.sum()))
        a, xs = solve(frame, p, grid, on, off, fee=CONTEXT['fee'], mode='asap', solver='copt')
        if xs is None:
            seq = dict(a)
        else:
            seq, _ = solve(frame, p, grid, on, off, fee=CONTEXT['fee'], mode='cost', fixed_x=xs, solver='copt')
            seq['asap_seconds'] = a['solve_seconds']
        joint, _ = solve(frame, p, grid, on, off, fee=CONTEXT['fee'], mode='cost', solver='copt')
        rows.append(dict(key, mode='seq', **seq))
        rows.append(dict(key, mode='joint', **joint))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--project', required=True, type=Path, help='第一节全球计算使用的项目副本（含 compute_load）')
    ap.add_argument('--asi', required=True)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--intensities', default='0,0.25,0.5,0.75')
    ap.add_argument('--fee', type=float, default=120.)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--grid-ids', type=Path, help='JSON {"grid_ids": [...]}，用于试跑')
    ap.add_argument('--scenarios', default='')
    ap.add_argument('--tiers', default='', help='如 ideal,max：第二节三层补算，只算顺序调度')
    ap.add_argument('--max-table', help='max_intensity.py 输出；给出时 --intensities 解释为上限的比例')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    intensities = [float(v) for v in args.intensities.split(',')]
    tiers = args.tiers.split(',') if args.tiers else None
    initialize(args.config, args.project, args.asi, args.fee, intensities, args.max_table, tiers)
    scenarios = args.scenarios.split(',') if args.scenarios else CONTEXT['own']['scenarios']
    grids = json.loads(args.grid_ids.read_text('utf-8'))['grid_ids'] if args.grid_ids else list(CONTEXT['meta'])
    runs = args.out / 'runs.jsonl'
    done = set()
    if runs.exists():
        for line in runs.read_text('utf-8').splitlines():
            r = json.loads(line)
            done.add((r['grid_id'], r['scenario']))
    tasks = [(g, s) for g in grids for s in scenarios if (g, s) not in done]
    total = len(grids) * len(scenarios)
    (args.out / 'run-config.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, ensure_ascii=False, indent=2), 'utf-8')
    started, finished, errors = time.time(), len(done), 0
    with runs.open('a', encoding='utf-8') as sink, ProcessPoolExecutor(
            args.workers, initializer=initialize, initargs=(args.config, args.project, args.asi, args.fee, intensities, args.max_table, tiers)) as ex:
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
                    rows = [dict(grid_id=task[0], scenario=task[1], status='error', error=traceback.format_exc()[-2000:])]
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


if __name__ == '__main__':
    main()
