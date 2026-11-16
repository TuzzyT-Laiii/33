import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import faulthandler
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import pandas as pd

CONTEXT = {}
STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def initialize_worker(args):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    faulthandler.enable()
    prepare(args)


def write_progress(path, progress):
    staging = path.with_suffix('.writing')
    staging.write_text(json.dumps(progress))
    staging.replace(path)


def prepare(args):
    sys.path.append(str(args.project / 'analysis'))
    from flex_econ_copt import load_config
    p, own = load_config(args.project)
    manifest = json.loads((args.resources / 'resource-manifest.json').read_text())
    scenarios = {}
    for scenario, fleet in manifest.items():
        frame = pd.read_parquet(args.inputs / 'templates' / f'{scenario}.parquet')
        task = frame[['rigid', 'unknown', 'whole', 'split']].sum(axis=1).to_numpy()
        extra = np.zeros(len(frame))
        sold = 0.
        if fleet['gpus_per_unit']:
            resource = pd.read_parquet(args.resources / f'{scenario}.parquet')
            np.testing.assert_array_equal(frame.source_hour_index, resource.source_hour_index)
            idle = fleet['baseline_units'] * fleet['gpus_per_unit'] - resource.gpu_hours.to_numpy()
            if idle.min() < -1e-6:
                raise ValueError('原时序 GPU 需求超出机群容量')
            idle = np.maximum(idle, 0)
            extra = idle * task.sum() / resource.gpu_hours.sum()
            sold = float(idle.sum())
            if np.max(frame.base.to_numpy() + task + extra - frame.cap_it_mw.to_numpy()) > 1e-7:
                raise ValueError('全卖情境超出既有 IT 功率上限')
        scenarios[scenario] = dict(frame=frame, extra=extra, sold=sold)
    reference = pd.read_parquet(args.reference)
    if len(reference) != 2 * 4842 * 11 or reference.extra_task_mw.max() > 1e-8:
        raise ValueError('已有基准不满足零新增容量与完整配对条件')
    worker_state = args.output / 'worker-state'
    worker_state.mkdir(exist_ok=True)
    CONTEXT.update(p=p, own=own, scenarios=scenarios, inputs=args.inputs, worker_state=worker_state,
                   reference=reference.set_index(['grid_id', 'scenario', 'mode']),
                   meta=pd.read_parquet(args.inputs / 'selected-grids.parquet').set_index('grid_id'))


def one(task):
    from global_flex_inputs import facility_frame
    from flex_econ_copt import solve, card_spec
    grid, scenario = task
    started = time.monotonic()
    state = CONTEXT['scenarios'][scenario]
    frame = state['frame'].copy()
    with np.load(CONTEXT['inputs'] / 'grids' / f'{grid}.npz') as raw:
        weather = pd.DataFrame({k: raw[k] for k in ('temperature_c', 'dew_point_c', 'pressure_pa')})
        hours = frame.weather_hour_index.to_numpy(int)
        frame['pue'] = facility_frame(frame, weather).pue.to_numpy()
        frame['cf_pv'], frame['cf_wind'] = raw['cf_pv'][hours], raw['cf_wind'][hours]
    p = {**CONTEXT['p'], 'emission_factor': {grid: float(CONTEXT['meta'].loc[grid].carbon_t_mwh)}}
    e0 = float(np.dot(frame.it_mw, frame.pue))
    rows = []
    for mode in (['baseline', 'coordination', 'sell_all'] if state['sold'] else ['baseline', 'coordination']):
        (CONTEXT['worker_state'] / f'{os.getpid()}.json').write_text(json.dumps(dict(
            grid_id=grid, scenario=scenario, mode=mode, utc=datetime.now(timezone.utc).isoformat())))
        data = frame.copy()
        cards = None
        if mode == 'sell_all':
            data['rigid'] = data.rigid + state['extra']
            data['it_mw'] = data.it_mw + state['extra']
        elif mode == 'coordination':
            cards = card_spec(p, CONTEXT['own'], data, scenario, 0.)
            cards['extra_max_mw'] = 0.
        wh, sh = (12, 48) if mode == 'coordination' else (0, 0)
        if mode == 'sell_all':
            selected_p, multipliers, factor = p, [1., .5, 1.5], 1.
        else:
            selected_p = {**p, 'grid_price_usd_mwh': .5 * p['grid_price_usd_mwh']}
            multipliers, factor = [1., 3.], .5
            existing = CONTEXT['reference'].loc[(grid, scenario, mode)]
            fields = ['total_cost_usd', 'grid_mwh', 'pv_mw', 'wind_mw', 'battery_mw',
                      'battery_mwh', 'grid_cap_mw', 'max_residual']
            rows.append(dict(grid_id=grid, scenario=scenario, mode=mode, hours=len(frame),
                             E0_mwh=e0, sold_gpu_hours=0., multiplier=1.,
                             **{key: float(existing[key]) for key in fields}))
        row, _ = solve(data, selected_p, grid, alpha=None, design='opt', whole_h=wh, split_h=sh,
                       cards=cards, grid_price_multipliers=multipliers)
        if row['status'] != 'optimal' or row['max_residual'] > 1e-5:
            raise RuntimeError(f'电价求解失败 {grid} {scenario} {mode} {row}')
        for price in row['grid_price_scan']:
            price = {**price, 'multiplier': price['multiplier'] * factor}
            rows.append(dict(grid_id=grid, scenario=scenario, mode=mode, hours=len(frame),
                             E0_mwh=e0, sold_gpu_hours=state['sold'] if mode == 'sell_all' else 0.,
                             **price))
    return rows, time.monotonic() - started


def main():
    parser = argparse.ArgumentParser()
    for name in ('project', 'inputs', 'resources', 'reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--scenarios', default='')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare(args)
    grids = [g for g in CONTEXT['meta'].index if (args.inputs / 'grids' / f'{g}.npz').exists()]
    scenarios = args.scenarios.split(',') if args.scenarios else list(CONTEXT['scenarios'])
    tasks = [(g, s) for g in grids for s in scenarios]
    if args.limit:
        tasks = tasks[:args.limit]
    log = args.output / 'groups.jsonl'
    done = set()
    if log.exists():
        for line in log.read_text().splitlines():
            group = json.loads(line)
            done.add((group['grid_id'], group['scenario']))
    remaining = iter(t for t in tasks if t not in done)
    initial, started = len(done), time.monotonic()
    progress = dict(utc=datetime.now(timezone.utc).isoformat(), finished=len(done), total=len(tasks),
                    groups_per_hour=0.)
    signal.signal(signal.SIGTERM, request_stop)
    with log.open('a', buffering=1) as output:
        while not STOP_REQUESTED:
            submitted = 0
            # 独立启动解释器并按批次释放本地求解器状态，保留每组已完成结果。
            with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                                     initializer=initialize_worker, initargs=(args,)) as pool:
                pending = {}

                def submit():
                    nonlocal submitted
                    if STOP_REQUESTED or submitted >= 16 * args.workers:
                        return
                    task = next(remaining, None)
                    if task is not None:
                        pending[pool.submit(one, task)] = task
                        submitted += 1

                for _ in range(2 * args.workers):
                    submit()
                if not pending:
                    break
                while pending:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        task = pending.pop(future)
                        rows, seconds = future.result()
                        output.write(json.dumps(dict(grid_id=task[0], scenario=task[1], rows=rows, seconds=seconds),
                                                allow_nan=False) + '\n')
                        done.add(task)
                        submit()
                    progress = dict(utc=datetime.now(timezone.utc).isoformat(), finished=len(done), total=len(tasks),
                                    groups_per_hour=(len(done) - initial) / max(.01, time.monotonic() - started) * 3600)
                    write_progress(args.output / 'progress.json', progress)
    if STOP_REQUESTED:
        print(json.dumps(dict(checkpoint_saved=len(done), total=len(tasks))), flush=True)
        return
    groups = [json.loads(line) for line in log.read_text().splitlines()]
    result = pd.DataFrame([row for group in groups for row in group['rows']])
    if result.duplicated(['grid_id', 'scenario', 'mode', 'multiplier']).any():
        raise RuntimeError('电价配对键重复')
    result.to_parquet(args.output / 'electricity-price.parquet', index=False)
    write_progress(args.output / 'progress.json', progress)
    print(progress, flush=True)


if __name__ == '__main__':
    main()
