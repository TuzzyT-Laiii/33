import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import faulthandler
import json
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

CONTEXT = {}
STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def initialize_worker():
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    faulthandler.enable()


def write_progress(path, progress):
    staging = path.with_suffix('.writing')
    staging.write_text(json.dumps(progress))
    staging.replace(path)


def prepare(args):
    sys.path[:0] = [str(args.capacity / 'analysis'), str(args.joint / 'analysis'),
                    str(args.project / 'analysis')]
    from bounded_city_energy import feasible_dispatch
    from flex_econ_copt import crf, load_config
    p, own = load_config(args.joint)
    source = args.joint / 'inputs'
    plan = SimpleNamespace(reference=source / 'reference', resources=source / 'resources',
                           whole_jobs=args.capacity / 'inputs/exact-jobs',
                           capacities=args.capacity / 'results/capacity')
    manifest = json.loads((plan.resources / 'resource-manifest.json').read_text())
    annuity = crf(p['discount_rate'], p['compute_life_years']) + p['compute_fom_fraction']
    scenarios = {}
    for scenario, fleet in manifest.items():
        certificate = json.loads((plan.capacities / f'{scenario}-w12-s48.json').read_text())
        template = pd.read_parquet(args.inputs / 'templates' / f'{scenario}.parquet')
        city = pd.read_parquet(source / 'cities' / f'frankfurt-{scenario}.parquet')
        np.testing.assert_array_equal(template.source_hour_index, city.source_hour_index)
        np.testing.assert_allclose(template[['base', 'rigid', 'unknown', 'whole', 'split']],
                                   city[['base', 'rigid', 'unknown', 'whole', 'split']], atol=1e-8, rtol=0)
        dispatch, checks, _, _ = feasible_dispatch(plan, scenario, fleet, certificate, city)
        unit = own['cards']['server_price_usd'][scenario[0]]['value'] * annuity
        scenarios[scenario] = dict(fleet=fleet, certificate=certificate, template=template,
                                   dispatch=dispatch[:, 2], unit=unit, checks=checks,
                                   reuse_baseline=bool(np.max(np.abs(template.base - fleet['baseline_units'] *
                                                                    fleet['idle_mw_per_unit'])) < 1e-12))
    meta = pd.read_parquet(args.inputs / 'selected-grids.parquet').set_index('grid_id')
    reference = pd.read_parquet(args.reference)
    reference = reference[reference['mode'] == 'baseline']
    if (len(reference) != 4842 * 11 or reference.duplicated(['grid_id', 'scenario']).any() or
            reference.extra_task_mw.max() > 1e-8 or (reference.status != 'optimal').any()):
        raise ValueError('全球原机群基准不满足完整配对与零加卡条件')
    CONTEXT.update(p=p, scenarios=scenarios, meta=meta, inputs=args.inputs,
                   reference=reference.set_index(['grid_id', 'scenario']))


def solve_one(task):
    from flex_econ_copt import solve
    from global_flex_inputs import facility_frame
    grid, scenario = task
    started = time.monotonic()
    state = CONTEXT['scenarios'][scenario]
    frame = state['template'].copy()
    meta = CONTEXT['meta'].loc[grid]
    with np.load(CONTEXT['inputs'] / 'grids' / f'{grid}.npz') as source:
        weather = pd.DataFrame({k: source[k] for k in ('temperature_c', 'dew_point_c', 'pressure_pa')})
        index = frame.weather_hour_index.to_numpy(int)
        frame['pue'] = facility_frame(frame, weather).pue.to_numpy()
        frame['cf_pv'], frame['cf_wind'] = source['cf_pv'][index], source['cf_wind'][index]
    p = {**CONTEXT['p'], 'emission_factor': {grid: float(meta.carbon_t_mwh)}}
    n = len(frame)
    annual = 8760 / n
    e0 = float(np.dot(frame.pue, frame.it_mw) * annual)
    fleet, cert = state['fleet'], state['certificate']
    original = frame[['rigid', 'unknown', 'whole', 'split']].sum(axis=1).to_numpy()

    def energy(count, fixed_task):
        data = frame.copy()
        data['base'] = count * fleet['idle_mw_per_unit']
        if fixed_task is not None:
            data['rigid'] = fixed_task
            data[['unknown', 'whole', 'split']] = 0.
        # 松弛下界移除设备资源和任务功率容量限制，保留期限、工作量和供能约束。
        data['cap_it_mw'] = float(original.sum() + data.base.max() + 1.)
        data['it_mw'] = data[['base', 'rigid', 'unknown', 'whole', 'split']].sum(axis=1)
        row, _ = solve(data, p, grid, alpha=None, design='opt', whole_h=12, split_h=48)
        if row['status'] != 'optimal' or row['max_residual'] > 1e-5:
            raise RuntimeError(f'{grid} {scenario} 供能求解失败 {row}')
        return row

    if state['reuse_baseline']:
        baseline = CONTEXT['reference'].loc[(grid, scenario)].to_dict()
        baseline['facility_mwh'] = float(np.dot(frame.pue, original +
                                               fleet['baseline_units'] * fleet['idle_mw_per_unit']))
    else:
        baseline = energy(fleet['baseline_units'], original)
    feasible = energy(cert['servers'], state['dispatch'])
    relaxed = energy(cert['lower_servers'], None)
    unit = state['unit']
    a_total = fleet['baseline_units'] * unit + baseline['total_cost_usd'] * annual
    c_high = cert['servers'] * unit + feasible['total_cost_usd'] * annual
    c_low = cert['lower_servers'] * unit + relaxed['total_cost_usd'] * annual
    if c_low > c_high + 1.:
        raise RuntimeError('联合规划费用下界高于可行费用')
    low, high = (a_total - c_high) / e0, (a_total - c_low) / e0
    return dict(grid_id=grid, scenario=scenario, continent=meta.continent,
                hours=n, E0_mwh_year=e0, A_facility_mwh_year=baseline['facility_mwh'] * annual,
                servers_A=fleet['baseline_units'], servers_C_feasible=cert['servers'],
                servers_C_lower=cert['lower_servers'],
                equipment_saving_feasible_usd_year=(fleet['baseline_units'] - cert['servers']) * unit,
                baseline_energy_usd_year=baseline['total_cost_usd'] * annual,
                feasible_energy_usd_year=feasible['total_cost_usd'] * annual,
                relaxed_energy_usd_year=relaxed['total_cost_usd'] * annual,
                idle_energy_reduction_mwh_year=float(frame.pue.sum() * annual * fleet['idle_mw_per_unit'] *
                                                     (fleet['baseline_units'] - cert['servers'])),
                net_saving_lower_usd_mwh=low, net_saving_upper_usd_mwh=high,
                max_residual=max(baseline['max_residual'], feasible['max_residual'], relaxed['max_residual']),
                seconds=time.monotonic() - started, status='verified_bounds')


def main():
    parser = argparse.ArgumentParser()
    for name in ('project', 'joint', 'capacity', 'inputs', 'reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=48)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--scenarios', default='')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare(args)
    grids = [g for g in CONTEXT['meta'].index if (args.inputs / 'grids' / f'{g}.npz').is_file()]
    scenarios = args.scenarios.split(',') if args.scenarios else list(CONTEXT['scenarios'])
    tasks = [(g, s) for g in grids for s in scenarios]
    if args.limit:
        tasks = tasks[:args.limit]
    log = args.output / 'runs.jsonl'
    done = set()
    if log.exists():
        for line in log.read_text().splitlines():
            row = json.loads(line)
            done.add((row['grid_id'], row['scenario']))
    remaining = iter(t for t in tasks if t not in done)
    started = time.monotonic()
    initial = len(done)
    progress = dict(utc=datetime.now(timezone.utc).isoformat(), finished=len(done), total=len(tasks),
                    groups_per_hour=0.)
    signal.signal(signal.SIGTERM, request_stop)
    with log.open('a', buffering=1) as output, ProcessPoolExecutor(args.workers, initializer=initialize_worker) as pool:
        pending = {}

        def submit():
            if STOP_REQUESTED:
                return
            task = next(remaining, None)
            if task is not None:
                pending[pool.submit(solve_one, task)] = task

        for _ in range(2 * args.workers):
            submit()
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                task = pending.pop(future)
                row = future.result()
                output.write(json.dumps(row, allow_nan=False) + '\n')
                done.add(task)
                submit()
            elapsed = time.monotonic() - started
            progress = dict(utc=datetime.now(timezone.utc).isoformat(), finished=len(done), total=len(tasks),
                            groups_per_hour=(len(done) - initial) / max(elapsed, .01) * 3600)
            write_progress(args.output / 'progress.json', progress)
    if STOP_REQUESTED:
        print(json.dumps(dict(checkpoint_saved=len(done), total=len(tasks))), flush=True)
        return
    result = pd.read_json(log, lines=True)
    if result.duplicated(['grid_id', 'scenario']).any() or len(result) != len(tasks):
        raise RuntimeError('全球计算配对键或完成数量不一致')
    result.to_parquet(args.output / 'global-capacity.parquet', index=False)
    write_progress(args.output / 'progress.json', progress)
    result.groupby('scenario').agg(
        grids=('grid_id', 'size'), lower_median=('net_saving_lower_usd_mwh', 'median'),
        upper_median=('net_saving_upper_usd_mwh', 'median'),
        lower_min=('net_saving_lower_usd_mwh', 'min'), upper_max=('net_saving_upper_usd_mwh', 'max'),
        max_residual=('max_residual', 'max')).to_csv(args.output / 'summary.csv')
    print(progress, flush=True)


if __name__ == '__main__':
    main()
