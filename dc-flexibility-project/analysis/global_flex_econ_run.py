import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
from hashlib import sha256
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import resource
import shutil
import time
import traceback

import numpy as np
import pandas as pd
import psutil

from audit_real_time_flex_dispatch import fifo_check
from flex_econ_copt import card_spec, load_config, solve
from global_flex_inputs import digest, facility_frame, settings, utc_now, write_json


ASSETS = ['pv_mw', 'wind_mw', 'battery_mw', 'battery_mwh', 'grid_cap_mw']
CONTEXT = {}


def case_key(alpha, boundary, flex):
    return f'a{alpha}-{boundary}-{flex}'


def load_rows(path):
    if not path.exists():
        return []
    payload = path.read_bytes()
    if payload and not payload.endswith(b'\n'):
        # 中断时只删除尚未写完的最后一条JSON记录，完整结果保持原字节。
        boundary = payload.rfind(b'\n') + 1
        with path.open('r+b') as stream:
            stream.truncate(boundary)
        payload = payload[:boundary]
    return [json.loads(line) for line in payload.splitlines()]


def append_row(path, row):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=lambda value: value.item()) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def initialize(config, output, events, keep_grids, audit_all):
    own, paths = settings(config)
    project = Path(config).parent.parent
    parameters, city_own = load_config(project)
    scope = json.loads((Path(output) / 'run-manifest.json').read_text('utf-8'))['scope']
    input_hashes = pd.read_csv(paths['run_root'] / f'inputs/grid-input-index-{scope}.csv').set_index('grid_id').sha256.to_dict()
    templates = {scenario: pd.read_parquet(paths['run_root'] / 'inputs/templates' / (scenario + '.parquet')) for scenario in own['scenarios']}
    grid_data = pd.read_parquet(paths['run_root'] / 'inputs/selected-grids.parquet')
    metadata = grid_data.astype(object).where(grid_data.notna(), None).set_index('grid_id').to_dict('index')
    land_counts = pd.read_parquet(paths['run_root'] / 'inputs/grid-points.parquet').groupby('grid_id').is_land.sum()
    for grid in metadata:
        metadata[grid]['land_points'] = int(land_counts[grid])
    CONTEXT.update(own=own, paths=paths, parameters=parameters, city_own=city_own, templates=templates,
                   metadata=metadata, input_hashes=input_hashes, output=Path(output), events=events, keep_grids=set(keep_grids), audit_all=audit_all,
                   source_version=json.loads((Path(output) / 'active-source-version.json').read_text('utf-8'))['source_version'])


def build_frame(grid, scenario):
    run = CONTEXT['paths']['run_root']
    frame = CONTEXT['templates'][scenario].copy()
    target = run / 'inputs/grids' / (grid + '.npz')
    if not target.exists():
        if CONTEXT['metadata'][grid]['land_points'] == 0:
            return None, 'input_no_land_points', None
        raise FileNotFoundError('网格输入尚未构建：' + str(target))
    input_hash = digest(target)
    if input_hash != CONTEXT['input_hashes'][grid]:
        raise ValueError('网格输入与冻结索引SHA不一致：' + grid)
    with np.load(target) as data:
        weather = pd.DataFrame({name: data[name] for name in ('temperature_c', 'dew_point_c', 'pressure_pa')})
        index = frame.weather_hour_index.to_numpy(int)
        selected = weather.iloc[index]
        cooling = facility_frame(frame, weather)
        frame['pue'], frame['facility_mw'] = cooling.pue.to_numpy(), cooling.facility_mw.to_numpy()
        frame['cf_pv'], frame['cf_wind'] = data['cf_pv'][index], data['cf_wind'][index]
    frame.attrs.update(scenario=scenario, **{key: cooling.attrs[key] for key in (
        'raw_min_temperature_c', 'raw_max_temperature_c', 'low_temperature_clipped_hours', 'high_temperature_clipped_hours',
        'temperature_clipped_hours', 'dewpoint_clipped_hours')})
    return frame, 'ready', input_hash


def audit_dispatch(frame, summary, hourly, whole_h, split_h, alpha):
    tolerance = 1e-4
    clock = frame.source_hour_index.to_numpy(float)
    errors = []
    fifo = {}
    for category, hours in (('whole', whole_h), ('split', split_h)):
        fifo[category] = fifo_check(frame[category].to_numpy(float), hourly[category + '_exec_mw'].to_numpy(float), clock, hours, tolerance)
        record = fifo[category]
        errors.extend([record[key] for key in ('early_execution_mwh', 'late_execution_mwh', 'unserved_mwh', 'conservation_error_mwh')])
        errors.append(max(0., -record['minimum_cumulative_backlog_mwh']))
    task = frame.rigid.to_numpy() + frame.unknown.to_numpy() + hourly.whole_exec_mw.to_numpy() + hourly.split_exec_mw.to_numpy()
    capacity_error = max(0., float(np.max(task - summary['peak_task_mw'] - summary['extra_task_mw'])))
    expected = frame.pue.to_numpy() * (frame.base.to_numpy() + task + summary['extra_idle_mwh'] / len(frame))
    facility_error = float(np.max(np.abs(expected - hourly.facility_mw.to_numpy())))
    balance = hourly.pv_mw + hourly.wind_mw - hourly.charge_mw - hourly.curtail_mw + hourly.grid_mw + hourly.discharge_mw - hourly.facility_mw
    balance_error = float(balance.abs().max())
    alpha_error = 0. if alpha is None else max(0., float(hourly.grid_mw.sum() - (1 - alpha) * hourly.facility_mw.sum()))
    errors.extend([capacity_error, facility_error, balance_error, alpha_error, summary['max_residual']])
    return {'passed': bool(max(errors) <= tolerance), 'maximum_violation': float(max(errors)), 'fifo': fifo,
            'capacity_violation_mw': capacity_error, 'facility_violation_mw': facility_error,
            'balance_violation_mw': balance_error, 'alpha_violation_mwh': alpha_error, 'tolerance': tolerance}


def run_group(task):
    grid, scenario = task
    ctx, run = CONTEXT, CONTEXT['paths']['run_root']
    path = ctx['output'] / 'shards' / (grid + '-' + scenario + '.jsonl')
    rows = load_rows(path)
    done = {row['case_key']: row for row in rows}
    frame, input_status, input_hash = build_frame(grid, scenario)
    meta = ctx['metadata'][grid]
    base_fields = {key: meta[key] for key in ('lat', 'lon', 'block_id', 'block_lat', 'block_lon', 'iso3', 'country', 'continent', 'carbon_source', 'carbon_gco2_kwh', 'carbon_t_mwh', 'land_points',
                                           'country_assignment_method', 'nearest_boundary_distance_km', 'country_assignment_distance_km', 'm49_region')}
    base_fields.update(grid_id=grid, scenario=scenario, input_sha256=input_hash, execution_source_version=ctx['source_version'])
    if frame is not None:
        base_fields.update({key: frame.attrs[key] for key in ('raw_min_temperature_c', 'raw_max_temperature_c', 'low_temperature_clipped_hours',
                            'high_temperature_clipped_hours', 'temperature_clipped_hours', 'dewpoint_clipped_hours')})
    p = dict(ctx['parameters'])
    p['emission_factor'] = {grid: meta['carbon_t_mwh']}
    if meta['carbon_t_mwh'] is None:
        input_status = 'input_country_or_carbon_unknown'
    for alpha in ctx['own']['alpha']:
        baseline_key = case_key(alpha, 'planning', 'none')
        for boundary, flex in (('planning', 'none'), ('planning', 'both'), ('fixed', 'none'), ('fixed', 'both')):
            key = case_key(alpha, boundary, flex)
            if key in done:
                continue
            started = time.monotonic()
            summary = dict(base_fields, alpha=alpha, boundary=boundary, flex=flex, case_key=key,
                           supply='fixed_noflex_opt' if boundary == 'fixed' else 'planning_opt',
                           card_price='forced0' if flex == 'none' else '1.0', lp_performed=False, reused_baseline=False)
            baseline = done.get(baseline_key)
            if input_status != 'ready':
                summary.update(status=input_status, solve_seconds=0., hours=len(ctx['templates'][scenario]))
            elif boundary == 'fixed' and flex == 'none':
                summary = {**baseline, **summary, 'solve_seconds': 0., 'reused_baseline': True, 'source_case_key': baseline_key}
                if baseline['status'] == 'optimal':
                    expected_assets = [baseline[name] for name in ASSETS]
                    actual_assets = [summary[name] for name in ASSETS]
                    summary.update(fixed_asset_max_abs_difference=float(np.max(np.abs(np.asarray(actual_assets) - expected_assets))),
                                   fixed_asset_equal=bool(np.array_equal(actual_assets, expected_assets)))
                    assert summary['fixed_asset_equal']
            elif boundary == 'fixed' and baseline['status'] != 'optimal':
                summary.update(status='dependency_nonoptimal', dependency_status=baseline['status'], solve_seconds=0., hours=len(frame))
            else:
                spec = card_spec(p, ctx['city_own'], frame, scenario, 0. if flex == 'none' else 1.)
                if flex == 'none':
                    spec['extra_max_mw'] = 0.
                elif boundary == 'fixed':
                    spec['extra_max_mw'] *= 2
                whole_h, split_h = (0, 0) if flex == 'none' else (12, 48)
                assets = [baseline[name] for name in ASSETS] if boundary == 'fixed' else None
                keep = grid in ctx['keep_grids'] and scenario in ('D-TRN', 'D-SAI', 'A-HPC', 'D-INF') and alpha in (None, 1.)
                solved, hourly = solve(frame, p, grid, alpha=alpha, design='opt', whole_h=whole_h, split_h=split_h,
                                       keep_hourly=keep or ctx['audit_all'], cards=spec, fixed_assets=assets)
                summary.update(solved)
                summary.update(lp_performed=True, card_usd_per_mw_year=spec['usd_per_mw_year'], card_market_usd_per_mw_year=card_spec(p, ctx['city_own'], frame, scenario, 1.)['usd_per_mw_year'],
                               idle_per_mw=spec['idle_per_mw'], baseline_fleet_usd_year=spec['baseline_fleet_usd_year'], servers_per_mw=spec['servers_per_mw'])
                if summary['status'] == 'optimal' and boundary == 'fixed':
                    difference = float(np.max(np.abs(np.asarray([summary[name] for name in ASSETS]) - assets)))
                    summary.update(fixed_asset_max_abs_difference=difference, fixed_asset_equal=bool(difference <= 1e-6))
                    if not summary['fixed_asset_equal']:
                        raise AssertionError('固定资产与无柔性基线不一致')
                if hourly is not None:
                    audit = audit_dispatch(frame, summary, hourly, whole_h, split_h, alpha)
                    summary.update(dispatch_audit_passed=audit['passed'], dispatch_audit_max_violation=audit['maximum_violation'])
                    if not audit['passed']:
                        summary.update(solver_status=summary['status'], status='audit_failed')
                    if keep:
                        label = f'{grid}-{scenario}-{key}'
                        hourly['source_hour_index'] = frame.source_hour_index.to_numpy()
                        hourly.to_parquet(ctx['output'] / 'hourly' / (label + '.parquet'), index=False, compression='zstd')
                        write_json(ctx['output'] / 'audits' / (label + '.json'), audit)
                if summary['status'] == 'optimal' and summary['max_residual'] > 1e-4:
                    summary.update(solver_status=summary['status'], status='audit_failed')
            summary['completed_utc'] = utc_now()
            summary['elapsed_seconds'] = time.monotonic() - started
            summary['worker_peak_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            append_row(path, summary)
            done[key] = summary
            ctx['events'].put((summary['status'], summary['lp_performed'], summary['reused_baseline'], summary['solve_seconds'], summary['worker_peak_rss_mb']))
    return grid, scenario, len(done)


def run(args):
    own, paths = settings(args.config)
    run_root = paths['run_root']
    output = run_root / args.output
    for folder in ('shards', 'hourly', 'audits', 'errors'):
        (output / folder).mkdir(parents=True, exist_ok=True)
    lock = (output / 'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.workers < 1 or args.workers > 96:
        raise ValueError('并行数必须在1至96之间')
    if args.scope == 'all':
        for name in ('city-validation-accepted.json', 'pilot-validation-accepted.json', 'global-inputs-audit.json'):
            acceptance = json.loads((run_root / 'validation' / name).read_text('utf-8'))
            if acceptance['passed'] is not True:
                raise ValueError('对照验证尚未通过：' + name)
            if name == 'global-inputs-audit.json':
                if acceptance['input_index_sha256'] != digest(run_root / 'inputs/grid-input-index-all.csv'):
                    raise ValueError('输入索引与全量验收版本不同')
    project = args.config.parent.parent
    version_files = [args.config, project / 'analysis/flex-econ-copt-20261102.json', project / 'analysis/eight-city-flex-value-20261029.json',
                     project / 'analysis/flex_econ_copt.py', project / 'analysis/eight_city_physics.py',
                     project / 'analysis/global_flex_inputs.py', project / 'analysis/compute_load/model.py',
                     project / 'analysis/compute_load/cooling.py', project / 'analysis/compute_load/assets/cooling-20261028-v2.json',
                     run_root / 'inputs/selected-grids.parquet', run_root / 'inputs/template-index.csv', run_root / f'inputs/grid-input-index-{args.scope}.csv']
    if args.grid_ids:
        version_files.append(args.grid_ids)
    versions = {str(path): digest(path) for path in version_files}
    fingerprint = sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest()
    manifest_path = output / 'run-manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text('utf-8'))
        if previous['scientific_fingerprint'] != fingerprint:
            raise ValueError('续跑输入或科学代码与已保存结果不一致，请先核验版本')
    else:
        write_json(manifest_path, {'utc': utc_now(), 'scientific_fingerprint': fingerprint, 'versions': versions,
                                  'runner_sha256': digest(Path(__file__)), 'config': own, 'scope': args.scope})
    source_version = sha256((fingerprint + digest(Path(__file__))).encode()).hexdigest()[:16]
    archive = output / 'source-versions' / source_version
    for source in [*version_files, Path(__file__)]:
        source = source.absolute()
        relative = source.relative_to(project.absolute()) if source.is_relative_to(project.absolute()) else source.relative_to(run_root)
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if digest(destination) != digest(source):
                raise ValueError('运行源码存档与本次版本不同：' + str(destination))
        else:
            temporary = destination.with_name(destination.name + '.part')
            shutil.copy2(source, temporary)
            temporary.replace(destination)
    write_json(output / 'active-source-version.json', {'source_version': source_version, 'scientific_fingerprint': fingerprint})
    grids = pd.read_parquet(run_root / 'inputs/selected-grids.parquet').grid_id.tolist()
    if args.scope == 'pilot':
        pilot = json.loads((run_root / 'inputs/pilot-grids.json').read_text('utf-8'))
        grids = pilot['city_grids'] + pilot['random_grids']
    if args.grid_ids:
        selected = json.loads(args.grid_ids.read_text('utf-8'))['grid_ids']
        if not selected or len(selected) != len(set(selected)) or not set(selected).issubset(grids):
            raise ValueError('指定重算网格必须唯一且属于当前抽样')
        grids = selected
    if args.limit_groups:
        tasks = [(grid, scenario) for grid in grids for scenario in own['scenarios']][:args.limit_groups]
    else:
        tasks = [(grid, scenario) for grid in grids for scenario in own['scenarios']]
    total_rows = len(tasks) * len(own['alpha']) * 4
    planned_lps = len(tasks) * len(own['alpha']) * 3
    counters = Counter()
    lp_counters = Counter()
    peak_rss, total_solve_seconds, performed, reused = 0., 0., 0, 0
    pending = []
    for grid, scenario in tasks:
        existing = load_rows(output / 'shards' / (grid + '-' + scenario + '.jsonl'))
        assert len({row['case_key'] for row in existing}) == len(existing)
        if len(existing) != len(own['alpha']) * 4:
            pending.append((grid, scenario))
        for row in existing:
            counters[row['status']] += 1
            if row['lp_performed']:
                lp_counters[row['status']] += 1
            performed += int(row['lp_performed'])
            reused += int(row['reused_baseline'])
            total_solve_seconds += row['solve_seconds']
            peak_rss = max(peak_rss, row['worker_peak_rss_mb'])
    initial_done = sum(counters.values())
    started, last_save, last_work_state = time.monotonic(), 0., -300.
    context = mp.get_context('spawn')
    events = context.Queue()
    pilot_selection = json.loads((run_root / 'inputs/pilot-grids.json').read_text('utf-8'))
    keep_grids = list(pilot_selection['city_grids'])
    continents = pd.read_parquet(run_root / 'inputs/selected-grids.parquet', columns=['grid_id', 'continent']).set_index('grid_id').continent.to_dict()
    for continent in ('Africa', 'South America'):
        candidate = next(grid for grid in pilot_selection['random_grids'] if continents[grid] == continent)
        keep_grids.append(candidate)
    write_json(output / 'hourly-grid-selection.json', {'grid_ids': keep_grids, 'method': '八城市代表格，加上固定随机顺序中的非洲、南美洲各一格'})
    # 共享服务器在扫描分片期间也会改变负载，创建进程池前重新按实测峰值留出余量。
    startup_available_gb = psutil.virtual_memory().available / 1024**3
    memory_reserve_gb, worker_memory_budget_gb = 16., .60
    memory_worker_limit = int(max(0., startup_available_gb - memory_reserve_gb) / worker_memory_budget_gb)
    workers = min(args.workers, own['runtime']['maximum_workers'], memory_worker_limit)
    if workers < 1:
        raise MemoryError(f'可用内存{startup_available_gb:.2f}GiB不足以预留16GiB并运行一个求解进程，请待资源释放后续跑')
    print('worker budget', {'requested': args.workers, 'actual': workers, 'available_gb': startup_available_gb,
                           'reserve_gb': memory_reserve_gb, 'per_worker_gb': worker_memory_budget_gb}, flush=True)
    append_row(output / 'execution-attempts.jsonl', {'event': 'start', 'utc': utc_now(), 'pid': os.getpid(),
               'workers': workers, 'requested_workers': args.workers, 'startup_available_memory_gb': startup_available_gb,
               'memory_reserve_gb': memory_reserve_gb, 'worker_memory_budget_gb': worker_memory_budget_gb,
               'completed_before': initial_done, 'total': total_rows, 'source_version': source_version})

    def progress(state):
        nonlocal last_work_state
        elapsed = time.monotonic() - started
        completed = sum(counters.values())
        rate = (completed - initial_done) / elapsed if elapsed > 0 else 0.
        value = {'utc': utc_now(), 'pid': os.getpid(), 'state': state, 'scope': args.scope, 'workers': workers,
                 'requested_workers': args.workers, 'startup_available_memory_gb': startup_available_gb,
                 'memory_reserve_gb': memory_reserve_gb, 'worker_memory_budget_gb': worker_memory_budget_gb,
                 'completed': completed, 'total': total_rows, 'planned_lps': planned_lps, 'performed_lps': performed,
                 'reused_baselines': reused, 'status_counts': dict(counters), 'failed': completed - counters['optimal'],
                 'lp_status_counts': dict(lp_counters), 'failed_lps': performed - lp_counters['optimal'],
                 'actual_lp_optimal_rate': lp_counters['optimal'] / performed if performed else None,
                 'input_unavailable_rows': sum(number for status, number in counters.items() if status.startswith('input_')),
                 'optimal_rate': counters['optimal'] / completed if completed else None, 'session_elapsed_seconds': elapsed,
                 'total_solve_seconds': total_solve_seconds, 'mean_lp_seconds': total_solve_seconds / performed if performed else None,
                 'worker_peak_rss_mb': peak_rss, 'available_memory_gb': psutil.virtual_memory().available / 1024**3,
                 'estimated_remaining_seconds': (total_rows - completed) / rate if rate > 0 else None}
        write_json(output / 'progress.json', value)
        write_json(run_root / 'progress.json', value)
        if args.scope == 'all' and (elapsed - last_work_state >= 300 or state != 'running'):
            with (run_root / 'WORK_STATE.md').open('a', encoding='utf-8') as stream:
                stream.write(f"\n- 自动进度 {value['utc']}：{state}；{workers}并行；完成{completed}/{total_rows}条；实际LP {performed}；LP状态{dict(lp_counters)}；可用内存{value['available_memory_gb']:.1f}GiB；预计剩余{value['estimated_remaining_seconds']}秒。\n")
            last_work_state = elapsed

    progress('running')
    with ProcessPoolExecutor(workers, mp_context=context, initializer=initialize,
                             initargs=(str(args.config), str(output), events, keep_grids, args.audit_all)) as pool:
        iterator = iter(pending)
        futures = {}
        for _ in range(min(len(pending), workers * 2)):
            task = next(iterator)
            futures[pool.submit(run_group, task)] = task
        while futures:
            finished, _ = wait(futures, timeout=2, return_when=FIRST_COMPLETED)
            while True:
                try:
                    status, lp, reuse, seconds, rss = events.get_nowait()
                except Empty:
                    break
                counters[status] += 1
                if lp:
                    lp_counters[status] += 1
                performed += int(lp)
                reused += int(reuse)
                total_solve_seconds += seconds
                peak_rss = max(peak_rss, rss)
            for future in finished:
                task = futures.pop(future)
                if future.exception() is not None:
                    write_json(output / 'errors' / ('-'.join(task) + '.json'), {'task': task, 'utc': utc_now(), 'traceback': ''.join(traceback.format_exception(future.exception()))})
                    progress('controller_error')
                    future.result()
                future.result()
                replacement = next(iterator, None)
                if replacement is not None:
                    futures[pool.submit(run_group, replacement)] = replacement
            if time.monotonic() - last_save >= 20:
                progress('running')
                last_save = time.monotonic()
                print('progress', sum(counters.values()), total_rows, dict(counters), flush=True)
    # 最后一个worker结果返回时，消息队列可能仍有已提交的记录。
    while True:
        try:
            status, lp, reuse, seconds, rss = events.get(timeout=0.2)
        except Empty:
            break
        counters[status] += 1
        if lp:
            lp_counters[status] += 1
        performed += int(lp)
        reused += int(reuse)
        total_solve_seconds += seconds
        peak_rss = max(peak_rss, rss)
    assert sum(counters.values()) == total_rows
    progress('complete')
    append_row(output / 'execution-attempts.jsonl', {'event': 'complete', 'utc': utc_now(), 'pid': os.getpid(),
               'workers': workers, 'completed': sum(counters.values()), 'session_elapsed_seconds': time.monotonic() - started})
    events.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('global-flex-econ-20261103.json'))
    parser.add_argument('--scope', choices=['pilot', 'all'], default='pilot')
    parser.add_argument('--output', default='pilot')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit-groups', type=int)
    parser.add_argument('--grid-ids', type=Path)
    parser.add_argument('--audit-all', action='store_true')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
