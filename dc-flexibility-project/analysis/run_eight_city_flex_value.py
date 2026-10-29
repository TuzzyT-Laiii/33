import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import time

for thread_variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[thread_variable] = '1'

import numpy as np
import pandas as pd
import psutil

from eight_city_onsite_solver import MODEL_VERSION, sha256, solve_onsite


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT / 'analysis/eight-city-flex-value-20261029.json'


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def load_input(directory, city, scenario, config, project=PROJECT):
    meta = config['scenarios'][scenario]
    path = directory / f'{city}-{scenario}-10MW.parquet'
    if not path.exists():
        path = directory / f"{city}-{meta['configuration_id']}-{meta['demand']}.parquet"
    supply_provenance = None
    if not path.exists():
        thermal = directory / f'{city}-{scenario}-10MW-thermal-only.parquet'
        supply = directory / f'{city}-supply.parquet'
        if not thermal.exists() or not supply.exists():
            return None
        path = thermal
        supply_before = sha256(supply)
        supply_frame = pd.read_parquet(supply)
        if sha256(supply) != supply_before:
            raise RuntimeError('供电输入在读取期间变化')
        supply_provenance = {'path': str(supply), 'sha256': supply_before}
    if not path.exists():
        return None
    before = sha256(path)
    frame = pd.read_parquet(path).reset_index(drop=True)
    if before != sha256(path):
        raise RuntimeError(f'输入在读取期间发生变化：{path}')
    frame['time_utc'] = pd.to_datetime(frame.time_utc, utc=True)
    if supply_provenance is not None:
        supply_frame['time_utc'] = pd.to_datetime(supply_frame.time_utc, utc=True)
        frame = frame.merge(supply_frame[['time_utc', 'pv_capacity_factor', 'wind_capacity_factor']],
                            on='time_utc', how='left', validate='one_to_one')
    year = config['sweep']['weather_year']
    expected = pd.date_range(f'{year}-01-01', f'{year+1}-01-01', inclusive='left', freq='h', tz='UTC')
    selected = frame.loc[frame.time_utc.dt.year.eq(year)].copy()
    if not pd.DatetimeIndex(selected.time_utc).equals(expected):
        raise ValueError(f'天气年不连续或不完整：{path}')
    if config['sweep']['leap_policy'] != 'drop_feb29':
        raise ValueError('尚未实现指定的闰年处理方式')
    frame = selected.loc[~(selected.time_utc.dt.month.eq(2) & selected.time_utc.dt.day.eq(29))].reset_index(drop=True)
    if len(frame) != 8760:
        raise ValueError('主算输入必须为8760小时')
    if not {'pv_capacity_factor', 'wind_capacity_factor'}.issubset(frame.columns):
        return None
    parameters = pd.read_csv(project / 'results/typical-load-curves-20261029-v2/power-parameters.csv')
    method = 'cpu_spec' if meta['configuration_id'] == 'C20' else 'ratio_p50'
    selected = parameters.loc[parameters.configuration_id.eq(meta['configuration_id']) & parameters.method.eq(method)]
    if len(selected) != 1:
        raise ValueError('硬件功率参数不能唯一定位')
    row = selected.iloc[0]
    installed_reference_mw = float(frame.installed_reference_it_mw.iloc[0])
    idle_per_unit = row.base_w + row.cpu_idle_w_each * row.cpu_packages_per_unit + row.unallocated_gpu_idle_w_each * row.gpu_devices_per_unit
    idle_mw = float(installed_reference_mw / row.sizing_reference_w * idle_per_unit)
    gpu_full = 0. if meta['configuration_id'] == 'C20' else row.gpu_tdp_w_each * row.allocated_gpu_power_ratio
    calibrated_full_per_unit = row.base_w + row.cpu_full_w_each * row.cpu_packages_per_unit + gpu_full * row.gpu_devices_per_unit
    design_mw = float(installed_reference_mw / row.sizing_reference_w * calibrated_full_per_unit)
    numerical_bytes = frame[['it_mw', 'pue', 'pv_capacity_factor', 'wind_capacity_factor']].to_numpy(dtype='<f8').tobytes()
    numerical_bytes += frame.time_utc.astype('int64').to_numpy(dtype='<i8').tobytes()
    numerical_bytes += np.array([idle_mw, design_mw], dtype='<f8').tobytes()
    provenance = {'path': str(path.resolve()), 'sha256': sha256_bytes(numerical_bytes),
                  'file_sha256': before, 'source_hours': len(expected),
                  'supply': supply_provenance,
                  'reported_hours': len(frame), 'weather_year': year, 'leap_policy': 'drop_feb29',
                  'hardware_parameter_path': 'results/typical-load-curves-20261029-v2/power-parameters.csv',
                  'hardware_parameter_sha256': sha256(project / 'results/typical-load-curves-20261029-v2/power-parameters.csv'),
                  'installed_reference_it_mw': installed_reference_mw,
                  'calibrated_full_it_mw': design_mw,
                  'idle_it_mw': idle_mw,
                  'power_proxy': '动态IT电量作为可分割工作量；硬件空闲功率不移动；r新增空闲功率完整计入',
                  'cooling_proxy': '执行时刻PUE固定为第一阶段该小时PUE；不重新估计负载率对PUE的反馈'}
    return frame, idle_mw, design_mw, provenance


def combinations(config, tier):
    sweep = config['sweep']
    axes = ['phi', 'rmax', 'carbon_price', 'alpha']
    main = sweep['main']
    priority = list(itertools.product(*(main[key] for key in axes)))
    remainder = [item for item in itertools.product(*(sweep[key] for key in axes)) if item not in priority]
    result = priority if tier == 'main' else remainder if tier == 'remaining' else priority + remainder
    return [dict(zip(axes, item)) for item in result]


def case_token(case):
    return sha256_bytes(json.dumps(case, sort_keys=True, ensure_ascii=True).encode())[:20]


def sha256_bytes(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def run_group(task):
    config, directory, output, city, scenario, settings, fingerprint = task
    loaded = load_input(Path(directory), city, scenario, config)
    if loaded is None:
        return {'city': city, 'scenario': scenario, 'status': 'input_pending'}
    frame, idle, design, provenance = loaded
    group = {'city': city, 'scenario': scenario, **settings, 'config_fingerprint': fingerprint,
             'config_sha256': config['_source_config_sha256'],
             'input_sha256': provenance['sha256'], 'model_version': MODEL_VERSION}
    common = {**settings, 'idle_mw': idle, 'design_mw': design,
              'emission_factor': config['cities'][city]['emission_factor']}
    if common['emission_factor'] is None:
        raise ValueError('国家排放因子缺失')
    counter = {'optimal': 0, 'infeasible': 0, 'not_solved': 0, 'cached': 0}
    for storage in config['sweep']['S']:
        for deadline in config['sweep']['D']:
            case = {**group, 'storage': storage, 'D': deadline}
            folder = Path(output) / 'cases' / city / scenario / case_token(case)
            target = folder / 'summary.json'
            if target.exists():
                summary = json.loads(target.read_text(encoding='utf-8'))
                if summary['status'] in ('optimal', 'infeasible'):
                    counter['cached'] += 1
                    continue
            solved = solve_onsite(frame, config, deadline=deadline, fraction=common['phi'],
                rmax=common['rmax'], storage=storage, alpha=common['alpha'],
                carbon_price=common['carbon_price'], emission_factor=common['emission_factor'], idle_mw=idle, design_mw=design)
            summary = {**solved['summary'], **case, 'input_provenance': provenance,
                       'finished_utc': datetime.now(timezone.utc).isoformat()}
            if not solved['hourly'].empty:
                folder.mkdir(parents=True, exist_ok=True)
                temporary = folder / 'hourly.partial.parquet'
                solved['hourly'].to_parquet(temporary, index=False, compression='zstd')
                temporary.replace(folder / 'hourly.parquet')
            write_json(target, summary)
            counter[summary['status']] += 1
            print(json.dumps({'case': str(target), 'status': summary['status'], 'D': deadline,
                              'S': storage, 'seconds': summary['solve_seconds']}), flush=True)
    return {'city': city, 'scenario': scenario, 'status': 'completed_with_not_solved' if counter['not_solved'] else 'completed', **counter}


def run_physical_group(task):
    config, directory, output, city, scenario, settings, fingerprint = task
    loaded = load_input(Path(directory), city, scenario, config)
    if loaded is None:
        return {'city': city, 'scenario': scenario, 'status': 'input_pending'}
    frame, idle, design, provenance = loaded
    emission = config['cities'][city]['emission_factor']
    reference = config['sweep']['physical_reference']
    reference_carbon_price = config['sweep']['main']['carbon_price'][0]
    if reference['mode'] != 'given' or min(reference['pv_mw'], reference['wind_mw']) < 0:
        raise ValueError('物理参照必须是配置中明确给定的有效现场装机')
    for storage in config['sweep']['S']:
        base = solve_onsite(frame, config, storage=storage, alpha=None, carbon_price=reference_carbon_price,
                           emission_factor=emission, idle_mw=idle, design_mw=design)
        if base['summary']['status'] != 'optimal':
            continue
        assets = {key: base['summary'][key] for key in ('pv_mw','wind_mw','battery_power_mw','battery_energy_mwh')}
        assets.update({key: reference[key] for key in ('pv_mw', 'wind_mw')})
        for deadline in config['sweep']['D']:
            case = {'city': city, 'scenario': scenario, 'D': deadline, 'storage': storage,
                    'phi': settings['phi'], 'rmax': settings['rmax'], 'physical': True,
                    'config_fingerprint': fingerprint, 'input_sha256': provenance['sha256'], 'model_version': MODEL_VERSION}
            case['config_sha256'] = config['_source_config_sha256']
            folder = Path(output) / 'physical' / city / scenario / case_token(case)
            if (folder / 'summary.json').exists():
                cached = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
                if cached['status'] in ('optimal', 'infeasible'):
                    continue
            result = solve_onsite(frame, config, deadline=deadline, fraction=settings['phi'], rmax=settings['rmax'],
                                  storage=storage, fixed_assets=assets, physical=True,
                                  carbon_price=reference_carbon_price, emission_factor=emission, idle_mw=idle, design_mw=design)
            folder.mkdir(parents=True, exist_ok=True)
            if not result['hourly'].empty:
                result['hourly'].to_parquet(folder / 'hourly.parquet', index=False, compression='zstd')
            write_json(folder / 'summary.json', {**result['summary'], **case, 'fixed_assets': assets,
                'input_provenance': provenance, 'physical_reference': reference})
    return {'city': city, 'scenario': scenario, 'status': 'physical_completed'}


def main():
    parser = argparse.ArgumentParser(description='八城市onsite柔性收益；默认断点续跑，配置改变自动使用新算例键')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'data/processed/eight-city-flex-value-20261029')
    parser.add_argument('--city', nargs='+')
    parser.add_argument('--scenario', nargs='+')
    parser.add_argument('--tier', choices=['main', 'remaining', 'all'], default='all')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--minimum-available-gb', type=float, default=12)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--physical', action='store_true')
    parser.add_argument('--wait-inputs', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    config['_source_config_sha256'] = sha256(args.config)
    gate = json.loads(args.gate.read_text(encoding='utf-8'))
    if gate.get('status') != 'passed' or gate.get('annual_real_input_gate') is not True:
        raise ValueError('全年真实输入核验尚未通过，禁止放开批量')
    if gate['config_sha256'] != sha256(args.config) or gate['solver_sha256'] != sha256(Path(__file__).with_name('eight_city_onsite_solver.py')):
        raise ValueError('核验版本与当前配置或模型不一致，需重新核验')
    fingerprint = sha256_bytes((sha256(args.config) + sha256(Path(__file__).with_name('eight_city_onsite_solver.py')) + sha256(Path(__file__))).encode())
    cities, scenarios = args.city or list(config['cities']), args.scenario or list(config['scenarios'])
    if not set(cities).issubset(config['cities']) or not set(scenarios).issubset(config['scenarios']):
        raise ValueError('城市或情境名称未登记')
    if not 1 <= args.workers <= 16:
        raise ValueError('并行进程数必须位于1至16')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    queue_label = ('physical' if args.physical else 'economic') + '-' + args.tier
    write_json(args.output_dir / f'run-manifest-{queue_label}.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'pid': os.getpid(), 'config_fingerprint': fingerprint, 'config_sha256': sha256(args.config),
        'solver_sha256': sha256(Path(__file__).with_name('eight_city_onsite_solver.py')),
        'workers': args.workers, 'cities': cities, 'scenarios': scenarios, 'tier': args.tier,
        'physical': args.physical, 'gate_sha256': sha256(args.gate), 'input_dir': str(args.input_dir)})
    settings_list = combinations(config, args.tier)
    if args.physical:
        settings_list = list({(item['phi'], item['rmax']): dict(phi=item['phi'], rmax=item['rmax']) for item in settings_list}.values())
    completed = set()
    unresolved_groups = []
    while True:
        pending_inputs = []
        for settings in settings_list:
            tasks = []
            for city, scenario in itertools.product(cities, scenarios):
                key = (city, scenario, json.dumps(settings, sort_keys=True))
                if key in completed:
                    continue
                if load_input(args.input_dir, city, scenario, config) is None:
                    pending_inputs.append([city, scenario])
                    continue
                tasks.append((config, str(args.input_dir), str(args.output_dir), city, scenario, settings, fingerprint))
            if tasks:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    futures = {}
                    for task in tasks:
                        while psutil.virtual_memory().available < args.minimum_available_gb * 1024**3:
                            print('等待服务器可用内存', flush=True)
                            time.sleep(30)
                        function = run_physical_group if args.physical else run_group
                        future = pool.submit(function, task)
                        futures[future] = (task[3], task[4], json.dumps(settings, sort_keys=True))
                    for future in as_completed(futures):
                        result = future.result()
                        if result['status'] == 'completed_with_not_solved':
                            unresolved_groups.append(result)
                        completed.add(futures[future])
                        print(json.dumps(result, ensure_ascii=False), flush=True)
                        write_json(args.output_dir / f'progress-{queue_label}.json', {'utc': datetime.now(timezone.utc).isoformat(),
                            'pid': os.getpid(), 'completed_groups': len(completed),
                            'latest_group': result, 'pending_input_pairs': sorted(set(map(tuple, pending_inputs)))})
        pending_inputs = sorted(set(map(tuple, pending_inputs)))
        write_json(args.output_dir / f'queue-state-{queue_label}.json', {'utc': datetime.now(timezone.utc).isoformat(),
                   'completed_groups': len(completed), 'pending_inputs': pending_inputs,
                   'unresolved_groups': unresolved_groups,
                   'status': 'waiting_inputs' if pending_inputs else ('completed_with_not_solved' if unresolved_groups else 'completed')})
        if not pending_inputs or not args.wait_inputs:
            break
        time.sleep(30)
    if unresolved_groups:
        raise RuntimeError('存在未完成求解的算例，请检查queue-state记录后续跑')


if __name__ == '__main__':
    main()
