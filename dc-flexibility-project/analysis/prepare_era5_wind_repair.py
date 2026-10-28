import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import threading
import time

from ecmwf.datastores import Client
import numpy as np
import pandas as pd
import requests
import xarray as xr

from acquire_buildability_protection import sha, utc_now, write_json
from acquire_eligible_era5_weather import open_source
from era5_cds_request import CDSRequestGate, PersistentCDSRequest, saved_job_status
from era5_source_validation import overlap_check_passed, overlap_statistics


REPAIR_GROUPS = {
    'wind': (('u100', 'v100'), ('100m_u_component_of_wind', '100m_v_component_of_wind'), 'm s**-1'),
    'dewpoint': (('d2m',), ('2m_dewpoint_temperature',), 'K'),
}


def cds_read_session():
    # 网络等待统一交给持久请求管理，确保每次失败都记录并计入24小时预算。
    return requests.Session()


def prepare(root, label, group='wind', monthly=False, missing_only=False):
    if monthly and group != 'dewpoint':
        raise ValueError('按月加速仅适用于本次露点补取')
    if missing_only and (not monthly or group != 'dewpoint'):
        raise ValueError('限定缺失月份要求按月露点补取')
    variables, cds_variables, expected_units = REPAIR_GROUPS[group]
    selection_path = root / 'data/metadata/buildability-20261025/selection-manifest.json'
    selection = json.loads(selection_path.read_text())
    if selection['status'] != 'complete' or not selection['protected_areas_applied'] or not selection['slope_applied']:
        raise ValueError('定向获取要求完整的保护地与坡度筛选清单')
    points_path = Path(selection['outputs']['points']['path'])
    if sha(points_path) != selection['outputs']['points']['sha256']:
        raise ValueError('定向获取的合格节点清单SHA不一致')
    points = pd.read_parquet(points_path).sort_values('native_flat_index')
    latitude = np.rint((points.latitude + 90) * 4).astype(int) // 4
    longitude = np.rint(((points.longitude_0_360 + 180) % 360) * 4).astype(int) // 4
    labels = pd.Series([f'lat-{lat:03d}-lon-{lon:03d}' for lat, lon in zip(latitude, longitude)], index=points.index)
    points = points.loc[labels.eq(label)]
    if points.empty:
        raise ValueError('定向获取的空间块没有对应合格天气节点')
    proof_path = root / 'data/metadata/buildability-20261025/weather-cds-repairs' / (label + '.json')
    if proof_path.exists():
        proof = json.loads(proof_path.read_text())
        if proof['status'] == 'verified':
            if (proof['selection_manifest_sha256'] != sha(selection_path)
                    or tuple(proof.get('variables', ('u100', 'v100'))) != variables
                    or proof['native_flat_indices'] != points.native_flat_index.astype(int).tolist()
                    or proof['source_metadata_sha256'] != sha(root / 'data/raw/era5-eligible-ecmwf-geo/.zmetadata')
                    or sha(Path(proof['output_path'])) != proof['output_sha256']):
                raise ValueError('已核验补取数据与当前研究输入不同')
            for source_record in proof['raw_sources']:
                if sha(Path(source_record['path'])) != source_record['sha256']:
                    raise ValueError('已核验补取原文SHA不一致')
            print(json.dumps({'status': 'reused_verified', 'block': label}), flush=True)
            return
    source, session, times, metadata_sha = open_source(root)
    original = {}
    with source:
        for name in variables:
            original[name] = source[name].sel(time=times).sel(
                latitude=xr.DataArray(points.latitude.to_numpy(), dims='weather_point'),
                longitude=xr.DataArray(((points.longitude_0_360.to_numpy() + 180) % 360 - 180), dims='weather_point')).values
    session.close()
    if missing_only:
        missing = np.isnan(original['d2m'])
        expected_missing = np.repeat(((times >= np.datetime64('2023-01-01')) &
                                      (times < np.datetime64('2024-07-01')))[:, None], len(points), axis=1)
        july = (times >= np.datetime64('2024-07-01')) & (times < np.datetime64('2024-08-01'))
        if not np.array_equal(missing, expected_missing) or not np.isfinite(original['d2m'][july]).all():
            raise ValueError('缺失月份或2024-07有效重合月不符合本次授权')
    if not any(np.isnan(value).any() for value in original.values()):
        raise ValueError('该空间块没有指定变量需要补取的缺失值')
    if any(np.isinf(value).any() for value in original.values()):
        raise ValueError('指定变量存在无穷值，需要单独核查')
    area = [float(points.latitude.max()), float(((points.longitude_0_360 + 180) % 360 - 180).min()),
            float(points.latitude.min()), float(((points.longitude_0_360 + 180) % 360 - 180).max())]
    directory = root / 'data/raw/era5-eligible-cds-repairs' / label
    directory.mkdir(parents=True, exist_ok=True)
    key = json.loads((root / '.codex-temp/buildability-20261025/cds-credential.json').read_text())['key']
    dataset = 'reanalysis-era5-single-levels'
    scope = group if group == 'dewpoint' else f'{group}-{label}'
    state_path = root / 'status' / f'era5-{scope}-repair.json'
    state = {'status': f'fetching_official_{group}', 'block': label, 'variables': list(variables),
             'strategy': 'missing_months_and_july_overlap' if missing_only else ('monthly_sequential_with_existing_annual_reuse' if monthly else 'annual'),
             'started_utc': utc_now(), 'jobs': {}}
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous['block'] != label or previous['variables'] != list(variables):
            raise ValueError('已有补取状态与当前空间块或变量不同')
        state['jobs'] = previous['jobs']
    state_lock = threading.Lock()
    stopped = threading.Event()
    known_path = root / '.codex-temp/buildability-20261025' / f'cds-{group}-repair/known-jobs.json'
    known = json.loads(known_path.read_text())['jobs'] if known_path.exists() else []

    def repair_ready():
        if not proof_path.exists():
            return False
        proof = json.loads(proof_path.read_text())
        return proof['status'] == 'verified'

    def update(year, status, request_id=None, **details):
        with state_lock:
            job = state['jobs'].setdefault(str(year), {})
            job.update(status=status, request_id=request_id, **details)
            state['updated_utc'] = utc_now()
            write_json(state_path, state)

    def request_for(year, month=None):
        request = {'product_type': ['reanalysis'],
                   'variable': list(cds_variables),
                   'year': [str(year)], 'month': [f'{month:02d}' for month in range(1, 13)] if year < 2026 else ['01'],
                   'day': [f'{day:02d}' for day in range(1, 32)] if year < 2026 else ['01'],
                   'time': [f'{hour:02d}:00' for hour in range(24)] if year < 2026 else ['00:00'],
                   'area': area, 'data_format': 'netcdf', 'download_format': 'unarchived'}
        if month is not None:
            request['month'] = [f'{month:02d}']
            request['day'] = [f'{day:02d}' for day in range(1, calendar.monthrange(year, month)[1] + 1)]
        return request

    annual_received = {}
    annual_locks = {year: threading.Lock() for year in (2024, 2025)}
    annual_checked = {year: 0.0 for year in (2024, 2025)}

    def completed_annual(year):
        # 保留既有年度队列；年度先返回时复用原文，后续月份不再提交。
        with annual_locks[year]:
            if year in annual_received:
                return True
            if time.monotonic() - annual_checked[year] < 60:
                return False
            annual_checked[year] = time.monotonic()
            target = directory / f'{group}-{year}.nc'
            saved_path = target.with_suffix('.request.json')
            if target.exists():
                annual_received[year] = fetch(year)
                return True
            if not saved_path.exists():
                return False
            saved = json.loads(saved_path.read_text())
            if saved['dataset'] != dataset or saved['request'] != request_for(year):
                raise ValueError('保留的年度请求与本次补取范围不一致')
            client = Client(url='https://cds.climate.copernicus.eu/api', key=key, cleanup=False,
                            progress=False, timeout=60, maximum_tries=1, session=cds_read_session())
            job = PersistentCDSRequest(client, dataset, request_for(year), saved_path,
                                       lambda status, request_id, **details: update(year, status, request_id, **details),
                                       stopped, gate=CDSRequestGate(root, client))
            status = job.poll()
            if status == 'successful':
                annual_received[year] = fetch(year)
                return True
            if status not in ('accepted', 'running'):
                raise RuntimeError(f'保留的CDS年度请求失败：{remote.request_id}/{status}')
            return False

    def fetch(year, month=None):
        if missing_only and repair_ready():
            return None
        if stopped.is_set():
            raise RuntimeError('其他补取请求失败，停止继续提交')
        period = str(year) if month is None else f'{year}-{month:02d}'
        if month is not None and not missing_only and completed_annual(year):
            update(period, 'covered_by_annual')
            return None
        request = request_for(year, month)
        target = directory / f'{group}-{year}.nc' if month is None else directory / f'{group}-monthly/{period}.nc'
        target.parent.mkdir(parents=True, exist_ok=True)
        receipt_path = target.with_suffix('.json')
        if target.exists():
            record = json.loads(receipt_path.read_text())
            if record['request'] != request or record['sha256'] != sha(target):
                raise ValueError('已有CDS原文与定向请求不一致')
        else:
            update(period, 'connecting')
            client = Client(url='https://cds.climate.copernicus.eu/api', key=key, cleanup=False,
                            progress=False, timeout=60, maximum_tries=1, sleep_max=15,
                            session=cds_read_session())
            request_path = target.with_suffix('.request.json')
            known_id = None
            if not request_path.exists():
                matched = [job for job in known if job['request'] == request and job['status'] in ('accepted', 'running', 'successful')]
                if len(matched) > 1:
                    raise ValueError('同一CDS补取请求存在多个可用编号，需要核查')
                known_id = matched[0]['request_id'] if matched else None
            job = PersistentCDSRequest(client, dataset, request, request_path,
                                       lambda status, request_id, **details: update(period, status, request_id, **details),
                                       stopped, known_id, gate=CDSRequestGate(root, client))
            while True:
                if missing_only and repair_ready():
                    return None
                if stopped.is_set():
                    raise RuntimeError('其他年度请求失败；现有CDS请求编号已保留')
                if month is not None and not missing_only and completed_annual(year):
                    update(period, 'covered_by_annual', job.remote.request_id)
                    return None
                status = job.poll()
                if status == 'successful':
                    break
                time.sleep(15)
            remote = job.remote
            temporary = target.with_suffix('.nc.part')
            job.download(temporary)
            record = {'dataset': dataset, 'request': request, 'downloaded_utc': utc_now(),
                      'request_id': remote.request_id,
                      'sha256': sha(temporary), 'bytes': temporary.stat().st_size, 'path': str(target)}
            if month is not None:
                record['source_layout'] = 'monthly'
            write_json(receipt_path, record)
            temporary.replace(target)
        update(period, 'downloaded', record.get('request_id'))
        print(json.dumps({'event': 'official_source_received', 'block': label, 'year': year,
                          'month': month, 'bytes': record['bytes'], 'utc': utc_now()}), flush=True)
        return record

    received = {}
    if monthly:
        periods = ([(2023, None)] + [(2024, month) for month in range(1, 8)]) if missing_only else (
            [(2023, None), (2024, 1), (2025, 1), (2026, None)] + [
                (year, month) for month in range(2, 13) for year in (2024, 2025)])
        for year, month in periods:
            received[(year, month)] = fetch(year, month)
    else:
        if group == 'wind':
            client = Client(url='https://cds.climate.copernicus.eu/api', key=key, cleanup=False,
                            progress=False, timeout=60, maximum_tries=1, session=cds_read_session())
            ready_years = []
            dismissed_years = []
            remaining_years = []
            for year in range(2023, 2027):
                target = directory / f'{group}-{year}.nc'
                saved_path = target.with_suffix('.request.json')
                saved = json.loads(saved_path.read_text()) if saved_path.exists() else None
                if saved and (saved['dataset'] != dataset or saved['request'] != request_for(year)):
                    raise ValueError('已有风数据请求与授权年份或区域不同')
                status = None
                if saved and saved.get('request_id') and not target.exists():
                    job = PersistentCDSRequest(client, dataset, request_for(year), saved_path,
                                               lambda status, request_id, year=year, **details: update(
                                                   year, status, request_id, **details),
                                               stopped, gate=CDSRequestGate(root, client))
                    status = job.network_call('priority_status', lambda: saved_job_status(client, saved['request_id']))
                if target.exists() or status in ('accepted', 'running', 'successful'):
                    ready_years.append(year)
                elif status == 'dismissed':
                    dismissed_years.append(year)
                else:
                    remaining_years.append(year)
            for year in ready_years + dismissed_years + remaining_years:
                received[(year, None)] = fetch(year)
        else:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {executor.submit(fetch, year): (year, None) for year in range(2023, 2027)}
                for future in as_completed(futures):
                    error = future.exception()
                    if error is not None:
                        stopped.set()
                        for pending in futures:
                            pending.cancel()
                        with state_lock:
                            state.update(status='failed', failed_year=futures[future], error_type=type(error).__name__, updated_utc=utc_now())
                            write_json(state_path, state)
                        raise error
                    received[futures[future]] = future.result()
    if missing_only and repair_ready():
        print(json.dumps({'status': 'verified_alternative_ready', 'block': label}), flush=True)
        return
    records = [record for (year, month), record in received.items()
               if record is not None and (month is None or year not in annual_received)]
    records.extend(annual_received.values())
    records.sort(key=lambda item: (item['request']['year'][0], item['request']['month'][0]))
    datasets = [xr.open_dataset(record['path']) for record in records]
    combined = xr.concat(datasets, dim='valid_time').sortby('valid_time')
    if combined.indexes['valid_time'].has_duplicates or (not missing_only and combined.indexes['valid_time'].get_indexer(times).min() < 0):
        raise ValueError('CDS定向原文存在重复或缺少研究小时')
    table = pd.DataFrame({'time': np.repeat(times, len(points)),
                          'native_flat_index': np.tile(points.native_flat_index.to_numpy(dtype=np.int32), len(times))})
    checks = {}
    for name in variables:
        if combined[name].attrs.get('units') != expected_units:
            raise ValueError('CDS指定变量单位与研究要求不同')
        covered_times = combined.valid_time.values if missing_only else times
        official = combined[name].sel(valid_time=covered_times).sel(
            latitude=xr.DataArray(points.latitude.to_numpy(), dims='weather_point'),
            longitude=xr.DataArray(((points.longitude_0_360.to_numpy() + 180) % 360 - 180), dims='weather_point')).values
        if (official.dtype != np.dtype('float32') or not np.isfinite(official).all()
                or official.shape != (len(covered_times), len(points))):
            raise ValueError('官方补取原文的类型、坐标或完整性不符合要求')
        if missing_only:
            positions = pd.Index(times).get_indexer(covered_times)
            if (positions.min() < 0 or official.dtype != np.dtype('float32') or not np.isfinite(official).all()
                    or official.shape != (len(covered_times), len(points))):
                raise ValueError('限定月份补取的时间、类型、坐标或数值无效')
            covered_original = original[name][positions]
            overlap = np.isfinite(covered_original)
            comparison = overlap_statistics(official, covered_original)
            if int(overlap.sum()) != int(july.sum()) * len(points) or not overlap_check_passed(comparison):
                raise ValueError(f'2024-07有效重合值超过绝对容差：{comparison}')
            values = original[name].copy()
            patch = np.isnan(covered_original)
            selected = values[positions].copy()
            selected[patch] = official[patch]
            values[positions] = selected
        else:
            comparison = overlap_statistics(official, original[name])
            if not overlap_check_passed(comparison):
                raise ValueError(f'{name}有效重合值超过绝对容差：{comparison}')
            values = original[name].copy()
            patch = np.isnan(values)
            values[patch] = official[patch]
        if values.shape != original[name].shape or values.dtype != np.dtype('float32') or not np.isfinite(values).all():
            raise ValueError('CDS定向原文的形状、类型或完整性不符合要求')
        valid = np.isfinite(original[name])
        if not np.array_equal(values[valid], original[name][valid]):
            raise ValueError('补取数值表改变了冻结原文中的有效值')
        checks[name] = {**comparison, 'original_nan_count': int(np.isnan(original[name]).sum()),
                        'overlap_units': expected_units, 'unchanged_original_finite_count': int(valid.sum()),
                        'overlap_source': 'official_vs_frozen_arco'}
        if missing_only:
            checks[name].update(verified_overlap_count=int(overlap.sum()), overlap_period='2024-07',
                                unchanged_original_finite_count=int(valid.sum()), overlap_source='official_month_vs_frozen_arco')
        table[name] = values.reshape(-1)
    output = root / 'data/interim/era5-eligible-cds-repairs' / (label + '.parquet')
    if missing_only:
        from era5_dewpoint_candidate import publish_candidate

        report = {'block': label, 'variables': list(variables), 'source_file_prefix': group,
                  'selection_manifest_sha256': sha(selection_path), 'source_metadata_sha256': metadata_sha,
                  'hours': 26305, 'native_flat_indices': points.native_flat_index.astype(int).tolist(),
                  'official_dataset': dataset, 'raw_sources': records, 'checks': checks, 'missing_only': True,
                  'created_utc': utc_now(), 'acceptance': '2024-07有效重合值绝对差≤1e-4 K；仅使用官方同时间同节点数值填补NaN，保留全部原有效值'}
        report = publish_candidate(root, table, report)
        for dataset_object in datasets:
            dataset_object.close()
        state.update(status='verified', updated_utc=utc_now())
        write_json(state_path, state)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.parquet.part')
    table.to_parquet(temporary, index=False, compression='zstd')
    pd.testing.assert_frame_equal(table, pd.read_parquet(temporary))
    temporary.replace(output)
    for dataset_object in datasets:
        dataset_object.close()
    report = {'status': 'awaiting_overlap_validation', 'created_utc': utc_now(), 'block': label,
              'variables': list(variables), 'source_file_prefix': group,
              'selection_manifest_sha256': sha(selection_path), 'source_metadata_sha256': metadata_sha,
              'hours': 26305, 'native_flat_indices': points.native_flat_index.astype(int).tolist(),
              'official_dataset': dataset, 'raw_sources': records, 'checks': checks,
              'missing_only': missing_only,
              'output_path': str(output), 'output_sha256': sha(output), 'output_bytes': output.stat().st_size}
    write_json(root / 'data/metadata/buildability-20261025/weather-cds-repairs' / (label + '.json'), report)
    state.update(status='awaiting_overlap_validation', updated_utc=utc_now())
    write_json(state_path, state)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--block', required=True)
    parser.add_argument('--group', choices=REPAIR_GROUPS, default='wind')
    parser.add_argument('--monthly', action='store_true')
    parser.add_argument('--missing-only', action='store_true')
    args = parser.parse_args()
    scope = args.group if args.group == 'dewpoint' else f'{args.group}-{args.block}'
    with (args.root.resolve() / f'.era5-{scope}-repair.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(args.root.resolve(), args.block, args.group, args.monthly, args.missing_only)
