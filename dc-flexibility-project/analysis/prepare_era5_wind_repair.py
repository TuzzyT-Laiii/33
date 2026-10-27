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
import xarray as xr

from acquire_buildability_protection import sha, utc_now, write_json
from acquire_eligible_era5_weather import open_source


REPAIR_GROUPS = {
    'wind': (('u100', 'v100'), ('100m_u_component_of_wind', '100m_v_component_of_wind'), 'm s**-1'),
    'dewpoint': (('d2m',), ('2m_dewpoint_temperature',), 'K'),
}


def prepare(root, label, group='wind', monthly=False):
    if monthly and group != 'dewpoint':
        raise ValueError('按月加速仅适用于本次露点补取')
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
    state_path = root / 'status' / f'era5-{group}-repair.json'
    state = {'status': f'fetching_official_{group}', 'block': label, 'variables': list(variables),
             'strategy': 'monthly_sequential_with_existing_annual_reuse' if monthly else 'annual',
             'started_utc': utc_now(), 'jobs': {}}
    state_lock = threading.Lock()
    stopped = threading.Event()
    known_path = root / '.codex-temp/buildability-20261025' / f'cds-{group}-repair/known-jobs.json'
    known = json.loads(known_path.read_text())['jobs'] if known_path.exists() else []

    def update(year, status, request_id=None):
        with state_lock:
            state['jobs'][str(year)] = {'status': status, 'request_id': request_id}
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
                            progress=False, timeout=60, maximum_tries=3, retry_after=2)
            remote = client.get_remote(saved['request_id'])
            status = remote.status
            update(year, status, remote.request_id)
            if status == 'successful':
                annual_received[year] = fetch(year)
                return True
            if status not in ('accepted', 'running'):
                raise RuntimeError(f'保留的CDS年度请求失败：{remote.request_id}/{status}')
            return False

    def fetch(year, month=None):
        if stopped.is_set():
            raise RuntimeError('其他补取请求失败，停止继续提交')
        period = str(year) if month is None else f'{year}-{month:02d}'
        if month is not None and completed_annual(year):
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
                            progress=False, timeout=60, maximum_tries=3, retry_after=2, sleep_max=15)
            request_path = target.with_suffix('.request.json')
            if request_path.exists():
                saved = json.loads(request_path.read_text())
                if saved['request'] != request or saved['dataset'] != dataset:
                    raise ValueError('CDS持久请求与当前合格区域请求不同')
                remote = client.get_remote(saved['request_id'])
            else:
                matched = [job for job in known if job['request'] == request and job['status'] in ('accepted', 'running', 'successful')]
                if len(matched) > 1:
                    raise ValueError('同一CDS补取请求存在多个可用编号，需要核查')
                remote = client.get_remote(matched[0]['request_id']) if matched else client.submit(dataset, request)
                write_json(request_path, {'dataset': dataset, 'request': request, 'request_id': remote.request_id, 'created_utc': utc_now()})
            if remote.collection_id != dataset or remote.request != request:
                raise ValueError('CDS服务端请求内容与本研究区域不同')
            quota_rejections = 0
            while True:
                if stopped.is_set():
                    raise RuntimeError('其他年度请求失败；现有CDS请求编号已保留')
                if month is not None and completed_annual(year):
                    update(period, 'covered_by_annual', remote.request_id)
                    return None
                status = remote.status
                update(period, status, remote.request_id)
                if status == 'successful':
                    break
                if status == 'rejected' and monthly:
                    response = remote.session.get(remote.url + '/results', headers=remote.headers, timeout=60)
                    if response.status_code != 400:
                        raise RuntimeError(f'CDS拒绝详情返回异常：{response.status_code}')
                    rejection = response.json()
                    expected = 'Maximum number queued requests for this dataset has been reached. Please configure your scripts to request data sequentially'
                    if rejection.get('traceback') != expected or quota_rejections >= 3:
                        raise RuntimeError(f'CDS请求拒绝：{remote.request_id}/{rejection}')
                    quota_rejections += 1
                    archived = request_path.with_name(request_path.stem + f'.quota-rejected-{remote.request_id}.json')
                    request_path.replace(archived)
                    write_json(archived.with_suffix('.reason.json'), rejection)
                    update(period, 'waiting_for_queue_slot', remote.request_id)
                    time.sleep(60)
                    remote = client.submit(dataset, request)
                    write_json(request_path, {'dataset': dataset, 'request': request,
                                             'request_id': remote.request_id, 'created_utc': utc_now()})
                    if remote.collection_id != dataset or remote.request != request:
                        raise ValueError('CDS重试请求内容与授权范围不同')
                    continue
                if status not in ('accepted', 'running'):
                    raise RuntimeError(f'CDS定向请求失败：{remote.request_id}/{status}')
                time.sleep(15)
            temporary = target.with_suffix('.nc.part')
            remote.download(str(temporary))
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
        periods = [(2023, None), (2024, 1), (2025, 1), (2026, None)] + [
            (year, month) for month in range(2, 13) for year in (2024, 2025)]
        for year, month in periods:
            received[(year, month)] = fetch(year, month)
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
    records = [record for (year, month), record in received.items()
               if record is not None and (month is None or year not in annual_received)]
    records.extend(annual_received.values())
    records.sort(key=lambda item: (item['request']['year'][0], item['request']['month'][0]))
    datasets = [xr.open_dataset(record['path']) for record in records]
    combined = xr.concat(datasets, dim='valid_time').sortby('valid_time')
    if combined.indexes['valid_time'].has_duplicates or combined.indexes['valid_time'].get_indexer(times).min() < 0:
        raise ValueError('CDS定向原文存在重复或缺少研究小时')
    table = pd.DataFrame({'time': np.repeat(times, len(points)),
                          'native_flat_index': np.tile(points.native_flat_index.to_numpy(dtype=np.int32), len(times))})
    checks = {}
    for name in variables:
        if combined[name].attrs.get('units') != expected_units:
            raise ValueError('CDS指定变量单位与研究要求不同')
        values = combined[name].sel(valid_time=times).sel(
            latitude=xr.DataArray(points.latitude.to_numpy(), dims='weather_point'),
            longitude=xr.DataArray(((points.longitude_0_360.to_numpy() + 180) % 360 - 180), dims='weather_point')).values
        if values.shape != original[name].shape or values.dtype != np.dtype('float32') or not np.isfinite(values).all():
            raise ValueError('CDS定向原文的形状、类型或完整性不符合要求')
        valid = np.isfinite(original[name])
        difference = np.abs(values[valid].astype(np.float64) - original[name][valid].astype(np.float64))
        checks[name] = {'original_nan_count': int(np.isnan(original[name]).sum()), 'verified_overlap_count': int(valid.sum()),
                        'max_abs_overlap_difference': float(difference.max()) if difference.size else None,
                        'exact_overlap_equal': bool(np.array_equal(values[valid], original[name][valid]))}
        table[name] = values.reshape(-1)
    output = root / 'data/interim/era5-eligible-cds-repairs' / (label + '.parquet')
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
    args = parser.parse_args()
    with (args.root.resolve() / f'.era5-{args.group}-repair.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(args.root.resolve(), args.block, args.group, args.monthly)
