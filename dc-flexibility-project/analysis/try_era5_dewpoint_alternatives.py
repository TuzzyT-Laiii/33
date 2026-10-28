import argparse
import fcntl
import json
import io
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import zipfile

from ecmwf.datastores import Client
import fsspec
from fsspec.exceptions import FSTimeoutError
import numpy as np
import pandas as pd
import requests
import xarray as xr

from acquire_buildability_protection import sha, utc_now, write_json
from acquire_eligible_era5_weather import open_source
from era5_dewpoint_candidate import differences, publish_candidate


def context(root, label):
    selection_path = root / 'data/metadata/buildability-20261025/selection-manifest.json'
    selection = json.loads(selection_path.read_text())
    points = pd.read_parquet(selection['outputs']['points']['path']).sort_values('native_flat_index')
    if sha(Path(selection['outputs']['points']['path'])) != selection['outputs']['points']['sha256']:
        raise ValueError('替代来源的筛选输入SHA不同')
    lat = np.rint((points.latitude + 90) * 4).astype(int) // 4
    lon = np.rint(((points.longitude_0_360 + 180) % 360) * 4).astype(int) // 4
    selected = np.array([f'lat-{a:03d}-lon-{b:03d}' == label for a, b in zip(lat, lon)])
    points = points.loc[selected]
    source, session, times, metadata_sha = open_source(root)
    with source:
        original = source.d2m.sel(time=times).sel(latitude=xr.DataArray(points.latitude.to_numpy(), dims='weather_point'),
            longitude=xr.DataArray(((points.longitude_0_360.to_numpy() + 180) % 360 - 180), dims='weather_point')).values
    session.close()
    if original.shape != (26305, 10) or original.dtype != np.dtype('float32'):
        raise ValueError('替代来源仅适用于已核查的10个节点')
    return points, times, original, sha(selection_path), metadata_sha


def open_timeseries(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith('.nc')]
            if len(names) != 1:
                raise ValueError('点位时间序列压缩包必须只含一份NetCDF')
            return xr.open_dataset(io.BytesIO(archive.read(names[0])))
    return xr.open_dataset(path)


def compare_and_publish(root, label, source_name, points, times, original, selection_sha, metadata_sha, values, supplied_times, records):
    positions = pd.Index(times).get_indexer(supplied_times)
    if positions.min() < 0 or values.shape != (len(supplied_times), len(points)) or values.dtype != np.dtype('float32'):
        raise ValueError('替代来源时间、形状或float32不符合要求')
    annual_path = root / 'data/raw/era5-eligible-cds-repairs' / label / 'dewpoint-2023.nc'
    annual_receipt = json.loads(annual_path.with_suffix('.json').read_text())
    if sha(annual_path) != annual_receipt['sha256']:
        raise ValueError('2023官方验证文件SHA不同')
    index = pd.DatetimeIndex(supplied_times)
    january = (index >= '2023-01-01') & (index < '2023-02-01')
    july = (index >= '2024-07-01') & (index < '2024-08-01')
    if january.sum() != 744 or july.sum() != 744:
        raise ValueError('替代来源缺少完整的2023-01及2024-07重合月')
    with xr.open_dataset(annual_path) as source:
        official = source.d2m.sel(valid_time=supplied_times[january]).sel(
            latitude=xr.DataArray(points.latitude.to_numpy(), dims='weather_point'),
            longitude=xr.DataArray(((points.longitude_0_360.to_numpy() + 180) % 360 - 180), dims='weather_point')).values
    checks = {'official_2023_01': differences(official, values[january]),
              'arco_2024_07': differences(original[positions[july]], values[july])}
    fill_months = (index >= '2024-01-01') & (index < '2024-07-01')
    finite = bool(np.isfinite(values).all())
    result = {'checked_utc': utc_now(), 'source': source_name, 'block': label, 'comparison': checks,
              'missing_values_2024_jan_jun': int(np.isnan(values[fill_months]).sum()),
              'candidate_finite': finite,
              'status': 'compatible' if finite and all(item['exact_equal'] for item in checks.values()) else 'incompatible',
              'raw_sources': records}
    write_json(root / f'results/buildability-20261025/dewpoint-alternative-{source_name}-comparison.json', result)
    if result['status'] != 'compatible':
        return result
    present = np.isfinite(original[positions])
    if not np.array_equal(values[present], original[positions][present]):
        result.update(status='incompatible', all_covered_finite=differences(original[positions][present], values[present]))
        write_json(root / f'results/buildability-20261025/dewpoint-alternative-{source_name}-comparison.json', result)
        return result
    repaired = original.copy()
    annual_times = times[(times >= np.datetime64('2023-01-01')) & (times < np.datetime64('2024-01-01'))]
    annual_positions = pd.Index(times).get_indexer(annual_times)
    with xr.open_dataset(annual_path) as source:
        annual_values = source.d2m.sel(valid_time=annual_times).sel(latitude=xr.DataArray(points.latitude.to_numpy(), dims='weather_point'),
            longitude=xr.DataArray(((points.longitude_0_360.to_numpy() + 180) % 360 - 180), dims='weather_point')).values
    if annual_values.dtype != np.dtype('float32') or not np.isfinite(annual_values).all():
        raise ValueError('既有2023官方补取文件类型或有限值不同')
    annual_selected = repaired[annual_positions].copy()
    annual_missing = np.isnan(annual_selected)
    annual_selected[annual_missing] = annual_values[annual_missing]
    repaired[annual_positions] = annual_selected
    selected = repaired[positions].copy()
    missing = np.isnan(selected)
    selected[missing] = values[missing]
    repaired[positions] = selected
    valid = np.isfinite(original)
    if not np.isfinite(repaired).all() or not np.array_equal(repaired[valid], original[valid]):
        raise ValueError('替代来源未补齐全部NaN或改变了原有效值')
    table = pd.DataFrame({'time': np.repeat(times, len(points)),
                          'native_flat_index': np.tile(points.native_flat_index.to_numpy(dtype=np.int32), len(times)),
                          'd2m': repaired.reshape(-1)})
    proof = {'block': label, 'variables': ['d2m'], 'source_file_prefix': 'dewpoint', 'hours': 26305,
             'selection_manifest_sha256': selection_sha, 'source_metadata_sha256': metadata_sha,
             'native_flat_indices': points.native_flat_index.astype(int).tolist(), 'official_dataset': source_name,
             'raw_sources': [annual_receipt, *records], 'missing_only': True, 'alternative_comparisons': checks,
             'checks': {'d2m': {'original_nan_count': int(np.isnan(original).sum()), 'verified_overlap_count': int(july.sum()) * len(points),
                                'exact_overlap_equal': True, 'max_abs_overlap_difference': 0.0, 'overlap_period': '2024-07',
                                'unchanged_original_finite_count': int(valid.sum())}}, 'created_utc': utc_now()}
    winner = publish_candidate(root, table, proof)
    result.update(status='published_verified', winner=winner['official_dataset'])
    from complete_era5_wind_repair import verify
    result['formal'] = verify(root, label)
    import subprocess
    import sys
    subprocess.run([sys.executable, str(root / 'analysis/grid_job_manager.py'), '--start', 'era5-eligible', '--ensure-running'], check=True)
    return result


def timeseries(root, label, points, times, original, selection_sha, metadata_sha, state, state_path):
    key = json.loads((root / '.codex-temp/buildability-20261025/cds-credential.json').read_text())['key']
    client = Client(url='https://cds.climate.copernicus.eu/api', key=key, cleanup=False, progress=False,
                    timeout=40, maximum_tries=2, retry_after=2)
    dataset = 'reanalysis-era5-single-levels-timeseries'
    directory = root / 'data/raw/era5-eligible-cds-repairs' / label / 'timeseries'
    directory.mkdir(parents=True, exist_ok=True)
    columns, records = [], []
    supplied_times = pd.date_range('2023-01-01', '2024-07-31T23:00:00', freq='h').to_numpy()
    for point in points.itertuples():
        target = directory / f'{point.native_flat_index}.nc'
        request = {'variable': ['2m_dewpoint_temperature'], 'date': ['2023-01-01', '2024-07-31'],
                   'location': {'latitude': float(point.latitude), 'longitude': float((point.longitude_0_360 + 180) % 360 - 180)},
                   'data_format': 'netcdf'}
        saved_path = target.with_suffix('.request.json')
        if saved_path.exists():
            saved = json.loads(saved_path.read_text())
            if saved['request'] != request or saved['dataset'] != dataset:
                raise ValueError('点位时间序列持久请求范围不同')
            remote = client.get_remote(saved['request_id'])
        else:
            remote = client.submit(dataset, request)
            write_json(saved_path, {'request_id': remote.request_id, 'dataset': dataset, 'request': request, 'created_utc': utc_now()})
        if remote.collection_id != dataset or remote.request != request:
            raise ValueError('点位时间序列官方请求范围不同')
        while not target.exists():
            status = remote.status
            state.update(status=status, node=int(point.native_flat_index), request_id=remote.request_id, updated_utc=utc_now())
            write_json(state_path, state)
            if status == 'successful':
                temporary = target.with_suffix('.nc.part')
                remote.download(str(temporary))
                temporary.replace(target)
                break
            if status not in ('accepted', 'running'):
                raise RuntimeError(f'点位时间序列请求未成功：{remote.request_id}/{status}')
            proof_path = root / 'data/metadata/buildability-20261025/weather-cds-repairs' / (label + '.json')
            if proof_path.exists() and json.loads(proof_path.read_text())['status'] == 'verified':
                return {'status': 'other_source_verified', 'request_id': remote.request_id}
            time.sleep(15)
        record = {'dataset': dataset, 'source_layout': 'point_timeseries', 'request': request,
                  'request_id': remote.request_id, 'path': str(target), 'sha256': sha(target), 'bytes': target.stat().st_size}
        write_json(target.with_suffix('.json'), record)
        with open_timeseries(target) as source:
            variable = source['d2m']
            if variable.attrs['units'] != 'K' or float(source.latitude.values) != request['location']['latitude'] or float(source.longitude.values) != request['location']['longitude']:
                raise ValueError('时间序列单位或精确坐标不同')
            if source.indexes['valid_time'].has_duplicates:
                raise ValueError('时间序列存在重复小时')
            values = variable.sel(valid_time=supplied_times).values.reshape(-1)
        columns.append(values)
        records.append(record)
        state.update(received_nodes=len(records), updated_utc=utc_now())
        write_json(state_path, state)
    return compare_and_publish(root, label, dataset, points, times, original, selection_sha, metadata_sha,
                               np.column_stack(columns), supplied_times, records)


def ncar(root, label, points, times, original, selection_sha, metadata_sha, state, state_path):
    bucket = 'https://nsf-ncar-era5.s3.us-west-2.amazonaws.com'
    directory = root / 'data/raw/era5-eligible-cds-repairs' / label / 'ncar'
    directory.mkdir(parents=True, exist_ok=True)
    series, records = [], []
    # 两个完整重合月先核验；数值不同时停止，不下载待填补月份。
    for month in ['202301', '202407', '202401', '202402', '202403', '202404', '202405', '202406']:
        prefix = f'e5.oper.an.sfc/{month}/e5.oper.an.sfc.128_168_2d.'
        response = requests.get(bucket, params={'list-type': '2', 'prefix': prefix}, timeout=40)
        response.raise_for_status()
        document = ET.fromstring(response.content)
        names = [item.text for item in document.findall('.//{*}Contents/{*}Key')]
        if len(names) != 1:
            raise ValueError(f'NCAR指定月份露点文件数量异常：{month}/{len(names)}')
        url = bucket + '/' + names[0]
        state.update(status='reading_public_month', month=month, updated_utc=utc_now())
        write_json(state_path, state)
        with fsspec.open(url, mode='rb', block_size=256 * 1024, cache_type='readahead', timeout=40).open() as stream:
            with xr.open_dataset(stream, engine='h5netcdf', chunks=None) as source:
                names = [name for name in ('D2M', 'd2m', 'VAR_2D') if name in source]
                if len(names) != 1:
                    raise ValueError('NCAR露点变量名称需要核对：' + str(list(source.data_vars)))
                variable = source[names[0]]
                if variable.attrs['units'] != 'K':
                    raise ValueError('NCAR露点单位不同')
                extracted = variable.sel(latitude=xr.DataArray(points.latitude.to_numpy(), dims='weather_point'),
                    longitude=xr.DataArray(points.longitude_0_360.to_numpy(), dims='weather_point')).load()
        extracted = extracted.rename(names[0]).to_dataset()
        target = directory / f'{month}.nc'
        extracted.to_netcdf(target)
        record = {'dataset': 'nsf-ncar-era5', 'source_layout': 'ncar_extracted', 'month': month,
                  'source_url': url, 'path': str(target), 'sha256': sha(target), 'bytes': target.stat().st_size}
        write_json(target.with_suffix('.json'), record)
        records.append(record)
        series.append(extracted[names[0]])
        if len(series) == 2:
            first_times = series[0].time.values
            july_times = series[1].time.values
            official_path = root / 'data/raw/era5-eligible-cds-repairs' / label / 'dewpoint-2023.nc'
            official_receipt = json.loads(official_path.with_suffix('.json').read_text())
            if sha(official_path) != official_receipt['sha256']:
                raise ValueError('2023官方文件SHA不同')
            with xr.open_dataset(official_path) as official:
                reference = official.d2m.sel(valid_time=first_times).sel(latitude=xr.DataArray(points.latitude.to_numpy(), dims='weather_point'),
                    longitude=xr.DataArray(((points.longitude_0_360.to_numpy() + 180) % 360 - 180), dims='weather_point')).values
            comparison = {'official_2023_01': differences(reference, series[0].values),
                          'arco_2024_07': differences(original[pd.Index(times).get_indexer(july_times)], series[1].values)}
            if not all(item['exact_equal'] for item in comparison.values()):
                return {'status': 'incompatible', 'comparison': comparison, 'raw_sources': records}
    combined = xr.concat(series, dim='time').sortby('time')
    return compare_and_publish(root, label, 'nsf-ncar-era5', points, times, original, selection_sha, metadata_sha,
                               combined.values, combined.time.values, records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', choices=('timeseries', 'ncar'), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    label = 'lat-084-lon-290'
    state_path = root / f'status/era5-dewpoint-alternative-{args.source}.json'
    state = {'status': 'starting', 'started_utc': utc_now(), 'source': args.source, 'block': label}
    with (root / f'.era5-dewpoint-alternative-{args.source}.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(state_path, state)
        try:
            values = context(root, label)
            function = timeseries if args.source == 'timeseries' else ncar
            result = function(root, label, *values, state, state_path)
            state.update(result, updated_utc=utc_now())
        except (ValueError, KeyError, OSError, RuntimeError, requests.RequestException, FSTimeoutError) as error:
            state.update(status='failed', error_type=type(error).__name__, error=str(error)[:1500], updated_utc=utc_now())
            write_json(state_path, state)
            raise
        write_json(state_path, state)
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
