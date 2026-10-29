import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import time
import zipfile

from ecmwf.datastores import Client
import numpy as np
import pandas as pd
import requests
import xarray as xr

from era5_cds_request import CDSRequestGate


VARIABLES = {
    '2m_temperature': ('t2m', 'K'),
    '2m_dewpoint_temperature': ('d2m', 'K'),
    'surface_pressure': ('sp', 'Pa'),
    'surface_solar_radiation_downwards': ('ssrd', 'J m**-2'),
    'total_sky_direct_solar_radiation_at_surface': ('fdir', 'J m**-2'),
    '10m_u_component_of_wind': ('u10', 'm s**-1'),
    '10m_v_component_of_wind': ('v10', 'm s**-1'),
    '100m_u_component_of_wind': ('u100', 'm s**-1'),
    '100m_v_component_of_wind': ('v100', 'm s**-1'),
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_file(path, city):
    expected = pd.date_range('2023-01-01', '2026-01-01', freq='h', inclusive='left')
    with xr.open_dataset(path) as ds:
        clock = 'valid_time' if 'valid_time' in ds.coords else 'time'
        actual = pd.DatetimeIndex(ds[clock].values)
        if not actual.equals(expected):
            raise ValueError(f'{city["city_id"]} UTC覆盖错误：{len(actual)}')
        checks, table = {}, pd.DataFrame({'time_utc': actual.tz_localize('UTC')})
        for long_name, (short_name, units) in VARIABLES.items():
            name = short_name if short_name in ds else long_name
            values = ds[name].values.squeeze()
            if values.shape != (len(expected),) or not np.isfinite(values).all():
                raise ValueError(f'{name} 存在缺失或不符合点时间序列维度')
            actual_units = ds[name].attrs['units']
            if actual_units != units:
                raise ValueError(f'{name} 单位异常：{actual_units}，预期{units}')
            table[short_name] = values
            checks[short_name] = {'units': units, 'min': float(values.min()), 'max': float(values.max()), 'missing': 0}
        lat, lon = float(ds.latitude.values.squeeze()), float(ds.longitude.values.squeeze())
    table.to_parquet(path.with_suffix('.parquet'), index=False, compression='zstd')
    return {'hours': len(expected), 'year_hours': {str(y): int((actual.year == y).sum()) for y in (2023, 2024, 2025)},
            'grid_latitude': lat, 'grid_longitude': lon, 'variables': checks,
            'sha256': sha(path), 'bytes': path.stat().st_size,
            'parquet_sha256': sha(path.with_suffix('.parquet'))}


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    cities = json.loads(args.locations.read_text(encoding='utf-8'))
    key = json.loads((args.shared_root / '.codex-temp/buildability-20261025/cds-credential.json').read_text())['key']
    client = Client(url='https://cds.climate.copernicus.eu/api', key=key, cleanup=False,
                    progress=False, timeout=60, maximum_tries=3, retry_after=5, sleep_max=30)
    gate = CDSRequestGate(args.shared_root, client)
    state = {'started_utc': utc_now(), 'status': 'running', 'cities': {}, 'max_own_active': 1,
             'coordinate_source': 'https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv'}
    state_path = args.output / 'status.json'
    for city in cities:
        name = city['city_id']
        request = {'variable': list(VARIABLES), 'date': ['2023-01-01', '2025-12-31'],
                   'location': {'latitude': city.get('weather_latitude', city['latitude']),
                                'longitude': city.get('weather_longitude', city['longitude'])}, 'data_format': 'netcdf'}
        dataset = 'reanalysis-era5-single-levels-timeseries'
        receipt = args.output / (name + '.request.json')
        target = args.output / (name + '.nc')
        state['cities'][name] = {'status': 'waiting_for_global_slot', 'request': request, 'updated_utc': utc_now()}
        write_json(state_path, state)
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved['request'] != request:
                raise ValueError('已保存的城市请求与当前输入不同')
            remote = client.get_remote(saved['request_id'])
        else:
            while True:
                with gate.locked():
                    snapshot = gate.snapshot()
                    if snapshot['active_count'] < snapshot['max_active_requests']:
                        # 保存提交意图，网络中断后禁止无编号重复提交。
                        intent = receipt.with_suffix('.intent.json')
                        if intent.exists():
                            raise RuntimeError(f'存在不确定提交，先核对CDS账号编号：{intent}')
                        write_json(intent, {'dataset': dataset, 'request': request, 'utc': utc_now()})
                        remote = client.submit(dataset, request)
                        write_json(receipt, {'dataset': dataset, 'request': request,
                                            'request_id': remote.request_id, 'submitted_utc': utc_now()})
                        gate.remember(remote.request_id, receipt)
                        intent.unlink()
                        break
                state['cities'][name]['global_gate'] = snapshot
                write_json(state_path, state)
                time.sleep(30)
        while True:
            status = remote.status
            state['cities'][name].update(status=status, request_id=remote.request_id, updated_utc=utc_now())
            write_json(state_path, state)
            if status == 'successful':
                break
            if status not in ('accepted', 'running'):
                raise RuntimeError(f'城市请求失败，保留编号等待核查：{name}/{remote.request_id}/{status}')
            time.sleep(20)
        if not target.exists():
            temporary = target.with_suffix('.nc.part')
            # 流式完整下载避免SDK对压缩HTTP响应的长度比较；完成后仍核验NetCDF覆盖与数值。
            results = remote.get_results()
            with requests.get(results.location, stream=True, timeout=120) as response:
                response.raise_for_status()
                with temporary.open('wb') as stream:
                    for block in response.iter_content(chunk_size=1024*1024):
                        stream.write(block)
                if response.headers.get('Content-Encoding', 'identity') == 'identity':
                    length = response.headers.get('Content-Length')
                    if length is not None and temporary.stat().st_size != int(length):
                        raise ValueError('CDS下载字节数与HTTP声明不一致')
            temporary.replace(target)
        if zipfile.is_zipfile(target):
            archive = target.with_suffix('.response.zip')
            target.replace(archive)
            with zipfile.ZipFile(archive) as zipped:
                members = [m for m in zipped.infolist() if not m.is_dir()]
                if len(members) != 1 or not members[0].filename.endswith('.nc'):
                    raise ValueError('CDS点时间序列压缩包需恰好包含一个NetCDF文件')
                with zipped.open(members[0]) as source, target.open('wb') as output:
                    for block in iter(lambda: source.read(1024*1024), b''):
                        output.write(block)
            write_json(args.output / (name + '.response.json'), {
                'request_id': remote.request_id, 'archive_sha256': sha(archive),
                'archive_bytes': archive.stat().st_size, 'netcdf_member': members[0].filename})
        validation = check_file(target, city)
        write_json(args.output / (name + '.validation.json'), validation)
        state['cities'][name].update(status='verified', validation=validation, updated_utc=utc_now())
        write_json(state_path, state)
        print(json.dumps({'city': name, 'event': 'verified', 'request_id': remote.request_id,
                          'hours': validation['hours'], 'utc': utc_now()}), flush=True)
    state.update(status='complete', completed_utc=utc_now())
    write_json(state_path, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--locations', type=Path, default=Path(__file__).with_name('eight_city_locations.json'))
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    with (arguments.output / '.controller.lock').open('a+') as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(arguments)
