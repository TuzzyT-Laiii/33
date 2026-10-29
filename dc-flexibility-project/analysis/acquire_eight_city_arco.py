import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from acquire_eligible_era5_weather import source_session
from trial_era5_landuse_download import CachedHTTPStore, SOURCE_URL
from acquire_eight_city_weather import VARIABLES, check_file, sha, utc_now, write_json


def run(args):
    cities = json.loads(args.locations.read_text(encoding='utf-8'))
    store_dir = args.output / 'arco-raw'
    with source_session(args.shared_root) as session:
        store = CachedHTTPStore(store_dir, session)
        with xr.open_zarr(store, consolidated=True, chunks=None) as source:
            for city in cities:
                target = args.output / (city['city_id'] + '-arco.nc')
                receipt_path = target.with_suffix('.source.json')
                if target.exists() and receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text())
                    if receipt['sha256'] != sha(target):
                        raise ValueError('ARCO城市输出SHA不一致')
                    continue
                lat = round(city.get('weather_latitude', city['latitude']) * 4) / 4
                lon = round(city.get('weather_longitude', city['longitude']) * 4) / 4
                base_path = args.output / (city['city_id'] + '-frozen-base.nc')
                if not base_path.exists():
                    raise ValueError('先提取和核验完整的七变量冻结ERA5原始输入')
                with xr.open_dataset(base_path) as base:
                    selected = base.load()
                current_wind = source[['u10','v10']].sel(
                    time=slice('2023-01-01', '2025-12-31T23:00:00'), latitude=lat, longitude=lon).load()
                selected = xr.merge([selected, current_wind], join='exact', compat='equals')
                missing = {v: int(np.isnan(selected[v].values).sum()) for v in selected.data_vars}
                write_json(args.output / (city['city_id'] + '-arco-coverage.json'),
                           {'source_url': SOURCE_URL, 'missing': missing, 'utc': utc_now()})
                if any(missing.values()):
                    print(json.dumps({'city': city['city_id'], 'status': 'source_missing', 'missing': missing}), flush=True)
                    continue
                selected.to_netcdf(target)
                validation = check_file(target, city)
                write_json(target.with_suffix('.validation.json'), validation)
                write_json(receipt_path, {'source_url': SOURCE_URL, 'access_method': 'official_ARCO_backing_CDS_timeseries',
                    'request_id': None, 'reason': 'existing_CDS_global_gate_fully_occupied; direct_official_backing_store',
                    'created_utc': utc_now(), 'sha256': sha(target), 'validation': validation,
                    'frozen_base': {'path': str(base_path), 'sha256': sha(base_path),
                                    'receipt': city['city_id'] + '-frozen.source.json'},
                    'current_arco_variables': ['u10', 'v10'],
                    'objects': [{'path': str(p.relative_to(args.output)), 'sha256': sha(p), 'bytes': p.stat().st_size}
                                for p in store_dir.rglob('*') if p.is_file()]})
                print(json.dumps({'city': city['city_id'], 'event': 'arco_verified', 'utc': utc_now()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--locations', required=True, type=Path)
    run(parser.parse_args())
