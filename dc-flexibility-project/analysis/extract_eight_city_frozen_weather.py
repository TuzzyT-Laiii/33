import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import xarray as xr

from acquire_eight_city_weather import VARIABLES, check_file, sha, utc_now, write_json


def run(args):
    source_dir = args.shared_root / 'data/raw/era5-eligible-ecmwf-geo'
    cities = json.loads(args.locations.read_text(encoding='utf-8'))
    frozen_raw = args.output / 'frozen-raw'
    frozen_raw.mkdir(exist_ok=True)
    shutil.copyfile(source_dir / '.zmetadata', frozen_raw / '.zmetadata')
    with xr.open_zarr(source_dir, consolidated=True, chunks=None) as source:
        for city in cities:
            target = args.output / (city['city_id'] + '-frozen.nc')
            lat = round(city.get('weather_latitude', city['latitude']) * 4) / 4
            lon = round(city.get('weather_longitude', city['longitude']) * 4) / 4
            selected = source[[v[0] for v in VARIABLES.values()]].sel(
                time=slice('2023-01-01', '2025-12-31T23:00:00'), latitude=lat, longitude=lon).load()
            missing = {v: int(np.isnan(selected[v].values).sum()) for v in selected.data_vars}
            record = {'city_id': city['city_id'], 'source': str(source_dir),
                'metadata_sha256': sha(source_dir / '.zmetadata'), 'missing': missing, 'checked_utc': utc_now()}
            # 温湿度先行输出也必须来自真实且完整的同一冻结原始版本。
            raw_objects = []
            for variable in [v[0] for v in VARIABLES.values() if v[0] not in ('u10','v10')]:
                lat_chunk, lon_chunk = round((lat+90)*4)//4, round((lon+180)*4)//4
                for chunk in (10, 11):
                    key = f'{variable}/{chunk}.{lat_chunk}.{lon_chunk}'
                    src, dst = source_dir / key, frozen_raw / key
                    if not src.exists():
                        raise ValueError('冻结源实际原始对象缺失：' + key)
                    dst.parent.mkdir(exist_ok=True)
                    if not dst.exists():
                        shutil.copyfile(src, dst)
                    raw_objects.append({'path': str(dst), 'sha256': sha(dst), 'bytes': dst.stat().st_size})
            record['raw_objects'] = raw_objects
            for label, names in [('thermal', ['t2m','d2m','sp']),
                ('base', [v[0] for v in VARIABLES.values() if v[0] not in ('u10','v10')]),
                ('complete', [v[0] for v in VARIABLES.values()])]:
                if not any(missing[name] for name in names):
                    path = target if label == 'complete' else args.output / (city['city_id'] + '-frozen-' + label + '.nc')
                    selected[names].to_netcdf(path)
                    record[label] = {'path': str(path), 'sha256': sha(path)}
            if 'complete' in record:
                validation = check_file(target, city)
                write_json(target.with_suffix('.validation.json'), validation)
                record['validation'] = validation
            write_json(args.output / (city['city_id'] + '-frozen.source.json'), record)
            print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--locations', required=True, type=Path)
    run(parser.parse_args())
