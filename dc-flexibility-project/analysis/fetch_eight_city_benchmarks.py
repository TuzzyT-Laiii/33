import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import rasterio
from rasterio.warp import transform
import requests


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    cities = json.loads(args.locations.read_text(encoding='utf-8'))
    session = requests.Session()
    receipt = []

    def get(name, url, params=None):
        path = output / name
        if not path.exists():
            response = session.get(url, params=params, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
            access_url = response.url
        else:
            access_url = requests.Request('GET', url, params=params).prepare().url
        receipt.append({'file': name, 'url': access_url, 'sha256': digest(path), 'bytes': path.stat().st_size})
        return path

    get('IEA_Reference_3.4MW_130.csv', 'https://raw.githubusercontent.com/NREL/turbine-models/main/turbine_models/data/Onshore/IEA_Reference_3.4MW_130.csv')
    get('IEA_Reference_3.4MW_130.yaml', 'https://raw.githubusercontent.com/NREL/turbine-models/main/turbine_models/specs/Onshore/IEA_Reference_3.4MW_130.yaml')
    get('isd-history.csv', 'https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv')
    get('gwa-mapframe.bundle.js', 'https://globalwindatlas.info/mapframe.bundle.js')
    station = pd.read_csv(output / 'isd-history.csv', dtype={'USAF': str})
    records = []
    for city in cities:
        match = station.loc[station.USAF.eq(city['station_id'])]
        if len(match) != 1 or abs(match.LAT.iloc[0] - city['latitude']) > 1e-6 or abs(match.LON.iloc[0] - city['longitude']) > 1e-6:
            raise ValueError('城市代表点与NOAA站点记录不一致')
        solar = get(city['city_id'] + '-gsa.json', 'https://api.globalsolaratlas.info/data/lta',
                    {'loc': f'{city["latitude"]},{city["longitude"]}'})
        solar_data = json.loads(solar.read_text())['annual']
        data = solar_data['data']
        if 'PVOUT_csi' in data:
            layer = solar_data['metadata']['layers']['PVOUT_csi']
            pv_specific, pv_source = data['PVOUT_csi'], 'Global Solar Atlas '+layer['version']
            pv_period, pv_update = layer['period'], layer['updated']
        elif city['latitude'] > 60:
            pvgis = get(city['city_id'] + '-pvgis.json', 'https://re.jrc.ec.europa.eu/api/v5_3/PVcalc',
                {'lat': city['latitude'], 'lon': city['longitude'], 'peakpower': 1, 'loss': 14,
                 'angle': min(abs(city['latitude']), 60), 'aspect': 0, 'outputformat': 'json'})
            pv_specific = json.loads(pvgis.read_text())['outputs']['totals']['fixed']['E_y']
            pv_source = 'JRC PVGIS 5.3; GSA solar coverage unavailable north of 60N'
            pv_period = json.loads(pvgis.read_text())['inputs']['meteo_data']
            pv_update = 'PVGIS 5.3 API retrieval'
        else:
            raise ValueError('GSA太阳能基准缺失且未定义适用的其他公开数据源')
        records.append({**city, 'benchmark_pv_source': pv_source, 'benchmark_pv_specific_kwh_kwp_year': pv_specific,
                        'benchmark_pv_cf_dc_basis': pv_specific / 8760,
                        'benchmark_pv_period_metadata': json.dumps(pv_period,ensure_ascii=False),
                        'benchmark_pv_update': pv_update,
                        'gsa_ghi_kwh_m2_year': data.get('GHI'), 'gsa_temperature_c': data.get('TEMP'),
                        'gsa_optimum_tilt_deg': data.get('OPTA')})
        # 从GWA4官网实际使用的GeoTIFF图层读取同一点；第三方格式库负责解析栅格。
        zoom = 10
        tile_x = math.floor((city['longitude'] + 180) / 360 * 2**zoom)
        tile_y = math.floor((1 - math.asinh(math.tan(math.radians(city['latitude']))) / math.pi) / 2 * 2**zoom)
        wind = get(city['city_id'] + '-gwa-iec3.tif',
            f'https://tiles-stag.ramtt.xyz/titiler/gwa4/cf_iec3/tiles/WebMercatorQuad/{zoom}/{tile_x}/{tile_y}.tif')
        with rasterio.open(wind) as dataset:
            x, y = transform('EPSG:4326', dataset.crs, [city['longitude']], [city['latitude']])
            value = float(next(dataset.sample([(x[0], y[0])]))[0])
            if value == dataset.nodata or not 0 <= value <= 1:
                raise ValueError(f'GWA容量因子像元无效：{city["city_id"]}/{value}')
            records[-1]['gwa_iec3_capacity_factor'] = value
            records[-1]['gwa_pixel_width_m'] = abs(dataset.transform.a)
        print(city['city_id'], 'PV', records[-1]['benchmark_pv_cf_dc_basis'], 'GWA', value, flush=True)
    pd.DataFrame(records).to_csv(output / 'city-benchmarks.csv', index=False, encoding='utf-8-sig')
    (output / 'source-receipts.json').write_text(json.dumps({'checked_utc': datetime.now(timezone.utc).isoformat(),
        'sources': receipt}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--locations', type=Path, default=Path(__file__).with_name('eight_city_locations.json'))
    parser.add_argument('--output', required=True, type=Path)
    fetch(parser.parse_args())
