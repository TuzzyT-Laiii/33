import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
from functools import lru_cache
import inspect
import json
from io import StringIO
import os
from pathlib import Path
import shutil
import threading
import time

import numpy as np
import pandas as pd

from compute_load.cooling import CoolingModel
from flex_econ_copt import CITIES, LAYERS, SCENARIOS


VARIABLES = {
    't2m': '2m_temperature', 'd2m': '2m_dewpoint_temperature', 'sp': 'surface_pressure',
    'u100': '100m_u_component_of_wind', 'v100': '100m_v_component_of_wind',
    'ssrd': 'mean_surface_downward_short_wave_radiation_flux',
    'fdir': 'mean_surface_direct_short_wave_radiation_flux',
}
WEATHER_CLOCK = pd.date_range('2025-01-01', '2026-01-01', freq='h', inclusive='left', tz='UTC')
WIND10_URL = 'https://arco.datastores.ecmwf.int/cadl-arco-geo-002/arco/reanalysis_era5_single_levels/sfc/geoChunked.zarr'
HTTP_CONTEXT = threading.local()


def digest(path):
    value = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def settings(config):
    own = json.loads(Path(config).read_text('utf-8'))
    return own, {key: Path(value) for key, value in own['paths'].items()}


def spherical_distance(lat, lon, target_lat, target_lon):
    lat, target_lat = np.deg2rad(lat), np.deg2rad(target_lat)
    dlon = np.deg2rad((np.asarray(lon) - target_lon + 180) % 360 - 180)
    a = np.sin((lat - target_lat) / 2) ** 2 + np.cos(lat) * np.cos(target_lat) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def nearest_boundary(lat, lon, countries, restricted_index=None):
    from pyproj import CRS, Geod, Transformer
    import shapely
    from shapely.ops import nearest_points, transform

    geod = Geod(ellps='WGS84')
    local = CRS.from_proj4(f'+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m')
    forward = Transformer.from_crs(4326, local, always_xy=True).transform
    inverse = Transformer.from_crs(local, 4326, always_xy=True).transform
    origin = shapely.Point(0, 0)
    radius = 25.
    while radius <= 25600:
        dlat = radius / 110.
        latitude_edge = min(89.999, abs(lat) + dlat)
        dlon = min(180., radius / (110 * np.cos(np.deg2rad(latitude_edge))))
        south, north = max(-90., lat - dlat), min(90., lat + dlat)
        if lon - dlon < -180:
            boxes = [shapely.box(-180, south, lon + dlon, north), shapely.box(lon - dlon + 360, south, 180, north)]
        elif lon + dlon > 180:
            boxes = [shapely.box(lon - dlon, south, 180, north), shapely.box(-180, south, lon + dlon - 360, north)]
        else:
            boxes = [shapely.box(lon - dlon, south, lon + dlon, north)]
        window = shapely.union_all(boxes)
        candidates = [restricted_index] if restricted_index is not None else countries.sindex.query(window, predicate='intersects')
        best = None
        for index in candidates:
            edge = countries.loc[int(index), 'boundary_geometry'].intersection(window)
            if edge.is_empty:
                continue
            # 在中心点的等距投影内求最近边界；细分使投影后的折线误差低于制图数据精度。
            edge = shapely.segmentize(edge, max_segment_length=.01)
            projected = transform(forward, edge)
            point = nearest_points(origin, projected)[1]
            boundary_lon, boundary_lat = inverse(point.x, point.y)
            distance = geod.inv(lon, lat, boundary_lon, boundary_lat)[2] / 1000
            candidate = (float(distance), countries.loc[int(index), 'iso3'], int(index), float(boundary_lat), float(boundary_lon))
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is not None and best[0] < radius:
            return best
        radius *= 2
    raise ValueError(f'无法确定最近国家边界：{lat}, {lon}')


def country_attribution(selected, countries, run):
    import geopandas as gpd

    points = gpd.GeoDataFrame(selected, geometry=gpd.points_from_xy(selected.lon, selected.lat), crs='EPSG:4326')
    assigned = gpd.sjoin(points, countries[['iso3', 'ADMIN', 'CONTINENT', 'geometry']], how='left', predicate='intersects')
    assigned = assigned.sort_values(['grid_id', 'iso3']).drop_duplicates('grid_id')
    countries = countries.copy()
    countries['boundary_geometry'] = countries.geometry.boundary
    cache = run / 'inputs/country-boundary-cache.parquet'
    cache_meta = run / 'inputs/country-boundary-cache.json'
    source_hash = digest(run / 'sources/countries-10m.zip')
    records = pd.read_parquet(cache).to_dict('records') if cache.exists() else []
    if records and json.loads(cache_meta.read_text('utf-8'))['boundary_source_sha256'] != source_hash:
        raise ValueError('国家边界缓存与来源版本不一致')
    done = {row['grid_id']: row for row in records}
    for number, row in enumerate(assigned.itertuples(), 1):
        if row.grid_id in done:
            assert (done[row.grid_id]['lat'], done[row.grid_id]['lon']) == (row.lat, row.lon)
            continue
        inside = pd.notna(row.index_right)
        distance, _, index, boundary_lat, boundary_lon = nearest_boundary(row.lat, row.lon, countries, int(row.index_right) if inside else None)
        country = countries.loc[index]
        record = {'grid_id': row.grid_id, 'lat': row.lat, 'lon': row.lon, 'iso3': country.iso3,
                  'country': country.ADMIN, 'continent': country.CONTINENT,
                  'center_country_found': bool(inside), 'country_assignment_method': 'center_in_polygon' if inside else 'nearest_boundary',
                  'nearest_boundary_distance_km': distance, 'country_assignment_distance_km': 0. if inside else distance,
                  'nearest_boundary_lat': boundary_lat, 'nearest_boundary_lon': boundary_lon}
        records.append(record)
        done[row.grid_id] = record
        if number % 50 == 0 or number == len(assigned):
            temporary = cache.with_suffix('.part.parquet')
            pd.DataFrame(records).to_parquet(temporary, index=False, compression='zstd')
            temporary.replace(cache)
            write_json(cache_meta, {'utc': utc_now(), 'boundary_source_sha256': source_hash, 'completed': len(records),
                                   'distance_method': 'WGS84中心等距投影；边界按0.01度细分，最近点用Geod复核；同距按ISO3排序'})
            print('country boundaries', number, len(assigned), flush=True)
    return selected.merge(pd.DataFrame(records).drop(columns=['lat', 'lon']), on='grid_id', validate='one_to_one')


def m49_regions(run):
    source = run / 'sources/un-m49.html'
    tables = pd.read_html(StringIO(source.read_text('utf-8')))
    candidates = [table for table in tables if list(table.columns) == ['Country or Area', 'M49 code', 'ISO-alpha3 code', 'Other groupings']]
    if not candidates:
        raise ValueError('UN M49地理区域表缺失')
    table = candidates[0]
    records = []
    region_name, region_code, continent = None, None, None
    macro = {2: 'Africa', 19: 'Americas', 142: 'Asia', 150: 'Europe', 9: 'Oceania'}
    for row in table.itertuples(index=False, name=None):
        name, code, iso3, _ = row
        if pd.isna(iso3):
            region_name, region_code = name, int(code)
            continent = macro.get(int(code), continent)
        else:
            records.append({'iso3': iso3, 'm49_region': region_name, 'm49_region_code': region_code, 'm49_continent': continent})
    frame = pd.DataFrame(records).set_index('iso3')
    assert frame.index.is_unique
    frame.to_csv(run / 'inputs/un-m49-country-regions.csv')
    write_json(run / 'sources/un-m49-receipt.json', {'url': 'https://unstats.un.org/unsd/methodology/m49/', 'downloaded_utc': utc_now(),
                                                  'sha256': digest(source), 'region_level': '地理表中最细区域，包括South America等中间区域'})
    return frame


def select(config):
    import geopandas as gpd
    import zarr

    own, paths = settings(config)
    run, source = paths['run_root'], paths['global_root']
    folder = source / 'data/processed/buildability-20261025'
    grids = pd.read_parquet(folder / 'grids.parquet')
    grids = grids.loc[grids.screen_status.eq('eligible')].copy()
    assert len(grids) == 60430
    grids['lat'], grids['lon'] = (grids.north + grids.south) / 2, (grids.east + grids.west) / 2
    grids['block_row'] = np.floor((90 - grids.lat) / 2).astype(int)
    grids['block_col'] = np.floor((grids.lon + 180) / 2).astype(int)
    grids['block_lat'], grids['block_lon'] = 89 - 2 * grids.block_row, -179 + 2 * grids.block_col
    grids['block_id'] = ['g2_r%02d_c%03d' % pair for pair in zip(grids.block_row, grids.block_col)]
    grids['center_distance_km'] = spherical_distance(grids.lat, grids.lon, grids.block_lat, grids.block_lon)
    selected = grids.sort_values(['center_distance_km', 'grid_id']).drop_duplicates('block_id').sort_values('grid_id').reset_index(drop=True)
    assert selected.block_id.nunique() == grids.block_id.nunique()
    countries = gpd.read_file('zip://' + str(run / 'sources/countries-10m.zip'))
    countries['iso3'] = countries.ISO_A3_EH
    assert not countries.iso3.isna().any()
    joined = country_attribution(selected, countries, run)
    joined['natural_earth_subregion'] = joined.country.map(countries.set_index('ADMIN').SUBREGION)
    m49 = m49_regions(run)
    joined['m49_region'] = joined.iso3.map(m49.m49_region)
    joined['m49_region_code'] = joined.iso3.map(m49.m49_region_code)
    joined['m49_continent'] = joined.iso3.map(m49.m49_continent)
    joined['natural_earth_continent'] = joined.continent
    continent = joined.m49_continent.copy()
    continent.loc[continent.eq('Americas')] = np.where(joined.loc[continent.eq('Americas'), 'm49_region'].eq('South America'), 'South America', 'North America')
    joined['continent'] = continent.where(continent.notna(), joined.continent)
    ember_path = run / 'sources/ember-yearly-complete.csv'
    ember = pd.read_csv(ember_path, low_memory=False)
    rows = ember.loc[ember.Year.eq(2024) & ember['Electricity source'].eq('Total generation')].copy()
    metric = 'Emissions intensity (gCO2e/kWh)'
    national = rows.loc[rows['Area type'].eq('Country or economy') & rows['ISO 3 code'].notna()].set_index('ISO 3 code')
    assert national.index.is_unique
    national['m49_region'] = national.index.map(m49.m49_region)
    joined['carbon_gco2_kwh'] = joined.iso3.map(national[metric])
    joined['carbon_source'] = np.where(joined.carbon_gco2_kwh.notna(), 'country', 'regional_fallback')
    joined['ember_region'] = joined.iso3.map(national['Ember region'])
    joined['carbon_reference_area'] = joined.iso3.map(national.Area)
    joined['carbon_matched_iso3'] = joined.iso3.where(joined.carbon_gco2_kwh.notna())
    joined['carbon_match_method'] = np.where(joined.carbon_gco2_kwh.notna(), 'country_iso3', 'regional_fallback')
    for index in joined.index[joined.carbon_gco2_kwh.isna()]:
        region = joined.loc[index, 'm49_region']
        if pd.isna(region):
            aliases = {'Somaliland': 'SOM', 'Northern Cyprus': 'CYP', 'Kosovo': 'SRB'}
            alias = aliases.get(joined.loc[index, 'country'])
            if alias is None:
                label = str(joined.loc[index, 'natural_earth_subregion']).casefold().replace('-', '').replace(' ', '')
                matches = [name for name in m49.m49_region.unique() if str(name).casefold().replace('-', '').replace(' ', '') == label]
                if len(matches) != 1:
                    raise ValueError('UN M49区域无法确定：' + str(joined.loc[index, 'country']))
                region = matches[0]
                region_code = m49.loc[m49.m49_region.eq(region), 'm49_region_code'].iloc[0]
                assignment = 'natural_earth_subregion_matched_to_un_m49'
            else:
                region = m49.loc[alias, 'm49_region']
                region_code = m49.loc[alias, 'm49_region_code']
                assignment = 'geographical_area_alias'
            joined.loc[index, 'm49_region'] = region
            joined.loc[index, 'm49_region_code'] = region_code
            joined.loc[index, 'm49_region_assignment'] = assignment
        named = national.loc[national.Area.eq(joined.loc[index, 'country']) & national[metric].notna()]
        if len(named) > 1:
            raise ValueError('Ember国家名称匹配不唯一：' + str(joined.loc[index, 'country']))
        if len(named) == 1:
            country = named.iloc[0]
            joined.loc[index, 'carbon_gco2_kwh'] = country[metric]
            joined.loc[index, 'carbon_source'] = 'country'
            joined.loc[index, 'carbon_reference_area'] = country.Area
            joined.loc[index, 'ember_region'] = country['Ember region']
            joined.loc[index, 'carbon_matched_iso3'] = named.index[0]
            joined.loc[index, 'carbon_match_method'] = 'country_name_unique'
            continue
        # 用同一UN M49区域内有数据国家的2024发电量加权，保留分子来源及覆盖数量。
        pool = national.loc[national.m49_region.eq(region) & national[metric].notna() & national['Generation (TWh)'].gt(0)]
        if pool.empty:
            raise ValueError('UN M49区域无可用碳强度：' + str(region))
        joined.loc[index, 'carbon_gco2_kwh'] = np.average(pool[metric], weights=pool['Generation (TWh)'])
        joined.loc[index, 'carbon_reference_area'] = region
        joined.loc[index, 'carbon_regional_country_count'] = len(pool)
        joined.loc[index, 'carbon_regional_generation_twh'] = pool['Generation (TWh)'].sum()
    joined['carbon_t_mwh'] = joined.carbon_gco2_kwh / 1000
    joined['carbon_year'] = 2024
    assert joined.iso3.notna().all() and joined.carbon_gco2_kwh.notna().all()
    joined.to_parquet(run / 'inputs/selected-grids.parquet', index=False, compression='zstd')
    mapping = pd.read_parquet(folder / 'mapping.parquet').merge(joined[['grid_id']], on='grid_id', validate='many_to_one')
    assert mapping.groupby('grid_id').size().eq(9).all()
    points = pd.read_parquet(folder / 'points.parquet')
    mapping = mapping.merge(points, on=['weather_point', 'native_flat_index'], validate='many_to_one')
    land = zarr.open_group(str(source / 'data/processed/era5-hourly-land-2023-2025.zarr'), mode='r')
    land_index = pd.Series(np.arange(len(land['native_flat_index'])), index=land['native_flat_index'][:])
    mapping['land_point_index'] = mapping.native_flat_index.map(land_index)
    mapping['is_land'] = mapping.land_point_index.notna()
    mapping.to_parquet(run / 'inputs/grid-points.parquet', index=False, compression='zstd')
    cities = json.loads((Path(config).parent / 'eight_city_locations.json').read_text('utf-8'))
    city_rows = []
    for city in cities:
        block_row = int(np.floor((90 - city['latitude']) / 2))
        block_col = int(np.floor((city['longitude'] + 180) / 2))
        representative = joined.loc[joined.block_row.eq(block_row) & joined.block_col.eq(block_col)]
        assert len(representative) == 1, city['city_id']
        lat = round(city.get('weather_latitude', city['latitude']) * 4) / 4
        lon = round(city.get('weather_longitude', city['longitude']) * 4) / 4
        native = round((90 - lat) * 4) * 1440 + round((lon % 360) * 4)
        assert native in land_index.index
        city_rows.append({'city': city['city_id'], 'grid_id': representative.grid_id.iloc[0], 'lat': lat, 'lon': lon,
                          'native_flat_index': native, 'land_point_index': int(land_index[native])})
    pd.DataFrame(city_rows).to_csv(run / 'inputs/city-grid-points.csv', index=False)
    city_ids = [row['grid_id'] for row in city_rows]
    random_ids = joined.loc[~joined.grid_id.isin(city_ids)].sample(n=12, random_state=own['sampling']['seed']).grid_id.tolist()
    write_json(run / 'inputs/pilot-grids.json', {'city_grids': city_ids, 'random_grids': random_ids, 'seed': own['sampling']['seed']})
    write_json(run / 'inputs/selection-manifest.json', {'utc': utc_now(), 'eligible_count': len(grids), 'selected_count': len(joined),
        'missing_center_country_count': int((~joined.center_country_found).sum()), 'land_point_count_distribution': mapping.groupby('grid_id').is_land.sum().value_counts().sort_index().to_dict(),
        'carbon_source_counts': joined.carbon_source.value_counts().to_dict(),
        'sources': {str(path): digest(path) for path in [folder / 'grids.parquet', folder / 'mapping.parquet', folder / 'points.parquet', ember_path, run / 'sources/countries-10m.zip']}})
    print('selected', len(joined), 'missing_center_country', int((~joined.center_country_found).sum()), flush=True)


def templates(config):
    own, paths = settings(config)
    run, source = paths['run_root'], paths['global_root']
    index = pd.read_csv(paths['city_input_root'] / 'input-index.csv')
    receipts = []
    (run / 'inputs/templates').mkdir(exist_ok=True)
    for scenario in SCENARIOS:
        expected = pd.read_parquet(paths['city_input_root'] / f'frankfurt-{scenario}.parquet')
        loadpath = index.loc[index.city.eq('frankfurt') & index.scenario.eq(scenario), 'load_path'].iloc[0]
        raw = pd.read_parquet(source / loadpath, columns=['source_hour_index', 'weather_time_utc', 'it_mw', 'installed_reference_it_mw'])
        assert np.array_equal(raw.source_hour_index, expected.source_hour_index)
        np.testing.assert_allclose(raw.it_mw, expected.it_mw, atol=1e-12, rtol=0)
        clock = pd.DatetimeIndex(pd.to_datetime(raw.weather_time_utc, utc=True))
        position = WEATHER_CLOCK.get_indexer(clock)
        assert (position >= 0).all()
        result = expected[['time_utc', 'source_hour_index', *LAYERS, 'it_mw', 'cap_it_mw']].copy()
        result['weather_hour_index'] = position
        result['weather_time_utc'] = clock
        for city in CITIES:
            other = pd.read_parquet(paths['city_input_root'] / f'{city}-{scenario}.parquet')
            np.testing.assert_allclose(other[[*LAYERS, 'it_mw', 'cap_it_mw', 'source_hour_index']], expected[[*LAYERS, 'it_mw', 'cap_it_mw', 'source_hour_index']], rtol=0, atol=1e-12)
        target = run / 'inputs/templates' / f'{scenario}.parquet'
        result.to_parquet(target, index=False, compression='zstd')
        receipts.append({'scenario': scenario, 'hours': len(result), 'weather_time_sha256': sha256(clock.asi8.tobytes()).hexdigest(),
                         'template_sha256': digest(target), 'load_path': str(source / loadpath), 'load_sha256': digest(source / loadpath)})
    pd.DataFrame(receipts).to_csv(run / 'inputs/template-index.csv', index=False)
    print('templates completed', flush=True)


def extraction_points(run, scope):
    mapping = pd.read_parquet(run / 'inputs/grid-points.parquet')
    if scope == 'pilot':
        pilot = json.loads((run / 'inputs/pilot-grids.json').read_text('utf-8'))
        mapping = mapping.loc[mapping.grid_id.isin(pilot['city_grids'] + pilot['random_grids'])]
    points = mapping.loc[mapping.is_land, ['native_flat_index', 'latitude', 'longitude_0_360']].drop_duplicates('native_flat_index')
    cities = pd.read_csv(run / 'inputs/city-grid-points.csv')
    extra = cities[['native_flat_index', 'lat', 'lon']].rename(columns={'lat': 'latitude', 'lon': 'longitude_0_360'})
    extra['longitude_0_360'] %= 360
    points = pd.concat([points, extra], ignore_index=True).drop_duplicates('native_flat_index').sort_values('native_flat_index').reset_index(drop=True)
    points['source_lat_index'] = 720 - points.native_flat_index // 1440
    points['source_lon_index'] = (points.native_flat_index % 1440 + 720) % 1440
    points['block_label'] = ['lat-%03d-lon-%03d' % pair for pair in zip(points.source_lat_index // 4, points.source_lon_index // 4)]
    return points


def extract_block(task):
    import zarr

    source, target, label, flats = task
    receipt = target / (label + '.json')
    destination = target / (label + '.npz')
    if receipt.exists():
        info = json.loads(receipt.read_text('utf-8'))
        if info['sha256'] != digest(destination):
            raise ValueError('气象分块SHA不一致：' + label)
        return info
    path = source / 'data/processed/era5-eligible-2023-2025' / (label + '.parquet')
    data = pd.read_parquet(path, filters=[('time', '>=', pd.Timestamp('2025-01-01')), ('time', '<', pd.Timestamp('2026-01-01'))])
    times = WEATHER_CLOCK.tz_localize(None)
    values = np.empty((len(times), len(flats), len(VARIABLES)), dtype='float32')
    for j, flat in enumerate(flats):
        single = data.loc[data.native_flat_index.eq(flat)].set_index('time').reindex(times)
        if single[list(VARIABLES.values())].isna().any().any():
            # 城市对照点可能未进入eligible列表；同一冻结源的该4×4块仍有真实原生值。
            raw = zarr.open_consolidated(str(source / 'data/raw/era5-eligible-ecmwf-geo'), mode='r')
            wanted = WEATHER_CLOCK.as_unit('s').asi8
            clock = raw['time'][:]
            positions = np.searchsorted(clock, wanted)
            np.testing.assert_array_equal(clock[positions], wanted)
            ilat, ilon = int(720 - flat // 1440), int((flat % 1440 + 720) % 1440)
            for k, short in enumerate(VARIABLES):
                value = raw[short].oindex[positions, ilat, ilon]
                values[:, j, k] = value / (3600 if short in ('ssrd', 'fdir') else 1)
        else:
            values[:, j] = single[list(VARIABLES.values())].to_numpy(dtype='float32')
    if not np.isfinite(values).all():
        raise ValueError('2025气象分块存在非有限数值：' + label)
    temporary = target / (label + '.part.npz')
    np.savez_compressed(temporary, values=values, native_flat_index=np.asarray(flats, dtype='int32'))
    temporary.replace(destination)
    info = {'label': label, 'points': len(flats), 'hours': len(times), 'source': str(path),
            'source_sha256': digest(path), 'sha256': digest(destination), 'utc': utc_now()}
    write_json(receipt, info)
    return info


def extract(config, scope, workers):
    import zarr
    from numcodecs import Blosc

    own, paths = settings(config)
    run, source = paths['run_root'], paths['global_root']
    points = extraction_points(run, scope)
    target = run / 'inputs/raw-blocks'
    target.mkdir(exist_ok=True)
    tasks = [(source, target, label, group.native_flat_index.tolist()) for label, group in points.groupby('block_label', sort=True)]
    # 每个块保存全部选中点；全量扩展时只重建点集变化的块。
    for _, _, label, flats in tasks:
        receipt = target / (label + '.json')
        if receipt.exists():
            with np.load(target / (label + '.npz')) as archive:
                if not np.array_equal(archive['native_flat_index'], flats):
                    receipt.unlink()
    started = time.monotonic()
    with ThreadPoolExecutor(workers) as pool:
        for number, info in enumerate(pool.map(extract_block, tasks), 1):
            if number % 20 == 0 or number == len(tasks):
                write_json(run / 'inputs/extraction-progress.json', {'utc': utc_now(), 'scope': scope, 'done': number, 'total': len(tasks), 'elapsed_seconds': time.monotonic() - started})
                print('extract', number, len(tasks), flush=True)
    points.to_parquet(run / f'inputs/extracted-points-{scope}.parquet', index=False)
    store_path = run / f'inputs/weather-{scope}.zarr'
    store = zarr.open_group(str(store_path), mode='a')
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)
    shape = (8760, len(points), len(VARIABLES))
    array = store.require_dataset('values', shape=shape, chunks=(8760, 16, 7), dtype='f4', compressor=compressor)
    store.require_dataset('native_flat_index', shape=(len(points),), dtype='i4')[:] = points.native_flat_index.to_numpy()
    store.attrs.update({'variables': list(VARIABLES), 'units': ['K', 'K', 'Pa', 'm/s', 'm/s', 'W/m2', 'W/m2'],
                        'time_start_utc': str(WEATHER_CLOCK[0]), 'hours': 8760, 'source': '完整eligible parquet；非eligible城市点读取同一冻结ECMWF块', 'source_zarr_2025_incomplete': True})
    @lru_cache(maxsize=64)
    def raw_block(label):
        with np.load(target / (label + '.npz')) as archive:
            return archive['native_flat_index'], archive['values']

    # 按目标Zarr块顺序一次写完，避免对同一压缩块多次读写。
    for start in range(0, len(points), 16):
        group = points.iloc[start:start + 16]
        batch = np.empty((8760, len(group), 7), dtype='float32')
        for j, row in enumerate(group.itertuples()):
            flats, values = raw_block(row.block_label)
            location = np.flatnonzero(flats == row.native_flat_index)
            assert len(location) == 1
            batch[:, j, :] = values[:, int(location[0]), :]
        array[:, start:start + len(group), :] = batch
        if start % 1600 == 0 or start + len(group) == len(points):
            write_json(run / 'inputs/extraction-progress.json', {'utc': utc_now(), 'scope': scope, 'stage': 'compact_zarr',
                'done': start + len(group), 'total': len(points), 'elapsed_seconds': time.monotonic() - started})
            print('compact zarr points', start + len(group), len(points), flush=True)
    write_json(run / f'inputs/extraction-{scope}.json', {'utc': utc_now(), 'scope': scope, 'points': len(points), 'blocks': len(tasks), 'seconds': time.monotonic() - started, 'zarr': str(store_path)})


def fetch_wind10(task):
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    run, global_root, key = task
    target = run / 'inputs/wind10-source' / key
    receipt = target.with_name(target.name + '.receipt.json')
    if target.exists() and receipt.exists():
        info = json.loads(receipt.read_text('utf-8'))
        if digest(target) != info['sha256']:
            raise ValueError('10m风速对象SHA不一致：' + key)
        return info
    target.parent.mkdir(parents=True, exist_ok=True)
    cached = Path('/data3/dataset/ljt/eight-city-weather-20261029/arco-raw') / key
    if target.exists():
        method = 'resumed_atomic_download_cache'
    elif cached.exists():
        shutil.copyfile(cached, target)
        method = 'existing_official_source_cache'
    else:
        # 使用已有授权凭据读取同一官方ERA5源；凭据不进入输出与日志。
        if not hasattr(HTTP_CONTEXT, 'session'):
            credential = json.loads((global_root / '.codex-temp/buildability-20261025/cds-credential.json').read_text('utf-8'))['key']
            session = requests.Session()
            session.headers.update({'Authorization': 'Bearer ' + credential, 'User-Agent': 'dc-flexibility-research/1.0'})
            retry = Retry(total=4, connect=4, read=4, status=4, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=('GET',))
            session.mount('https://', HTTPAdapter(max_retries=retry))
            HTTP_CONTEXT.session = session
        for attempt in range(3):
            try:
                response = HTTP_CONTEXT.session.get(WIND10_URL + '/' + key, timeout=(20, 180))
                response.raise_for_status()
                payload = response.content
                break
            except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as error:
                if attempt == 2:
                    raise RuntimeError('ERA5对象传输未完成：' + key) from error
                print(json.dumps({'event': 'retry_object_transfer', 'key': key, 'attempt': attempt + 1, 'error_type': type(error).__name__}), flush=True)
                time.sleep(2 ** attempt)
        temporary = target.with_name(target.name + '.download')
        temporary.write_bytes(payload)
        temporary.replace(target)
        method = 'official_ecmwf_arco_download'
    info = {'key': key, 'url': WIND10_URL + '/' + key, 'sha256': digest(target), 'bytes': target.stat().st_size, 'method': method, 'utc': utc_now()}
    write_json(receipt, info)
    return info


def wind10(config, scope, workers):
    import zarr
    from numcodecs import Blosc

    own, paths = settings(config)
    run, source = paths['run_root'], paths['global_root']
    points = extraction_points(run, scope)
    original = source / 'data/raw/era5-eligible-ecmwf-geo'
    raw = zarr.open_consolidated(str(original), mode='r')
    clock = raw['time'][:]
    wanted = WEATHER_CLOCK.as_unit('s').asi8
    positions = np.searchsorted(clock, wanted)
    np.testing.assert_array_equal(clock[positions], wanted)
    target = run / 'inputs/wind10-source'
    target.mkdir(exist_ok=True)
    metadata_path = Path('/data3/dataset/ljt/eight-city-weather-20261029/arco-raw/.zmetadata')
    if not (target / '.zmetadata').exists():
        shutil.copyfile(metadata_path, target / '.zmetadata')
    if digest(metadata_path) != digest(target / '.zmetadata'):
        raise ValueError('10m风速源元数据版本发生变化')
    specification = json.loads(metadata_path.read_text('utf-8'))['metadata']['u10/.zarray']
    assert specification['chunks'] == [67584, 4, 4]
    keys = sorted({f'{variable}/{int(tchunk)}.{int(row.source_lat_index // 4)}.{int(row.source_lon_index // 4)}'
                   for variable in ('u10', 'v10') for tchunk in np.unique(positions // 67584) for row in points.itertuples()})
    started = time.monotonic()
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(fetch_wind10, (run, source, key)) for key in keys]
        for number, future in enumerate(as_completed(futures), 1):
            future.result()
            if number % 10 == 0 or number == len(keys):
                write_json(run / 'inputs/wind10-progress.json', {'utc': utc_now(), 'scope': scope, 'done': number, 'total': len(keys), 'elapsed_seconds': time.monotonic() - started})
                print('wind10', number, len(keys), flush=True)
    supplement = zarr.open_consolidated(str(target), mode='r')
    out = zarr.open_group(str(run / f'inputs/weather-{scope}.zarr'), mode='a')
    array = out.require_dataset('wind10', shape=(8760, len(points), 2), chunks=(8760, 16, 2), dtype='f4', compressor=Blosc(cname='zstd', clevel=3))
    assert np.diff(positions).min() == np.diff(positions).max() == 1
    time_slice = slice(int(positions[0]), int(positions[-1]) + 1)

    @lru_cache(maxsize=512)
    def wind_block(lat_block, lon_block):
        return np.stack([supplement[variable][time_slice, slice(lat_block * 4, lat_block * 4 + 4),
                                             slice(lon_block * 4, lon_block * 4 + 4)] for variable in ('u10', 'v10')], axis=-1)

    for start in range(0, len(points), 16):
        group = points.iloc[start:start + 16]
        values = np.empty((8760, len(group), 2), dtype='float32')
        for j, row in enumerate(group.itertuples()):
            block = wind_block(int(row.source_lat_index // 4), int(row.source_lon_index // 4))
            values[:, j, :] = block[:, int(row.source_lat_index % 4), int(row.source_lon_index % 4), :]
        if not np.isfinite(values).all():
            raise ValueError('10m风速数据存在缺失')
        array[:, start:start + len(group), :] = values
        if start % 1600 == 0 or start + len(group) == len(points):
            write_json(run / 'inputs/wind10-progress.json', {'utc': utc_now(), 'scope': scope, 'stage': 'compact_zarr',
                'done': start + len(group), 'total': len(points), 'downloaded_objects': len(keys), 'elapsed_seconds': time.monotonic() - started})
            print('compact wind10 points', start + len(group), len(points), flush=True)
    write_json(run / f'inputs/wind10-{scope}.json', {'utc': utc_now(), 'points': len(points), 'objects': len(keys), 'seconds': time.monotonic() - started,
                                                  'metadata_sha256': digest(metadata_path), 'source_url': WIND10_URL})


def weather_frame(values, wind10_values):
    result = pd.DataFrame(values, columns=list(VARIABLES))
    result['time_utc'] = WEATHER_CLOCK
    result['temperature_c'] = result.t2m.to_numpy(dtype=float) - 273.15
    result['dew_point_c'] = result.d2m.to_numpy(dtype=float) - 273.15
    result['pressure_pa'] = result.sp.to_numpy(dtype=float)
    result['ssrd'] = result.ssrd.to_numpy(dtype=float) * 3600
    result['fdir'] = result.fdir.to_numpy(dtype=float) * 3600
    result['u10'], result['v10'] = wind10_values[:, 0], wind10_values[:, 1]
    return result


def facility_frame(template, weather):
    position = template.weather_hour_index.to_numpy(int)
    selected = weather.iloc[position][['temperature_c', 'dew_point_c', 'pressure_pa']].reset_index(drop=True)
    raw_temperature = selected.temperature_c.to_numpy(float).copy()
    raw_dew = selected.dew_point_c.to_numpy(float).copy()
    # 用户口径：输入端夹取温度，保留原始气象；风光计算继续使用原始温度。
    selected['temperature_c'] = np.clip(raw_temperature, -40., 50.)
    selected['dew_point_c'] = np.minimum(raw_dew, selected.temperature_c.to_numpy())
    # 冷却计算逐小时无记忆；临时时标只满足接口的唯一递增检查，天气映射仍保留在模板中。
    selected['time_utc'] = pd.date_range('2001-01-01', periods=len(selected), freq='h', tz='UTC')
    cooling = CoolingModel(cooling_type='air', parameter_set='central').calculate(selected, template.it_mw.to_numpy(float), float(template.cap_it_mw.iloc[0]), full_year=False)
    cooling.attrs.update(raw_min_temperature_c=float(raw_temperature.min()), raw_max_temperature_c=float(raw_temperature.max()),
                         low_temperature_clipped_hours=int((raw_temperature < -40).sum()),
                         high_temperature_clipped_hours=int((raw_temperature > 50).sum()),
                         temperature_clipped_hours=int(((raw_temperature < -40) | (raw_temperature > 50)).sum()),
                         dewpoint_clipped_hours=int((raw_dew > selected.temperature_c.to_numpy()).sum()))
    return cooling


def compute(config, scope, workers=4, grid_directory='grids'):
    import zarr
    from eight_city_physics import resource_capacity_factors

    own, paths = settings(config)
    run = paths['run_root']
    points = pd.read_parquet(run / f'inputs/extracted-points-{scope}.parquet')
    store = zarr.open_group(str(run / f'inputs/weather-{scope}.zarr'), mode='r')
    np.testing.assert_array_equal(store['native_flat_index'][:], points.native_flat_index)
    point_index = pd.Series(points.index.to_numpy(), index=points.native_flat_index)
    curve = pd.read_csv(paths['city_weather_root'] / 'benchmarks/IEA_Reference_3.4MW_130.csv')
    mapping = pd.read_parquet(run / 'inputs/grid-points.parquet')
    if scope == 'pilot':
        pilot = json.loads((run / 'inputs/pilot-grids.json').read_text('utf-8'))
        mapping = mapping.loc[mapping.grid_id.isin(pilot['city_grids'] + pilot['random_grids'])]
    if Path(grid_directory).name != grid_directory:
        raise ValueError('网格输入子目录必须是单个目录名')
    folder = run / 'inputs' / grid_directory
    folder.mkdir(exist_ok=True)
    suffix = '' if grid_directory == 'grids' else '-' + grid_directory
    index_path = run / f'inputs/grid-input-index-{scope}{suffix}.csv'
    version_path = run / f'inputs/cf-build-{scope}{suffix}.json'
    versions = {'physics_sha256': digest(Path(config).parent / 'eight_city_physics.py'),
                'curve_sha256': digest(paths['city_weather_root'] / 'benchmarks/IEA_Reference_3.4MW_130.csv'),
                'points_sha256': digest(run / f'inputs/extracted-points-{scope}.parquet'),
                'extraction_sha256': digest(run / f'inputs/extraction-{scope}.json'),
                'wind10_sha256': digest(run / f'inputs/wind10-{scope}.json'),
                'weather_function_sha256': sha256(inspect.getsource(weather_frame).encode()).hexdigest()}
    if version_path.exists() and json.loads(version_path.read_text('utf-8'))['versions'] != versions:
        raise ValueError('容量因子续跑来源版本发生变化')
    write_json(version_path, {'versions': versions, 'scope': scope, 'grid_directory': grid_directory})
    reports = pd.read_csv(index_path).to_dict('records') if index_path.exists() else []
    if scope == 'all' and grid_directory == 'grids' and not reports:
        reports = pd.read_csv(run / 'inputs/grid-input-index-pilot.csv').to_dict('records')
    done = {row['grid_id']: row for row in reports}
    tasks = []
    for grid, group in mapping.groupby('grid_id', sort=True):
        if grid in done:
            previous = done[grid]
            if previous['input_status'] == 'ready':
                target = folder / (grid + '.npz')
                if digest(target) != previous['sha256']:
                    raise ValueError('既有容量因子输入SHA不一致：' + grid)
                with np.load(target) as archive:
                    np.testing.assert_array_equal(archive['native_flat_index'], group.loc[group.is_land].native_flat_index)
            continue
        tasks.append((grid, group))

    def build_grid(item):
        grid, group = item
        chosen = group.loc[group.is_land]
        target = folder / (grid + '.npz')
        if chosen.empty:
            return {'grid_id': grid, 'land_points': 0, 'input_status': 'no_land_points'}
        positions = point_index.loc[chosen.native_flat_index].to_numpy(int)
        values = store['values'].oindex[:, positions, :].astype(float)
        winds = store['wind10'].oindex[:, positions, :].astype(float)
        factors = []
        for j, point in enumerate(chosen.itertuples()):
            weather = weather_frame(values[:, j], winds[:, j])
            cf, _ = resource_capacity_factors(weather, {'latitude': point.latitude, 'longitude': point.longitude_0_360}, curve)
            factors.append(cf[['pv_capacity_factor', 'wind_capacity_factor']].to_numpy(float))
        cf = np.mean(factors, axis=0)
        mean = values.mean(axis=1)
        temperature = mean[:, 0] - 273.15
        dew = mean[:, 1] - 273.15
        pressure = mean[:, 2]
        if not np.isfinite(np.c_[cf, temperature, dew, pressure]).all():
            raise ValueError('网格输入含非有限值：' + grid)
        temporary = target.with_suffix('.part.npz')
        np.savez_compressed(temporary, cf_pv=cf[:, 0], cf_wind=cf[:, 1], temperature_c=temperature, dew_point_c=dew,
                            pressure_pa=pressure, land_points=np.asarray(len(chosen)), native_flat_index=chosen.native_flat_index.to_numpy())
        temporary.replace(target)
        return {'grid_id': grid, 'land_points': len(chosen), 'input_status': 'ready', 'sha256': digest(target),
                'minimum_temperature_c': float(temperature.min()), 'maximum_temperature_c': float(temperature.max()),
                'temperature_outside_model_hours': int(((temperature < -40) | (temperature > 50)).sum()),
                'cf_pv_mean': float(cf[:, 0].mean()), 'cf_wind_mean': float(cf[:, 1].mean())}

    started = time.monotonic()
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(build_grid, item) for item in tasks]
        for number, future in enumerate(as_completed(futures), 1):
            reports.append(future.result())
            if number % 20 == 0 or number == len(tasks):
                temporary = index_path.with_suffix('.part.csv')
                pd.DataFrame(reports).sort_values('grid_id').to_csv(temporary, index=False)
                temporary.replace(index_path)
                write_json(run / f'inputs/compute-progress-{scope}{suffix}.json', {'utc': utc_now(), 'scope': scope,
                    'done': len(reports), 'total': mapping.grid_id.nunique(), 'workers': workers, 'elapsed_seconds': time.monotonic() - started})
                print('compute grids', len(reports), mapping.grid_id.nunique(), flush=True)
    if not tasks:
        print('compute grids already complete', len(reports), flush=True)


def city_validation(config, scope):
    import zarr
    from eight_city_physics import resource_capacity_factors

    own, paths = settings(config)
    run, source = paths['run_root'], paths['global_root']
    store = zarr.open_group(str(run / f'inputs/weather-{scope}.zarr'), mode='r')
    point_index = pd.Series(np.arange(store['native_flat_index'].shape[0]), index=store['native_flat_index'][:])
    cities = pd.read_csv(run / 'inputs/city-grid-points.csv')
    locations = {city['city_id']: city for city in json.loads((Path(config).parent / 'eight_city_locations.json').read_text('utf-8'))}
    inputs = pd.read_csv(paths['city_input_root'] / 'input-index.csv')
    curve = pd.read_csv(paths['city_weather_root'] / 'benchmarks/IEA_Reference_3.4MW_130.csv')
    rows, weather_rows = [], []
    nearest_store = zarr.open_group(str(run / 'inputs/weather-nearest-cities.zarr'), mode='a')
    nearest_values = nearest_store.require_dataset('values', shape=(8760, len(cities), 7), chunks=(8760, 1, 7), dtype='f4')
    nearest_wind = nearest_store.require_dataset('wind10', shape=(8760, len(cities), 2), chunks=(8760, 1, 2), dtype='f4')
    nearest_store.attrs.update({'cities': cities.city.tolist(), 'variables': list(VARIABLES), 'time_start_utc': str(WEATHER_CLOCK[0]), 'source': '冻结ECMWF原生点；保留独立zarr供地理最近点对照'})
    for city in cities.itertuples():
        position = int(point_index[city.native_flat_index])
        data, winds = store['values'][:, position, :].astype(float), store['wind10'][:, position, :].astype(float)
        weather = weather_frame(data, winds)
        factors, _ = resource_capacity_factors(weather, locations[city.city], curve)
        original = pd.read_parquet(paths['city_weather_root'] / (city.city + '.parquet'))
        original = original.loc[pd.to_datetime(original.time_utc, utc=True).dt.year.eq(2025)].reset_index(drop=True)
        # 先逐字段追踪天气来源差异，再隔离物理算法的数值差异。
        for variable in (*VARIABLES, 'u10', 'v10'):
            expected = original[variable].to_numpy(float)
            if variable in ('ssrd', 'fdir'):
                expected /= 3600
            actual = data[:, list(VARIABLES).index(variable)] if variable in VARIABLES else winds[:, ('u10', 'v10').index(variable)]
            gap = np.abs(actual - expected)
            weather_rows.append({'city': city.city, 'variable': variable, 'max_abs_difference': float(gap.max()), 'mean_abs_difference': float(gap.mean())})
        for scenario in SCENARIOS:
            template = pd.read_parquet(run / 'inputs/templates' / (scenario + '.parquet'))
            baseline = pd.read_parquet(paths['city_input_root'] / (city.city + '-' + scenario + '.parquet'))
            np.testing.assert_array_equal(template.source_hour_index, baseline.source_hour_index)
            cooling = facility_frame(template, weather)
            actuals = {'cf_pv': factors.pv_capacity_factor.to_numpy()[template.weather_hour_index],
                       'cf_wind': factors.wind_capacity_factor.to_numpy()[template.weather_hour_index],
                       'pue': cooling.pue.to_numpy(), 'facility_mw': cooling.facility_mw.to_numpy()}
            for variable, actual in actuals.items():
                gap = np.abs(actual - baseline[variable].to_numpy(float))
                rows.append({'city': city.city, 'scenario': scenario, 'variable': variable, 'hours': len(template),
                             'coordinate_kind': 'baseline_weather_point', 'weather_lat': city.lat, 'weather_lon': city.lon,
                             'max_abs_difference': float(gap.max()), 'mean_abs_difference': float(gap.mean()),
                             'baseline_mean': float(baseline[variable].mean()), 'recomputed_mean': float(np.mean(actual))})
        location = locations[city.city]
        nearest_lat, nearest_lon = round(location['latitude'] * 4) / 4, round(location['longitude'] * 4) / 4
        nearest_data, nearest_winds = data, winds
        if (nearest_lat, nearest_lon) != (city.lat, city.lon):
            raw = zarr.open_consolidated(str(source / 'data/raw/era5-eligible-ecmwf-geo'), mode='r')
            raw_clock = raw['time'][:]
            positions = np.searchsorted(raw_clock, WEATHER_CLOCK.as_unit('s').asi8)
            np.testing.assert_array_equal(raw_clock[positions], WEATHER_CLOCK.as_unit('s').asi8)
            ilat, ilon = round((nearest_lat + 90) * 4), round((nearest_lon + 180) * 4) % 1440
            nearest_data = np.column_stack([raw[short].oindex[positions, ilat, ilon] / (3600 if short in ('ssrd', 'fdir') else 1) for short in VARIABLES])
            supplement = zarr.open_consolidated(str(run / 'inputs/wind10-source'), mode='r')
            nearest_winds = np.column_stack([supplement[variable].oindex[positions, ilat, ilon] for variable in ('u10', 'v10')])
        nearest_values[:, city.Index, :], nearest_wind[:, city.Index, :] = nearest_data, nearest_winds
        nearest_weather = weather_frame(nearest_values[:, city.Index, :].astype(float), nearest_wind[:, city.Index, :].astype(float))
        nearest_factors, _ = resource_capacity_factors(nearest_weather, {'latitude': nearest_lat, 'longitude': nearest_lon}, curve)
        for scenario in SCENARIOS:
            template = pd.read_parquet(run / 'inputs/templates' / (scenario + '.parquet'))
            baseline = pd.read_parquet(paths['city_input_root'] / (city.city + '-' + scenario + '.parquet'))
            cooling = facility_frame(template, nearest_weather)
            for variable, actual in {'cf_pv': nearest_factors.pv_capacity_factor.to_numpy()[template.weather_hour_index],
                                     'cf_wind': nearest_factors.wind_capacity_factor.to_numpy()[template.weather_hour_index],
                                     'pue': cooling.pue.to_numpy(), 'facility_mw': cooling.facility_mw.to_numpy()}.items():
                gap = np.abs(actual - baseline[variable].to_numpy(float))
                rows.append({'city': city.city, 'scenario': scenario, 'variable': variable, 'hours': len(template),
                             'coordinate_kind': 'geographic_nearest', 'weather_lat': nearest_lat, 'weather_lon': nearest_lon,
                             'max_abs_difference': float(gap.max()), 'mean_abs_difference': float(gap.mean()),
                             'baseline_mean': float(baseline[variable].mean()), 'recomputed_mean': float(np.mean(actual))})
    pd.DataFrame(rows).to_csv(run / 'validation/city-hourly-differences.csv', index=False)
    pd.DataFrame(weather_rows).to_csv(run / 'validation/city-weather-differences.csv', index=False)
    statistics = []
    for (kind, variable), current in pd.DataFrame(rows).groupby(['coordinate_kind', 'variable']):
        statistics.append({'coordinate_kind': kind, 'variable': variable,
                           'max_abs_difference': float(current.max_abs_difference.max()),
                           'mean_abs_difference': float(np.average(current.mean_abs_difference, weights=current.hours)),
                           'hourly_observations': int(current.hours.sum())})
    write_json(run / 'validation/city-validation-summary.json', {'utc': utc_now(), 'cases': len(CITIES) * len(SCENARIOS),
        'variables': statistics, 'mean_definition': '各城市情境MAE按原生小时数加权，即合并所有小时的MAE',
        'source_zarr_2025_missing': True, 'reconstructed_zarr': str(run / f'inputs/weather-{scope}.zarr'),
        'automatic_gate_passed': bool(pd.DataFrame(rows).loc[lambda f: f.variable.eq('pue') & f.coordinate_kind.eq('baseline_weather_point'), 'max_abs_difference'].max() < 1e-5),
        'gate_note': 'PUE先检验1e-5；风光与天气差异须逐字段解释并由本次运行记录验收。'})
    print('city validation', json.loads((run / 'validation/city-validation-summary.json').read_text('utf-8')), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['select', 'templates', 'extract', 'wind10', 'compute', 'validate-cities'])
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('global-flex-econ-20261103.json'))
    parser.add_argument('--scope', choices=['pilot', 'all'], default='pilot')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--grid-directory', default='grids')
    args = parser.parse_args()
    if args.stage == 'extract':
        extract(args.config, args.scope, args.workers)
    elif args.stage == 'wind10':
        wind10(args.config, args.scope, args.workers)
    elif args.stage == 'compute':
        compute(args.config, args.scope, args.workers, args.grid_directory)
    elif args.stage == 'validate-cities':
        city_validation(args.config, args.scope)
    else:
        {'select': select, 'templates': templates}[args.stage](args.config)


if __name__ == '__main__':
    main()
