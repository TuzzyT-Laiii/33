import hashlib
import json
import os

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from acquire_buildability_protection import utc_now, write_json


OVERLAP_ABSOLUTE_TOLERANCE = 1e-4


def overlap_statistics(official, original):
    if official.shape != original.shape:
        raise ValueError('重合值比较的形状不同')
    valid = np.isfinite(original)
    if not np.isfinite(official[valid]).all():
        raise ValueError('官方重合值存在非有限值')
    delta = np.abs(official[valid].astype(np.float64) - original[valid].astype(np.float64))
    return {'verified_overlap_count': int(delta.size),
            'max_abs_overlap_difference': float(delta.max()) if delta.size else None,
            'mean_abs_overlap_difference': float(delta.mean()) if delta.size else None,
            'overlap_absolute_tolerance': OVERLAP_ABSOLUTE_TOLERANCE,
            'overlap_within_tolerance': bool(delta.size and np.all(delta <= OVERLAP_ABSOLUTE_TOLERANCE)),
            'exact_overlap_equal': bool(delta.size and np.all(delta == 0))}


def overlap_check_passed(check):
    if check['verified_overlap_count'] <= 0:
        return False
    if 'overlap_absolute_tolerance' not in check:
        return check['exact_overlap_equal'] and check['max_abs_overlap_difference'] == 0
    maximum = check['max_abs_overlap_difference']
    mean = check['mean_abs_overlap_difference']
    return (check['overlap_absolute_tolerance'] == OVERLAP_ABSOLUTE_TOLERANCE
            and check['overlap_within_tolerance'] is True
            and maximum is not None and mean is not None
            and np.isfinite(maximum) and np.isfinite(mean)
            and 0 <= mean <= maximum <= OVERLAP_ABSOLUTE_TOLERANCE)


def analysis_chunk_values(cache, key, payload):
    variable, chunk_name = key.split('/')
    hour, latitude, longitude = map(int, chunk_name.split('.'))
    metadata = json.loads((cache / '.zmetadata').read_bytes())['metadata']
    spec = metadata[variable + '/.zarray']
    if spec['chunks'] != [67584, 4, 4] or spec['dtype'] != '<f4':
        raise ValueError(f'ERA5冻结分块结构不符合已核验布局：{key}')
    with xr.open_zarr(zarr.storage.DirectoryStore(str(cache)), consolidated=True, chunks=None) as source:
        requested = pd.date_range('2023-01-01', '2026-01-01', freq='h')
        positions = source.indexes['time'].get_indexer(requested)
    if len(positions) != 26305 or positions.min() < 0 or not np.all(np.diff(positions) == 1):
        raise ValueError('ERA5冻结时间轴不符合研究小时范围')
    start = max(int(positions[0]), hour * 67584)
    stop = min(int(positions[-1]) + 1, (hour + 1) * 67584)
    if stop <= start:
        raise ValueError(f'ERA5对象不属于本研究所需时间分块：{key}')
    memory = zarr.storage.MemoryStore()
    memory['.zarray'] = json.dumps(spec).encode()
    memory[chunk_name] = payload
    array = zarr.open_array(memory, mode='r')
    values = array[start:stop, latitude * 4:latitude * 4 + 4, longitude * 4:longitude * 4 + 4]
    if values.shape != (stop - start, 4, 4) or values.dtype != np.dtype('float32'):
        raise ValueError(f'ERA5研究窗口解码形状或类型异常：{key}')
    return values


def retain_equivalent_source(cache, key, incoming, receipt):
    existing_path = cache / key
    existing = existing_path.read_bytes()
    old_values = analysis_chunk_values(cache, key, existing)
    new_values = analysis_chunk_values(cache, key, incoming)
    old_digest = hashlib.sha256(np.ascontiguousarray(old_values).tobytes()).hexdigest()
    new_digest = hashlib.sha256(np.ascontiguousarray(new_values).tobytes()).hexdigest()
    if old_digest != new_digest:
        raise ValueError(f'ERA5源对象更新改变研究窗口数值：{key}')
    incoming_sha = hashlib.sha256(incoming).hexdigest()
    if incoming_sha != receipt['sha256'] or len(incoming) != receipt['bytes']:
        raise ValueError(f'ERA5更新对象原始字节与传输清单不一致：{key}')
    # 源对象包含研究时段以后的小时；保留冻结原文并逐值验证研究窗口。
    variants = cache / '.source-variants'
    variants.mkdir(parents=True, exist_ok=True)
    path = variants / incoming_sha
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != incoming_sha:
            raise ValueError(f'ERA5原文版本归档SHA不一致：{key}')
    else:
        temporary = path.with_name(path.name + f'.{os.getpid()}.part')
        temporary.write_bytes(incoming)
        temporary.replace(path)
    retained = {**receipt, 'sha256': hashlib.sha256(existing).hexdigest(), 'bytes': len(existing),
                'downloaded_utc': None, 'validated_utc': utc_now(), 'origin': 'retained_same_source_cache',
                'incoming_source_version': receipt, 'incoming_archive_path': str(path),
                'analysis_values_sha256': old_digest,
                'analysis_nan_count': int(np.isnan(old_values).sum()),
                'analysis_inf_count': int(np.isinf(old_values).sum()),
                'analysis_window_utc': ['2023-01-01T00:00:00', '2026-01-01T00:00:00'],
                'comparison': '研究窗口内全部16个空间节点逐小时float32字节及缺失位置完全一致；正式整理独立检查全部数值有限'}
    audit = cache.parent.parent.parent / 'data/metadata/buildability-20261025/weather-source-versions'
    write_json(audit / (key.replace('/', '__') + '.' + incoming_sha[:16] + '.json'), retained)
    return retained


def apply_verified_source_repair(root, label, name, points, times, values, selection_sha, metadata_sha):
    if name not in ('u100', 'v100', 'd2m') or np.isinf(values).any():
        raise ValueError('定向源数据补取仅适用于已核查变量的NaN缺失值')
    path = root / 'data/metadata/buildability-20261025/weather-cds-repairs' / (label + '.json')
    record = json.loads(path.read_text())
    variables = tuple(record.get('variables', ('u100', 'v100')))
    prefix = record.get('source_file_prefix', 'wind')
    if ((prefix, variables) not in (('wind', ('u100', 'v100')), ('dewpoint', ('d2m',)))
            or name not in variables
            or not overlap_check_passed(record['checks'][name])):
        raise ValueError('CDS补取变量或重合时段证明不符合要求')
    if (record['status'] != 'verified' or record['block'] != label or record['hours'] != 26305
            or record['official_dataset'] not in ('reanalysis-era5-single-levels', 'reanalysis-era5-single-levels-timeseries', 'nsf-ncar-era5')
            or record['selection_manifest_sha256'] != selection_sha or record['source_metadata_sha256'] != metadata_sha
            or record['native_flat_indices'] != points.native_flat_index.astype(int).tolist()):
        raise ValueError('CDS补取证明与合格节点、时间或来源版本不一致')
    if record['official_dataset'] != 'reanalysis-era5-single-levels':
        comparisons = record['alternative_comparisons']
        if set(comparisons) != {'official_2023_01', 'arco_2024_07'} or not all(
                item['exact_equal'] and item['count'] == 744 * len(points) and item['max_abs_difference'] == 0
                for item in comparisons.values()):
            raise ValueError('替代来源未通过两个完整月份的逐值一致检查')
    for source in record['raw_sources']:
        directory = root / 'data/raw/era5-eligible-cds-repairs' / label
        layout = source.get('source_layout')
        if layout == 'point_timeseries':
            if record['official_dataset'] != 'reanalysis-era5-single-levels-timeseries' or source['dataset'] != record['official_dataset']:
                raise ValueError('点位时间序列来源与证明不同')
            request = source['request']
            location = request['location']
            matches = points.loc[(points.latitude == location['latitude']) &
                                 (((points.longitude_0_360 + 180) % 360 - 180) == location['longitude'])]
            if len(matches) != 1 or request['variable'] != ['2m_dewpoint_temperature'] or request['date'] != ['2023-01-01', '2024-07-31']:
                raise ValueError('点位时间序列范围不符合授权')
            raw = directory / 'timeseries' / f'{int(matches.native_flat_index.iloc[0])}.nc'
        elif layout == 'ncar_extracted':
            if record['official_dataset'] != 'nsf-ncar-era5' or source['dataset'] != 'nsf-ncar-era5' or source['month'] not in ['202301', *[f'2024{month:02d}' for month in range(1, 8)]]:
                raise ValueError('NCAR补取来源或月份不符合授权')
            if not source['source_url'].startswith('https://nsf-ncar-era5.s3.us-west-2.amazonaws.com/e5.oper.an.sfc/' + source['month'] + '/e5.oper.an.sfc.128_168_2d.'):
                raise ValueError('NCAR补取原始对象不符合来源约束')
            raw = directory / 'ncar' / (source['month'] + '.nc')
        elif layout == 'monthly':
            year = source['request']['year'][0]
            if prefix != 'dewpoint' or len(source['request']['month']) != 1 or year not in ('2024', '2025'):
                raise ValueError('CDS按月原文的变量或时段超出授权补取范围')
            month = source['request']['month'][0]
            if month not in {f'{value:02d}' for value in range(1, 13)}:
                raise ValueError('CDS按月原文月份无效')
            if record.get('missing_only') and (year != '2024' or int(month) > 7):
                raise ValueError('限定缺失月份的CDS原文超出2024年1至7月')
            raw = directory / f'{prefix}-monthly/{year}-{month}.nc'
        elif source.get('source_layout') is None:
            year = source['request']['year'][0]
            raw = directory / f'{prefix}-{year}.nc'
        else:
            raise ValueError('未知的CDS原文时间布局')
        if str(raw) != source['path'] or hashlib.sha256(raw.read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('CDS补取原文的路径或SHA不一致')
    output = root / 'data/interim/era5-eligible-cds-repairs' / (label + '.parquet')
    if str(output) != record['output_path'] or hashlib.sha256(output.read_bytes()).hexdigest() != record['output_sha256']:
        raise ValueError('CDS补取数值表的路径或SHA不一致')
    table = pd.read_parquet(output)
    if (not np.array_equal(table.time.to_numpy(), np.repeat(times, len(points)))
            or not np.array_equal(table.native_flat_index.to_numpy(), np.tile(points.native_flat_index.to_numpy(), len(times)))):
        raise ValueError('CDS补取数据的时间或节点顺序不一致')
    official = table[name].to_numpy().reshape(values.shape)
    if official.dtype != np.dtype('float32') or not np.isfinite(official).all():
        raise ValueError('CDS补取数据仍有缺失或类型错误')
    present = np.isfinite(values)
    if not np.array_equal(official[present], values[present]):
        raise ValueError('CDS补取与现有有效数据不完全一致')
    repaired = values.copy()
    missing = np.isnan(values)
    repaired[missing] = official[missing]
    proof = {'manifest_path': str(path), 'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
             'raw_sources': record['raw_sources'], 'replaced_nan_count': int(missing.sum()),
             'unchanged_finite_values': int(present.sum()), 'method': '只使用已核验官方原文中同时间同节点的数值补齐NaN'}
    return repaired, proof


def apply_verified_wind_repair(root, label, name, points, times, values, selection_sha, metadata_sha):
    if name not in ('u100', 'v100'):
        raise ValueError('风数据补取接口要求风变量')
    return apply_verified_source_repair(root, label, name, points, times, values, selection_sha, metadata_sha)
