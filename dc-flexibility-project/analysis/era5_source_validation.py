import hashlib
import json
import os

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from acquire_buildability_protection import utc_now, write_json


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
            or not record['checks'][name]['exact_overlap_equal']
            or record['checks'][name]['verified_overlap_count'] <= 0):
        raise ValueError('CDS补取变量或重合时段证明不符合要求')
    if (record['status'] != 'verified' or record['block'] != label or record['hours'] != 26305
            or record['official_dataset'] != 'reanalysis-era5-single-levels'
            or record['selection_manifest_sha256'] != selection_sha or record['source_metadata_sha256'] != metadata_sha
            or record['native_flat_indices'] != points.native_flat_index.astype(int).tolist()):
        raise ValueError('CDS补取证明与合格节点、时间或来源版本不一致')
    for source in record['raw_sources']:
        raw = root / 'data/raw/era5-eligible-cds-repairs' / label / (f"{prefix}-{source['request']['year'][0]}.nc")
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
