import argparse
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import xarray as xr

from acquire_buildability_protection import sha, utc_now, write_json
from era5_source_validation import analysis_chunk_values, retain_equivalent_source
from trial_era5_landuse_download import SOURCE_URL, VARIABLES


def verify(root):
    selection_path = root / 'data/metadata/buildability-20261025/selection-manifest.json'
    selection = json.loads(selection_path.read_text())
    if selection['status'] != 'complete' or not selection['protected_areas_applied'] or not selection['slope_applied']:
        raise ValueError('验证输入尚未完成保护地与坡度筛选')
    points_path = Path(selection['outputs']['points']['path'])
    if sha(points_path) != selection['outputs']['points']['sha256']:
        raise ValueError('合格天气节点SHA不一致')
    points = pd.read_parquet(points_path)
    lat = np.rint((points.latitude + 90) * 4).astype(int) // 4
    lon = np.rint(((points.longitude_0_360 + 180) % 360) * 4).astype(int) // 4
    expected = set(zip(lat, lon))
    plan = json.loads((root / 'data/metadata/buildability-20261025/weather-stream-relay-plan.json').read_text())
    if expected != {tuple(pair) for pair in plan['blocks']}:
        raise ValueError('下载源空间块集合与最终合格天气节点不同')
    cache = root / 'data/raw/era5-eligible-ecmwf-geo'
    if sha(cache / '.zmetadata') != plan['source_metadata_sha256']:
        raise ValueError('冻结元数据SHA与下载计划不同')
    checked = 0
    audit = root / 'data/metadata/buildability-20261025/weather-source-versions'
    for path in audit.glob('*.json'):
        record = json.loads(path.read_text())
        if not record['url'].startswith(SOURCE_URL + '/'):
            raise ValueError('原文版本证明引用了其他数据源')
        key = record['url'][len(SOURCE_URL) + 1:]
        old = (cache / key).read_bytes()
        incoming = Path(record['incoming_archive_path']).read_bytes()
        if hashlib.sha256(old).hexdigest() != record['sha256'] or hashlib.sha256(incoming).hexdigest() != record['incoming_source_version']['sha256']:
            raise ValueError('原文版本证明与实际字节不同')
        old_values = analysis_chunk_values(cache, key, old)
        new_values = analysis_chunk_values(cache, key, incoming)
        if old_values.tobytes() != new_values.tobytes() or hashlib.sha256(old_values.tobytes()).hexdigest() != record['analysis_values_sha256']:
            raise ValueError('原文更新改变了研究窗口数值或缺失位置')
        checked += 1
    if checked == 0:
        raise ValueError('缺少真实源文件更新样本')
    negative = next(json.loads(path.read_text()) for path in audit.glob('u100__11.68.228.*.json'))
    incoming = Path(negative['incoming_archive_path']).read_bytes()
    with unittest.TestCase().assertRaisesRegex(ValueError, '改变研究窗口数值'):
        retain_equivalent_source(cache, 'u100/11.129.294', incoming, negative['incoming_source_version'])
    probe = root / '.codex-temp/buildability-20261025/cds-wind-repair/wind-20250101-lat068-lon228.nc'
    with xr.open_dataset(probe) as data:
        if data.sizes['valid_time'] != 24 or data.sizes['latitude'] != 4 or data.sizes['longitude'] != 3:
            raise ValueError('官方原文探针时间或节点范围不同')
        for name in ('u100', 'v100'):
            if data[name].attrs.get('units') != 'm s**-1' or not np.isfinite(data[name].values).all():
                raise ValueError('官方原文探针未通过单位与有限值检查')
    known = json.loads((root / '.codex-temp/buildability-20261025/cds-wind-repair/known-jobs.json').read_text())['jobs']
    for year in (2023, 2024):
        request_path = root / 'data/raw/era5-eligible-cds-repairs/lat-068-lon-228' / f'wind-{year}.request.json'
        persisted = json.loads(request_path.read_text())
        if not any(item['request_id'] == persisted['request_id'] and item['request'] == persisted['request'] for item in known):
            raise ValueError('后台补取没有接续已记录的CDS请求')
    checkpoint_path = root / 'checkpoints/era5-eligible-blocks/lat-068-lon-228.json'
    formal_status = 'pending_official_annual_files'
    if checkpoint_path.exists():
        record = json.loads(checkpoint_path.read_text())
        table = pd.read_parquet(record['output_path'])
        if record['hours'] != 26305 or sha(Path(record['output_path'])) != record['output_sha256']:
            raise ValueError('补取后的正式输出SHA或小时数不同')
        if not np.isfinite(table[list(VARIABLES)].to_numpy()).all():
            raise ValueError('补取后的正式输出仍有非有限值')
        formal_status = 'verified_single_block'
    report = {'checked_utc': utc_now(), 'status': 'raw_recovery_verified',
              'selection_manifest_sha256': sha(selection_path), 'selected_grids': selection['selected_grids'],
              'authorized_source_blocks': len(expected), 'verified_source_versions': checked,
              'real_different_values_rejected': True, 'official_probe_hours': 24, 'official_probe_nodes': 12,
              'persistent_cds_request_ids_reused': True,
              'formal_repair_validation': formal_status}
    write_json(root / 'results/buildability-20261025/source-recovery-validation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    verify(args.root.resolve())
