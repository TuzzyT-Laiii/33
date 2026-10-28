import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from acquire_buildability_protection import sha, utc_now, write_json
from acquire_eligible_era5_weather import open_source
from era5_source_validation import apply_verified_source_repair, overlap_check_passed, overlap_statistics


def verify(root):
    label = 'lat-084-lon-290'
    selection_path = root / 'data/metadata/buildability-20261025/selection-manifest.json'
    selection = json.loads(selection_path.read_text())
    points_path = Path(selection['outputs']['points']['path'])
    if sha(points_path) != selection['outputs']['points']['sha256']:
        raise ValueError('真实验证的授权节点清单SHA不同')
    points = pd.read_parquet(points_path).sort_values('native_flat_index')
    lat = np.rint((points.latitude + 90) * 4).astype(int) // 4
    lon = np.rint(((points.longitude_0_360 + 180) % 360) * 4).astype(int) // 4
    points = points.loc[(lat == 84) & (lon == 290)]
    source, session, times, metadata_sha = open_source(root)
    with source:
        original = source.d2m.sel(time=times).sel(
            latitude=xr.DataArray(points.latitude.to_numpy(), dims='weather_point'),
            longitude=xr.DataArray((points.longitude_0_360.to_numpy() + 180) % 360 - 180,
                                   dims='weather_point')).values
    session.close()
    if original.shape != (26305, 10) or original.dtype != np.dtype('float32') or np.isnan(original).sum() != 131280:
        raise ValueError('真实露点缺失、节点、小时或float32基线不同')
    raw_directory = root / 'data/raw/era5-eligible-cds-repairs' / label
    july_path = raw_directory / 'dewpoint-monthly/2024-07.nc'
    with xr.open_dataset(july_path) as july_data:
        official_july = july_data.d2m.sel(
            latitude=xr.DataArray(points.latitude.to_numpy(), dims='weather_point'),
            longitude=xr.DataArray((points.longitude_0_360.to_numpy() + 180) % 360 - 180,
                                   dims='weather_point')).values
        july_times = july_data.valid_time.values
        if july_data.d2m.attrs['units'] != 'K':
            raise ValueError('真实7月原文的单位不同')
    july = (times >= np.datetime64('2024-07-01')) & (times < np.datetime64('2024-08-01'))
    if not np.array_equal(july_times, times[july]) or official_july.dtype != np.dtype('float32'):
        raise ValueError('真实7月原文的小时或float32不同')
    comparison = overlap_statistics(official_july, original[july])
    if comparison['verified_overlap_count'] != 7440 or not overlap_check_passed(comparison):
        raise ValueError('真实7月重合值未通过绝对容差验证')
    # 使用同一真实原文中其他小时的数值验证超过容差时被拒绝。
    mismatched_hours = overlap_statistics(np.roll(official_july, 1, axis=0), original[july])
    if overlap_check_passed(mismatched_hours) or mismatched_hours['max_abs_overlap_difference'] <= 1e-4:
        raise ValueError('真实错时数值未被绝对容差检查拒绝')
    repaired, supplement = apply_verified_source_repair(
        root, label, 'd2m', points, times, original, sha(selection_path), metadata_sha)
    valid = np.isfinite(original)
    if (not np.isfinite(repaired).all() or repaired.dtype != np.dtype('float32')
            or not np.array_equal(repaired[valid], original[valid])):
        raise ValueError('NaN补齐或原有效值保留验证失败')
    proof_path = root / 'data/metadata/buildability-20261025/weather-cds-repairs' / (label + '.json')
    proof = json.loads(proof_path.read_text())
    if proof['checks']['d2m']['max_abs_overlap_difference'] != comparison['max_abs_overlap_difference']:
        raise ValueError('补取证明未记录实际官方重合差值')
    checkpoint_path = root / 'checkpoints/era5-eligible-blocks' / (label + '.json')
    checkpoint = json.loads(checkpoint_path.read_text())
    output_path = Path(checkpoint['output_path'])
    table = pd.read_parquet(output_path)
    if (checkpoint['status'] != 'done' or checkpoint['hours'] != 26305
            or sha(output_path) != checkpoint['output_sha256']
            or not np.array_equal(table.time.to_numpy(), np.repeat(times, len(points)))
            or not np.array_equal(table.native_flat_index.to_numpy(),
                                  np.tile(points.native_flat_index.to_numpy(), len(times)))):
        raise ValueError('正式输出的SHA、小时或授权节点顺序不同')
    for name in checkpoint['output_values_sha256']:
        values = table[name].to_numpy()
        if values.dtype != np.dtype('float32') or not np.isfinite(values).all():
            raise ValueError(f'正式输出变量不满足float32及全部有限：{name}')
    if not np.array_equal(table['2m_dewpoint_temperature'].to_numpy().reshape(original.shape), repaired):
        raise ValueError('正式露点输出与已核验NaN补齐结果不同')
    old_path = root / 'data/metadata/buildability-20261025/weather-cds-repairs/lat-068-lon-228.json'
    old_proof = json.loads(old_path.read_text())
    if not all(overlap_check_passed(check) for check in old_proof['checks'].values()):
        raise ValueError('既有精确一致风数据证明未保持兼容')
    result = {'checked_utc': utc_now(), 'block': label, 'hours': len(times), 'nodes': len(points),
              'official_overlap': comparison, 'real_mismatched_hours_rejected': mismatched_hours,
              'replaced_nan_count': supplement['replaced_nan_count'],
              'unchanged_original_finite_count': int(valid.sum()), 'all_variables_float32_and_finite': True,
              'formal_output_sha256': checkpoint['output_sha256'], 'proof_sha256': sha(proof_path),
              'official_raw_sources': [{'path': item['path'], 'sha256': item['sha256']} for item in proof['raw_sources']],
              'existing_exact_wind_proof_compatible': True}
    write_json(root / 'results/buildability-20261025/overlap-1e4-validation.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    verify(parser.parse_args().root.resolve())
