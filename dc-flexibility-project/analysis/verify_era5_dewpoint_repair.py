import argparse
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import xarray as xr

from acquire_buildability_protection import sha, utc_now, write_json
from acquire_eligible_era5_weather import open_source
from complete_era5_wind_repair import verify
from era5_source_validation import apply_verified_source_repair


def verify_repair(root):
    selection_path = root / 'data/metadata/buildability-20261025/selection-manifest.json'
    selection = json.loads(selection_path.read_text())
    if selection['status'] != 'complete' or not selection['slope_applied'] or not selection['protected_areas_applied']:
        raise ValueError('核查要求已完成保护地与坡度筛选')
    points_path = Path(selection['outputs']['points']['path'])
    if sha(points_path) != selection['outputs']['points']['sha256']:
        raise ValueError('授权天气节点清单SHA不一致')
    points = pd.read_parquet(points_path).sort_values('native_flat_index')
    latitude = np.rint((points.latitude + 90) * 4).astype(int) // 4
    longitude = np.rint(((points.longitude_0_360 + 180) % 360) * 4).astype(int) // 4
    selected = points.loc[(latitude == 84) & (longitude == 290)]
    wind_points = points.loc[(latitude == 68) & (longitude == 228)]
    source, session, times, metadata_sha = open_source(root)
    with source:
        values = source['d2m'].sel(time=times).sel(
            latitude=xr.DataArray(selected.latitude.to_numpy(), dims='weather_point'),
            longitude=xr.DataArray((selected.longitude_0_360.to_numpy() + 180) % 360 - 180,
                                   dims='weather_point')).values
        wind_values = source['u100'].sel(time=times).sel(
            latitude=xr.DataArray(wind_points.latitude.to_numpy(), dims='weather_point'),
            longitude=xr.DataArray((wind_points.longitude_0_360.to_numpy() + 180) % 360 - 180,
                                   dims='weather_point')).values
    session.close()
    if values.shape != (26305, 10) or values.dtype != np.dtype('float32'):
        raise ValueError('真实露点输入的小时、节点数或类型不同')
    if np.isnan(values).sum() != 131280 or np.isinf(values).any():
        raise ValueError('真实露点缺失位置数量与已核查输入不同')
    wind_result = verify(root, 'lat-068-lon-228')
    repaired_wind, wind_proof = apply_verified_source_repair(
        root, 'lat-068-lon-228', 'u100', wind_points, times, wind_values,
        sha(selection_path), metadata_sha)
    if not np.isfinite(repaired_wind).all() or not np.array_equal(
            repaired_wind[np.isfinite(wind_values)], wind_values[np.isfinite(wind_values)]):
        raise ValueError('通用补取入口未保留已有真实风数值')
    with unittest.TestCase().assertRaisesRegex(ValueError, '变量或重合时段证明'):
        apply_verified_source_repair(root, 'lat-068-lon-228', 'd2m', wind_points,
                                     times, wind_values, sha(selection_path), metadata_sha)
    proof_path = root / 'data/metadata/buildability-20261025/weather-cds-repairs/lat-084-lon-290.json'
    dewpoint = 'pending_official_files_and_exact_overlap'
    if proof_path.exists():
        proof = json.loads(proof_path.read_text())
        if proof['status'] in ('awaiting_overlap_validation', 'verified'):
            dewpoint = verify(root, 'lat-084-lon-290')
    report = {'checked_utc': utc_now(), 'selection_manifest_sha256': sha(selection_path),
              'source_metadata_sha256': metadata_sha, 'block': 'lat-084-lon-290',
              'variable': 'd2m', 'selected_points': len(selected), 'hours': len(times),
              'original_nan_values': int(np.isnan(values).sum()),
              'original_missing_hours': int(np.isnan(values).any(axis=1).sum()),
              'existing_wind_repair': wind_result,
              'existing_wind_nan_filled': wind_proof['replaced_nan_count'],
              'wind_proof_rejected_for_dewpoint': True,
              'official_dewpoint_validation': dewpoint}
    write_json(root / 'results/buildability-20261025/dewpoint-recovery-validation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    verify_repair(args.root.resolve())
