import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_trn_power_model import BASE, RUN
from build_typical_load_curves_v3 import checks
from compute_load.model import sha256
from load_curve_scenarios_v3 import catalogue, TDP
from trn_power_models import cpu_response


def verify(project, all_locations=False):
    output = project / 'results' / RUN
    old_root = project / 'results' / BASE
    manifest = json.loads((output / 'manifest.json').read_text('utf-8'))
    old_manifest = json.loads((old_root / 'manifest.json').read_text('utf-8'))
    shape = json.loads((output / 'cpu-shape.json').read_text('utf-8'))
    configs = catalogue(project)
    u = np.linspace(0, 1, 10001)
    response = cpu_response(u, shape)
    if response[0] != 0 or response[-1] != 1 or (np.diff(response) < 0).any():
        raise ValueError('CPU 端点或单调性不满足约束')
    step = float(np.diff(cpu_response(np.array([.5, .6]), shape))[0] * 410)
    if step >= 60:
        raise ValueError('CPU 50% 至 60% 区间仍有过大跃升')
    table = pd.read_csv(output / 'annual-summary.csv').set_index('scenario_id')
    immutable = ['time_utc', 'u_cpu_proxy', 'allocated_gpu_fraction', 'unit_count', 'installed_reference_it_mw',
                 'temperature_c', 'relative_humidity_pct', 'pressure_pa', 'pressure_assumed']
    verified, hours = 0, 0
    for path in sorted((output / 'hourly').glob('*.parquet')):
        old_path = old_root / 'hourly' / path.name
        old, new = pd.read_parquet(old_path), pd.read_parquet(path)
        if sha256(old_path) != old_manifest['hourly_sha256'][str(old_path.relative_to(project)).replace('\\', '/')]:
            raise ValueError('v3 原始小时结果发生变化')
        if sha256(path) != manifest['hourly_sha256'][path.name]:
            raise ValueError('新小时文件与清单不符')
        pd.testing.assert_frame_equal(new[immutable], old[immutable], check_exact=True)
        checks(new)
        np.testing.assert_allclose(new.it_mw, new.base_it_mw + new.cpu_it_mw + new.gpu_it_mw, atol=1e-10)
        np.testing.assert_allclose(new.it_mw, new.idle_it_mw + new.task_it_mw, atol=1e-10)
        key, use = new.server_type.iloc[0], new.use.iloc[0]
        if use in ('SAI', 'INF'):
            np.testing.assert_array_equal(new.gpu_it_mw, old.gpu_it_mw)
        if use == 'TRN':
            a, n = new.allocated_gpu_fraction.to_numpy(), configs[key]['gpu_devices_per_unit']
            expected = n * (a * TDP[key] * .8 + (1 - a) * (70 if key == 'D' else 50)) * new.unit_count.iloc[0] / 1e6
            np.testing.assert_allclose(new.gpu_it_mw, expected, rtol=1e-12)
        row = table.loc[new.scenario_id.iloc[0]]
        for col in ('it', 'facility'):
            np.testing.assert_allclose([new[col + '_mw'].mean(), new[col + '_mw'].max()], [row['mean_' + col + '_mw'], row['peak_' + col + '_mw']], rtol=1e-12)
        np.testing.assert_allclose(row.task_share_of_it_pct, 100 * new.task_it_mw.sum() / new.it_mw.sum(), rtol=1e-12)
        hours += len(new)
        verified += 1
    if verified != 44 or hours != 385440:
        raise ValueError('参考网格情境和小时数不符')
    sensitivity = pd.read_csv(output / 'sensitivity-summary.csv')
    for key in ('B', 'C', 'D'):
        rows = sensitivity.loc[sensitivity.server_type.eq(key) & sensitivity.use.eq('TRN')].set_index('sensitivity')
        if not rows.loc['gpu_low', 'mean_it_mw'] < rows.loc['central', 'mean_it_mw'] < rows.loc['gpu_high', 'mean_it_mw']:
            raise ValueError('TRN GPU 功率敏感性排序不符合预期')
        if not rows.loc['idle_low', 'mean_it_mw'] < rows.loc['central', 'mean_it_mw'] < rows.loc['idle_high', 'mean_it_mw']:
            raise ValueError('空闲功率敏感性排序不符合预期')
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'reference_scenarios': verified, 'reference_hours': hours,
              'cpu_50_to_60_dynamic_increment_W': step, 'frozen_demand_weather_units': True,
              'SAI_INF_GPU_arrays_unchanged': True, 'IT_and_facility_energy_conservation': True,
              'training_formula_independently_checked': True, 'sensitivity_ordering': True,
              'tests_use_real_v3_inputs': True}
    if all_locations:
        city_manifest = json.loads((output / 'all-locations-manifest.json').read_text('utf-8'))
        count, city_hours = 0, 0
        for name, digest in city_manifest['hourly_sha256'].items():
            path = project / name
            if sha256(path) != digest:
                raise ValueError('城市小时 SHA-256 不符：' + name)
            frame = pd.read_parquet(path)
            checks(frame)
            city_hours += len(frame)
            count += 1
        if count != 1100 or city_hours != 9644448:
            raise ValueError('全部城市情境数量不符')
        result.update(all_location_scenarios=count, all_location_hours=city_hours, all_location_sha256_verified=True)
    (output / ('validation-all-locations.json' if all_locations else 'validation.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--all-locations', action='store_true')
    args = parser.parse_args()
    verify(args.project.resolve(), args.all_locations)
