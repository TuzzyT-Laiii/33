import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compute_load.cooling import CoolingModel
from compute_load.model import sha256
from load_curve_scenarios_v3 import RUN, SCALES, LEGACY, DAYS, catalogue, combinations


def verify(args):
    project = args.project.resolve()
    output = project / 'results' / RUN
    manifest = json.loads((output / 'manifest.json').read_text('utf-8'))
    summary = pd.read_csv(output / 'annual-summary-all-locations.csv')
    full = manifest['stage'] == 'full'
    active = [(key, use) for key, use in combinations() if full or use != 'SAI']
    expected_count = len(active) * 4
    expected_ids = {f'{key}-{use}-{scale}MW' for key, use in active for scale in SCALES}
    checks = {}
    checks['reference_scenario_coverage'] = set(summary.loc[summary.location_id.eq('reference-grid'), 'scenario_id']) == expected_ids
    checks['no_out_of_scope_server_types'] = set(summary.server_type) <= set('ABCD')
    checks['all_air_cooling'] = summary.cooling_type.eq('air').all()
    checks['all_location_years_complete'] = summary.groupby(['location_id', 'year']).size().eq(expected_count).all()
    checks['unique_scenario_location_year'] = not summary.duplicated(['location_id', 'year', 'scenario_id']).any()
    checks['annual_hours'] = summary.hours.eq(np.where(summary.year.eq(2024), 8784, 8760)).all()
    cld = pd.read_csv(project / 'results/alibaba-cpu-cld-20261031/complete-window.csv')
    inf = pd.read_csv(project / 'results/asi-185day-workload-types-20261031/inf-hourly-quality.csv', float_precision='round_trip')
    ratios = json.loads((project / 'results/typical-load-curves-20261029-v2/mit-power-ratio-summary.json').read_text('utf-8'))['transfer_distribution']
    mit = json.loads((output / 'mit-calibration.json').read_text('utf-8')) if full else None
    configs = catalogue(project)
    read_rows, maximum_energy_error, max_component_error = 0, 0., 0.
    for relative, digest in manifest['hourly_sha256'].items():
        if args.reference_only and not relative.startswith('results/'):
            continue
        path = project / relative
        if sha256(path) != digest:
            raise ValueError('v3 小时输出 SHA-256 不符：' + relative)
        frame = pd.read_parquet(path)
        scenario = frame.scenario_id.iloc[0]
        key, use = frame.server_type.iloc[0], frame.use.iloc[0]
        year = int(frame.time_utc.iloc[0].year)
        location = 'reference-grid' if relative.startswith('results/') else path.parent.name
        if not frame.location_id.eq(location).all() or not frame.year.eq(year).all():
            raise ValueError('小时文件地点或年份字段与索引不符')
        if location == 'reference-grid' and (not frame.pressure_assumed.all() or not frame.pressure_pa.eq(101325.).all()):
            raise ValueError('参考网格气压假设标识未保留')
        row = summary.loc[summary.scenario_id.eq(scenario) & summary.location_id.eq(location) & summary.year.eq(year)].iloc[0]
        index = pd.DatetimeIndex(frame.time_utc)
        expected = pd.date_range(f'{year}-01-01', f'{year+1}-01-01', inclusive='left', freq='h', tz='UTC')
        if not index.equals(expected):
            raise ValueError('小时索引错误：' + relative)
        position = np.arange(len(frame)) % (DAYS[use] * 24)
        np.testing.assert_array_equal(frame.source_hour_index, position)
        if use == 'CLD':
            np.testing.assert_allclose(frame.u_cpu_proxy, cld.resource_fraction.to_numpy()[position], rtol=1e-13)
        elif use == 'INF':
            expected_gpu = inf.combined_used_gpu_hours.to_numpy()[position] / inf.combined_used_gpu_hours.max()
            np.testing.assert_allclose(frame.allocated_gpu_fraction, expected_gpu, rtol=1e-13)
            np.testing.assert_allclose(frame.u_cpu_proxy, .25 * expected_gpu, rtol=1e-13)
            np.testing.assert_array_equal(frame.source_quality_flag, inf.source_quality_flag.to_numpy()[position])
            np.testing.assert_array_equal(frame.source_unresolved, position == 126)
        if use in ('SAI', 'TRN', 'INF'):
            devices = configs[key]['gpu_devices_per_unit']
            tdp = 700. if key == 'D' else 350.
            allocated_ratio = mit['measured_power_to_tdp_ratio'] if use == 'SAI' else ratios['p50']
            idle = tdp * mit['idle_W_per_gpu'] / mit['tdp_W'] if use == 'SAI' else (70. if key == 'D' else 50.)
            gpu_w = (frame.allocated_gpu_fraction * allocated_ratio * tdp + (1 - frame.allocated_gpu_fraction) * idle)
            expected_gpu_mw = gpu_w * devices * frame.unit_count / 1e6
            np.testing.assert_allclose(frame.gpu_it_mw, expected_gpu_mw, rtol=1e-12, atol=1e-10)
        error = np.max(np.abs(frame.facility_mw - frame.it_mw - frame.cooling_mw - frame.other_mw))
        max_component_error = max(max_component_error, float(error))
        np.testing.assert_allclose(frame.it_mw, frame.base_it_mw + frame.cpu_it_mw + frame.gpu_it_mw, rtol=1e-12)
        np.testing.assert_allclose(frame.cooling_mw, frame[[name for name in frame if name in (
            'room_fan_mw', 'chilled_pump_mw', 'cdu_mw', 'primary_pump_mw', 'condenser_pump_mw',
            'tower_fan_mw', 'dry_cooler_fan_mw', 'air_compressor_mw', 'liquid_compressor_mw', 'freeze_heater_mw')]].sum(axis=1), atol=1e-9)
        for metric in ('it', 'facility', 'cooling', 'other'):
            actual = float(frame[metric + '_mw'].sum())
            expected_energy = float(row['annual_' + metric + '_mwh'])
            maximum_energy_error = max(maximum_energy_error, abs(actual - expected_energy))
            if not np.isclose(actual, expected_energy, rtol=1e-12, atol=1e-7):
                raise ValueError('小时积分与年汇总不符')
        if location == 'reference-grid' and row.reference_it_mw == 10:
            weather = frame[['time_utc', 'temperature_c', 'dew_point_c', 'pressure_pa']]
            calculated = CoolingModel(cooling_type='air').calculate(weather, frame.it_mw.to_numpy(), frame.installed_reference_it_mw.iloc[0])
            np.testing.assert_allclose(calculated.facility_mw, frame.facility_mw, rtol=1e-12, atol=1e-10)
        read_rows += len(frame)
    checks['hourly_conservation'] = max_component_error < 1e-8
    checks['hourly_annual_integral'] = maximum_energy_error < 1e-6
    comparison = pd.read_csv(output / 'v2-v3-comparison.csv')
    unchanged = comparison.loc[comparison.use.isin(['HPC', 'TRN'])]
    checks['HPC_TRN_unchanged_hourly'] = unchanged.it_hourly_max_abs_difference_mw.lt(1e-9).all() and unchanged.facility_hourly_max_abs_difference_mw.lt(1e-9).all()
    checks['CLD_new_scenarios_have_no_v2_baseline'] = comparison.loc[comparison.use.eq('CLD'), 'v2_available'].eq(False).all()
    if full:
        checks['MIT_calibration_snapshot_hash'] = sha256(output / 'mit-calibration.json') == manifest['mit_calibration_sha256']
        checks['MIT_release_status_explicit'] = mit['release_status'] == ('阶段版' if mit['partial_data'] else '六月窗口完成版')
        legacy = json.loads((project / 'results/typical-load-curves-20261029-v2/manifest.json').read_text('utf-8'))
        for key in ('B', 'C', 'D'):
            path = args.baseline / f'{LEGACY[key]}-MIT-ratio_p50-central.parquet'
            if sha256(path) != legacy['processed_sha256'][path.name]:
                raise ValueError('MIT 原分配输入被修改')
            old = pd.read_parquet(path)
            current = pd.read_parquet(output / 'hourly' / f'{key}-SAI-10MW-2025.parquet')
            np.testing.assert_array_equal(old.allocated_gpu_fraction, current.allocated_gpu_fraction)
            np.testing.assert_array_equal(old.u_cpu_proxy, current.u_cpu_proxy)
        checks['MIT_237day_allocations_unchanged'] = True
    if manifest['location_years'] == 25:
        old = pd.read_csv(project / 'results/eight-city-phase-a-20261029/annual-load-central.csv')
        for key, use in [('A', 'HPC'), ('B', 'TRN'), ('C', 'TRN'), ('D', 'TRN')]:
            rows = summary.loc[summary.scenario.eq(f'{key}-{use}') & summary.reference_it_mw.eq(10) & ~summary.location_id.eq('reference-grid')]
            for row in rows.itertuples():
                prior = old.loc[old.legacy_configuration_id.eq(LEGACY[key]) & old.demand_source.eq('Lisa' if use == 'HPC' else 'Acme')
                                & old.city_id.eq(row.location_id) & old.year.eq(row.year)].iloc[0]
                np.testing.assert_allclose([row.annual_it_mwh, row.annual_facility_mwh, row.peak_facility_mw],
                    [prior.annual_it_mwh, prior.annual_facility_mwh, prior.peak_facility_mw], rtol=1e-10, atol=1e-7)
        checks['eight_city_HPC_TRN_matches_existing_results'] = True
    if not all(checks.values()):
        raise ValueError({key: bool(value) for key, value in checks.items() if not value})
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'checks': {k: bool(v) for k, v in checks.items()},
              'stage': manifest['stage'], 'reference_only': args.reference_only, 'checked_hourly_rows': read_rows,
              'maximum_facility_balance_error_mw': max_component_error,
              'maximum_annual_integration_error_mwh': maximum_energy_error, 'status': 'passed'}
    name = 'validation-reference.json' if args.reference_only else 'validation.json'
    (output / name).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--reference-only', action='store_true')
    verify(parser.parse_args())
