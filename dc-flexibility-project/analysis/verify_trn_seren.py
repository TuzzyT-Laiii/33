import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_trn_seren import cases
from build_typical_load_curves_v3 import checks, read_baseline
from compute_load.cooling import CoolingModel, WEATHER_COLUMNS
from compute_load.model import hourly_index, sha256
from load_curve_scenarios_v3 import TDP, catalogue, size
from prepare_trn_seren import RUN, TYPES, save_json
from trn_power_models import power


def verify(args):
    project = args.project.resolve()
    output = project / 'results' / RUN
    audit = json.loads((output / 'calculation-audit.json').read_text('utf-8'))
    source_audit = json.loads((output / 'source-audit.json').read_text('utf-8'))
    source = pd.read_csv(output / 'source-hourly.csv', parse_dates=['source_clock'])
    assert sha256(output / 'source-hourly.csv') == audit['source_hourly_sha256']
    np.testing.assert_allclose(source.allocated_gpu_hours, source.filter(like='gpu_hours_').sum(axis=1), atol=1e-6)
    np.testing.assert_allclose(source.host_cpu_core_hours, source.filter(like='host_cpu_core_hours_').sum(axis=1), atol=1e-6)
    np.testing.assert_allclose(source.u_cpu_proxy, .25 * source.allocated_gpu_fraction, rtol=1e-12)
    np.testing.assert_allclose(source.allocated_gpu_fraction * source_audit['concurrent_gpu_peak_normalizer'], source.allocated_gpu_hours, rtol=1e-12)
    variants = cases(source, project)
    old_seren = json.loads((project / 'results/trn-power-model-20261101/job-power-evidence/seren/seren-summary.json').read_text('utf-8'))
    assert len(variants['main']) == old_seren['quality_eligible_hour_bins'] == 2057
    np.testing.assert_allclose(source_audit['central_power_statistics']['gpu_hour_weighted_mean_ratio'], old_seren['central_idle60']['gpu_hour_weighted_mean_ratio'], rtol=1e-12)
    assert variants['main'].allocated_gpu_power_ratio.gt(1).sum() == 8
    assert variants['clip_0_1'].allocated_gpu_power_ratio.between(0, 1).all()
    full = variants['full_observed_plus_profile']
    assert len(full) == source_audit['full_source_hours']
    np.testing.assert_array_equal(full.loc[~full.power_imputed, 'allocated_gpu_power_ratio'], variants['main'].allocated_gpu_power_ratio)
    assert variants['main'].power_imputed.sum() == 0
    configs = catalogue(project)
    mit = json.loads((project / 'results/typical-load-curves-20261101-v3/mit-calibration.json').read_text('utf-8'))
    cpu = json.loads((project / 'results/trn-power-model-20261101/cpu-shape.json').read_text('utf-8'))
    old_ratios = json.loads((project / 'results/typical-load-curves-20261029-v2/mit-power-ratio-summary.json').read_text('utf-8'))['transfer_distribution']
    summary = pd.read_csv(output / 'window-summary-all-locations.csv')
    day = pd.read_csv(output / 'typical-day.csv')
    week = pd.read_csv(output / 'typical-week.csv')
    files, rows, gap_files = 0, 0, 0
    for relative, expected in audit['hourly_sha256'].items():
        path = output / relative
        assert sha256(path) == expected, relative
        frame = pd.read_parquet(path)
        checks(frame)
        case, key = frame.case.iloc[0], frame.server_type.iloc[0]
        selected = variants[case]
        clock = pd.DatetimeIndex(pd.to_datetime(frame.source_clock, utc=True))
        assert clock.is_unique and clock.is_monotonic_increasing
        pd.testing.assert_index_equal(clock, pd.DatetimeIndex(selected.source_clock), check_names=False)
        np.testing.assert_array_equal(frame.source_hour_index, selected.source_hour_index)
        weather_clock = pd.DatetimeIndex(frame.time_utc)
        np.testing.assert_array_equal(clock.strftime('%m-%d %H'), weather_clock.strftime('%m-%d %H'))
        assert np.all(weather_clock.year == frame.weather_year.iloc[0])
        gap_files += int(np.any(np.diff(clock.asi8) != 3600 * 10**9))
        ratio = selected.allocated_gpu_power_ratio.to_numpy()
        bounded = np.minimum(ratio, 1)
        previous_power = power(configs[key], key, 'TRN', selected.u_cpu_proxy.to_numpy(), selected.allocated_gpu_fraction.to_numpy(), old_ratios, mit, cpu, training_ratio=bounded)
        count = frame.unit_count.iloc[0]
        np.testing.assert_allclose(frame.base_it_mw, previous_power['base_w'] * count / 1e6, rtol=1e-12)
        np.testing.assert_allclose(frame.cpu_it_mw, previous_power['cpu_w'] * count / 1e6, rtol=1e-12)
        delta = selected.allocated_gpu_fraction.to_numpy() * configs[key]['gpu_devices_per_unit'] * TDP[key] * (ratio - bounded)
        np.testing.assert_allclose(frame.gpu_it_mw, (previous_power['gpu_w'] + delta) * count / 1e6, rtol=1e-12)
        expected_units, capacity = size(configs[key], frame.reference_it_mw.iloc[0])
        assert count == expected_units
        np.testing.assert_allclose(frame.installed_reference_it_mw, capacity, rtol=1e-12)
        row = summary.loc[summary.scenario_id.eq(frame.scenario_id.iloc[0]) & summary.case.eq(case)
            & summary.location_id.eq(frame.location_id.iloc[0]) & summary.weather_year.eq(frame.weather_year.iloc[0])].iloc[0]
        assert len(frame) == row.hours == len(selected)
        for name in ('it', 'facility'):
            values = frame[name + '_mw']
            for reducer, label in [('mean', 'mean'), ('max', 'peak'), ('min', 'minimum'), ('sum', 'window')]:
                column = f'{label}_{name}_' + ('mwh' if label == 'window' else 'mw')
                np.testing.assert_allclose(getattr(values, reducer)(), row[column], rtol=1e-12)
            if frame.location_id.iloc[0] == 'reference-grid':
                for grouping, table in [('hour_of_source_day', day), ('hour_of_source_week', week)]:
                    expected_group = values.groupby(frame[grouping]).mean()
                    published = table.loc[table.scenario_id.eq(frame.scenario_id.iloc[0]) & table.case.eq(case)].set_index(grouping)
                    np.testing.assert_allclose(expected_group, published.loc[expected_group.index, name + '_mw'], rtol=1e-12)
        files += 1
        rows += len(frame)
    assert files == audit['saved_hourly_files']
    assert len(summary) == audit['summary_rows'] == 12 * 10 * audit['location_years']
    assert summary.groupby(['case', 'scenario_id']).size().eq(audit['location_years']).all()
    # 真实旧版输入验证默认连续路径与新增缺口路径的逐小时数值相同。
    baseline, _ = read_baseline(project, args.baseline)
    golden = baseline[('A', 'HPC')]
    weather = golden[[name for name in WEATHER_COLUMNS if name in golden]].copy()
    if golden.pressure_assumed.all():
        weather = weather.drop(columns='pressure_pa')
    calculated = CoolingModel().calculate(weather, golden.it_mw.to_numpy(), golden.installed_reference_it_mw.iloc[0], full_year=True)
    np.testing.assert_allclose(calculated.facility_mw, golden.facility_mw, rtol=1e-12)
    positions = np.arange(0, len(weather), 3)
    gapped = CoolingModel().calculate(weather.iloc[positions], golden.it_mw.to_numpy()[positions], golden.installed_reference_it_mw.iloc[0], full_year=False, allow_gaps=True)
    np.testing.assert_allclose(gapped.facility_mw, calculated.facility_mw.iloc[positions], rtol=1e-12)
    pd.testing.assert_index_equal(pd.DatetimeIndex(gapped.time_utc), pd.DatetimeIndex(weather.time_utc.iloc[positions]), check_names=False)
    try:
        hourly_index(weather.time_utc.iloc[positions], full_year=False)
    except ValueError:
        rejected = True
    else:
        rejected = False
    assert rejected, '默认接口应拒绝缺口'
    save_json(output / 'validation.json', {'utc': datetime.now(timezone.utc).isoformat(),
        'hourly_files_sha256_verified': files, 'saved_hour_rows_verified': rows, 'gapped_hourly_files': gap_files,
        'all_summary_rows': len(summary), 'seren_old_audit_reproduced': True,
        'gpu_and_host_cpu_type_conservation': True, 'no_power_imputation_in_body': True,
        'source_calendar_and_gaps_preserved': True, 'pr36_cpu_base_inventory_and_bounded_gpu_formula_match': True,
        'all_hourly_electric_and_heat_balances': True, 'diurnal_and_weekly_group_means_verified': True,
        'legacy_weather_full_year_power_reproduced': True, 'gapped_cooling_equals_full_series_subset': True,
        'default_continuity_validation_unchanged': True, 'software': {'numpy': np.__version__, 'pandas': pd.__version__}})
    print(json.dumps({'verified_files': files, 'verified_rows': rows, 'summary_rows': len(summary), 'cooling_regression': 'passed'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--baseline', type=Path, required=True)
    verify(parser.parse_args())
