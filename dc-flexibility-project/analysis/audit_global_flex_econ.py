import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

from global_flex_inputs import digest, facility_frame, settings, utc_now, write_json


def compare_runtime(run):
    folders = [run / name / 'shards' for name in ('memory-sample', 'memory-sample-runtime')]
    frames = [pd.DataFrame([json.loads(line) for path in folder.glob('*.jsonl')
                            for line in path.read_text('utf-8').splitlines()]) for folder in folders]
    columns = ['status', 'total_cost_usd', 'pv_mw', 'wind_mw', 'battery_mw', 'battery_mwh',
               'grid_cap_mw', 'extra_task_mw', 'facility_mwh', 'grid_mwh']
    frames = [frame.set_index('case_key').sort_index()[columns] for frame in frames]
    pd.testing.assert_frame_equal(*frames, check_exact=True)
    progress = json.loads((run / 'memory-sample-runtime/progress.json').read_text('utf-8'))
    assert progress['completed'] == 24 and progress['performed_lps'] == 18
    assert progress['actual_lp_optimal_rate'] == 1 and progress['failed_lps'] == 0
    attempts = [json.loads(line) for line in (run / 'memory-sample-runtime/execution-attempts.jsonl').read_text('utf-8').splitlines()]
    assert attempts[-2]['event'] == 'start' and attempts[-2]['completed_before'] == 24
    return {'passed': True, 'summary_rows': 24, 'columns_compared': columns,
            'results_exactly_equal': True, 'resumed_without_additional_lps': True}


def pilot(config, report_folder):
    own, paths = settings(config)
    run = paths['run_root']
    progress = json.loads((run / 'pilot-confirmed/progress.json').read_text('utf-8'))
    report = json.loads((report_folder / 'summary-manifest.json').read_text('utf-8'))
    tests = json.loads((run / 'validation/confirmed-input-tests.json').read_text('utf-8'))
    assert progress['state'] == 'complete' and progress['performed_lps'] == 3960
    assert progress['completed'] == 5280 and progress['status_counts'] == {'optimal': 5280}
    assert report['fixed_asset_errors'] == report['missing_dispatch_audits'] == 0
    assert tests['passed'] and report['valid_pairs'] == 2640
    comparison = pd.read_csv(report_folder / 'eight-city-grid-comparison.csv')
    assert len(comparison) == 8 * len(own['scenarios']) * len(own['alpha']) * 2
    assert comparison.savings_pct.notna().all() and comparison.city_savings_pct.notna().all()
    # 接近零的收益不计算比值，保留绝对差供审阅。
    measurable = comparison.loc[comparison.city_savings_pct.abs() >= .1].copy()
    measurable['grid_to_city_ratio'] = measurable.savings_pct / measurable.city_savings_pct
    measurable.to_csv(run / 'validation/pilot-city-measurable-comparison.csv', index=False)
    economic = comparison.loc[comparison.scenario.eq('D-TRN') & comparison.alpha.isna()]
    assert np.all((economic.savings_pct / economic.city_savings_pct).between(.2, 5))
    groups = comparison.groupby(['scenario', 'boundary'], dropna=False).agg(
        comparisons=('savings_pct', 'size'),
        minimum_city_pct=('city_savings_pct', 'min'), maximum_city_pct=('city_savings_pct', 'max'),
        minimum_grid_pct=('savings_pct', 'min'), maximum_grid_pct=('savings_pct', 'max'),
        minimum_difference_pp=('grid_minus_city_percentage_points', 'min'),
        maximum_difference_pp=('grid_minus_city_percentage_points', 'max'))
    groups.to_csv(run / 'validation/pilot-city-comparison-ranges.csv')
    return {'passed': True, 'grids': 20, 'summary_rows': 5280, 'performed_lps': 3960,
            'optimal_lps': 3960, 'reused_baselines': 1320, 'actual_lp_optimal_rate': 1.,
            'fixed_asset_errors': 0, 'missing_dispatch_audits': 0,
            'all_city_scenario_alpha_boundary_comparisons': len(comparison),
            'measurable_ratio_min': float(measurable.grid_to_city_ratio.min()),
            'measurable_ratio_max': float(measurable.grid_to_city_ratio.max()),
            'mean_lp_seconds': progress['mean_lp_seconds'],
            'controller_elapsed_seconds': progress['session_elapsed_seconds'],
            'measured_workers': progress['workers'],
            'economic_dtrn': economic[['city', 'boundary', 'city_savings_pct', 'savings_pct']].to_dict('records')}


def inputs(config):
    own, paths = settings(config)
    run = paths['run_root']
    selected = pd.read_parquet(run / 'inputs/selected-grids.parquet')
    points = pd.read_parquet(run / 'inputs/grid-points.parquet')
    index = pd.read_csv(run / 'inputs/grid-input-index-all.csv')
    templates = pd.read_csv(run / 'inputs/template-index.csv')
    assert set(templates.scenario) == set(own['scenarios'])
    for row in templates.itertuples():
        assert digest(run / 'inputs/templates' / (row.scenario + '.parquet')) == row.template_sha256
        assert digest(row.load_path) == row.load_sha256
    assert len(selected) == len(index) == 4900 and not index.grid_id.duplicated().any()
    assert set(selected.grid_id) == set(index.grid_id)
    assert selected.country_assignment_method.value_counts().to_dict() == {'center_in_polygon': 3820, 'nearest_boundary': 1080}
    assert selected[['iso3', 'carbon_gco2_kwh', 'nearest_boundary_distance_km']].notna().all().all()
    assert selected.loc[selected.carbon_source.eq('regional_fallback'), 'm49_region'].notna().all()
    assert selected.carbon_source.isin(['country', 'regional_fallback']).all()
    assert selected.nearest_boundary_distance_km.ge(0).all()
    land_counts = points.groupby('grid_id').is_land.sum()
    climate_rows = []
    for row in index.itertuples():
        assert row.land_points == int(land_counts[row.grid_id])
        if row.land_points == 0:
            assert row.input_status == 'no_land_points'
            continue
        assert row.input_status == 'ready'
        path = run / 'inputs/grids' / (row.grid_id + '.npz')
        assert digest(path) == row.sha256
        with np.load(path) as archive:
            for key in ('cf_pv', 'cf_wind', 'temperature_c', 'dew_point_c', 'pressure_pa'):
                assert archive[key].shape == (8760,) and np.isfinite(archive[key]).all()
            assert ((archive['cf_pv'] >= 0) & (archive['cf_pv'] <= 1)).all()
            assert ((archive['cf_wind'] >= 0) & (archive['cf_wind'] <= 1)).all()
            np.testing.assert_array_equal(archive['native_flat_index'], points.loc[points.grid_id.eq(row.grid_id) & points.is_land].native_flat_index)
            climate_rows.append({'grid_id': row.grid_id, 'raw_min_temperature_c': float(archive['temperature_c'].min()),
                                 'raw_max_temperature_c': float(archive['temperature_c'].max()),
                                 'low_temperature_hours_2025': int((archive['temperature_c'] < -40).sum()),
                                 'high_temperature_hours_2025': int((archive['temperature_c'] > 50).sum())})
    climate = pd.DataFrame(climate_rows)
    climate.to_csv(run / 'validation/grid-weather-range-audit.csv', index=False)
    return {'passed': True, 'selected_grids': len(index), 'ready_grids': len(climate),
            'no_land_grids': int(index.land_points.eq(0).sum()),
            'grids_below_minus40': int(climate.low_temperature_hours_2025.gt(0).sum()),
            'grids_above50': int(climate.high_temperature_hours_2025.gt(0).sum()),
            'raw_min_temperature_c': float(climate.raw_min_temperature_c.min()),
            'raw_max_temperature_c': float(climate.raw_max_temperature_c.max()),
            'national_factor_grids_without_separate_un_m49_entry': int(selected.m49_region.isna().sum()),
            'expected_actual_lps': len(climate) * 198, 'requested_summary_rows': len(index) * 264,
            'input_index_sha256': digest(run / 'inputs/grid-input-index-all.csv')}


def cooling(config):
    own, paths = settings(config)
    run = paths['run_root']
    ranges = pd.read_csv(run / 'validation/grid-weather-range-audit.csv')
    high = ranges.loc[ranges.high_temperature_hours_2025.gt(0), 'grid_id'].tolist()
    low = ranges.nsmallest(2, 'raw_min_temperature_c').grid_id.tolist()
    templates = {scenario: pd.read_parquet(run / 'inputs/templates' / (scenario + '.parquet')) for scenario in own['scenarios']}
    records = []
    for grid in sorted(set(high + low)):
        with np.load(run / 'inputs/grids' / (grid + '.npz')) as archive:
            weather = pd.DataFrame({name: archive[name] for name in ('temperature_c', 'dew_point_c', 'pressure_pa')})
        original = weather.to_numpy(copy=True)
        for scenario, template in templates.items():
            selected = original[template.weather_hour_index.to_numpy(int)]
            temperature = np.clip(selected[:, 0], -40., 50.)
            dew = np.minimum(selected[:, 1], temperature)
            result = facility_frame(template, weather)
            np.testing.assert_array_equal(result.temperature_c, temperature)
            np.testing.assert_array_equal(result.dew_point_c, dew)
            np.testing.assert_array_equal(weather.to_numpy(), original)
            assert result.attrs['low_temperature_clipped_hours'] == int((selected[:, 0] < -40).sum())
            assert result.attrs['high_temperature_clipped_hours'] == int((selected[:, 0] > 50).sum())
            assert result.attrs['temperature_clipped_hours'] == int(((selected[:, 0] < -40) | (selected[:, 0] > 50)).sum())
            assert result.attrs['raw_min_temperature_c'] == float(selected[:, 0].min())
            assert result.attrs['raw_max_temperature_c'] == float(selected[:, 0].max())
            assert np.isfinite(result.pue).all() and result.pue.ge(1).all()
            records.append({'grid_id': grid, 'scenario': scenario, **result.attrs,
                            'minimum_pue': float(result.pue.min()), 'maximum_pue': float(result.pue.max())})
    table = pd.DataFrame(records)
    table.to_csv(run / 'validation/extreme-temperature-real-input-audit.csv', index=False)
    return {'passed': True, 'high_temperature_grids': high, 'minimum_temperature_grids': low,
            'real_grid_scenarios': len(table), 'high_clipped_grid_scenarios': int(table.high_temperature_clipped_hours.gt(0).sum()),
            'low_clipped_grid_scenarios': int(table.low_temperature_clipped_hours.gt(0).sum()),
            'clipped_observation_hours_sum': int(table.temperature_clipped_hours.sum()), 'original_weather_unchanged': True}


def environment():
    packages = ['numpy', 'pandas', 'pyarrow', 'coptpy', 'pvlib', 'zarr', 'geopandas',
                'shapely', 'pyproj', 'duckdb', 'matplotlib', 'cartopy', 'frykit']
    return {'python': sys.version, 'platform': platform.platform(),
            'packages': {name: importlib.metadata.version(name) for name in packages}}


def report_reader(config, source):
    import duckdb
    import resource
    from report_global_flex_econ import load_shards

    own, paths = settings(config)
    folder = source / 'shards'
    files = sorted(folder.glob('*.jsonl'))[:80]
    points = pd.read_parquet(paths['run_root'] / 'inputs/grid-points.parquet')
    no_land = points.groupby('grid_id').is_land.sum().loc[lambda value: value.eq(0)].index
    for grid in no_land:
        candidates = sorted(folder.glob(grid + '-*.jsonl'))
        if candidates:
            files.extend(candidates)
            break
    files = sorted(set(files))
    rows = [json.loads(line) for path in files for line in path.read_text('utf-8').splitlines()]
    expected = pd.DataFrame(rows)
    assert len(files) > 64 and expected.status.eq('input_no_land_points').any()
    database = paths['run_root'] / 'validation/report-reader-check.duckdb'
    connection = duckdb.connect(str(database), config={'memory_limit': '2GB', 'threads': 2, 'preserve_insertion_order': False})
    load_shards(connection, [str(path) for path in files], 'runs')
    actual = connection.sql('SELECT * FROM runs').df()
    for name in actual.select_dtypes(include=['datetime', 'datetimetz']).columns:
        expected[name] = pd.to_datetime(expected[name], utc=True)
    keys = ['grid_id', 'scenario', 'case_key']
    expected = expected.set_index(keys).sort_index().sort_index(axis=1)
    actual = actual.set_index(keys).sort_index().sort_index(axis=1)
    expected = expected.astype(object).where(expected.notna(), None)
    actual = actual.astype(object).where(actual.notna(), None)
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    connection.close()
    return {'passed': True, 'real_shards': len(files), 'rows': len(rows), 'columns': len(expected.columns),
            'all_fields_compared': True, 'null_representation_normalized': True,
            'input_no_land_points_preserved': True, 'duckdb_memory_limit': '2GB',
            'process_peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def city_statistics(config):
    own, paths = settings(config)
    run = paths['run_root']
    data = pd.read_csv(run / 'validation/city-hourly-differences.csv')
    hours = pd.read_csv(run / 'inputs/template-index.csv').set_index('scenario').hours
    np.testing.assert_array_equal(data.hours, data.scenario.map(hours))
    rows = []
    for (coordinates, variable), current in data.groupby(['coordinate_kind', 'variable']):
        rows.append({'coordinate_kind': coordinates, 'variable': variable, 'city_scenario_cases': len(current),
                     'hourly_observations': int(current.hours.sum()),
                     'maximum_absolute_difference': float(current.max_abs_difference.max()),
                     'mean_absolute_difference_hour_weighted': float(np.average(current.mean_abs_difference, weights=current.hours)),
                     'mean_case_mae': float(current.mean_abs_difference.mean()),
                     'maximum_case_mae': float(current.mean_abs_difference.max())})
    definition = '各城市情境MAE按原生小时数加权，即合并所有小时的MAE'
    summary_path = run / 'validation/city-validation-summary.json'
    summary = json.loads(summary_path.read_text('utf-8'))
    summary['variables'] = [{'coordinate_kind': row['coordinate_kind'], 'variable': row['variable'],
                             'max_abs_difference': row['maximum_absolute_difference'],
                             'mean_abs_difference': row['mean_absolute_difference_hour_weighted'],
                             'hourly_observations': row['hourly_observations']} for row in rows]
    summary['mean_definition'] = definition
    write_json(summary_path, summary)
    return {'passed': True, 'mean_definition': definition, 'values': rows,
            'source_sha256': digest(run / 'validation/city-hourly-differences.csv')}


def hourly(config, source):
    import pyarrow.parquet as pq

    own, paths = settings(config)
    grids = json.loads((source / 'hourly-grid-selection.json').read_text('utf-8'))['grid_ids']
    assert len(grids) == len(set(grids)) == 10
    scenarios = ['D-TRN', 'D-SAI', 'A-HPC', 'D-INF']
    templates = {name: pd.read_parquet(paths['run_root'] / 'inputs/templates' / (name + '.parquet'),
                                     columns=['source_hour_index']) for name in scenarios}
    expected = {}
    for grid in grids:
        for scenario in scenarios:
            for alpha in (None, 1.):
                for boundary, flex in [('planning', 'none'), ('planning', 'both'), ('fixed', 'both')]:
                    expected[f'{grid}-{scenario}-a{alpha}-{boundary}-{flex}'] = scenario
    assert {path.stem for path in (source / 'hourly').glob('*.parquet')} == set(expected)
    assert {path.stem for path in (source / 'audits').glob('*.json')} == set(expected)
    maximum = 0.
    for stem, scenario in expected.items():
        path = source / 'hourly' / (stem + '.parquet')
        assert pq.read_metadata(path).num_rows == len(templates[scenario])
        actual = pq.read_table(path, columns=['source_hour_index']).column(0).to_numpy()
        np.testing.assert_array_equal(actual, templates[scenario].source_hour_index.to_numpy())
        audit = json.loads((source / 'audits' / (stem + '.json')).read_text('utf-8'))
        assert audit['passed'] and audit['maximum_violation'] <= audit['tolerance']
        maximum = max(maximum, audit['maximum_violation'])
    return {'passed': True, 'grids': grids, 'hourly_files': len(expected), 'audit_files': len(expected),
            'native_clock_and_rows_match_templates': True, 'maximum_dispatch_violation': maximum}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['pilot', 'inputs', 'runtime', 'cooling', 'environment', 'report-reader', 'city-statistics', 'hourly'])
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('global-flex-econ-20261103.json'))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    own, paths = settings(args.config)
    run = paths['run_root']
    if args.stage == 'pilot':
        result = pilot(args.config, args.report or run / 'report-pilot-confirmed')
    elif args.stage == 'inputs':
        result = inputs(args.config)
    elif args.stage == 'runtime':
        result = compare_runtime(run)
    elif args.stage == 'cooling':
        result = cooling(args.config)
    elif args.stage == 'report-reader':
        result = report_reader(args.config, args.report or run / 'full')
    elif args.stage == 'city-statistics':
        result = city_statistics(args.config)
    elif args.stage == 'hourly':
        result = hourly(args.config, args.report or run / 'full')
    else:
        result = environment()
    result.update(utc=utc_now(), auditor_sha256=digest(Path(__file__)))
    write_json(run / 'validation' / ('global-' + args.stage + '-audit.json'), result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
