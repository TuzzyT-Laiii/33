import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from compute_load.cooling import CoolingModel
from compute_load.model import sha256 as digest


RUN = 'eight-city-native-flex-20261102'
SEREN = 'typical-load-curves-20261101-v4-seren'
COMBINED = 'typical-load-curves-20261101-v3-native-combined'
FLEX = 'flex-class-20261101'
CLASSES = ['刚性', '可推迟·整体', '可推迟·可拆分', '未知', '基础与空载 IT']
HOURS = {'HPC': 7416, 'CLD': 120, 'SAI': 5688, 'TRN': 2057, 'INF': 4440}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(project, inputs):
    inputs.mkdir(parents=True, exist_ok=True)
    source = project / 'results'
    combined = json.loads((source / COMBINED / 'manifest.json').read_text('utf-8'))
    seren = json.loads((source / 'trn-seren-20261101/calculation-audit.json').read_text('utf-8'))
    weather = json.loads((source / 'eight-city-phase-a-20261029/calculation-manifest.json').read_text('utf-8'))['weather_sources']
    names = {row['city_id']: row for row in json.loads((project / 'analysis/eight_city_locations.json').read_text('utf-8'))}
    grids_path = source / 'eight-city-phase-a-20261029/city-coordinates-climate-resource.csv'
    grids = pd.read_csv(grids_path).set_index('city_id')
    cases = pd.read_csv(source / SEREN / 'window-summary-all-locations.csv', float_precision='round_trip')
    cases = cases.loc[cases.reference_it_mw.eq(10) & cases.location_id.ne('reference-grid')].copy()
    paths, hashes = [], []
    for row in cases.to_dict('records'):
        if row['use'] == 'TRN':
            relative = f'hourly/{row["location_id"]}/main/{row["scenario_id"]}-{row["weather_year"]}.parquet'
            paths.append('results/trn-seren-20261101/' + relative)
            hashes.append(seren['hourly_sha256'][relative])
        else:
            relative = f'data/processed/{COMBINED}/{row["location_id"]}/{row["scenario_id"]}-{row["weather_year"]}-{row["placement"]}.parquet'
            paths.append(relative)
            hashes.append(combined['hourly_sha256'][relative])
    cases['upstream_path'], cases['upstream_sha256'] = paths, hashes
    assert len(cases) == 1320 and cases.upstream_path.is_unique
    cases.to_csv(inputs / 'cases.csv', index=False, encoding='utf-8-sig')
    records = []
    for row in weather:
        city = row['city_id']
        linked = [x for x in seren['weather'] if x['location_id'] == city]
        assert len(linked) == 3 and all(x['source']['sha256'] == row['sha256'] for x in linked)
        records.append({**names[city], 'weather_file': row['path'].replace('\\', '/').split('/')[-1],
            'weather_sha256': row['sha256'], 'method': row['method'],
            'grid_latitude': float(grids.loc[city, 'grid_latitude']), 'grid_longitude': float(grids.loc[city, 'grid_longitude'])})
    save(inputs / 'weather-index.json', records)
    shutil.copyfile(source / FLEX / 'shares-10MW.csv', inputs / 'fifth-shares.csv')
    day = pd.read_csv(source / FLEX / 'selected-days/D-TRN-hourly.csv', float_precision='round_trip')
    parts = pd.read_csv(source / FLEX / 'selected-days/D-TRN-by-task.csv', float_precision='round_trip')
    fractions = day[['source_hour_index', 'source_clock', 'hour']].copy()
    for label in CLASSES:
        fractions[label] = parts.loc[parts.task_type.eq(label)].sort_values('hour').it_mw.to_numpy() / day.it_mw
    np.testing.assert_allclose(fractions[CLASSES].sum(axis=1), 1., atol=1e-12)
    fractions.to_csv(inputs / 'training-day-fractions.csv', index=False, encoding='utf-8-sig')
    source_paths = [source / SEREN / 'window-summary-all-locations.csv', source / FLEX / 'shares-10MW.csv',
        source / COMBINED / 'manifest.json', source / 'trn-seren-20261101/calculation-audit.json',
        source / 'eight-city-phase-a-20261029/calculation-manifest.json', source / FLEX / 'selected-days/D-TRN-by-task.csv',
        source / FLEX / 'selected-days/D-TRN-hourly.csv', grids_path, project / 'analysis/compute_load/assets/cooling-20261028-v2.json']
    expected_cooling = digest(source_paths[-1])
    assert expected_cooling == seren['cooling_parameters_sha256'] == combined['cooling']['parameters_sha256']
    save(inputs / 'provenance.json', {'base_commit': 'f1588457559b2e2e5f5c9d2216c1a586e0d1654a',
        'source_sha256': {p.relative_to(project).as_posix(): digest(p) for p in source_paths},
        'input_sha256': {p.name: digest(p) for p in inputs.iterdir() if p.is_file() and p.name != 'provenance.json'},
        'cooling_parameters_sha256': expected_cooling,
        'reuse': '非训练1248个10MW窗口复用原组合结果，训练72个窗口复用#38 Seren main；每份小时与冻结气象重新核验'})


def collect(inputs, source_root, weather_root, output):
    output.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((inputs / 'provenance.json').read_text('utf-8'))
    assert all(digest(inputs / p) == value for p, value in provenance['input_sha256'].items())
    asset = Path(__file__).parent / 'compute_load/assets/cooling-20261028-v2.json'
    assert digest(asset) == provenance['cooling_parameters_sha256']
    cases = pd.read_csv(inputs / 'cases.csv', float_precision='round_trip')
    shares = pd.read_csv(inputs / 'fifth-shares.csv', float_precision='round_trip').set_index('scenario')
    day = pd.read_csv(inputs / 'training-day-fractions.csv', float_precision='round_trip')
    cities = json.loads((inputs / 'weather-index.json').read_text('utf-8'))
    climate_rows, cache, weather_receipts = [], {}, []
    expected = pd.date_range('2023-01-01', '2026-01-01', freq='h', inclusive='left', tz='UTC')
    for row in cities:
        path = weather_root / row['weather_file']
        assert digest(path) == row['weather_sha256'], path
        full = pd.read_parquet(path)
        clock = pd.DatetimeIndex(pd.to_datetime(full.time_utc, utc=True))
        assert clock.equals(expected) and full[['t2m', 'd2m', 'sp']].notna().all().all()
        t, dew = full.t2m.to_numpy(float) - 273.15, full.d2m.to_numpy(float) - 273.15
        cache[row['city_id']] = pd.DataFrame({'temperature_c': t, 'dew_point_c': dew, 'pressure_pa': full.sp.to_numpy(float)}, index=clock)
        climate_rows.append({'city_id': row['city_id'], 'city_zh': row['city_zh'], 'hours': len(full),
            'mean_temperature_c': float(t.mean()), 'mean_dew_point_c': float(dew.mean()),
            'hot_30c_pct': float(100 * (t >= 30).mean()), 'humid_dew20c_pct': float(100 * (dew >= 20).mean()),
            'mean_pressure_pa': float(full.sp.mean()), 'grid_latitude': row['grid_latitude'],
            'grid_longitude': row['grid_longitude'], 'grid_note': row.get('grid_note', '')})
        weather_receipts.append({'city_id': row['city_id'], 'file': str(path), 'sha256': digest(path), 'hours': len(full),
            'start_utc': str(clock[0]), 'end_utc': str(clock[-1])})
    verified, typical, recooling = [], [], []
    max_error, total_hours = 0., 0
    for number, row in enumerate(cases.to_dict('records')):
        path = source_root / row['upstream_path']
        assert digest(path) == row['upstream_sha256'], path
        frame = pd.read_parquet(path)
        clock = pd.DatetimeIndex(pd.to_datetime(frame.time_utc, utc=True))
        assert len(frame) == row['hours'] and row['native_hours'] == HOURS[row['use']]
        weather = cache[row['location_id']].loc[clock]
        original_dew = frame['input_dew_point_c'] if 'input_dew_point_c' in frame else frame.dew_point_c
        np.testing.assert_allclose(frame.temperature_c, weather.temperature_c, atol=1e-9, rtol=0)
        np.testing.assert_allclose(original_dew, weather.dew_point_c, atol=1e-9, rtol=0)
        np.testing.assert_allclose(frame.pressure_pa, weather.pressure_pa, atol=1e-8, rtol=0)
        assert not frame.pressure_assumed.any()
        assert frame.cooling_version.eq('cooling-20261028-v2').all() and frame.cooling_type.eq('air').all()
        if row['has_real_dates']:
            source_clock = pd.DatetimeIndex(pd.to_datetime(frame.source_clock, utc=True))
            assert np.array_equal(clock.month, source_clock.month) and np.array_equal(clock.day, source_clock.day) and np.array_equal(clock.hour, source_clock.hour)
            assert len(frame) == HOURS[row['use']]
        else:
            month = int(row['placement'][1:])
            begin, end = pd.Timestamp(year=int(row['weather_year']), month=month, day=1, tz='UTC'), pd.Timestamp(year=int(row['weather_year'])+1, month=1, day=1, tz='UTC')
            hours = min(HOURS[row['use']], int((end-begin).total_seconds()/3600))
            assert len(frame) == hours and clock.equals(pd.date_range(begin, periods=hours, freq='h'))
        for component in ('it', 'facility', 'cooling', 'other'):
            values = frame[component + '_mw']
            np.testing.assert_allclose([values.mean(), values.max(), values.min(), values.sum()],
                [row['mean_' + component + '_mw'], row['peak_' + component + '_mw'], row['minimum_' + component + '_mw'], row['window_' + component + '_mwh']], rtol=1e-11, atol=1e-8)
        np.testing.assert_allclose(frame.facility_mw, frame.it_mw + frame.cooling_mw + frame.other_mw, atol=1e-9, rtol=1e-12)
        record = dict(row)
        record['mean_non_it_mw'] = row['mean_facility_mw'] - row['mean_it_mw']
        scenario = row['scenario_id'].removesuffix('-10MW')
        for label in CLASSES:
            record[label] = row['mean_it_mw'] * shares.loc[scenario, label] / 100
        np.testing.assert_allclose(sum(record[label] for label in CLASSES) + record['mean_non_it_mw'], row['mean_facility_mw'], atol=1e-9)
        if scenario == 'D-TRN':
            selected = frame.set_index('source_hour_index').loc[day.source_hour_index].reset_index()
            assert len(selected) == 24
            for hour in range(24):
                current = selected.iloc[hour]
                item = {'city_id': row['location_id'], 'city_zh': row['location_zh'], 'year': row['weather_year'],
                    'scenario_zh': row['scenario_zh'], 'source_date': '2023-05-31', 'hour_utc': hour,
                    'weather_time_utc': str(current.time_utc), 'it_mw': float(current.it_mw), 'facility_mw': float(current.facility_mw),
                    'temperature_c': float(current.temperature_c), 'dew_point_c': float(current.dew_point_c),
                    '冷却等非 IT': float(current.facility_mw - current.it_mw)}
                for label in CLASSES:
                    item[label] = float(current.it_mw * day.iloc[hour][label])
                typical.append(item)
            recomputed = CoolingModel().calculate(weather.reset_index(names='time_utc'), frame.it_mw.to_numpy(),
                float(frame.installed_reference_it_mw.iloc[0]), full_year=False, allow_gaps=True)
            error = float(np.abs(recomputed.facility_mw.to_numpy() - frame.facility_mw.to_numpy()).max())
            assert error < 1e-8
            max_error = max(max_error, error)
            recooling.append({'city_id': row['location_id'], 'year': row['weather_year'], 'hours': len(frame), 'facility_max_absolute_difference_mw': error})
        verified.append(record)
        total_hours += len(frame)
        if number % 25 == 0:
            state = {'utc': datetime.now(timezone.utc).isoformat(), 'verified_files': number+1, 'total_files': len(cases), 'hours': total_hours}
            save(output / 'progress.json', state)
            print(json.dumps(state), flush=True)
    table = pd.DataFrame(verified)
    annual, central_windows = [], []
    for (city, year, sid), group in table.groupby(['location_id', 'weather_year', 'scenario_id'], sort=False):
        ordered = group.sort_values(['mean_facility_mw', 'placement'], kind='stable')
        n = len(ordered)
        middle = ordered.iloc[[(n-1)//2, n//2]].drop_duplicates('placement')
        numeric = ['mean_it_mw', 'mean_facility_mw', 'mean_non_it_mw', 'mean_cooling_mw', 'mean_other_mw', *CLASSES]
        record = {'city_id': city, 'city_zh': group.location_zh.iloc[0], 'year': year, 'scenario_id': sid, 'scenario_zh': group.scenario_zh.iloc[0],
            **{name: float(middle[name].mean()) for name in numeric}, 'window_pue': float(group.window_pue.median()),
            'facility_range_min_mw': float(group.mean_facility_mw.min()), 'facility_range_max_mw': float(group.mean_facility_mw.max()),
            'pue_range_min': float(group.window_pue.min()), 'pue_range_max': float(group.window_pue.max()), 'placements': n}
        np.testing.assert_allclose(record['mean_facility_mw'], group.mean_facility_mw.median(), atol=1e-12)
        annual.append(record)
        for item in middle.to_dict('records'):
            central_windows.append({'city_id': city, 'year': year, 'scenario_id': sid, 'placement': item['placement'],
                'hours': item['hours'], 'weight_within_year': 1/len(middle), 'facility_mean_mw': item['mean_facility_mw']})
    years = pd.DataFrame(annual)
    city_records = []
    for (city, sid), group in years.groupby(['city_id', 'scenario_id'], sort=False):
        assert len(group) == 3
        numeric = ['mean_it_mw', 'mean_facility_mw', 'mean_non_it_mw', 'mean_cooling_mw', 'mean_other_mw', 'window_pue', *CLASSES]
        city_records.append({'city_id': city, 'city_zh': group.city_zh.iloc[0], 'scenario_id': sid, 'scenario_zh': group.scenario_zh.iloc[0],
            **{name: float(group[name].mean()) for name in numeric}, 'facility_range_min_mw': float(group.facility_range_min_mw.min()),
            'facility_range_max_mw': float(group.facility_range_max_mw.max()), 'pue_range_min': float(group.pue_range_min.min()), 'pue_range_max': float(group.pue_range_max.max())})
    for name, data in [('case-summary', table), ('year-summary', years), ('city-summary', pd.DataFrame(city_records)),
        ('central-windows', pd.DataFrame(central_windows)), ('climate-summary', pd.DataFrame(climate_rows).sort_values('mean_temperature_c')),
        ('typical-day-city-year', pd.DataFrame(typical))]:
        data.to_csv(output / (name + '.csv'), index=False, encoding='utf-8-sig')
    save(output / 'verification.json', {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed', 'hourly_files_verified': len(verified),
        'hour_rows_verified': total_hours, 'weather_files_verified': weather_receipts, 'cooling_recomputed': recooling,
        'cooling_maximum_difference_mw': max_error, 'cooling_parameters_sha256': digest(asset), 'old_results_reused_without_writes': True,
        'composition': '四类和基础空载按第五章固定原生IT份额归因城市IT，冷却等非IT单列；典型日按第五章同日逐小时IT份额归因。',
        'aggregation': '主表每年按设施均值取中位放置，同批指标配对汇总，再等权平均三年；PUE每年按窗口电量比取放置中位数后平均三年；范围为三年所有放置极值。',
        'upstream_hourly_sha256': dict(zip(cases.upstream_path, cases.upstream_sha256)), 'input_provenance': provenance,
        'code_sha256': digest(Path(__file__)), 'model_code_sha256': {name: digest(Path(__file__).parent / 'compute_load' / name) for name in ('cooling.py', 'model.py')}})
    save(output / 'progress.json', {'status': 'complete', 'hourly_files': len(verified), 'hour_rows': total_hours, 'city_scenarios': len(city_records)})
    print(json.dumps({'status': 'complete', 'files': len(verified), 'city_scenarios': len(city_records)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--weather-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.project, args.inputs)
    else:
        collect(args.inputs, args.source_root, args.weather_root, args.output)
