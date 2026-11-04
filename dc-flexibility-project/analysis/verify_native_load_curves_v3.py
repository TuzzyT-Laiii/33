import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_typical_load_curves_v3 import checks, read_baseline, weather_inputs
from compute_load.model import sha256
from load_curve_scenarios_v3 import DAYS, TDP, catalogue, combinations
from native_load_inputs_v3 import RUN, POWER_COLUMNS, native_sources, save_json


def verify(args):
    project = args.project.resolve()
    output = project / 'results' / RUN
    manifest = json.loads((output / 'manifest.json').read_text('utf-8'))
    summary = pd.read_csv(output / 'window-summary-all-locations.csv')
    aggregate = pd.read_csv(output / 'placement-aggregate-all-locations.csv')
    native = pd.read_csv(output / 'native-it-summary.csv').set_index('scenario_id')
    source, metadata = native_sources(project, args.source_project, args.mit_hourly)
    expected_ids = {f'{key}-{use}-{scale}MW' for key, use in combinations() for scale in (1, 10, 100, 500)}
    if manifest['scenario_count'] != 44 or set(native.index) != expected_ids:
        raise ValueError('44情境覆盖不完整')
    configs = catalogue(project)
    mit = json.loads((output / 'mit-calibration.json').read_text('utf-8'))
    if mit['partial_data'] or sha256(output / 'mit-calibration.json') != manifest['mit_calibration_sha256']:
        raise ValueError('六月完成版标定不符')
    baseline, _ = read_baseline(project, args.baseline)
    weather = {(row['location_id'], row['year']): row['weather'].set_index('time_utc') for row in
               weather_inputs(project, args.weather, baseline, args.reference_only)}
    source_frames = {}
    for relative, digest in manifest['source_hourly_sha256'].items():
        path = project / relative
        if sha256(path) != digest:
            raise ValueError('原生IT文件哈希不符：' + relative)
        frame = pd.read_parquet(path)
        scenario, use, key = frame.scenario_id.iloc[0], frame.use.iloc[0], frame.server_type.iloc[0]
        if len(frame) != DAYS[use] * 24 or frame.source_hour_index.duplicated().any():
            raise ValueError('原生小时数或唯一性错误')
        np.testing.assert_array_equal(frame.source_hour_index, np.arange(DAYS[use] * 24))
        for column in ('source_clock', 'u_cpu_proxy', 'allocated_gpu_fraction', 'source_unresolved'):
            np.testing.assert_array_equal(frame[column], source[use][column])
        np.testing.assert_allclose(frame.it_mw, frame.base_it_mw + frame.cpu_it_mw + frame.gpu_it_mw, rtol=1e-12)
        if use == 'SAI':
            fraction = source[use].allocated_gpu_fraction.to_numpy()
            expected = int(frame.unit_count.iloc[0]) * configs[key]['gpu_devices_per_unit'] * TDP[key] * (
                fraction * mit['measured_power_to_tdp_ratio'] + (1 - fraction) * mit['idle_W_per_gpu'] / 250) / 1e6
            np.testing.assert_allclose(frame.gpu_it_mw, expected, rtol=1e-12)
        for metric, reducer in [('mean', np.mean), ('peak', np.max), ('minimum', np.min)]:
            np.testing.assert_allclose(native.loc[scenario, metric + '_it_mw'], reducer(frame.it_mw), rtol=1e-12)
        np.testing.assert_allclose(native.loc[scenario, 'window_it_mwh'], frame.it_mw.sum(), rtol=1e-12)
        source_frames[scenario] = frame
    index_columns = ['location_id', 'weather_year', 'scenario_id', 'placement']
    if summary.duplicated(index_columns).any():
        raise ValueError('放置汇总存在重复')
    lookup = summary.set_index(index_columns)
    count, max_balance, max_energy_error = 0, 0., 0.
    visited = set()
    for relative, digest in manifest['hourly_sha256'].items():
        if args.reference_only and '/hourly/' not in relative:
            continue
        path = project / relative
        if sha256(path) != digest:
            raise ValueError('气象匹配小时文件哈希不符：' + relative)
        frame = pd.read_parquet(path)
        key = tuple(frame[name].iloc[0] for name in index_columns)
        row = lookup.loc[key]
        original = source_frames[key[2]]
        use = frame.use.iloc[0]
        dated = metadata[use]['has_real_dates']
        if dated:
            clock = pd.DatetimeIndex(pd.to_datetime(original.source_clock, utc=True))
            year = int(key[1])
            mapped = []
            for value in clock:
                if value.month == 2 and value.day == 29 and not pd.Timestamp(year=year, month=12, day=31).is_leap_year:
                    mapped.append(value.replace(year=year, day=28))
                else:
                    mapped.append(value.replace(year=year))
            expected_hours = len(original)
            target = pd.DatetimeIndex(mapped)
        else:
            start = pd.Timestamp(year=int(key[1]), month=int(key[3][1:]), day=1, tz='UTC')
            remaining = int((pd.Timestamp(year=int(key[1]) + 1, month=1, day=1, tz='UTC') - start).total_seconds() / 3600)
            expected_hours = min(len(original), remaining)
            target = pd.date_range(start, periods=expected_hours, freq='h')
        if len(frame) != expected_hours or int(row.hours) != expected_hours:
            raise ValueError('气象放置长度错误')
        np.testing.assert_array_equal(frame.source_hour_index, np.arange(expected_hours))
        if not pd.DatetimeIndex(frame.weather_time_utc).equals(target):
            raise ValueError('源月日小时或月初放置对应错误')
        selected_weather = weather[(key[0], int(key[1]))].loc[target]
        np.testing.assert_allclose(frame.temperature_c, selected_weather.temperature_c, rtol=0, atol=1e-12)
        dew = selected_weather['input_dew_point_c'] if 'input_dew_point_c' in selected_weather else selected_weather.dew_point_c
        np.testing.assert_allclose(frame.input_dew_point_c, dew, rtol=0, atol=1e-12)
        if 'pressure_pa' in selected_weather:
            np.testing.assert_allclose(frame.pressure_pa, selected_weather.pressure_pa, rtol=0, atol=1e-10)
        for column in ('it_mw', 'base_it_mw', 'cpu_it_mw', 'gpu_it_mw'):
            np.testing.assert_array_equal(frame[column], original[column].iloc[:expected_hours])
        checks(frame)
        residual = frame.facility_mw - frame.it_mw - frame.cooling_mw - frame.other_mw
        max_balance = max(max_balance, float(residual.abs().max()))
        for component in POWER_COLUMNS:
            values = frame[component + '_mw']
            for metric, reducer in [('mean', np.mean), ('peak', np.max), ('minimum', np.min)]:
                np.testing.assert_allclose(row[f'{metric}_{component}_mw'], reducer(values), rtol=1e-11, atol=1e-10)
            error = abs(row[f'window_{component}_mwh'] - values.sum())
            max_energy_error = max(max_energy_error, float(error))
            np.testing.assert_allclose(row[f'window_{component}_mwh'], values.sum(), rtol=1e-11, atol=1e-8)
            if component in ('it', 'facility'):
                for dimension, column in [('diurnal', 'hour_of_source_day'), ('weekly', 'hour_of_source_week')]:
                    if use == 'CLD' and dimension == 'weekly':
                        continue
                    grouped = values.groupby(frame[column]).mean()
                    expected_ratio = grouped.max() / grouped.min()
                    np.testing.assert_allclose(row[f'{component}_{dimension}_peak_valley'], expected_ratio, rtol=1e-12)
        if use == 'CLD' and (bool(row.weekly_shape_available) or pd.notna(row.it_weekly_peak_valley) or pd.notna(row.facility_weekly_peak_valley)):
            raise ValueError('CLD五天不能报告周内形状')
        count += len(frame)
        visited.add(key)
    selected_summary = summary.loc[summary.location_id.eq('reference-grid')] if args.reference_only else summary
    if len(visited) != len(selected_summary) or count != int(selected_summary.hours.sum()):
        raise ValueError('小时文件与汇总覆盖不一致')
    for row in aggregate.to_dict('records'):
        group = summary.loc[summary.location_id.eq(row['location_id']) & summary.weather_year.eq(row['weather_year']) & summary.scenario_id.eq(row['scenario_id'])]
        if len(group) != (1 if row['has_real_dates'] else 12):
            raise ValueError('放置数量错误')
        for component in ('it', 'facility'):
            for metric in ('mean', 'peak', 'minimum'):
                values = group[f'{metric}_{component}_mw']
                for statistic in ('median', 'min', 'max'):
                    np.testing.assert_allclose(row[f'placement_{metric}_{component}_mw_{statistic}'], getattr(values, statistic)(), rtol=1e-12)
    week = pd.read_csv(output / 'typical-week-placements.csv', usecols=['scenario_id'])
    if week.scenario_id.str.contains('-CLD-').any():
        raise ValueError('周形状表含CLD')
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed', 'reference_only': args.reference_only,
        'native_scenarios': len(native), 'source_hours_checked': int(native.native_hours.sum()), 'weather_matched_rows_checked': count,
        'placement_files_checked': len(visited), 'max_facility_balance_residual_mw': max_balance,
        'max_window_energy_error_mwh': max_energy_error, 'independent_checks': [
            '原始来源SHA与原生小时序列、无重复无补齐', '独立SAI功率公式', '真实日期月日小时对应与跨年源顺序',
            '月初12放置及年末截断', '目标闰年不新增源小时', '温度/气压逐小时查表', 'IT与设施分项守恒',
            '窗口平均/峰值/最小值/电量积分', '12放置中位数与极值', 'CLD五日不报周形状']}
    save_json(output / ('validation-reference.json' if args.reference_only else 'validation.json'), result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source-project', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--weather', type=Path, required=True)
    parser.add_argument('--mit-hourly', type=Path, required=True)
    parser.add_argument('--reference-only', action='store_true')
    verify(parser.parse_args())
