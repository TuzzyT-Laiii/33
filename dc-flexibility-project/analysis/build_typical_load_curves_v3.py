import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alibaba_cpu_profile import load_alibaba_source
from annual_task_profiles import replay_hourly
from asi_inference_profiles import load_asi_sources
from compute_load.cooling import CoolingModel
from compute_load.model import sha256
from load_curve_scenarios_v3 import RUN, SCALES, LEGACY, DEMAND, DAYS, TDP, catalogue, combinations, identity, power, size


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def note(output, state, **fields):
    item = {'utc': datetime.now(timezone.utc).isoformat(), 'state': state, **fields}
    save_json(output / 'progress.json', item)
    with (output / 'progress.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(json.dumps(item, ensure_ascii=False), flush=True)


def read_baseline(project, directory):
    manifest = json.loads((project / 'results/typical-load-curves-20261029-v2/manifest.json').read_text('utf-8'))
    frames, inputs = {}, []
    for key, use in combinations():
        if use == 'CLD':
            continue
        method = 'cpu_spec' if key == 'A' else 'ratio_p50'
        path = directory / f'{LEGACY[key]}-{DEMAND[use]}-{method}-central.parquet'
        digest = sha256(path)
        if digest != manifest['processed_sha256'][path.name]:
            raise ValueError('v2 小时输入 SHA-256 不符：' + path.name)
        frame = pd.read_parquet(path)
        if len(frame) != 8760:
            raise ValueError('v2 小时数错误')
        frames[(key, use)] = frame
        inputs.append({'scenario': f'{key}-{use}', 'legacy_input_file': path.name, 'sha256': digest})
    return frames, inputs


def weather_inputs(project, directory, baseline, reference_only):
    frame = baseline[('A', 'HPC')]
    if not frame.pressure_assumed.all() or not frame.pressure_pa.eq(101325.).all():
        raise ValueError('参考网格气压与v2固定假设不符')
    columns = ['grid_id', 'time_utc', 'temperature_c', 'relative_humidity_pct', 'dew_point_c', 'input_dew_point_c']
    weather = frame[columns].copy()
    records = [{'location_id': 'reference-grid', 'location_zh': '参考网格', 'year': 2025, 'weather': weather,
                'source': {'source': 'v2逐小时文件内固定天气', 'grid': 'g05_r100_c592',
                           'pressure_status': '沿用v2的101325Pa假设', 'source_weather_sha256':
                           json.loads((project / 'results/typical-load-curves-20261029-v2/manifest.json').read_text('utf-8'))['weather_file_sha256']}}]
    if reference_only:
        return records
    manifest = json.loads((project / 'results/eight-city-phase-a-20261029/calculation-manifest.json').read_text('utf-8'))
    names = {row['city_id']: row['city_zh'] for row in json.loads((project / 'analysis/eight_city_locations.json').read_text('utf-8'))}
    if len(manifest['weather_sources']) != 8:
        raise ValueError('八城市冻结天气清单缺失')
    for source in manifest['weather_sources']:
        name = source['path'].replace('\\', '/').split('/')[-1]
        path = directory / name
        if sha256(path) != source['sha256']:
            raise ValueError('八城市天气 SHA-256 不符：' + name)
        data = pd.read_parquet(path)
        data.time_utc = pd.to_datetime(data.time_utc, utc=True)
        expected = pd.date_range('2023-01-01', '2026-01-01', freq='h', inclusive='left', tz='UTC')
        if not pd.DatetimeIndex(data.time_utc).equals(expected):
            raise ValueError('城市天气三年索引不完整')
        for year in (2023, 2024, 2025):
            selected = data.loc[data.time_utc.dt.year.eq(year)].reset_index(drop=True)
            weather = pd.DataFrame({'time_utc': selected.time_utc,
                                    'temperature_c': selected.t2m.to_numpy(dtype=float) - 273.15,
                                    'dew_point_c': selected.d2m.to_numpy(dtype=float) - 273.15,
                                    'pressure_pa': selected.sp.to_numpy(dtype=float)})
            records.append({'location_id': source['city_id'], 'location_zh': names[source['city_id']],
                            'year': year, 'weather': weather,
                            'source': {'input_file': name, 'sha256': source['sha256'], 'method': source['method']}})
    return records


def demands(project, baseline, times):
    result, mappings = {}, {}
    length = len(times)
    cld = load_alibaba_source(project)
    cpu_cld, cld_map = replay_hourly(cld, times)
    components = pd.read_csv(project / 'results/alibaba-cpu-cld-20261031/complete-component-window.csv')
    if not np.array_equal(components.relative_hour, np.arange(48, 168)):
        raise ValueError('CLD 业务拆分须与120小时需求窗口相同')
    cld_map['online_share'] = components.online_share_of_machine.to_numpy()[cld_map.source_hour_index]
    cld_map['batch_share'] = components.batch_share_of_machine.to_numpy()[cld_map.source_hour_index]
    cld_map['residual_share'] = components.residual_share_of_machine.to_numpy()[cld_map.source_hour_index]
    mappings['A-CLD'] = cld_map
    sources = load_asi_sources(project)
    inf, inf_map = replay_hourly(sources['asi_gpu'], times)
    inf_map['online_fraction'] = replay_hourly(sources['asi_online_gpu'], times)[0].fraction
    inf_map['offline_fraction'] = replay_hourly(sources['asi_offline_gpu'], times)[0].fraction
    anomalies = pd.read_csv(project / 'results/asi-185day-workload-types-20261031/inventory-attribution-anomalies.csv')
    inf_map['all_business_inventory_excess'] = inf_map.source_hour_index.isin(anomalies.hour_index)
    mappings['INF'] = inf_map
    for key, use in combinations():
        if use == 'CLD':
            cpu, gpu = cpu_cld.fraction, np.zeros(length)
            mapping = cld_map
        elif use == 'INF':
            gpu = inf.fraction
            cpu = .25 * gpu
            mapping = inf_map
        else:
            source = baseline[(key, use)]
            position = np.arange(length) % (DAYS[use] * 24)
            cpu = source.u_cpu_proxy.to_numpy()[position]
            gpu = source.allocated_gpu_fraction.to_numpy()[position]
            mapping = pd.DataFrame({'time_utc': times, 'source_hour_index': position,
                                    'source_quality_flag': 'allocation_proxy'})
        result[(key, use)] = (cpu, gpu, mapping)
    return result, mappings


def ratio(values):
    minimum = float(np.min(values))
    return float(np.max(values) / minimum) if minimum > 0 else None


def summarize(frame, fields):
    time = pd.DatetimeIndex(frame.time_utc)
    result = {**fields, 'hours': len(frame), 'unit_count': int(frame.unit_count.iloc[0]),
              'installed_reference_it_mw': float(frame.installed_reference_it_mw.iloc[0]),
              'source_quality_flagged_hours': int(frame.source_unresolved.sum()),
              'all_business_inventory_excess_hours': int(frame.all_business_inventory_excess.sum()),
              'annual_pue': float(frame.facility_mw.sum() / frame.it_mw.sum())}
    for name in ('it', 'cooling', 'other', 'facility', 'base_it', 'cpu_it', 'gpu_it'):
        values = frame[name + '_mw']
        result['annual_' + name + '_mwh'] = float(values.sum())
        result['mean_' + name + '_mw'] = float(values.mean())
        result['peak_' + name + '_mw'] = float(values.max())
        if name in ('it', 'facility'):
            result[name + '_annual_peak_valley'] = ratio(values)
            result[name + '_positive_source_peak_valley'] = ratio(values.loc[~frame.source_unresolved])
            result[name + '_diurnal_peak_valley'] = ratio(values.groupby(time.hour).mean())
            result[name + '_weekly_peak_valley'] = ratio(values.groupby(time.dayofweek * 24 + time.hour).mean())
    return result


def decorate(frame, key, use, scale, units, cpu, gpu, mapping, unit_power):
    for name in ('base', 'cpu', 'gpu'):
        frame[name + '_it_mw'] = unit_power[name + '_w'] * units / 1e6
    for name, value in identity(key, use, scale).items():
        frame[name] = value
    frame['unit_count'], frame['u_cpu_proxy'], frame['allocated_gpu_fraction'] = units, cpu, gpu
    frame['source_hour_index'] = mapping.source_hour_index.to_numpy()
    frame['source_unresolved'] = mapping.get('source_quality_flag', pd.Series('', index=mapping.index)).eq('all_resources_zero_with_pod_inventory').to_numpy()
    frame['source_quality_flag'] = mapping.get('source_quality_flag', pd.Series('short_window_replay', index=mapping.index)).to_numpy()
    frame['all_business_inventory_excess'] = mapping.get('all_business_inventory_excess', pd.Series(False, index=mapping.index)).to_numpy()
    return frame


def checks(frame):
    columns = ['it_mw', 'cooling_mw', 'other_mw', 'facility_mw', 'base_it_mw', 'cpu_it_mw', 'gpu_it_mw']
    if not np.isfinite(frame[columns]).all().all() or frame[columns].lt(0).any().any():
        raise ValueError('小时功率存在非有限或负值')
    np.testing.assert_allclose(frame.it_mw, frame.base_it_mw + frame.cpu_it_mw + frame.gpu_it_mw, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(frame.facility_mw, frame.it_mw + frame.cooling_mw + frame.other_mw, rtol=1e-12, atol=1e-9)
    if frame.heat_balance_residual_mw.abs().max() > 1e-8:
        raise ValueError('逐小时热平衡未通过')


def comparison(frame, old, fields, units):
    result = {**fields, 'v2_available': old is not None,
              'change_reason': {'HPC': '需求、功率及制冷方法保持', 'CLD': '新增120小时整机CPU需求',
                                'TRN': '需求、功率及制冷方法保持', 'SAI': '237天需求保持；六月实测均值及空载比例迁移',
                                'INF': '单日混合业务改为185天推理子集；改用共同观测峰值归一化'}[fields['use']]}
    if old is None:
        return result
    factor = units / float(old.unit_count.iloc[0])
    for name in ('it', 'facility'):
        before = old[name + '_mw'].to_numpy() * factor
        after = frame[name + '_mw'].to_numpy()
        for metric, reducer in [('mean', np.mean), ('peak', np.max)]:
            start, end = float(reducer(before)), float(reducer(after))
            result[f'v2_{metric}_{name}_mw'] = start
            result[f'v3_{metric}_{name}_mw'] = end
            result[f'{metric}_{name}_change_pct'] = 100 * (end / start - 1)
        result[f'{name}_hourly_max_abs_difference_mw'] = float(np.max(np.abs(after - before)))
    return result


def profiles(frames, output):
    day, week, envelope, actual = [], [], [], []
    for frame in frames.values():
        time = pd.DatetimeIndex(frame.time_utc)
        columns = ['it_mw', 'cooling_mw', 'other_mw', 'facility_mw', 'base_it_mw', 'cpu_it_mw', 'gpu_it_mw']
        fields = {name: frame[name].iloc[0] for name in ('scenario_id', 'scenario_zh', 'reference_it_mw')}
        for grouper, label, target in [(time.hour, 'hour_utc', day), (time.dayofweek * 24 + time.hour, 'hour_of_target_week_utc', week)]:
            value = frame[columns].groupby(grouper).mean().reset_index(names=label)
            target.append(value.assign(**fields))
        value = frame[columns].groupby(time.floor('D')).agg(['mean', 'min', 'max'])
        value.columns = [a + '_' + b for a, b in value.columns]
        envelope.append(value.reset_index(names='date_utc').assign(**fields))
    feature_frames = [frame for frame in frames.values() if frame.reference_it_mw.iloc[0] == 10]
    features = np.column_stack([frame.it_mw.to_numpy() / frame.it_mw.mean() for frame in feature_frames]
                               + [feature_frames[0].temperature_c.to_numpy()])
    time = pd.DatetimeIndex(feature_frames[0].time_utc)
    starts = [i for i, t in enumerate(time) if t.dayofweek == 0 and t.hour == 0 and i + 168 <= len(time)]
    weeks = np.stack([features[i:i + 168] for i in starts])
    scores = (((weeks - weeks.mean(axis=0)) / np.maximum(features.std(axis=0), 1e-8)) ** 2).mean(axis=(1, 2))
    start = starts[int(np.argmin(scores))]
    for frame in frames.values():
        actual.append(frame.iloc[start:start + 168][['time_utc', 'scenario_id', 'scenario_zh', *columns]])
    for name, values in [('typical-day.csv', day), ('typical-week.csv', week), ('annual-envelope.csv', envelope), ('representative-week.csv', actual)]:
        pd.concat(values, ignore_index=True).to_csv(output / name, index=False, encoding='utf-8-sig')
    pd.DataFrame({'start_utc': time[starts], 'standardized_mse': scores}).to_csv(output / 'representative-week-scores.csv', index=False)
    return str(time[start])


def build(args):
    project = args.project.resolve()
    output = project / 'results' / RUN
    processed = project / 'data/processed' / RUN
    output.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    (output / 'hourly').mkdir(exist_ok=True)
    note(output, 'reading_inputs', stage=args.stage)
    baseline, baseline_inputs = read_baseline(project, args.baseline)
    configs = catalogue(project)
    ratios = json.loads((project / 'results/typical-load-curves-20261029-v2/mit-power-ratio-summary.json').read_text('utf-8'))['transfer_distribution']
    mit = json.loads((output / 'mit-calibration.json').read_text('utf-8')) if args.stage == 'full' else None
    weather_records = weather_inputs(project, args.weather, baseline, args.reference_only)
    summary, comparisons, sensitivity, hourly_hashes, reference_frames, weather_manifest = [], [], [], {}, {}, []
    parameter_rows = []
    active = [(key, use) for key, use in combinations() if args.stage == 'full' or use != 'SAI']
    for record in weather_records:
        weather, location, year = record['weather'], record['location_id'], record['year']
        weather_manifest.append({k: v for k, v in record.items() if k != 'weather'})
        resources, mappings = demands(project, baseline, weather.time_utc)
        if location == 'reference-grid':
            for name, mapping in mappings.items():
                mapping.to_csv(output / (name + '-annual-replay-map.csv'), index=False)
        for key, use in active:
            config = configs[key]
            cpu, gpu, mapping = resources[(key, use)]
            unit_power = power(config, key, use, cpu, gpu, ratios, mit)
            for scale in SCALES:
                units, capacity = size(config, scale)
                frame = CoolingModel(cooling_type='air').calculate(weather, unit_power['unit_w'] * units / 1e6, capacity)
                decorate(frame, key, use, scale, units, cpu, gpu, mapping, unit_power)
                checks(frame)
                fields = {**identity(key, use, scale), 'location_id': location, 'location_zh': record['location_zh'],
                          'year': year, 'power_method': 'mit_june_mean_scaled_idle' if use == 'SAI' else ('cpu_spec' if key == 'A' else 'v2_ratio_p50'),
                          'mit_release_status': mit['release_status'] if use == 'SAI' else 'not_applicable'}
                for name, value in fields.items():
                    frame[name] = value
                summary.append(summarize(frame, fields))
                directory = output / 'hourly' if location == 'reference-grid' else processed / location
                directory.mkdir(exist_ok=True)
                path = directory / f'{fields["scenario_id"]}-{year}.parquet'
                frame.to_parquet(path, index=False, compression='zstd')
                hourly_hashes[str(path.relative_to(project))] = sha256(path)
                if location == 'reference-grid':
                    reference_frames[fields['scenario_id']] = frame
                    comparisons.append(comparison(frame, baseline.get((key, use)), fields, units))
                    if scale == 10:
                        methods = ('central', 'ratio_p10', 'ratio_p90', 'legacy_idle', 'idle_half', 'idle_double') if use == 'SAI' else ('central',)
                        for method in methods:
                            candidate = power(config, key, use, cpu, gpu, ratios, mit, method)
                            for case in ('low', 'central', 'high'):
                                alternative = CoolingModel(cooling_type='air', parameter_set=case).calculate(weather, candidate['unit_w'] * units / 1e6, capacity)
                                decorate(alternative, key, use, scale, units, cpu, gpu, mapping, candidate)
                                checks(alternative)
                                sensitivity.append(summarize(alternative, {**fields, 'power_sensitivity': method, 'cooling_case': case}))
                        parameter_rows.append({**identity(key, use, scale), 'unit_count': units,
                                               'reference_unit_full_W': capacity * 1e6 / units, 'target_gpu_tdp_W': TDP[key],
                                               'cpu_packages_per_unit': config['cpu_packages_per_unit'], 'gpu_devices_per_unit': config['gpu_devices_per_unit'],
                                               'allocated_gpu_ratio': mit['measured_power_to_tdp_ratio'] if use == 'SAI' else (ratios['p50'] if key != 'A' else 0),
                                               'idle_gpu_W': TDP[key] * mit['idle_W_per_gpu'] / mit['tdp_W'] if use == 'SAI' else (70 if key == 'D' else (50 if key != 'A' else 0))})
        note(output, 'location_complete', location=location, year=year, summary_rows=len(summary))
    all_summary = pd.DataFrame(summary)
    all_summary.to_csv(output / 'annual-summary-all-locations.csv', index=False, encoding='utf-8-sig')
    central = all_summary.loc[all_summary.location_id.eq('reference-grid')]
    central.to_csv(output / 'annual-summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(comparisons).to_csv(output / 'v2-v3-comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(sensitivity).to_csv(output / 'sensitivity-summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(parameter_rows).to_csv(output / 'power-parameters.csv', index=False, encoding='utf-8-sig')
    central[['scenario_id', 'scenario_zh', 'reference_it_mw', 'source_days', 'hours', 'power_method']].assign(status='computed').to_csv(output / 'coverage.csv', index=False, encoding='utf-8-sig')
    week_start = profiles(reference_frames, output)
    save_json(output / 'manifest.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'stage': args.stage,
              'base_commit': args.base_commit, 'reference_computed_scenarios': len(central), 'location_years': len(weather_records),
              'annual_rows': len(all_summary), 'hourly_rows': int(all_summary.hours.sum()), 'hourly_sha256': hourly_hashes,
              'baseline_inputs': baseline_inputs, 'weather': weather_manifest, 'representative_week_start_utc': week_start,
              'dependency_prs': [27, 30, 32, 33], 'cooling_model_sha256': sha256(project / 'analysis/compute_load/cooling.py'),
              'cooling_asset_sha256': sha256(project / 'analysis/compute_load/assets/cooling-20261028-v2.json'),
              'mit_calibration_sha256': sha256(output / 'mit-calibration.json') if mit else None,
              'code_sha256': {name: sha256(project / 'analysis' / name) for name in (
                  'build_typical_load_curves_v3.py', 'load_curve_scenarios_v3.py', 'typical_power_models.py',
                  'typical_power_models_v2.py', 'annual_task_profiles.py', 'alibaba_cpu_profile.py',
                  'asi_inference_profiles.py', 'compute_load/model.py', 'compute_load/cooling.py')},
              'publication': '参考网格小时功率及所有城市年汇总为聚合情境结果；城市逐小时保留共享卷；无逐设备或逐作业发布',
              'scientific_scope': '需求重放及工程功率情境；无跨设备吞吐等价、季节需求识别或优化收益结论'})
    note(output, 'complete', stage=args.stage, reference_scenarios=len(central), annual_rows=len(all_summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--weather', type=Path, required=True)
    parser.add_argument('--stage', choices=('cld-inf', 'full'), default='full')
    parser.add_argument('--reference-only', action='store_true')
    parser.add_argument('--base-commit', required=True)
    build(parser.parse_args())
