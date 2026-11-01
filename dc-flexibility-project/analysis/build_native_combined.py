import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from build_native_load_curves_v3 import progress, reduce_placements, source_metrics
from build_typical_load_curves_v3 import checks
from compute_load.cooling import CoolingModel, WEATHER_COLUMNS
from compute_load.model import sha256
from native_combined_power import CombinedPower, NATIVE, RUN
from native_load_inputs_v3 import calculate, metrics, POWER_COLUMNS, save_json


COMMITS = {'31': '3b11b5866890fa47149663dcddf7e01052358f44', '34': 'c836bbaf095dfe4660bad8abe9ab9e0ddee4cd6c',
    '35': '520f68d57c105d255d867d13e1f7498aa59c1ded', '36': '1594fb77148f3c9307d00230c25e864f6a286625'}


def run(project, inputs):
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    output = project / 'results' / RUN
    output.mkdir(parents=True, exist_ok=False)
    (output / 'source-hourly').mkdir()
    (output / 'hourly').mkdir()
    old_root = inputs / 'results' / NATIVE
    manifest = json.loads((old_root / 'manifest.json').read_text('utf-8'))
    model = CombinedPower(project)
    if CoolingModel().metadata['parameters_sha256'] != manifest['cooling_asset_sha256'] or CoolingModel().metadata['cooling_code_sha256'] != manifest['cooling_model_sha256']:
        raise ValueError('冷却模型必须与原生窗口冻结版本一致')
    progress(output, 'source_power')
    sources, native, source_hashes, old_hashes = {}, {}, {}, {}
    for relative, digest in manifest['source_hourly_sha256'].items():
        path = inputs / relative
        if sha256(path) != digest:
            raise ValueError('源小时文件指纹不符：' + relative)
        source = pd.read_parquet(path)
        key, use, scenario = source.server_type.iloc[0], source.use.iloc[0], source.scenario_id.iloc[0]
        unit = model.unit(source, key, use)
        count = int(source.unit_count.iloc[0])
        for name in ('base', 'cpu', 'gpu'):
            source[name + '_it_mw'] = unit[name + '_w'] * count / 1e6
        source['it_mw'] = unit['unit_w'] * count / 1e6
        source['power_model_version'] = RUN
        fields = {name: source[name].iloc[0] for name in ('scenario_id', 'scenario_zh', 'scenario', 'server_type', 'use', 'reference_it_mw', 'unit_count', 'installed_reference_it_mw', 'has_real_dates', 'time_basis')}
        fields['has_real_dates'] = bool(fields['has_real_dates'])
        native[scenario] = source_metrics(source, fields)
        sources[scenario] = source
        target = output / 'source-hourly' / path.name
        source.to_parquet(target, index=False, compression='zstd')
        source_hashes[str(target.relative_to(project))] = sha256(target)
        old_hashes[relative] = digest
    pd.DataFrame(native.values()).to_csv(output / 'native-it-summary.csv', index=False, encoding='utf-8-sig')
    rows, days, weeks, times, hashes = [], [], [], [], {}
    old_summary = pd.read_csv(old_root / 'window-summary-all-locations.csv')
    identity = ['scenario_id', 'location_id', 'weather_year', 'placement']
    old_by_key = old_summary.set_index(identity)
    model_cache = {}
    for number, (relative, digest) in enumerate(manifest['hourly_sha256'].items()):
        path = inputs / relative
        if sha256(path) != digest:
            raise ValueError('天气匹配小时文件指纹不符：' + relative)
        old = pd.read_parquet(path)
        scenario, location, year, placement = [old[name].iloc[0] for name in identity]
        key, use = old.server_type.iloc[0], old.use.iloc[0]
        cache_key = (scenario, location, int(year))
        if cache_key not in model_cache:
            model_cache[cache_key] = model.unit(sources[scenario], key, use, location, year)
        unit = model_cache[cache_key]
        index = old.source_hour_index.to_numpy(int)
        count, capacity = int(old.unit_count.iloc[0]), float(old.installed_reference_it_mw.iloc[0])
        weather = old[[name for name in WEATHER_COLUMNS if name in old]].copy()
        if old.pressure_assumed.all():
            weather = weather.drop(columns='pressure_pa')
        current = old.copy()
        thermal = calculate(weather, unit['unit_w'][index] * count / 1e6, capacity)
        for name in thermal:
            current[name] = thermal[name].to_numpy()
        for name in ('base', 'cpu', 'gpu'):
            current[name + '_it_mw'] = unit[name + '_w'][index] * count / 1e6
        for name in ('request_intensity', 'online_gpu_w_each', 'allocated_gpu_power_ratio'):
            if name in unit:
                current[name] = unit[name][index]
        if use == 'INF':
            for name in ('online_gpu', 'offline_gpu', 'unallocated_gpu'):
                current[name + '_it_mw'] = unit[name + '_w'][index] * count / 1e6
        current['power_model_version'] = RUN
        checks(current)
        for name in ('source_clock', 'source_hour_index', 'u_cpu_proxy', 'allocated_gpu_fraction', 'unit_count', 'time_utc', 'temperature_c', 'relative_humidity_pct', 'source_quality_flag', 'source_unresolved'):
            if name in old:
                pd.testing.assert_series_equal(current[name], old[name])
        np.testing.assert_allclose(current.base_it_mw + current.cpu_it_mw + current.gpu_it_mw, current.it_mw, rtol=1e-12)
        fields = old_by_key.loc[(scenario, location, year, placement)].to_dict()
        fields = {name: value for name, value in fields.items() if not name.startswith(('mean_', 'peak_', 'minimum_', 'window_', 'it_', 'facility_'))}
        fields.update(scenario_id=scenario, location_id=location, weather_year=int(year), placement=placement)
        rows.append(metrics(current, fields))
        if location == 'reference-grid':
            target = output / 'hourly' / path.name
            for grouping, table in [('hour_of_source_day', days), ('hour_of_source_week', weeks)]:
                if use == 'CLD' and grouping == 'hour_of_source_week':
                    continue
                table.append(current.groupby(grouping)[[name + '_mw' for name in POWER_COLUMNS]].mean().reset_index().assign(**{name: old[name].iloc[0] for name in ('scenario_id', 'reference_it_mw', 'use', 'placement')}))
            if float(old.reference_it_mw.iloc[0]) == 10:
                times.append(current[['scenario_id', 'source_hour_index', 'facility_mw']].assign(placement=placement))
        else:
            target = project / 'data/processed' / RUN / location / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
        current.to_parquet(target, index=False, compression='zstd')
        hashes[str(target.relative_to(project))] = sha256(target)
        old_hashes[relative] = digest
        if number % 100 == 99:
            progress(output, 'weather_power', completed=number + 1, total=len(manifest['hourly_sha256']))
    cases = pd.DataFrame(rows)
    if len(cases) != 5500 or int(cases.hours.sum()) != 15861600:
        raise ValueError('原生窗口覆盖必须与#34完全一致')
    cases.to_csv(output / 'window-summary-all-locations.csv', index=False, encoding='utf-8-sig')
    aggregate = reduce_placements(cases, native)
    aggregate.to_csv(output / 'placement-aggregate-all-locations.csv', index=False, encoding='utf-8-sig')
    reference = aggregate.loc[aggregate.location_id.eq('reference-grid')]
    reference.to_csv(output / 'native-summary.csv', index=False, encoding='utf-8-sig')
    cases.loc[cases.location_id.eq('reference-grid')].to_csv(output / 'placement-summary.csv', index=False, encoding='utf-8-sig')
    pd.concat(days).to_csv(output / 'typical-day-placements.csv', index=False, encoding='utf-8-sig')
    pd.concat(weeks).to_csv(output / 'typical-week-placements.csv', index=False, encoding='utf-8-sig')
    combined = pd.concat(times).groupby(['scenario_id', 'source_hour_index']).agg(facility_mw_median=('facility_mw', 'median'), facility_mw_min=('facility_mw', 'min'), facility_mw_max=('facility_mw', 'max'), available_placements=('facility_mw', 'count')).reset_index()
    combined.to_parquet(output / 'native-timeseries-10MW.parquet', index=False, compression='zstd')
    before = pd.read_csv(old_root / 'native-summary.csv').set_index('scenario_id')
    comparison = reference[['scenario_id', 'scenario_zh', 'reference_it_mw', 'native_days']].copy().set_index('scenario_id')
    for column in ('mean_it_mw', 'peak_it_mw', 'minimum_it_mw', 'it_diurnal_peak_valley', 'it_weekly_peak_valley', 'window_it_mwh',
            'placement_mean_facility_mw_median', 'placement_peak_facility_mw_median', 'placement_facility_diurnal_peak_valley_median', 'placement_window_facility_mwh_median'):
        comparison['old_' + column] = before[column]
        comparison['new_' + column] = reference.set_index('scenario_id')[column]
        comparison['change_pct_' + column] = 100 * (comparison['new_' + column] / comparison['old_' + column] - 1)
    comparison.reset_index().to_csv(output / 'old-model-comparison.csv', index=False, encoding='utf-8-sig')
    # 核对旧窗口与资源逐项守恒，再次读取指纹确认原文件未改写。
    for relative, digest in old_hashes.items():
        if sha256(inputs / relative) != digest:
            raise ValueError('原目录字节发生变化：' + relative)
    save_json(output / 'source-time-bases.json', manifest['sources'])
    save_json(output / 'manifest.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'dependency_commits': COMMITS,
        'input_native_manifest_sha256': sha256(old_root / 'manifest.json'), 'input_hourly_sha256': old_hashes,
        'hourly_sha256': hashes, 'source_hourly_sha256': source_hashes, 'scenario_count': len(native),
        'location_years': 25, 'placement_rows': len(cases), 'weather_matched_hourly_rows': int(cases.hours.sum()),
        'native_it_rows': sum(len(s) for s in sources.values()), 'old_input_bytes_unchanged': True,
        'inference_calibration_policy': '冻结#35各地点/气象年的全年标定gain；原生窗口不重新标定，不生成循环需求。源相对小时使用BurstGPT相位0日形状。',
        'training_policy': '冻结#36 shape_central；UTC日形状；原生90天GPU小时加权均值0.80；无星期修正。',
        'cooling': CoolingModel().metadata, 'code_sha256': {name: sha256(project / 'analysis' / name) for name in ('native_combined_power.py', 'build_native_combined.py', 'trn_power_models.py', 'native_load_inputs_v3.py')}})
    progress(output, 'complete', scenarios=len(native), placement_rows=len(cases), hourly_rows=int(cases.hours.sum()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    args = parser.parse_args()
    run(args.project.resolve(), args.inputs.resolve())
