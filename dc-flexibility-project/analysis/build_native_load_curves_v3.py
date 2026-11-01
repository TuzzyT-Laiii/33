import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from build_typical_load_curves_v3 import checks, read_baseline, weather_inputs
from compute_load.model import sha256
from load_curve_scenarios_v3 import SCALES, catalogue, combinations, identity, power, size
from native_load_inputs_v3 import RUN, PREVIOUS, POWER_COLUMNS, calculate, flexibility_inputs, metrics, native_sources, ratio, save_json, weather_placements


def progress(output, state, **fields):
    value = {'utc': datetime.now(timezone.utc).isoformat(), 'state': state, **fields}
    save_json(output / 'progress.json', value)
    with (output / 'progress.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + '\n')
    print(json.dumps(value, ensure_ascii=False), flush=True)


def source_metrics(frame, fields):
    row = {**fields, 'native_hours': len(frame), 'native_days': len(frame) / 24,
           'weekly_shape_available': fields['use'] != 'CLD'}
    for name in ('it', 'base_it', 'cpu_it', 'gpu_it'):
        values = frame[name + '_mw']
        row.update({f'mean_{name}_mw': float(values.mean()), f'peak_{name}_mw': float(values.max()),
                    f'minimum_{name}_mw': float(values.min()), f'window_{name}_mwh': float(values.sum())})
    row['it_window_peak_valley'] = ratio(frame.it_mw)
    row['it_diurnal_peak_valley'] = ratio(frame.groupby('hour_of_source_day').it_mw.mean())
    row['it_weekly_peak_valley'] = ratio(frame.groupby('hour_of_source_week').it_mw.mean()) if row['weekly_shape_available'] else None
    return row


def reduce_placements(cases, native):
    keys = ['location_id', 'location_zh', 'weather_year', 'scenario_id']
    metrics_names = [name for name in cases if name.startswith(('mean_', 'peak_', 'minimum_', 'window_', 'it_', 'facility_')) and name not in ('it_capacity_mw',)]
    metrics_names = [name for name in metrics_names if pd.api.types.is_numeric_dtype(cases[name])]
    rows = []
    for values, group in cases.groupby(keys, sort=False):
        source = native[values[-1]]
        row = {**dict(zip(keys, values)), **source,
               'placement_count': len(group), 'placement_hours_min': int(group.hours.min()),
               'placement_hours_max': int(group.hours.max()), 'truncated_placements': int(group.truncated.sum())}
        for name in metrics_names:
            series = group[name].dropna()
            for statistic in ('median', 'min', 'max'):
                row[f'placement_{name}_{statistic}'] = float(getattr(series, statistic)()) if len(series) else None
            if name not in source:
                row[name] = float(series.median()) if len(series) else None
        rows.append(row)
    return pd.DataFrame(rows)


def compare(project, summary, baseline):
    prior = pd.read_csv(project / 'results' / PREVIOUS / 'annual-summary.csv').set_index('scenario_id')
    rows = []
    for record in summary.to_dict('records'):
        row = {name: record[name] for name in ('scenario_id', 'scenario_zh', 'reference_it_mw', 'use', 'native_hours', 'native_days')}
        old = baseline.get((record['server_type'], record['use']))
        for label, data in [('previous_v3_annual', prior.loc[record['scenario_id']]), ('v2_annual', None)]:
            if label == 'v2_annual' and old is not None:
                factor = record['unit_count'] / float(old.unit_count.iloc[0])
                data = {'mean_it_mw': float(old.it_mw.mean()) * factor, 'mean_facility_mw': float(old.facility_mw.mean()) * factor}
            for component in ('it', 'facility'):
                after = record['mean_it_mw'] if component == 'it' else record['placement_mean_facility_mw_median']
                row[f'native_mean_{component}_mw'] = after
                if data is not None:
                    before = float(data[f'mean_{component}_mw'])
                    row[f'{label}_mean_{component}_mw'] = before
                    row[f'{label}_{component}_change_pct'] = 100 * (after / before - 1)
        row['comparison_scope'] = '平均功率描述性对比；历史全年窗口与原生窗口、日期气象对应不同，不能分离单一因素'
        rows.append(row)
    return pd.DataFrame(rows)


def build(args):
    project = args.project.resolve()
    output, processed = project / 'results' / RUN, project / 'data/processed' / RUN
    output.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    for name in ('hourly', 'source-hourly'):
        (output / name).mkdir(exist_ok=True)
    progress(output, 'reading_native_inputs', input_commit=args.input_commit)
    baseline, baseline_inputs = read_baseline(project, args.baseline)
    sources, metadata = native_sources(project, args.source_project, args.mit_hourly)
    configs = catalogue(project)
    ratios = json.loads((project / 'results/typical-load-curves-20261029-v2/mit-power-ratio-summary.json').read_text('utf-8'))['transfer_distribution']
    previous = project / 'results' / PREVIOUS
    mit = json.loads((previous / 'mit-calibration.json').read_text('utf-8'))
    if mit['partial_data']:
        raise ValueError('必须使用六月完成版功率标定')
    for path in previous.glob('mit-*'):
        if path.is_file():
            shutil.copyfile(path, output / path.name)
    stats, candidates = flexibility_inputs(project, args.source_project, args.mit_reference, sources)
    save_json(output / 'source-time-bases.json', metadata)
    save_json(output / 'flexibility-source-statistics.json', stats)
    native, source_hashes, flexible = {}, {}, []
    unit_powers = {}
    for key, use in combinations():
        source = sources[use]
        cpu, gpu = source.u_cpu_proxy.to_numpy(), source.allocated_gpu_fraction.to_numpy()
        unit = power(configs[key], key, use, cpu, gpu, ratios, mit)
        unit_powers[(key, use)] = unit
        for scale in SCALES:
            units, capacity = size(configs[key], scale)
            fields = {**identity(key, use, scale), 'unit_count': units, 'installed_reference_it_mw': capacity,
                      'has_real_dates': metadata[use]['has_real_dates'], 'time_basis': metadata[use]['time_basis']}
            frame = source.copy().assign(**fields)
            for name in ('base', 'cpu', 'gpu'):
                frame[name + '_it_mw'] = unit[name + '_w'] * units / 1e6
            frame['it_mw'] = unit['unit_w'] * units / 1e6
            row = source_metrics(frame, fields)
            native[fields['scenario_id']] = row
            path = output / 'source-hourly' / (fields['scenario_id'] + '.parquet')
            frame.to_parquet(path, index=False, compression='zstd')
            source_hashes[path.relative_to(project).as_posix()] = sha256(path)
            hint = {**fields, **stats[use], 'native_hours': len(frame), 'native_days': len(frame) / 24,
                    'interpretation': '原生窗口内资源分类候选；期限、暂停和补做容量未验证，未运行优化'}
            if use in ('CLD', 'INF'):
                if use == 'CLD':
                    after_cpu, after_gpu = cpu * (1 - frame.batch_share.to_numpy()), gpu
                else:
                    after_gpu = np.maximum(gpu - candidates['INF'] / 124677.704, 0)
                    after_cpu = .25 * after_gpu
                after = power(configs[key], key, use, after_cpu, after_gpu, ratios, mit)
                decrement = (unit['unit_w'] - after['unit_w']) * units / 1e6
                if np.any(decrement < -1e-9):
                    raise ValueError('候选反事实功率差为负')
                hint.update(illustrative_mean_it_decrement_mw=float(decrement.mean()),
                            illustrative_peak_it_decrement_mw=float(decrement.max()),
                            illustrative_window_it_energy_decrement_mwh=float(decrement.sum()),
                            illustrative_it_energy_decrement_share=float(decrement.sum() / frame.it_mw.sum()))
            flexible.append(hint)
    pd.DataFrame(native.values()).to_csv(output / 'native-it-summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(flexible).to_csv(output / 'flexibility-hints.csv', index=False, encoding='utf-8-sig')
    records = weather_inputs(project, args.weather, baseline, args.reference_only)
    summaries, sensitivities, day_rows, week_rows, time_rows, weather_manifest, hourly_hashes = [], [], [], [], [], [], {}
    for record in records:
        location, year = record['location_id'], record['year']
        weather_manifest.append({k: v for k, v in record.items() if k != 'weather'})
        for key, use in combinations():
            source, unit = sources[use], unit_powers[(key, use)]
            for placement, weather, selected, leap in weather_placements(source, record['weather'], year, metadata[use]['has_real_dates']):
                index = selected.source_hour_index.to_numpy()
                for scale in SCALES:
                    units, capacity = size(configs[key], scale)
                    fields = {**identity(key, use, scale), 'location_id': location, 'location_zh': record['location_zh'],
                        'weather_year': year, 'placement': placement, 'has_real_dates': metadata[use]['has_real_dates'],
                        'native_hours': len(source), 'native_days': len(source) / 24,
                        'truncated': len(selected) != len(source), 'dropped_source_tail_hours': len(source) - len(selected),
                        'feb29_mapped_to_feb28_hours': int(leap.sum()), 'unit_count': units,
                        'installed_reference_it_mw': capacity, 'time_basis': metadata[use]['time_basis']}
                    frame = calculate(weather, unit['unit_w'][index] * units / 1e6, capacity)
                    for name in ('base', 'cpu', 'gpu'):
                        frame[name + '_it_mw'] = unit[name + '_w'][index] * units / 1e6
                    for name in selected:
                        frame[name] = selected[name].to_numpy()
                    for name, value in fields.items():
                        frame[name] = value
                    frame['weather_time_utc'] = frame.time_utc
                    frame['feb29_mapped_to_feb28'] = leap
                    checks(frame)
                    summary = metrics(frame, fields)
                    summary['weather_start'] = str(frame.time_utc.iloc[0])
                    summary['weather_end_in_source_order'] = str(frame.time_utc.iloc[-1])
                    summaries.append(summary)
                    directory = output / 'hourly' if location == 'reference-grid' else processed / location
                    directory.mkdir(exist_ok=True)
                    path = directory / f'{fields["scenario_id"]}-{year}-{placement}.parquet'
                    frame.to_parquet(path, index=False, compression='zstd')
                    hourly_hashes[path.relative_to(project).as_posix()] = sha256(path)
                    if location == 'reference-grid':
                        for column, target, values in [('hour_of_source_day', day_rows, range(24)), ('hour_of_source_week', week_rows, range(168))]:
                            if use == 'CLD' and column == 'hour_of_source_week':
                                continue
                            grouped = frame.groupby(column)[[name + '_mw' for name in POWER_COLUMNS]].mean().reindex(values).reset_index()
                            target.append(grouped.assign(**fields))
                        if scale == 10:
                            time_rows.append(frame[['scenario_id', 'source_hour_index', 'facility_mw', 'cooling_mw']].assign(placement=placement))
                            if use == 'SAI':
                                for method in ('central', 'ratio_p10', 'ratio_p90', 'legacy_idle', 'idle_half', 'idle_double'):
                                    alternative = power(configs[key], key, use, source.u_cpu_proxy.to_numpy(), source.allocated_gpu_fraction.to_numpy(), ratios, mit, method)
                                    for cooling_case in ('low', 'central', 'high'):
                                        candidate = calculate(weather, alternative['unit_w'] * units / 1e6, capacity, cooling_case)
                                        for name in ('base', 'cpu', 'gpu'):
                                            candidate[name + '_it_mw'] = alternative[name + '_w'] * units / 1e6
                                        for name in selected:
                                            candidate[name] = selected[name].to_numpy()
                                        candidate['use'] = use
                                        checks(candidate)
                                        sensitivities.append(metrics(candidate, {**fields, 'power_sensitivity': method, 'cooling_case': cooling_case}))
        progress(output, 'location_complete', location=location, weather_year=year, placement_rows=len(summaries))
    cases = pd.DataFrame(summaries)
    cases.to_csv(output / 'window-summary-all-locations.csv', index=False, encoding='utf-8-sig')
    aggregate = reduce_placements(cases, native)
    aggregate.to_csv(output / 'placement-aggregate-all-locations.csv', index=False, encoding='utf-8-sig')
    reference = aggregate.loc[aggregate.location_id.eq('reference-grid')]
    reference.to_csv(output / 'native-summary.csv', index=False, encoding='utf-8-sig')
    cases.loc[cases.location_id.eq('reference-grid')].to_csv(output / 'placement-summary.csv', index=False, encoding='utf-8-sig')
    pd.concat(day_rows, ignore_index=True).to_csv(output / 'typical-day-placements.csv', index=False, encoding='utf-8-sig')
    pd.concat(week_rows, ignore_index=True).to_csv(output / 'typical-week-placements.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(sensitivities).to_csv(output / 'sensitivity-summary.csv', index=False, encoding='utf-8-sig')
    compare(project, reference, baseline).to_csv(output / 'mean-power-comparison.csv', index=False, encoding='utf-8-sig')
    combined = pd.concat(time_rows, ignore_index=True)
    series = combined.groupby(['scenario_id', 'source_hour_index']).agg(
        facility_mw_median=('facility_mw', 'median'), facility_mw_min=('facility_mw', 'min'),
        facility_mw_max=('facility_mw', 'max'), available_placements=('facility_mw', 'count')).reset_index()
    series.to_parquet(output / 'native-timeseries-10MW.parquet', index=False, compression='zstd')
    reference[['scenario_id', 'scenario_zh', 'native_hours', 'native_days', 'placement_count', 'placement_hours_min', 'placement_hours_max', 'truncated_placements']].to_csv(output / 'coverage.csv', index=False, encoding='utf-8-sig')
    save_json(output / 'manifest.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'input_commit': args.input_commit,
        'scenario_count': len(native), 'location_years': len(records), 'placement_rows': len(cases),
        'native_it_rows': sum(row['native_hours'] for row in native.values()), 'weather_matched_hourly_rows': int(cases.hours.sum()),
        'hourly_sha256': hourly_hashes, 'source_hourly_sha256': source_hashes, 'sources': metadata,
        'baseline_inputs': baseline_inputs, 'weather': weather_manifest,
        'unknown_date_policy': '12个月1日00:00 UTC分别放置；年末截断源尾部，不循环、不跨到下一年气象、不补齐',
        'dated_policy': '每个源小时按月日小时对应目标气象年；保持源顺序；源2月29日遇非闰年对应2月28日并标识；当前来源无2月29日',
        'calendar_week_policy': '有日期按源日期星期；无日期只计算相对七日分组；CLD五日不报周形状',
        'energy_scope': '每个实际计算窗口内小时功率之和，不年化',
        'dependency_prs': [27, 30, 32, 33], 'downstream_8760_results_modified': False,
        'mit_calibration_sha256': sha256(output / 'mit-calibration.json'),
        'cooling_model_sha256': sha256(project / 'analysis/compute_load/cooling.py'),
        'cooling_asset_sha256': sha256(project / 'analysis/compute_load/assets/cooling-20261028-v2.json'),
        'code_sha256': {name: sha256(project / 'analysis' / name) for name in ('native_load_inputs_v3.py', 'build_native_load_curves_v3.py', 'load_curve_scenarios_v3.py', 'build_typical_load_curves_v3.py')},
        'publication': '公开聚合源小时IT和参考网格气象匹配功率；城市小时文件留在服务器；无逐作业或逐设备记录'})
    progress(output, 'complete', scenarios=len(native), location_years=len(records), placement_rows=len(cases), hourly_rows=int(cases.hours.sum()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source-project', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--weather', type=Path, required=True)
    parser.add_argument('--mit-hourly', type=Path, required=True)
    parser.add_argument('--mit-reference', type=Path, required=True)
    parser.add_argument('--input-commit', required=True)
    parser.add_argument('--reference-only', action='store_true')
    build(parser.parse_args())
