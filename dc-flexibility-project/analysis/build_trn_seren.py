import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_trn_diurnal import metrics
from build_typical_load_curves_v3 import checks, read_baseline, weather_inputs
from compute_load.cooling import CoolingModel
from compute_load.model import sha256
from load_curve_scenarios_v3 import SCALES, TDP, catalogue, identity, size
from native_load_inputs_v3 import weather_placements
from prepare_trn_seren import RUN, TYPES, quality, save_json
from trn_power_models import cpu_response, idle_power
from typical_power_models import PARAMETERS, validate_utilization


def cases(source, project):
    central = quality(source)
    result = {}
    for name, idle in [('main', '60.000'), ('idle50', '50.000'), ('idle70', '70.000'), ('idle_zero_proxy', '67.300')]:
        frame = source.loc[central].copy()
        frame['allocated_gpu_power_ratio'] = frame['inverse_ratio_idle_' + idle]
        frame['power_imputed'] = False
        result[name] = frame
    for threshold in (.75, .85):
        frame = source.loc[quality(source, threshold)].copy()
        frame['allocated_gpu_power_ratio'] = frame['inverse_ratio_idle_60.000']
        frame['power_imputed'] = False
        result['coverage' + str(int(threshold * 100))] = frame
    clipped = result['main'].copy()
    clipped['allocated_gpu_power_ratio'] = clipped.allocated_gpu_power_ratio.clip(0, 1)
    result['clip_0_1'] = clipped
    profiles = pd.read_csv(project / 'results' / RUN / 'power-profiles.csv')
    day = profiles.loc[profiles.grouping.eq('hour_of_source_day')].set_index('position').gpu_hour_weighted_mean_ratio
    full = source.copy()
    full['power_imputed'] = ~central
    full['allocated_gpu_power_ratio'] = full['inverse_ratio_idle_60.000'].where(central, full.hour_of_source_day.map(day))
    result['full_observed_plus_profile'] = full
    fixed = result['main'].copy()
    fixed['allocated_gpu_power_ratio'] = .8
    result['seren_constant_080'] = fixed
    kalos = pd.read_csv(project / 'results/trn-power-model-20261101/diurnal-acme/transfer-profile.csv')
    borrowed = result['main'].copy()
    values = kalos.set_index('hour_utc').shape_central.reindex(borrowed.hour_of_source_day).to_numpy()
    borrowed['allocated_gpu_power_ratio'] = values * .8 / np.average(values, weights=borrowed.allocated_gpu_hours)
    result['seren_080_Kalos_shape'] = borrowed
    for name, frame in result.items():
        if frame.empty or not np.isfinite(frame.allocated_gpu_power_ratio).all() or frame.allocated_gpu_power_ratio.lt(0).any():
            raise ValueError('功率比例无效：' + name)
        result[name] = frame.reset_index(drop=True)
    return result


def unit_power(config, key, source, shape):
    gpu = validate_utilization(source.allocated_gpu_fraction.to_numpy())
    cpu = validate_utilization(source.u_cpu_proxy.to_numpy())
    ratio = source.allocated_gpu_power_ratio.to_numpy(float)
    if not np.isfinite(ratio).all() or (ratio < 0).any():
        raise ValueError('反推比例必须非负且有限')
    base, ci, cf, _, _ = PARAMETERS[config['configuration_id']]['central']
    base_w = np.full(len(source), base)
    cpu_w = config['cpu_packages_per_unit'] * (ci + (cf - ci) * cpu_response(cpu, shape))
    # 原始条件估计保留超TDP标记；clip_0_1方案单独改变该端点。
    gpu_w = config['gpu_devices_per_unit'] * (gpu * TDP[key] * ratio + (1 - gpu) * (70. if key == 'D' else 50.))
    return {'base_w': base_w, 'cpu_w': cpu_w, 'gpu_w': gpu_w, 'unit_w': base_w + cpu_w + gpu_w}


def build(args):
    project = args.project.resolve()
    output = project / 'results' / RUN
    source_path = output / 'source-hourly.csv'
    source = pd.read_csv(source_path, parse_dates=['source_clock'])
    variants = cases(source, project)
    configs = catalogue(project)
    mit = json.loads((project / 'results/typical-load-curves-20261101-v3/mit-calibration.json').read_text('utf-8'))
    shape = json.loads((project / 'results/trn-power-model-20261101/cpu-shape.json').read_text('utf-8'))
    baseline, baseline_inputs = read_baseline(project, args.baseline)
    weather_records = weather_inputs(project, args.weather, baseline, args.reference_only)
    records, days, weeks, power_composition, hashes, validation = [], [], [], [], {}, []
    keep_source = ['source_clock', 'source_hour_index', 'hour_of_source_day', 'hour_of_source_week',
        'allocated_gpu_hours', 'allocated_gpu_fraction', 'host_cpu_core_hours', 'independent_cpu_core_hours', 'u_cpu_proxy',
        'allocated_gpu_power_ratio', 'power_imputed', 'power_ratio_above_tdp', 'physical_device_coverage',
        'overlap_samples', 'source_partial_boundary_hour', 'source_dataset', *['gpu_hours_' + x for x in TYPES],
        *['host_cpu_core_hours_' + x for x in TYPES]]
    for record in weather_records:
        for case, source_case in variants.items():
            _, weather, selected, leap = next(weather_placements(source_case, record['weather'], record['year'], True))
            if leap.any():
                raise ValueError('Seren 2023原生窗口不应包含闰日')
            for key in ('B', 'C', 'D'):
                unit = unit_power(configs[key], key, selected, shape)
                for scale in SCALES:
                    units, capacity = size(configs[key], scale)
                    fields = {**identity(key, 'TRN', scale), 'source_days': len(selected) / 24,
                        'unit_count': units, 'location_id': record['location_id'], 'location_zh': record['location_zh'],
                        'weather_year': record['year'], 'power_model_version': RUN, 'case': case,
                        'time_basis': 'Seren原时间带+08:00已换算UTC；按源月日小时对应目标气象年',
                        'gap_policy': '仅保留本方案的合格真实小时；缺口不填充、不压缩时间、不循环'}
                    frame = CoolingModel().calculate(weather, unit['unit_w'] * units / 1e6, capacity, full_year=False, allow_gaps=True)
                    for name in ('base', 'cpu', 'gpu'):
                        frame[name + '_it_mw'] = unit[name + '_w'] * units / 1e6
                    for name in keep_source:
                        frame[name] = selected[name].to_numpy()
                    for name, value in fields.items():
                        frame[name] = value
                    frame['weather_time_utc'] = frame.time_utc
                    frame['idle_it_mw'] = idle_power(configs[key], key, 'TRN', mit) * units / 1e6
                    frame['task_it_mw'] = frame.it_mw - frame.idle_it_mw
                    frame['source_unresolved'] = False
                    checks(frame)
                    summary = metrics(frame, case)
                    summary.update(imputed_hours=int(selected.power_imputed.sum()), ratio_above_tdp_hours=int(selected.allocated_gpu_power_ratio.gt(1).sum()),
                        location_zh=record['location_zh'], source_dataset='Acme Seren', normalizer_gpu_count=float(source.allocated_gpu_hours.max() / source.allocated_gpu_fraction.max()))
                    records.append(summary)
                    if record['location_id'] == 'reference-grid':
                        labels = {name: fields[name] for name in ('scenario_id', 'server_type', 'reference_it_mw', 'case')}
                        columns = ['it_mw', 'facility_mw', 'task_it_mw', 'gpu_it_mw', 'allocated_gpu_power_ratio']
                        for grouping, target in [('hour_of_source_day', days), ('hour_of_source_week', weeks)]:
                            grouped = frame.groupby(grouping)[columns].mean().reset_index()
                            grouped['available_hours'] = frame.groupby(grouping).size().to_numpy()
                            target.append(grouped.assign(**labels))
                        if scale == 10:
                            for kind in TYPES:
                                allocated_w = units * configs[key]['gpu_devices_per_unit'] * TDP[key] * selected.allocated_gpu_fraction * selected.allocated_gpu_power_ratio
                                kind_share = selected['gpu_hours_' + kind].div(selected.allocated_gpu_hours).fillna(0)
                                power_composition.append({**labels, 'type': kind, 'gpu_hours': float(selected['gpu_hours_' + kind].sum()),
                                    'gpu_share': float(selected['gpu_hours_' + kind].sum() / selected.allocated_gpu_hours.sum()),
                                    'modeled_mean_allocated_gpu_mw': float((allocated_w * kind_share).mean() / 1e6),
                                    'attribution': 'common hourly cluster inferred ratio apportioned by GPU allocation; not per-type measured power'})
                    # 全部地点保留正文和六个月对照；参考网格同时保留全部敏感性。
                    if record['location_id'] == 'reference-grid' or case in ('main', 'full_observed_plus_profile'):
                        target = output / 'hourly' / record['location_id'] / case / f'{fields["scenario_id"]}-{record["year"]}.parquet'
                        target.parent.mkdir(parents=True, exist_ok=True)
                        frame.to_parquet(target, index=False, compression='zstd')
                        hashes[target.relative_to(output).as_posix()] = sha256(target)
                    validation.append({'case': case, 'scenario_id': fields['scenario_id'], 'location': record['location_id'], 'year': record['year'],
                        'hours': len(frame), 'heat_balance_max_mw': float(frame.heat_balance_residual_mw.abs().max())})
        print(json.dumps({'location': record['location_id'], 'year': record['year'], 'summary_rows': len(records)}), flush=True)
    summary = pd.DataFrame(records)
    summary.to_csv(output / 'window-summary-all-locations.csv', index=False)
    reference = summary.loc[summary.location_id.eq('reference-grid')]
    reference.to_csv(output / 'window-summary.csv', index=False)
    pd.concat(days, ignore_index=True).to_csv(output / 'typical-day.csv', index=False)
    pd.concat(weeks, ignore_index=True).to_csv(output / 'typical-week.csv', index=False)
    pd.DataFrame(power_composition).to_csv(output / 'task-power-composition.csv', index=False)
    previous_path = project / 'results/trn-power-model-20261101/diurnal-acme/window-summary-all-locations.csv'
    previous = pd.read_csv(previous_path)
    old = previous.loc[previous.case.isin(['constant_080', 'shape_central'])].copy()
    old['case'] = old.case.map({'constant_080': 'old_Kalos_constant_080', 'shape_central': 'old_Kalos_shape_080'})
    old['source_dataset'] = 'Acme Kalos'
    pd.concat([old, summary], ignore_index=True).to_csv(output / 'comparison-all-locations.csv', index=False)
    comparison = pd.concat([old.loc[old.location_id.eq('reference-grid')], reference], ignore_index=True)
    comparison.to_csv(output / 'comparison.csv', index=False)
    save_json(output / 'calculation-audit.json', {'utc': datetime.now(timezone.utc).isoformat(), 'input_commit': args.input_commit,
        'source_hourly_sha256': sha256(source_path), 'previous_summary_sha256': sha256(previous_path),
        'scenarios': 12, 'location_years': len(weather_records), 'cases': list(variants), 'summary_rows': len(summary),
        'computed_hour_rows': int(summary.hours.sum()), 'saved_hourly_files': len(hashes), 'hourly_sha256': hashes,
        'weather': [{k: v for k, v in record.items() if k != 'weather'} for record in weather_records], 'baseline_inputs': baseline_inputs,
        'cooling_parameters_sha256': sha256(project / 'analysis/compute_load/assets/cooling-20261028-v2.json'),
        'cpu_shape_sha256': sha256(project / 'results/trn-power-model-20261101/cpu-shape.json'),
        'case_hours': {name: len(frame) for name, frame in variants.items()}, 'checks': validation,
        'energy_scope': 'Actual selected hour bins only; gaps omitted, not imputed in main; no annualization.',
        'full_window': 'Qualified hours use observed inference; other hours use qualified GPU-hour-weighted UTC-hour mean.',
        'old_new_comparison': 'Different source, normalization, window and task mix; descriptive comparison. Seren controls hold selected hours and demand fixed.',
        'code_sha256': {name: sha256(project / 'analysis' / name) for name in ['prepare_trn_seren.py', 'build_trn_seren.py', 'compute_load/model.py', 'compute_load/cooling.py']}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--weather', type=Path, required=True)
    parser.add_argument('--reference-only', action='store_true')
    parser.add_argument('--input-commit', default='6de2e4cde4dbf66f055c0802364c96e511945bbe')
    build(parser.parse_args())
