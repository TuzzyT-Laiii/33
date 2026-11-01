import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_typical_load_curves_v3 import checks, summarize
from compute_load.cooling import CoolingModel, WEATHER_COLUMNS
from compute_load.model import sha256
from inf_diurnal_power import online_power
from inf_request_shapes import shape
from load_curve_scenarios_v3 import identity


RUN = 'inf-diurnal-20261101'


def calculate(old, mapping, parameters, intensity, case='central', mean_factor=1.):
    online = mapping.online_fraction.to_numpy()[old.source_hour_index]
    offline = mapping.offline_fraction.to_numpy()[old.source_hour_index]
    np.testing.assert_allclose(online + offline, old.allocated_gpu_fraction, rtol=1e-12)
    tdp = float(parameters.target_gpu_tdp_W)
    mean_w = tdp * float(parameters.allocated_gpu_ratio)
    idle_unallocated = float(parameters.idle_gpu_W)
    cards = int(parameters.gpu_devices_per_unit) * int(old.unit_count.iloc[0])
    reconstructed = cards / 1e6 * ((online + offline) * mean_w + (1 - online - offline) * idle_unallocated)
    np.testing.assert_allclose(reconstructed, old.gpu_it_mw, rtol=1e-12, atol=1e-12)
    per_card, calibration = online_power(intensity, online, tdp, mean_w * mean_factor, case)
    gpu = cards / 1e6 * (online * per_card + offline * mean_w + (1 - online - offline) * idle_unallocated)
    it = old.base_it_mw + old.cpu_it_mw + gpu
    weather = old[[c for c in WEATHER_COLUMNS if c in old]].copy()
    if old.pressure_assumed.all():
        weather = weather.drop(columns='pressure_pa')
    new = old.copy()
    thermal = CoolingModel(cooling_type='air').calculate(weather, it, float(old.installed_reference_it_mw.iloc[0]))
    for column in thermal:
        new[column] = thermal[column]
    new['gpu_it_mw'] = gpu
    new['online_allocated_gpu_fraction'] = online
    new['offline_allocated_gpu_fraction'] = offline
    new['request_intensity'] = intensity
    new['online_gpu_w_each'] = per_card
    new['online_gpu_it_mw'] = cards / 1e6 * online * per_card
    new['offline_gpu_it_mw'] = cards / 1e6 * offline * mean_w
    new['unallocated_gpu_it_mw'] = cards / 1e6 * (1 - online - offline) * idle_unallocated
    new['power_method'] = 'request_intensity_bounded_power_energy_calibrated'
    checks(new)
    np.testing.assert_allclose(new.gpu_it_mw, new.online_gpu_it_mw + new.offline_gpu_it_mw + new.unallocated_gpu_it_mw, rtol=1e-12)
    expected_delta = cards / 1e6 * online.sum() * mean_w * (mean_factor - 1)
    np.testing.assert_allclose(float(new.it_mw.sum() - old.it_mw.sum()), expected_delta, atol=1e-7, rtol=1e-10)
    return new, calibration


def variants():
    yield 'power-low', {'case': 'low'}
    yield 'power-high', {'case': 'high'}
    yield 'input-share-20pct', {'alpha': .2}
    yield 'input-share-80pct', {'alpha': .8}
    yield 'requests-only', {'mode': 'requests'}
    yield 'positive-days', {'policy': 'complete_positive_days'}
    yield 'mean-online-minus10pct', {'mean_factor': .9}
    yield 'mean-online-plus10pct', {'mean_factor': 1.1}
    for source in ('azure-2024-code', 'azure-2024-conv'):
        yield source + '-daily', {'source': source}
        yield source + '-weekly', {'source': source, 'mode': 'weekly'}
    for phase in range(7):
        yield f'burst-weekly-day0-{phase}', {'mode': 'weekly', 'phase': phase}
    for shift in (-8, 8):
        yield f'clock-shift-{shift}', {'shift': shift}


def fields(old):
    return {c: old[c].iloc[0] for c in ('scenario_id', 'scenario_zh', 'scenario', 'server_type', 'use',
             'reference_it_mw', 'source_days', 'cooling_type', 'location_id', 'location_zh', 'year')}


def build(args):
    project, storage = args.project.resolve(), args.storage.resolve()
    output = project / 'results' / RUN
    output.mkdir(parents=True, exist_ok=True)
    (output / 'hourly').mkdir(exist_ok=True)
    v3 = project / 'results/typical-load-curves-20261101-v3'
    manifest = json.loads((v3 / 'manifest.json').read_text('utf-8'))
    cooling = CoolingModel().metadata
    if cooling['cooling_code_sha256'] != manifest['cooling_model_sha256'] or cooling['parameters_sha256'] != manifest['cooling_asset_sha256']:
        raise ValueError('冷却模型或参数与v3冻结版本不一致')
    source_map = pd.read_csv(v3 / 'INF-annual-replay-map.csv').iloc[:4440]
    parameters = pd.read_csv(v3 / 'power-parameters.csv').set_index('server_type')
    parameters = parameters.loc[parameters.use.eq('INF')]
    summaries, comparisons, calibrations, sensitivity, sensitivity_days, profiles, week_profiles, hashes, inputs = [], [], [], [], [], [], [], {}, {}
    source_paths = [p for p in manifest['hourly_sha256'] if '-INF-' in p]
    if args.reference_only:
        source_paths = [p for p in source_paths if '/hourly/' in p]
    if len(source_paths) != (12 if args.reference_only else 300):
        raise ValueError('v3清单中的推理情境覆盖数量不完整')
    for number, relative in enumerate(source_paths):
        path = storage / relative
        digest = sha256(path)
        if digest != manifest['hourly_sha256'][relative]:
            raise ValueError('v3输入校验失败：' + relative)
        inputs[relative] = digest
        old = pd.read_parquet(path)
        time = pd.DatetimeIndex(pd.to_datetime(old.time_utc, utc=True))
        key, scale = old.server_type.iloc[0], int(old.reference_it_mw.iloc[0])
        for column, value in identity(key, 'INF', scale).items():
            old[column] = value
        intensity = shape(output, time)
        new, calibration = calculate(old, source_map, parameters.loc[key], intensity)
        meta = fields(old)
        for label, frame in [('v3', old), ('diurnal', new)]:
            summaries.append(summarize(frame, {**meta, 'version': label}))
        calibrations.append({**meta, **calibration, 'annual_it_delta_mwh': float(new.it_mw.sum() - old.it_mw.sum())})
        location, year, sid = meta['location_id'], int(meta['year']), meta['scenario_id']
        directory = output / 'hourly' if location == 'reference-grid' else storage / 'data/processed' / RUN / location
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f'{sid}-{year}.parquet'
        new.to_parquet(destination, index=False, compression='zstd')
        hashes[str(destination.relative_to(project if location == 'reference-grid' else storage))] = sha256(destination)
        if location == 'reference-grid':
            before, after = summaries[-2:]
            comparisons.append({**meta, **{f'{version}_{k}': record[k] for version, record in [('v3', before), ('new', after)]
                for k in record if k.startswith(('mean_', 'peak_', 'annual_', 'it_', 'facility_'))}})
            for label, frame in [('v3', old), ('diurnal', new)]:
                columns = ['it_mw', 'gpu_it_mw', 'facility_mw', 'cooling_mw']
                profiles.append(frame[columns].groupby(time.hour).mean().reset_index(names='hour').assign(**meta, version=label))
                week_profiles.append(frame[columns].groupby(time.dayofweek * 24 + time.hour).mean().reset_index(names='hour_of_week').assign(**meta, version=label))
            if scale == 10:
                sensitivity.append(summarize(new, {**meta, 'sensitivity': 'central'}))
                sensitivity_days.append(new[['it_mw', 'facility_mw']].groupby(time.hour).mean().reset_index(names='hour').assign(**meta, sensitivity='central'))
                for name, spec in variants():
                    spec = spec.copy()
                    case, mean_factor, shift = spec.pop('case', 'central'), spec.pop('mean_factor', 1.), spec.pop('shift', 0)
                    request = shape(output, time + pd.Timedelta(hours=shift), **spec)
                    changed, cal = calculate(old, source_map, parameters.loc[key], request, case, mean_factor)
                    sensitivity.append({**summarize(changed, {**meta, 'sensitivity': name}), **cal})
                    sensitivity_days.append(changed[['it_mw', 'facility_mw']].groupby(time.hour).mean().reset_index(names='hour').assign(**meta, sensitivity=name))
        if number % 12 == 11:
            print(json.dumps(dict(completed=number + 1, total=len(source_paths), location=location, year=year)), flush=True)
    pd.DataFrame(summaries).to_csv(output / 'annual-summary-all-locations.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(comparisons).to_csv(output / 'v3-comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(calibrations).to_csv(output / 'power-calibration.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(sensitivity).to_csv(output / 'sensitivity-summary.csv', index=False, encoding='utf-8-sig')
    pd.concat(sensitivity_days).to_csv(output / 'sensitivity-typical-day.csv', index=False, encoding='utf-8-sig')
    pd.concat(profiles).to_csv(output / 'typical-day.csv', index=False, encoding='utf-8-sig')
    pd.concat(week_profiles).to_csv(output / 'typical-week.csv', index=False, encoding='utf-8-sig')
    audit = dict(created_utc=datetime.now(timezone.utc).isoformat(), base_commit='ab07a41157442e837659535e1a33579ced5af8fa',
        dependencies=[34, 30], inputs_sha256=inputs, hourly_sha256=hashes, hourly_files=len(hashes),
        baseline_rows=len(summaries) // 2, new_hourly_rows=int(pd.DataFrame(summaries).query("version == 'diurnal'").hours.sum()),
        primary_shape='BurstGPT 119-day 24h input/output mean-normalized blend 0.5/0.5; target UTC hour phase=0',
        energy_policy='allocation-weighted online GPU energy equals v3 each location-year; offline, CPU, base and installed units unchanged',
        weekly_policy='Burst calendar unknown: daily central; seven weekly phase sensitivities',
        aggregate_inputs_sha256={str(p.relative_to(project)): sha256(p) for p in
            [v3 / 'manifest.json', v3 / 'INF-annual-replay-map.csv', v3 / 'power-parameters.csv',
             output / 'request-input-audit.json', output / 'request-shape-24h.csv', output / 'request-shape-168h.csv']},
        cooling=CoolingModel().metadata)
    (output / 'manifest.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--storage', type=Path, required=True)
    parser.add_argument('--reference-only', action='store_true')
    build(parser.parse_args())
