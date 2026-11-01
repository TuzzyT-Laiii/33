import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_typical_load_curves_v3 import checks
from compute_load.cooling import CoolingModel, WEATHER_COLUMNS
from compute_load.model import sha256
from load_curve_scenarios_v3 import catalogue, TDP
from trn_power_models import idle_power, power


RUN = 'trn-power-model-20261101/diurnal-acme'
NATIVE = 'typical-load-curves-20261101-v3-native'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def recalculate(source, config, ratios, mit, cpu_shape, profile):
    key = source.server_type.iloc[0]
    unit = power(config, key, 'TRN', source.u_cpu_proxy.to_numpy(), source.allocated_gpu_fraction.to_numpy(),
                 ratios, mit, cpu_shape, training_ratio=profile)
    count = int(source.unit_count.iloc[0])
    weather = source[[name for name in WEATHER_COLUMNS if name in source]].copy()
    if source.pressure_assumed.all():
        weather = weather.drop(columns='pressure_pa')
    current = CoolingModel().calculate(weather, unit['unit_w'] * count / 1e6,
        float(source.installed_reference_it_mw.iloc[0]), full_year=False)
    for name in source.columns.difference(current.columns):
        current[name] = source[name].to_numpy()
    for name in ('base', 'cpu', 'gpu'):
        current[name + '_it_mw'] = unit[name + '_w'] * count / 1e6
    current['allocated_gpu_power_ratio'] = profile
    current['idle_it_mw'] = idle_power(config, key, 'TRN', mit) * count / 1e6
    current['task_it_mw'] = current.it_mw - current.idle_it_mw
    current['power_model_version'] = RUN
    checks(current)
    expected = config['gpu_devices_per_unit'] * count / 1e6 * (
        source.allocated_gpu_fraction * TDP[key] * profile + (1 - source.allocated_gpu_fraction) * (70. if key == 'D' else 50.))
    np.testing.assert_allclose(expected, current.gpu_it_mw, rtol=1e-12, atol=1e-12)
    for name in ('source_clock', 'source_hour_index', 'allocated_gpu_fraction', 'u_cpu_proxy', 'unit_count', 'time_utc', 'temperature_c'):
        pd.testing.assert_series_equal(source[name], current[name], check_names=False)
    return current


def metrics(frame, case):
    row = {name: frame[name].iloc[0] for name in ('scenario_id', 'scenario_zh', 'server_type', 'reference_it_mw', 'location_id', 'weather_year')}
    row.update(case=case, hours=len(frame), source_start=str(frame.source_clock.iloc[0]), source_end=str(frame.source_clock.iloc[-1]),
        allocated_power_ratio_gpu_hour_mean=float(np.average(frame.allocated_gpu_power_ratio, weights=frame.allocated_gpu_fraction)),
        task_share_of_it_pct=float(100 * frame.task_it_mw.sum() / frame.it_mw.sum()),
        window_pue=float(frame.facility_mw.sum() / frame.it_mw.sum()))
    for name in ('it', 'facility'):
        values = frame[name + '_mw']
        day = values.groupby(frame.hour_of_source_day).mean()
        week = values.groupby(frame.hour_of_source_week).mean()
        row.update({f'mean_{name}_mw': float(values.mean()), f'peak_{name}_mw': float(values.max()),
            f'minimum_{name}_mw': float(values.min()), f'window_{name}_mwh': float(values.sum()),
            f'{name}_diurnal_peak_valley': float(day.max() / day.min()),
            f'{name}_weekly_peak_valley': float(week.max() / week.min())})
    return row


def main(args):
    project, inputs = args.project.resolve(), args.input_project.resolve()
    output = project / 'results' / RUN
    profile = pd.read_csv(output / 'transfer-profile.csv')
    if not np.array_equal(profile.hour_utc, np.arange(24)):
        raise ValueError('迁移形状必须显式列出24小时及各小时证据状态')
    configs = catalogue(project)
    ratios = json.loads((project / 'results/typical-load-curves-20261029-v2/mit-power-ratio-summary.json').read_text('utf-8'))['transfer_distribution']
    mit = json.loads((project / 'results/typical-load-curves-20261101-v3/mit-calibration.json').read_text('utf-8'))
    shape = json.loads((project / 'results/trn-power-model-20261101/cpu-shape.json').read_text('utf-8'))
    manifest_path = inputs / 'results' / NATIVE / 'manifest.json'
    manifest = json.loads(manifest_path.read_text('utf-8'))
    summaries, days, weeks, hashes, sources = [], [], [], {}, {}
    validation = []
    for relative, expected_hash in manifest['hourly_sha256'].items():
        if '-TRN-' not in relative or (args.reference_only and '/hourly/' not in relative):
            continue
        path = inputs / relative
        digest = sha256(path)
        if digest != expected_hash:
            raise ValueError('原生小时输入哈希不符：' + str(path))
        old = pd.read_parquet(path)
        clock = pd.DatetimeIndex(pd.to_datetime(old.source_clock, utc=True))
        if len(old) != 2160 or clock[0] != pd.Timestamp('2023-05-18', tz='UTC') or clock[-1] != pd.Timestamp('2023-08-15T23:00:00Z'):
            raise ValueError('TRN 必须保留一次原生90天窗口')
        key = old.server_type.iloc[0]
        position = old.hour_of_source_day.to_numpy(int)
        variants = {'constant_080': np.full(len(old), .8)}
        variants.update({name: profile[name].to_numpy()[position] for name in profile.columns if name.startswith(('shape_', 'direct_'))})
        for case, hourly_ratio in variants.items():
            current = recalculate(old, configs[key], ratios, mit, shape, hourly_ratio)
            record = metrics(current, case)
            summaries.append(record)
            if old.location_id.iloc[0] == 'reference-grid':
                fields = {name: record[name] for name in ('scenario_id', 'scenario_zh', 'server_type', 'reference_it_mw', 'case')}
                for grouping, table in [('hour_of_source_day', days), ('hour_of_source_week', weeks)]:
                    grouped = current.groupby(grouping)[['it_mw', 'facility_mw', 'gpu_it_mw', 'allocated_gpu_power_ratio']].mean().reset_index()
                    table.append(grouped.assign(**fields))
            target = output / 'hourly' / str(old.location_id.iloc[0]) / case / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            current.to_parquet(target, index=False, compression='zstd')
            hashes[str(target.relative_to(output))] = sha256(target)
            validation.append({'source': relative, 'case': case, 'hours': len(current),
                'maximum_heat_balance_residual_mw': float(current.heat_balance_residual_mw.abs().max()),
                'maximum_gpu_ratio': float(current.allocated_gpu_power_ratio.max()),
                'minimum_gpu_ratio': float(current.allocated_gpu_power_ratio.min())})
        sources[relative] = digest
        print(old.scenario_id.iloc[0], old.location_id.iloc[0], old.weather_year.iloc[0], flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / 'window-summary-all-locations.csv', index=False)
    reference = summary.loc[summary.location_id.eq('reference-grid')].copy()
    if reference.scenario_id.nunique() != 12:
        raise ValueError('需要B/C/D各四档规模的12个TRN情境')
    reference.to_csv(output / 'window-summary.csv', index=False)
    pd.concat(days, ignore_index=True).to_csv(output / 'typical-day.csv', index=False)
    pd.concat(weeks, ignore_index=True).to_csv(output / 'typical-week.csv', index=False)
    baseline = reference.loc[reference.case.eq('constant_080')].set_index('scenario_id')
    comparison = reference.loc[reference.case.eq('shape_central')].copy().set_index('scenario_id')
    comparison = comparison[['scenario_zh', 'server_type', 'reference_it_mw', 'hours']]
    for name in ('mean_it_mw', 'peak_it_mw', 'minimum_it_mw', 'mean_facility_mw', 'peak_facility_mw', 'minimum_facility_mw',
                 'it_diurnal_peak_valley', 'it_weekly_peak_valley', 'facility_diurnal_peak_valley', 'facility_weekly_peak_valley'):
        comparison['constant_' + name] = baseline[name]
        comparison['shape_' + name] = reference.loc[reference.case.eq('shape_central')].set_index('scenario_id')[name]
    comparison.reset_index().to_csv(output / 'comparison.csv', index=False)
    write_json(output / 'native-calculation-audit.json', {'utc': datetime.now(timezone.utc).isoformat(),
        'source_native_commit': 'c836bbaf095dfe4660bad8abe9ab9e0ddee4cd6c', 'source_manifest_sha256': sha256(manifest_path),
        'inputs_sha256': sources, 'output_hourly_sha256': hashes, 'transfer_profile_sha256': sha256(output / 'transfer-profile.csv'),
        'scenarios': int(reference.scenario_id.nunique()), 'location_years': int(summary[['location_id', 'weather_year']].drop_duplicates().shape[0]),
        'variants': list(variants), 'hourly_rows': int(sum(row['hours'] for row in validation)),
        'energy_scope': '2160 native hours per scenario and weather placement; no annual replay or annual energy extrapolation',
        'code_sha256': {name: sha256(project / 'analysis' / name) for name in ('build_trn_diurnal.py', 'trn_power_models.py')},
        'checks': validation})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--input-project', type=Path, required=True)
    parser.add_argument('--reference-only', action='store_true')
    main(parser.parse_args())
