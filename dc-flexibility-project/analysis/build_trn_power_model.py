import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_typical_load_curves_v3 import checks, summarize, save_json
from compute_load.cooling import CoolingModel, WEATHER_COLUMNS
from compute_load.model import sha256
from load_curve_scenarios_v3 import catalogue, identity, power as power_v3
from trn_power_models import idle_power, power, TRN_RATIOS


RUN = 'trn-power-model-20261101'
BASE = 'typical-load-curves-20261101-v3'


def calculate(old, config, key, use, ratios, mit, shape, **options):
    cpu, gpu = old.u_cpu_proxy.to_numpy(), old.allocated_gpu_fraction.to_numpy()
    unit = power(config, key, use, cpu, gpu, ratios, mit, shape, **options)
    count = int(old.unit_count.iloc[0])
    weather = old[[col for col in WEATHER_COLUMNS if col in old]].copy()
    # 沿用 v3 的气压是否假设标记，避免把固定假设转写成观测气压。
    if old.pressure_assumed.all():
        weather = weather.drop(columns='pressure_pa')
    frame = CoolingModel().calculate(weather, unit['unit_w'] * count / 1e6,
                                     float(old.installed_reference_it_mw.iloc[0]))
    for column in old.columns.difference(frame.columns):
        frame[column] = old[column].to_numpy()
    for name in ('base', 'cpu', 'gpu'):
        frame[name + '_it_mw'] = unit[name + '_w'] * count / 1e6
    frame['idle_it_mw'] = idle_power(config, key, use, mit, options.get('idle_case', 'central')) * count / 1e6
    frame['task_it_mw'] = frame.it_mw - frame.idle_it_mw
    frame['power_model_version'] = RUN
    frame['allocated_gpu_power_ratio'] = TRN_RATIOS[options.get('training_case', 'central')] if use == 'TRN' else (mit['measured_power_to_tdp_ratio'] if use == 'SAI' else ratios['p50'] if key != 'A' else np.nan)
    checks(frame)
    if (frame.task_it_mw < -1e-9).any():
        raise ValueError('任务增量功率不能为负')
    return frame


def summary(frame, fields):
    values = summarize(frame, fields)
    values['mean_idle_it_mw'] = float(frame.idle_it_mw.mean())
    values['mean_task_it_mw'] = float(frame.task_it_mw.mean())
    values['task_share_of_it_pct'] = float(100 * frame.task_it_mw.sum() / frame.it_mw.sum())
    values['idle_share_of_it_pct'] = 100 - values['task_share_of_it_pct']
    return values


def build(args):
    project = args.project.resolve()
    baseline = project / 'results' / BASE
    output = project / 'results' / RUN
    hourly = output / 'hourly'
    hourly.mkdir(parents=True, exist_ok=True)
    configs = catalogue(project)
    ratios = json.loads((project / 'results/typical-load-curves-20261029-v2/mit-power-ratio-summary.json').read_text('utf-8'))['transfer_distribution']
    mit = json.loads((baseline / 'mit-calibration.json').read_text('utf-8'))
    shape = json.loads((output / 'cpu-shape.json').read_text('utf-8'))
    source_manifest = json.loads((baseline / 'manifest.json').read_text('utf-8'))
    summaries, comparisons, sensitivities, days, inputs, hashes = [], [], [], [], [], {}
    for path in sorted((baseline / 'hourly').glob('*.parquet')):
        digest = sha256(path)
        if digest != source_manifest['hourly_sha256'][f'results/{BASE}/hourly/{path.name}']:
            raise ValueError('v3 小时输入 SHA-256 不符：' + path.name)
        old = pd.read_parquet(path)
        key, use, scale = old.server_type.iloc[0], old.use.iloc[0], int(old.reference_it_mw.iloc[0])
        config, fields = configs[key], identity(key, use, scale)
        # 对冻结 v3 输入重新计算完整链路，检查比较基准和分项口径。
        original = power_v3(config, key, use, old.u_cpu_proxy.to_numpy(), old.allocated_gpu_fraction.to_numpy(), ratios, mit)
        for name in ('base', 'cpu', 'gpu'):
            np.testing.assert_allclose(original[name + '_w'] * old.unit_count.iloc[0] / 1e6, old[name + '_it_mw'], rtol=1e-12, atol=1e-10)
        old['idle_it_mw'] = idle_power(config, key, use, mit) * old.unit_count.iloc[0] / 1e6
        old['task_it_mw'] = old.it_mw - old.idle_it_mw
        current = calculate(old, config, key, use, ratios, mit, shape)
        before, after = summary(old, {**fields, 'version': 'v3'}), summary(current, {**fields, 'version': RUN})
        summaries.append(after)
        comparison = fields.copy()
        for name in ('mean_it_mw', 'peak_it_mw', 'mean_facility_mw', 'peak_facility_mw', 'it_diurnal_peak_valley',
                     'facility_diurnal_peak_valley', 'task_share_of_it_pct', 'mean_idle_it_mw', 'annual_pue'):
            comparison['v3_' + name], comparison['new_' + name] = before[name], after[name]
            comparison[name + '_change_pct'] = 100 * (after[name] / before[name] - 1)
        comparisons.append(comparison)
        for version, frame in [('v3', old), ('new', current)]:
            day = frame[['it_mw', 'facility_mw', 'idle_it_mw', 'task_it_mw']].groupby(pd.DatetimeIndex(frame.time_utc).hour).mean().reset_index(names='hour_utc')
            days.append(day.assign(version=version, **fields))
        target = hourly / path.name
        current.to_parquet(target, index=False)
        hashes[path.name] = sha256(target)
        inputs.append({'file': path.name, 'sha256': digest})
        if scale == 10:
            variants = [('central', {})]
            if use == 'TRN':
                variants += [(f'gpu_{case}', {'training_case': case}) for case in ('low', 'high')]
                variants += [('gpu_only', {'cpu_method': 'v3'})]
            variants += [(f'idle_{case}', {'idle_case': case}) for case in ('low', 'high')]
            for label, options in variants:
                frame = calculate(old, config, key, use, ratios, mit, shape, **options)
                sensitivities.append(summary(frame, {**fields, 'sensitivity': label}))
        print(fields['scenario_id'], after['mean_it_mw'], flush=True)
    if len(summaries) != 44:
        raise ValueError('参考网格需要44个情境')
    pd.DataFrame(summaries).to_csv(output / 'annual-summary.csv', index=False)
    pd.DataFrame(comparisons).to_csv(output / 'v3-comparison.csv', index=False)
    pd.DataFrame(sensitivities).to_csv(output / 'sensitivity-summary.csv', index=False)
    pd.concat(days).to_csv(output / 'typical-day.csv', index=False)
    save_json(output / 'manifest.json', {'utc': datetime.now(timezone.utc).isoformat(), 'base_commit': 'ab07a41157442e837659535e1a33579ced5af8fa',
              'scope': 'reference-grid 2025, 44 scenarios, frozen v3 demand/weather/unit count; see all-locations-manifest.json for the additional 1056 city-year scenarios',
              'inputs': inputs, 'hourly_sha256': hashes, 'cpu_shape_sha256': sha256(output / 'cpu-shape.json'),
              'source_v3_manifest_sha256': sha256(baseline / 'manifest.json'),
              'scripts': {name: sha256(project / 'analysis' / name) for name in
                          ('trn_power_models.py', 'build_trn_power_model.py', 'typical_power_models.py',
                           'typical_power_models_v2.py', 'load_curve_scenarios_v3.py', 'compute_load/cooling.py',
                           'compute_load/assets/cooling-20261028-v2.json')},
              'inf_merge_note': 'INF GPU power and demand identical to #34; only CPU response changes. Reapply shared CPU response to inference diurnal PR and regenerate combined outputs.',
              'idle_central_changed': False, 'training_power_ratio': TRN_RATIOS})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    build(parser.parse_args())
