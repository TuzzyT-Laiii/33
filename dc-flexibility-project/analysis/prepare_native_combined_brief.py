import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_scenario_typical_days_v4 import split_layers
from compute_load.model import sha256
from load_curve_scenarios_v3 import catalogue
from native_combined_power import RUN
from reselect_scenario_typical_days import daily_metrics, select_day, audit_day, RULE
from trn_power_models import idle_power


def run(project, resources):
    output = project / 'results' / RUN
    typical = output / 'selected-days'
    typical.mkdir(exist_ok=True)
    shape = json.loads((project / 'results/trn-power-model-20261101/cpu-shape.json').read_text('utf-8'))
    mit = json.loads((project / 'results/typical-load-curves-20261101-v3/mit-calibration.json').read_text('utf-8'))
    ratios = json.loads((project / 'results/typical-load-curves-20261029-v2/mit-power-ratio-summary.json').read_text('utf-8'))['transfer_distribution']
    configs = catalogue(project)
    placements = pd.read_csv(output / 'placement-summary.csv').query('reference_it_mw == 10')
    summary = pd.read_csv(output / 'native-summary.csv').query('reference_it_mw == 10').set_index('scenario_id')
    previous = json.loads((project / 'results/scenario-typical-day-by-task-20261101-v6/summary.json').read_text('utf-8'))
    resource_manifest = json.loads((project / 'results/scenario-typical-day-by-task-20261101-v4/manifest.json').read_text('utf-8'))
    expected_resources = {item['role']: item['sha256'] for item in resource_manifest['input_files']}
    previous_rows = {r['scenario']: r for r in previous['scenarios']}
    records, selections, comparisons, audits, inputs = [], {}, [], [], {}
    for use in ('HPC', 'CLD', 'SAI', 'TRN', 'INF'):
        keys = ['A'] if use in ('HPC', 'CLD') else ['B', 'C', 'D']
        reference_id = f'{keys[-1]}-{use}-10MW'
        options = placements.loc[placements.scenario_id.eq(reference_id)]
        full = options.loc[options.hours.eq(options.native_hours)].copy()
        median = float(options.mean_facility_mw.median())
        chosen = full.assign(distance=(full.mean_facility_mw - median).abs()).sort_values(['distance', 'placement'], kind='stable').iloc[0]
        placement = chosen.placement
        full_frame = pd.read_parquet(output / 'hourly' / f'{reference_id}-2025-{placement}.parquet')
        count = len(full_frame) // 24
        day, scores, selection = select_day(daily_metrics(full_frame, count))
        scores.to_csv(typical / f'{use}-day-selection.csv', index=False, encoding='utf-8-sig')
        if use in ('CLD', 'INF'):
            source_date = f'相对第{day + (2 if use == "CLD" else 0)}天'
            weekday = '星期未知'
        else:
            date = pd.Timestamp(full_frame.source_clock.iloc[day * 24])
            source_date = date.strftime('%Y-%m-%d')
            weekday = '星期' + '一二三四五六日'[date.dayofweek]
        selections[use] = {**selection, 'day_index': day, 'source_date': source_date, 'placement': placement,
            'placement_policy': '先限定完整原生窗口，再取设施平均最接近12次放置中位数的放置；同需求按GPU主导型共用放置与源日'}
        cache_path = resources / f'{use}-resources.parquet'
        if sha256(cache_path) != expected_resources[use + '_local_classified_resource_hours']:
            raise ValueError('分类资源小时与已核验的#31指纹不符：' + str(cache_path))
        resource = pd.read_parquet(cache_path)
        if len(resource) != len(full_frame):
            raise ValueError('分类资源小时与原生窗口长度不符')
        inputs[str(cache_path)] = sha256(cache_path)
        resource = resource.iloc[day*24:(day+1)*24].reset_index(drop=True)
        for key in keys:
            scenario = key + '-' + use
            path = output / 'hourly' / f'{scenario}-10MW-2025-{placement}.parquet'
            full_frame = pd.read_parquet(path)
            frame = full_frame.iloc[day*24:(day+1)*24].reset_index(drop=True)
            np.testing.assert_allclose(resource.filter(regex='^cpu:').sum(axis=1), frame.u_cpu_proxy, atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(resource.filter(regex='^gpu:').sum(axis=1), frame.allocated_gpu_fraction, atol=1e-10, rtol=1e-10)
            units = int(frame.unit_count.iloc[0])
            frame['idle_it_mw'] = idle_power(configs[key], key, use, mit) * units / 1e6
            layers = split_layers(frame, resource, configs[key], key, use, ratios, mit, shape)
            name = str(frame.scenario_zh.iloc[0])
            fields = dict(scenario=scenario, scenario_name=name, source_date=source_date, weekday=weekday)
            hourly = frame.assign(**fields, hour=np.arange(24))
            hourly.to_csv(typical / f'{scenario}-hourly.csv', index=False, encoding='utf-8-sig')
            parts = [pd.DataFrame({**fields, 'hour': range(24), 'task_type': label, 'facility_mw': values,
                'it_mw': np.zeros(24) if label == '冷却等非 IT' else values}) for label, values in layers.items()]
            pd.concat(parts).to_csv(typical / f'{scenario}-by-task.csv', index=False, encoding='utf-8-sig')
            np.testing.assert_allclose(sum(layers.values()), frame.facility_mw, atol=1e-8)
            metrics = daily_metrics(full_frame, len(full_frame) // 24)
            metrics_row = metrics.iloc[day]
            records.append({**fields, 'placement': placement, 'model_weather_date': str(frame.time_utc.iloc[0]),
                'daily_facility_mwh': float(frame.facility_mw.sum()), 'daily_it_mwh': float(frame.it_mw.sum()),
                'peak_mw': float(frame.facility_mw.max()), 'minimum_mw': float(frame.facility_mw.min()),
                'peak_valley_ratio': float(frame.facility_mw.max() / frame.facility_mw.min()),
                'rmse_mw': float(metrics_row.rmse_mw), 'energy_deviation_pct': float(metrics_row.energy_deviation_pct),
                'native_window_facility_mean_mw': float(full_frame.facility_mw.mean()), 'native_days': count})
            audits.append({**fields, **audit_day(full_frame, metrics, day)})
            old = previous_rows[scenario]
            comparisons.append({**fields, 'old_source_date': old['source_date'], 'new_source_date': source_date,
                'old_selected_peak_mw': old['peak_mw'], 'new_selected_peak_mw': float(frame.facility_mw.max()),
                'old_selected_ratio': old['peak_valley_ratio'], 'new_selected_ratio': float(frame.facility_mw.max() / frame.facility_mw.min()),
                'old_selected_daily_mwh': old['daily_facility_mwh'], 'new_selected_daily_mwh': float(frame.facility_mw.sum()),
                'new_window_mean_it_mw': float(full_frame.it_mw.mean()), 'chapter5_source_mean_it_mw': float(summary.loc[scenario + '-10MW', 'mean_it_mw']),
                'window_it_mean_absolute_difference_mw': abs(float(full_frame.it_mw.mean()) - float(summary.loc[scenario + '-10MW', 'mean_it_mw'])),
                'selected_hourly_maximum_difference_mw': float((hourly.facility_mw.to_numpy() - full_frame.facility_mw.iloc[day*24:(day+1)*24].to_numpy()).max()),
                'comparison_scope': '前版第四章与本版各自选日；同时改变功率模型、天气映射及选日，不能解释为单因素效应'})
    (typical / 'summary.json').write_text(json.dumps({'selection_rule': RULE, 'selections': selections, 'scenarios': records}, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(records).to_csv(typical / 'summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(comparisons).to_csv(output / 'chapter4-comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(audits).to_csv(typical / 'selected-day-diagnostics.csv', index=False, encoding='utf-8-sig')
    (typical / 'resource-input-sha256.json').write_text(json.dumps(inputs, ensure_ascii=False, indent=2), encoding='utf-8')
    facts_path = project / 'results/scenario-typical-day-by-task-20261101-v6/brief-facts.json'
    facts = json.loads(facts_path.read_text('utf-8'))
    calculation = json.loads((output / 'manifest.json').read_text('utf-8'))
    training_profile = project / 'results/trn-power-model-20261101/diurnal-acme/transfer-profile.csv'
    profile = pd.read_csv(training_profile).shape_central
    facts['power']['commits'].update(calculation['dependency_commits'])
    facts['power']['commits']['31'] = 'c7c3df07adc481f6209de418191117734c43ac03'
    facts['power'].update(TRN_GPU_TDP_ratio_interpretation='原生90天分配GPU小时加权均值',
        TRN_diurnal_GPU_TDP_range=[float(profile.min()), float(profile.max())],
        training_profile_sha256=sha256(training_profile), power_model_version=RUN)
    facts['business_statistics_source_sha256'] = sha256(facts_path)
    (typical / 'brief-facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--resources', type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.resources)
