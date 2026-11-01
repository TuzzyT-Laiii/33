import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa

from build_scenario_typical_days import save_json
from build_scenario_typical_days_v4 import split_layers
from compute_load.model import sha256
from load_curve_scenarios_v3 import catalogue, DAYS
from plot_scenario_typical_days_v4 import plot_all


RUN = 'scenario-typical-day-by-task-20261101-v5'
BASE = 'scenario-typical-day-by-task-20261101-v4'
PRIMARY_WINDOW = .02
FALLBACK_WINDOW = .05
MINIMUM_CANDIDATES = 5
RULE_CAPTION = '日电量中位数±2%内候选，少于5天放宽至±5%；候选中与全部完整日平均曲线RMSE最小'
RULE = RULE_CAPTION + '；RMSE相同则按电量绝对偏差、源日期依次排序。科研AI、训练、推理以GPU主导型选日，三种规格共用日期。'


def daily_metrics(frame, days):
    facility = frame.facility_mw.to_numpy()[:days * 24].reshape(days, 24)
    it = frame.it_mw.to_numpy()[:days * 24].reshape(days, 24)
    energy = facility.sum(axis=1)
    median = float(np.median(energy))
    if median <= 0:
        raise ValueError('完整日中位电量必须为正')
    return pd.DataFrame({'day_index': np.arange(days), 'facility_mwh': energy, 'median_mwh': median,
        'energy_deviation_pct': 100 * (energy / median - 1),
        'absolute_distance_mwh': np.abs(energy - median),
        'rmse_mw': np.sqrt(np.mean((facility - facility.mean(axis=0)) ** 2, axis=1)),
        'peak_mw': facility.max(axis=1), 'minimum_mw': facility.min(axis=1),
        'peak_valley_ratio': facility.max(axis=1) / facility.min(axis=1),
        'max_hourly_IT_step_over_mean': np.max(np.abs(np.diff(it, axis=1)), axis=1) / it.mean(),
        'low_IT_hours': np.sum(it < .6 * it.mean(axis=0), axis=1)})


def select_day(metrics):
    primary = metrics.energy_deviation_pct.abs().le(100 * PRIMARY_WINDOW + 1e-10)
    width = PRIMARY_WINDOW if int(primary.sum()) >= MINIMUM_CANDIDATES else FALLBACK_WINDOW
    candidates = metrics.energy_deviation_pct.abs().le(100 * width + 1e-10)
    if not candidates.any():
        raise ValueError('中位电量±5%以内没有完整日，无法按既定规则选日')
    chosen = int(metrics.loc[candidates].sort_values(['rmse_mw', 'absolute_distance_mwh', 'day_index'], kind='stable').iloc[0].day_index)
    scores = metrics.assign(candidate_at_2pct=primary, candidate=candidates, applied_window_pct=width * 100,
        selected=metrics.day_index.eq(chosen))
    return chosen, scores, {'complete_days': len(metrics), 'candidate_count_at_2pct': int(primary.sum()),
        'candidate_count': int(candidates.sum()), 'applied_window_pct': width * 100,
        'minimum_candidates': MINIMUM_CANDIDATES, 'selection_rule': RULE}


def audit_day(frame, metrics, index):
    daily = frame.it_mw.to_numpy()[:len(metrics) * 24].reshape(-1, 24)
    low = daily[index] < .6 * daily.mean(axis=0)
    run, longest = 0, 0
    for value in low:
        run = run + 1 if value else 0
        longest = max(longest, run)
    q1, q3 = np.quantile(metrics.rmse_mw, [.25, .75])
    rmse_limit = float(q3 + 1.5 * (q3-q1))
    flags = []
    if metrics.iloc[index].max_hourly_IT_step_over_mean > .30:
        flags.append('相邻小时IT跳变超过全期IT均值30%')
    if longest >= 3:
        flags.append('连续至少3小时IT低于全期同小时均值60%')
    if metrics.iloc[index].rmse_mw > rmse_limit:
        flags.append('RMSE超过Q3+1.5IQR')
    unresolved = int(frame.source_unresolved.iloc[index*24:(index+1)*24].sum())
    if unresolved:
        flags.append('来源存在未解决质量标记')
    return {'audit_flags': '；'.join(flags), 'longest_low_IT_hours': longest,
        'max_IT_step_fraction': float(metrics.iloc[index].max_hourly_IT_step_over_mean),
        'rmse_outlier_limit_mw': rmse_limit, 'unresolved_hours': unresolved}


def report(output, summary, comparison, diagnostics):
    lines = ['# 各情境典型日：中位电量窗口内按形状选日', '', '![典型日总览](overview.png)', '',
        RULE, '',
        '日电量和RMSE均由设施功率计算；RMSE参照全部完整日的平均24小时曲线。±2%候选少于5天时只放宽到±5%，若放宽后仍少于5天则使用实际候选；没有候选时明确报错。数值边界容差为1e-10个百分点。', '',
        '模型、需求、天气及每小时功率保持v4组合模型（#35推理GPU映射＋#36训练与CPU），参考IT 10MW。直接读取并核对v4全年小时文件SHA-256，不重算或改写全年负荷，不访问流式进程和STOP。', '',
        '## 新旧典型日', '',
        '表中箭头为v4选日→新规则；电量单位MWh，RMSE单位MW，偏差为本情境自身中位数的有符号百分比。GPU需求统一按GPU主导型候选和RMSE选日，B/C自身偏差与参考偏差另列于CSV。', '',
        '| 情境 | 日期 | 日电量 | 中位数偏差% | RMSE | 日内最高/最低比 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in comparison:
        lines.append(f"| {row['scenario_name']} | {row['old_date']} → {row['new_date']} | {row['old_energy_mwh']:.2f} → {row['new_energy_mwh']:.2f} | {row['old_deviation_pct']:+.2f} → {row['new_deviation_pct']:+.2f} | {row['old_rmse_mw']:.3f} → {row['new_rmse_mw']:.3f} | {row['old_peak_valley_ratio']:.3f} → {row['new_peak_valley_ratio']:.3f} |")
    lines += ['', '## 候选数量与筛查', '',
        '| 需求 | ±2%候选数 | 实际窗口 | 实际候选数 | 新日期 |', '| --- | ---: | ---: | ---: | --- |']
    for use, row in summary['selections'].items():
        lines.append(f"| {row['reference_scenario_name']} | {row['candidate_count_at_2pct']} | ±{row['applied_window_pct']:g}% | {row['candidate_count']} | {row['source_date']} |")
    lines += ['', '筛查采用固定提示条件：相邻小时IT变化超过全期IT均值的30%；连续至少3小时低于全期同小时IT均值的60%；设施曲线RMSE高于全期Q3+1.5IQR；来源有未解决质量标记。筛查不参与选日、不剔除候选，也不据此认定发生维护或停机。', '']
    for version in ('old', 'new'):
        flagged = [row for row in diagnostics if row['version'] == version and row['audit_flags']]
        lines.append(('旧日' if version == 'old' else '新日') + '提示：' + ('；'.join(row['scenario_name'] + '（' + row['source_date'] + '）：' + row['audit_flags'] for row in flagged) if flagged else '11个情境均未触发上述条件。'))
    lines += ['', '## 数据、图件与复现', '',
        'day-selection-comparison.csv为11情境新旧对照；selected-day-diagnostics.csv为22个新旧日的指标和提示；五份*-day-selection.csv保留全部日、窗口、候选及选中标记。每情境另有PNG（300dpi）/SVG、24行hourly.csv及分任务by-task.csv。年度汇总和参数表按v4原字节复制，v4选日数据和图件原样保留。', '',
        '敏感性参数、功率依据与原始来源见[固定的v4模型说明](../scenario-typical-day-by-task-20261101-v4/README.md)。ASI、Alibaba实际日历未知；请求相位为合成映射。MIT仍为六月完成版，DNN含官方标签和推断，逐作业数据仅留本机。', '',
        '```powershell', "$env:OPENBLAS_NUM_THREADS='1'", "$env:OMP_NUM_THREADS='1'",
        'python dc-flexibility-project/analysis/reselect_scenario_typical_days.py', '```', '',
        '脚本依赖沿用requirements-scenario-typical-days-v3.txt。需要已有v4冻结全年小时文件及本地分类资源小时汇总。简表继续由docs/job-flexibility-brief/build.js生成；Word转PDF逐页检查后运行verify_typical_day_reselection.py。', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def build(project):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    old_root, output = project / 'results' / BASE, project / 'results' / RUN
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((old_root / 'manifest.json').read_text(encoding='utf-8'))
    old_summary = json.loads((old_root / 'summary.json').read_text(encoding='utf-8'))
    inputs = {item['role']: item for item in manifest['input_files']}
    for use in ('HPC', 'CLD', 'SAI', 'TRN', 'INF'):
        item = inputs[use + '_local_classified_resource_hours']
        assert sha256(Path(item['file'])) == item['sha256']
    configs = catalogue(project)
    ratios = json.loads(Path(inputs['frozen_GPU_ratios']['file']).read_text(encoding='utf-8'))['transfer_distribution']
    mit = json.loads(Path(inputs['MIT_calibration']['file']).read_text(encoding='utf-8'))
    shape = json.loads(Path(inputs['PR36_CPU_shape']['file']).read_text(encoding='utf-8'))
    rows_by_id = {row['scenario']: row for row in old_summary['scenarios']}
    records, selections, comparison, diagnostics, checks, full_hashes = [], {}, [], [], {}, {}
    for use in ('HPC', 'CLD', 'SAI', 'TRN', 'INF'):
        keys = ['A'] if use in ('HPC', 'CLD') else ['B', 'C', 'D']
        curves = {}
        for key in keys:
            filename = f'{key}-{use}-10MW-2025.parquet'
            path = project / 'data/processed' / BASE / filename
            assert sha256(path) == manifest['local_annual_hourly_sha256'][filename]
            full_hashes[str(path.resolve())] = sha256(path)
            curves[key] = pd.read_parquet(path)
        reference = keys[-1] + '-' + use
        metrics = daily_metrics(curves[keys[-1]], DAYS[use])
        day, scores, selection = select_day(metrics)
        old_scores = pd.read_csv(old_root / (use + '-day-selection.csv'))
        np.testing.assert_allclose(metrics.facility_mwh, old_scores.facility_mwh, rtol=1e-12)
        np.testing.assert_allclose(metrics.rmse_mw, old_scores.rmse_mw, rtol=1e-12)
        scores.insert(0, 'source_day', old_scores.source_day)
        scores.to_csv(output / (use + '-day-selection.csv'), index=False, encoding='utf-8-sig')
        source_date = str(scores.source_day.iloc[day]) if use in ('CLD', 'INF') else pd.Timestamp(scores.source_day.iloc[day]).strftime('%Y-%m-%d')
        weekday = '未知' if use in ('CLD', 'INF') else '星期' + '一二三四五六日'[pd.Timestamp(source_date).dayofweek]
        selections[use] = {**selection, 'day_index': day, 'source_date': source_date, 'weekday': weekday,
            'reference_scenario': reference, 'reference_scenario_name': rows_by_id[reference]['scenario_name'],
            'model_weather_date': str(curves[keys[-1]].time_utc.iloc[day*24].date())}
        cache = pd.read_parquet(inputs[use + '_local_classified_resource_hours']['file'])
        resource = cache.iloc[day*24:(day+1)*24].reset_index(drop=True)
        prefix = 'cpu:' if use in ('HPC', 'CLD') else 'gpu:'
        amounts = resource.filter(regex='^' + prefix).sum()
        composition = {column[4:]: float(value / amounts.sum()) for column, value in amounts.items()}
        for key in keys:
            scenario = key + '-' + use
            full = curves[key]
            own = daily_metrics(full, DAYS[use])
            frame = full.iloc[day*24:(day+1)*24].reset_index(drop=True)
            old_day = old_summary['selections'][use]['day_index']
            for version, index in (('old', old_day), ('new', day)):
                date = rows_by_id[scenario]['source_date'] if version == 'old' else source_date
                diagnostics.append({'scenario': scenario, 'scenario_name': rows_by_id[scenario]['scenario_name'], 'version': version,
                    'source_date': date, **own.iloc[index].to_dict(), **audit_day(full, own, index)})
            old, new = own.iloc[old_day], own.iloc[day]
            comparison.append({'scenario': scenario, 'scenario_name': rows_by_id[scenario]['scenario_name'],
                'old_date': rows_by_id[scenario]['source_date'], 'new_date': source_date,
                'old_energy_mwh': old.facility_mwh, 'new_energy_mwh': new.facility_mwh,
                'old_deviation_pct': old.energy_deviation_pct, 'new_deviation_pct': new.energy_deviation_pct,
                'old_rmse_mw': old.rmse_mw, 'new_rmse_mw': new.rmse_mw,
                'old_peak_valley_ratio': old.peak_valley_ratio, 'new_peak_valley_ratio': new.peak_valley_ratio,
                'selection_reference': reference, 'reference_old_deviation_pct': float(metrics.energy_deviation_pct.iloc[old_day]),
                'reference_new_deviation_pct': float(metrics.energy_deviation_pct.iloc[day]),
                'candidate_count': selection['candidate_count'], 'applied_window_pct': selection['applied_window_pct']})
            layers = split_layers(frame, resource, configs[key], key, use, ratios, mit, shape)
            fields = {'scenario': scenario, 'scenario_name': rows_by_id[scenario]['scenario_name'], 'source_date': source_date, 'weekday': weekday}
            hourly = pd.DataFrame({**fields, 'hour': range(24), 'model_time_utc': frame.time_utc, 'reference_it_mw': 10.,
                'installed_reference_it_mw': frame.installed_reference_it_mw, 'it_mw': frame.it_mw,
                'facility_mw': frame.facility_mw, 'pue': frame.pue, 'source_quality_flag': frame.source_quality_flag})
            hourly.to_csv(output / (scenario + '-hourly.csv'), index=False, encoding='utf-8-sig')
            parts = [pd.DataFrame({**fields, 'hour': range(24), 'task_type': label,
                'it_mw': np.zeros(24) if label == '冷却等非 IT' else values, 'facility_mw': values,
                'component_scope': 'non_IT' if label == '冷却等非 IT' else 'IT'}) for label, values in layers.items()]
            pd.concat(parts).to_csv(output / (scenario + '-by-task.csv'), index=False, encoding='utf-8-sig')
            np.testing.assert_allclose(sum(layers.values()), frame.facility_mw, atol=1e-8)
            annual = float(full.facility_mw.sum())
            records.append({**fields, 'complete_days': DAYS[use], 'peak_mw': float(new.peak_mw), 'minimum_mw': float(new.minimum_mw),
                'peak_valley_ratio': float(new.peak_valley_ratio), 'daily_facility_mwh': float(new.facility_mwh),
                'daily_it_mwh': float(frame.it_mw.sum()), 'annual_facility_mwh': annual,
                'daily_to_annual_mean_ratio': float(new.facility_mwh * 365 / annual), 'rmse_mw': float(new.rmse_mw),
                'energy_deviation_pct': float(new.energy_deviation_pct), 'composition': composition,
                'composition_basis': '核心小时' if prefix == 'cpu:' else 'GPU小时',
                'task_share_of_daily_it': float((frame.it_mw - frame.idle_it_mw).sum() / frame.it_mw.sum()),
                'model_weather_date': selections[use]['model_weather_date'], 'unresolved_selected_hours': int(frame.source_unresolved.sum())})
            checks[scenario + '_frozen_v4_hourly_values_and_task_balance'] = True
        print(json.dumps({'use': use, 'new_date': source_date, 'window_pct': selection['applied_window_pct'], 'candidates': selection['candidate_count']}, ensure_ascii=False), flush=True)
    summary = {'data_version': 'v4-model-new-day-rule', 'power_model_baseline_commit': 'a67ebd030d1255d4e533c419b229229c000b995a',
        'selection_parameters': {'primary_window': PRIMARY_WINDOW, 'fallback_window': FALLBACK_WINDOW, 'minimum_candidates': MINIMUM_CANDIDATES},
        'selection_rule': RULE, 'selections': selections, 'scenarios': records}
    save_json(output / 'summary.json', summary)
    pd.DataFrame([{**row, 'composition': json.dumps(row['composition'], ensure_ascii=False)} for row in records]).to_csv(output / 'summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(comparison).to_csv(output / 'day-selection-comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(diagnostics).to_csv(output / 'selected-day-diagnostics.csv', index=False, encoding='utf-8-sig')
    for name in ('annual-summary.csv', 'brief-facts.json', 'idle-endpoints.csv'):
        shutil.copyfile(old_root / name, output / name)
    report(output, summary, comparison, diagnostics)
    checks.update(plot_all(output, rule_caption=RULE_CAPTION))
    assert all(checks.values())
    save_json(output / 'validation.json', {'status': 'passed', 'checks': checks})
    save_json(output / 'manifest.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'v4_manifest_sha256': sha256(old_root / 'manifest.json'), 'frozen_annual_hourly_sha256': full_hashes,
        'resource_inputs': [inputs[use + '_local_classified_resource_hours'] for use in ('HPC', 'CLD', 'SAI', 'TRN', 'INF')],
        'model_and_data_unchanged': True, 'selection_parameters': summary['selection_parameters'],
        'code_sha256': {name: sha256(project / 'analysis' / name) for name in ('reselect_scenario_typical_days.py', 'plot_scenario_typical_days_v4.py')}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    build(parser.parse_args().project)
