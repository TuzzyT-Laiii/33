import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa

from build_native_load_curves_v3 import source_metrics, reduce_placements
from build_trn_seren import unit_power
from load_curve_scenarios_v3 import catalogue
from native_load_inputs_v3 import metrics, POWER_COLUMNS
from plot_native_combined import native, selected_days
from plot_scenario_typical_days import setup
from reselect_scenario_typical_days import daily_metrics, select_day, audit_day
from update_chapter4_trn_seren import complete_days, LABELS


RUN = 'typical-load-curves-20261101-v4-seren'
PREVIOUS = 'typical-load-curves-20261101-v3-native-combined'
COMMIT = 'bd6c8cdcb99a08b994136372881cee4a98d55d46'
TRN = ['B-TRN', 'C-TRN', 'D-TRN']


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return pd.read_csv(path, float_precision='round_trip')


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def replace_training(old, current):
    result = pd.concat([old.loc[~old['use'].eq('TRN')], current], ignore_index=True)
    return result.reindex(columns=old.columns)


def build(project, inputs):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    previous = project / 'results' / PREVIOUS
    output = project / 'results' / RUN
    upstream = project / 'results/trn-seren-20261101'
    v7 = project / 'results/scenario-typical-day-by-task-20261101-v7'
    for name in ('hourly', 'source-hourly', 'selected-days'):
        (output / name).mkdir(parents=True, exist_ok=True)
    baseline_hashes = {p.relative_to(previous).as_posix(): digest(p) for p in previous.rglob('*') if p.is_file()}
    preserved = {}
    # 非训练小时文件与独立图件保留原始字节。
    for folder in ('hourly', 'source-hourly', 'selected-days'):
        for path in (previous / folder).iterdir():
            if not path.is_file() or '-TRN' in path.name or path.name.startswith(('TRN-', 'overview', 'summary', 'brief-', 'selected-day', 'resource-input')):
                continue
            target = output / folder / path.name
            shutil.copyfile(path, target)
            preserved[target.relative_to(output).as_posix()] = digest(target)
    audit = json.loads((upstream / 'calculation-audit.json').read_text('utf-8'))
    old_cases = read(previous / 'window-summary-all-locations.csv')
    old_native = read(previous / 'native-summary.csv')
    old_aggregate = read(previous / 'placement-aggregate-all-locations.csv')
    upstream_cases = read(upstream / 'window-summary-all-locations.csv').query('case == "main"')
    fields_index = old_cases.loc[old_cases.use.eq('TRN')].set_index(['scenario_id', 'location_id', 'weather_year']).sort_index()
    assert fields_index.index.is_unique
    configs = catalogue(project)
    shape = json.loads((project / 'results/trn-power-model-20261101/cpu-shape.json').read_text('utf-8'))
    frames, sources, case_rows, hashes, profile_day, profile_week, series = {}, {}, [], {}, [], [], []
    for number, row in enumerate(upstream_cases.to_dict('records')):
        relative = f'hourly/{row["location_id"]}/main/{row["scenario_id"]}-{row["weather_year"]}.parquet'
        path = inputs / relative
        hashes[relative] = digest(path)
        assert hashes[relative] == audit['hourly_sha256'][relative], relative
        frame = pd.read_parquet(path)
        assert len(frame) == 2057 and not frame.power_imputed.any()
        clock = pd.DatetimeIndex(pd.to_datetime(frame.source_clock, utc=True))
        assert clock.is_unique and clock.is_monotonic_increasing
        assert int((clock.to_series().diff() > pd.Timedelta(hours=1)).sum()) == 53
        fields = fields_index.loc[(row['scenario_id'], row['location_id'], row['weather_year'])].to_dict()
        fields = {k: v for k, v in fields.items() if not k.startswith(('mean_', 'peak_', 'minimum_', 'window_', 'it_', 'facility_'))}
        fields.update(scenario_id=row['scenario_id'], location_id=row['location_id'], weather_year=row['weather_year'],
            source_days=2057/24, native_days=2057/24, native_hours=2057, has_real_dates=True, truncated=False,
            dropped_source_tail_hours=0, placement='dated', time_basis=str(frame.time_basis.iloc[0]),
            weather_start=str(frame.time_utc.iloc[0]), weather_end_in_source_order=str(frame.time_utc.iloc[-1]))
        result = metrics(frame, fields)
        for name in ('mean_it_mw', 'peak_it_mw', 'minimum_it_mw', 'window_it_mwh', 'mean_facility_mw',
                     'peak_facility_mw', 'minimum_facility_mw', 'window_facility_mwh',
                     'it_diurnal_peak_valley', 'facility_diurnal_peak_valley', 'it_weekly_peak_valley', 'facility_weekly_peak_valley'):
            np.testing.assert_allclose(result[name], row[name], rtol=1e-12, atol=1e-10)
        case_rows.append(result)
        if row['location_id'] == 'reference-grid':
            target = output / 'hourly' / f'{row["scenario_id"]}-{row["weather_year"]}-dated.parquet'
            shutil.copyfile(path, target)
            frames[row['scenario_id']] = frame
            key = str(frame.server_type.iloc[0])
            unit = unit_power(configs[key], key, frame, shape)
            for component in ('base', 'cpu', 'gpu'):
                np.testing.assert_allclose(unit[component + '_w'] * frame.unit_count / 1e6, frame[component + '_it_mw'], rtol=1e-12)
            source = frame[[c for c in frame if c.startswith(('source_', 'hour_of_', 'allocated_', 'gpu_hours_', 'host_cpu_')) or c in
                            ('u_cpu_proxy', 'it_mw', 'base_it_mw', 'cpu_it_mw', 'gpu_it_mw', 'power_imputed', 'power_ratio_above_tdp')]].copy()
            for field in ('scenario_id', 'scenario', 'scenario_zh', 'server_type', 'use', 'reference_it_mw', 'unit_count', 'installed_reference_it_mw'):
                source[field] = frame[field]
            source['has_real_dates'] = True
            source['time_basis'] = frame.time_basis
            source.to_parquet(output / 'source-hourly' / (row['scenario_id'] + '.parquet'), index=False, compression='zstd')
            template = old_native.loc[old_native.scenario_id.eq(row['scenario_id'])].iloc[0].to_dict()
            source_fields = {k: template[k] for k in ('scenario_id', 'scenario', 'scenario_zh', 'server_type', 'use',
                'reference_it_mw', 'unit_count', 'installed_reference_it_mw', 'has_real_dates')}
            source_fields['time_basis'] = str(frame.time_basis.iloc[0])
            sources[row['scenario_id']] = source_metrics(frame, source_fields)
            for grouping, target_list in [('hour_of_source_day', profile_day), ('hour_of_source_week', profile_week)]:
                target_list.append(frame.groupby(grouping)[[x + '_mw' for x in POWER_COLUMNS]].mean().reset_index().assign(
                    scenario_id=row['scenario_id'], reference_it_mw=row['reference_it_mw'], use='TRN', placement='dated'))
            if row['reference_it_mw'] == 10:
                series.append(frame[['scenario_id', 'source_hour_index', 'facility_mw']].rename(columns={'facility_mw': 'facility_mw_median'}).assign(
                    facility_mw_min=frame.facility_mw, facility_mw_max=frame.facility_mw, available_placements=1))
        if number % 50 == 0:
            print(json.dumps({'verified_Seren_hourly': number + 1, 'total': len(upstream_cases)}), flush=True)
    training_cases = pd.DataFrame(case_rows)
    cases = replace_training(old_cases, training_cases)
    aggregate_training = reduce_placements(training_cases, sources)
    aggregate = replace_training(old_aggregate, aggregate_training)
    summary = replace_training(old_native, aggregate_training.loc[aggregate_training.location_id.eq('reference-grid')])
    assert len(cases) == 5500 and len(summary) == 44 and len(hashes) == 300
    for name, table in [('window-summary-all-locations', cases), ('placement-aggregate-all-locations', aggregate),
                        ('native-summary', summary), ('placement-summary', cases.loc[cases.location_id.eq('reference-grid')])]:
        table.to_csv(output / (name + '.csv'), index=False, encoding='utf-8-sig')
    for name, additions in [('typical-day-placements', profile_day), ('typical-week-placements', profile_week)]:
        before = read(previous / (name + '.csv'))
        replace_training(before, pd.concat(additions)).to_csv(output / (name + '.csv'), index=False, encoding='utf-8-sig')
    before_series = pd.read_parquet(previous / 'native-timeseries-10MW.parquet')
    pd.concat([before_series.loc[~before_series.scenario_id.str.contains('-TRN-')], *series], ignore_index=True).to_parquet(
        output / 'native-timeseries-10MW.parquet', index=False, compression='zstd')
    times = json.loads((previous / 'source-time-bases.json').read_text('utf-8'))
    times['TRN'] = {'source_days': 2057/24, 'source_hours': 2057, 'has_real_dates': True,
        'source_start': '2023-05-16T10:00:00Z', 'source_end_inclusive': '2023-08-17T23:00:00Z',
        'time_basis': 'Seren原+08:00已换算UTC；按月日小时对应气象；189小时缺口保留',
        'weekly_basis': '真实UTC星期；仅保留合格小时', 'input_sha256': digest(upstream / 'source-hourly.csv')}
    save(output / 'source-time-bases.json', times)
    typical = output / 'selected-days'
    selected = json.loads((previous / 'selected-days/summary.json').read_text('utf-8'))
    old_selected = {r['scenario']: r.copy() for r in selected['scenarios']}
    complete, dates, available = complete_days(frames['D-TRN-10MW'])
    day, scores, selection = select_day(daily_metrics(complete, len(dates)))
    scores.insert(0, 'source_date', dates.strftime('%Y-%m-%d'))
    scores.to_csv(typical / 'TRN-day-selection.csv', index=False, encoding='utf-8-sig')
    available.to_csv(typical / 'TRN-day-availability.csv', index=False, encoding='utf-8-sig')
    date = dates[day]
    assert len(dates) == 55 and date.strftime('%Y-%m-%d') == '2023-05-31'
    selection.update(day_index=day, source_date=date.strftime('%Y-%m-%d'), placement='dated', qualified_hours=2057,
        placement_policy='Seren正文真实UTC日期；仅完整日参与选日；三训练规格共用GPU主导型日期')
    selected['selections']['TRN'] = selection
    diagnostics, comparisons = [], []
    summary_index = summary.set_index('scenario_id')
    old_index = old_native.set_index('scenario_id')
    for row in selected['scenarios']:
        scenario = row['scenario']
        if scenario not in TRN:
            continue
        full = frames[scenario + '-10MW']
        clock = pd.DatetimeIndex(pd.to_datetime(full.source_clock, utc=True))
        current = full.loc[clock.floor('d') == date].reset_index(drop=True)
        fields = dict(scenario=scenario, scenario_name=row['scenario_name'], source_date=date.strftime('%Y-%m-%d'),
            weekday='星期' + '一二三四五六日'[date.dayofweek])
        current.assign(**fields, hour=np.arange(24)).to_csv(typical / (scenario + '-hourly.csv'), index=False, encoding='utf-8-sig')
        layers = {'基础与空载 IT': current.idle_it_mw.to_numpy()}
        for kind, label in LABELS.items():
            layers[label] = (current.it_mw - current.idle_it_mw) * current['gpu_hours_' + kind] / current.allocated_gpu_hours
        layers['冷却等非 IT'] = (current.facility_mw - current.it_mw).to_numpy()
        np.testing.assert_allclose(sum(layers.values()), current.facility_mw, atol=1e-10)
        pd.concat([pd.DataFrame({**fields, 'hour': range(24), 'task_type': label, 'facility_mw': values,
            'it_mw': np.zeros(24) if label == '冷却等非 IT' else values}) for label, values in layers.items()]).to_csv(
                typical / (scenario + '-by-task.csv'), index=False, encoding='utf-8-sig')
        complete_key, key_dates, _ = complete_days(full)
        assert key_dates.equals(dates)
        metric = daily_metrics(complete_key, len(dates))
        selected_metrics = metric.iloc[day]
        row.update(**fields, placement='dated', model_weather_date=str(current.time_utc.iloc[0]),
            daily_facility_mwh=float(current.facility_mw.sum()), daily_it_mwh=float(current.it_mw.sum()),
            peak_mw=float(current.facility_mw.max()), minimum_mw=float(current.facility_mw.min()),
            peak_valley_ratio=float(current.facility_mw.max() / current.facility_mw.min()),
            rmse_mw=float(selected_metrics.rmse_mw), energy_deviation_pct=float(selected_metrics.energy_deviation_pct),
            native_window_facility_mean_mw=float(full.facility_mw.mean()), native_days=2057/24, complete_days=55)
        diagnostics.append({**fields, **audit_day(complete_key, metric, day)})
        comparison = {'scenario': scenario, 'scenario_name': row['scenario_name'],
            'old_source_date': old_selected[scenario]['source_date'], 'new_source_date': row['source_date']}
        for name in ('mean_it_mw', 'peak_it_mw', 'placement_mean_facility_mw_median', 'placement_peak_facility_mw_median',
                     'it_diurnal_peak_valley', 'placement_facility_diurnal_peak_valley_median'):
            comparison['old_' + name] = old_index.loc[scenario + '-10MW', name]
            comparison['new_' + name] = summary_index.loc[scenario + '-10MW', name]
        comparison.update(old_selected_day_ratio=old_selected[scenario]['peak_valley_ratio'], new_selected_day_ratio=row['peak_valley_ratio'],
            old_selected_day_peak_mw=old_selected[scenario]['peak_mw'], new_selected_day_peak_mw=row['peak_mw'])
        comparisons.append(comparison)
    save(typical / 'summary.json', selected)
    pd.DataFrame(selected['scenarios']).to_csv(typical / 'summary.csv', index=False, encoding='utf-8-sig')
    old_diagnostics = read(previous / 'selected-days/selected-day-diagnostics.csv')
    pd.concat([old_diagnostics.loc[~old_diagnostics.scenario.isin(TRN)], pd.DataFrame(diagnostics)], ignore_index=True).to_csv(
        typical / 'selected-day-diagnostics.csv', index=False, encoding='utf-8-sig')
    shutil.copyfile(v7 / 'brief-facts.json', typical / 'brief-facts.json')
    shutil.copyfile(v7 / 'seren-job-statistics.json', typical / 'seren-job-statistics.json')
    pd.DataFrame(comparisons).to_csv(output / 'training-10MW-comparison.csv', index=False, encoding='utf-8-sig')
    crosscheck = []
    for row in selected['scenarios']:
        scenario = row['scenario']
        hourly = read(typical / (scenario + '-hourly.csv'))
        full = pd.read_parquet(output / 'hourly' / f'{scenario}-10MW-2025-{row["placement"]}.parquet')
        if scenario in TRN:
            selected_full = full.loc[pd.to_datetime(full.source_clock, utc=True).dt.strftime('%Y-%m-%d').eq(row['source_date'])]
        else:
            index = selected['selections'][scenario.split('-')[1]]['day_index']
            selected_full = full.iloc[index*24:(index+1)*24]
        delta = float(np.max(np.abs(hourly.facility_mw.to_numpy() - selected_full.facility_mw.to_numpy())))
        assert delta < 1e-12
        crosscheck.append({'scenario': scenario, 'source_date': row['source_date'], 'selected_hourly_maximum_difference_mw': delta,
            'window_it_mean_absolute_difference_mw': abs(float(full.it_mw.mean()) - summary_index.loc[scenario + '-10MW', 'mean_it_mw'])})
    pd.DataFrame(crosscheck).to_csv(output / 'chapter4-comparison.csv', index=False, encoding='utf-8-sig')
    order = ['A-HPC', 'A-CLD', 'B-SAI', 'C-SAI', 'D-SAI', *TRN, 'B-INF', 'C-INF', 'D-INF']
    ten = summary.loc[summary.reference_it_mw.eq(10)].set_index('scenario_id').loc[[x + '-10MW' for x in order]]
    rows = [['情境（参考IT 10MW）', '原生\n长度', 'IT平均\nMW', 'IT峰值\nMW', '设施平均\nMW', '设施峰值\nMW', '设施日内\n峰谷比', '设施平均范围\nMW']]
    for row in ten.to_dict('records'):
        duration = '2057小时\n约86天\n有缺口' if row['use'] == 'TRN' else str(int(row['native_days'])) + '天'
        interval = '—' if row['has_real_dates'] else f'{row["placement_mean_facility_mw_min"]:.3f}–{row["placement_mean_facility_mw_max"]:.3f}'
        rows.append([row['scenario_zh'], duration, *[f'{row[x]:.3f}' for x in ('mean_it_mw', 'peak_it_mw',
            'placement_mean_facility_mw_median', 'placement_peak_facility_mw_median', 'placement_facility_diurnal_peak_valley_median')], interval])
    save(output / 'brief-summary.json', {'rows': rows, 'model': '#38 Seren main＋#35推理＋#36 CPU/基础空闲；cooling-20261028-v2'})
    summary.to_csv(output / 'all-44-scenarios.csv', index=False, encoding='utf-8-sig')
    setup()
    series_frame = pd.read_parquet(output / 'native-timeseries-10MW.parquet')
    for weekly, averaged, profile in [(False, False, None), (False, True, 'typical-day-placements.csv'), (True, False, 'typical-week-placements.csv')]:
        native(output, summary.loc[summary.reference_it_mw.eq(10)].set_index('scenario_id'), series_frame,
            read(output / profile) if profile else None, weekly=weekly, averaged=averaged)
    selected_days(output, only_scenarios=TRN, training_seren=True)
    assert all(digest(previous / p) == value for p, value in baseline_hashes.items())
    assert all(digest(output / p) == value == digest(previous / p) for p, value in preserved.items())
    for file in ('native-summary.csv', 'placement-summary.csv', 'placement-aggregate-all-locations.csv', 'window-summary-all-locations.csv'):
        old, new = read(previous / file), read(output / file)
        pd.testing.assert_frame_equal(old.loc[~old.use.eq('TRN')].reset_index(drop=True), new.loc[~new.use.eq('TRN')].reset_index(drop=True), check_dtype=False)
    save(output / 'manifest.json', {'utc': datetime.now(timezone.utc).isoformat(), 'PR38_commit': COMMIT,
        'PR31_statistics_commit': 'c16ac75623516a2ff7ac633f764def34704c9265', 'previous_result': PREVIOUS,
        'upstream_main_hourly_sha256': hashes, 'preserved_files_sha256': preserved, 'previous_all_files_sha256': baseline_hashes,
        'cases': len(cases), 'hour_rows': int(cases.hours.sum()), 'reference_cases': 220, 'native_scenarios': 44,
        'training_hours': 2057, 'training_segments': 54, 'training_excluded_interior_hours': 189,
        'complete_UTC_days': 55, 'selected_UTC_date': '2023-05-31', 'nontraining_summary_values_unchanged': True,
        'chapter4_chapter5_same_hourly_values': True, 'previous_directory_unchanged': True,
        'comparison_scope': '旧Kalos与新Seren来源、窗口、归一化与功率同时变化，仅描述性比较',
        'code_sha256': {name: digest(project / 'analysis' / name) for name in ('update_brief_seren.py', 'plot_native_combined.py')}})
    report(output, rows, comparisons, selection)
    print(json.dumps({'output': str(output), 'selection': selection, 'comparisons': comparisons, 'preserved_files': len(preserved)}, ensure_ascii=False), flush=True)


def report(output, rows, comparisons, selection):
    lines = ['# Seren训练与组合模型原生窗口简表', '',
        '训练采用PR #38 bd6c8cd正文main：2023-05-16 10:00至08-17 23:00 UTC范围内2057个合格小时，合计85.7083天小时量。范围内189小时缺口保留，54个连续片段；不填充、不循环、不压缩时间、不年化。', '',
        'Seren同期GPU监测反推已分配GPU功率/TDP：源GPU采样时间加权均值0.7623，P10/P50/P90=0.6139/0.7683/0.8963。2288卡在线库存、报送均功率代表缺测GPU、源未分配GPU空闲60W为假设；无逐作业设备映射。A100 400W按TDP比例迁移至H100 PCIe 350W或SXM 700W，吞吐及进度未标定。8个超过TDP的小时保留原值及标记。4192小时观测＋UTC日形状补缺方案仅为上游对照。', '',
        '其余八个组合、32个规模情境原生源、208份参考气象时序与8套典型日图/表按上一版原字节复制。推理#35、CPU/基础空闲#36、科研AI的MIT六月标定与冷却cooling-20261028-v2保持。Lisa309天、MIT237天按源月日小时对应气象；Lisa时区对应UTC为假设，MIT保留Slurm epoch。Alibaba5天与ASI185天无真实日期，在12个月初放置、年末截断，范围同时含气象及源前缀变化；Alibaba不报告周形状，ASI为相对七日。', '',
        '## 10MW原生窗口', '',
        '参考网格2025气象。IT取全部原生合格小时；设施为各放置指标中位数，有日期来源一次对应。日内峰谷比为全窗口按源小时算术均值的最大/最小；周内同理按源星期小时分组。窗口电量为实际小时求和，均不年化。', '',
        '| ' + ' | '.join(x.replace('\n', '') for x in rows[0]) + ' |', '| ' + ' | '.join(['---']*8) + ' |']
    lines += ['| ' + ' | '.join(x.replace('\n', '；') for x in row) + ' |' for row in rows[1:]]
    lines += ['', '![原生窗口](native-overview.png)', '', '训练面板以真实UTC经过天数显示，缺口插入NaN仅供绘图断线；数值表与时序不插入任何补缺值。其他面板源经过天数保持。', '',
        '## 第四章典型日', '', f'仅{selection["complete_days"]}个完整UTC日参与训练选日，中位电量±2%内{selection["candidate_count"]}个候选，按与全部完整日平均24小时设施曲线RMSE最小选中2023-05-31；GPU主导型选日、三种规格共日。部分日不参与选日和RMSE参照，仍保留在第五章2057小时窗口统计中。', '',
        '训练任务分层共用同小时集群反推功率比例，CPU/GPU动态IT增量按各类别GPU分配小时占比分摊；MLLM独立单列。分任务功率为模型归因，未使用文献或受控实验的逐类型功率比例。', '', '![典型日](selected-days/overview.png)', '',
        '## 与PR #37上一版训练数字对照', '', '旧版为Kalos90天，GPU功率与日形状为旧情境假设；本表为历史对照。来源、窗口、归一化、任务构成及功率均变化，差值不能归因于单独的功率模型。', '',
        '| 训练情境 | IT平均MW | IT峰值MW | 设施平均MW | 设施峰值MW | 设施日内比 | 典型日 |', '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in comparisons:
        values = [f'{row["old_"+name]:.3f}→{row["new_"+name]:.3f}' for name in ('mean_it_mw','peak_it_mw',
            'placement_mean_facility_mw_median','placement_peak_facility_mw_median','placement_facility_diurnal_peak_valley_median')]
        lines.append('| ' + ' | '.join([row['scenario_name'], *values, row['old_source_date']+'→'+row['new_source_date']]) + ' |')
    lines += ['', '## 文件、验证与复现', '',
        'native-summary.csv为44情境，placement-summary.csv为参考网格220次放置；window-summary-all-locations.csv为25组地点年份5500次放置，训练替换300行。source-hourly保留32条未变源和12条Seren源；hourly保留208条未变时序及12条与#38 SHA-256完全相同的Seren正文时序。城市训练小时读取上游缓存，全部300份匹配#38指纹；汇总与上游指标逐项核对。', '',
        'training-10MW-comparison.csv保存训练新旧平均/峰值、日内比和典型日日期；chapter4-comparison.csv核对11情境第四章逐小时值与第五章原生时序一致。selected-days保存完整日可用性、选日评分、诊断、24小时与任务分层。第二章Seren任务统计来自#31 c16ac75重新扫描原始作业所得seren-job-statistics.json，完整作业分母与正文合格小时构成分别记录。', '',
        'manifest.json保存上游300份小时指纹、旧目录全部文件指纹及非训练保留文件指纹。脚本校验非训练汇总全部值不变、上游GPU/CPU/基础公式、任务分层加总及两章小时取值。brief-audit.json记录Word/PDF页数、表格边界与逐页检查。', '',
        '已核验的#38小时文件缓存置于项目.codex-temp/seren-brief-inputs/hourly/<地点>/main/。运行python analysis/update_brief_seren.py后，在docs/job-flexibility-brief运行node build.js 数据中心任务时间柔性简表.docx README.md，再运行analysis/export_brief_pdf.ps1及analysis/audit_brief_seren.py。旧结果目录保持原字节。', '',
        '## 局限与审阅', '',
        'Seren无逐作业设备映射，反推依赖在线库存与缺测代表性；A100→H100按TDP迁移，未标定吞吐。MIT单月标定迁移H100；Alibaba仅5天、ASI/Alibaba日期未知。任务标签、历史排队与功率变化均不赋予延后或暂停权限。', '',
        '依赖顺序：#27；#30→#32、#33；#34、#35、#36→#38；#31；#37。本PR保持待审，不合并。', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--inputs', type=Path)
    args = parser.parse_args()
    build(args.project, args.inputs or args.project / '.codex-temp/seren-brief-inputs')
