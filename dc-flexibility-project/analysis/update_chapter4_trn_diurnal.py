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
from build_trn_diurnal import metrics, recalculate, NATIVE
from compute_load.model import sha256
from load_curve_scenarios_v3 import catalogue
from plot_scenario_typical_days_v4 import plot_all
from reselect_scenario_typical_days import daily_metrics, select_day, audit_day, RULE, RULE_CAPTION


RUN = 'scenario-typical-day-by-task-20261101-v6'
PREVIOUS = 'scenario-typical-day-by-task-20261101-v5'
UPSTREAM = 'trn-power-model-20261101/diurnal-acme'
COMMIT = '1594fb77148f3c9307d00230c25e864f6a286625'
TRN = ['B-TRN', 'C-TRN', 'D-TRN']


def write_report(output, summary, comparison):
    lines = ['# 分任务典型日：Acme 训练日内形状', '', '![各情境总览](overview.png)', '',
        '参考IT 10MW；科研计算、云通用计算、科研AI、推理服务的八套单图及逐小时数据按v5原字节保留。本次只更新三种大模型训练/开发规格及总览。', '',
        RULE, '',
        '日电量与RMSE均由设施功率计算，GPU需求以GPU主导型筛选，B/C/D共用源日期及作业构成。候选、评分、窗口见TRN-day-selection.csv。', '',
        '## 训练来源与映射', '',
        f'- PR #36 `{COMMIT}`，功率实现`d7f6bca`，依赖#34原生窗口`c836bba`。采用2023-05-18至08-15的90个完整源UTC日；天气为参考网格2025同月日小时。只读下载其汇总小时文件，逐份匹配上游SHA-256并按#36公式独立复算。',
        '- 中央方案：Kalos固定271卡（功率报送覆盖阈值99%）的各UTC小时设备均值等权合成形状s(h)，r(h)=0.80×s(h)/GPU分配小时加权均值[s(h)]。GPU/TDP为0.729818–0.839815，加权中央水平严格0.80。CPU三份SPEC平滑、服务器数及基础空载保持#36；按新IT功率重新计算冷却与PUE。',
        '- 分配功率反推为[集群估计总功率−(全群设备数−分配设备数)×空闲功率]/分配设备数。估计总功率要求报送设备代表全群；原生窗口只重叠16.496采样小时、报送覆盖11.45%–35.84%，合格标定小时为零。因此反推只作诊断，采用固定集合的日形状迁移，未直接套用反推比例。',
        '- 形状含混合业务、分配变化与运行状态，无作业映射，不能识别预训练专属周期或检查点原因；没有星期调制。A100集群形状迁移至H100属于工程假设，可能重复计入部分分配变化。',
        '- 敏感性：形状幅度0/0.5/1/1.5，固定集合覆盖95%/99%/99.9%；每种形状均保持GPU小时加权中央0.80。中央水平0.65–0.95和基础空载端点沿用v4的独立敏感性，不与本次形状方案叠加，不合成为置信区间。', '',
        '## 典型日新旧对照', '',
        '旧列为PR31 v5所选典型日；新列为本次#36原生窗口重选日。设施功率包含冷却等非IT，表内比值为所选日24个小时最高/最低。上游的日内比是90天按源小时分组的均值比，两者不同。', '',
        '| 情境 | 源日期，旧→新 | 设施峰值MW | 设施谷值MW | 最高/最低比 | 日电量MWh |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in comparison:
        lines.append(f"| {row['scenario_name']} | {row['old_source_date']} → {row['new_source_date']} | {row['old_peak_mw']:.3f} → {row['new_peak_mw']:.3f} | {row['old_minimum_mw']:.3f} → {row['new_minimum_mw']:.3f} | {row['old_peak_valley_ratio']:.3f} → {row['new_peak_valley_ratio']:.3f} | {row['old_daily_facility_mwh']:.2f} → {row['new_daily_facility_mwh']:.2f} |")
    selected = summary['selections']['TRN']
    lines += ['', f"训练新日为{selected['source_date']}（{selected['weekday']}），±{selected['applied_window_pct']:g}%候选{selected['candidate_count']}天。具体偏差与RMSE见评分表。", '',
        '原生窗口下恒定0.80模型按同一规则也选中2023-06-27；日期从v5的2023-06-13切换同时涉及原生输入和天气口径，不能把日期变化单独归因于日形状。恒定模型评分另存TRN-constant-day-selection.csv。', '',
        'comparison.csv另列IT峰谷，matched-day-comparison.csv保存新日同天气下恒定0.80与中央形状的24小时曲线；sensitivity-summary.csv/hourly.csv为新日及原生90天敏感性。敏感性不重新选日。', '',
        'period-summary.csv明确每行时间范围；TRN用2160小时原生窗口的日电量均值，其他八情境沿用v5年度均值，不将90天外推为年电量。全部情境的典型日电量与各自期间日均之比保持同数量级。', '',
        '任务层为超过基础空载的IT增量：按同小时资源份额分摊CPU和GPU增量；CPU分类代理严格采用#36的0.25×各类GPU占用。训练形状为所有已分配训练GPU共享的用途级情境参数，各类别功率均由模型推算。每小时IT层之和等于IT总功率，加非IT层等于设施功率。', '',
        '## 复现与文件', '',
        '```powershell', "$env:OPENBLAS_NUM_THREADS='1'", "$env:OMP_NUM_THREADS='1'",
        'python dc-flexibility-project/analysis/update_chapter4_trn_diurnal.py', '```', '',
        '依赖本机已核验TRN分类资源小时缓存和已提交的#34原生小时文件。服务器共享卷results/trn-power-model-20261101/diurnal-acme/hourly/reference-grid下七个方案各三份*-TRN-10MW-2025-dated.parquet，只读复制到data/processed/' + RUN + '/upstream/<方案名>/，并由脚本逐份核对native-calculation-audit.json清单。不扫描逐作业原始数据，不访问MIT进程或STOP。所有输入、代码及保留文件有SHA-256记录。验证脚本verify_chapter4_trn_diurnal.py核对选日、上游功率、分层、其他八情境及文档PDF。', '',
        '其他情境来源、请求映射、MIT含推断类别及Alibaba尖峰说明见[保留的v5报告](../' + PREVIOUS + '/README.md)与[v4模型报告](../scenario-typical-day-by-task-20261101-v4/README.md)。训练完整证据见[PR36报告](../' + UPSTREAM + '/README.md)。', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def build(project):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    previous, output = project / 'results' / PREVIOUS, project / 'results' / RUN
    upstream = project / 'results' / UPSTREAM
    output.mkdir(parents=True, exist_ok=True)
    local = project / 'data/processed' / RUN
    local.mkdir(parents=True, exist_ok=True)
    inputs = []

    def track(path):
        inputs.append({'file': str(path.resolve()), 'sha256': sha256(path)})
        return path

    def read_json(path):
        return json.loads(track(path).read_text(encoding='utf-8'))

    summary = read_json(previous / 'summary.json')
    old_rows = {row['scenario']: row.copy() for row in summary['scenarios']}
    v4_manifest = read_json(project / 'results/scenario-typical-day-by-task-20261101-v4/manifest.json')
    resource_input = next(item for item in v4_manifest['input_files'] if item['role'] == 'TRN_local_classified_resource_hours')
    assert sha256(Path(resource_input['file'])) == resource_input['sha256']
    resource = pd.read_parquet(track(Path(resource_input['file'])))
    # 原生TRN的CPU代理为GPU占用的四分之一，分类层沿用相同定义。
    for column in resource.filter(regex='^gpu:').columns:
        resource['cpu:' + column[4:]] = .25 * resource[column]
    configs = catalogue(project)
    ratios = read_json(project / 'results/typical-load-curves-20261029-v2/mit-power-ratio-summary.json')['transfer_distribution']
    mit = read_json(project / 'results/typical-load-curves-20261101-v3/mit-calibration.json')
    cpu_shape = read_json(project / 'results/trn-power-model-20261101/cpu-shape.json')
    audit = read_json(upstream / 'native-calculation-audit.json')
    profile = pd.read_csv(track(upstream / 'transfer-profile.csv'))
    reference = pd.read_csv(track(upstream / 'window-summary.csv')).query('reference_it_mw == 10').set_index(['server_type', 'case'])
    assert profile.hour_utc.tolist() == list(range(24))
    full_curves, variants, local_hashes, checks = {}, {}, {}, {}
    for key in ('B', 'C', 'D'):
        scenario = key + '-TRN'
        relative = f'results/{NATIVE}/hourly/{scenario}-10MW-2025-dated.parquet'
        path = track(project / relative)
        assert sha256(path) == audit['inputs_sha256'][relative]
        source = pd.read_parquet(path)
        assert len(source) == len(resource) == 2160
        for prefix, column in (('cpu:', 'u_cpu_proxy'), ('gpu:', 'allocated_gpu_fraction')):
            np.testing.assert_allclose(resource.filter(regex='^' + prefix).sum(axis=1), source[column], rtol=1e-8, atol=1e-7)
        position = source.hour_of_source_day.to_numpy(int)
        variants[key] = {}
        cases = ['constant_080'] + [name for name in profile if name.startswith('shape_')]
        for case in cases:
            ratio = np.full(2160, .8) if case == 'constant_080' else profile[case].to_numpy()[position]
            recalculated = recalculate(source, configs[key], ratios, mit, cpu_shape, ratio)
            upstream_file = track(local / 'upstream' / case / f'{scenario}-10MW-2025-dated.parquet')
            expected_key = f'hourly/reference-grid/{case}/{upstream_file.name}'
            assert sha256(upstream_file) == audit['output_hourly_sha256'][expected_key]
            current = pd.read_parquet(upstream_file)
            pd.testing.assert_frame_equal(current, recalculated, rtol=1e-12, atol=1e-12)
            expected = reference.loc[(key, case)]
            for name in ('it', 'facility'):
                np.testing.assert_allclose(current[name + '_mw'].mean(), expected['mean_' + name + '_mw'], rtol=1e-12)
            variants[key][case] = current
        current = variants[key]['shape_central']
        path = local / f'{scenario}-10MW-2025-dated.parquet'
        shutil.copyfile(local / 'upstream/shape_central' / path.name, path)
        expected_key = f'hourly/reference-grid/shape_central/{path.name}'
        assert sha256(path) == audit['output_hourly_sha256'][expected_key]
        local_hashes[str(path.resolve())] = sha256(path)
        full_curves[key] = current
        checks[scenario + '_upstream_hourly_sha256_and_seven_case_means'] = True
    day, scores, selection = select_day(daily_metrics(full_curves['D'], 90))
    clocks = pd.DatetimeIndex(pd.to_datetime(full_curves['D'].source_clock, utc=True))
    constant_day, constant_scores, constant_selection = select_day(daily_metrics(variants['D']['constant_080'], 90))
    constant_scores.insert(0, 'source_day', clocks[::24].strftime('%Y-%m-%d'))
    constant_scores.to_csv(output / 'TRN-constant-day-selection.csv', index=False, encoding='utf-8-sig')
    constant_selection.update(day_index=constant_day, source_date=clocks[constant_day*24].strftime('%Y-%m-%d'))
    summary['TRN_constant_080_selection'] = constant_selection
    scores.insert(0, 'source_day', clocks[::24].strftime('%Y-%m-%d'))
    scores.to_csv(output / 'TRN-day-selection.csv', index=False, encoding='utf-8-sig')
    start = day * 24
    source_date = clocks[start].strftime('%Y-%m-%d')
    weekday = '星期' + '一二三四五六日'[clocks[start].dayofweek]
    selection.update(day_index=day, source_date=source_date, weekday=weekday,
        reference_scenario='D-TRN', reference_scenario_name=old_rows['D-TRN']['scenario_name'],
        model_weather_date=str(full_curves['D'].time_utc.iloc[start].date()))
    summary['selections']['TRN'] = selection
    selected_resource = resource.iloc[start:start+24].reset_index(drop=True)
    amounts = selected_resource.filter(regex='^gpu:').sum()
    composition = {column[4:]: float(value/amounts.sum()) for column, value in amounts.items()}
    preserved = {}
    for row in summary['scenarios']:
        scenario = row['scenario']
        if scenario in TRN:
            continue
        for suffix in ('-hourly.csv', '-by-task.csv', '.png', '.svg'):
            name = scenario + suffix
            shutil.copyfile(previous / name, output / name)
            preserved[name] = sha256(output / name)
    for name in ('HPC-day-selection.csv', 'CLD-day-selection.csv', 'SAI-day-selection.csv', 'INF-day-selection.csv', 'idle-endpoints.csv'):
        shutil.copyfile(previous / name, output / name)
        preserved[name] = sha256(output / name)
    facts = read_json(previous / 'brief-facts.json')
    facts['power'].update(TRN_diurnal_GPU_TDP_range=[float(profile.shape_central.min()), float(profile.shape_central.max())],
        TRN_diurnal_source='Acme Kalos fixed 271 GPUs; source UTC hourly shape; allocated GPU-hour weighted 0.80',
        TRN_diurnal_commit=COMMIT)
    save_json(output / 'brief-facts.json', facts)
    comparisons, matched, diagnostics, sensitivity_rows, sensitivity_hours = [], [], [], [], []
    for row in summary['scenarios']:
        scenario = row['scenario']
        if scenario not in TRN:
            continue
        key = scenario[0]
        full = full_curves[key]
        frame = full.iloc[start:start+24].reset_index(drop=True)
        layers = split_layers(frame, selected_resource, configs[key], key, 'TRN', ratios, mit, cpu_shape)
        fields = {'scenario': scenario, 'scenario_name': row['scenario_name'], 'source_date': source_date, 'weekday': weekday}
        pd.DataFrame({**fields, 'hour': range(24), 'model_time_utc': frame.time_utc, 'reference_it_mw': 10.,
            'installed_reference_it_mw': frame.installed_reference_it_mw, 'allocated_gpu_power_ratio': frame.allocated_gpu_power_ratio,
            'it_mw': frame.it_mw, 'facility_mw': frame.facility_mw, 'pue': frame.pue,
            'source_quality_flag': frame.source_quality_flag}).to_csv(output / (scenario + '-hourly.csv'), index=False, encoding='utf-8-sig')
        pd.concat([pd.DataFrame({**fields, 'hour': range(24), 'task_type': label,
            'it_mw': np.zeros(24) if label == '冷却等非 IT' else values, 'facility_mw': values,
            'component_scope': 'non_IT' if label == '冷却等非 IT' else 'IT'}) for label, values in layers.items()]).to_csv(output / (scenario + '-by-task.csv'), index=False, encoding='utf-8-sig')
        energy = float(frame.facility_mw.sum())
        own_metrics = daily_metrics(full, 90)
        diagnosis = audit_day(full, own_metrics, day)
        diagnostics.append({**fields, **diagnosis})
        row.pop('annual_facility_mwh')
        row.pop('daily_to_annual_mean_ratio')
        row.update(**fields, peak_mw=float(frame.facility_mw.max()), minimum_mw=float(frame.facility_mw.min()),
            peak_valley_ratio=float(frame.facility_mw.max()/frame.facility_mw.min()),
            daily_it_mwh=float(frame.it_mw.sum()), daily_facility_mwh=energy,
            period_facility_mwh=float(full.facility_mw.sum()), period_hours=2160,
            daily_to_period_mean_ratio=energy/full.facility_mw.mean()/24,
            model_weather_date=selection['model_weather_date'], composition=composition,
            task_share_of_daily_it=float((frame.it_mw-frame.idle_it_mw).sum()/frame.it_mw.sum()),
            unresolved_selected_hours=int(frame.source_unresolved.sum()))
        old = old_rows[scenario]
        old_it = pd.read_csv(previous / (scenario + '-hourly.csv')).it_mw
        new_it = frame.it_mw
        comparisons.append({'scenario': scenario, 'scenario_name': row['scenario_name'],
            'old_source_date': old['source_date'], 'new_source_date': source_date,
            **{version + '_' + name: record[name] for version, record in [('old', old), ('new', row)]
               for name in ('peak_mw', 'minimum_mw', 'peak_valley_ratio', 'daily_facility_mwh')},
            **{version + '_it_' + name: float(value) for version, series in [('old', old_it), ('new', new_it)]
               for name, value in [('peak_mw', series.max()), ('minimum_mw', series.min()), ('peak_valley_ratio', series.max()/series.min())]}})
        constant = variants[key]['constant_080'].iloc[start:start+24]
        matched.append(pd.DataFrame({**fields, 'hour': range(24), 'constant_it_mw': constant.it_mw.to_numpy(),
            'shape_it_mw': frame.it_mw, 'constant_facility_mw': constant.facility_mw.to_numpy(), 'shape_facility_mw': frame.facility_mw}))
        for case, current in variants[key].items():
            selected = current.iloc[start:start+24]
            record = metrics(current, case)
            sensitivity_rows.append({**record, 'source_date': source_date,
                'daily_facility_mwh': float(selected.facility_mw.sum()), 'day_peak_mw': float(selected.facility_mw.max()),
                'day_minimum_mw': float(selected.facility_mw.min()), 'day_peak_valley_ratio': float(selected.facility_mw.max()/selected.facility_mw.min())})
            sensitivity_hours.append(pd.DataFrame({**fields, 'case': case, 'hour': range(24),
                'it_mw': selected.it_mw.to_numpy(), 'facility_mw': selected.facility_mw.to_numpy()}))
        checks[scenario + '_local_resource_alignment_layer_balance_and_energy_scale'] = bool(.5 < row['daily_to_period_mean_ratio'] < 2)
    summary.update(data_version='v5-eight-scenarios-plus-PR36-native-TRN-shape', training_commit=COMMIT,
        selection_rule=RULE, training_window='90 source UTC days; reference-grid 2025 same month/day/hour weather')
    save_json(output / 'summary.json', summary)
    pd.DataFrame([{**row, 'composition': json.dumps(row['composition'], ensure_ascii=False)} for row in summary['scenarios']]).to_csv(output / 'summary.csv', index=False, encoding='utf-8-sig')
    for name, rows in [('comparison', comparisons), ('sensitivity-summary', sensitivity_rows), ('selected-day-diagnostics', diagnostics)]:
        pd.DataFrame(rows).to_csv(output / (name + '.csv'), index=False, encoding='utf-8-sig')
    pd.concat(matched).to_csv(output / 'matched-day-comparison.csv', index=False, encoding='utf-8-sig')
    pd.concat(sensitivity_hours).to_csv(output / 'sensitivity-hourly.csv', index=False, encoding='utf-8-sig')
    annual = pd.read_csv(previous / 'annual-summary.csv').set_index('scenario')
    periods = []
    for row in summary['scenarios']:
        scenario = row['scenario']
        train = scenario in TRN
        energy = row['period_facility_mwh'] if train else row['annual_facility_mwh']
        hours = 2160 if train else int(annual.loc[scenario, 'hours'])
        periods.append({'scenario': scenario, 'scenario_name': row['scenario_name'], 'period': 'native_90_days' if train else 'v5_annual_replay',
            'hours': hours, 'period_facility_mwh': energy, 'mean_daily_facility_mwh': energy/hours*24,
            'daily_to_period_mean_ratio': row['daily_facility_mwh']/(energy/hours*24)})
    pd.DataFrame(periods).to_csv(output / 'period-summary.csv', index=False, encoding='utf-8-sig')
    write_report(output, summary, comparisons)
    checks.update(plot_all(output, rule_caption=RULE_CAPTION, individual_scenarios=TRN, training_diurnal=True))
    assert all(checks.values())
    save_json(output / 'validation.json', {'status': 'passed', 'checks': checks})
    save_json(output / 'manifest.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'PR36_commit': COMMIT,
        'input_files': inputs, 'local_TRN_hourly_sha256': local_hashes, 'preserved_v5_files_sha256': preserved,
        'code_sha256': {name: sha256(project / 'analysis' / name) for name in (Path(__file__).name,
            'build_trn_diurnal.py', 'trn_power_models.py', 'reselect_scenario_typical_days.py', 'plot_scenario_typical_days_v4.py')}})
    print(json.dumps({'selection': selection, 'comparison': comparisons, 'checks': len(checks)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    build(parser.parse_args().project)
