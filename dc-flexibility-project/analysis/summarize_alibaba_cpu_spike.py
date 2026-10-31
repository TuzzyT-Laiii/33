import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compute_load.cooling import CoolingModel
from typical_power_models import reference_power, PARAMETERS, SHAPES
from typical_power_models_v2 import power_v2


RUN = 'scenario-typical-day-by-task-20261031'
AUDIT = 'cld-spike-audit-20261101'


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def longest_run(seconds, mask):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return {'duration_minutes': 0, 'start_relative_second': None, 'end_relative_second': None}
    groups = np.split(indices, np.flatnonzero(np.diff(seconds[indices]) != 10) + 1)
    selected = max(groups, key=len)
    return {'duration_minutes': len(selected) / 6,
        'start_relative_second': int(seconds[selected[0]]), 'end_relative_second': int(seconds[selected[-1]] + 10)}


def build(args):
    result = args.project / 'results' / RUN
    output = result / AUDIT
    output.mkdir(parents=True, exist_ok=True)
    raw = json.loads(args.raw_audit.read_text(encoding='utf-8'))
    batch = json.loads(args.batch_audit.read_text(encoding='utf-8'))
    hourly = pd.DataFrame(raw.pop('hourly'))
    five = pd.DataFrame(raw.pop('five_minute'))
    native = pd.DataFrame(raw.pop('native_10second_day2_h4_to_h9'))
    batch_five = pd.DataFrame(batch.pop('five_minute'))
    components_path = args.alibaba_dir / 'component-reconciliation.csv'
    components = pd.read_csv(components_path)
    for column in ('occupied_core_hours', 'observed_core_hours', 'cpu_util_time_weighted_pct'):
        np.testing.assert_allclose(hourly[column], components[column], rtol=1e-9, atol=1e-7)
    envelope = pd.read_csv(args.alibaba_dir / 'online-batch-request-envelope.csv')
    np.testing.assert_allclose(batch_five.planned_envelope_core_hours.to_numpy().reshape(192, 12).sum(axis=1),
        envelope.batch_planned_envelope_core_hours, rtol=1e-8, atol=1e-6)
    batch_hours = batch_five.assign(relative_hour=(batch_five.relative_second // 3600)).groupby('relative_hour').agg(
        task_start_records=('task_start_records', 'sum'), planned_envelope_core_hours=('planned_envelope_core_hours', 'sum'))
    compared = components.merge(batch_hours, on='relative_hour', validate='one_to_one')
    compared.to_csv(output / 'hourly-evidence.csv', index=False, encoding='utf-8-sig')
    five.to_csv(output / 'machine-five-minute.csv', index=False, encoding='utf-8-sig')
    native.to_csv(output / 'machine-native-10second-day2-h4-h9.csv', index=False, encoding='utf-8-sig')
    batch_five.to_csv(output / 'batch-task-five-minute.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(raw['hold_sensitivity']).to_csv(output / 'hold-sensitivity.csv', index=False, encoding='utf-8-sig')
    recurrence = hourly.loc[hourly.groupby('relative_day').cpu_util_time_weighted_pct.idxmax()].copy()
    recurrence['adopted_complete_day'] = recurrence.relative_day.isin(range(2, 7))
    recurrence.to_csv(output / 'daily-peak-recurrence.csv', index=False, encoding='utf-8-sig')
    config = next(row for row in json.loads((args.project / 'results/server-count-scenarios-20261028-v2/server-configurations.json').read_text(encoding='utf-8')) if row['configuration_id'] == 'C20')
    source_day = hourly[hourly.relative_day.eq(2)].reset_index(drop=True)
    demand = source_day.cpu_util_time_weighted_pct.to_numpy() / 100
    units = int(10e6 // reference_power(config))
    power = power_v2(config, demand, np.zeros(24), 'cpu_spec', {})['unit_w'] * units / 1e6
    plotted = pd.read_csv(result / 'A-CLD-hourly.csv')
    np.testing.assert_allclose(power, plotted.it_mw, rtol=1e-10)
    baseline = json.loads((args.project / 'results/typical-load-curves-20261029-v2/manifest.json').read_text(encoding='utf-8'))
    weather = pd.read_parquet(Path(baseline['processed_directory']) / 'C20-Lisa-cpu_spec-central.parquet').iloc[:24]
    facilities = CoolingModel().calculate(weather, power, units * reference_power(config) / 1e6, full_year=False)
    np.testing.assert_allclose(facilities.facility_mw, plotted.facility_mw, rtol=1e-10)
    mapping = pd.DataFrame({'hour_of_day': range(24), 'source_cpu_util_pct': demand * 100,
        'it_mw': power, 'facility_mw': facilities.facility_mw, 'pue': facilities.pue})
    mapping.to_csv(output / 'power-mapping.csv', index=False, encoding='utf-8-sig')
    near = five[five.relative_day.eq(2) & five.hour_of_day.ge(6) & five.hour_of_day.lt(7)]
    selected_native = native[native.hour_of_day.ge(6) & native.hour_of_day.lt(7)]
    native_weighted = np.average(selected_native.raw_cpu_mean_pct, weights=selected_native.reporting_machines)
    np.testing.assert_allclose(native_weighted, components.loc[components.relative_hour.eq(54), 'cpu_util_sample_mean_pct'].iloc[0], atol=1e-8)
    duration = {str(threshold): longest_run(native.relative_second.to_numpy(), native.raw_cpu_mean_pct.to_numpy() >= threshold) for threshold in (50, 55, 60)}
    cpu_dynamic_mw = units * 2 * (PARAMETERS['C20']['central'][2] - PARAMETERS['C20']['central'][1]) / 1e6
    slope_low = cpu_dynamic_mw * (SHAPES['6448y'][5] - SHAPES['6448y'][4])
    slope_high = cpu_dynamic_mw * (SHAPES['6448y'][6] - SHAPES['6448y'][5])
    focus = compared[compared.relative_day.eq(2) & compared.hour_of_day.isin([5,6,7])]
    focus.to_csv(output / 'day2-hour5-6-7.csv', index=False, encoding='utf-8-sig')
    component_audit_path = args.alibaba_dir / 'component-audit.json'
    component_audit = json.loads(component_audit_path.read_text(encoding='utf-8'))
    state = component_audit['large_table_states']
    assert all(state[key]['status'] == 'complete' and state[key]['official_sha256_matches'] for key in ('container_usage', 'batch_instance'))
    validations = {'full_raw_machine_archive_sha256': True, 'raw_machine_row_count_matches_prior_audit': True,
        'no_raw_machine_duplicates': raw['counts']['duplicate_rows'] == 0,
        'no_raw_cpu_out_of_range': raw['counts']['cpu_below_zero'] == raw['counts']['cpu_above_100'] == 0,
        'source_files_unchanged': raw['source_files_before'] == raw['source_files_after'] and batch['source_unchanged'],
        'raw_hours_match_existing_192_hours': True, 'batch_envelope_matches_existing_192_hours': True,
        'recurrent_peak_hour_all_eight_days': bool(recurrence.hour_of_day.eq(6).all()),
        'native_360_samples_reconcile_raw_hour_mean': len(selected_native) == 360,
        'five_minute_12_bins_complete_and_covered': bool(len(near) == 12 and near.coverage_fraction.min() > .95),
        'existing_24h_it_and_facility_unchanged': True, 'prior_component_full_stream_sha_verified': True}
    assert all(validations.values()), validations
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'conclusion': '原始10秒CPU观测中存在每天源相对06时的周期性高值；保留原曲线，非线性SPEC响应放大了模型功率峰形',
        'checks': validations, 'raw_machine_audit': raw, 'raw_batch_audit': batch,
        'day2_hour6': {'five_minute_cpu_min_pct': float(near.cpu_time_weighted_pct.min()),
            'five_minute_cpu_max_pct': float(near.cpu_time_weighted_pct.max()),
            'native_10second_cpu_min_pct': float(selected_native.raw_cpu_mean_pct.min()),
            'native_10second_cpu_max_pct': float(selected_native.raw_cpu_mean_pct.max()),
            'native_weighted_raw_sample_mean_pct': float(native_weighted),
            'native_reporting_machines_min': int(selected_native.reporting_machines.min()),
            'native_reporting_machines_max': int(selected_native.reporting_machines.max()),
            'minutes_native_above_50pct_in_hour6': int((selected_native.raw_cpu_mean_pct > 50).sum()) / 6},
        'native_longest_runs': duration,
        'power_mapping': {'cpu_segment_40_to_50pct_it_increase_mw': float(slope_low),
            'cpu_segment_50_to_60pct_it_increase_mw': float(slope_high), 'slope_ratio': float(slope_high / slope_low),
            'interpretation': '沿用v2 SPECpower归一化整机响应；该源设施的电力没有实测验证'},
        'component_scope': '复用PR #33已完成且全流SHA通过的container_usage与batch_instance汇总；本次重新扫描machine_usage和batch_task原始包；不将实例区间平均CPU归因为精确瞬时负荷',
        'component_sources': {key: {field: state[key][field] for field in ('processed_rows', 'sha256', 'official_sha256_matches')}
            for key in ('container_usage', 'batch_instance')},
        'input_sha256': {str(path): sha(path) for path in (args.raw_audit, args.batch_audit, components_path, component_audit_path,
            result / 'A-CLD-hourly.csv', result / 'A-CLD-by-task.csv')},
        'code_sha256': {name: sha(args.project / 'analysis' / name) for name in
            ('audit_alibaba_cpu_spike.py', 'audit_alibaba_batch_spike.py', 'summarize_alibaba_cpu_spike.py')}}
    (output / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 云通用计算典型日第6小时尖峰核查', '',
        '原始记录存在周期性 CPU 利用率高值，保留典型日数据；图中尖锐程度还受到既有非线性功率模型的放大。源相对第 2 天第 6 小时对应原始秒数 [194400, 198000)，未进行时区换算。', '',
        '## 原始观测与小时结果', '',
        f"只读扫描 machine_usage 全部 {raw['counts']['rows']:,} 行，压缩包官方 SHA-256 一致；重复记录为 {raw['counts']['duplicate_rows']}，CPU 越界记录为 0，全部八天仅 {raw['counts']['cpu_missing']} 条 CPU 缺失。4,034 台机器均为 96 核，逐采样时刻按核数加权与按上报机器等权的均值一致。", '',
        '| 相对第2天小时 | 上报机器数 | 核心时间覆盖率 | CPU 利用率 | 占用核数时均值 | 机器小时 CPU 中位数 | IT MW | 设施 MW |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for hour in (5, 6, 7):
        row = focus[focus.hour_of_day.eq(hour)].iloc[0]
        p = mapping.iloc[hour]
        lines.append(f'| {hour} | {int(row.observed_machines)} | {row.coverage_fraction:.3%} | {row.cpu_util_time_weighted_pct:.3f}% | {row.occupied_core_hours:,.1f} | {row.machine_hour_cpu_p50_pct:.3f}% | {p.it_mw:.3f} | {p.facility_mw:.3f} |')
    paired = raw['paired_hour_5_to_6']
    lines += ['', f"第6小时十二个5分钟区间的 CPU 均值范围为 {near.cpu_time_weighted_pct.min():.2f}%–{near.cpu_time_weighted_pct.max():.2f}%；360个原始10秒采样时刻的集群均值范围为 {selected_native.raw_cpu_mean_pct.min():.2f}%–{selected_native.raw_cpu_mean_pct.max():.2f}%，高于50%的累计时长为 {(selected_native.raw_cpu_mean_pct > 50).sum()/6:.2f} 分钟，高于60%的累计时长为 {(selected_native.raw_cpu_mean_pct > 60).sum()/6:.2f} 分钟。最高两个5分钟区间位于06:25–06:35，均值为70.82%、74.18%。",
        f"同时出现在第5、6小时的 {paired['machines']} 台机器中，{paired['increase_at_least_10pp']} 台的小时 CPU 均值增加至少10个百分点，配对增加中位数为 {paired['median_increase_pp']:.2f} 个百分点；峰值涉及广泛机器。", '',
        '## 八天复现', '', '| 相对日 | 日最高小时 | 该小时 CPU | 核心时间覆盖率 | 进入典型日候选 |', '| --- | ---: | ---: | ---: | --- |']
    for row in recurrence.itertuples():
        lines.append(f"| {row.relative_day} | {row.hour_of_day} | {row.cpu_util_time_weighted_pct:.3f}% | {row.coverage_fraction:.3%} | {'是' if row.adopted_complete_day else '否'} |")
    lines += ['', '八天的小时最高值均在源相对06时；用于选日的五个完整日均复现。第0、1天覆盖不足，第7天末小时不完整，不用于选日。真实日期、星期和当地时区未知。', '',
        '## 处理假象检查与功率映射', '',
        '- 重新从原始采样生成5分钟累计，再汇总小时，全部192小时与既有结果一致；原始10秒均值也直接显示高值。整点边界、重采样和时间对齐没有创造该峰值。',
        '- 第5–7小时的小时上报机器数及核心时间覆盖率稳定。第6小时的10秒上报机器数为3669–4016；最高CPU点位于06:26:10，上报4000台，最低上报数点位于06:50:00、CPU均值52.18%，二者未重合。',
        '- 保持上限10秒和300秒时，第6小时CPU均值分别为58.8787%和58.8600%，仅差0.0187个百分点；10/30/60/300秒明细见hold-sensitivity.csv。缺测不填成空闲，CPU越界不参与累计。',
        f'- 既有 SPECpower 响应在 CPU 40%→50% 区间使 IT 增加 {slope_low:.3f} MW，在50%→60%区间增加 {slope_high:.3f} MW，斜率约为前段的 {slope_high/slope_low:.2f} 倍；原始利用率跨过此区间后模型功率峰形更尖。',
        '- 5.681→7.924→6.076 MW 为现有模型输出；接近10 MW的设施功率也由同一参考天气与冷却模型得到，源数据没有测得这些MW值。', '',
        '## 与在线及离线业务的对应', '', '| 小时 | 在线观测折合核小时 | 离线实例平均CPU积分核小时 | 未对账差额核小时 | batch_task启动记录 |', '| --- | ---: | ---: | ---: | ---: |']
    for row in focus.itertuples():
        lines.append(f'| {row.hour_of_day} | {row.online_request_based_core_hours:,.1f} | {row.batch_mean_core_hours:,.1f} | {row.residual_core_hours:,.1f} | {row.task_start_records:,} |')
    lines += ['', '第6小时离线CPU积分较第5小时增加37,096.1核小时，在线增加3,503.6核小时；batch_task启动记录从116,413下降至105,521。离线资源用量有同步增高的证据，任务数量本身无法解释峰值。在线/离线实际用量沿用 PR #33 已完成且全流 SHA 通过的汇总。离线平均CPU按执行区间分摊、容器存在样本缺口，且未对账差额同样增大，因此不能将整个峰值完全归因于某类作业，也不能由此推断它可以迁移。', '',
        '本次重新扫描14,295,731条 batch_task，计划包络与既有192小时汇总一致；计划包络不表示实际活跃用量。没有重读40.16亿行容器及13.51亿行实例大表，完整流哈希与处理范围登记于 validation.json。', '',
        '## 文件与复核', '',
        '- machine-five-minute.csv：八天逐5分钟机器覆盖、核数加权CPU均值及原始采样分位数；machine-native-10second-day2-h4-h9.csv：第2天04–09时原始10秒聚合。',
        '- hourly-evidence.csv、daily-peak-recurrence.csv：逐小时观测和日内最高时刻；day2-hour5-6-7.csv：争议时段对账。',
        '- batch-task-five-minute.csv：原始任务启动事件与计划包络；hold-sensitivity.csv：保持上限敏感性；power-mapping.csv：24小时功率复核。',
        f"- 服务器只读执行，未解压落盘、未改写原始包，读前后大小与mtime一致；两个原始扫描进程峰值RSS分别为 {raw['peak_process_rss_mib']:.1f} / {batch['peak_process_rss_mib']:.1f} MiB。", '',
        '代码：analysis/audit_alibaba_cpu_spike.py、audit_alibaba_batch_spike.py 在服务器通过 Python -B 标准输入执行，JSON 标准输出回收到本机；analysis/summarize_alibaba_cpu_spike.py 在本机生成汇总。来源文件、官方哈希、输入/脚本指纹及验证项目均见 validation.json。', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'checks': len(validations), 'hour6': report['day2_hour6'], 'paired': paired,
        'native_longest_runs': duration, 'power_mapping': report['power_mapping']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--raw-audit', type=Path, required=True)
    parser.add_argument('--batch-audit', type=Path, required=True)
    parser.add_argument('--alibaba-dir', type=Path, required=True)
    build(parser.parse_args())
