import argparse
from datetime import datetime, timezone
from hashlib import file_digest
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa

from build_scenario_typical_days_v4 import split_layers
from load_curve_scenarios_v3 import catalogue
from plot_flex_classes import plot_all


RUN = 'flex-class-20261101'
BASE = 'typical-load-curves-20261101-v4-seren'
CLASSES = ['刚性', '可推迟·整体', '可推迟·可拆分', '未知']
LAYERS = ['基础与空载 IT', *CLASSES, '冷却等非 IT']
MAPPING = {
    'HPC': {'Jupyter': 0, 'CPU 科研分区': 1, '课程分区': 3, '其他分区': 3},
    'CLD': {'在线服务': 0, '离线批处理': 1, '未对账部分': 3},
    'SAI': {'交互式': 0, '低 GPU 活动占用': 0, '批处理': 1, 'MapReduce': 1, '数组批量': 1,
            'DNN 训练（含推断）': 2, '长时单任务': 2, '多节点大作业': 2, '其他与无法判断': 3},
    'TRN': {'调试': 0, '评测': 1, '预训练': 2, 'SFT': 2, 'MLLM（多模态大模型）': 2, '其他': 3},
    'INF': {'在线推理': 0, '离线推理': 1},
}
REPRESENTATIVES = [
    ['用途', '① 刚性', '② 可推迟·整体', '③ 可推迟·可拆分', '未知'],
    ['科研计算', '交互', '批作业', '—', '课程等'],
    ['云通用计算', '在线服务', '离线批处理', '—', '未对账'],
    ['科研 AI', '交互', '批量作业', '模型训练', '无法判断'],
    ['大模型训练/开发', '调试', '评测', '训练', '其他'],
    ['推理服务', '在线推理', '离线推理', '—', '—'],
]
ORDER = ['A-HPC', 'A-CLD', 'B-SAI', 'C-SAI', 'D-SAI', 'B-TRN', 'C-TRN', 'D-TRN', 'B-INF', 'C-INF', 'D-INF']


def digest(path):
    with path.open('rb') as handle:
        return file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def combine(fine, use):
    normalized = {('评测' if name == '评估' else name): np.asarray(value) for name, value in fine.items()}
    assert set(normalized) == set(MAPPING[use]) | {'基础与空载 IT', '冷却等非 IT'}
    result = {name: np.zeros(len(next(iter(normalized.values())))) for name in LAYERS}
    for name, values in normalized.items():
        target = name if name in ('基础与空载 IT', '冷却等非 IT') else CLASSES[MAPPING[use][name]]
        result[target] += values
    np.testing.assert_allclose(sum(fine.values()), sum(result.values()), atol=1e-10, rtol=1e-12)
    assert min(value.min() for value in result.values()) >= -1e-8
    return result


def resource_frame(frame):
    labels = {'Pretrain': '预训练', 'SFT': 'SFT', 'MLLM': 'MLLM（多模态大模型）', 'Eval': '评测', 'Debug': '调试', 'Other': '其他'}
    return pd.DataFrame({prefix + label: frame['gpu_hours_' + kind].to_numpy() / 2032 * factor
        for kind, label in labels.items() for prefix, factor in [('gpu:', 1), ('cpu:', .25)]})


def build(project, resources):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    base, output = project / 'results' / BASE, project / 'results' / RUN
    for folder in ('native-hourly', 'native-by-task', 'selected-days'):
        (output / folder).mkdir(parents=True, exist_ok=True)
    old_hashes = {p.relative_to(base).as_posix(): digest(p) for p in base.rglob('*') if p.is_file()}
    inputs = {}
    def track(path):
        inputs[path.relative_to(project).as_posix()] = digest(path)
        return path
    configs = catalogue(project)
    ratios = json.loads(track(project / 'results/typical-load-curves-20261029-v2/mit-power-ratio-summary.json').read_text('utf-8'))['transfer_distribution']
    mit = json.loads(track(project / 'results/typical-load-curves-20261101-v3/mit-calibration.json').read_text('utf-8'))
    shape = json.loads(track(project / 'results/trn-power-model-20261101/cpu-shape.json').read_text('utf-8'))
    selected = json.loads(track(base / 'selected-days/summary.json').read_text('utf-8'))
    selected_rows = {r['scenario']: r for r in selected['scenarios']}
    summary = pd.read_csv(track(base / 'native-summary.csv'), float_precision='round_trip').query('reference_it_mw == 10').set_index('scenario_id')
    placements = pd.read_csv(track(base / 'placement-summary.csv'), float_precision='round_trip')
    expected = json.loads(track(project / 'results/scenario-typical-day-by-task-20261101-v4/manifest.json').read_text('utf-8'))['input_files']
    expected = {r['role']: r['sha256'] for r in expected}
    resource_cache, records, wide_records, checks, chosen_placements = {}, [], [], [], []
    for scenario in ORDER:
        key, use = scenario.split('-')
        sid = scenario + '-10MW'
        native = summary.loc[sid]
        options = placements.loc[placements.scenario_id.eq(sid)]
        complete = options.loc[options.hours.eq(options.native_hours)].copy()
        median = float(options.mean_facility_mw.median())
        chosen = complete.assign(distance=(complete.mean_facility_mw - median).abs()).sort_values(['distance', 'placement'], kind='stable').iloc[0]
        placement = chosen.placement
        chosen_placements.append({'scenario': scenario, 'scenario_name': native.scenario_zh, 'placement': placement,
            'native_hours': int(native.native_hours), 'facility_mean_mw': float(chosen.mean_facility_mw),
            'all_placements_facility_mean_mw_median': median, 'facility_mean_range_min_mw': float(options.mean_facility_mw.min()),
            'facility_mean_range_max_mw': float(options.mean_facility_mw.max()), 'placements': len(options),
            'policy': '限定完整原生窗口，再取设施均值最接近全部实际放置均值中位数者；并列按placement排序'})
        frame = pd.read_parquet(track(base / 'hourly' / f'{sid}-2025-{placement}.parquet'))
        assert len(frame) == int(native.native_hours)
        if use == 'TRN':
            resource = resource_frame(frame)
        else:
            if use not in resource_cache:
                path = track(resources / (use + '-resources.parquet'))
                assert digest(path) == expected[use + '_local_classified_resource_hours']
                resource_cache[use] = pd.read_parquet(path)
            resource = resource_cache[use]
        assert len(resource) == len(frame)
        np.testing.assert_allclose(resource.filter(regex='^cpu:').sum(axis=1), frame.u_cpu_proxy, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(resource.filter(regex='^gpu:').sum(axis=1), frame.allocated_gpu_fraction, atol=1e-10, rtol=1e-10)
        fine = split_layers(frame, resource, configs[key], key, use, ratios, mit, shape)
        classes = combine(fine, use)
        np.testing.assert_allclose(sum(classes[x] for x in LAYERS[:-1]), frame.it_mw, atol=1e-8, rtol=1e-12)
        np.testing.assert_allclose(sum(classes.values()), frame.facility_mw, atol=1e-8, rtol=1e-12)
        hours = len(frame)
        it_energy = float(frame.it_mw.sum())
        wide = {'scenario': scenario, 'scenario_name': native.scenario_zh, 'native_hours': hours,
            'reference_it_mw': 10, 'location_id': 'reference-grid', 'weather_year': 2025,
            'total_it_mwh': it_energy, 'total_mean_it_mw': float(frame.it_mw.mean())}
        identity = wide.copy()
        for label in LAYERS[:-1]:
            energy, mean = float(classes[label].sum()), float(classes[label].mean())
            pct_energy, pct_power = 100 * energy / it_energy, 100 * mean / frame.it_mw.mean()
            np.testing.assert_allclose(pct_energy, pct_power, atol=1e-12)
            records.append({**identity, 'class': label, 'it_mwh': energy, 'mean_it_mw': mean,
                'share_of_it_energy_pct': pct_energy, 'share_of_mean_it_power_pct': pct_power})
            wide[label] = pct_energy
        np.testing.assert_allclose(sum(wide[x] for x in LAYERS[:-1]), 100, atol=1e-9)
        wide_records.append(wide)
        metadata = frame[[c for c in ('source_hour_index', 'source_clock', 'time_utc', 'hour_of_source_day', 'hour_of_source_week') if c in frame]].reset_index(drop=True)
        pd.concat([metadata, pd.DataFrame(fine)], axis=1).rename(columns={'评估': '评测'}).to_parquet(output / 'native-by-task' / (scenario + '.parquet'), index=False, compression='zstd')
        pd.concat([metadata, pd.DataFrame(classes)], axis=1).assign(it_mw=frame.it_mw, facility_mw=frame.facility_mw,
            scenario=scenario, scenario_name=native.scenario_zh, placement=placement).to_parquet(output / 'native-hourly' / (scenario + '.parquet'), index=False, compression='zstd')
        selected_path = track(base / 'selected-days' / (scenario + '-by-task.csv'))
        parts = pd.read_csv(selected_path, float_precision='round_trip')
        day_fine = {label: group.sort_values('hour').facility_mw.to_numpy() for label, group in parts.groupby('task_type', sort=False)}
        day_classes = combine(day_fine, use)
        row = selected_rows[scenario]
        fields = {name: row[name] for name in ('scenario', 'scenario_name', 'source_date', 'weekday')}
        pd.concat([pd.DataFrame({**fields, 'hour': range(24), 'task_type': label, 'facility_mw': values,
            'it_mw': np.zeros(24) if label == '冷却等非 IT' else values}) for label, values in day_classes.items()]).to_csv(
                output / 'selected-days' / (scenario + '-by-task.csv'), index=False, encoding='utf-8-sig')
        source_hourly = track(base / 'selected-days' / (scenario + '-hourly.csv'))
        shutil.copyfile(source_hourly, output / 'selected-days' / source_hourly.name)
        current = pd.read_csv(source_hourly, float_precision='round_trip')
        np.testing.assert_allclose(sum(day_classes.values()), current.facility_mw, atol=1e-8)
        if row['placement'] == placement:
            if use == 'TRN':
                indices = np.flatnonzero(pd.to_datetime(frame.source_clock, utc=True).dt.strftime('%Y-%m-%d').eq(row['source_date']))
            else:
                day_index = selected['selections'][use]['day_index']
                indices = np.arange(day_index * 24, (day_index + 1) * 24)
            for label in LAYERS:
                np.testing.assert_allclose(day_classes[label], classes[label][indices], atol=1e-8, rtol=1e-10)
        checks.append({'scenario': scenario, 'hours': hours, 'native_IT_balance_max_mw': float(np.abs(sum(classes[x] for x in LAYERS[:-1]) - frame.it_mw).max()),
            'native_facility_balance_max_mw': float(np.abs(sum(classes.values()) - frame.facility_mw).max()),
            'selected_day_total_unchanged_max_mw': float(np.abs(sum(day_classes.values()) - current.facility_mw).max())})
    save(output / 'selected-days/summary.json', selected)
    pd.DataFrame(records).to_csv(output / 'native-class-summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(wide_records).to_csv(output / 'shares-10MW.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(chosen_placements).to_csv(output / 'representative-placements.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([{'use': use, 'source_label': label, 'class': CLASSES[index]} for use, mapping in MAPPING.items() for label, index in mapping.items()]).to_csv(
        output / 'task-class-mapping.csv', index=False, encoding='utf-8-sig')
    rows = [['情境（参考IT 10MW）', '① 刚性\n%', '② 可推迟·整体\n%', '③ 可推迟·可拆分\n%', '未知\n%', '基础与空载 IT\n%']]
    rows += [[r['scenario_name'], *[f'{r[x]:.2f}' for x in [*CLASSES, '基础与空载 IT']]] for r in wide_records]
    save(output / 'brief-summary.json', {'classification_rows': REPRESENTATIVES, 'share_rows': rows,
        'class_order': CLASSES, 'legend_order': [*CLASSES, '基础与空载 IT', '冷却等非 IT'],
        'share_denominator': '同一原生小时集合的总IT电量或总IT平均功率，两种比例相同；基础与空载IT单列；冷却非IT排除'})
    plot_all(output)
    assert all(digest(base / path) == value for path, value in old_hashes.items())
    save(output / 'manifest.json', {'utc': datetime.now(timezone.utc).isoformat(), 'base_commit': '1b839396f49a4445f7b5b404a9a6768daf39c6ac',
        'source_result': BASE, 'mapping_authority': '用户2026-11-01确认，2026-11-02补充全部图统一四类',
        'input_sha256': inputs, 'old_result_all_files_sha256': old_hashes, 'previous_directory_unchanged': True,
        'scenarios': 11, 'checks': checks, 'task_classification_is_scenario_assumption': True,
        'code_sha256': {name: digest(project / 'analysis' / name) for name in ('build_flex_classes.py', 'plot_flex_classes.py')}})
    report(output, rows)
    print(json.dumps({'output': str(output), 'rows': rows, 'checks': checks}, ensure_ascii=False), flush=True)


def report(output, rows):
    lines = ['# 按推迟与拆分能力统一的四类任务负荷', '',
        '采用用户确认的①刚性、②可推迟·整体、③可推迟·可拆分及未知分类；未知默认按刚性处理、仍单列以显示证据缺口。此映射为研究情景分类，未从排队、优先级或运行形态推断真实调度权限。', '',
        '正文代表名称表每格一个名称，原始细类完整映射见task-class-mapping.csv。评测统一用中文“评测”，源Seren字段Eval对应此类。', '',
        '## 10MW原生窗口IT份额', '',
        '分母为参考网格2025气象下同一完整原生窗口的总IT电量。每类平均功率=该类窗口电量/保留小时数，所以平均功率份额与电量份额完全相同；同表报告两者。基础与空载IT单列，与四类合计100%；冷却等非IT不计入IT分母。', '',
        '按原有模型，用逐小时CPU/GPU任务资源份额分摊超过无任务端点的动态IT功率，再按确认映射合并；推理在线/离线GPU仍采用各自功率公式，Seren共用同小时监测反推比例。表中为模型归因份额，缺少推迟时长、最短分段、存档恢复成本及服务期限数据，不据此给出可调电量。', '',
        '| ' + ' | '.join(x.replace('\n', '') for x in rows[0]) + ' |', '| ' + ' | '.join(['---']*6) + ' |']
    lines += ['| ' + ' | '.join(row) + ' |' for row in rows[1:]]
    lines += ['', '![原生窗口IT电量构成](shares-10MW.png)', '',
        '## 同一分类与配色的负荷曲线', '', '![典型日](selected-days/overview.png)', '', '![原生窗口](native-overview.png)', '',
        '图例固定为刚性、可推迟·整体、可推迟·可拆分、未知、基础与空载IT、冷却等非IT。典型日日期、小时功率与原版完全相同；仅合并分类并重画。', '',
        '原生时序保留各来源时长：Lisa7416小时，Alibaba120小时，MIT5688小时，Seren2057个合格小时（189内部缺口小时保留，真实UTC时间不压缩），ASI4440小时。沿用日期气象对应及无日期来源12个月初放置、年末截断；原生统计不年化。', '',
        '无日期来源先限定覆盖完整原生窗口的放置，再选择设施均值最接近全部12次实际放置设施均值中位数者；代表值是实际一次放置，不是逐小时中位数合成曲线。representative-placements.csv记录选择及全部放置的设施均值范围，范围同时含气象和截断源前缀差异。IT任务份额不受天气放置影响，直接取完整原生窗口；原有第五章汇总仍报告全部放置的中位数和范围。', '',
        '## 文件与核验', '',
        'native-class-summary.csv为每情境四类及基础层的窗口MWh、平均MW、电量占比和平均功率占比；shares-10MW.csv为11行情境汇总；native-by-task保留小时级原细类功率归因，native-hourly为六层小时功率；selected-days保留分类后24小时表及11份独立PNG/PDF。三个Word图件为典型日总览、原生堆叠时序及份额条形图。', '',
        'manifest.json记录所有输入指纹、旧结果目录全部文件指纹及逐小时守恒/典型日不变检查。brief-audit.json记录独立数值核验、Word/PDF版面及逐页检查。未重算功率、冷却或典型日日期；MIT流式目录、STOP、看门狗及旧结果目录保持原状。', '',
        '```powershell', 'python dc-flexibility-project/analysis/build_flex_classes.py', '```', '',
        '本机分类资源小时缓存为data/processed/scenario-typical-day-by-task-20261101-v3，逐份匹配既有manifest；Seren直接读取#38的合格小时与六类资源列，不使用旧训练缓存。Word由docs/job-flexibility-brief/build.js生成，用export_brief_pdf.ps1导出PDF并逐页检查，再运行audit_flex_classes.py。', '',
        '同一PR #37继续审阅；依赖顺序：#27；#30→#32、#33；#34、#35、#36→#38；#31；#37。不合并。', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--resources', type=Path)
    args = parser.parse_args()
    build(args.project, args.resources or args.project / 'data/processed/scenario-typical-day-by-task-20261101-v3')
