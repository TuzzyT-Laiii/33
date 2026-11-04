import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from audit_research_gpu_demand import tres_count
from build_scenario_typical_days import choose_day, read_task_resources, save_json
from compute_load.model import sha256
from load_curve_scenarios_v3 import catalogue, combinations, DAYS, power, size, TYPES, USES
from task_trace_inputs import interval_resource_hours


RUN = 'scenario-typical-day-by-task-20261101-v3'
UPSTREAM = 'typical-load-curves-20261101-v3'
OTHER_MAP = {'labelled_dnn': 'DNN 训练（含推断）', 'user_dnn_high': 'DNN 训练（含推断）',
    'user_dnn_medium': 'DNN 训练（含推断）', 'similar_dnn': 'DNN 训练（含推断）',
    'long_single': '长时单任务', 'array': '数组批量', 'large': '多节点大作业',
    'idle': '低 GPU 活动占用', 'interactive': '其他与无法判断', 'startup': '其他与无法判断',
    'short_test': '其他与无法判断', 'unknown': '其他与无法判断'}
OTHER_LABELS = list(dict.fromkeys(OTHER_MAP.values()))
SAI_LABELS = ['批处理', 'MapReduce', '交互式'] + OTHER_LABELS


def integrate(frame, column, edges):
    return interval_resource_hours(frame.time_start.to_numpy(np.int64) * 10**9,
        frame.time_end.to_numpy(np.int64) * 10**9, frame[column].to_numpy(float), edges.asi8)


def sai_resources(args, clocks, baseline, output, inputs):
    columns = ['id_job', 'time_start', 'time_end', 'gpus', 'cpu_allocated', 'category_id', 'classification_version']
    private = pd.read_parquet(args.other_jobs, columns=columns)
    assert private.id_job.is_unique and private.classification_version.eq('mit-other-refine-20261101-v2').all()
    private['label'] = private.category_id.map(OTHER_MAP)
    assert private.label.notna().all()
    private = private.set_index('id_job')
    hours = len(clocks)
    edges = pd.date_range(clocks[0], periods=hours + 1, freq='h')
    values = {resource: {label: np.zeros(hours) for label in SAI_LABELS} for resource in ('cpu', 'gpu')}
    original_other = {resource: np.zeros(hours) for resource in values}
    all_hours = {label: 0. for label in OTHER_LABELS}
    raw = args.mit_root / 'reference/slurm-log.csv'
    source_columns = ['id_job', 'time_start', 'time_end', 'time_suspended', 'tres_req', 'tres_alloc', 'job_type']
    original_labels = {'LLSUB:BATCH': '批处理', 'LLMAPREDUCE:MAP': 'MapReduce', 'LLSUB:INTERACTIVE': '交互式'}
    matched = 0
    for frame in pd.read_csv(raw, usecols=source_columns, chunksize=20000, dtype={'tres_req': str, 'tres_alloc': str}):
        requested = sum(tres_count(frame.tres_req, key).fillna(0) for key in (1001, 1002))
        frame['gpu'] = sum(tres_count(frame.tres_alloc, key).fillna(0) for key in (1001, 1002))
        frame['cpu'] = tres_count(frame.tres_alloc, 1).fillna(0)
        selected = requested.gt(0) & frame.gpu.gt(0) & frame.time_start.gt(0) & frame.time_end.ge(frame.time_start) & frame.time_suspended.eq(0)
        frame = frame[selected].copy()
        frame['label'] = frame.job_type.map(original_labels)
        other = frame.job_type.eq('OTHER')
        part = frame[other]
        tags = private.loc[part.id_job]
        assert np.array_equal(tags.time_start, part.time_start)
        assert np.array_equal(tags.time_end, part.time_end)
        assert np.array_equal(tags.gpus, part.gpu)
        assert np.array_equal(tags.cpu_allocated, part.cpu)
        frame.loc[other, 'label'] = tags.label.to_numpy()
        matched += int(other.sum())
        assert frame.label.notna().all()
        for label, group in frame.groupby('label'):
            for resource in values:
                values[resource][label] += integrate(group, resource, edges)
            if label in OTHER_LABELS:
                all_hours[label] += float(((group.time_end - group.time_start) * group.gpu / 3600).sum())
        for resource in values:
            original_other[resource] += integrate(part, resource, edges)
    assert matched == len(private) == 76404
    published = pd.read_csv(args.project / 'results/mit-other-breakdown-20261031/other-breakdown.csv')
    published['label'] = published.category_id.map(OTHER_MAP)
    expected = published.groupby('label').gpu_hours.sum()
    for label in OTHER_LABELS:
        np.testing.assert_allclose(all_hours[label], expected[label], rtol=1e-10, atol=1e-6)
    prior = pd.read_parquet(args.project / 'data/processed/scenario-typical-day-by-task-20261031/SAI-hourly-resource-fractions.parquet')
    for resource in values:
        np.testing.assert_allclose(sum(values[resource][label] for label in OTHER_LABELS), original_other[resource], rtol=1e-9, atol=1e-6)
        denominator = baseline['sources']['mit_' + resource]['normalization_denominator']
        for label in values[resource]:
            values[resource][label] /= denominator
        np.testing.assert_allclose(sum(values[resource][label] for label in OTHER_LABELS), prior[resource + ':OTHER'], rtol=1e-8, atol=1e-7)
    total_other = float(sum(all_hours.values()))
    total_mit = float(pd.read_csv(args.project / 'results/job-flexibility-constraints-20261030/mit-classes.csv').query('resource_pool == "gpu_hours"').resource_hours.sum())
    shares = pd.DataFrame([{'task_type': label, 'gpu_hours': all_hours[label],
        'share_of_other': all_hours[label] / total_other, 'share_of_mit_gpu_hours': all_hours[label] / total_mit}
        for label in OTHER_LABELS])
    shares.to_csv(output / 'mit-other-full-source-shares.csv', index=False, encoding='utf-8-sig')
    inputs.extend([{'role': 'MIT_private_classification_local_only', 'file': str(args.other_jobs), 'sha256': sha256(args.other_jobs)},
        {'role': 'MIT_slurm_readonly', 'file': str(raw), 'sha256': sha256(raw)}])
    return values


def resources(args, use, baseline, output, inputs):
    hours = DAYS[use] * 24
    if use in ('HPC', 'TRN'):
        source = 'lisa_cpu' if use == 'HPC' else 'acme_gpu'
        clocks = pd.date_range(baseline['sources'][source]['source_start'], periods=hours, freq='h').tolist()
        values = read_task_resources(args, use, clocks, baseline['sources'], inputs)
    elif use == 'SAI':
        clocks = pd.date_range(baseline['sources']['mit_gpu']['source_start'], periods=hours, freq='h').tolist()
        values = sai_resources(args, clocks, baseline, output, inputs)
    elif use == 'CLD':
        path = args.project / 'results/alibaba-cpu-cld-20261031/complete-component-window.csv'
        frame = pd.read_csv(path)
        assert len(frame) == hours and frame.relative_hour.tolist() == list(range(48, 168))
        labels = {'在线服务': 'online_share_of_machine', '离线批处理': 'batch_share_of_machine', '未对账部分': 'residual_share_of_machine'}
        values = {'cpu': {label: (frame.resource_fraction * frame[column]).to_numpy() for label, column in labels.items()},
            'gpu': {label: np.zeros(hours) for label in labels}}
        clocks = [f'相对第{int(day)}天' for day in frame.relative_day]
        inputs.append({'role': use + '_hourly_components', 'file': str(path), 'sha256': sha256(path)})
    else:
        path = args.project / 'results/asi-185day-workload-types-20261031/inf-hourly.csv'
        frame = pd.read_csv(path)
        assert len(frame) == hours and frame.hour_index.tolist() == list(range(hours))
        values = {'gpu': {'在线推理': frame.online_shape_fraction.to_numpy(), '离线推理': frame.offline_shape_fraction.to_numpy()}}
        values['cpu'] = {label: .25 * fraction for label, fraction in values['gpu'].items()}
        clocks = [f'相对第{int(day)}天' for day in frame.day]
        inputs.append({'role': use + '_hourly_components', 'file': str(path), 'sha256': sha256(path)})
    return clocks, values


def split(frame, values, config, key, use, ratios, mit):
    n = len(frame)
    idle = power(config, key, use, np.zeros(n), np.zeros(n), ratios, mit)
    units = int(frame.unit_count.iloc[0])
    layers = {'基础与空载 IT': idle['unit_w'] * units / 1e6}
    for label in values['cpu']:
        cpu_share = np.divide(values['cpu'][label], frame.u_cpu_proxy, out=np.zeros(n), where=frame.u_cpu_proxy.to_numpy() > 0)
        gpu_share = np.divide(values['gpu'][label], frame.allocated_gpu_fraction, out=np.zeros(n), where=frame.allocated_gpu_fraction.to_numpy() > 0)
        layers[label] = (frame.cpu_it_mw.to_numpy() - idle['cpu_w'] * units / 1e6) * cpu_share + (frame.gpu_it_mw.to_numpy() - idle['gpu_w'] * units / 1e6) * gpu_share
    np.testing.assert_allclose(sum(layers.values()), frame.it_mw, rtol=1e-9, atol=1e-8)
    layers['冷却等非 IT'] = (frame.facility_mw - frame.it_mw).to_numpy()
    assert min(float(value.min()) for value in layers.values()) >= -1e-8
    return layers


def brief_facts(project, output):
    other = pd.read_csv(project / 'results/mit-other-breakdown-20261031/other-breakdown.csv')
    other['label'] = other.category_id.map(OTHER_MAP)
    grouped = other.groupby('label')[['pct_mit_gpu_hours', 'pct_other_gpu_hours']].sum()
    asi = pd.read_csv(project / 'results/asi-185day-workload-types-20261031/task-type-summary.csv').set_index('job_type_public')
    flexibility = json.loads((project / 'results' / UPSTREAM / 'flexibility-source-statistics.json').read_text(encoding='utf-8'))
    facts = {'MIT_OTHER': grouped.to_dict(orient='index'), 'ASI_all_business_share': asi.gpu_hour_share.to_dict(),
        'CLD': flexibility['CLD'], 'INF': flexibility['INF'], 'MIT_partial_data': False,
        'MIT_release_status': '六月窗口完成版', 'OTHER_definition': 'PR28互斥主类；百分比为各自总体GPU小时，不能当作可调功率'}
    save_json(output / 'brief-facts.json', facts)


def report(output, summary):
    lines = ['# v3 各情境分任务典型日负荷曲线', '', '![典型日总览](overview.png)', '',
        '采用 PR #34 完成版 v3 的参考网格2025年逐小时 IT、PUE 和设施功率。参考 IT 10 MW；其他规模按整数设备台数比缩放，形状保持。', '',
        '完整日先按日电量与中位数的绝对距离选取，并列时按与全部日平均24小时曲线的RMSE选取，再按日期排序。科研AI、训练和推理均以GPU主导型选日，三规格共用源日期、任务资源构成和天气对应日。', '',
        '图中任务层为资源驱动的IT增量，基础与空载IT及冷却等非IT独立列示。CPU非线性响应按小时资源份额分摊；功率、PUE与v3逐值一致，未重新拟合功率模型。', '',
        '| 情境 | 源典型日 | 星期 | 设施峰值 MW | 设施谷值 MW | 日电量 MWh | 与全年日均之比 | 主要任务构成 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | --- |']
    for row in summary['scenarios']:
        composition = '；'.join(f'{key} {value:.1%}' for key, value in sorted(row['composition'].items(), key=lambda item: -item[1]) if value >= .02)
        lines.append(f"| {row['scenario_name']} | {row['source_date']} | {row['weekday']} | {row['peak_mw']:.3f} | {row['minimum_mw']:.3f} | {row['daily_facility_mwh']:.3f} | {row['daily_to_annual_mean_ratio']:.3f} | {composition} |")
    lines += ['', '## 版本与证据范围', '',
        '- v3：PR #34，ab07a41；OTHER：PR #28，d3fd452。ASI185天与Alibaba拆分分别来自#32、#33。',
        '- MIT来源：Samsi等，The MIT Supercloud Dataset，DOI:10.1109/HPEC49654.2021.9622850；许可CC BY-NC-ND 4.0。仅发布小时资源与功率聚合，逐作业分类及标识保留本机。',
        '- MIT六月窗口完成版：partial_data=false，12,181文件完成；有效监测覆盖68.33%，合格验证3小时。V100→H100及未分配空载仍为迁移假设，文件完成不等于全设施实测。',
        '- DNN训练合并官方标签和三类推断；长时、数组、多节点为运行形态；低GPU活动不证明整个作业空闲。其他与无法判断合并交互候选、短时异常、测试及待定。六类互斥并逐小时加总回原OTHER。',
        '- MIT完整数据集OTHER六层占比见mit-other-full-source-shares.csv，所选典型日OTHER内部及科研AI总体占比见mit-other-typical-day-shares.csv；分母不同，不能互换。前三类原提交方式分别保留。',
        '- INF按v3仅使用在线和离线推理；训练、开发等全业务不进入推理情境。185天均参与完整日选取，索引126未确认原因的零资源小时沿用v3原值与质量标识。',
        '- ASI与Alibaba没有可核实的日历日期及星期，只报告源相对日；Lisa源时区、MIT当地时区未核实。图内日期是源日期，2025年参考天气对应日另列于CSV。',
        '- Alibaba在线、离线和未对账量来自有效核心小时；未对账部分保留，不任意分到业务。原始数据的源相对06时高值真实存在，单份SPECpower曲线50%→60%的陡升进一步放大功率峰形；见旧版cld-spike-audit-20261101证据包。', '',
        '## 文件与复现', '',
        '每情境PNG（300 dpi）/SVG、24行hourly.csv和分层by-task.csv；另有候选日评分、summary.csv/json、输入版本和SHA-256的manifest.json、validation.json。所有逐作业输入和资源累计中间表只留本地忽略目录。', '',
        'independent-verification.json记录独立核查：逐值对照v3、重新排序选日、直接按区间截断计算OTHER资源小时、全源237天OTHER守恒、PNG分辨率与SVG、Word嵌图和四页PDF。manifest.json另记录人工逐页检查状态及PDF指纹；PDF和页面渲染只留本地。', '',
        '```powershell', 'python -m pip install -r dc-flexibility-project/analysis/requirements-scenario-typical-days-v3.txt', "$env:OPENBLAS_NUM_THREADS='1'", "$env:OMP_NUM_THREADS='1'",
        'python dc-flexibility-project/analysis/build_scenario_typical_days_v3.py --input-project "<原始输入项目>" --mit-root "D:/mit-supercloud-gpu-summary-20261028" --other-jobs "D:/mit-supercloud-gpu-summary-20261028/derived/other-refine-20261101/other-job-categories.parquet"',
        '```', '',
        '随后在docs/job-flexibility-brief目录运行`node build.js 数据中心任务时间柔性简表.docx README.md`，将Word导出PDF并逐页检查，再在仓库根目录运行：', '',
        '```powershell', 'python dc-flexibility-project/analysis/verify_scenario_typical_days_v3.py --other-jobs "D:/mit-supercloud-gpu-summary-20261028/derived/other-refine-20261101/other-job-categories.parquet" --pdf "<本机检查后的PDF路径>" --visual-qa-confirmed', '```', '',
        'MIT按20,000行读取Slurm，仅加载76,404条类别的必要字段；不访问流式数据库、锁或STOP文件。旧版结果保留在scenario-typical-day-by-task-20261031。', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def build(args):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    output = args.project / 'results' / RUN
    output.mkdir(parents=True, exist_ok=True)
    local = args.project / 'data/processed' / RUN
    local.mkdir(parents=True, exist_ok=True)
    upstream = args.project / 'results' / UPSTREAM
    source_manifest = json.loads((upstream / 'manifest.json').read_text(encoding='utf-8'))
    baseline = json.loads((args.project / 'results/typical-load-curves-20261029-v2/manifest.json').read_text(encoding='utf-8'))
    mit = json.loads((upstream / 'mit-calibration.json').read_text(encoding='utf-8'))
    assert mit['partial_data'] is False and mit['release_status'] == '六月窗口完成版'
    ratios = json.loads((args.project / 'results/typical-load-curves-20261029-v2/mit-power-ratio-summary.json').read_text(encoding='utf-8'))['transfer_distribution']
    configs = catalogue(args.project)
    annual = pd.read_csv(upstream / 'annual-summary.csv').query('reference_it_mw == 10')
    selections, records, inputs, checks = {}, [], [], {}
    for use in ('HPC', 'CLD', 'SAI', 'TRN', 'INF'):
        clocks, values = resources(args, use, baseline, output, inputs)
        pd.DataFrame({resource + ':' + label: fraction for resource, labels in values.items() for label, fraction in labels.items()}).to_parquet(local / (use + '-resources.parquet'), index=False)
        keys = ['A'] if use in ('HPC', 'CLD') else ['B', 'C', 'D']
        curves = {}
        for key in keys:
            path = upstream / 'hourly' / f'{key}-{use}-10MW-2025.parquet'
            assert sha256(path) == source_manifest['hourly_sha256'][path.relative_to(args.project).as_posix()]
            curves[key] = pd.read_parquet(path)
            for resource, column in (('cpu', 'u_cpu_proxy'), ('gpu', 'allocated_gpu_fraction')):
                np.testing.assert_allclose(sum(values[resource].values()), curves[key][column].to_numpy()[:len(clocks)], rtol=1e-8, atol=1e-7)
            inputs.append({'role': key + '-' + use + '_v3_curve', 'file': str(path), 'sha256': sha256(path)})
        day, selection = choose_day(curves[keys[-1]], clocks, use, output)
        start = day * 24
        source_date = str(clocks[start]) if use in ('CLD', 'INF') else pd.Timestamp(clocks[start]).strftime('%Y-%m-%d')
        weekday = '未知' if use in ('CLD', 'INF') else '星期' + '一二三四五六日'[pd.Timestamp(clocks[start]).dayofweek]
        selections[use] = {**selection, 'day_index': day, 'source_date': source_date, 'weekday': weekday,
            'model_weather_date': str(curves[keys[-1]].time_utc.iloc[start].date())}
        selected_values = {resource: {label: value[start:start + 24] for label, value in labels.items()} for resource, labels in values.items()}
        resource = 'cpu' if use in ('HPC', 'CLD') else 'gpu'
        totals = {label: float(value.sum()) for label, value in selected_values[resource].items()}
        composition = {label: amount / sum(totals.values()) for label, amount in totals.items()}
        if use == 'SAI':
            denominator = baseline['sources']['mit_gpu']['normalization_denominator']
            other_total = sum(totals[label] for label in OTHER_LABELS)
            pd.DataFrame([{'task_type': label, 'gpu_hours': totals[label] * denominator,
                'share_of_other': totals[label] / other_total, 'share_of_sai': composition[label]} for label in OTHER_LABELS]).to_csv(output / 'mit-other-typical-day-shares.csv', index=False, encoding='utf-8-sig')
            checks['OTHER_CPU_GPU_hourly_and_PR28_full_source_conservation'] = True
        for key in keys:
            curve = curves[key]
            selected = curve.iloc[start:start + 24].reset_index(drop=True)
            scenario = key + '-' + use
            name = TYPES[key] + '·' + USES[use]
            layers = split(selected, selected_values, configs[key], key, use, ratios, mit)
            hourly = pd.DataFrame({'scenario': scenario, 'scenario_name': name, 'source_date': source_date, 'weekday': weekday,
                'hour': range(24), 'model_time_utc': selected.time_utc, 'reference_it_mw': 10.,
                'installed_reference_it_mw': selected.installed_reference_it_mw,
                'it_mw': selected.it_mw, 'facility_mw': selected.facility_mw, 'pue': selected.pue,
                'source_quality_flag': selected.source_quality_flag})
            hourly.to_csv(output / (scenario + '-hourly.csv'), index=False, encoding='utf-8-sig')
            parts = [pd.DataFrame({'scenario': scenario, 'scenario_name': name, 'source_date': source_date, 'weekday': weekday,
                'hour': range(24), 'task_type': label, 'it_mw': np.zeros(24) if label == '冷却等非 IT' else value,
                'facility_mw': value, 'component_scope': 'non_IT' if label == '冷却等非 IT' else 'IT'}) for label, value in layers.items()]
            pd.concat(parts).to_csv(output / (scenario + '-by-task.csv'), index=False, encoding='utf-8-sig')
            expected = annual[annual.scenario.eq(scenario)].iloc[0]
            np.testing.assert_allclose(curve.facility_mw.sum(), expected.annual_facility_mwh, rtol=1e-10)
            daily = float(selected.facility_mw.sum())
            relative = daily / (float(expected.annual_facility_mwh) / 365)
            assert .5 < relative < 2
            checks[scenario + '_v3_resources_and_annual_energy'] = True
            checks[scenario + '_task_and_facility_balance'] = bool(np.allclose(sum(layers.values()), selected.facility_mw, atol=1e-8))
            records.append({'scenario': scenario, 'scenario_name': name, 'source_date': source_date, 'weekday': weekday,
                'complete_days': DAYS[use], 'peak_mw': float(selected.facility_mw.max()), 'minimum_mw': float(selected.facility_mw.min()),
                'daily_it_mwh': float(selected.it_mw.sum()), 'daily_facility_mwh': daily,
                'annual_facility_mwh': float(expected.annual_facility_mwh), 'daily_to_annual_mean_ratio': relative,
                'composition': composition, 'composition_basis': '核心小时' if resource == 'cpu' else 'GPU小时',
                'model_weather_date': selections[use]['model_weather_date'], 'unresolved_selected_hours': int(selected.source_unresolved.sum())})
        print(json.dumps({'completed': use, 'source_date': source_date}, ensure_ascii=False), flush=True)
    summary = {'data_version': 'v3-ab07a41', 'mit_release_status': mit['release_status'], 'partial_data': mit['partial_data'],
        'selections': selections, 'scenarios': records}
    save_json(output / 'summary.json', summary)
    pd.DataFrame([{**row, 'composition': json.dumps(row['composition'], ensure_ascii=False)} for row in records]).to_csv(output / 'summary.csv', index=False, encoding='utf-8-sig')
    report(output, summary)
    brief_facts(args.project, output)
    from plot_scenario_typical_days_v3 import plot_all
    checks.update(plot_all(output))
    assert all(checks.values())
    save_json(output / 'validation.json', {'status': 'passed', 'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'checks': checks})
    save_json(output / 'manifest.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'upstream_pr': 34, 'upstream_commit': 'ab07a41157442e837659535e1a33579ced5af8fa',
        'upstream_manifest_sha256': sha256(upstream / 'manifest.json'), 'other_pr': 28,
        'other_commit': 'd3fd452e6e03b2cd00b74352b91dbdd57f437095', 'input_files': inputs,
        'reference': 'v3参考网格g05_r100_c592，2025，10MW，中央风冷；源日首次重放对应天气',
        'category_scope': '六类互斥OTHER主类，DNN含官方和推断；形态不等于用途或柔性权限',
        'privacy': 'MIT逐作业及标识只留本机；仅小时聚合入Git，不接触流式进程或STOP',
        'code_sha256': {name: sha256(args.project / 'analysis' / name) for name in ('build_scenario_typical_days_v3.py', 'plot_scenario_typical_days_v3.py', 'load_curve_scenarios_v3.py')}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--input-project', type=Path, required=True)
    parser.add_argument('--mit-root', type=Path, required=True)
    parser.add_argument('--other-jobs', type=Path, required=True)
    build(parser.parse_args())
