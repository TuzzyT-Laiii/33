import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from annual_task_profiles import load_sources
from audit_research_gpu_demand import tres_count
from compute_load.cooling import CoolingModel
from compute_load.model import sha256
from task_trace_inputs import interval_resource_hours, load_acme
from typical_power_models import reference_power
from typical_power_models_v2 import power_v2


RUN = 'scenario-typical-day-by-task-20261031'
BASELINE = 'typical-load-curves-20261029-v2'
SPECS = {'A': ('C20', '纯 CPU 型'), 'B': ('C21', 'CPU 主导型'),
         'C': ('C22', '均衡型'), 'D': ('C28', 'GPU 主导型')}
USES = {'HPC': ('Lisa', '科研计算'), 'CLD': ('Alibaba', '云通用计算'),
        'SAI': ('MIT', '科研 AI'), 'TRN': ('Acme', '大模型训练/开发'), 'INF': ('ASI', '推理服务')}
TYPES = {'HPC': ['CPU 科研分区', 'Jupyter', '课程分区', '其他分区'],
         'CLD': ['业务未分解'], 'SAI': ['批处理', 'MapReduce', '交互式', 'OTHER'],
         'TRN': ['预训练', 'SFT', '评估', '调试', '其他'],
         'INF': ['训练', '在线推理', '离线推理', '开发', '其他/未知']}
RULE = '完整日按日电量与中位数的绝对距离升序；数值并列（1e-9 MWh）按与全部日平均24小时曲线的RMSE升序，再按源日期升序；GPU需求以GPU主导型选日，三规格沿用同日同资源构成。'


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def integrate(frame, start, end, units, edges):
    return interval_resource_hours(pd.DatetimeIndex(frame[start]).as_unit('ns').asi8,
        pd.DatetimeIndex(frame[end]).as_unit('ns').asi8, frame[units], edges.asi8)


def accumulate(frame, labels, start, end, resource, edges, output):
    for label, group in frame.groupby(labels, observed=True):
        if label not in output:
            raise ValueError('未识别的任务分类：' + str(label))
        output[label] += integrate(group, start, end, resource, edges)


def empty_resources(use, hours):
    return {resource: {label: np.zeros(hours) for label in TYPES[use]} for resource in ('cpu', 'gpu')}


def read_task_resources(args, use, clocks, metadata, inputs):
    hours = len(clocks)
    values = empty_resources(use, hours)
    if use == 'INF':
        path = args.project / 'results/job-flexibility-constraints-20261030/asi-hourly-type-priority.csv'
        data = pd.read_csv(path)
        labels = {'training': '训练', 'online_inference': '在线推理',
                  'offline_inference': '离线推理', 'dev': '开发'}
        data['label'] = data.job_type_public.map(labels).fillna('其他/未知')
        for label, group in data.groupby('label'):
            values['gpu'][label] = group.groupby('hour').used_gpu_hour_fraction.sum().reindex(range(24), fill_value=0).to_numpy()
            values['cpu'][label] = .25 * values['gpu'][label]
        inputs.append({'role': use + '_hourly_types', 'sha256': sha256(path), 'file': str(path)})
        return values
    edges = pd.date_range(clocks[0], periods=hours + 1, freq='h')
    if use == 'HPC':
        path = args.input_project / 'data/processed/task-trace-audit-20261026/lisa/lisa-tasks.parquet'
        columns = ['start_date', 'end_date', 'nodetypes', 'numcores', 'hardware_group']
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32768, columns=columns, use_threads=False):
            frame = batch.to_pandas()
            frame = frame[frame.hardware_group.eq('cpu_partition')].copy()
            frame['label'] = np.select([frame.nodetypes.str.contains('jupyter'), frame.nodetypes.str.contains('course'),
                frame.nodetypes.isin(['normal', 'shared', 'fat_soil_shared', 'sw', 'short', 'fat', 'shared_52c_384g'])],
                ['Jupyter', '课程分区', 'CPU 科研分区'], default='其他分区')
            accumulate(frame, 'label', 'start_date', 'end_date', 'numcores', edges, values['cpu'])
        denominator = metadata['lisa_cpu']['reference_resource_count']
        for label in TYPES[use]:
            values['cpu'][label] /= denominator
    elif use == 'SAI':
        path = args.mit_root / 'reference/slurm-log.csv'
        columns = ['time_start', 'time_end', 'time_suspended', 'tres_req', 'tres_alloc', 'job_type']
        labels = {'LLSUB:BATCH': '批处理', 'LLMAPREDUCE:MAP': 'MapReduce', 'LLSUB:INTERACTIVE': '交互式', 'OTHER': 'OTHER'}
        for frame in pd.read_csv(path, usecols=columns, chunksize=20000, dtype={'tres_req': str, 'tres_alloc': str}):
            requested = sum(tres_count(frame.tres_req, key).fillna(0) for key in (1001, 1002))
            frame['gpu'] = sum(tres_count(frame.tres_alloc, key).fillna(0) for key in (1001, 1002))
            frame['cpu'] = tres_count(frame.tres_alloc, 1).fillna(0)
            valid = requested.gt(0) & frame.gpu.gt(0) & frame.time_start.gt(0) & frame.time_end.ge(frame.time_start) & frame.time_suspended.eq(0)
            frame = frame[valid].copy()
            frame['start'] = pd.to_datetime(frame.time_start, unit='s')
            frame['end'] = pd.to_datetime(frame.time_end, unit='s')
            frame['label'] = frame.job_type.map(labels)
            if frame.label.isna().any():
                raise ValueError('MIT 出现未识别的提交方式')
            for resource in ('cpu', 'gpu'):
                accumulate(frame, 'label', 'start', 'end', resource, edges, values[resource])
        for resource in ('cpu', 'gpu'):
            denominator = metadata['mit_' + resource]['normalization_denominator']
            for label in TYPES[use]:
                values[resource][label] /= denominator
    elif use == 'TRN':
        path = args.input_project / 'data/raw/task-trace-audit-20261026/trace_kalos.csv'
        frame = load_acme(path)
        frame = frame[frame.interval_valid].copy()
        labels = {'Pretrain': '预训练', 'SFT': 'SFT', 'Eval': '评估', 'Debug': '调试', 'Other': '其他', 'CPU': '其他'}
        frame['label'] = frame.type.map(labels)
        if frame.label.isna().any():
            raise ValueError('Acme 出现未识别的任务类型')
        for resource in ('cpu', 'gpu'):
            selected = frame[frame.hardware.eq(resource)]
            accumulate(selected, 'label', 'start_time', 'end_time', resource + '_num', edges, values[resource])
            denominator = metadata['acme_' + resource]['reference_resource_count']
            for label in TYPES[use]:
                values[resource][label] /= denominator
        # GPU 作业宿主 CPU 沿用 v2 的 max(独立CPU需求, 0.25×GPU需求) 口径。
        cpu = sum(values['cpu'].values())
        gpu = sum(values['gpu'].values())
        total = np.maximum(cpu, .25 * gpu)
        for label in TYPES[use]:
            host_share = np.divide(values['gpu'][label], gpu, out=np.zeros(hours), where=gpu > 0)
            independent_share = np.divide(values['cpu'][label], cpu, out=np.zeros(hours), where=cpu > 0)
            values['cpu'][label] = total * np.where(.25 * gpu >= cpu, host_share, independent_share)
    else:
        raise ValueError(use)
    inputs.append({'role': use + '_jobs_local_only', 'sha256': sha256(path), 'file': str(path)})
    return values


def choose_day(frame, clocks, use, output):
    count = len(clocks) // 24
    daily = frame.facility_mw.to_numpy()[:count * 24].reshape(count, 24)
    energy = daily.sum(axis=1)
    median = float(np.median(energy))
    distance = np.abs(energy - median)
    rmse = np.sqrt(np.mean((daily - daily.mean(axis=0)) ** 2, axis=1))
    tied = np.flatnonzero(np.isclose(distance, distance.min(), rtol=0, atol=1e-9))
    chosen = int(tied[np.argmin(rmse[tied])])
    table = pd.DataFrame({'source_day': [str(clocks[i * 24]) for i in range(count)],
        'day_index': np.arange(count), 'facility_mwh': energy, 'median_mwh': median,
        'absolute_distance_mwh': distance, 'rmse_mw': rmse, 'selected': np.arange(count) == chosen})
    table.to_csv(output / (use + '-day-selection.csv'), index=False, encoding='utf-8-sig')
    return chosen, {'complete_days': count, 'reference_specification': '纯 CPU 型' if use in ('HPC', 'CLD') else 'GPU 主导型',
        'median_daily_facility_mwh': median, 'selected_distance_mwh': float(distance[chosen]),
        'selected_rmse_mw': float(rmse[chosen]), 'selection_rule': RULE if count > 1 else '唯一完整观测日；单日演示'}


def split_it(frame, values, config, ratios):
    cpu, gpu = frame.u_cpu_proxy.to_numpy(), frame.allocated_gpu_fraction.to_numpy()
    method = 'cpu_spec' if config['configuration_id'] == 'C20' else 'ratio_p50'
    units = int(10e6 // reference_power(config))
    zero = np.zeros(len(frame))
    idle = power_v2(config, zero, zero, method, ratios)
    power = power_v2(config, cpu, gpu, method, ratios)
    layers = {'基础与空载 IT': idle['unit_w'] * units / 1e6}
    for label in next(iter(values.values())):
        cpu_share = np.divide(values['cpu'][label], cpu, out=zero.copy(), where=cpu > 0)
        gpu_share = np.divide(values['gpu'][label], gpu, out=zero.copy(), where=gpu > 0)
        layers[label] = ((power['cpu_w'] - idle['cpu_w']) * cpu_share
            + (power['gpu_w'] - idle['gpu_w']) * gpu_share) * units / 1e6
    np.testing.assert_allclose(sum(layers.values()), frame.it_mw, rtol=1e-9, atol=1e-8)
    layers['冷却等非 IT'] = (frame.facility_mw - frame.it_mw).to_numpy()
    if min(float(value.min()) for value in layers.values()) < -1e-8:
        raise AssertionError('分层功率出现负值')
    return layers


def write_readme(output, records, selections):
    lines = ['# 各情境典型日 24 小时负荷曲线', '',
        '参考 IT 10 MW，11 个组合；总览按需求分行、服务器规格分列，全部图共用纵轴。', '',
        '![典型日总览](overview.png)', '', '## 日期与选取依据', '', RULE, '',
        '日历日期和星期使用源时间；源时间与气象模型日期分开记录。选日评分使用设施功率，气象取既有 v2 全年曲线首次重放该源日期时对应的参考天气。', '',
        '| 需求与数据源 | 完整日 | 典型日 | 星期 | 选取依据 |', '| --- | ---: | --- | --- | --- |']
    for use in ('HPC', 'CLD', 'SAI', 'TRN', 'INF'):
        selected = selections[use]
        if selected['status'] != 'computed':
            lines.append('| 云通用计算 / Alibaba | 0 | 待数据 | 未知 | 无完整小时数据 |')
            continue
        reason = '唯一完整日；单日演示' if use == 'INF' else '日电量接近中位数；并列时 RMSE 最小'
        lines.append(f"| {USES[use][1]} / {USES[use][0]} | {selected['complete_days']} | {selected['source_date']} | {selected['weekday']} | {reason} |")
    lines += ['', '## 典型日峰谷与任务构成', '',
        '峰谷为逐小时设施平均功率。构成为所选日资源小时占比，独立于基础与空载 IT，不能解释为可释放功率占比。', '',
        '| 情境 | 峰值 MW | 谷值 MW | 日电量 MWh | 日电量 / 全年日均 | 主要任务构成 |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for row in records:
        composition = '；'.join(f'{key} {value:.2%}' for key, value in sorted(row['composition'].items(), key=lambda item: -item[1]) if value >= .001)
        lines.append(f"| {row['scenario_name']} | {row['peak_mw']:.3f} | {row['minimum_mw']:.3f} | {row['daily_facility_mwh']:.3f} | {row['daily_to_annual_mean_ratio']:.3f} | {composition} |")
    lines += ['', '## 分层与功率口径', '',
        '- Lisa 仅使用 CPU 分区；科研分区合并 normal、shared、fat、short 等，Jupyter 单列。本日 CPU 课程分区为零；GPU 课程分区不进入纯 CPU 情境。',
        '- MIT 沿用 v2 的 GPU 请求、有效执行、无暂停作业筛选；分配 GPU 和同批宿主 CPU 分别按原观测小时峰值归一化。BATCH、MapReduce、INTERACTIVE、OTHER 仅代表提交方式。',
        '- Acme 使用 start_time 至 end_time 的有效区间，GPU 与独立 CPU 分别累计；宿主 CPU 需求沿用 max(独立 CPU 需求, 0.25 × GPU 需求)。当宿主代理较大时按 GPU 任务份额分配 CPU 增量，否则按独立 CPU 任务份额分配。',
        '- ASI 直接累加第 109 天各小时、类型与优先级的 GPU 小时/库存份额；开发以外的零散类型合并为其他/未知。未从小时出现时长推断作业时长。',
        '- Alibaba 采用相对第 2–6 天的五个完整日，最低核心时间覆盖率约 98.19%；CPU 利用率按有效核心秒加权。缺少在线容器监测和逐实例执行记录，业务未分解，不能按请求配置包络拆分实际电耗。',
        '- 功率调用既有 typical_power_models_v2.power_v2：CPU_SPEC 或 GPU ratio_p50，冷却调用 CoolingModel 风冷中央参数。CPU 非线性响应在总需求上计算，再按小时各类型 CPU 资源占比分配增量；GPU 增量按 GPU 资源占比分配。基础与空载 IT 使用零需求功率，单独列示。',
        '- 冷却等非 IT = 设施功率 − IT 功率，含冷却、UPS 配电损耗及照明控制。各任务层仅贡献其 IT 部分，未将非 IT 任意分摊给任务。',
        '- 10 MW 为参考满载 IT 预算，整机台数向下取整；实际安装参考容量略低于 10 MW。1/100/500 MW 按相应台数比缩放，形状和 PUE 不变。', '',
        '## 文件与来源', '',
        '- `overview.png/svg`：11 情境总览；`A-HPC.png/svg` 等为单图，PNG 300 dpi，SVG 中文字形转路径。',
        '- `*-hourly.csv`：每情境 24 行，记录源日期、星期、气象模型 UTC 日期、IT/设施功率及 PUE。',
        '- `*-by-task.csv`：每小时按任务层列示 IT 功率与对设施功率的贡献；基础与空载 IT、冷却等非 IT 单列。按小时相加分别等于 hourly 表的 IT 和设施功率。功率单位 MW，1 小时时段求和得到 MWh。',
        '- `*-day-selection.csv`：所有候选日的日电量、中位数距离和 RMSE；`summary.csv/json`：峰谷、任务构成和全年核对；`manifest.json`：输入与脚本 SHA-256。',
        '- 任务分类：[PR #26 汇总](../job-flexibility-constraints-20261030/README.md)；功率、PUE 与全年基准：[v2 结果](../typical-load-curves-20261029-v2/README.md)。情境名称遵循 PR #27 与 2026-10-31 移出整柜液冷型的决定。',
        '- Alibaba 输入来自同机 alibaba-cpu-cld-20261031 工作树的 complete-window.csv 与 source-audit.json，SHA-256 及本地位置见 manifest。其全年电量由本脚本在相同气象和 v2 模型下新增计算，旧全年结果没有 Alibaba。',
        '- MIT 数据引用：Samsi et al., The MIT Supercloud Dataset (HPEC 2021), DOI 10.1109/HPEC49654.2021.9622850，许可 CC-BY-NC-ND-4.0；本目录仅公开小时聚合，逐作业记录、标识与分层资源中间表全部留在本地忽略目录。', '',
        '## 不确定性与验证', '',
        '资源分配不是活跃利用率；CPU 响应、跨 GPU 代际和用途的功率比例以及参考冷却均含假设，未建立吞吐等价或原设施实测负荷。ASI 仅单日演示。Alibaba 与 ASI 没有公开可恢复的日历日期和星期；Lisa/MIT 当地时区也未核实。', '',
        '气象沿用 v2 的 g05_r100_c592、2025 年参考网格；源日日期与天气日期的对应是重放假设，不能解释为原设施同期天气。全年日均用于数量级核对，不要求典型日电量等于全年日均。', '',
        '数值检查见 validation.json 与 independent-validation.json；Word 原生导出 PDF 后逐页检查的记录见 visual-review.json。', '',
        '## 重新生成', '',
        '在仓库根目录运行（路径指向本机经审计输入，运行时限制线程数）：', '',
        '```powershell', "$env:OPENBLAS_NUM_THREADS='1'", "$env:OMP_NUM_THREADS='1'",
        'python dc-flexibility-project/analysis/build_scenario_typical_days.py --input-project <原始输入项目目录> --mit-root <MIT本地根目录> --alibaba-dir <Alibaba汇总目录>',
        'python dc-flexibility-project/analysis/verify_scenario_typical_days.py --input-project <原始输入项目目录> --mit-root <MIT本地根目录>',
        '```', '',
        '脚本复用 v2 manifest 登记的本地全年缓存，也可用 --baseline-processed 指定相同 SHA-256 的缓存目录。Lisa 以 32768 行批次读取，MIT 以 20000 行批次读取；不打开 MIT 数据库或 current 目录。', '',
        '简表重新生成命令见 [简表 README](../../docs/job-flexibility-brief/README.md#重新生成)。', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def build(args):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    output = args.project / 'results' / RUN
    output.mkdir(parents=True, exist_ok=True)
    local = args.project / 'data/processed' / RUN
    local.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((args.project / 'results' / BASELINE / 'manifest.json').read_text(encoding='utf-8'))
    cached = args.baseline_processed or Path(baseline['processed_directory'])
    configs = {c['configuration_id']: c for c in json.loads((args.project / 'results/server-count-scenarios-20261028-v2/server-configurations.json').read_text(encoding='utf-8'))}
    ratios = json.loads((args.project / 'results' / BASELINE / 'mit-power-ratio-summary.json').read_text(encoding='utf-8'))['transfer_distribution']
    sources = load_sources(args.input_project)
    metadata = {**{key: source['metadata'] for key, source in sources.items()},
        **{key: baseline['sources'][key] for key in ('mit_cpu', 'mit_gpu')}}
    annual_summary = pd.read_csv(args.project / 'results' / BASELINE / 'annual-summary.csv')
    frames, tasks, selections, records, checks, inputs = {}, {}, {}, [], {}, []
    source_keys = {'HPC': 'lisa_cpu', 'TRN': 'acme_gpu', 'INF': 'asi_gpu'}
    for use in ('HPC', 'SAI', 'TRN', 'INF', 'CLD'):
        demand = USES[use][0]
        specs = ('A',) if use in ('HPC', 'CLD') else ('B', 'C', 'D')
        if use == 'CLD':
            path = args.alibaba_dir / 'complete-window.csv' if args.alibaba_dir else None
            if path is None or not path.exists():
                selections[use] = {'status': '待数据', 'source_date': '待数据', 'weekday': '未知'}
                continue
            data = pd.read_csv(path)
            audit = json.loads((args.alibaba_dir / 'source-audit.json').read_text(encoding='utf-8'))
            if sha256(path) != audit['outputs_sha256'][path.name]:
                raise ValueError('Alibaba 完整小时文件与获取审计不符')
            if not data.coverage_pass.all() or len(data) % 24 or not np.all(np.diff(data.relative_hour) == 1):
                raise ValueError('Alibaba 完整日覆盖或连续性不足')
            inputs.append({'role': 'Alibaba_complete_hourly', 'file': str(path), 'sha256': sha256(path),
                'minimum_core_coverage': float(data.coverage_fraction.min())})
            clocks = [f'相对第{int(day)}天' for day in data.relative_day]
            fraction = data.resource_fraction.to_numpy()
            config = configs[SPECS['A'][0]]
            units = int(10e6 // reference_power(config))
            weather = frames[('A', 'HPC')]
            cpu = np.resize(fraction, 8760)
            power = power_v2(config, cpu, np.zeros(8760), 'cpu_spec', ratios)
            curve = CoolingModel().calculate(weather, power['unit_w'] * units / 1e6, units * reference_power(config) / 1e6)
            curve['u_cpu_proxy'], curve['allocated_gpu_fraction'] = cpu, 0.
            frames[('A', use)] = curve
            tasks[use] = empty_resources(use, len(clocks))
            tasks[use]['cpu']['业务未分解'] = fraction
        else:
            if use == 'SAI':
                source_path = Path(baseline['sources']['mit_gpu']['input_file'])
                if sha256(source_path) != baseline['sources']['mit_gpu']['input_sha256']:
                    raise ValueError('MIT 基准小时文件发生变化')
                clocks = pd.read_csv(source_path, parse_dates=['source_time']).source_time.tolist()
            elif use == 'INF':
                clocks = ['相对第109天'] * 24
            else:
                clocks = pd.to_datetime(sources[source_keys[use]]['clock']).tolist()
            for spec in specs:
                key = SPECS[spec][0]
                method = 'cpu_spec' if spec == 'A' else 'ratio_p50'
                path = cached / f'{key}-{demand}-{method}-central.parquet'
                if sha256(path) != baseline['processed_sha256'][path.name]:
                    raise ValueError('v2 缓存曲线与 manifest 不符：' + str(path))
                frames[(spec, use)] = pd.read_parquet(path)
                inputs.append({'role': spec + '-' + use + '_v2_curve', 'file': str(path), 'sha256': sha256(path)})
            tasks[use] = read_task_resources(args, use, clocks, metadata, inputs)
        reference = frames[(specs[-1], use)]
        for resource, column in (('cpu', 'u_cpu_proxy'), ('gpu', 'allocated_gpu_fraction')):
            np.testing.assert_allclose(sum(tasks[use][resource].values()), reference[column].to_numpy()[:len(clocks)], rtol=1e-8, atol=1e-7)
            checks[use + '_' + resource + '_hourly_reconciles_v2'] = True
        pd.DataFrame({resource + ':' + label: values for resource, labels in tasks[use].items()
            for label, values in labels.items()}).to_parquet(local / (use + '-hourly-resource-fractions.parquet'), index=False)
        day, selection = choose_day(reference, clocks, use, output)
        start = day * 24
        source = clocks[start]
        relative = use in ('CLD', 'INF')
        source_date = str(source) if relative else pd.Timestamp(source).strftime('%Y-%m-%d')
        weekday = '未知' if relative else '星期' + '一二三四五六日'[pd.Timestamp(source).dayofweek]
        selections[use] = {**selection, 'status': 'computed', 'source_date': source_date, 'weekday': weekday,
            'source_clock': '源相对小时；日历日期与星期未公开' if relative else ('UTC' if use == 'TRN' else '源时间；真实当地时区未核实'),
            'model_weather_date': str(reference.time_utc.iloc[start].date()), 'day_index': day}
        selected_tasks = {resource: {label: value[start:start + 24] for label, value in labels.items()}
            for resource, labels in tasks[use].items()}
        primary_resource = 'cpu' if use in ('HPC', 'CLD') else 'gpu'
        composition = {label: float(values.sum()) for label, values in selected_tasks[primary_resource].items()}
        total = sum(composition.values())
        composition = {label: value / total for label, value in composition.items()} if total > 0 else composition
        for spec in specs:
            curve = frames[(spec, use)]
            selected = curve.iloc[start:start + 24].reset_index(drop=True)
            config = configs[SPECS[spec][0]]
            layers = split_it(selected, selected_tasks, config, ratios)
            name = SPECS[spec][1] + '·' + USES[use][1]
            scenario = spec + '-' + use
            hourly = pd.DataFrame({'scenario': scenario, 'scenario_name': name, 'source_date': source_date,
                'weekday': weekday, 'hour': np.arange(24), 'model_time_utc': selected.time_utc,
                'reference_it_mw': 10., 'installed_reference_it_mw': selected.installed_reference_it_mw,
                'it_mw': selected.it_mw, 'facility_mw': selected.facility_mw, 'pue': selected.pue})
            hourly.to_csv(output / (scenario + '-hourly.csv'), index=False, encoding='utf-8-sig')
            components = []
            for label, values in layers.items():
                components.append(pd.DataFrame({'scenario': scenario, 'scenario_name': name, 'source_date': source_date,
                    'weekday': weekday, 'hour': np.arange(24), 'task_type': label,
                    'it_mw': np.zeros(24) if label == '冷却等非 IT' else values, 'facility_mw': values,
                    'component_scope': 'non_IT' if label == '冷却等非 IT' else 'IT'}))
            pd.concat(components).to_csv(output / (scenario + '-by-task.csv'), index=False, encoding='utf-8-sig')
            expected = annual_summary[annual_summary.configuration_id.eq(SPECS[spec][0]) & annual_summary.demand.eq(demand) & annual_summary.reference_it_mw.eq(10)]
            if use != 'CLD':
                if len(expected) != 1:
                    raise AssertionError('全年汇总匹配不唯一')
                np.testing.assert_allclose(curve.facility_mw.sum(), expected.annual_facility_mwh.iloc[0], rtol=1e-10)
            annual_mean = float(curve.facility_mw.sum() / 365)
            daily = float(selected.facility_mw.sum())
            ratio = daily / annual_mean
            checks[scenario + '_daily_annual_same_order'] = .5 < ratio < 2
            checks[scenario + '_component_balance'] = bool(np.allclose(sum(layers.values()), selected.facility_mw, atol=1e-8))
            records.append({'scenario': scenario, 'scenario_name': name, 'source_date': source_date, 'weekday': weekday,
                'complete_days': len(clocks) // 24, 'peak_mw': float(selected.facility_mw.max()),
                'minimum_mw': float(selected.facility_mw.min()), 'daily_it_mwh': float(selected.it_mw.sum()),
                'daily_facility_mwh': daily, 'daily_pue': daily / float(selected.it_mw.sum()),
                'annual_facility_mwh': float(curve.facility_mw.sum()), 'annual_daily_mean_mwh': annual_mean,
                'daily_to_annual_mean_ratio': ratio, 'composition': composition,
                'composition_basis': '核心小时' if primary_resource == 'cpu' else 'GPU 小时',
                'model_weather_date': selections[use]['model_weather_date'],
                'baseline_it_mwh': float(layers['基础与空载 IT'].sum()),
                'annual_comparison': '复用v2全年结果' if use != 'CLD' else '新增CPU小时输入按相同v2功率和冷却模型重放全年'})
        print(json.dumps({'completed': use, 'selected': source_date, 'weekday': weekday}, ensure_ascii=False), flush=True)
    if not all(checks.values()):
        raise AssertionError([key for key, value in checks.items() if not value])
    save_json(output / 'summary.json', {'selection_rule': RULE, 'selections': selections, 'scenarios': records})
    pd.DataFrame([{**row, 'composition': json.dumps(row['composition'], ensure_ascii=False)} for row in records]).to_csv(output / 'summary.csv', index=False, encoding='utf-8-sig')
    write_readme(output, records, selections)
    from plot_scenario_typical_days import plot_all
    plots = plot_all(output)
    checks.update(plots)
    if not all(checks.values()):
        raise AssertionError([key for key, value in checks.items() if not value])
    save_json(output / 'validation.json', {'status': 'numerical_passed', 'checks': checks,
        'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'scenario_count': len(records)})
    save_json(output / 'manifest.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'baseline_manifest_sha256': sha256(args.project / 'results' / BASELINE / 'manifest.json'),
        'input_files': inputs, 'selection_rule': RULE, 'reference_it_mw': 10,
        'sizing': '沿用v2参考满载功率预算与整机台数向下取整；其他规模按台数比缩放，形状不变',
        'task_power': '非线性CPU增量按该小时各类CPU资源份额分摊；GPU增量按GPU资源份额分摊；整机基础与空载独立列示',
        'facility_component_csv': '各任务facility_mw表示其对设施总功率的IT贡献；冷却等非IT单列，所有层相加等于设施总功率；未将非IT任意分配给任务',
        'weather': '沿用v2网格g05_r100_c592的2025年气象及首次重放日期，源日期与气象模型日期分别列示；并非原设施当地天气',
        'MIT_scope': 'CC-BY-NC-ND-4.0；仅发布典型日小时聚合，不发布逐作业记录或标识；只读reference文件，未接触流式进程、数据库及current目录',
        'uncertainty': '资源份额、CPU响应、GPU跨代功率比例及冷却均含假设；未建立吞吐等价或目标设施实测负荷；ASI单日演示；Alibaba业务实际占用未分解',
        'code_sha256': {name: sha256(args.project / 'analysis' / name) for name in
            ('build_scenario_typical_days.py', 'plot_scenario_typical_days.py', 'typical_power_models.py', 'typical_power_models_v2.py', 'compute_load/cooling.py')}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--input-project', type=Path, required=True)
    parser.add_argument('--baseline-processed', type=Path)
    parser.add_argument('--mit-root', type=Path, required=True)
    parser.add_argument('--alibaba-dir', type=Path)
    build(parser.parse_args())
