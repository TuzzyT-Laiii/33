import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from build_inf_diurnal import calculate as inference_power
from build_scenario_typical_days import choose_day, save_json
from build_trn_power_model import calculate as training_power
from build_typical_load_curves_v3 import checks, summarize
from compute_load.cooling import CoolingModel, WEATHER_COLUMNS
from compute_load.model import sha256
from load_curve_scenarios_v3 import catalogue, DAYS, TYPES, USES
from trn_power_models import power, idle_power
from typical_power_models import PARAMETERS


RUN = 'scenario-typical-day-by-task-20261101-v4'
PREVIOUS = 'scenario-typical-day-by-task-20261101-v3'
BASE = 'typical-load-curves-20261101-v3'
TRAINING = 'trn-power-model-20261101'
INFERENCE = 'inf-diurnal-20261101'
COMMITS = {'35': '520f68d57c105d255d867d13e1f7498aa59c1ded',
    '36': '6c83cf11e601bd9babf8d0075928e4bd207503ea', '28': 'd3fd452e6e03b2cd00b74352b91dbdd57f437095',
    '34': 'ab07a41157442e837659535e1a33579ced5af8fa'}


def combine_inference(request, cpu, config, key, mit, idle_case='central'):
    frame = request.copy()
    frame['base_it_mw'] = cpu.base_it_mw
    frame['cpu_it_mw'] = cpu.cpu_it_mw
    gpu = request.gpu_it_mw.to_numpy().copy()
    if idle_case != 'central':
        central_idle = 70. if key == 'D' else 50.
        alternative = PARAMETERS[config['configuration_id']][idle_case][3]
        gpu += config['gpu_devices_per_unit'] * int(frame.unit_count.iloc[0]) / 1e6 * (1 - frame.allocated_gpu_fraction) * (alternative - central_idle)
    frame['gpu_it_mw'] = gpu
    total = frame.base_it_mw + frame.cpu_it_mw + frame.gpu_it_mw
    weather = frame[[column for column in WEATHER_COLUMNS if column in frame]].copy()
    if frame.pressure_assumed.all():
        weather = weather.drop(columns='pressure_pa')
    thermal = CoolingModel().calculate(weather, total, float(frame.installed_reference_it_mw.iloc[0]))
    for column in thermal:
        frame[column] = thermal[column]
    frame['idle_it_mw'] = idle_power(config, key, 'INF', mit, idle_case) * frame.unit_count.iloc[0] / 1e6
    frame['task_it_mw'] = frame.it_mw - frame.idle_it_mw
    frame['power_model_version'] = RUN
    checks(frame)
    return frame


def split_layers(frame, resource, config, key, use, ratios, mit, shape):
    zero = np.zeros(len(frame))
    idle = power(config, key, use, zero, zero, ratios, mit, shape)
    units = int(frame.unit_count.iloc[0])
    layers = {'基础与空载 IT': idle['unit_w'] * units / 1e6}
    labels = [column[4:] for column in resource.columns if column.startswith('cpu:')]
    for label in labels:
        cpu_share = np.divide(resource['cpu:' + label], frame.u_cpu_proxy, out=zero.copy(), where=frame.u_cpu_proxy.to_numpy() > 0)
        cpu = (frame.cpu_it_mw - idle['cpu_w'] * units / 1e6) * cpu_share
        if use == 'INF':
            cards = config['gpu_devices_per_unit'] * units
            idle_each = 70. if key == 'D' else 50.
            observed = frame.online_gpu_it_mw if label == '在线推理' else frame.offline_gpu_it_mw
            gpu = observed - cards / 1e6 * resource['gpu:' + label] * idle_each
        else:
            share = np.divide(resource['gpu:' + label], frame.allocated_gpu_fraction, out=zero.copy(), where=frame.allocated_gpu_fraction.to_numpy() > 0)
            gpu = (frame.gpu_it_mw - idle['gpu_w'] * units / 1e6) * share
        layers[label] = np.asarray(cpu + gpu)
    np.testing.assert_allclose(sum(layers.values()), frame.it_mw, atol=1e-8, rtol=1e-10)
    layers['冷却等非 IT'] = (frame.facility_mw - frame.it_mw).to_numpy()
    assert min(float(layer.min()) for layer in layers.values()) >= -1e-8
    return layers


def write_report(output, summary):
    lines = ['# 各情境分任务典型日负荷曲线：组合模型', '', '![典型日总览](overview.png)', '',
        '参考IT 10MW，沿用冻结的设备台数与参考天气。功率超过或低于参考容量均保留，不按10MW截断。其他规模按整数设备台数比缩放。', '',
        '完整日先最小化日电量与中位数的距离，并列时最小化与全部日平均24小时曲线的RMSE。科研AI、训练、推理分别以GPU主导型选日，三规格共用日期与资源构成。源日期与2025年天气重放日分开记录。', '',
        '## 组合口径与来源', '',
        '- #36（6c83cf1）：全部情境采用三份SPECpower归一化后拟合的单调CPU响应；八个非推理情境逐小时功率直接采用该PR结果。训练GPU/TDP中央0.80，敏感性0.65–0.95，700W H100对应560W、455–665W。分配轨迹保持原样，未添加固定小时训练周期。',
        '- #35（520f68d）：推理在线GPU叠加BurstGPT 119日请求的输入/输出token各50%日内形状，离线推理保持原分配与固定每卡功率。在线能量按ASI分配加权标定至原年度GPU电量；CPU及基础部分采用#36，随后完整重算冷却和PUE。未使用#36冻结的推理GPU表覆盖#35。',
        '- 两任务中央基础/空闲值一致，分别保留其已分配在线常驻空闲与未分配GPU空闲定义。推理在线SXM端点中央120/525W，PCIe60/262.5W；低到高波动敏感性为SXM空闲100–160W、高负载350–700W、曲率0.7–1.3，PCIe按TDP比例迁移。',
        '- 基础与空闲独立敏感性：纯CPU型120–320W/台，CPU主导型195–500W/台，均衡型270–680W/台，GPU主导型1370–4110W/台；科研AI保持MIT的GPU空闲代理，只改变基础与CPU空闲。所有范围属于工程情境，均非置信区间。',
        '- MIT六月完成版与#28 OTHER互斥六层保持；任务层为超过无任务整机端点的IT增量，基础与空载IT、冷却等非IT分别列示。推理在线/离线GPU增量分别计算，不按总GPU份额分摊不同每卡功率。',
        '- Alibaba相对06时利用率高值来自原始观测；单份SPECpower 50%→60%陡升的放大效应已由#36三曲线平滑替换，双路CPU该区间动态增量由约125W降至41W。', '',
        '| 情境 | 源典型日 | 星期 | 设施峰值MW | 设施谷值MW | 日内最高/最低比 | 日电量MWh | 与新全年日均之比 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in summary['scenarios']:
        lines.append(f"| {row['scenario_name']} | {row['source_date']} | {row['weekday']} | {row['peak_mw']:.3f} | {row['minimum_mw']:.3f} | {row['peak_valley_ratio']:.3f} | {row['daily_facility_mwh']:.3f} | {row['daily_to_annual_mean_ratio']:.3f} |")
    lines += ['', '## 核对与解释边界', '',
        'comparison.csv对照上一版各自按同一规则选出的典型日，日期可能不同；matched-day-comparison.csv另对同一源日、同一天气小时比较功率。此处日内最高/最低比取所选日24个小时的最大/最小，区别于上游报告中全年按小时分组平均的比值。', '',
        'sensitivity-hourly.csv在中央模型选中日分别改变训练GPU比例、推理波动参数和基础空闲端点；sensitivity-summary.csv给出同日及全年量。敏感性不重新选日，各因素单独改变，不将独立端点叠加成置信范围。', '',
        'ASI与Alibaba真实日期和星期未知，BurstGPT的日历与时区未知；请求叠加采用#35规定的小时相位，属于合成映射。ASI索引126零资源异常沿用原值和质量标识。Acme缺少逐作业设备关联，不能给出各任务类型GPU实测功率；训练比例为用途级情境参数。', '',
        'MIT源为Samsi等The MIT Supercloud Dataset，DOI:10.1109/HPEC49654.2021.9622850，许可CC BY-NC-ND 4.0。仅使用本地已核验分类资源小时汇总；逐作业输入、标识和全年组合小时中间表只留本机忽略目录，未访问MIT流式进程或STOP。', '',
        '## 文件与复现', '',
        '每情境PNG（300dpi）/SVG、24行hourly.csv和分任务by-task.csv；总览、选日评分、summary、comparison、annual-summary、敏感性与输入/代码SHA-256一并保存。idle-endpoints.csv列出各用途无任务整机低、中、高端点。图例、颜色沿用上一版。', '',
        '```powershell', 'python -m pip install -r dc-flexibility-project/analysis/requirements-scenario-typical-days-v3.txt', "$env:OPENBLAS_NUM_THREADS='1'", "$env:OMP_NUM_THREADS='1'",
        'python dc-flexibility-project/analysis/build_scenario_typical_days_v4.py --training-results "D:/codex-runs/trn-power-utilization-20261101/worktree/dc-flexibility-project/results/trn-power-model-20261101"', '```', '',
        '需要已核验的v3本地分类资源小时汇总，以及#36清单对应的本地全年小时文件；不重新扫描原始逐作业数据。简表按docs/job-flexibility-brief的重新生成说明构建，Word转PDF逐页检查。随后运行独立核验：', '',
        '```powershell', 'python dc-flexibility-project/analysis/verify_scenario_typical_days_v4.py --pdf "<逐页检查后的PDF>" --visual-qa-confirmed', '```', '',
        'independent-verification.json记录CPU/GPU来源、功率公式、选日、分层、全年和敏感性核对，以及新选日OTHER的逐作业独立交集检查。manifest记录本地全年组合文件及文档和PDF指纹；人工检查标识只在实际逐页检查后设置。旧版结果均保持原样。', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def save_idle_endpoints(configs, mit, records, output):
    rows = []
    for row in records:
        key, use = row['scenario'].split('-')
        rows.append({'scenario': row['scenario'], 'scenario_name': row['scenario_name'],
            **{case + '_unit_w': idle_power(configs[key], key, use, mit, case) for case in ('low', 'central', 'high')}})
    pd.DataFrame(rows).to_csv(output / 'idle-endpoints.csv', index=False, encoding='utf-8-sig')


def build(args):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    project = args.project
    output, local = project / 'results' / RUN, project / 'data/processed' / RUN
    output.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    inputs = []

    def track(path, role):
        inputs.append({'role': role, 'file': str(path.resolve()), 'sha256': sha256(path)})
        return path

    def read_json(path, role):
        return json.loads(track(path, role).read_text(encoding='utf-8'))

    baseline = project / 'results' / BASE
    previous = project / 'results' / PREVIOUS
    upstream = project / 'results' / TRAINING
    requests = project / 'results' / INFERENCE
    previous_summary = read_json(previous / 'summary.json', 'previous_typical_day_summary')
    source_manifest = read_json(baseline / 'manifest.json', 'v3_manifest')
    training_manifest = read_json(upstream / 'manifest.json', 'PR36_manifest')
    request_manifest = read_json(requests / 'manifest.json', 'PR35_manifest')
    shape = read_json(upstream / 'cpu-shape.json', 'PR36_CPU_shape')
    mit = read_json(baseline / 'mit-calibration.json', 'MIT_calibration')
    ratios = read_json(project / 'results/typical-load-curves-20261029-v2/mit-power-ratio-summary.json', 'frozen_GPU_ratios')['transfer_distribution']
    clocks_manifest = read_json(project / 'results/typical-load-curves-20261029-v2/manifest.json', 'source_clocks')
    mapping = pd.read_csv(track(baseline / 'INF-annual-replay-map.csv', 'inference_source_map')).iloc[:4440]
    parameters = pd.read_csv(track(baseline / 'power-parameters.csv', 'GPU_parameters')).query('use == "INF"').set_index('server_type')
    track(requests / 'request-shape-24h.csv', 'request_diurnal_shape')
    configs = catalogue(project)
    annual36 = pd.read_csv(track(upstream / 'annual-summary.csv', 'PR36_annual_summary')).query('reference_it_mw == 10').set_index('scenario')
    annual35 = pd.read_csv(track(requests / 'annual-summary-all-locations.csv', 'PR35_annual_summary'))
    annual35 = annual35.query('reference_it_mw == 10 and version == "diurnal" and year == 2025 and location_id == "reference-grid"').set_index('scenario')
    assert annual35.index.is_unique
    records, selections, annual_rows, sensitivity_rows, sensitivity_hours, comparisons, matched, local_hashes, checks_done = [], {}, [], [], [], [], [], {}, {}
    before_by_id = {row['scenario']: row for row in previous_summary['scenarios']}
    for use in ('HPC', 'CLD', 'SAI', 'TRN', 'INF'):
        cache_path = track(project / 'data/processed' / PREVIOUS / (use + '-resources.parquet'), use + '_local_classified_resource_hours')
        resource = pd.read_parquet(cache_path)
        assert len(resource) == DAYS[use] * 24 and np.isfinite(resource).all().all() and resource.min().min() >= -1e-8
        if use in ('HPC', 'SAI', 'TRN'):
            source = {'HPC': 'lisa_cpu', 'SAI': 'mit_gpu', 'TRN': 'acme_gpu'}[use]
            clocks = pd.date_range(clocks_manifest['sources'][source]['source_start'], periods=len(resource), freq='h').tolist()
        else:
            clocks = [f'相对第{day + (2 if use == "CLD" else 0)}天' for day in range(DAYS[use]) for _ in range(24)]
        keys = ['A'] if use in ('HPC', 'CLD') else ['B', 'C', 'D']
        curves, old_curves, variants_by_key = {}, {}, {}
        for key in keys:
            scenario = key + '-' + use
            filename = scenario + '-10MW-2025.parquet'
            old_path = track(baseline / 'hourly' / filename, scenario + '_v3_hourly')
            assert sha256(old_path) == source_manifest['hourly_sha256'][old_path.relative_to(project).as_posix()]
            old = pd.read_parquet(old_path)
            path = track(args.training_results / 'hourly' / filename, scenario + '_PR36_hourly')
            assert sha256(path) == training_manifest['hourly_sha256'][filename]
            current = pd.read_parquet(path)
            for source, column in (('cpu:', 'u_cpu_proxy'), ('gpu:', 'allocated_gpu_fraction')):
                np.testing.assert_allclose(resource.filter(regex='^' + source).sum(axis=1), old[column].to_numpy()[:len(resource)], rtol=1e-8, atol=1e-7)
            for column in ('u_cpu_proxy', 'allocated_gpu_fraction', 'source_hour_index', 'unit_count', 'temperature_c', 'relative_humidity_pct'):
                np.testing.assert_array_equal(current[column], old[column])
            for column in ('it_mw', 'facility_mw'):
                np.testing.assert_allclose(current[column].sum(), annual36.loc[scenario, 'annual_' + column.replace('_mw', '_mwh')], rtol=1e-12)
            variants = {}
            if use == 'TRN':
                for case in ('low', 'high'):
                    variants['training_' + case] = training_power(old, configs[key], key, use, ratios, mit, shape, training_case=case)
            for case in ('low', 'high'):
                variants['idle_' + case] = training_power(old, configs[key], key, use, ratios, mit, shape, idle_case=case)
            if use == 'INF':
                request_path = track(requests / 'hourly' / filename, scenario + '_PR35_hourly')
                assert sha256(request_path) == request_manifest['hourly_sha256'][request_path.relative_to(project).as_posix()]
                request = pd.read_parquet(request_path)
                central = combine_inference(request, current, configs[key], key, mit)
                for column in ('gpu_it_mw', 'request_intensity', 'online_gpu_w_each', 'source_hour_index', 'source_quality_flag'):
                    np.testing.assert_array_equal(central[column], request[column])
                for column in ('base_it_mw', 'cpu_it_mw'):
                    np.testing.assert_array_equal(central[column], current[column])
                np.testing.assert_allclose(central.it_mw.sum(), current.it_mw.sum(), rtol=1e-12)
                np.testing.assert_allclose(request.it_mw.sum(), annual35.loc[scenario, 'annual_it_mwh'], rtol=1e-12)
                for case in ('low', 'high'):
                    request_variant, _ = inference_power(old, mapping, parameters.loc[key], request.request_intensity.to_numpy(), case=case)
                    variants['inference_' + case] = combine_inference(request_variant, current, configs[key], key, mit)
                    variants['idle_' + case] = combine_inference(request, variants['idle_' + case], configs[key], key, mit, idle_case=case)
                current = central
            checks(current)
            curves[key], old_curves[key] = current, old
            variants_by_key[key] = {'central': current, **variants}
            local_file = local / filename
            current.to_parquet(local_file, index=False)
            local_hashes[filename] = sha256(local_file)
            fields = {'scenario': scenario, 'scenario_name': TYPES[key] + '·' + USES[use], 'reference_it_mw': 10}
            annual_rows.append(summarize(current, fields))
            checks_done[scenario + '_source_hash_demand_components_annual_and_thermal_balance'] = True
        day, selection = choose_day(curves[keys[-1]], clocks, use, output)
        start = day * 24
        source_date = clocks[start] if use in ('CLD', 'INF') else pd.Timestamp(clocks[start]).strftime('%Y-%m-%d')
        weekday = '未知' if use in ('CLD', 'INF') else '星期' + '一二三四五六日'[pd.Timestamp(clocks[start]).dayofweek]
        selections[use] = {**selection, 'day_index': day, 'source_date': source_date, 'weekday': weekday,
            'model_weather_date': str(curves[keys[-1]].time_utc.iloc[start].date())}
        selected_resource = resource.iloc[start:start + 24].reset_index(drop=True)
        basis = 'cpu:' if use in ('HPC', 'CLD') else 'gpu:'
        totals = selected_resource.filter(regex='^' + basis).sum()
        composition = {column[4:]: float(value / totals.sum()) for column, value in totals.items()}
        if use == 'SAI':
            labels = [label for label in composition if label not in ('批处理', 'MapReduce', '交互式')]
            denominator = clocks_manifest['sources']['mit_gpu']['normalization_denominator']
            total_other = sum(totals[basis + label] for label in labels)
            pd.DataFrame([{'task_type': label, 'gpu_hours': totals[basis + label] * denominator,
                'share_of_other': totals[basis + label] / total_other, 'share_of_sai': composition[label]} for label in labels]).to_csv(output / 'mit-other-typical-day-shares.csv', index=False, encoding='utf-8-sig')
        for key in keys:
            scenario = key + '-' + use
            frame = curves[key].iloc[start:start + 24].reset_index(drop=True)
            layers = split_layers(frame, selected_resource, configs[key], key, use, ratios, mit, shape)
            base_fields = {'scenario': scenario, 'scenario_name': TYPES[key] + '·' + USES[use], 'source_date': source_date, 'weekday': weekday}
            hourly = pd.DataFrame({**base_fields, 'hour': range(24), 'model_time_utc': frame.time_utc,
                'reference_it_mw': 10., 'installed_reference_it_mw': frame.installed_reference_it_mw,
                'it_mw': frame.it_mw, 'facility_mw': frame.facility_mw, 'pue': frame.pue,
                'source_quality_flag': frame.source_quality_flag})
            hourly.to_csv(output / (scenario + '-hourly.csv'), index=False, encoding='utf-8-sig')
            parts = [pd.DataFrame({**base_fields, 'hour': range(24), 'task_type': label,
                'it_mw': np.zeros(24) if label == '冷却等非 IT' else values, 'facility_mw': values,
                'component_scope': 'non_IT' if label == '冷却等非 IT' else 'IT'}) for label, values in layers.items()]
            pd.concat(parts).to_csv(output / (scenario + '-by-task.csv'), index=False, encoding='utf-8-sig')
            energy = float(frame.facility_mw.sum())
            annual_energy = float(curves[key].facility_mw.sum())
            ratio = energy * 365 / annual_energy
            assert .5 < ratio < 2
            row = {**base_fields, 'complete_days': DAYS[use], 'peak_mw': float(frame.facility_mw.max()),
                'minimum_mw': float(frame.facility_mw.min()), 'peak_valley_ratio': float(frame.facility_mw.max() / frame.facility_mw.min()),
                'daily_it_mwh': float(frame.it_mw.sum()), 'daily_facility_mwh': energy, 'annual_facility_mwh': annual_energy,
                'daily_to_annual_mean_ratio': ratio, 'composition': composition, 'composition_basis': '核心小时' if basis == 'cpu:' else 'GPU小时',
                'task_share_of_daily_it': float((frame.it_mw - frame.idle_it_mw).sum() / frame.it_mw.sum()),
                'model_weather_date': selections[use]['model_weather_date'], 'unresolved_selected_hours': int(frame.source_unresolved.sum())}
            records.append(row)
            before = before_by_id[scenario]
            comparisons.append({'scenario': scenario, 'scenario_name': row['scenario_name'],
                'old_source_date': before['source_date'], 'new_source_date': source_date,
                'old_peak_mw': before['peak_mw'], 'new_peak_mw': row['peak_mw'],
                'old_minimum_mw': before['minimum_mw'], 'new_minimum_mw': row['minimum_mw'],
                'old_peak_valley_ratio': before['peak_mw'] / before['minimum_mw'], 'new_peak_valley_ratio': row['peak_valley_ratio']})
            old = old_curves[key].iloc[start:start + 24].reset_index(drop=True)
            matched.append(pd.DataFrame({**base_fields, 'hour': range(24), 'old_it_mw': old.it_mw,
                'new_it_mw': frame.it_mw, 'old_facility_mw': old.facility_mw, 'new_facility_mw': frame.facility_mw}))
            for case, full in variants_by_key[key].items():
                selected = full.iloc[start:start + 24]
                sensitivity_rows.append({'scenario': scenario, 'scenario_name': row['scenario_name'], 'case': case,
                    'source_date': source_date, 'annual_mean_it_mw': float(full.it_mw.mean()), 'annual_mean_facility_mw': float(full.facility_mw.mean()),
                    'daily_facility_mwh': float(selected.facility_mw.sum()), 'peak_mw': float(selected.facility_mw.max()),
                    'minimum_mw': float(selected.facility_mw.min()), 'peak_valley_ratio': float(selected.facility_mw.max() / selected.facility_mw.min())})
                sensitivity_hours.append(pd.DataFrame({**base_fields, 'case': case, 'hour': range(24),
                    'it_mw': selected.it_mw.to_numpy(), 'facility_mw': selected.facility_mw.to_numpy()}))
            checks_done[scenario + '_hourly_task_layer_balance_and_daily_energy_scale'] = True
        print(json.dumps({'completed': use, 'source_date': source_date}, ensure_ascii=False), flush=True)
    summary = {'data_version': 'PR35+PR36', 'commits': COMMITS, 'selections': selections, 'scenarios': records}
    save_json(output / 'summary.json', summary)
    pd.DataFrame([{**row, 'composition': json.dumps(row['composition'], ensure_ascii=False)} for row in records]).to_csv(output / 'summary.csv', index=False, encoding='utf-8-sig')
    for name, values in [('annual-summary', annual_rows), ('sensitivity-summary', sensitivity_rows), ('comparison', comparisons)]:
        pd.DataFrame(values).to_csv(output / (name + '.csv'), index=False, encoding='utf-8-sig')
    pd.concat(sensitivity_hours).to_csv(output / 'sensitivity-hourly.csv', index=False, encoding='utf-8-sig')
    pd.concat(matched).to_csv(output / 'matched-day-comparison.csv', index=False, encoding='utf-8-sig')
    facts = json.loads((previous / 'brief-facts.json').read_text(encoding='utf-8'))
    facts['power'] = {'TRN_GPU_TDP_ratio': .80, 'TRN_GPU_TDP_range': [.65, .95],
        'H100_SXM_training_GPU_kwh_per_card_hour': .560, 'H100_SXM_training_GPU_kwh_per_card_hour_range': [.455, .665],
        'inference_source': 'BurstGPT119日中央形状；Azure2024两类各7日为敏感性', 'commits': COMMITS}
    save_json(output / 'brief-facts.json', facts)
    write_report(output, summary)
    save_idle_endpoints(configs, mit, records, output)
    from plot_scenario_typical_days_v4 import plot_all
    checks_done.update(plot_all(output))
    assert all(checks_done.values())
    save_json(output / 'validation.json', {'status': 'passed', 'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'checks': checks_done})
    save_json(output / 'manifest.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'commits': COMMITS,
        'composition_rule': 'PR35_GPU_and_request_mapping + PR36_CPU_and_base_idle; full_cooling_recalculation',
        'input_files': inputs, 'local_annual_hourly_sha256': local_hashes,
        'code_sha256': {name: sha256(project / 'analysis' / name) for name in ('build_scenario_typical_days_v4.py', 'plot_scenario_typical_days_v4.py',
            'trn_power_models.py', 'build_trn_power_model.py', 'build_inf_diurnal.py', 'inf_diurnal_power.py', 'compute_load/cooling.py')},
        'privacy': '逐作业输入与全年组合中间表只留本机；公开24小时任务聚合与年度汇总'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--training-results', type=Path, required=True)
    build(parser.parse_args())
