import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from annual_task_profiles import load_sources, replay_hourly
from build_typical_load_curves import add_alibaba
from compute_load.cooling import CoolingModel
from compute_load.model import ResourceProfile, normalize_weather, sha256
from global_grid_compute_load import grid_weather
from typical_power_models import reference_power
from typical_power_models_v2 import GPU_IDLE_W, GPU_METHODS, GPU_TDP_W, parameters_v2, power_v2


RUN = 'typical-load-curves-20261029-v2'
GRID = 'g05_r100_c592'
DEMANDS = {'lisa': 'Lisa', 'mit-supercloud': 'MIT', 'acme': 'Acme', 'asi': 'ASI', 'alibaba': 'Alibaba'}
SOURCE_KEYS = {'Lisa': 'lisa_cpu', 'MIT': 'mit_gpu', 'Acme': 'acme_gpu', 'ASI': 'asi_gpu', 'Alibaba': 'alibaba_cpu'}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def progress(project, text):
    path = project.parent / '.codex-temp/progress-note.md'
    with path.open('a', encoding='utf-8') as handle:
        handle.write(f'\n- {datetime.now(timezone.utc).isoformat()}：{text}\n')


def load_mit(project, source_dir, times):
    audit_path = project / 'data/metadata/research-gpu-demand-20261028/source-audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    path = source_dir / 'research-gpu-observed-hourly.csv'
    if sha256(path) != audit['adopted']['observed_hourly_sha256']:
        raise ValueError('MIT 小时需求与已合并矩阵的来源 SHA-256 不符')
    frame = pd.read_csv(path, parse_dates=['source_time'])
    if len(frame) != 5688 or not frame.source_time.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError('MIT 输入必须为237个连续完整日')
    results, metadata, mappings = {}, {}, {}
    for key, column, hardware in [('mit_gpu', 'gpu_allocation_hours', 'gpu'), ('mit_cpu', 'host_cpu_allocation_hours', 'cpu')]:
        denominator = float(frame[column].max())
        fraction = frame[column].to_numpy(dtype=float) / denominator
        if denominator <= 0 or not np.isfinite(fraction).all() or (fraction < 0).any():
            raise ValueError('MIT 分配需求无效')
        source = {'profile': ResourceProfile(hardware, fraction, {'source': key,
            'capacity_basis': '同批GPU作业的GPU/宿主CPU小时分别除以本资源观测小时峰值；跨硬件份额形状代理'}),
            'clock': frame.source_time.astype(str).to_numpy(), 'metadata': {
                'source': key, 'days': 237, 'input_file': str(path), 'input_sha256': sha256(path),
                'normalization_denominator': denominator, 'resource_column': column,
                'source_start': str(frame.source_time.iloc[0]), 'source_end': str(frame.source_time.iloc[-1]),
                'status': 'allocation_shape_proxy; separate_CPU_GPU_normalization; performance_not_matched'}}
        profile, mapping = replay_hourly(source, times)
        results[key], metadata[key], mappings[key] = profile.fraction, profile.provenance, mapping
    return results, metadata, mappings


def resource_pair(demand, replay):
    if demand in ('Lisa', 'Alibaba'):
        return replay[SOURCE_KEYS[demand]], np.zeros(8760)
    if demand == 'MIT':
        return replay['mit_cpu'], replay['mit_gpu']
    gpu = replay[SOURCE_KEYS[demand]]
    host = .25 * gpu
    return (np.maximum(replay['acme_cpu'], host) if demand == 'Acme' else host), gpu


def representative_week(weather, replay):
    features = np.column_stack([replay[key] for key in ('lisa_cpu', 'mit_gpu', 'mit_cpu', 'acme_gpu', 'asi_gpu')]
        + [weather.temperature_c.to_numpy()])
    starts = [index for index, value in enumerate(weather.time_utc)
        if value.dayofweek == 0 and value.hour == 0 and index + 168 <= len(weather)]
    weeks = np.stack([features[start:start + 168] for start in starts])
    scale = np.maximum(features.std(axis=0), 1e-8)
    scores = (((weeks - weeks.mean(axis=0)) / scale) ** 2).mean(axis=(1, 2))
    selected = int(np.argmin(scores))
    return starts[selected], pd.DataFrame({'start_utc': [str(weather.time_utc.iloc[start]) for start in starts],
        'hour_index': starts, 'standardized_mse': scores})


def summarize(frame, config, demand, method, cooling_case, source_days):
    energy_it = float(frame.it_mw.sum())
    energy = float(frame.facility_mw.sum())
    trust = ['CPU及基础功率含假设', '冷却v2工程情境未在目标设施实测验证', '资源份额跨硬件重放']
    if demand == 'ASI':
        trust.append('单日重复演示')
    if demand == 'Acme':
        trust.extend(['90日循环不代表季节需求', 'MIT功率分布跨工作负载迁移假设'])
    if demand == 'MIT':
        trust.append('237日循环;GPU与宿主CPU分别按观测峰值归一化')
    if config['gpu_devices_per_unit']:
        trust.append('未分配GPU空载功率为假设或官方模板代理')
    if method.startswith('ratio_'):
        trust.append('V100功率比例跨代假设;1574文件快照尚未稳定')
    return {'curve_id': config['configuration_id'] + '-' + demand, 'configuration_id': config['configuration_id'],
        'demand': demand, 'method': method, 'cooling_case': cooling_case,
        'cooling_type': config['cooling_type'], 'reference_it_mw': 10., 'hours': len(frame), 'source_days': source_days,
        'unit': config['research_unit'], 'units': int(frame.unit_count.iloc[0]),
        'sizing_reference_w': reference_power(config),
        'installed_reference_it_mw': float(frame.installed_reference_it_mw.iloc[0]),
        'annual_it_mwh': energy_it, 'annual_facility_mwh': energy,
        'annual_cooling_mwh': float(frame.cooling_mw.sum()), 'annual_other_mwh': float(frame.other_mw.sum()),
        **{f'annual_{component}_mwh': float(frame[component + '_it_mw'].sum()) for component in ('base', 'cpu', 'gpu')},
        'peak_mw': float(frame.facility_mw.max()), 'it_peak_mw': float(frame.it_mw.max()),
        'minimum_mw': float(frame.facility_mw.min()), 'average_mw': energy / len(frame),
        'average_load_factor': float(frame.facility_mw.mean() / frame.facility_mw.max()),
        'peak_valley_ratio': float(frame.facility_mw.max() / frame.facility_mw.min()),
        'annual_pue': energy / energy_it, 'cooling_share': float(frame.cooling_mw.sum() / energy),
        'allocated_power_below_idle': bool(config['gpu_devices_per_unit'] and method.startswith('ratio_') and frame.allocated_gpu_w_each.iloc[0] < GPU_IDLE_W[config['configuration_id']]),
        'data_trust_flag': ';'.join(trust), 'status': 'computed'}


def comparison_table(project, central):
    old = pd.read_csv(project / 'results/typical-load-curves-20261028/annual-summary.csv')
    records = []
    for row in central.itertuples():
        old_demand = 'Lisa' if row.demand == 'MIT' else row.demand
        previous = old.loc[old.configuration_id.eq(row.configuration_id) & old.demand.eq(old_demand)]
        if len(previous) != 1:
            raise ValueError('v1 对照必须唯一')
        previous = previous.iloc[0]
        comparable = row.configuration_id != 'R-GB200'
        records.append({'configuration_id': row.configuration_id, 'demand_v1': old_demand, 'demand_v2': row.demand,
            'v1_reported_scope': previous.reported_scope, 'v1_reported_annual_mwh': previous.reported_annual_mwh,
            'v2_annual_facility_mwh': row.annual_facility_mwh,
            'v1_annual_it_mwh': previous.annual_it_mwh, 'v2_annual_it_mwh': row.annual_it_mwh,
            'v1_reported_peak_mw': previous.peak_mw, 'v2_facility_peak_mw': row.peak_mw,
            'v1_annual_pue': previous.annual_pue, 'v2_annual_pue': row.annual_pue,
            'facility_energy_change_pct': (row.annual_facility_mwh / previous.annual_facility_mwh - 1) * 100 if comparable else None,
            'facility_peak_change_pct': (row.peak_mw / previous.peak_mw - 1) * 100 if comparable else None,
            'it_energy_change_pct': (row.annual_it_mwh / previous.annual_it_mwh - 1) * 100,
            'comparison_scope': 'facility_to_facility' if comparable else 'v1_IT_only; facility_change_not_defined',
            'demand_changed': row.demand == 'MIT'})
    return pd.DataFrame(records)


def build(args):
    project, inputs = args.project.resolve(), args.input_project.resolve()
    output = project / 'results' / RUN
    processed = project / 'data/processed' / RUN
    output.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    ratio_path = output / 'mit-power-ratio-summary.json'
    ratio_summary = json.loads(ratio_path.read_text(encoding='utf-8'))
    if ratio_summary['source_gpu_tdp_w'] != 250 or ratio_summary['transfer_gpu_type'] != 'gpu:volta':
        raise ValueError('当前参数要求V100 PCIe 250W，需先核对源型号')
    ratios = ratio_summary['transfer_distribution']
    catalog_path = project / 'results/server-count-scenarios-20261028-v2/server-configurations.json'
    matrix_path = project / 'results/server-count-scenarios-20261028-v2/scenario-matrix.csv'
    configs = {c['configuration_id']: c for c in json.loads(catalog_path.read_text(encoding='utf-8'))}
    matrix = pd.read_csv(matrix_path)
    if len(matrix) != 52:
        raise ValueError('v2矩阵必须包含52个设计单元')
    mapping_path = inputs / 'data/processed/era5-landuse-selection/grid-weather-mapping.parquet'
    weather_path = inputs / f'data/processed/era5-landuse-trial/{GRID}-hourly.parquet'
    baseline_manifest = json.loads((project / 'results/typical-load-curves-20261028/manifest.json').read_text(encoding='utf-8'))
    for path, key in [(mapping_path, 'weather_mapping_sha256'), (weather_path, 'weather_file_sha256')]:
        if sha256(path) != baseline_manifest[key]:
            raise ValueError('气象输入与v1固定输入不一致: ' + str(path))
    mapping = pd.read_parquet(mapping_path)
    center = mapping.loc[mapping.grid_id.eq(GRID) & mapping.row_offset.eq(1) & mapping.col_offset.eq(1)]
    if len(center) != 1:
        raise ValueError('天气中心节点必须唯一')
    weather = normalize_weather(grid_weather(ds.dataset(weather_path, format='parquet'), center.iloc[0], [2025]))
    sources = load_sources(inputs)
    if args.alibaba_hourly:
        add_alibaba(sources, args.alibaba_hourly)
    replay, metadata, hashes = {}, {}, {}
    for key, source in sources.items():
        profile, provenance = replay_hourly(source, weather.time_utc)
        replay[key], metadata[key] = profile.fraction, profile.provenance
        path = processed / (key + '-replay-map.parquet')
        provenance.to_parquet(path, index=False)
        hashes[path.name] = sha256(path)
    mit_replay, mit_metadata, mit_mappings = load_mit(project, args.mit_source_dir, weather.time_utc)
    replay.update(mit_replay)
    metadata.update(mit_metadata)
    for key, frame in mit_mappings.items():
        path = processed / (key + '-replay-map.parquet')
        frame.to_parquet(path, index=False)
        hashes[path.name] = sha256(path)
    progress(project, 'MIT快照1574文件与功率比例参数已落盘；气象、Lisa/Acme/ASI和MIT需求输入SHA-256通过。')
    frames, summaries, checks, pending = {}, [], {}, []
    for scenario in matrix.loc[matrix.reference_it_mw.eq(10)].itertuples():
        config = configs[scenario.configuration_id]
        demand = DEMANDS[scenario.scenario_id.split('_')[-1]]
        if demand == 'Alibaba' and 'alibaba_cpu' not in replay:
            pending.append({'configuration_id': config['configuration_id'], 'demand': demand, 'status': '待数据',
                'reason': 'Alibaba 2018机器小时CPU实值数据仍不可用，跳过功率计算'})
            continue
        key = config['configuration_id']
        cpu, gpu = resource_pair(demand, replay)
        units = int(10e6 // reference_power(config))
        capacity = units * reference_power(config) / 1e6
        methods = GPU_METHODS if config['gpu_devices_per_unit'] else ('cpu_spec',)
        for method in methods:
            power = power_v2(config, cpu, gpu, method, ratios)
            for cooling_case in ('low', 'central', 'high'):
                cooling = CoolingModel(cooling_type='direct_liquid' if key == 'R-GB200' else 'air', parameter_set=cooling_case)
                frame = cooling.calculate(weather, power['unit_w'] * units / 1e6, capacity)
                for component in ('base', 'cpu', 'gpu'):
                    frame[component + '_it_mw'] = power[component + '_w'] * units / 1e6
                frame['unit_count'], frame['configuration_id'], frame['demand'] = units, key, demand
                frame['reference_it_mw'] = 10.
                frame['u_cpu_proxy'], frame['allocated_gpu_fraction'] = cpu, gpu
                frame['method'], frame['cooling_case'] = method, cooling_case
                frame['allocated_gpu_w_each'] = GPU_TDP_W[key] * ratios[method.removeprefix('ratio_')] if method.startswith('ratio_') else np.nan
                path = processed / f'{key}-{demand}-{method}-{cooling_case}.parquet'
                frame.to_parquet(path, index=False)
                hashes[path.name] = sha256(path)
                power_columns = [column for column in frame if column.endswith('_mw') or column.endswith('_mw_th')]
                for scale in (1, 100, 500):
                    scale_units = int(scale * 1e6 // reference_power(config))
                    factor = scale_units / units
                    scaled_frame = frame.copy()
                    scaled_frame[power_columns] = frame[power_columns] * factor
                    scaled_frame['unit_count'], scaled_frame['reference_it_mw'] = scale_units, float(scale)
                    scale_path = processed / f'{scale}MW-{key}-{demand}-{method}-{cooling_case}.parquet'
                    scaled_frame.to_parquet(scale_path, index=False)
                    hashes[scale_path.name] = sha256(scale_path)
                source_days = int(metadata[SOURCE_KEYS[demand]]['days'])
                summaries.append(summarize(frame, config, demand, method, cooling_case, source_days))
                label = f'{key}-{demand}-{method}-{cooling_case}'
                checks[label + ':8760_hours'] = bool(len(frame) == 8760 and frame.time_utc.is_unique)
                checks[label + ':components'] = bool(np.allclose(frame.it_mw, frame[['base_it_mw', 'cpu_it_mw', 'gpu_it_mw']].sum(axis=1), atol=1e-10))
                checks[label + ':facility'] = bool(np.allclose(frame.facility_mw, frame.it_mw + frame.cooling_mw + frame.other_mw, atol=1e-10))
                checks[label + ':thermal_balance'] = bool(frame.heat_balance_residual_mw.abs().max() < 1e-9)
                checks[label + ':finite_positive'] = bool(np.isfinite(frame[['it_mw', 'facility_mw', 'cooling_mw', 'other_mw', 'pue']]).all().all() and frame.it_mw.gt(0).all() and frame.pue.gt(1).all())
                if method in ('ratio_p50', 'cpu_spec') and cooling_case == 'central':
                    frames[(key, demand)] = frame
        print(json.dumps({'completed_curve': key + '-' + demand}, ensure_ascii=False), flush=True)
    summary10 = pd.DataFrame(summaries)
    all_rows = []
    for row in summary10.to_dict(orient='records'):
        for scale in (1, 10, 100, 500):
            units = int(scale * 1e6 // row['sizing_reference_w'])
            factor = units / row['units']
            scaled = {**row, 'reference_it_mw': scale, 'units': units,
                'installed_reference_it_mw': row['installed_reference_it_mw'] * factor}
            for column in row:
                if column.startswith('annual_') and column.endswith('_mwh') or column in ('peak_mw', 'it_peak_mw', 'minimum_mw', 'average_mw'):
                    scaled[column] = row[column] * factor
            all_rows.append(scaled)
    summary = pd.DataFrame(all_rows)
    central = summary.loc[summary.method.isin(['ratio_p50', 'cpu_spec']) & summary.cooling_case.eq('central')]
    central10 = central.loc[central.reference_it_mw.eq(10)].copy()
    summary.to_csv(output / 'sensitivity-summary.csv', index=False, encoding='utf-8-sig')
    central.to_csv(output / 'annual-summary.csv', index=False, encoding='utf-8-sig')
    parameters_v2(list(configs.values()), ratios).to_csv(output / 'power-parameters.csv', index=False, encoding='utf-8-sig')
    design = matrix[['scenario_id', 'reference_it_mw', 'configuration_id']].copy()
    design['demand'] = [DEMANDS[value.split('_')[-1]] for value in design.scenario_id]
    design['status'] = ['待数据' if value == 'Alibaba' and 'alibaba_cpu' not in replay else 'computed' for value in design.demand]
    design.to_csv(output / 'design-cell-status.csv', index=False, encoding='utf-8-sig')
    comparison = comparison_table(project, central10)
    comparison.to_csv(output / 'v1-v2-comparison.csv', index=False, encoding='utf-8-sig')
    # 统一选周时纳入新增MIT需求，逐小时评分表保存在忽略目录。
    week, week_scores = representative_week(weather, replay)
    week_scores.to_csv(processed / 'representative-week-selection.csv', index=False)
    checks['52_design_cells'] = len(design) == 52
    checks['48_computed_cells'] = int(design.status.eq('computed').sum()) == (52 if 'alibaba_cpu' in replay else 48)
    checks['ASI_exact_daily_repetition'] = bool(np.array_equal(replay['asi_gpu'], np.tile(replay['asi_gpu'][:24], 365)))
    for (key, demand), frame in frames.items():
        rows = summary10.loc[summary10.configuration_id.eq(key) & summary10.demand.eq(demand)]
        for method in rows.method.unique():
            cases = rows.loc[rows.method.eq(method)].set_index('cooling_case')
            checks[f'{key}-{demand}-{method}:cooling_order'] = bool(cases.loc['low', 'annual_facility_mwh'] <= cases.loc['central', 'annual_facility_mwh'] <= cases.loc['high', 'annual_facility_mwh'])
        # 模型对IT与安装容量同倍缩放应保持q和PUE，仅检查实际矩阵四档。
        for scale in (1, 100, 500):
            units = int(scale * 1e6 // reference_power(configs[key]))
            factor = units / frame.unit_count.iloc[0]
            scaled = CoolingModel(cooling_type='direct_liquid' if key == 'R-GB200' else 'air').calculate(weather, frame.it_mw.to_numpy() * factor, frame.installed_reference_it_mw.iloc[0] * factor)
            checks[f'{key}-{demand}-{scale}:scale_shape'] = bool(np.allclose(scaled.facility_mw, frame.facility_mw * factor, atol=1e-9) and np.allclose(scaled.pue, frame.pue, atol=1e-10))
    progress(project, f'全年曲线完成：{len(frames)}个10MW中央组合，{len(summary10)}条10MW方法/冷却曲线；四档规模汇总{len(central)}单元，Alibaba待数据。')
    from plot_typical_load_curves_v2 import plot_all_v2
    figure_checks = plot_all_v2(frames, comparison, output, week)
    checks.update(figure_checks['checks'])
    if not all(checks.values()):
        raise AssertionError([key for key, passed in checks.items() if not passed])
    validation = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'numerical_passed; visual_review_pending',
        'checks': checks, 'design_cells': 52, 'computed_cells': int(design.status.eq('computed').sum()),
        'central_10mw_combinations': len(frames), 'hourly_10mw_curves': len(summary10),
        'hourly_curves': len(summary), 'hourly_rows': len(summary) * 8760, 'pending': pending, 'figures': figure_checks,
        'weather': {'grid': GRID, 'year': 2025, 'hours': 8760, 'native_flat_index': int(center.native_flat_index.iloc[0])}}
    save_json(output / 'validation.json', validation)
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project, text=True).strip(),
        'dependency_prs': [f'https://github.com/TuzzyT-Laiii/33/pull/{n}' for n in (13, 14, 15, 19, 20)],
        'matrix_sha256': sha256(matrix_path), 'configuration_sha256': sha256(catalog_path),
        'weather_file_sha256': sha256(weather_path), 'weather_mapping_sha256': sha256(mapping_path),
        'mit_ratio_summary_sha256': sha256(ratio_path), 'mit_snapshot_files': ratio_summary['completed_files'],
        'sources': metadata, 'input_project': str(inputs), 'processed_directory': str(processed),
        'processed_sha256': hashes, 'week_start_utc': str(weather.time_utc.iloc[week]),
        'power_method': 'allocated_share*TDP*GPU-hour-power-ratio + (1-allocated_share)*idle_W; constant quantile scenarios',
        'sizing': 'v1中央整机满载参考固定台数/柜数，四档规模分别向下取整；冷却同倍缩放经实际调用验证',
        'gpu_migration': 'V100 PCIe 250W至H100 PCIe350W/SXM700W/GB200中Blackwell GPU1200W，跨代和跨需求分布均为假设',
        'publication': '仅聚合统计及图件入Git；MIT逐作业/逐小时衍生输入、重放与全量曲线均保留于忽略目录',
        'scientific_scope': '需求与功率假设的情境重算，未建立吞吐等价、目标设备实测精度或CFE/柔性收益结论',
        'code_sha256': {name: sha256(project / 'analysis' / name) for name in ['build_typical_load_curves_v2.py', 'typical_power_models_v2.py', 'plot_typical_load_curves_v2.py', 'summarize_mit_power_ratios.py', 'aggregate_mit_gpu_telemetry.py', 'compute_load/cooling.py']}}
    save_json(output / 'manifest.json', manifest)
    print(json.dumps({'output': str(output), 'checks': len(checks), 'central_combinations': len(frames)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--input-project', type=Path, required=True)
    parser.add_argument('--mit-source-dir', type=Path, required=True)
    parser.add_argument('--alibaba-hourly', type=Path)
    build(parser.parse_args())
