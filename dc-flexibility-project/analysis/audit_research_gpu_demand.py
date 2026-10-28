import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from compute_load.model import sha256
from task_trace_inputs import interval_resource_hours


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def lisa_audit(root):
    expected = {'slurm_table_cleaned.zip': 'ae697e29d3d4ce10c4b1ab670b82d06c',
                'node_hardware_info.parquet': '9c004d3ac06a97f6de1e0917640a9464'}
    inputs = {}
    for name, md5 in expected.items():
        path = root / name
        content = path.read_bytes()
        if hashlib.md5(content).hexdigest() != md5:
            raise ValueError('Lisa 原始文件与官方 MD5 不符: ' + name)
        inputs[name] = {'path': str(path), 'bytes': len(content), 'md5': md5, 'sha256': sha256(path)}
    with zipfile.ZipFile(root / 'slurm_table_cleaned.zip') as archive:
        jobs = pd.concat([pd.read_parquet(io.BytesIO(archive.read(name)))
                          for name in archive.namelist() if name.endswith('.parquet')], ignore_index=True)
    if jobs.id.isna().any() or jobs.id.duplicated().any():
        raise ValueError('Lisa 作业标识缺失或重复')
    gpu = jobs[jobs.nodetypes.str.startswith('gpu')]
    if list(jobs.columns) != ['id', 'submit_date', 'start_date', 'end_date', 'node', 'nodetypes', 'numnodes', 'numcores', 'state']:
        raise ValueError('Lisa 原始 schema 发生变化，需要重新审阅 GPU 字段')
    left, right = gpu.start_date.min().ceil('D'), gpu.end_date.max().floor('D')
    hardware = pd.read_parquet(root / 'node_hardware_info.parquet')
    return {'source': 'SURF Lisa Zenodo v3', 'source_url': 'https://zenodo.org/records/13685426',
            'license': 'CC-BY-4.0', 'inputs': inputs, 'rows': len(jobs), 'gpu_partition_rows': len(gpu),
            'job_schema': {c: str(t) for c, t in jobs.dtypes.items()},
            'gpu_first_start': str(gpu.start_date.min()), 'gpu_last_end': str(gpu.end_date.max()),
            'full_calendar_days': int((right - left).days), 'full_day_begin': str(left), 'full_day_end_exclusive': str(right),
            'gpu_partition_host_core_hours': float(((gpu.end_date - gpu.start_date).dt.total_seconds() * gpu.numcores).sum() / 3600),
            'gpu_request_count_field': None, 'gpu_allocated_count_field': None,
            'gpu_utilization_field': None, 'gpu_power_field': None,
            'hardware_gpu_count': int(hardware.gpu_count.sum()),
            'hardware_count_scope': '节点静态库存，不能确定共享分区各作业的 GPU 分配数',
            'gpu_occupancy_usable': False, 'usable_gpu_occupancy_days': 0,
            'monitoring_local_to_source_dir': [p.name for p in root.glob('*prom*') if p.is_file()],
            'monitoring_published': {'sampling_seconds': 30, 'prom_table_cleaned.zip_bytes': 10242740338,
                                     'prom_slurm_joined.zip_bytes': 6895904211, 'downloaded_this_task': False},
            'timezone': '源时间未携带时区',
            'conclusion': 'GPU 分区存在；现有作业及硬件原文不足以构造 GPU 卡小时；完整监测待获取核验'}


def tres_count(series, resource):
    # TRES 是 CSV 字段中的资源键值列表；只提取官方映射中明确列出的资源。
    value = series.str.extract(r'(?:^|,)' + str(resource) + r'=(\d+)(?=,|$)', expand=False)
    return pd.to_numeric(value, errors='raise')


def mit_audit(root, output):
    acquisition = json.loads((root / 'acquisition.json').read_text(encoding='utf-8'))
    for record in acquisition['files']:
        if sha256(root / record['file']) != record['sha256'] or not record['etag_verified']:
            raise ValueError('MIT 下载文件与获取记录不符')
    mapping = pd.read_csv(root / 'tres-mapping.txt')
    resources = dict(zip(mapping.tres_id, mapping.resource_type.str.strip()))
    if resources.get(1) != 'cpu' or resources.get(1001) != 'gpu:tesla' or resources.get(1002) != 'gpu:volta':
        raise ValueError('MIT GPU 资源映射与已审阅版本不同')
    jobs = pd.read_csv(root / 'slurm-log.csv', dtype={'tres_alloc': str, 'tres_req': str})
    if jobs.id_job.isna().any() or jobs.id_job.duplicated().any():
        raise ValueError('MIT 作业标识缺失或重复')
    for column in ('tres_alloc', 'tres_req'):
        unknown = set(jobs[column].dropna().str.findall(r'(?:^|,)(\d+)=').explode().dropna().astype(int)) - set(resources)
        if unknown:
            raise ValueError(f'MIT TRES 含未映射资源: {unknown}')
    jobs['gpu_allocated'] = sum(tres_count(jobs.tres_alloc, key).fillna(0) for key in (1001, 1002))
    jobs['gpu_requested'] = sum(tres_count(jobs.tres_req, key).fillna(0) for key in (1001, 1002))
    jobs['host_cpu_allocated'] = tres_count(jobs.tres_alloc, 1)
    requested = jobs.gpu_requested.gt(0)
    allocated = jobs.gpu_allocated.gt(0)
    valid = jobs.time_start.gt(0) & jobs.time_end.ge(jobs.time_start) & jobs.time_end.gt(0)
    suspended = jobs.time_suspended.gt(0)
    selected = jobs[requested & allocated & valid & ~suspended].copy()
    if selected.host_cpu_allocated.isna().any() or selected.host_cpu_allocated.le(0).any():
        raise ValueError('GPU 执行作业缺少有效宿主 CPU 分配数')
    selected['start'] = pd.to_datetime(selected.time_start, unit='s')
    selected['end'] = pd.to_datetime(selected.time_end, unit='s')
    begin, end = selected.start.min().floor('h'), selected.end.max().ceil('h')
    edges = pd.date_range(begin, end, freq='h')
    hourly = pd.DataFrame({'source_time': edges[:-1]})
    checks = {}
    for resource, column in [('gpu', 'gpu_allocated'), ('host_cpu', 'host_cpu_allocated')]:
        work = interval_resource_hours(selected.start.array.asi8, selected.end.array.asi8, selected[column], edges.asi8)
        expected = float(((selected.time_end - selected.time_start) * selected[column]).sum() / 3600)
        np.testing.assert_allclose(work.sum(), expected, rtol=1e-10)
        hourly[resource + '_allocation_hours'] = work
        checks[resource + '_hours_conserved'] = expected
        sample = np.linspace(0, len(edges) - 2, 48, dtype=int)
        direct = [float((np.maximum(0, np.minimum(selected.time_end, edges[i + 1].value / 1e9)
                                   - np.maximum(selected.time_start, edges[i].value / 1e9)) * selected[column]).sum() / 3600) for i in sample]
        np.testing.assert_allclose(work[sample], direct, rtol=1e-9, atol=1e-6)
        checks[resource + '_independent_overlap_checks'] = len(sample)
    left, right = selected.start.min().ceil('D'), selected.end.max().floor('D')
    complete = hourly[hourly.source_time.ge(left) & hourly.source_time.lt(right)].copy()
    if len(complete) < 168 or len(complete) % 24:
        raise ValueError('MIT 未获得至少一周的完整小时窗口')
    if not complete.source_time.diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError('MIT 小时索引不连续')
    complete['source_hour'] = np.arange(len(complete))
    output.mkdir(parents=True, exist_ok=True)
    complete.to_csv(output / 'research-gpu-observed-hourly.csv', index=False)
    selected[['id_job', 'start', 'end', 'gpu_allocated', 'gpu_requested', 'host_cpu_allocated', 'state']].to_parquet(output / 'research-gpu-jobs.parquet', index=False)
    schema = {c: str(t) for c, t in jobs.dtypes.items()}
    summary = {'source': 'MIT Supercloud 202201 科研 GPU 作业',
               'source_url': 'https://registry.opendata.aws/dcc/',
               'documentation_url': 'https://github.com/MIT-AI-Accelerator/MIT-Supercloud-Dataset',
               'citation': 'Samsi et al. (2021), The MIT Supercloud Dataset, DOI:10.1109/HPEC49654.2021.9622850',
               'license': 'CC-BY-NC-ND-4.0', 'acquisition': acquisition, 'job_schema': schema,
               'total_job_rows': len(jobs), 'gpu_requested_rows': int(requested.sum()),
               'gpu_allocated_rows': int(allocated.sum()), 'selected_gpu_execution_rows': len(selected),
               'excluded_allocated_gpu_invalid_interval': int((allocated & ~valid).sum()),
               'excluded_allocated_gpu_suspended': int((allocated & valid & suspended).sum()),
               'gpu_requested_without_allocation': int((requested & ~allocated).sum()),
               'gpu_allocation_without_request': int((allocated & ~requested).sum()),
               'selection_rule': 'GPU 请求数与分配数均大于零、执行区间有效且无暂停累计；排除只有整节点 GPU 预留而没有 GPU 请求的作业',
               'zero_duration_selected': int(selected.time_start.eq(selected.time_end).sum()),
               'selected_state_counts': {str(k): int(v) for k, v in selected.state.value_counts().items()},
               'gpu_first_start': str(selected.start.min()), 'gpu_last_end': str(selected.end.max()),
               'full_day_begin': str(left), 'full_day_end_exclusive': str(right),
               'full_calendar_days': len(complete) // 24, 'observed_hours': len(complete),
               'nonzero_gpu_days': int(complete.groupby(complete.source_time.dt.floor('D')).gpu_allocation_hours.sum().gt(0).sum()),
               'gpu_resource_ids': {'1001': 'gpu:tesla', '1002': 'gpu:volta'},
               'cpu_scope': '同一批实际分配 GPU 的执行作业，其 tres_alloc 的 CPU 数量；不含独立 CPU 作业',
               'quantity_scope': '执行区间内分配资源小时；未将分配量解释为实测活跃利用率或功率',
               'timezone': '保留由源 epoch 秒得到的源时间；未核实其真实当地时区，不连接天气',
               'coverage_quality': '时间跨度内的完整 24 小时时间窗；日志采集完整性与漏报率未独立证明',
               'power_and_performance_mapping': 'Tesla/Volta 向 H100 的性能、显存、活跃利用率和功率映射待核定',
               'validation': checks, 'observed_hourly_sha256': sha256(output / 'research-gpu-observed-hourly.csv')}
    summary['telemetry_samples'] = []
    for name in acquisition['telemetry_sample_files']:
        sample = pd.read_csv(root / name)
        summary['telemetry_samples'].append({'file': name, 'rows': len(sample),
                                             'schema': {c: str(t) for c, t in sample.dtypes.items()}})
    summary['telemetry_scope'] = '两份小样本仅核验字段，未用于集群活跃利用率或功率外推'
    write_json(output / 'mit-summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lisa-dir', type=Path, required=True)
    parser.add_argument('--mit-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lisa = lisa_audit(args.lisa_dir)
    write_json(args.output_dir / 'lisa-gpu-audit.json', lisa)
    print(json.dumps({'lisa': lisa}, ensure_ascii=False), flush=True)
    mit = mit_audit(args.mit_dir, args.output_dir)
    for name, record in lisa['inputs'].items():
        if sha256(args.lisa_dir / name) != record['sha256']:
            raise ValueError('核查期间 Lisa 原始文件发生变化')
    lisa['original_bytes_preserved'] = True
    write_json(args.output_dir / 'lisa-gpu-audit.json', lisa)
    write_json(args.output_dir / 'source-audit.json', {'lisa': lisa, 'adopted': mit})
    print(json.dumps({'MIT_gpu_jobs': mit['selected_gpu_execution_rows'], 'full_days': mit['full_calendar_days']}, ensure_ascii=False), flush=True)
