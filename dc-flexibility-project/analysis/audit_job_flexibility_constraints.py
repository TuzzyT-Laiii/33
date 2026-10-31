import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SLURM_BASE = {0: 'PENDING', 1: 'RUNNING', 2: 'SUSPENDED', 3: 'COMPLETED',
              4: 'CANCELLED', 5: 'FAILED', 6: 'TIMEOUT', 7: 'NODE_FAIL',
              8: 'PREEMPTED', 9: 'BOOT_FAIL', 10: 'DEADLINE', 11: 'OUT_OF_MEMORY'}
QUANTILES = {'p50': .5, 'p90': .9, 'p99': .99}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


class Audit:
    def __init__(self, output):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.inputs = []
        self.classes, self.states, self.scales, self.durations = [], [], [], []
        self.details = {}

    def source(self, path):
        path = path.resolve()
        if self.output == path or path in self.output.parents or self.output in path.parents:
            raise ValueError('输出目录必须与输入目录分开')
        stat = path.stat()
        self.inputs.append({'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                            'sha256': digest(path)})

    def finish(self, prefix):
        for entry in self.inputs:
            stat = Path(entry['path']).stat()
            if stat.st_size != entry['bytes'] or stat.st_mtime_ns != entry['mtime_ns']:
                raise ValueError('分析期间输入元数据发生变化: ' + entry['path'])
        for name, rows in [('classes', self.classes), ('states', self.states),
                           ('scales', self.scales), ('durations', self.durations)]:
            pd.DataFrame(rows).to_csv(self.output / f'{prefix}-{name}.csv', index=False)
        write_json(self.output / f'{prefix}-audit.json', {
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'inputs': self.inputs, 'details': self.details,
            'analysis_sha256': digest(Path(__file__)),
            'software': {'numpy': np.__version__, 'pandas': pd.__version__, 'pyarrow': pa.__version__},
            'input_metadata_unchanged': True,
            'privacy': '仅输出聚合；不输出作业、用户、pod、workload、节点标识及逐作业时间'})

    def jobs(self, data, dataset):
        if data['work_hours'].lt(0).any():
            raise ValueError('资源小时不能为负')
        total_rows = len(data)
        pools = data.groupby('resource_pool')['work_hours'].sum()
        pool_rows = data.groupby('resource_pool').size()
        for (pool, label), group in data.groupby(['resource_pool', 'class'], observed=True):
            work = float(group['work_hours'].sum())
            valid = group[group.valid_runtime]
            runtime, wait = valid.runtime_s, group.wait_s.dropna()
            row = {'dataset': dataset, 'resource_pool': pool, 'class': label, 'rows': len(group),
                   'row_fraction_of_dataset': len(group) / total_rows,
                   'row_fraction_of_resource_pool': len(group) / pool_rows[pool],
                   'valid_runtime_rows': len(valid), 'invalid_runtime_rows': len(group) - len(valid),
                   'resource_hours': work,
                   'work_fraction_of_resource_pool': work / pools[pool] if pools[pool] else 0,
                   'runtime_under60_rows': int(runtime.lt(60).sum()),
                   'runtime_under60_row_fraction': float(runtime.lt(60).mean()) if len(valid) else None,
                   'runtime_under60_work_fraction': float(group.loc[group.valid_runtime & group.runtime_s.lt(60), 'work_hours'].sum() / work) if work else 0,
                   'runtime_ge1h_work_fraction': float(group.loc[group.valid_runtime & group.runtime_s.ge(3600), 'work_hours'].sum() / work) if work else 0,
                   'runtime_ge6h_work_fraction': float(group.loc[group.valid_runtime & group.runtime_s.ge(21600), 'work_hours'].sum() / work) if work else 0,
                   'wait_valid_rows': len(wait), 'nodes_nonpositive_rows': int(group.nodes.le(0).sum())}
            for name, values in [('runtime_s', runtime), ('wait_s', wait), ('units', group.units), ('nodes', group.nodes)]:
                for suffix, quantile in QUANTILES.items():
                    row[f'{name}_{suffix}'] = float(values.quantile(quantile)) if len(values.dropna()) else None
                row[f'{name}_max'] = float(values.max()) if len(values.dropna()) else None
            self.classes.append(row)
            for state, chosen in group.groupby('state', dropna=False):
                self.states.append({'dataset': dataset, 'resource_pool': pool, 'class': label,
                                    'state': state, 'rows': len(chosen),
                                    'resource_hours': float(chosen.work_hours.sum())})
            for scale, chosen in group.groupby('units', dropna=False):
                self.scales.append({'dataset': dataset, 'resource_pool': pool, 'class': label,
                                    'dimension': 'units', 'value': scale, 'rows': len(chosen),
                                    'resource_hours': float(chosen.work_hours.sum())})
            for scale, chosen in group.groupby('nodes', dropna=False):
                self.scales.append({'dataset': dataset, 'resource_pool': pool, 'class': label,
                                    'dimension': 'nodes', 'value': scale, 'rows': len(chosen),
                                    'resource_hours': float(chosen.work_hours.sum())})
            bins = pd.cut(group.runtime_s.where(group.valid_runtime),
                          [-1e-9, 60, 300, 3600, 21600, 86400, np.inf], right=False,
                          labels=['<60s', '60-300s', '300-3600s', '1-6h', '6-24h', '>=24h'])
            for band in bins.cat.categories:
                chosen = group[bins.eq(band)]
                self.durations.append({'dataset': dataset, 'resource_pool': pool, 'class': label,
                                       'runtime_band': band, 'rows': len(chosen),
                                       'resource_hours': float(chosen.work_hours.sum())})
        np.testing.assert_allclose(sum(x['resource_hours'] for x in self.classes if x['dataset'] == dataset),
                                   data.work_hours.sum(), rtol=1e-12)


def acme(audit, root):
    path = root / 'data/raw/task-trace-audit-20261026/trace_kalos.csv'
    audit.source(path)
    data = pd.read_csv(path)
    if data.job_id.isna().any() or data.job_id.duplicated().any():
        raise ValueError('Acme 作业标识不唯一')
    for col in ['submit_time', 'start_time', 'end_time']:
        data[col] = pd.to_datetime(data[col], utc=True, format='mixed')
    data['runtime_s'] = (data.end_time - data.start_time).dt.total_seconds()
    wait = (data.start_time - data.submit_time).dt.total_seconds()
    data['wait_s'] = wait.where(wait.ge(0))
    data['valid_runtime'] = data.runtime_s.ge(0) & wait.ge(0)
    data['class'] = data.type
    data['resource_pool'] = np.where(data.gpu_num.gt(0), 'gpu_hours', 'independent_cpu_hours')
    data['units'] = np.where(data.gpu_num.gt(0), data.gpu_num, data.cpu_num)
    data['nodes'] = data.node_num
    data['work_hours'] = (data.units * data.runtime_s / 3600).where(data.valid_runtime, 0)
    audit.jobs(data, 'Acme_Kalos')
    sequence = data.sort_values(['user', 'type', 'gpu_num', 'node_num', 'start_time'])
    keys = ['user', 'type', 'gpu_num', 'node_num']
    prev_end = sequence.groupby(keys).end_time.shift()
    prev_state = sequence.groupby(keys).state.shift()
    gap = (sequence.start_time - prev_end).dt.total_seconds()
    candidate = (prev_state.isin(['CANCELLED', 'FAILED']) & gap.between(0, 3600)
                 & sequence.submit_time.ge(prev_end))
    repeats = []
    for label, group in sequence[candidate].groupby('type'):
        repeats.append({'class': label, 'candidate_successor_rows': len(group),
                        'successor_completed_rows': int(group.state.eq('COMPLETED').sum()),
                        'candidate_gap_s_p50': float(gap.loc[group.index].median()),
                        'successor_resource_hours': float(group.work_hours.sum())})
    audit.details['Acme_Kalos'] = {
        'schema': list(data.columns[:17]), 'rows': len(data),
        'first_submit': str(data.submit_time.min()), 'last_end': str(data.end_time.max()),
        'raw_duration_matches_turnaround_rows': int(data.duration.eq((data.end_time - data.submit_time).dt.total_seconds()).sum()),
        'raw_duration_matches_runtime_rows': int(data.duration.eq(data.runtime_s).sum()),
        'negative_wait_rows': int(wait.lt(0).sum()),
        'repeated_submission_candidates': repeats,
        'candidate_rule': '同用户、type、GPU数、节点数按启动排序，前一作业 CANCELLED/FAILED；后作业提交不早于前作业结束，启动间隔0至3600秒',
        'resume_evidence': '弱关联候选；没有逻辑任务或 checkpoint 关联，不能认定保存后恢复',
        'preempted_rows': int(data.state.eq('PREEMPTED').sum()), 'checkpoint_fields': [], 'requeue_fields': []}


def lisa(audit, root):
    path = root / 'data/processed/task-trace-audit-20261026/lisa/lisa-tasks.parquet'
    audit.source(path)
    cols = ['id', 'submit_date', 'start_date', 'end_date', 'nodetypes', 'numnodes', 'numcores', 'state']
    data = pq.read_table(path, columns=cols, use_threads=False).to_pandas()
    if data.id.isna().any() or data.id.duplicated().any():
        raise ValueError('Lisa 作业标识不唯一')
    data['runtime_s'] = (data.end_date - data.start_date).dt.total_seconds()
    data['wait_s'] = (data.start_date - data.submit_date).dt.total_seconds()
    data['valid_runtime'] = data.runtime_s.ge(0) & data.wait_s.ge(0)
    if not data.valid_runtime.all():
        raise ValueError('Lisa 时间区间无效')
    data['class'], data['units'], data['nodes'] = data.nodetypes, data.numcores, data.numnodes
    data['resource_pool'] = np.where(data.nodetypes.str.startswith('gpu'), 'gpu_host_cpu_hours', 'cpu_hours')
    data['work_hours'] = data.runtime_s * data.units / 3600
    audit.jobs(data, 'Lisa')
    audit.details['Lisa'] = {'rows': len(data), 'raw_schema': pq.read_schema(path).names,
        'first_submit': str(data.submit_date.min()), 'last_end': str(data.end_date.max()),
        'preempted_rows': int(data.state.eq('PREEMPTED').sum()),
        'requeue_named_rows': int(data.state.str.contains('REQUEUE', case=False).sum()),
        'user_field_present': False, 'checkpoint_fields': [],
        'gpu_count_field_present': False, 'timezone': '源时间无时区'}


def asi(audit, root):
    base = root / 'data/raw/task-trace-audit-20261026/asi_opensource_pod_hourly/day=109'
    tables, inventories = [], []
    cols = ['pod_id', 'workload_id', 'server_id', 'job_type_public', 'priority_class',
            'state_public', 'gpu_request', 'used_gpu_hours', 'schedule_delay_sec', 'ready_delay_sec']
    all_rows = 0
    for hour in range(24):
        path = base / f'hour={hour:02d}/part-000.parquet'
        audit.source(path)
        all_rows += pq.ParquetFile(path).metadata.num_rows
        table = pq.read_table(path, columns=cols, filters=[('gpu_request', '>', 0)], use_threads=False).to_pandas()
        if table.duplicated(['pod_id', 'server_id']).any():
            raise ValueError('ASI 同小时 pod/server 重复')
        table['hour'] = hour
        for col in ['job_type_public', 'priority_class', 'state_public']:
            table[col] = table[col].astype('category')
        tables.append(table)
        inventory = root / f'data/raw/task-trace-audit-20261026/asi_opensource_server_hourly/day=109/hour={hour:02d}/part-000.parquet'
        audit.source(inventory)
        servers = pq.read_table(inventory, columns=['server_id', 'gpu_count'], use_threads=False).to_pandas()
        if servers.server_id.duplicated().any() or not table.server_id.isin(servers.server_id).all():
            raise ValueError('ASI 服务器关联无效')
        inventories.append({'hour': hour, 'gpu_count': float(servers.gpu_count.sum()),
                            'request': float(table.gpu_request.sum()),
                            'used_gpu_hours': float(table.used_gpu_hours.sum())})
        print(f'ASI hour {hour:02d}: {len(table)} GPU observations', flush=True)
    data = pd.concat(tables, ignore_index=True)
    del tables
    if data.used_gpu_hours.lt(0).any() or data.used_gpu_hours.isna().any():
        raise ValueError('ASI 归属GPU小时无效')
    total = float(data.used_gpu_hours.sum())
    rows, priority_rows, state_rows, presence_rows, workload_rows = [], [], [], [], []
    for label, group in data.groupby('job_type_public', observed=True):
        row = {'dataset': 'Alibaba_ASI_day109', 'class': label, 'pod_hour_observations': len(group),
               'observation_fraction': len(group) / len(data), 'distinct_pods': group.pod_id.nunique(),
               'distinct_workloads': group.workload_id.nunique(),
               'missing_workload_observations': int(group.workload_id.isna().sum()),
               'used_gpu_hours': float(group.used_gpu_hours.sum()),
               'gpu_hour_fraction': float(group.used_gpu_hours.sum() / total),
               'lp_gpu_hours': float(group.loc[group.priority_class.eq('LP'), 'used_gpu_hours'].sum()),
               'runtime_s_p50': None, 'runtime_s_p90': None, 'runtime_s_p99': None,
               'lifetime_job_count': None}
        unique_pods = group.sort_values('hour').drop_duplicates('pod_id')
        for name, values in [('gpu_request_per_pod_server_hour', group.gpu_request),
                             ('first_observed_pod_schedule_delay_s', unique_pods.schedule_delay_sec),
                             ('first_observed_pod_ready_delay_s', unique_pods.ready_delay_sec)]:
            valid = values[values.ge(0)].dropna()
            row[name + '_valid_rows'] = len(valid)
            row[name + '_missing_or_negative_rows'] = len(values) - len(valid)
            for suffix, q in QUANTILES.items():
                row[name + '_' + suffix] = float(valid.quantile(q)) if len(valid) else None
            row[name + '_max'] = float(valid.max()) if len(valid) else None
        rows.append(row)
        for priority, selected in group.groupby('priority_class', observed=True):
            priority_rows.append({'class': label, 'priority': priority, 'pod_hour_observations': len(selected),
                'used_gpu_hours': float(selected.used_gpu_hours.sum()),
                'fraction_all_gpu_hours': float(selected.used_gpu_hours.sum() / total),
                'fraction_class_gpu_hours': float(selected.used_gpu_hours.sum() / group.used_gpu_hours.sum())})
        for state, selected in group.groupby('state_public', observed=True):
            state_rows.append({'class': label, 'state': state, 'pod_hour_observations': len(selected),
                               'used_gpu_hours': float(selected.used_gpu_hours.sum())})
        observed = group.groupby('pod_id').hour.agg(['min', 'max', 'nunique'])
        gap = observed['max'] - observed['min'] + 1 - observed['nunique']
        presence_rows.append({'class': label, 'distinct_pods': len(observed),
            'pods_observed_at_hour0': int(observed['min'].eq(0).sum()),
            'pods_observed_at_hour23': int(observed['max'].eq(23).sum()),
            'pods_with_internal_hour_gaps': int(gap.gt(0).sum()),
            'observed_hours_p50': float(observed['nunique'].median()),
            'observed_hours_p90': float(observed['nunique'].quantile(.9)),
            'observed_hours_p99': float(observed['nunique'].quantile(.99))})
        workload = group.dropna(subset=['workload_id']).groupby(['workload_id', 'hour']).agg(
            gpu_request=('gpu_request', 'sum'), servers=('server_id', 'nunique'), pods=('pod_id', 'nunique'))
        wrow = {'class': label, 'observed_workload_hours': len(workload)}
        for col in ['gpu_request', 'servers', 'pods']:
            for suffix, q in QUANTILES.items():
                wrow[col + '_' + suffix] = float(workload[col].quantile(q)) if len(workload) else None
            wrow[col + '_max'] = float(workload[col].max()) if len(workload) else None
        workload_rows.append(wrow)
    for name, values in [('classes', rows), ('priorities', priority_rows), ('states', state_rows),
                         ('presence', presence_rows), ('workload-scales', workload_rows), ('inventory', inventories)]:
        pd.DataFrame(values).to_csv(audit.output / f'asi-{name}.csv', index=False)
    published = pd.read_csv(root / 'data/processed/task-trace-audit-20261026/asi-hourly-type-priority.csv')
    expected = published.groupby('job_type_public').used_gpu_hours.sum().sort_index()
    actual = data.groupby('job_type_public', observed=True).used_gpu_hours.sum().sort_index()
    np.testing.assert_allclose(actual, expected, rtol=1e-10)
    audit.details['Alibaba_ASI_day109'] = {'all_pod_hour_rows': all_rows, 'gpu_pod_hour_rows': len(data),
        'distinct_gpu_pods': int(data.pod_id.nunique()), 'distinct_workloads': int(data.workload_id.nunique()),
        'unique_job_count': None, 'raw_schema': pq.read_schema(base / 'hour=00/part-000.parquet').names,
        'used_gpu_hours': total, 'priority_hours_conserved': True, 'previous_audit_reconciled': True,
        'runtime': '缺少提交/启动/结束和执行跨度表；小时出现时长不是作业运行时长',
        'presence': '缺席后重现只说明小时观测存在间断；不证明checkpoint或恢复',
        'schedule_delay': '每类每pod首次观测值，非负值分位数；-1和缺失排除，重复小时不重复计权',
        'workload_scale': '同workload同小时申请GPU总和与服务器数；非已证明的同步gang或最小运行规模',
        'checkpoint_fields': [], 'requeue_fields': [],
        'execution_summary_downloaded': False}


def mit(audit, root):
    from audit_research_gpu_demand import tres_count

    for name in ['slurm-log.csv', 'tres-mapping.txt', 'README.md', 'LICENSE']:
        audit.source(root / 'reference' / name)
    path = root / 'reference/slurm-log.csv'
    data = pd.read_csv(path, dtype={'tres_req': str, 'tres_alloc': str})
    if data.id_job.isna().any() or data.id_job.duplicated().any():
        raise ValueError('MIT 作业标识不唯一')
    mapping = pd.read_csv(root / 'reference/tres-mapping.txt')
    resources = dict(zip(mapping.tres_id, mapping.resource_type.str.strip()))
    if resources.get(1) != 'cpu' or resources.get(1001) != 'gpu:tesla' or resources.get(1002) != 'gpu:volta':
        raise ValueError('TRES 映射发生变化')
    for col in ['tres_req', 'tres_alloc']:
        unknown = set(data[col].dropna().str.findall(r'(?:^|,)(\d+)=').explode().dropna().astype(int)) - set(resources)
        if unknown:
            raise ValueError(f'未知TRES字段: {unknown}')
    requested = sum(tres_count(data.tres_req, key).fillna(0) for key in [1001, 1002])
    allocated = sum(tres_count(data.tres_alloc, key).fillna(0) for key in [1001, 1002])
    cpu = tres_count(data.tres_alloc, 1).fillna(0)
    raw_state = data.state.copy()
    data['state'] = (raw_state & 0xff).map(SLURM_BASE)
    if data.state.isna().any():
        raise ValueError('未知Slurm基础状态')
    requeue = (raw_state & 0x400).ne(0)
    data.loc[requeue, 'state'] = data.loc[requeue, 'state'] + '|REQUEUE_FLAG'
    data['runtime_s'] = data.time_end - data.time_start
    interval = data.time_start.gt(0) & data.time_end.gt(0) & data.time_end.ge(data.time_start) & data.time_suspended.eq(0)
    gpu = requested.gt(0)
    data['valid_runtime'] = interval & np.where(gpu, allocated.gt(0), cpu.gt(0))
    wait = data.time_start - data.time_submit
    data['wait_s'] = wait.where(data.time_start.gt(0) & data.time_submit.gt(0) & wait.ge(0))
    data['units'] = np.where(gpu, allocated, cpu)
    data['nodes'], data['class'] = data.nodes_alloc, data.job_type
    data['resource_pool'] = np.where(gpu, 'gpu_hours', 'independent_slurm_cpu_hours')
    data['work_hours'] = (data.runtime_s * data.units / 3600).where(data.valid_runtime, 0)
    audit.jobs(data, 'MIT_Supercloud')
    # Slurm NO_VAL 为0xfffffffe，匿名化后的非零数组ID不能识别数组成员。
    array = data[data.id_array_task.ne(0xfffffffe)]
    duplicate_array_keys = array.groupby(['id_user', 'id_array_job', 'id_array_task']).size()
    audit.details['MIT_Supercloud'] = {'rows': len(data), 'raw_schema': list(pd.read_csv(path, nrows=0).columns),
        'gpu_requested_rows': int(gpu.sum()), 'valid_gpu_execution_rows': int((gpu & data.valid_runtime).sum()),
        'independent_cpu_rows': int((~gpu).sum()),
        'no_gpu_request_but_gpu_allocation_rows': int((~gpu & allocated.gt(0)).sum()),
        'invalid_runtime_rows': int((~data.valid_runtime).sum()),
        'preempted_base_state_rows': int((raw_state & 0xff).eq(8).sum()),
        'requeue_flag_rows': int(requeue.sum()), 'requeue_flag_gpu_requested_rows': int((requeue & gpu).sum()),
        'requeue_flag_valid_execution_rows': int((requeue & data.valid_runtime).sum()),
        'nonzero_time_suspended_rows': int(data.time_suspended.ne(0).sum()),
        'explicit_requeue_column_present': 'requeue' in data.columns,
        'array_member_rows': len(array), 'distinct_arrays': int(array.id_array_job.nunique()),
        'duplicate_user_array_task_keys': int(duplicate_array_keys.gt(1).sum()),
        'array_scope': '按id_array_task排除Slurm NO_VAL后统计；数组成员不等于同一训练任务的恢复段',
        'state_numeric_source': 'SchedMD slurm-20-11-9-1/slurm/slurm.h.in；实际采集版本未公开，按兼容定义解码',
        'checkpoint_fields': [], 'first_submit_epoch_s': int(data.time_submit[data.time_submit.gt(0)].min()),
        'last_end_epoch_s': int(data.time_end.max()),
        'labelled_jobids_present': (root / 'reference/labelled_jobids.csv').exists()}
    snapshot = root / 'aggregates/current/summary.json'
    content = snapshot.read_bytes()
    summary = json.loads(content)
    audit.details['MIT_telemetry_context'] = {'path': str(snapshot),
        'sha256': hashlib.sha256(content).hexdigest(),
        'snapshot_created_at': summary['created_at'],
        'completed_files': summary.get('completed_files'),
        'scope': '仅读取既有汇总；未打开数据库，未改写current；闲置/功率波动不能归因为RL或可释放资源'}
    audit.finish('mit')


def verify(output):
    classes = pd.concat([pd.read_csv(output / f'{prefix}-classes.csv') for prefix in ['server', 'mit']], ignore_index=True)
    states = pd.concat([pd.read_csv(output / f'{prefix}-states.csv') for prefix in ['server', 'mit']], ignore_index=True)
    durations = pd.concat([pd.read_csv(output / f'{prefix}-durations.csv') for prefix in ['server', 'mit']], ignore_index=True)
    keys = ['dataset', 'resource_pool', 'class']
    np.testing.assert_allclose(classes.set_index(keys).resource_hours.sort_index(),
                               states.groupby(keys).resource_hours.sum().sort_index(), rtol=1e-11)
    np.testing.assert_allclose(classes.set_index(keys).resource_hours.sort_index(),
                               durations.groupby(keys).resource_hours.sum().sort_index(), rtol=1e-11)
    np.testing.assert_array_equal(classes.set_index(keys).valid_runtime_rows.sort_index(),
                                  durations.groupby(keys).rows.sum().sort_index())
    for pool, group in classes.groupby(['dataset', 'resource_pool']):
        np.testing.assert_allclose(group.work_fraction_of_resource_pool.sum(), 1, atol=1e-12)
        np.testing.assert_allclose(group.row_fraction_of_resource_pool.sum(), 1, atol=1e-12)
    a = pd.read_csv(output / 'asi-classes.csv')
    p = pd.read_csv(output / 'asi-priorities.csv')
    np.testing.assert_allclose(a.set_index('class').used_gpu_hours.sort_index(),
                               p.groupby('class').used_gpu_hours.sum().sort_index(), rtol=1e-11)
    np.testing.assert_allclose(a.gpu_hour_fraction.sum(), 1, atol=1e-12)
    np.testing.assert_array_equal(classes.groupby('dataset').rows.sum().sort_index(), [62413, 1596963, 395914])
    if int(a.pod_hour_observations.sum()) != 2816548:
        raise ValueError('ASI 样本量与既有核验不一致')
    metadata = Path(__file__).resolve().parents[1] / 'data/metadata/task-trace-audit-20261026'
    previous_lisa = json.loads((metadata / 'lisa-summary.json').read_text(encoding='utf-8'))
    lisa_classes = classes[classes.dataset.eq('Lisa')]
    for pool, label in [('cpu_hours', 'cpu_partition'), ('gpu_host_cpu_hours', 'gpu_partition')]:
        np.testing.assert_allclose(lisa_classes.loc[lisa_classes.resource_pool.eq(pool), 'resource_hours'].sum(),
                                   previous_lisa['groups'][label]['allocated_core_hours'], rtol=1e-11)
    previous = json.loads((metadata / 'asi-acme-validation.json').read_text(encoding='utf-8'))
    for pool, label in [('gpu_hours', 'gpu'), ('independent_cpu_hours', 'cpu')]:
        np.testing.assert_allclose(classes.loc[classes.dataset.eq('Acme_Kalos') & classes.resource_pool.eq(pool), 'resource_hours'].sum(),
                                   previous['acme']['resource_integrals'][label]['integrated_resource_hours'], rtol=1e-11)
    # 检查公开表中没有能够连接回逐作业记录的标识或时间字段。
    forbidden = {'job_id', 'id_job', 'user', 'id_user', 'pod_id', 'workload_id', 'server_id',
                 'node', 'nodelist', 'time_start', 'time_end', 'start_time', 'end_time'}
    for path in output.glob('*.csv'):
        columns = set(pd.read_csv(path, nrows=0).columns)
        if forbidden & columns:
            raise ValueError('公开表出现逐作业字段: ' + str(path))
    checks = {'resource_hours_by_state': 'passed', 'resource_hours_by_duration': 'passed',
              'valid_jobs_by_duration': 'passed', 'within_pool_shares': 'passed',
              'asi_priority_conservation': 'passed', 'known_dataset_sizes': 'passed',
              'previous_lisa_acme_resource_integrals': 'passed',
              'public_csv_identifier_check': 'passed',
              'scope': '真实输入聚合和既有审计对照；没有运行能源优化或赋予任务权限'}
    write_json(output / 'validation.json', checks)
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='只读任务输入，输出负载类别、规模、时长与柔性证据汇总')
    parser.add_argument('--mode', choices=['server', 'mit', 'verify'], required=True)
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    if args.mode == 'verify':
        verify(args.output)
    else:
        if args.input_root is None:
            parser.error('--input-root 为必需参数')
        audit = Audit(args.output)
        if args.mode == 'server':
            acme(audit, args.input_root)
            lisa(audit, args.input_root)
            asi(audit, args.input_root)
            audit.finish('server')
        else:
            mit(audit, args.input_root)
