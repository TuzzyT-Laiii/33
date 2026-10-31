from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

from audit_research_gpu_demand import tres_count


FEATURE_WEIGHTS = {
    'log2_gpus': 1, 'log2_cpu_request': .25, 'log2_memory_request': .25,
    'memory_request_missing': 4,
    'log2_nodes': .5, 'log2_runtime_s': .5, 'log2_finite_timelimit': .1,
    'unlimited_timelimit': .5, 'log2_wait_s': .025, 'is_array': 4,
    'partition_normal': 1, 'partition_gaia': 1, 'partition_test': 1,
    'submit_hour_sin': .1, 'submit_hour_cos': .1,
    'submit_weekday_sin': .05, 'submit_weekday_cos': .05,
}


def load_jobs(root):
    columns = ['id_job', 'id_user', 'id_array_task', 'nodes_alloc', 'cpus_req', 'mem_req',
               'derived_ec', 'exit_code', 'partition', 'state', 'timelimit', 'time_submit',
               'time_start', 'time_end', 'time_suspended', 'tres_alloc', 'tres_req', 'job_type']
    parts, seen = [], set()
    rows = 0
    for chunk in pd.read_csv(Path(root) / 'reference/slurm-log.csv', usecols=columns, chunksize=20000):
        if chunk.id_job.isna().any() or chunk.id_job.duplicated().any() or seen.intersection(chunk.id_job):
            raise ValueError('作业ID缺失或重复')
        seen.update(chunk.id_job)
        rows += len(chunk)
        requested = sum(tres_count(chunk.tres_req, key).fillna(0) for key in [1001, 1002])
        data = chunk[requested.gt(0)].copy()
        data['gpus'] = sum(tres_count(data.tres_alloc, key).fillna(0) for key in [1001, 1002])
        data['cpu_request'] = tres_count(data.tres_req, 1)
        data['memory_request_mb'] = tres_count(data.tres_req, 2)
        data['cpu_allocated'] = tres_count(data.tres_alloc, 1)
        parts.append(data.drop(columns=['tres_alloc', 'tres_req', 'mem_req']))
    jobs = pd.concat(parts, ignore_index=True)
    if rows != 395914 or len(jobs) != 98176:
        raise ValueError('输入计数与PR #26不同')
    jobs['runtime_s'] = jobs.time_end - jobs.time_start
    jobs['runtime_h'] = jobs.runtime_s / 3600
    jobs['wait_s'] = jobs.time_start - jobs.time_submit
    jobs['wait_h'] = jobs.wait_s / 3600
    jobs['gpu_hours'] = jobs.gpus * jobs.runtime_h
    valid = (jobs.time_start.gt(0) & jobs.runtime_s.ge(0) & jobs.time_suspended.eq(0)
             & jobs.gpus.gt(0) & jobs.wait_s.ge(0) & jobs.cpu_request.gt(0)
             & (jobs.memory_request_mb.gt(0) | jobs.memory_request_mb.isna()))
    if not valid.all():
        issues = {name: int(jobs[name].isna().sum()) for name in ['gpus', 'cpu_request', 'memory_request_mb', 'wait_s']}
        issues.update({name + '_nonpositive': int(jobs[name].le(0).sum()) for name in ['gpus', 'cpu_request', 'memory_request_mb']})
        raise ValueError(f'GPU区间或资源请求无效: {issues}; negative_wait={int(jobs.wait_s.lt(0).sum())}')
    if not np.isclose(jobs.gpu_hours.sum(), 1460190.6641666667, atol=1e-8, rtol=0):
        raise ValueError('GPU小时分母与PR #26不同')
    jobs['finite_timelimit_min'] = jobs.timelimit.where(jobs.timelimit.lt(0xfffffffe))
    jobs['unlimited_timelimit'] = jobs.finite_timelimit_min.isna()
    jobs['is_array'] = jobs.id_array_task.ne(0xfffffffe)
    state_names = {0: 'PENDING', 3: 'COMPLETED', 4: 'CANCELLED', 5: 'FAILED',
                   6: 'TIMEOUT', 7: 'NODE_FAIL', 11: 'OUT_OF_MEMORY'}
    jobs['terminal_state'] = (jobs.state & 0xff).map(state_names)
    if jobs.terminal_state.isna().any():
        raise ValueError('未知Slurm终态')
    jobs.loc[(jobs.state & 0x400).ne(0), 'terminal_state'] += '|REQUEUE_FLAG'
    # 匿名原始mem_req包含MEM_PER_CPU高位，资源相似度使用TRES中的总内存请求。
    submission = pd.to_datetime(jobs.time_submit, unit='s', utc=True)
    jobs['submit_hour_utc'] = submission.dt.hour + submission.dt.minute / 60
    jobs['submit_weekday_utc'] = submission.dt.dayofweek
    return jobs


def feature_matrix(jobs):
    data = pd.DataFrame(index=jobs.index)
    for name, column in [('gpus', 'gpus'), ('cpu_request', 'cpu_request'), ('nodes', 'nodes_alloc'),
                         ('runtime_s', 'runtime_s')]:
        data['log2_' + name] = np.log2(jobs[column])
    data['log2_memory_request'] = np.log2(1 + jobs.memory_request_mb.fillna(0))
    data['memory_request_missing'] = jobs.memory_request_mb.isna().astype(float)
    data['log2_finite_timelimit'] = np.log2(1 + jobs.finite_timelimit_min.fillna(0))
    data['unlimited_timelimit'] = jobs.unlimited_timelimit.astype(float)
    data['log2_wait_s'] = np.log2(1 + jobs.wait_s)
    data['is_array'] = jobs.is_array.astype(float)
    for part in ['normal', 'gaia', 'test']:
        data['partition_' + part] = jobs.partition.eq(part).astype(float)
    if not jobs.partition.isin(['normal', 'gaia', 'test']).all():
        raise ValueError('分区范围发生变化')
    for field, period in [('submit_hour', 24), ('submit_weekday', 7)]:
        raw = jobs[field + '_utc']
        data[field + '_sin'] = np.sin(2 * np.pi * raw / period)
        data[field + '_cos'] = np.cos(2 * np.pi * raw / period)
    return data[list(FEATURE_WEIGHTS)].to_numpy() * np.array(list(FEATURE_WEIGHTS.values()))


def user_propagation(reference, target):
    result = pd.DataFrame({'user_high_support': 0, 'user_near_support': 0}, index=target.index)
    keys = ['id_user', 'gpus', 'partition', 'unlimited_timelimit']
    lookup = {key: group for key, group in reference.dropna(subset=['memory_request_mb']).groupby(keys)}
    for key, group in target.dropna(subset=['memory_request_mb']).groupby(keys):
        if key not in lookup:
            continue
        seeds = lookup[key]
        def coordinates(frame):
            return np.column_stack([np.log2(frame.cpu_request), np.log2(frame.memory_request_mb),
                                    np.log2(1 + frame.finite_timelimit_min.fillna(0)),
                                    frame.time_submit / 86400])
        left, right = coordinates(seeds), coordinates(group)
        for column, tolerance in [
            ('user_high_support', [np.log2(1.25), np.log2(1.25), np.log2(1.25), 90]),
            ('user_near_support', [np.log2(1.5), np.log2(1.5), 1, 90]),
        ]:
            scale = np.array(tolerance)
            tree = cKDTree(left / scale)
            result.loc[group.index, column] = tree.query_ball_point(
                right / scale, r=1 + 1e-12, p=np.inf, return_length=True)
    result['user_high'] = result.user_high_support.ge(2)
    result['user_near'] = result.user_near_support.ge(1)
    return result


def similarity(reference, target, omit=None):
    reference_x, target_x = feature_matrix(reference), feature_matrix(target)
    if omit:
        keep = [i for i, name in enumerate(FEATURE_WEIGHTS) if name not in omit]
        reference_x, target_x = reference_x[:, keep], target_x[:, keep]
    model = NearestNeighbors(metric='manhattan', algorithm='ball_tree', n_neighbors=6, n_jobs=1)
    model.fit(reference_x)
    # 阈值仅由参考正例内部的留一邻域确定，检验作业不参与阈值选择。
    calibration = model.kneighbors(reference_x, n_neighbors=6)[0][:, 1:].mean(axis=1)
    threshold = float(np.quantile(calibration, .90))
    distances, indices = [], []
    for offset in range(0, len(target), 1000):
        distance, neighbor = model.kneighbors(target_x[offset:offset+1000], n_neighbors=5)
        distances.append(distance.mean(axis=1))
        indices.append(neighbor[:, 0])
    distance = np.concatenate(distances)
    first = reference.iloc[np.concatenate(indices)].reset_index(drop=True)
    selected = target.reset_index(drop=True)
    ratio_cpu = np.maximum(selected.cpu_request.to_numpy() / first.cpu_request.to_numpy(),
                           first.cpu_request.to_numpy() / selected.cpu_request.to_numpy())
    ratio_mem = np.maximum(selected.memory_request_mb.to_numpy() / first.memory_request_mb.to_numpy(),
                           first.memory_request_mb.to_numpy() / selected.memory_request_mb.to_numpy())
    ratio_runtime = np.maximum(selected.runtime_s.to_numpy() / first.runtime_s.to_numpy(),
                               first.runtime_s.to_numpy() / selected.runtime_s.to_numpy())
    compatible = (selected.gpus.eq(first.gpus) & selected.partition.eq(first.partition)
                  & selected.unlimited_timelimit.eq(first.unlimited_timelimit)
                  & ~selected.is_array & (ratio_cpu <= 1.25) & (ratio_mem <= 1.25) & (ratio_runtime <= 2))
    return pd.DataFrame({'similarity_distance': distance, 'similarity_threshold': threshold,
                         'similarity_compatible': compatible.to_numpy(),
                         'similarity_medium': compatible.to_numpy() & (distance <= threshold)}, index=target.index)


def infer(reference, target):
    return user_propagation(reference, target).join(similarity(reference, target))
