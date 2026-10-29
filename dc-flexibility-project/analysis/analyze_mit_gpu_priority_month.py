import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd

from mit_gpu_common import atomic_json, identity, load_jobs, process_lock, utc
from mit_gpu_priority import overlaps, window_bounds


HOUR_US = 3_600_000_000
CAPACITY_SOURCE = 'https://arxiv.org/pdf/2108.02037#page=2'
TDP_SOURCE = 'https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf'


def committed_snapshot(root, jobs, begin, finish):
    db = sqlite3.connect((root / 'state.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    db.execute('BEGIN')
    objects = [dict(r) for r in db.execute('SELECT * FROM objects ORDER BY seq')
               if overlaps(identity(r['key'], jobs)[2], begin, finish)]
    db.rollback()
    db.close()
    if not objects:
        raise ValueError('窗口没有关联监控文件')
    done = [o for o in objects if o['status'] == 'done']
    tables = {'files': [], 'hourly': [], 'joint_hist': []}
    for batch in sorted({o['batch'] for o in done}):
        selected = {o['seq'] for o in done if o['batch'] == batch}
        folder = root / 'batches' / batch
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        if not selected.issubset({o['seq'] for o in manifest['objects'] if o['status'] == 'done'}):
            raise ValueError('批次清单没有包含快照中的完成对象: ' + batch)
        for kind in tables:
            frame = pd.read_parquet(folder / (kind + '.parquet'))
            tables[kind].append(frame.loc[frame.seq.isin(selected)])
    if not done:
        raise ValueError('窗口尚未有完整提交的监控对象')
    tables = {kind: pd.concat(frames, ignore_index=True) for kind, frames in tables.items()}
    if tables['files'].seq.duplicated().any() or set(tables['files'].seq) != {o['seq'] for o in done}:
        raise ValueError('文件汇总重复或与 SQLite 快照不符')
    return objects, tables


def allocations(jobs, objects, begin, finish):
    hours = pd.date_range(begin, finish, freq='h', inclusive='left')
    index = hours.asi8 // 1000
    total, available, completed = [np.zeros(len(hours)) for _ in range(3)]
    nodes_by_job, done_by_job = {}, {}
    for obj in objects:
        job_id, node, _, mapping = identity(obj['key'], jobs)
        if mapping == 'matched':
            nodes_by_job.setdefault(job_id, set()).add(node)
            if obj['status'] == 'done':
                done_by_job.setdefault(job_id, set()).add(node)
    integrated = 0.0
    for job_id, job in jobs.items():
        if not overlaps(job, begin, finish) or not job['allocated_gpus']:
            continue
        start = max(int(job['start'] * 1_000_000), int(index[0]))
        end = min(int(job['end'] * 1_000_000), int(finish.value // 1000))
        first = (start - int(index[0])) // HOUR_US
        last = (end - 1 - int(index[0])) // HOUR_US
        positions = np.arange(first, last + 1, dtype=int)
        duration = np.minimum(index[positions] + HOUR_US, end) - np.maximum(index[positions], start)
        assigned = duration / HOUR_US * job['allocated_gpus']
        total[positions] += assigned
        integrated += (end - start) / HOUR_US * job['allocated_gpus']
        if job['nodes']:
            available[positions] += assigned * len(nodes_by_job.get(job_id, ())) / len(job['nodes'])
            completed[positions] += assigned * len(done_by_job.get(job_id, ())) / len(job['nodes'])
    if not np.isclose(total.sum(), integrated, rtol=1e-12, atol=1e-8):
        raise ValueError('Slurm 月内 GPU 小时积分不守恒')
    return pd.DataFrame({'hour_us': index, 'hour_utc': hours,
        'allocated_gpu_count': total, 'file_available_gpu_count': available,
        'file_completed_gpu_count': completed})


def device_hours(tables, begin, finish):
    files, source = tables['files'], tables['hourly']
    accepted = files.loc[files.mapping_status.eq('matched') & files.clock_status.eq('new_york_wall_time_inferred')
                         & files.eligibility.eq('eligible')].set_index('seq')
    frame = source.loc[source.seq.isin(accepted.index) & source.within_allocation
        & source.hour_us.ge(begin.value // 1000) & source.hour_us.lt(finish.value // 1000)].copy()
    if frame.empty:
        raise ValueError('窗口缺少时钟可定位且分配归属明确的实测功率')
    frame['span_start'] = np.maximum(frame.hour_us, frame.seq.map(accepted.job_start) * 1_000_000)
    frame['span_end'] = np.minimum(frame.hour_us + HOUR_US, frame.seq.map(accepted.job_end) * 1_000_000)
    dimensions = ['hour_us', 'node', 'gpu_index']
    frame = frame.sort_values(dimensions + ['span_start', 'span_end', 'seq'])
    frame['cumulative_end'] = frame.groupby(dimensions).span_end.cummax()
    previous = frame.groupby(dimensions).cumulative_end.shift()
    frame['potential_overlap'] = frame.span_start.lt(previous)
    result = frame.groupby(dimensions, as_index=False).agg(
        power_valid_us=('power_draw_W_valid_us', 'sum'),
        power_time_sum_us=('power_draw_W_time_sum_us', 'sum'),
        sample_count=('power_draw_W_count', 'sum'),
        files=('seq', 'nunique'), potential_overlap=('potential_overlap', 'max'))
    result['ambiguous'] = result.potential_overlap | result.power_valid_us.gt(HOUR_US)
    result['mean_power_W'] = result.power_time_sum_us / result.power_valid_us.replace(0, np.nan)
    result['power_energy_Wh'] = result.power_time_sum_us / HOUR_US
    result['coverage'] = result.power_valid_us / HOUR_US
    # 小时汇总不能恢复重复样本的精确区间，疑似重叠的设备小时单列并排除。
    return result, accepted


def metrics(actual, predicted):
    frame = pd.DataFrame({'actual': actual, 'predicted': predicted}).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return {'hours': 0, 'bias_W': None, 'MAE_W': None, 'RMSE_W': None, 'WAPE': None, 'pearson_r': None}
    error = frame.predicted - frame.actual
    correlation = frame.actual.corr(frame.predicted) if len(frame) > 1 and frame.actual.std() > 0 and frame.predicted.std() > 0 else None
    return {'hours': len(frame), 'bias_W': float(error.mean()), 'MAE_W': float(error.abs().mean()),
        'RMSE_W': float(np.sqrt((error ** 2).mean())),
        'WAPE': float(error.abs().sum() / frame.actual.abs().sum()) if frame.actual.abs().sum() else None,
        'pearson_r': float(correlation) if correlation is not None and pd.notna(correlation) else None}


def analyze(args):
    begin, finish = window_bounds(*args.window, timezone=args.timezone)
    if args.capacity <= 0 or args.tdp <= 0 or not 0 < args.minimum_coverage <= 1:
        raise ValueError('库存、TDP 和覆盖率参数不在有效范围')
    jobs = load_jobs(args.root)
    objects, tables = committed_snapshot(args.root, jobs, begin, finish)
    status = Counter(o['status'] for o in objects)
    complete = status['done'] == len(objects)
    if not args.allow_partial and not complete:
        raise ValueError(f'窗口未完成: {status["done"]}/{len(objects)}；部分数据诊断须显式使用 --allow-partial')
    hourly = allocations(jobs, objects, begin, finish)
    if hourly.allocated_gpu_count.max() > args.capacity + 1e-8:
        raise ValueError('Slurm 分配超过指定 GPU 库存，需核实库存及记录重叠')
    devices, accepted = device_hours(tables, begin, finish)
    valid = devices.loc[~devices.ambiguous & devices.power_valid_us.gt(0)]
    observed = valid.groupby('hour_us', as_index=False).agg(
        power_valid_us=('power_valid_us', 'sum'), power_time_sum_us=('power_time_sum_us', 'sum'),
        observed_device_count=('gpu_index', 'size'), sum_device_mean_power_W=('mean_power_W', 'sum'))
    ambiguous = devices.groupby('hour_us').ambiguous.sum().rename('ambiguous_device_hours')
    hourly = hourly.merge(observed, on='hour_us', how='left').merge(ambiguous, on='hour_us', how='left')
    # 覆盖计数的零表示没有观测；实测功率缺测保留为空。
    for column in ('power_valid_us', 'observed_device_count', 'ambiguous_device_hours'):
        hourly[column] = hourly[column].fillna(0)
    hourly['observed_gpu_equivalent'] = hourly.power_valid_us / HOUR_US
    hourly['monitoring_coverage'] = hourly.observed_gpu_equivalent / hourly.allocated_gpu_count.replace(0, np.nan)
    if hourly.monitoring_coverage.max() > 1 + 1e-8:
        raise ValueError('实测有效 GPU 小时超过分配量，需要核实设备身份及时间重叠')
    hourly['file_coverage'] = hourly.file_available_gpu_count / hourly.allocated_gpu_count.replace(0, np.nan)
    hourly['observed_energy_equivalent_W'] = hourly.power_time_sum_us / HOUR_US
    hourly['cluster_device_time_coverage'] = hourly.observed_gpu_equivalent / args.capacity
    hourly['whole_cluster_measured_W'] = hourly.observed_energy_equivalent_W.where(
        hourly.cluster_device_time_coverage.ge(1 - 1e-12) & hourly.ambiguous_device_hours.eq(0))
    split = begin + (finish - begin) / 2
    calibration = hourly.hour_utc.lt(split)
    calibration_us = float(hourly.loc[calibration, 'power_valid_us'].sum())
    if not calibration_us:
        raise ValueError('前半月缺少功率标定数据')
    allocated_watts = float(hourly.loc[calibration, 'power_time_sum_us'].sum() / calibration_us)
    ratio = allocated_watts / args.tdp
    hist = tables['joint_hist']
    idle_files = accepted.loc[accepted.job_start.ge(begin.timestamp()) & accepted.job_end.le(split.timestamp())]
    idle = hist.loc[hist.seq.isin(idle_files.index) & hist.within_allocation & hist.util_bin.eq(0)
                    & hist.month.eq(begin.strftime('%Y-%m'))]
    idle_us = int(idle.duration_us.sum())
    idle_watts = args.idle_watts
    if idle_watts is None:
        if not idle_us:
            raise ValueError('前半月没有可用于空载代理的零利用率观测；请提供有依据的 --idle-watts')
        idle_watts = float(((idle.power_bin * 5 + 2.5) * idle.duration_us).sum() / idle_us)
    if not np.isfinite(idle_watts) or idle_watts < 0:
        raise ValueError('空载功率参数无效')
    hourly['estimated_allocated_W'] = hourly.allocated_gpu_count * ratio * args.tdp
    hourly['estimated_unallocated_idle_W'] = (args.capacity - hourly.allocated_gpu_count) * idle_watts
    hourly['estimated_cluster_W'] = hourly.estimated_allocated_W + hourly.estimated_unallocated_idle_W
    hourly['estimated_on_observed_support_W'] = hourly.observed_gpu_equivalent * ratio * args.tdp
    hourly['monitoring_qualified'] = hourly.monitoring_coverage.ge(args.minimum_coverage) & hourly.ambiguous_device_hours.eq(0)
    hourly['measured_plus_idle_proxy_W'] = (hourly.observed_energy_equivalent_W + hourly.estimated_unallocated_idle_W).where(hourly.monitoring_qualified)
    hourly['allocated_error_W'] = (hourly.estimated_allocated_W - hourly.observed_energy_equivalent_W).where(hourly.monitoring_qualified)
    hourly['cluster_proxy_error_W'] = hourly.estimated_cluster_W - hourly.measured_plus_idle_proxy_W
    hourly['evaluation'] = hourly.hour_utc.ge(split)
    local = hourly.hour_utc.dt.tz_convert(args.shape_timezone)
    hourly['hour_of_day'], hourly['day_of_week'] = local.dt.hour, local.dt.dayofweek
    shapes = {}
    shape_metrics = {}
    for name, dimension in [('diurnal', 'hour_of_day'), ('weekly', 'day_of_week')]:
        selected = hourly.loc[hourly.evaluation & hourly.monitoring_qualified]
        shape = selected.groupby(dimension, as_index=False).agg(
            hours=('hour_us', 'size'), measured_W=('observed_energy_equivalent_W', 'mean'),
            estimated_W=('estimated_allocated_W', 'mean'), coverage=('monitoring_coverage', 'mean'),
            measured_plus_idle_proxy_W=('measured_plus_idle_proxy_W', 'mean'), estimated_cluster_W=('estimated_cluster_W', 'mean'))
        for column in ('measured_W', 'estimated_W', 'measured_plus_idle_proxy_W', 'estimated_cluster_W'):
            shape[column + '_normalized'] = shape[column] / shape[column].mean()
        shape_metrics[name] = metrics(shape.measured_W_normalized, shape.estimated_W_normalized)
        shapes[name] = shape
        diagnostic = hourly.loc[hourly.evaluation & hourly.observed_gpu_equivalent.gt(0)].groupby(dimension, as_index=False).agg(
            hours=('hour_us', 'size'), measured_W=('observed_energy_equivalent_W', 'mean'),
            estimated_W=('estimated_on_observed_support_W', 'mean'), coverage=('monitoring_coverage', 'mean'))
        for column in ('measured_W', 'estimated_W'):
            diagnostic[column + '_normalized'] = diagnostic[column] / diagnostic[column].mean()
        shape_metrics[name + '_observed_support_diagnostic'] = metrics(diagnostic.measured_W_normalized, diagnostic.estimated_W_normalized)
        shapes[name + '-observed-support'] = diagnostic
    output = args.output or args.root / 'analysis/priority-month'
    output.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output / 'hourly-cluster-power.csv', index=False)
    devices.to_parquet(output / 'device-hours.parquet', index=False, compression='zstd')
    for name, shape in shapes.items():
        shape.to_csv(output / (name + '-shape.csv'), index=False)
    evaluation = hourly.evaluation & hourly.monitoring_qualified
    support_evaluation = hourly.evaluation & hourly.observed_gpu_equivalent.gt(0)
    summary = {'created_at': utc(), 'window_start_utc': begin.isoformat(), 'window_end_utc': finish.isoformat(),
        'partial_data': not complete, 'completed_objects': status['done'], 'total_objects': len(objects),
        'failed_objects': status['failed'], 'hours': len(hourly), 'allocated_gpu_hours': float(hourly.allocated_gpu_count.sum()),
        'file_coverage': float(hourly.file_available_gpu_count.sum() / hourly.allocated_gpu_count.sum()),
        'completed_file_gpu_hour_share': float(hourly.file_completed_gpu_count.sum() / hourly.allocated_gpu_count.sum()),
        'measured_gpu_hours': float(hourly.observed_gpu_equivalent.sum()),
        'monitoring_coverage': float(hourly.observed_gpu_equivalent.sum() / hourly.allocated_gpu_count.sum()),
        'hours_with_observations': int(hourly.observed_gpu_equivalent.gt(0).sum()),
        'qualified_hours': int(hourly.monitoring_qualified.sum()),
        'fully_measured_cluster_hours': int(hourly.whole_cluster_measured_W.notna().sum()),
        'ambiguous_device_hours': int(devices.ambiguous.sum()),
        'excluded_file_eligibility': dict(Counter(tables['files'].loc[~tables['files'].seq.isin(accepted.index)].eligibility)),
        'observed_file_gpu_types': dict(Counter(tables['files'].gpu_type)),
        'capacity_gpus': args.capacity, 'capacity_source': CAPACITY_SOURCE,
        'tdp_W': args.tdp, 'tdp_source': TDP_SOURCE, 'tdp_interpretation': 'V100 PCIe 250 W 为归一化参考；源文件未声明实际功率上限，ratio×同一TDP代数抵消',
        'calibration_end_utc': split.isoformat(), 'allocated_power_W_per_gpu': allocated_watts,
        'measured_power_to_tdp_ratio': ratio, 'calibration_gpu_hours': calibration_us / HOUR_US,
        'idle_W_per_gpu': idle_watts, 'idle_calibration_gpu_hours': idle_us / HOUR_US,
        'idle_method': '用户提供参数' if args.idle_watts is not None else '前半月内完整作业的零GPU利用率功率5W直方图箱中心加权均值；未分配GPU空载代理',
        'idle_bin_uncertainty_W': None if args.idle_watts is not None else 2.5,
        'minimum_monitoring_coverage': args.minimum_coverage, 'shape_timezone': args.shape_timezone,
        'held_out_allocated_comparison': metrics(hourly.loc[evaluation, 'observed_energy_equivalent_W'], hourly.loc[evaluation, 'estimated_allocated_W']),
        'held_out_cluster_with_idle_proxy_comparison': metrics(hourly.loc[evaluation, 'measured_plus_idle_proxy_W'], hourly.loc[evaluation, 'estimated_cluster_W']),
        'held_out_observed_support_diagnostic': metrics(hourly.loc[support_evaluation, 'observed_energy_equivalent_W'], hourly.loc[support_evaluation, 'estimated_on_observed_support_W']),
        'held_out_shapes': shape_metrics,
        'whole_cluster_comparison': metrics(hourly.loc[hourly.evaluation, 'whole_cluster_measured_W'], hourly.loc[hourly.evaluation, 'estimated_cluster_W']),
        'limitations': ['作业监控未覆盖持续未分配GPU；全文件完成仍不保证整个物理集群纯实测功率可获得',
            'observed_energy_equivalent_W为有效观测能量/整小时，缺测未填零；sum_device_mean_power_W仅对出现的设备按各自覆盖区间求均值后相加',
            '相同node×gpu_index的跨作业疑似重叠设备小时排除；小时批次无法恢复样本并集，须原始区间才能进一步消除歧义',
            '前半月标定，后半月验证；部分完成文件依旧存在大小与随机顺序造成的选择偏差',
            '空载代理来自已分配但GPU利用率为零的时段，不能证明未分配GPU具有相同功率',
            '日内/周内形状只使用后半月监控覆盖达阈值的相同小时，缺少组合保持空白']}
    atomic_json(output / 'summary.json', summary)
    atomic_json(output / 'snapshot.json', {'created_at': summary['created_at'], 'objects': objects})
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--window', nargs=2, default=['2021-06-01', '2021-07-01'])
    parser.add_argument('--timezone', default='UTC')
    parser.add_argument('--shape-timezone', default='UTC')
    parser.add_argument('--capacity', type=int, default=448)
    parser.add_argument('--tdp', type=float, default=250)
    parser.add_argument('--idle-watts', type=float)
    parser.add_argument('--minimum-coverage', type=float, default=.99)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--wait-for-completion', action='store_true')
    parser.add_argument('--sync', action='store_true')
    args = parser.parse_args()
    output = args.output or args.root / 'analysis/priority-month'
    output.mkdir(parents=True, exist_ok=True)
    watcher_lock = process_lock(args.root / 'priority-month-watcher.lock') if args.wait_for_completion else None
    try:
        if args.wait_for_completion:
            begin, finish = window_bounds(*args.window, timezone=args.timezone)
            jobs = load_jobs(args.root)
            while True:
                db = sqlite3.connect((args.root / 'state.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
                rows = list(db.execute('SELECT key,status FROM objects'))
                db.close()
                states = Counter(status for key, status in rows if overlaps(identity(key, jobs)[2], begin, finish))
                atomic_json(output / 'watcher.json', {'updated_at': utc(), 'states': dict(states)})
                if states['failed']:
                    raise RuntimeError('窗口存在失败对象，月度分析等待已终止')
                if states['done'] and not states['pending']:
                    break
                time.sleep(30)
            del jobs, rows
        lock = process_lock(args.root / 'priority-month-analysis.lock')
        try:
            (output / 'verification.json').unlink(missing_ok=True)
            analyze(args)
            from verify_mit_gpu_priority_month import verify
            verify(args.root, output)
            if args.sync:
                from stream_mit_gpu_telemetry import synchronize
                args.sync_host = 'pku-test'
                args.sync_dir = '/data3/dataset/ljt/mit-supercloud-gpu-summary-20261028'
                synchronize(args, force=True)
        finally:
            lock.close()
    finally:
        if watcher_lock is not None:
            watcher_lock.close()


if __name__ == '__main__':
    main()
