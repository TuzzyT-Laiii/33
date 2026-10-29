import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from mit_gpu_common import atomic_json, load_jobs, utc


def verify(root, output):
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    snapshot = json.loads((output / 'snapshot.json').read_text(encoding='utf-8'))
    hourly = pd.read_csv(output / 'hourly-cluster-power.csv')
    devices = pd.read_parquet(output / 'device-hours.parquet')
    begin = pd.Timestamp(summary['window_start_utc']).timestamp()
    finish = pd.Timestamp(summary['window_end_utc']).timestamp()
    jobs = load_jobs(root)
    events = defaultdict(float)
    for job in jobs.values():
        if job['start'] <= 0 or job['end'] <= job['start'] or not job['allocated_gpus']:
            continue
        start, end = max(job['start'], begin), min(job['end'], finish)
        if start < end:
            events[start] += job['allocated_gpus']
            events[end] -= job['allocated_gpus']
    for boundary in range(int(begin), int(finish) + 1, 3600):
        events[boundary] += 0
    assigned = np.zeros(len(hourly))
    running = 0.0
    previous = begin
    for timestamp, change in sorted(events.items()):
        if timestamp > previous:
            position = int((previous - begin) // 3600)
            assigned[position] += running * (timestamp - previous) / 3600
        running += change
        previous = timestamp
    assert np.allclose(assigned, hourly.allocated_gpu_count, rtol=1e-12, atol=1e-8)
    assert abs(running) < 1e-8
    assert len(hourly) == int((finish - begin) / 3600)
    assert not hourly.hour_us.duplicated().any()
    assert not devices.duplicated(['hour_us', 'node', 'gpu_index']).any()
    clean = devices.loc[~devices.ambiguous & devices.power_valid_us.gt(0)]
    duration = clean.groupby('hour_us').power_valid_us.sum()
    energy = clean.groupby('hour_us').power_time_sum_us.sum() / 3_600_000_000
    actual = hourly.set_index('hour_us')
    assert np.allclose(actual.loc[duration.index, 'observed_gpu_equivalent'], duration / 3_600_000_000)
    assert np.allclose(actual.loc[energy.index, 'observed_energy_equivalent_W'], energy, rtol=1e-12, atol=1e-8)
    assert hourly.loc[hourly.observed_gpu_equivalent.eq(0), 'observed_energy_equivalent_W'].isna().all()
    assert np.allclose(hourly.estimated_cluster_W, hourly.allocated_gpu_count * summary['measured_power_to_tdp_ratio'] * summary['tdp_W']
                       + (summary['capacity_gpus'] - hourly.allocated_gpu_count) * summary['idle_W_per_gpu'])
    assert hourly.loc[hourly.cluster_device_time_coverage.lt(1 - 1e-12), 'whole_cluster_measured_W'].isna().all()
    qualified = hourly.monitoring_qualified
    assert np.allclose(hourly.loc[qualified, 'cluster_proxy_error_W'], hourly.loc[qualified, 'allocated_error_W'])
    evaluation = hourly.loc[hourly.evaluation & hourly.observed_gpu_equivalent.gt(0)]
    error = evaluation.estimated_on_observed_support_W - evaluation.observed_energy_equivalent_W
    diagnostic = summary['held_out_observed_support_diagnostic']
    assert len(evaluation) == diagnostic['hours']
    if len(evaluation):
        assert np.isclose(error.abs().mean(), diagnostic['MAE_W'])
        assert np.isclose(np.sqrt((error ** 2).mean()), diagnostic['RMSE_W'])
    if diagnostic['pearson_r'] is not None:
        assert np.isclose(np.corrcoef(evaluation.observed_energy_equivalent_W, evaluation.estimated_on_observed_support_W)[0, 1], diagnostic['pearson_r'])
    for name in ('diurnal', 'weekly'):
        shape = pd.read_csv(output / (name + '-observed-support-shape.csv'))
        assert int(shape.hours.sum()) == len(evaluation)
        if len(shape):
            assert np.isclose(shape.measured_W_normalized.mean(), 1)
            assert np.isclose(shape.estimated_W_normalized.mean(), 1)
    db = sqlite3.connect((root / 'state.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
    committed = [o for o in snapshot['objects'] if o['status'] == 'done']
    for obj in committed:
        status, batch, attempts = db.execute('SELECT status,batch,attempts FROM objects WHERE seq=?', (obj['seq'],)).fetchone()
        assert (status, batch, attempts) == ('done', obj['batch'], obj['attempts'])
    db.close()
    report = {'verified_at': utc(), 'input_analysis_created_at': summary['created_at'], 'completed_objects': len(committed),
        'checks': ['独立事件积分与逐小时分配一致', '窗口小时数和唯一性', 'node×gpu_index合并唯一性',
                   '设备时长和功率积分守恒', '缺测功率保持空白', '分配×比例×TDP+未分配空载公式',
                   '未完整监控的整集群实测保持空白', '验证集误差和相关性独立重算',
                   '同一空载代理下的整集群与分配功率误差一致',
                   '日内/周内共同观测范围计数与归一化', '分析快照已完成对象状态保持']}
    atomic_json(output / 'verification.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    verify(args.root, args.output or args.root / 'analysis/priority-month')
