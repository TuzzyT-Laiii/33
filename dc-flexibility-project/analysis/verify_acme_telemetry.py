import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


def verify(output):
    acquisition = json.loads((output / 'hf-acquisition.json').read_text('utf-8'))
    run = json.loads((output / 'telemetry-run.json').read_text('utf-8'))
    if not acquisition['complete'] or not run['complete'] or run['files'] != 9:
        raise ValueError('监测获取和统计未完整完成')
    for item in acquisition['files']:
        relative = 'hf/' + item['url'].split('/resolve/main/', 1)[1]
        if run['source_files'][relative] != item['sha256']:
            raise ValueError('下载与统计阶段的原件哈希不一致')
    records = pd.read_csv(output / 'gpu-telemetry-summary.csv')
    if len(records) != 6 or records[['cluster', 'metric']].duplicated().any():
        raise ValueError('GPU 指标组合数量不符')
    for _, row in records.iterrows():
        base = f'{row.cluster}-{row.metric}'
        if row.valid_cells + row.missing_cells + row.invalid_cells != row.cells or row.cells != row.rows * row.devices:
            raise ValueError('监测单元格数量不守恒')
        if row.nonpositive_time_steps:
            raise ValueError('监测时间存在重复或倒序，需要先处理其统计权重')
        histogram = pd.read_csv(output / f'{base}-histogram.csv')
        if histogram['count'].sum() != row.valid_cells:
            raise ValueError('直方图有效样本数不守恒')
        bin_mean = np.average(histogram.bin_lower, weights=histogram['count'])
        if not bin_mean - 1e-6 <= row['mean'] <= bin_mean + row.quantile_bin_width + 1e-6:
            raise ValueError('直方图与精确均值不相容')
        for grouping in ('daily', 'diurnal'):
            groups = pd.read_csv(output / f'{base}-{grouping}.csv')
            if groups.valid_cells.sum() != row.valid_cells or groups.rows.sum() != row.rows:
                raise ValueError('日期或日内分组样本数不守恒')
            np.testing.assert_allclose(np.average(groups['mean'], weights=groups.valid_cells), row['mean'], rtol=1e-12, atol=1e-8)
        values = row[[f'p{i:02d}' for i in (1, 5, 10, 25, 50, 75, 90, 95, 99)]].to_numpy(dtype=float)
        if (np.diff(values) < 0).any():
            raise ValueError('分位数顺序不正确')
    jobs = pd.read_csv(output / 'job-type-summary.csv')
    audits = json.loads((output / 'job-label-audit.json').read_text('utf-8'))
    if jobs.jobs.sum() != 880740 or jobs.jobs.sum() != sum(row['unique_job_ids'] for row in audits):
        raise ValueError('公开作业数量不守恒')
    gpu_jobs = jobs.loc[jobs.gpu_jobs.gt(0)]
    if not gpu_jobs.allocated_gpu_power.str.startswith('unknown:').all() or not gpu_jobs.gpu_utilization.str.startswith('unknown:').all():
        raise ValueError('缺少作业分配映射时不得声称逐作业实测')
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'telemetry_files': 9, 'gpu_metric_tables': 6,
              'source_hashes_match_acquisition': True, 'histogram_daily_diurnal_count_and_mean_conservation': True,
              'nonpositive_gpu_time_steps': 0, 'public_job_rows': 880740,
              'job_type_monitoring_join_status': 'unknown; allocation mapping absent', 'all_checks_passed': True}
    (output / 'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'results/trn-power-model-20261101/acme')
    verify(parser.parse_args().output.resolve())
