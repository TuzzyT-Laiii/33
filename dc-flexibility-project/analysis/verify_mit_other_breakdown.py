import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    private = args.reference.parent / 'derived/other-refine-20261101'
    derived = pd.read_parquet(private / 'other-job-categories.parquet')
    if derived.id_job.duplicated().any() or len(derived) != 76404:
        raise ValueError('本机逐作业结果计数或唯一性错误')
    lookup = derived.set_index('id_job')[['time_start', 'time_end', 'gpus', 'gpu_hours']].to_dict('index')
    with (args.reference / 'labelled_jobids.csv').open(newline='', encoding='utf-8') as handle:
        label_rows = list(csv.DictReader(handle))
    labels = {int(row['id_job']): row['model'] for row in label_rows}
    if len(labels) != len(label_rows):
        raise ValueError('标签ID重复')
    counts, label_counts = Counter(), Counter()
    hours, label_hours = defaultdict(float), defaultdict(float)
    all_rows = 0
    gpu_ids = re.compile(r'(?:^|,)(1001|1002)=(\d+)(?=,|$)')
    # 使用标准CSV库独立逐行累计，不读取主分析的逐作业中间对象。
    with (args.reference / 'slurm-log.csv').open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            all_rows += 1
            if sum(int(value) for _, value in gpu_ids.findall(row['tres_req'])) == 0:
                continue
            allocated = sum(int(value) for _, value in gpu_ids.findall(row['tres_alloc']))
            duration = int(row['time_end']) - int(row['time_start'])
            if duration < 0 or allocated <= 0 or int(row['time_suspended']) != 0:
                raise ValueError('GPU执行区间无效')
            resource_hours = allocated * duration / 3600
            if row['job_type'] == 'OTHER':
                result = lookup.pop(int(row['id_job']))
                if result['time_start'] != int(row['time_start']) or result['time_end'] != int(row['time_end']) or result['gpus'] != allocated:
                    raise ValueError('逐作业结果改变了原始执行区间或GPU数量')
                if not math.isclose(result['gpu_hours'], resource_hours, abs_tol=1e-9, rel_tol=0):
                    raise ValueError('逐作业GPU小时核验失败')
            counts[row['job_type']] += 1
            hours[row['job_type']] += resource_hours
            if int(row['id_job']) in labels:
                label_counts[row['job_type']] += 1
                label_hours[row['job_type']] += resource_hours
    if all_rows != 395914 or sum(counts.values()) != 98176 or counts['OTHER'] != 76404:
        raise ValueError('独立逐行计数与PR #26不同')
    if label_counts != Counter({'OTHER': 3430}):
        raise ValueError('独立逐行标签匹配与交付结果不同')
    if lookup:
        raise ValueError('逐作业结果包含源数据之外的作业')
    if not math.isclose(sum(hours.values()), 1460190.6641666666, abs_tol=1e-6, rel_tol=0):
        raise ValueError('独立GPU小时分母核验失败')
    if not math.isclose(label_hours['OTHER'], 40071.8875, abs_tol=1e-6, rel_tol=0):
        raise ValueError('独立标签GPU小时核验失败')
    saved = pd.read_csv(args.output / 'other-breakdown.csv')
    if saved.rows.sum() != counts['OTHER']:
        raise ValueError('最终类别计数不守恒')
    if not math.isclose(saved.gpu_hours.sum(), hours['OTHER'], abs_tol=1e-5, rel_tol=0):
        raise ValueError('最终类别GPU小时不守恒')
    if not math.isclose(saved.pct_other_gpu_hours.sum(), 100, abs_tol=1e-8, rel_tol=0):
        raise ValueError('最终类别占比不守恒')
    actual = derived.groupby('category_id').agg(rows=('id_job', 'size'), gpu_hours=('gpu_hours', 'sum')).sort_index()
    expected = saved.set_index('category_id')[['rows', 'gpu_hours']].sort_index()
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), atol=1e-5, rtol=0)
    cross = pd.read_csv(args.output / 'category-terminal-crosswalk.csv').groupby('category_id')[['rows', 'gpu_hours']].sum().sort_index()
    np.testing.assert_allclose(cross.to_numpy(), expected.to_numpy(), atol=1e-5, rtol=0)
    if derived[derived.model.notna()].category_id.ne('labelled_dnn').any():
        raise ValueError('官方标签作业未保留在标签类别')
    if set(derived['classification_version']) != {'mit-other-refine-20261101-v2'}:
        raise ValueError('逐作业版本标识错误')
    folds = pd.read_csv(args.output / 'label-holdout-validation.csv')
    random = folds[folds.protocol.eq('stratified_random_5fold') & folds.method.eq('combined_high_or_medium')]
    if random.held_jobs.sum() != 3430 or (folds.recovered_jobs > folds.held_jobs).any():
        raise ValueError('留出检验样本计数错误')
    cost = pd.read_csv(args.output / 'gpu-telemetry-download-cost.csv').set_index('stratum').loc['TOTAL']
    if cost['bytes'] <= 30_000_000_000 or int(cost['files']) != 12286:
        raise ValueError('监测样本成本核验失败')
    for path in args.output.glob('*.csv'):
        columns = pd.read_csv(path, nrows=0).columns
        if set(columns) & {'id_job', 'id_user', 'id_array_job', 'id_array_task', 'nodelist'}:
            raise ValueError(f'汇总文件包含逐作业标识: {path}')
    print('独立逐行核验通过：76404条执行区间与GPU数、资源守恒、标签、类别及终态交叉、留出计数和公开文件范围。')


if __name__ == '__main__':
    main()
