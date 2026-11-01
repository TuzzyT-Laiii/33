import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compute_load.model import sha256


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def family(model):
    if model in ('bert-base-uncased', 'distilbert-base-uncased'):
        return 'NLP_finetuning'
    if model in ('dimenet', 'schnet', 'conv', 'pna'):
        return 'GNN'
    return 'CNN'


def snapshot(args):
    root, output, private = args.mit_root.resolve(), args.output.resolve(), args.private.resolve()
    output.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(root / 'reference/labelled_jobids.csv')
    if labels.id_job.duplicated().any() or labels.isna().any().any():
        raise ValueError('MIT官方标签缺失或作业标识重复')
    labels['family'] = labels.model.map(family)
    pieces, hashes = [], {}
    for batch in sorted((root / 'batches').iterdir()):
        path = batch / 'files.parquet'
        if not (batch / 'manifest.json').exists() or not path.exists() or not (batch / 'hourly.parquet').exists():
            continue
        frame = pd.read_parquet(path)
        if not frame.schema_version.eq(2).all():
            raise ValueError('MIT批次版本不符合既有时区/分配审计')
        frame['batch'] = batch.name
        pieces.append(frame)
        hashes[batch.name] = sha256(path)
    files = pd.concat(pieces, ignore_index=True)
    if files.key.duplicated().any():
        duplicates = files.loc[files.key.duplicated(keep=False)]
        if duplicates.groupby('key').sha256.nunique().gt(1).any():
            raise ValueError('相同MIT对象存在不同内容的批次，须先审阅')
        files = files.drop_duplicates('key', keep='last')
    files['id_job'] = pd.to_numeric(files.job_id).astype('int64')
    files = files.merge(labels, on='id_job', how='left', validate='many_to_one')
    inventory = pd.read_parquet(root / 'derived/other-refine-20261101/gpu-s3-inventory.parquet')
    expected = inventory.loc[inventory.id_job.isin(labels.id_job)].copy()
    missing = expected.loc[~expected.key.isin(files.key)].copy()
    missing.to_parquet(private / 'missing-labelled-files.parquet', index=False)
    files.to_parquet(private / 'mit-file-snapshot.parquet', index=False)
    qualified = files.eligibility.eq('eligible') & files.gpu_type.eq('gpu:volta')
    record = {'utc': datetime.now(timezone.utc).isoformat(), 'official_label_jobs': len(labels),
        'label_models': labels.model.value_counts().astype(int).to_dict(), 'batch_count': len(hashes),
        'snapshot_files': len(files), 'snapshot_jobs': int(files.id_job.nunique()),
        'eligible_v100_files': int(qualified.sum()), 'eligible_v100_jobs': int(files.loc[qualified, 'id_job'].nunique()),
        'labelled_jobs_with_any_file': int(files.loc[files.model.notna(), 'id_job'].nunique()),
        'labelled_eligible_v100_jobs': int(files.loc[qualified & files.model.notna(), 'id_job'].nunique()),
        'labelled_expected_files': len(expected), 'labelled_expected_bytes': int(expected.bytes.sum()),
        'missing_labelled_files': len(missing), 'missing_labelled_jobs': int(missing.id_job.nunique()),
        'missing_labelled_bytes': int(missing.bytes.sum()), 'download_threshold_bytes': 30_000_000_000,
        'download_decision': 'within_threshold' if missing.bytes.sum() <= 30_000_000_000 else 'over_threshold_do_not_download',
        'eligibility_file_counts': files.eligibility.value_counts().astype(int).to_dict(),
        'source_label_sha256': sha256(root / 'reference/labelled_jobids.csv'),
        'source_inventory_sha256': sha256(root / 'derived/other-refine-20261101/gpu-s3-inventory.parquet'),
        'batch_file_table_sha256': hashes, 'read_only_source': True,
        'mapping': 'official id_job -> immutable per-job/node files -> GPU index and allocation-clipped telemetry',
        'clock': 'Existing schema v2 New York wall-clock correction against Slurm epoch; no new offset fitted.'}
    write_json(output / 'mit-input-audit.json', record)
    print(json.dumps({k: v for k, v in record.items() if k not in ('batch_file_table_sha256', 'label_models')}, ensure_ascii=False), flush=True)
    return files, labels


def aggregate(args, files, labels):
    output, private = args.output.resolve(), args.private.resolve()
    selected = files.loc[files.eligibility.eq('eligible') & files.gpu_type.eq('gpu:volta')]
    model_map = labels.set_index('id_job').model.to_dict()
    family_map = labels.set_index('id_job').family.to_dict()
    job_sum = defaultdict(lambda: np.zeros(4, dtype=float))
    hist, diurnal = defaultdict(float), defaultdict(lambda: np.zeros(3, dtype=float))
    training_hours = []
    inputs = {}
    total_hours_us = 0.
    columns = ['seq', 'job_id', 'within_allocation', 'hour_us', 'power_draw_W_valid_us',
        'power_draw_W_time_sum_us', 'power_draw_W_time_sumsq_us', 'utilization_gpu_pct_valid_us',
        'utilization_gpu_pct_time_sum_us']
    for count, (batch, metadata) in enumerate(selected.groupby('batch'), 1):
        directory = args.mit_root / 'batches' / batch
        hours = pd.read_parquet(directory / 'hourly.parquet', columns=columns)
        hours = hours.loc[hours.seq.isin(metadata.seq) & hours.within_allocation & hours.power_draw_W_valid_us.gt(0)].copy()
        hours['id_job'] = pd.to_numeric(hours.job_id).astype('int64')
        joint = pd.read_parquet(directory / 'joint_hist.parquet', columns=['seq', 'job_id', 'within_allocation', 'power_bin', 'duration_us'])
        joint = joint.loc[joint.seq.isin(metadata.seq) & joint.within_allocation]
        joint['id_job'] = pd.to_numeric(joint.job_id).astype('int64')
        # 同批次不重复文件；已有分配与设备歧义筛选在文件表中逐项保留。
        grouped = hours.groupby('id_job')[['power_draw_W_valid_us', 'power_draw_W_time_sum_us', 'power_draw_W_time_sumsq_us', 'utilization_gpu_pct_time_sum_us']].sum()
        for identifier, row in grouped.iterrows():
            job_sum[int(identifier)] += row.to_numpy()
        epoch = pd.to_datetime(hours.hour_us, unit='us', utc=True)
        hours['hour_local'] = epoch.dt.tz_convert('America/New_York').dt.hour
        hours['model'] = hours.id_job.map(model_map)
        hours['family'] = hours.id_job.map(family_map)
        subsets = {'all_observed_V100': np.ones(len(hours), bool), 'official_DNN': hours.model.notna(),
            'unlabelled': hours.model.isna()}
        subsets.update({name: hours.family.eq(name) for name in ('NLP_finetuning', 'CNN', 'GNN')})
        for cohort, mask in subsets.items():
            part = hours.loc[mask]
            for hour, row in part.groupby('hour_local')[['power_draw_W_valid_us', 'power_draw_W_time_sum_us', 'utilization_gpu_pct_time_sum_us']].sum().iterrows():
                diurnal[(cohort, int(hour))] += row.to_numpy()
            ids = set(part.id_job)
            for power_bin, duration in joint.loc[joint.id_job.isin(ids)].groupby('power_bin').duration_us.sum().items():
                hist[(cohort, int(power_bin))] += float(duration)
        known = hours.loc[hours.model.notna()].groupby(['id_job', 'hour_us'])[['power_draw_W_valid_us', 'power_draw_W_time_sum_us']].sum().reset_index()
        training_hours.append(known)
        total_hours_us += float(hours.power_draw_W_valid_us.sum())
        inputs[batch] = {'hourly': sha256(directory / 'hourly.parquet'), 'joint_hist': sha256(directory / 'joint_hist.parquet')}
        if count % 250 == 0:
            print('MIT batches', count, flush=True)
    metadata = selected.drop_duplicates('id_job').set_index('id_job')
    job_rows = []
    for identifier, a in job_sum.items():
        duration, energy, square, util = a
        row = metadata.loc[identifier]
        expected = (row.job_end - row.job_start) * row.allocated_gpus * 1e6
        mean = energy / duration
        job_rows.append({'id_job': identifier, 'model': model_map.get(identifier), 'family': family_map.get(identifier),
            'power_mean_w': mean, 'power_std_w': float(np.sqrt(max(0., square / duration - mean ** 2))),
            'gpu_hours': duration / 3.6e9, 'coverage': duration / expected if expected else np.nan,
            'start_epoch': int(row.job_start), 'end_epoch': int(row.job_end)})
    jobs = pd.DataFrame(job_rows)
    jobs.to_parquet(private / 'mit-job-power.parquet', index=False)
    pd.concat(training_hours, ignore_index=True).groupby(['id_job', 'hour_us'], as_index=False).sum().to_parquet(private / 'mit-labelled-job-hours.parquet', index=False)
    histogram = pd.DataFrame([{'cohort': cohort, 'power_bin_lower_w': power_bin * 5, 'gpu_hours': duration / 3.6e9}
        for (cohort, power_bin), duration in hist.items()])
    histogram.to_csv(output / 'mit-power-histogram.csv', index=False)
    day = pd.DataFrame([{'cohort': cohort, 'hour_new_york': hour, 'mean_power_w': a[1] / a[0],
        'power_tdp_ratio': a[1] / a[0] / 250, 'gpu_hours': a[0] / 3.6e9}
        for (cohort, hour), a in sorted(diurnal.items())])
    day.to_csv(output / 'mit-diurnal.csv', index=False)
    records = []
    cohorts = {'all_observed_V100': jobs, 'official_DNN': jobs.loc[jobs.model.notna()], 'unlabelled': jobs.loc[jobs.model.isna()]}
    cohorts.update({name: jobs.loc[jobs.family.eq(name)] for name in ('NLP_finetuning', 'CNN', 'GNN')})
    for name, part in cohorts.items():
        if part.empty:
            continue
        h = histogram.loc[histogram.cohort.eq(name)].sort_values('power_bin_lower_w')
        cumulative = h.gpu_hours.cumsum().to_numpy()
        quantiles = {f'p{int(q * 100):02d}_sample_ratio': float(h.iloc[np.searchsorted(cumulative, q * cumulative[-1])].power_bin_lower_w + 2.5) / 250 for q in (.1, .5, .9)}
        d = day.loc[day.cohort.eq(name)]
        records.append({'cohort': name, 'jobs': len(part), 'observed_gpu_hours': float(part.gpu_hours.sum()),
            'gpu_time_weighted_mean_w': float(np.average(part.power_mean_w, weights=part.gpu_hours)),
            'gpu_time_weighted_mean_ratio': float(np.average(part.power_mean_w, weights=part.gpu_hours) / 250),
            'job_equal_weight_mean_ratio': float(part.power_mean_w.mean() / 250),
            'median_job_mean_ratio': float(part.power_mean_w.median() / 250),
            'job_mean_ratio_p10': float(part.power_mean_w.quantile(.1) / 250), 'job_mean_ratio_p90': float(part.power_mean_w.quantile(.9) / 250),
            'median_observed_coverage': float(part.coverage.median()), 'coverage_at_least80pct_jobs': int(part.coverage.ge(.8).sum()),
            'median_within_job_cv': float((part.power_std_w / part.power_mean_w).median()),
            'observed_diurnal_peak_valley': float(d.mean_power_w.max() / d.mean_power_w.min()), **quantiles})
    pd.DataFrame(records).to_csv(output / 'mit-cohort-summary.csv', index=False)
    models = jobs.loc[jobs.model.notna()].groupby(['model', 'family']).apply(lambda part: pd.Series({
        'jobs': len(part), 'gpu_hours': part.gpu_hours.sum(), 'gpu_time_weighted_mean_ratio': np.average(part.power_mean_w, weights=part.gpu_hours) / 250,
        'median_job_ratio': part.power_mean_w.median() / 250, 'median_job_cv': (part.power_std_w / part.power_mean_w).median()}), include_groups=False).reset_index()
    models.to_csv(output / 'mit-model-summary.csv', index=False)
    run = {'utc': datetime.now(timezone.utc).isoformat(), 'selected_batches': len(inputs), 'jobs': len(jobs),
        'observed_gpu_hours': total_hours_us / 3.6e9, 'inputs_sha256': inputs,
        'quantiles': '5W histogram midpoints, GPU-duration weighted; sample quantiles conditional on joint-valid power and utilization',
        'job_variation': 'time-and-GPU-weighted within-job standard deviation; does not isolate time from device heterogeneity',
        'diurnal': 'observed New York hour, conditional on jobs and available device-time; not an LLM pretraining causal cycle',
        'raw_source_writes': False}
    write_json(output / 'mit-aggregation-audit.json', run)
    print(pd.DataFrame(records).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mit-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--snapshot-only', action='store_true')
    args = parser.parse_args()
    files, labels = snapshot(args)
    if not args.snapshot_only:
        aggregate(args, files, labels)
