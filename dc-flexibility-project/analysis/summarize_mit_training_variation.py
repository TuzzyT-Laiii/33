import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main(args):
    jobs = pd.read_parquet(args.private / 'mit-job-power.parquet')
    hours = pd.read_parquet(args.private / 'mit-labelled-job-hours.parquet').merge(jobs, on='id_job', validate='many_to_one')
    hours['hour_mean_w'] = hours.power_draw_W_time_sum_us / hours.power_draw_W_valid_us
    hours['relative_to_job_mean'] = hours.hour_mean_w / hours.power_mean_w
    time = pd.to_datetime(hours.hour_us, unit='us', utc=True)
    hours['hour_new_york'] = time.dt.tz_convert('America/New_York').dt.hour
    hours['source_day'] = time.dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
    hours['coverage_qualified'] = hours.coverage.between(.8, 1.01)
    hourly_midpoint = (np.maximum(hours.hour_us / 1e6, hours.start_epoch) + np.minimum(hours.hour_us / 1e6 + 3600, hours.end_epoch)) / 2
    hours['progress_bin'] = np.minimum(9, np.floor(10 * (hourly_midpoint - hours.start_epoch) / (hours.end_epoch - hours.start_epoch))).astype(int)
    cohorts = {'official_DNN': jobs.model.notna(), 'NLP_finetuning': jobs.family.eq('NLP_finetuning'),
        'CNN': jobs.family.eq('CNN'), 'GNN': jobs.family.eq('GNN'), 'all_observed_V100': jobs.id_job.notna()}
    summaries, progress, diurnal = [], [], []
    for name, mask in cohorts.items():
        selected = jobs.loc[mask & jobs.coverage.between(.8, 1.01)]
        if selected.empty:
            continue
        part = hours.loc[hours.id_job.isin(selected.id_job)].copy() if name != 'all_observed_V100' else hours.iloc[:0].copy()
        multi = part.groupby('id_job').hour_us.nunique()
        long = part.loc[part.id_job.isin(multi.index[multi >= 4])].sort_values(['id_job', 'hour_us'])
        adjacent = long.groupby('id_job').hour_us.diff().eq(3.6e9)
        changes = long.groupby('id_job').hour_mean_w.diff().abs().loc[adjacent] / 250
        record = {'cohort': name, 'coverage_qualified_jobs': len(selected), 'observed_gpu_hours': float(selected.gpu_hours.sum()),
            'qualified_mean_power_tdp_ratio': float(np.average(selected.power_mean_w, weights=selected.gpu_hours) / 250),
            'median_job_cv': float((selected.power_std_w / selected.power_mean_w).median()),
            'jobs_with_at_least4_hour_bins': int(long.id_job.nunique()) if name != 'all_observed_V100' else None,
            'adjacent_hour_pairs': len(changes) if name != 'all_observed_V100' else None,
            'median_absolute_adjacent_hour_change_tdp': float(changes.median()) if len(changes) else None,
            'p90_absolute_adjacent_hour_change_tdp': float(changes.quantile(.9)) if len(changes) else None}
        summaries.append(record)
        if not part.empty:
            grouped = part.groupby('hour_new_york').agg(relative_mean=('relative_to_job_mean', 'mean'),
                job_hour_observations=('id_job', 'size'), distinct_jobs=('id_job', 'nunique'), distinct_days=('source_day', 'nunique')).reset_index()
            diurnal.append(grouped.assign(cohort=name))
        if not long.empty:
            within = long.groupby(['id_job', 'progress_bin']).relative_to_job_mean.mean().reset_index()
            grouped = within.groupby('progress_bin').agg(relative_power_mean=('relative_to_job_mean', 'mean'),
                relative_power_p10=('relative_to_job_mean', lambda x: x.quantile(.1)),
                relative_power_p90=('relative_to_job_mean', lambda x: x.quantile(.9)),
                contributing_jobs=('id_job', 'nunique')).reset_index()
            progress.append(grouped.assign(cohort=name))
    pd.DataFrame(summaries).to_csv(args.output / 'mit-coverage-variation.csv', index=False)
    pd.concat(diurnal, ignore_index=True).to_csv(args.output / 'mit-diurnal-job-standardized.csv', index=False)
    pd.concat(progress, ignore_index=True).to_csv(args.output / 'mit-within-job-progress.csv', index=False)
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
