import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from joint_capacity_inputs import atomic_json, sha256


def aggregate(source, output, scenario):
    path = source / f'{scenario}.parquet'
    record_path = output / f'{scenario}.aggregation.json'
    if record_path.is_file():
        existing = json.loads(record_path.read_text('utf-8'))
        if (existing['input_sha256'] == sha256(path) and
                existing['output_sha256'] == sha256(output / f'{scenario}.parquet')):
            print(existing, flush=True)
            return
    jobs = pd.read_parquet(path).sort_values(['job', 'source_hour_index'])
    codes, labels = pd.factorize(jobs.job, sort=False)
    first = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    last = np.r_[first[1:] - 1, len(jobs) - 1]
    width = last - first + 1
    values = jobs[['cpu_units', 'gpu_units']].to_numpy()
    amplitude = np.maximum.reduceat(values.max(axis=1), first)
    divisor = np.where(amplitude > 0, amplitude, 1.)[:, None]
    head = values[first] / divisor
    tail = values[last] / divisor * (width > 1)[:, None]
    interior = values[(first + 1).clip(max=len(jobs) - 1)] / divisor * (width > 2)[:, None]
    keys = pd.MultiIndex.from_arrays([jobs.source_hour_index.to_numpy()[first],
        jobs.source_hour_index.to_numpy()[last], *(np.round(np.column_stack((head, interior, tail)), 12).T)])
    groups, _ = pd.factorize(keys, sort=False)
    mapping = pd.DataFrame({'job': labels, 'cohort': groups, 'amplitude': amplitude,
                             'job_count': jobs.job_count.to_numpy()[first]})
    weighted = jobs[['source_hour_index', 'cpu_units', 'gpu_units', 'mw']].copy()
    weighted[['cpu_units', 'gpu_units', 'mw']] = weighted[['cpu_units', 'gpu_units', 'mw']].mul(jobs.job_count.to_numpy(), axis=0)
    weighted['job'] = groups[codes]
    combined = weighted.groupby(['job', 'source_hour_index'], as_index=False)[['cpu_units', 'gpu_units', 'mw']].sum()
    combined['job_count'] = 1
    output.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output / f'{scenario}.parquet', index=False)
    mapping.to_parquet(output / f'{scenario}.mapping.parquet', index=False)
    record = {'scenario': scenario, 'original_groups': len(labels), 'lp_cohorts': int(groups.max() + 1),
              'represented_jobs': int(mapping.job_count.sum()), 'normalized_shape_decimal_places': 12,
              'lp_only': True, 'lower_bound_checked_with_original_jobs': True,
              'integer_plan_checked_with_original_jobs': True, 'input_sha256': sha256(path),
              'output_sha256': sha256(output / f'{scenario}.parquet')}
    atomic_json(output / f'{scenario}.aggregation.json', record)
    print(record, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenario', required=True)
    args = parser.parse_args()
    aggregate(args.source, args.output, args.scenario)


if __name__ == '__main__':
    main()
