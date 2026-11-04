import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


METRICS = ['utilization_gpu_pct', 'utilization_memory_pct', 'power_draw_W']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--refresh-snapshot', action='store_true')
    args = parser.parse_args()
    private = args.root / 'derived/other-refine-20261101'
    private.mkdir(parents=True, exist_ok=True)
    snapshot_path = private / 'existing-telemetry-snapshot.csv'
    expected = {}
    if snapshot_path.exists() and not args.refresh_snapshot:
        snapshot = pd.read_csv(snapshot_path)
        expected = dict(zip(snapshot.batch, snapshot.manifest_sha256))
        paths = [args.root / 'batches' / name / 'manifest.json' for name in snapshot.batch]
    else:
        paths = sorted(path for path in (args.root / 'batches').glob('*/manifest.json')
                       if re.fullmatch(r'\d{9}-\d{9}', path.parent.name))
    selected = {}
    snapshots = []
    for index, manifest_path in enumerate(paths):
        content = manifest_path.read_bytes()
        manifest = json.loads(content)
        folder = manifest_path.parent
        if folder.name in expected and hashlib.sha256(content).hexdigest() != expected[folder.name]:
            raise ValueError(f'已记录的监测批次发生变化: {folder.name}')
        snapshots.append({'batch': folder.name, 'created_at': manifest['created_at'],
                          'manifest_sha256': hashlib.sha256(content).hexdigest()})
        files_path = folder / 'files.parquet'
        if not files_path.exists():
            continue
        files = pd.read_parquet(files_path, columns=['seq', 'job_id', 'node', 'key', 'bytes',
                                                     'mapping_status', 'eligibility', 'clock_status'])
        for row in files.itertuples(index=False):
            previous = selected.get(row.seq)
            if previous is None or previous[0] < manifest['created_at']:
                selected[row.seq] = (manifest['created_at'], folder.name, row._asdict())
        if (index + 1) % 500 == 0:
            print(f'Existing telemetry manifests {index+1}/{len(paths)}', flush=True)
    folders = defaultdict(list)
    for seq, (_, folder, row) in selected.items():
        if row['mapping_status'] == 'matched' and row['eligibility'] == 'eligible':
            folders[folder].append(seq)
    accum = defaultdict(lambda: defaultdict(float))
    cols = ['seq', 'job_id', 'within_allocation']
    for metric in METRICS:
        cols.extend([metric + suffix for suffix in ['_valid_us', '_time_sum_us', '_time_sumsq_us']])
    for index, (name, seqs) in enumerate(folders.items()):
        folder = args.root / 'batches' / name
        hourly = pd.read_parquet(folder / 'hourly.parquet', columns=cols)
        hourly = hourly[hourly.seq.isin(seqs) & hourly.within_allocation]
        numeric = [col for col in cols if col not in ['seq', 'job_id', 'within_allocation']]
        for job, values in hourly.groupby('job_id')[numeric].sum().iterrows():
            for key, value in values.items():
                accum[int(job)][key] += value
        hist = pd.read_parquet(folder / 'util_hist.parquet',
                               columns=['seq', 'job_id', 'within_allocation', 'util_value', 'duration_us'])
        hist = hist[hist.seq.isin(seqs) & hist.within_allocation & hist.util_value.le(5)]
        for job, duration in hist.groupby('job_id').duration_us.sum().items():
            accum[int(job)]['idle_us'] += duration
        if (index + 1) % 500 == 0:
            print(f'Existing telemetry aggregates {index+1}/{len(folders)}', flush=True)
    rows = []
    for job, values in accum.items():
        row = {'id_job': job}
        for metric in METRICS:
            duration = values[metric + '_valid_us']
            mean = values[metric + '_time_sum_us'] / duration if duration > 0 else np.nan
            variance = values[metric + '_time_sumsq_us'] / duration - mean ** 2 if duration > 0 else np.nan
            row[metric + '_mean'] = mean
            row[metric + '_std'] = np.sqrt(max(0, variance)) if np.isfinite(variance) else np.nan
            row[metric + '_observed_gpu_hours'] = duration / 3_600_000_000
        row['idle_fraction_le5pct'] = values['idle_us'] / values['utilization_gpu_pct_valid_us']
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_parquet(private / 'existing-job-telemetry.parquet', index=False)
    pd.DataFrame(snapshots).to_csv(snapshot_path, index=False)
    pd.DataFrame([{'metric': 'snapshot_batches', 'value': len(paths)},
                  {'metric': 'unique_source_files', 'value': len(selected)},
                  {'metric': 'eligible_source_files', 'value': sum(map(len, folders.values()))},
                  {'metric': 'jobs_with_existing_telemetry', 'value': len(frame)},
                  {'metric': 'new_raw_gpu_files_downloaded', 'value': 0},
                  {'metric': 'memory_used_MiB_available_in_existing_aggregates', 'value': 0}]).to_csv(
        args.output / 'existing-telemetry-extraction.csv', index=False)
    print(f'Existing telemetry: {len(selected)} unique source files, {len(frame)} jobs', flush=True)


if __name__ == '__main__':
    main()
