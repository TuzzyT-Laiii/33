import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compute_load.model import sha256


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


class Distribution:
    def __init__(self, upper, step):
        self.upper, self.step = upper, step
        self.hist = np.zeros(int(upper / step) + 1, dtype=np.int64)
        self.total = self.valid = self.missing = self.invalid = 0
        self.sum = self.sumsq = 0.
        self.minimum, self.maximum = np.inf, -np.inf
        self.zero = self.over_tdp = 0

    def add(self, values, tdp=None):
        self.total += values.size
        finite = np.isfinite(values)
        self.missing += int(np.isnan(values).sum())
        good = finite & (values >= 0) & (values <= self.upper)
        self.invalid += int((~good & ~np.isnan(values)).sum())
        selected = values[good].astype(np.float64)
        if selected.size:
            self.valid += selected.size
            self.sum += selected.sum()
            self.sumsq += np.square(selected).sum()
            self.minimum = min(self.minimum, float(selected.min()))
            self.maximum = max(self.maximum, float(selected.max()))
            self.zero += int((selected == 0).sum())
            self.over_tdp += int((selected > tdp).sum()) if tdp else 0
            self.hist += np.bincount(np.floor(selected / self.step).astype(int), minlength=len(self.hist))
        return np.where(good, values, np.nan)

    def result(self):
        if not self.valid:
            raise ValueError('监测列没有有效数值')
        cdf = self.hist.cumsum()
        quantiles = {f'p{int(q * 100):02d}': min(self.upper, (np.searchsorted(cdf, q * self.valid, side='left') + .5) * self.step)
                     for q in (.01, .05, .10, .25, .50, .75, .90, .95, .99)}
        return {'cells': self.total, 'valid_cells': self.valid, 'missing_cells': self.missing,
                'invalid_cells': self.invalid, 'valid_fraction': self.valid / self.total,
                'mean': self.sum / self.valid, 'std': float(np.sqrt(max(0, self.sumsq / self.valid - (self.sum / self.valid) ** 2))),
                'min': self.minimum, 'max': self.maximum, 'zero_fraction': self.zero / self.valid,
                'above_400W_fraction': self.over_tdp / self.valid if self.upper == 2000 else None,
                'quantile_bin_width': self.step, **quantiles}


def metric_file(path, output, chunk_rows):
    cluster, metric = path.parent.name, path.stem
    setting = {'POWER_USAGE': (2000, 1, 'W'), 'GPU_UTIL': (100, 1, '%'), 'FB_USED': (100000, 64, 'MiB')}[metric]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if header[0] != 'Time' or any(name in header for name in ('job_id', 'type')):
        raise ValueError('GPU 监测字段与预期不符')
    dtype = {column: np.float32 for column in header[1:]}
    dist = Distribution(*setting[:2])
    daily, hourly = defaultdict(lambda: [0., 0, 0]), defaultdict(lambda: [0., 0, 0])
    first = last = previous = None
    rows = nonpositive_steps = irregular_steps = 0
    max_gap = 0.
    for frame in pd.read_csv(path, chunksize=chunk_rows, dtype=dtype):
        times = pd.to_datetime(frame.pop('Time'), utc=True, errors='raise')
        if times.isna().any():
            raise ValueError('时间字段缺失')
        if first is None:
            first = times.iloc[0]
        stamps = times.astype('int64').to_numpy()
        steps = np.diff(np.r_[previous, stamps] if previous is not None else stamps) / 1e9
        nonpositive_steps += int((steps <= 0).sum())
        irregular_steps += int((steps != 15).sum())
        if len(steps):
            max_gap = max(max_gap, float(steps.max()))
        previous, last = stamps[-1], times.iloc[-1]
        values = dist.add(frame.to_numpy(), 400 if metric == 'POWER_USAGE' else None)
        sums = np.nansum(values, axis=1, dtype=np.float64)
        counts = np.isfinite(values).sum(axis=1)
        local = pd.DatetimeIndex(times).tz_convert('Asia/Shanghai')
        for labels, target in ((local.strftime('%Y-%m-%d'), daily), (local.hour, hourly)):
            for label in np.unique(labels):
                mask = labels == label
                target[str(label)][0] += float(sums[mask].sum())
                target[str(label)][1] += int(counts[mask].sum())
                target[str(label)][2] += int(mask.sum())
        rows += len(frame)
    record = {'cluster': cluster, 'metric': metric, 'unit': setting[2], 'devices': len(header) - 1,
              'rows': rows, 'start_utc': first.isoformat(), 'end_utc': last.isoformat(),
              'nonpositive_time_steps': nonpositive_steps, 'non_15s_time_steps': irregular_steps,
              'maximum_gap_s': max_gap, 'sha256': sha256(path), **dist.result(),
              'population': 'all monitored devices, allocated and unallocated mixed; sample weighted; no job join',
              'sampling_note': 'No imputation or gap holding; quantiles are histogram bin midpoints; all valid values within declared physical screening range retained.'}
    if metric == 'POWER_USAGE':
        record['mean_to_a100_400W_tdp'] = record['mean'] / 400
    write_json(output / f'{cluster}-{metric}-summary.json', record)
    for label, table in (('daily', daily), ('diurnal', hourly)):
        pd.DataFrame([{'cluster': cluster, 'metric': metric, label: key, 'mean': item[0] / item[1] if item[1] else np.nan,
                       'valid_cells': item[1], 'rows': item[2], 'possible_cells': item[2] * (len(header) - 1)}
                      for key, item in sorted(table.items())]).to_csv(output / f'{cluster}-{metric}-{label}.csv', index=False)
    pd.DataFrame({'bin_lower': np.arange(len(dist.hist)) * dist.step, 'count': dist.hist}).query('count > 0').to_csv(
        output / f'{cluster}-{metric}-histogram.csv', index=False)
    print(cluster, metric, rows, record['mean'], flush=True)
    return record


def ipmi_file(path, output, chunk_rows):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    columns = [name for name in header if name not in ('Time', 'node') and not name.startswith('Unnamed:')]
    distributions = {name: Distribution(20000, 5) for name in columns}
    rows = 0
    for frame in pd.read_csv(path, chunksize=chunk_rows, usecols=columns, dtype=np.float32):
        for name in columns:
            distributions[name].add(frame[name].to_numpy())
        rows += len(frame)
    records = [{'source': path.stem, 'field': name, 'unit': 'W', 'rows': rows, **dist.result()}
               for name, dist in distributions.items() if dist.valid]
    pd.DataFrame(records).to_csv(output / f'{path.stem}-summary.csv', index=False)
    write_json(output / f'{path.stem}-metadata.json', {'sha256': sha256(path), 'rows': rows,
               'columns': header, 'omitted_all_missing_fields': [name for name, dist in distributions.items() if not dist.valid],
               'interpretation': 'Mixed running states; node mappings withheld from aggregates; Sys_Total_Power, PSU input and component DC are separate meters and must not be summed.'})
    print(path.stem, rows, flush=True)


def jobs(root, output):
    records, audit = [], []
    for cluster in ('kalos', 'seren'):
        path = root / 'github/data/job_trace' / f'trace_{cluster}.csv'
        frame = pd.read_csv(path)
        if frame.job_id.isna().any() or frame.job_id.duplicated().any():
            raise ValueError('作业标识缺失或重复')
        raw_fields = frame.columns.tolist()
        raw_nulls = frame.isna().sum().astype(int).to_dict()
        total = frame.gpu_time.sum()
        runtime = (pd.to_datetime(frame.end_time, utc=True) - pd.to_datetime(frame.start_time, utc=True)).dt.total_seconds()
        runtime_gpu_hours = runtime.clip(lower=0) * frame.gpu_num / 3600
        frame['timestamp_gpu_hours'] = runtime_gpu_hours
        gpu_rows = frame.gpu_num > 0
        for kind, part in frame.groupby('type', dropna=False):
            monitoring_status = 'unknown: no job/device allocation join' if (part.gpu_num > 0).any() else 'not_applicable: no GPUs requested'
            records.append({'cluster': cluster, 'type': kind if pd.notna(kind) else '<null>', 'jobs': len(part),
                            'gpu_jobs': int((part.gpu_num > 0).sum()), 'gpu_hours': part.gpu_time.sum() / 3600,
                            'gpu_time_share_pct': 100 * part.gpu_time.sum() / total,
                            'start_end_gpu_hours': part.timestamp_gpu_hours.sum(),
                            'start_end_gpu_time_share_pct': 100 * part.timestamp_gpu_hours.sum() / runtime_gpu_hours.sum(),
                            'allocated_gpu_power': monitoring_status, 'gpu_utilization': monitoring_status})
        audit.append({'cluster': cluster, 'rows': len(frame), 'unique_job_ids': frame.job_id.nunique(),
                      'type_nulls': int(frame.type.isna().sum()), 'fields': raw_fields,
                      'field_null_counts': raw_nulls,
                      'gpu_time_equals_gpu_num_times_duration_on_gpu_jobs': bool(np.allclose(frame.loc[gpu_rows, 'gpu_time'], (frame.gpu_num * frame.duration)[gpu_rows])),
                      'duration_start_end_disagreements': int((runtime.notna() & (runtime != frame.duration)).sum()),
                      'missing_end_timestamps': int(runtime.isna().sum()),
                      'negative_start_end_durations': int((runtime < 0).sum()),
                      'gpu_time_definition': 'official gpu_time / 3600; separate start_end_gpu_hours computed from nonnegative end_time-start_time, without filling missing intervals',
                      'sha256': sha256(path), 'gpu_hours': float(total / 3600)})
    result = pd.DataFrame(records)
    if result.jobs.sum() != sum(item['rows'] for item in audit):
        raise ValueError('作业分类数量不守恒')
    result.to_csv(output / 'job-type-summary.csv', index=False)
    write_json(output / 'job-label-audit.json', audit)


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.jobs_only:
        jobs(args.raw.resolve(), output)
        return
    paths = sorted((args.raw / 'hf/data/utilization').rglob('*.csv'))
    if args.metric:
        paths = [path for path in paths if path.stem == args.metric]
    if not args.available and not args.metric and len(paths) != 9:
        raise ValueError(f'需要九份完整监测文件，当前只有 {len(paths)} 份')
    records = []
    for path in paths:
        target = output / f'{path.parent.name}-{path.stem}-summary.json'
        if path.parent.name == 'ipmi':
            if not (output / f'{path.stem}-metadata.json').exists():
                ipmi_file(path, output, args.chunk_rows)
        elif target.exists():
            records.append(json.loads(target.read_text('utf-8')))
        else:
            records.append(metric_file(path, output, args.chunk_rows))
    pd.DataFrame(records).to_csv(output / 'gpu-telemetry-summary.csv', index=False)
    write_json(output / 'telemetry-run.json', {'utc': datetime.now(timezone.utc).isoformat(), 'files': len(paths),
               'complete': len(paths) == 9, 'chunk_rows': args.chunk_rows,
               'scope': 'CC BY 4.0 AcmeTrace official monitoring; raw samples and device identifiers remain outside Git',
               'source_files': {str(path.relative_to(args.raw)): sha256(path) for path in paths}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk-rows', type=int, default=2048)
    parser.add_argument('--jobs-only', action='store_true')
    parser.add_argument('--available', action='store_true')
    parser.add_argument('--metric', choices=['POWER_USAGE', 'GPU_UTIL', 'FB_USED'])
    main(parser.parse_args())
