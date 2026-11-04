import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import tarfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv


SCHEMA = 'machine_id timestamp cpu mem mem_gps mkpi net_in net_out disk'.split()
META = 'machine_id timestamp failure_domain_1 failure_domain_2 cpu_num mem_size status'.split()
END = 8 * 86400
ABSENT, INVALID, MISSING = 255, 253, 254


class DigestStream:
    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()
        self.bytes = 0

    def read(self, size=-1):
        block = self.handle.read(size)
        self.digest.update(block)
        self.bytes += len(block)
        return block


def file_signature(path):
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def histogram_quantile(hist, probability):
    result = np.full(len(hist), np.nan)
    for index, counts in enumerate(hist):
        total = counts.sum()
        if total:
            rank = probability * (total - 1)
            lower = np.searchsorted(counts.cumsum(), np.floor(rank) + 1)
            upper = np.searchsorted(counts.cumsum(), np.ceil(rank) + 1)
            result[index] = lower + (upper - lower) * (rank - np.floor(rank))
    return result


def bin_intervals(starts, ends, weights, seconds):
    # 前一采样保持到下一采样，最多300秒；以真实重叠秒数累计，不填补长缺口。
    first = (starts // seconds).astype(int)
    boundary = (first + 1) * seconds
    first_seconds = np.minimum(ends, boundary) - starts
    output = np.bincount(first, weights=weights * first_seconds, minlength=END // seconds)
    crossed = ends > boundary
    output += np.bincount(first[crossed] + 1, weights=weights[crossed] * (ends[crossed] - boundary[crossed]), minlength=len(output))
    return output


def scan_machine(project):
    raw = project / 'data/raw/alibaba-cpu-2018'
    source_audit = json.loads((project / 'results/alibaba-cpu-cld-20261031/source-audit.json').read_text())
    machine_path = raw / 'machine_usage.tar.gz'
    metadata_path = raw / 'machine_meta.tar.gz'
    signatures = [file_signature(path) for path in (machine_path, metadata_path)]
    with tarfile.open(metadata_path, 'r:gz') as archive:
        members = [item for item in archive.getmembers() if item.isfile()]
        assert len(members) == 1
        metadata = pd.read_csv(archive.extractfile(members[0]), names=META)
    assert metadata.groupby('machine_id').cpu_num.nunique().eq(1).all()
    inventory = metadata.groupby('machine_id', sort=True).cpu_num.first()
    assert inventory.gt(0).all()
    machines, slots, bins = len(inventory), END // 10 + 1, END // 300
    ids = pa.array(inventory.index.tolist())
    cores = inventory.to_numpy(float)
    matrix = np.full((machines, slots), ABSENT, dtype=np.uint8)
    histogram = np.zeros((bins, 101), dtype=np.int64)
    native_count = np.zeros(slots, dtype=np.int64)
    native_sum = np.zeros(slots)
    counts = Counter()
    options = pacsv.ConvertOptions(include_columns=['machine_id', 'timestamp', 'cpu'],
        column_types={'machine_id': pa.string(), 'timestamp': pa.float64(), 'cpu': pa.float64()})
    last_progress = 0
    with machine_path.open('rb') as file:
        digest = DigestStream(file)
        with tarfile.open(fileobj=digest, mode='r|gz') as archive:
            members = 0
            for member in archive:
                if not member.isfile():
                    continue
                assert Path(member.name).name == 'machine_usage.csv' and members == 0
                members += 1
                reader = pacsv.open_csv(archive.extractfile(member),
                    read_options=pacsv.ReadOptions(column_names=SCHEMA, block_size=8 * 1024 * 1024, use_threads=False),
                    convert_options=options)
                for batch in reader:
                    times = batch.column(1).to_numpy(zero_copy_only=False)
                    cpu = batch.column(2).to_numpy(zero_copy_only=False)
                    mapped = pc.index_in(batch.column(0), value_set=ids)
                    assert mapped.null_count == 0
                    machine = mapped.to_numpy(zero_copy_only=False).astype(np.int64)
                    assert np.isfinite(times).all() and (times >= 0).all() and (times < END).all() and (times % 10 == 0).all()
                    tick = (times / 10).astype(np.int64)
                    valid = np.isfinite(cpu) & (cpu >= 0) & (cpu <= 100)
                    assert (cpu[valid] % 1 == 0).all()
                    counts['rows'] += len(cpu)
                    counts['cpu_missing'] += int((~np.isfinite(cpu)).sum())
                    counts['cpu_below_zero'] += int((cpu < 0).sum())
                    counts['cpu_above_100'] += int((cpu > 100).sum())
                    value = np.full(len(cpu), INVALID, dtype=np.uint8)
                    value[~np.isfinite(cpu)] = MISSING
                    value[valid] = cpu[valid].astype(np.uint8)
                    position = machine * slots + tick
                    unique, take, inverse = np.unique(position, return_index=True, return_inverse=True)
                    assert np.array_equal(value, value[take][inverse]), '同机器同时间存在冲突值'
                    old = matrix.ravel()[unique]
                    assert np.array_equal(old[old != ABSENT], value[take][old != ABSENT]), '跨批次重复记录冲突'
                    fresh = take[old == ABSENT]
                    counts['duplicate_rows'] += len(cpu) - len(fresh)
                    matrix.ravel()[position[fresh]] = value[fresh]
                    fresh = fresh[valid[fresh]]
                    native_count += np.bincount(tick[fresh], minlength=slots)
                    native_sum += np.bincount(tick[fresh], weights=cpu[fresh], minlength=slots)
                    location = (times[fresh] // 300).astype(int) * 101 + cpu[fresh].astype(int)
                    histogram += np.bincount(location, minlength=bins * 101).reshape(bins, 101)
                    if counts['rows'] - last_progress >= 20000000:
                        print(json.dumps({'stage': 'raw_machine_usage', 'rows': counts['rows']}), file=sys.stderr, flush=True)
                        last_progress = counts['rows']
            assert members == 1
        while digest.read(1024 * 1024):
            pass
        archive_sha = digest.digest.hexdigest()
        archive_bytes = digest.bytes
    assert archive_sha == '3e6ee87fd204bb85b9e234c5c75a5096580fdabc8f085b224033080090753a7a'
    assert archive_bytes == machine_path.stat().st_size
    assert counts['rows'] == source_audit['counts']['raw_rows']
    occupied = np.zeros(bins)
    exposure = np.zeros(bins)
    observed = np.zeros(bins, dtype=int)
    hour_means = np.full((machines, 192), np.nan)
    hold_sensitivity = {str(hold): {'occupied': np.zeros(192), 'exposure': np.zeros(192)} for hold in (10, 30, 60, 300)}
    for machine, row in enumerate(matrix):
        ticks = np.flatnonzero(row != ABSENT)
        starts = ticks * 10
        following = np.r_[starts[1:], END]
        good = row[ticks] <= 100
        ends = np.minimum(following, starts + 300)
        weights = row[ticks][good].astype(float) / 100 * cores[machine]
        a, b = starts[good], ends[good]
        current = bin_intervals(a, b, weights, 300)
        coverage = bin_intervals(a, b, np.full(len(a), cores[machine]), 300)
        occupied += current
        exposure += coverage
        observed[np.unique((starts[good] // 300).astype(int))] += 1
        numerator = current.reshape(192, 12).sum(axis=1)
        denominator = coverage.reshape(192, 12).sum(axis=1)
        hour_means[machine] = np.divide(numerator, denominator, out=np.full(192, np.nan), where=denominator > 0) * 100
        for hold, data in hold_sensitivity.items():
            capped = np.minimum(following, starts + int(hold))[good]
            data['occupied'] += bin_intervals(a, capped, weights, 3600)
            data['exposure'] += bin_intervals(a, capped, np.full(len(a), cores[machine]), 3600)
    five = pd.DataFrame({'relative_second': np.arange(bins) * 300,
        'relative_day': np.arange(bins) // 288, 'hour_of_day': np.arange(bins) % 288 / 12,
        'observed_machines': observed, 'valid_samples': histogram.sum(axis=1),
        'coverage_fraction': exposure / (cores.sum() * 300),
        'mean_occupied_cores': occupied / 300, 'cpu_time_weighted_pct': occupied / exposure * 100,
        'cpu_raw_sample_mean_pct': histogram @ np.arange(101) / histogram.sum(axis=1)})
    for q in (.05, .25, .5, .75, .95):
        five[f'raw_cpu_p{int(q*100):02d}_pct'] = histogram_quantile(histogram, q)
    hourly = pd.DataFrame({'relative_hour': np.arange(192), 'relative_day': np.arange(192) // 24, 'hour_of_day': np.arange(192) % 24,
        'occupied_core_hours': occupied.reshape(192, 12).sum(axis=1) / 3600,
        'observed_core_hours': exposure.reshape(192, 12).sum(axis=1) / 3600})
    hourly['coverage_fraction'] = hourly.observed_core_hours / cores.sum()
    hourly['cpu_util_time_weighted_pct'] = hourly.occupied_core_hours / hourly.observed_core_hours * 100
    for q in (.05, .25, .5, .75, .95):
        hourly[f'machine_hour_cpu_p{int(q*100):02d}_pct'] = np.nanquantile(hour_means, q, axis=0)
    reference = pd.read_csv(project / 'results/alibaba-cpu-cld-20261031/alibaba-hourly-cpu-demand.csv')
    for column in ('occupied_core_hours', 'observed_core_hours', 'cpu_util_time_weighted_pct'):
        np.testing.assert_allclose(hourly[column], reference[column], rtol=1e-9, atol=1e-7)
    selected = (np.arange(slots) >= 2*8640 + 4*360) & (np.arange(slots) < 2*8640 + 9*360)
    time = np.arange(slots)[selected] * 10
    native = pd.DataFrame({'relative_second': time, 'hour_of_day': (time - 2*86400) / 3600,
        'reporting_machines': native_count[selected],
        'raw_cpu_mean_pct': native_sum[selected] / native_count[selected]})
    changes = hour_means[:, 54] - hour_means[:, 53]
    pair = np.isfinite(changes)
    sensitivity = [{'maximum_hold_seconds': int(hold), 'relative_hour': hour,
        'cpu_util_pct': float(data['occupied'][hour] / data['exposure'][hour] * 100),
        'coverage_fraction': float(data['exposure'][hour] / 3600 / cores.sum())}
        for hold, data in hold_sensitivity.items() for hour in (53, 54, 55)]
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'counts': dict(counts),
        'archive_sha256': archive_sha, 'archive_bytes': archive_bytes,
        'inventory_machines': machines, 'inventory_cores': float(cores.sum()),
        'cores_per_machine': sorted(inventory.unique().astype(float).tolist()),
        'source_files_before': signatures, 'source_files_after': [file_signature(Path(item['path'])) for item in signatures],
        'raw_hourly_matches_existing': True,
        'peak_process_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'paired_hour_5_to_6': {'machines': int(pair.sum()), 'increased': int((changes[pair] > 0).sum()),
            'increase_at_least_10pp': int((changes[pair] >= 10).sum()),
            'median_increase_pp': float(np.median(changes[pair]))},
        'hold_sensitivity': sensitivity, 'hourly': json.loads(hourly.to_json(orient='records')),
        'five_minute': json.loads(five.to_json(orient='records')),
        'native_10second_day2_h4_to_h9': json.loads(native.to_json(orient='records'))}
    assert result['source_files_before'] == result['source_files_after']
    return result


def main():
    parser = argparse.ArgumentParser(description='只读扫描 Alibaba 原始机器 CPU，输出聚合 JSON 到标准输出')
    parser.add_argument('--project', type=Path, default=Path('/data3/dataset/ljt/global-grid-flexibility-20261023'))
    args = parser.parse_args()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    print(json.dumps(scan_machine(args.project), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
