import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import tarfile

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv


END, WIDTH = 8 * 86400, 300
SCHEMA = 'task_name instance_num job_name task_type status start end plan_cpu plan_mem'.split()


def scan(project):
    path = project / 'data/raw/alibaba-cpu-2018/batch_task.tar.gz'
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    histogram = np.zeros(END // WIDTH)
    starts_count = np.zeros_like(histogram, dtype=np.int64)
    starting_instances = np.zeros_like(histogram)
    rows, valid_rows = 0, 0
    with tarfile.open(path, 'r|gz') as archive:
        members = 0
        for member in archive:
            if not member.isfile():
                continue
            assert Path(member.name).name == 'batch_task.csv' and members == 0
            members += 1
            reader = pacsv.open_csv(archive.extractfile(member),
                read_options=pacsv.ReadOptions(column_names=SCHEMA, block_size=8 * 1024 * 1024, use_threads=False),
                convert_options=pacsv.ConvertOptions(include_columns=['instance_num', 'start', 'end', 'plan_cpu'],
                    column_types={name: pa.float64() for name in ('instance_num', 'start', 'end', 'plan_cpu')}))
            for batch in reader:
                count, start, end, cpu = [batch.column(index).to_numpy(zero_copy_only=False) for index in range(4)]
                rows += len(start)
                good = np.isfinite(start) & np.isfinite(end) & np.isfinite(cpu) & np.isfinite(count) & (start >= 0) & (end > start) & (cpu >= 0) & (count > 0)
                valid_rows += int(good.sum())
                a, b, weight = np.maximum(start[good], 0), np.minimum(end[good], END), count[good] * cpu[good] / 100
                count = count[good]
                keep = b > a
                a, b, weight, count = a[keep], b[keep], weight[keep], count[keep]
                first = (a // WIDTH).astype(int)
                last = (np.nextafter(b, -np.inf) // WIDTH).astype(int)
                starts_count += np.bincount(first, minlength=len(histogram))
                starting_instances += np.bincount(first, weights=count, minlength=len(histogram))
                same = first == last
                histogram += np.bincount(first[same], weights=(b[same] - a[same]) * weight[same] / 3600, minlength=len(histogram))
                f, l, w, left, right = first[~same], last[~same], weight[~same], a[~same], b[~same]
                histogram += np.bincount(f, weights=((f + 1) * WIDTH - left) * w / 3600, minlength=len(histogram))
                histogram += np.bincount(l, weights=(right - l * WIDTH) * w / 3600, minlength=len(histogram))
                delta = np.bincount(f + 1, weights=w, minlength=len(histogram) + 1) - np.bincount(l, weights=w, minlength=len(histogram) + 1)
                histogram += delta.cumsum()[:-1] * WIDTH / 3600
        assert members == 1
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == '7c4b32361bd1ec2083647a8f52a6854a03bc125ca5c202652316c499fbf978c6'
    assert before == (path.stat().st_size, path.stat().st_mtime_ns)
    return {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source_file': str(path),
        'source_sha256': digest.hexdigest(), 'source_bytes': before[0], 'source_mtime_ns': before[1],
        'source_unchanged': True, 'raw_rows': rows, 'valid_envelope_rows': valid_rows,
        'scope': '任务开始事件和计划请求包络，不等同于实例实际活跃时长或CPU利用率；不作因果归因',
        'peak_process_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'five_minute': [{'relative_second': index * WIDTH, 'relative_day': index // 288,
            'hour_of_day': (index % 288) / 12, 'task_start_records': int(starts_count[index]),
            'instances_in_starting_task_records': float(starting_instances[index]),
            'planned_envelope_core_hours': float(histogram[index])} for index in range(len(histogram))]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='只读扫描原始 batch_task，输出5分钟计划包络与启动事件聚合')
    parser.add_argument('--project', type=Path, default=Path('/data3/dataset/ljt/global-grid-flexibility-20261023'))
    args = parser.parse_args()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    print(json.dumps(scan(args.project), ensure_ascii=False, allow_nan=False))
