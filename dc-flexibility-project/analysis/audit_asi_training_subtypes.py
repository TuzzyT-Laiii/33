import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def run(root, output):
    output.mkdir(parents=True, exist_ok=True)
    pieces, sources = [], []
    dimensions = ['job_type_public', 'model_type_public', 'priority_class']
    for hour in range(24):
        path = root / f'data/raw/task-trace-audit-20261026/asi_opensource_pod_hourly/day=109/hour={hour:02d}/part-000.parquet'
        if output.resolve() == path.parent.resolve() or output.resolve() in path.resolve().parents:
            raise ValueError('输出与原始数据目录必须分开')
        stat = path.stat()
        table = pq.read_table(path, columns=dimensions + ['used_gpu_hours'],
                              filters=[('gpu_request', '>', 0)], use_threads=False).to_pandas()
        summary = table.groupby(dimensions, dropna=False).agg(
            pod_hour_observations=('used_gpu_hours', 'size'), used_gpu_hours=('used_gpu_hours', 'sum')).reset_index()
        pieces.append(summary)
        sources.append({'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
        print(f'ASI model types hour {hour:02d}', flush=True)
    combined = pd.concat(pieces).groupby(dimensions, dropna=False)[['pod_hour_observations', 'used_gpu_hours']].sum().reset_index()
    published = pd.read_csv(root / 'data/processed/task-trace-audit-20261026/asi-hourly-type-priority.csv')
    np.testing.assert_allclose(combined.groupby('job_type_public').used_gpu_hours.sum().sort_index(),
                               published.groupby('job_type_public').used_gpu_hours.sum().sort_index(), rtol=1e-10)
    training = combined[combined.job_type_public.eq('training')].groupby('model_type_public')[['pod_hour_observations', 'used_gpu_hours']].sum().reset_index()
    training['gpu_hour_fraction_within_training'] = training.used_gpu_hours / training.used_gpu_hours.sum()
    for entry in sources:
        stat = Path(entry['path']).stat()
        if stat.st_size != entry['bytes'] or stat.st_mtime_ns != entry['mtime_ns']:
            raise ValueError('分析期间输入元数据发生变化')
    combined.to_csv(output / 'asi-model-type-priority.csv', index=False)
    training.to_csv(output / 'asi-training-model-types.csv', index=False)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_metadata': sources, 'source_sha256_record': 'server-audit.json 的同24份pod输入',
        'validation': {'type_gpu_hours_conserved': 'passed', 'source_metadata_unchanged': 'passed'},
        'scope': '模型领域标签；genai、rec等不能区分预训练、SFT或RL算法；仅发布聚合'}
    (output / 'asi-training-model-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    run(args.input_root, args.output)
