import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

from audit_research_gpu_demand import tres_count


BASE = 'https://mit-supercloud-dataset.s3.us-west-2.amazonaws.com/'
PREFIX = 'datacenter-challenge/202201/gpu/'
NS = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}


def list_directory(directory, proxy):
    session = requests.Session()
    session.trust_env = False
    session.proxies = {'http': proxy, 'https': proxy}
    token = None
    rows = []
    while True:
        params = {'list-type': 2, 'prefix': f'{PREFIX}{directory:04d}/', 'max-keys': 1000}
        if token:
            params['continuation-token'] = token
        response = session.get(BASE, params=params, timeout=(30, 120))
        response.raise_for_status()
        tree = ET.fromstring(response.content)
        for item in tree.findall('s:Contents', NS):
            key = item.findtext('s:Key', namespaces=NS)
            match = re.fullmatch(r'(\d+)-(r\d+-n\d+)\.csv', Path(key).name)
            if match is None:
                raise ValueError(f'未知GPU文件名: {key}')
            rows.append({'id_job': int(match[1]), 'node': match[2], 'key': key,
                         'bytes': int(item.findtext('s:Size', namespaces=NS)),
                         'etag': item.findtext('s:ETag', namespaces=NS)})
        token = tree.findtext('s:NextContinuationToken', namespaces=NS)
        if not token:
            break
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--proxy', default='http://127.0.0.1:7897')
    parser.add_argument('--per-stratum', type=int, default=200)
    args = parser.parse_args()
    private = args.root / 'derived/other-refine-20261101'
    private.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    path = private / 'gpu-s3-inventory.parquet'
    if path.exists():
        inventory = pd.read_parquet(path)
    else:
        entries = []
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=6) as executor:
            for index, result in enumerate(executor.map(lambda d: list_directory(d, args.proxy), range(100))):
                entries.extend(result)
                if (index + 1) % 10 == 0:
                    print(f'S3 metadata: {index+1}/100 directories, {len(entries)} objects', flush=True)
        inventory = pd.DataFrame(entries)
        if inventory.key.duplicated().any():
            raise ValueError('S3清单键重复')
        inventory.to_parquet(path, index=False)
        print(f'S3 metadata completed in {time.perf_counter()-started:.1f}s', flush=True)
    labels = pd.read_csv(args.root / 'reference/labelled_jobids.csv')
    parts = []
    for chunk in pd.read_csv(args.root / 'reference/slurm-log.csv', chunksize=20000,
                             usecols=['id_job', 'job_type', 'tres_req', 'tres_alloc', 'nodes_alloc',
                                      'time_submit', 'time_start', 'time_end', 'id_array_task', 'timelimit']):
        requested = sum(tres_count(chunk.tres_req, key).fillna(0) for key in [1001, 1002])
        chosen = chunk[requested.gt(0) & chunk.job_type.eq('OTHER')].copy()
        chosen['gpus'] = sum(tres_count(chosen.tres_alloc, key).fillna(0) for key in [1001, 1002])
        parts.append(chosen.drop(columns=['tres_req', 'tres_alloc']))
    jobs = pd.concat(parts, ignore_index=True)
    runtime = jobs.time_end - jobs.time_start
    wait = jobs.time_start - jobs.time_submit
    jobs['stratum'] = np.select([
        jobs.id_job.isin(labels.id_job), jobs.id_array_task.ne(0xfffffffe),
        jobs.nodes_alloc.ge(2) | jobs.gpus.ge(8), runtime.le(600),
        wait.le(2) & runtime.lt(21600) & jobs.timelimit.le(720), runtime.ge(21600)],
        ['labelled_dnn', 'array', 'large', 'short', 'interactive_candidate', 'long'], default='remaining')
    jobs['gpu_hours'] = runtime * jobs.gpus / 3600
    selected = []
    for name, group in jobs.groupby('stratum'):
        if name == 'labelled_dnn':
            sample = group
        else:
            # 固定种子分层抽样；层内按GPU小时四分位进一步平衡规模。
            group = group.copy()
            group['size_band'] = pd.qcut(group.gpu_hours.rank(method='first'), 4, labels=False)
            sample = pd.concat([band.sample(n=min(len(band), args.per_stratum // 4), random_state=20261101)
                                for _, band in group.groupby('size_band')])
        selected.append(sample[['id_job', 'stratum', 'gpu_hours']])
    sample = pd.concat(selected, ignore_index=True)
    plan = inventory.merge(sample[['id_job', 'stratum']], on='id_job', how='inner', validate='many_to_one')
    plan.to_parquet(private / 'gpu-download-plan.parquet', index=False)
    sample.to_parquet(private / 'gpu-sample-jobs.parquet', index=False)
    costs = []
    for name, group in sample.groupby('stratum'):
        objects = plan[plan.stratum.eq(name)]
        size = int(objects.bytes.sum())
        costs.append({'stratum': name, 'sample_jobs': len(group), 'matched_jobs': objects.id_job.nunique(),
                      'files': len(objects), 'bytes': size, 'decimal_GB': size / 1e9,
                      'hours_at_1_MiB_s': size / 1048576 / 3600,
                      'hours_at_3_MiB_s': size / 1048576 / 3600 / 3})
    size = int(plan.bytes.sum())
    costs.append({'stratum': 'TOTAL', 'sample_jobs': len(sample), 'matched_jobs': plan.id_job.nunique(),
                  'files': len(plan), 'bytes': size, 'decimal_GB': size / 1e9,
                  'hours_at_1_MiB_s': size / 1048576 / 3600,
                  'hours_at_3_MiB_s': size / 1048576 / 3600 / 3})
    pd.DataFrame(costs).to_csv(args.output / 'gpu-telemetry-download-cost.csv', index=False)
    print(pd.DataFrame(costs).to_string(index=False), flush=True)
    (private / 'gpu-cost-decision.json').write_text(json.dumps({
        'inventory_objects': len(inventory), 'inventory_bytes': int(inventory.bytes.sum()),
        'sample_bytes': size, 'limit_bytes': 30_000_000_000,
        'download_authorized_by_cost': size <= 30_000_000_000,
        'inventory_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
