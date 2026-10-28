import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import requests


BASE = 'https://mit-supercloud-dataset.s3.us-west-2.amazonaws.com/'
PREFIX = 'datacenter-challenge/202201/'
NS = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
PART_BYTES = 8 * 1024 * 1024


def acquire(root, work):
    root.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    response = requests.get(BASE, params={'list-type': '2', 'prefix': PREFIX, 'delimiter': '/'}, timeout=60)
    response.raise_for_status()
    listing = ET.fromstring(response.content)
    if listing.findtext('s:IsTruncated', namespaces=NS) != 'false':
        raise ValueError('顶层对象清单不完整')
    objects = {item.findtext('s:Key', namespaces=NS).removeprefix(PREFIX):
               {'bytes': int(item.findtext('s:Size', namespaces=NS)),
                'etag': item.findtext('s:ETag', namespaces=NS).strip('"')}
               for item in listing.findall('s:Contents', NS)}
    response = requests.get(BASE, params={'list-type': '2', 'prefix': PREFIX + 'gpu/0000/', 'max-keys': 100}, timeout=60)
    response.raise_for_status()
    samples = {}
    for item in ET.fromstring(response.content).findall('s:Contents', NS):
        size = int(item.findtext('s:Size', namespaces=NS))
        name = item.findtext('s:Key', namespaces=NS).removeprefix(PREFIX)
        if name.endswith('.csv') and 0 < size <= 256 * 1024:
            samples[name] = {'bytes': size, 'etag': item.findtext('s:ETag', namespaces=NS).strip('"')}
            if len(samples) == 2:
                break
    if len(samples) != 2:
        raise ValueError('当前 GPU 监测目录没有获得两份范围内的小样本')
    objects.update(samples)
    records = []
    for name in ['LICENSE', 'README.md', 'tres-mapping.txt', 'slurm-log.csv', *samples]:
        info = objects[name]
        if info['bytes'] > 128 * 1024 * 1024:
            raise ValueError('所需对象超出本次 128 MiB 单文件下载范围')
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        url = BASE + PREFIX + name
        started = time.monotonic()
        if not path.exists():
            ranges = [(i, min(i + PART_BYTES, info['bytes']) - 1) for i in range(0, info['bytes'], PART_BYTES)]

            def fetch(bounds):
                begin, end = bounds
                chunk = work / f'{name}.{begin}.part'
                chunk.parent.mkdir(parents=True, exist_ok=True)
                if chunk.exists() and chunk.stat().st_size == end - begin + 1:
                    return chunk
                response = requests.get(url, headers={'Range': f'bytes={begin}-{end}', 'If-Match': '"' + info['etag'] + '"'}, timeout=(30, 180))
                response.raise_for_status()
                if response.status_code != 206 or len(response.content) != end - begin + 1:
                    raise ValueError('HTTP Range 响应长度或状态错误')
                if response.headers.get('Content-Range') != f'bytes {begin}-{end}/{info["bytes"]}':
                    raise ValueError('HTTP Range 响应边界错误')
                chunk.write_bytes(response.content)
                print(json.dumps({'file': name, 'range_start': begin, 'range_bytes': len(response.content)}), flush=True)
                return chunk

            with ThreadPoolExecutor(max_workers=8) as pool:
                chunks = list(pool.map(fetch, ranges))
            partial = work / (name + '.assembled')
            partial.parent.mkdir(parents=True, exist_ok=True)
            with partial.open('wb') as stream:
                for chunk in chunks:
                    with chunk.open('rb') as source:
                        shutil.copyfileobj(source, stream)
            shutil.move(str(partial), str(path))
        if path.stat().st_size != info['bytes']:
            raise ValueError('原始文件长度与 S3 清单不一致: ' + name)
        digest = hashlib.sha256()
        part_md5 = []
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(PART_BYTES), b''):
                digest.update(block)
                part_md5.append(hashlib.md5(block).digest())
        actual_etag = (hashlib.md5(b''.join(part_md5)).hexdigest() + '-' + str(len(part_md5))
                       if '-' in info['etag'] else hashlib.md5(path.read_bytes()).hexdigest())
        if actual_etag != info['etag']:
            raise ValueError('S3 ETag 与完整原始对象校验不一致: ' + name)
        records.append({'file': name, 'url': url, **info, 'sha256': digest.hexdigest(),
                        'etag_verified': True, 'seconds': time.monotonic() - started})
        print(json.dumps(records[-1]), flush=True)
    result = {'source': 'MIT Supercloud 202201', 'license': 'CC-BY-NC-ND-4.0',
              'files': records, 'downloaded_bytes': sum(r['bytes'] for r in records),
              'raw_data_location': str(root), 'telemetry_sample_files': list(samples),
              'full_cluster_telemetry_downloaded': False}
    (root / 'acquisition.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    acquire(args.raw_dir, args.work_dir)
