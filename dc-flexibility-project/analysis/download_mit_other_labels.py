import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


BASE_URL = 'https://mit-supercloud-dataset.s3.us-west-2.amazonaws.com/datacenter-challenge/202201/'
FILES = ['labelled_jobids.csv', 'labelled_job_stats.csv', 'README.md', 'LICENSE', 'tres-mapping.txt']
MAX_BYTES = 2_000_000_000


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--proxy', default='http://127.0.0.1:7897')
    args = parser.parse_args()
    args.reference.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    session.proxies = {'http': args.proxy, 'https': args.proxy}
    entries = []
    # 全部文件先核对大小，超过上限时不启动任何下载。
    for name in FILES:
        url = BASE_URL + name
        with session.head(url, timeout=60, allow_redirects=True) as response:
            response.raise_for_status()
            size = int(response.headers['Content-Length'])
            if size > MAX_BYTES:
                raise ValueError(f'文件超过2 GB，停止下载: {url}, {size} bytes')
            entries.append({'file': name, 'url': response.url, 'bytes': size,
                            'etag': response.headers['ETag'],
                            'last_modified': response.headers['Last-Modified']})
    for entry in entries:
        path = args.reference / entry['file']
        # 已有说明文件也通过本次GET逐字节校验，保持原文件不变。
        partial = path.with_name(path.name + '.download-part')
        total = 0
        with session.get(entry['url'], timeout=(30, 120), stream=True,
                         headers={'If-Match': entry['etag']}) as response:
            response.raise_for_status()
            with partial.open('xb') as handle:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    total += len(block)
                    if total > min(MAX_BYTES, entry['bytes']):
                        raise ValueError(f'下载内容超过HEAD大小: {entry["file"]}')
                    handle.write(block)
        if total != entry['bytes']:
            raise ValueError(f'下载不完整: {entry["file"]}')
        entry['sha256'] = sha256(partial)
        if path.exists():
            if sha256(path) != entry['sha256']:
                raise ValueError(f'已有文件与官方版本不同，保留两份文件并停止: {path}')
            partial.unlink()
            entry['local_action'] = 'existing_file_verified_against_fresh_download'
        else:
            partial.rename(path)
            entry['local_action'] = 'downloaded'
        print(f'{entry["file"]}: {total} bytes, sha256={entry["sha256"]}', flush=True)
    manifest = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                'release': 'MIT Supercloud datacenter-challenge/202201',
                'license': 'CC-BY-NC-ND-4.0', 'max_single_file_bytes': MAX_BYTES, 'files': entries}
    (args.reference / 'other-label-acquisition-20261031.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
