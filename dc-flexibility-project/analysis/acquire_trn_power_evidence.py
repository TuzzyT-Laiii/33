import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def download(session, url, path, direct_cdn=False):
    if path.exists():
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                digest.update(block)
        return {'url': url, 'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    response = session.get(url, stream=True, timeout=(30, 180))
    response.raise_for_status()
    part = path.with_suffix(path.suffix + '.part')
    offset = part.stat().st_size if direct_cdn and part.exists() else 0
    if direct_cdn:
        location = response.url
        response.close()
        direct = requests.Session()
        direct.trust_env = False
        response = direct.get(location, stream=True, timeout=(30, 180), headers={'Range': f'bytes={offset}-'} if offset else {})
        if offset and (response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
            raise ValueError('官方 CDN 没有确认断点续传起点')
    with response:
        response.raise_for_status()
        digest = hashlib.sha256()
        if offset:
            with part.open('rb') as existing:
                for block in iter(lambda: existing.read(4 * 1024 * 1024), b''):
                    digest.update(block)
        with part.open('ab' if offset else 'wb') as handle:
            total = offset
            for block in response.iter_content(4 * 1024 * 1024):
                handle.write(block)
                digest.update(block)
                total += len(block)
                if total % (256 * 1024 * 1024) < len(block):
                    print(f'{path.parent.name}/{path.name}: {total / 1e9:.2f} GB', flush=True)
        part.replace(path)
    return {'url': url, 'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def main(args):
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    if args.proxy:
        session.proxies.update({'https': args.proxy, 'http': args.proxy})
    if args.hf_only:
        inventory = json.loads((root / 'huggingface-tree.json').read_text('utf-8'))
        chosen = [row for row in inventory if row['type'] == 'file' and
                  (row['path'].endswith(('/POWER_USAGE.csv', '/GPU_UTIL.csv', '/FB_USED.csv')) or
                   row['path'].startswith('data/utilization/ipmi/'))]
        if any(row['size'] >= 10_000_000_000 for row in chosen):
            raise ValueError('单文件达到10GB，停止自动下载')
        print(json.dumps({'selected_files': len(chosen), 'total_bytes': sum(row['size'] for row in chosen)}), flush=True)
        def acquire(row):
            worker = requests.Session()
            worker.proxies.update(session.proxies)
            path = root / 'hf' / row['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            print('acquiring ' + row['path'], flush=True)
            item = download(worker, 'https://huggingface.co/datasets/Qinghao/AcmeTrace/resolve/main/' + row['path'], path, direct_cdn=args.direct_cdn)
            if item['bytes'] != row['size']:
                raise ValueError('下载字节数不符：' + row['path'])
            expected = row.get('lfs', {}).get('oid')
            if expected and item['sha256'] != expected:
                raise ValueError('Hugging Face LFS SHA-256 不符：' + row['path'])
            return item
        records = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(acquire, row) for row in chosen]):
                records.append(future.result())
                save_json(root / 'hf-acquisition.json', {'files': records, 'complete': len(records) == len(chosen)})
        return
    api = 'https://api.github.com/repos/InternLM/AcmeTrace'
    if args.github_tree:
        tree = json.loads(args.github_tree.read_text('utf-8-sig'))
    else:
        response = session.get(api + '/git/trees/master?recursive=1', timeout=60)
        response.raise_for_status()
        tree = response.json()
    save_json(root / 'github-tree.json', tree)
    selected = [row for row in tree['tree'] if row['type'] == 'blob' and
                (row['path'].startswith('data/utilization/') or row['path'] in
                 ('README.md', 'LICENSE.txt', 'analysis.ipynb', 'data/generate_utilization_pkl.ipynb',
                  'data/utils.py', 'data/cluster_summary.csv', 'data/job_trace/trace_kalos.csv',
                  'data/job_trace/trace_seren.csv'))]
    manifest = []
    for row in selected:
        path = root / 'github' / row['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest.append(download(session, f'https://raw.githubusercontent.com/InternLM/AcmeTrace/{tree["sha"]}/{row["path"]}', path))
    url = 'https://huggingface.co/api/datasets/Qinghao/AcmeTrace/tree/main?recursive=true&limit=1000'
    response = session.get(url, timeout=90)
    response.raise_for_status()
    inventory = response.json()
    if response.links.get('next'):
        raise ValueError('Hugging Face 清单出现分页，需要继续读取')
    save_json(root / 'huggingface-tree.json', inventory)
    save_json(root / 'acquisition.json', {'utc': datetime.now(timezone.utc).isoformat(),
              'github_commit': tree['sha'], 'files': manifest})
    print(json.dumps({'github_commit': tree['sha'], 'downloaded_files': len(manifest),
                      'hf_files': [{'path': row['path'], 'size': row.get('size')} for row in inventory]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--proxy')
    parser.add_argument('--github-tree', type=Path)
    parser.add_argument('--hf-only', action='store_true')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--direct-cdn', action='store_true')
    main(parser.parse_args())
