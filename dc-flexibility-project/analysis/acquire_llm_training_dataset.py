import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import requests

from compute_load.model import sha256


def main(args):
    metadata = json.loads(args.metadata.read_text('utf-8'))
    files = metadata['files']
    if len(files) != 1 or files[0]['size'] > 1_000_000_000:
        raise ValueError('公开实验数据的文件结构或体积与已审阅元数据不同')
    item = files[0]
    args.raw.mkdir(parents=True, exist_ok=True)
    path = args.raw / item['name']
    if not path.exists():
        session = requests.Session()
        session.trust_env = False
        session.headers['User-Agent'] = 'Mozilla/5.0'
        if args.proxy:
            session.proxies = {'http': args.proxy, 'https': args.proxy}
        response = session.get(args.download_url or item['download_url'], params=None if args.download_url else {'download': '1'},
            stream=True, timeout=(15, 60))
        response.raise_for_status()
        with path.with_suffix('.part').open('wb') as target:
            for chunk in response.iter_content(1024 * 1024):
                target.write(chunk)
        with path.with_suffix('.part').open('rb') as stream:
            temporary_md5 = hashlib.file_digest(stream, 'md5').hexdigest()
        if path.with_suffix('.part').stat().st_size != item['size'] or temporary_md5 != item['supplied_md5']:
            raise ValueError('Figshare下载未得到完整的校验通过对象')
        path.with_suffix('.part').replace(path)
    with path.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'md5').hexdigest()
    if path.stat().st_size != item['size'] or checksum != item['supplied_md5']:
        raise ValueError('Figshare实验数据长度或MD5不符')
    with zipfile.ZipFile(path) as archive:
        entries = [{'path': member.filename, 'bytes': member.file_size} for member in archive.infolist() if not member.is_dir()]
    record = {'doi': metadata['doi'], 'metadata_url': 'https://api.figshare.com/v2/articles/31654879',
        'download_url': args.download_url or item['download_url'], 'bytes': item['size'], 'md5': checksum, 'sha256': sha256(path),
        'license': metadata['license'], 'zip_entries': entries}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'llm-experiment-acquisition.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DATASET', metadata['doi'], item['size'], metadata['license'], flush=True)
    print(json.dumps(entries, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--proxy')
    parser.add_argument('--download-url')
    main(parser.parse_args())
