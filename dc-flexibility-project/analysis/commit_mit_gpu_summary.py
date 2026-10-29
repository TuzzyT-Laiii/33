import argparse
import json
import os
from pathlib import Path
import shutil
import tarfile
import time


def commit(root):
    root = root.resolve(strict=True)
    incoming = root / '.incoming'
    archive = incoming / 'delta.tar'
    if shutil.disk_usage(root).free - archive.stat().st_size < 50_000_000_000:
        raise OSError('服务器解包后余量将低于 50 GB')
    staging = incoming / ('unpack-' + str(time.time_ns()))
    staging.mkdir()
    with tarfile.open(archive, 'r') as source:
        for member in source.getmembers():
            target = (staging / member.name).resolve()
            if not target.is_relative_to(staging) or member.issym() or member.islnk():
                raise ValueError('归档出现越界路径或链接')
        source.extractall(staging, filter='data')
    (root / 'batches').mkdir(exist_ok=True)
    if (staging / 'batches').exists():
        for batch in (staging / 'batches').iterdir():
            destination = root / 'batches' / batch.name
            if destination.exists():
                if (destination / 'manifest.json').read_bytes() != (batch / 'manifest.json').read_bytes():
                    raise ValueError('服务器同名批次的内容标识不同: ' + batch.name)
            else:
                os.replace(batch, destination)
    for path in staging.iterdir():
        if path.is_file():
            os.replace(path, root / path.name)
    for directory in ('aggregates', 'analysis/priority-month'):
        for path in (staging / directory).rglob('*'):
            if path.is_file():
                destination = root / path.relative_to(staging)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, destination)
    if not staging.resolve().is_relative_to(incoming.resolve()):
        raise ValueError('暂存清理路径超出同步目录')
    shutil.rmtree(staging)
    archive.unlink()
    print(json.dumps({'committed_root': str(root)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    commit(parser.parse_args().root)
