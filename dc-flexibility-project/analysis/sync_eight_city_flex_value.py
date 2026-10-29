import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import time
import zipfile

import pandas as pd

from eight_city_onsite_solver import sha256
from run_eight_city_flex_value import PROJECT, write_json


def main():
    parser = argparse.ArgumentParser(description='只读第一阶段已完成文件，复制到第二阶段服务器独立输入目录')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--host', default='pku-test')
    parser.add_argument('--server-root', default='/data3/dataset/ljt/eight-city-flex-value-20261029')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--max-hours', type=float, default=12)
    args = parser.parse_args()
    temporary = PROJECT / '.codex-temp/flex-value-sync'
    temporary.mkdir(parents=True, exist_ok=True)
    state_path = temporary / 'sync-state.json'
    done = json.loads(state_path.read_text(encoding='utf-8'))['files'] if state_path.exists() else {}
    deadline = time.monotonic() + args.max_hours * 3600
    while True:
        selected = []
        for path in sorted(args.input_dir.glob('*.parquet')):
            if path.name.endswith('-thermal-only.parquet') or path.name.endswith('-supply.parquet') or path.name.endswith('-10MW.parquet'):
                if time.time() - path.stat().st_mtime < 15:
                    continue
                digest = sha256(path)
                if done.get(path.name) != digest:
                    selected.append((path, digest))
        if selected:
            package = temporary / 'inputs.zip'
            with zipfile.ZipFile(package, 'w', compression=zipfile.ZIP_STORED) as archive:
                for path, digest in selected:
                    # Parquet尾部可读与哈希稳定之后才发布到独立输入副本。
                    pd.read_parquet(path, columns=['time_utc'])
                    archive.write(path, path.name)
                    if digest != sha256(path):
                        raise RuntimeError(f'输入在打包时改变：{path}')
            subprocess.run(['scp', str(package), f'{args.host}:{args.server_root}/input-transfer.zip'], check=True)
            command = ('cd ' + shlex.quote(args.server_root + '/inputs/phase-a') +
                       ' && /data3/dataset/ljt/global-grid-flexibility-20261023/.venv/bin/python -m zipfile -e ' +
                       shlex.quote(args.server_root + '/input-transfer.zip') + ' .')
            subprocess.run(['ssh', args.host, command], check=True)
            done.update({path.name: digest for path, digest in selected})
            write_json(state_path, {'utc': datetime.now(timezone.utc).isoformat(), 'files': done,
                                    'server_root': args.server_root, 'source': str(args.input_dir)})
            print(json.dumps({'synced': len(selected), 'total': len(done)}, ensure_ascii=False), flush=True)
        complete = sum(name.endswith('-10MW.parquet') for name in done) >= 96
        complete |= (sum(name.endswith('-thermal-only.parquet') for name in done) >= 96 and sum(name.endswith('-supply.parquet') for name in done) >= 8)
        if complete or not args.watch or time.monotonic() >= deadline:
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
