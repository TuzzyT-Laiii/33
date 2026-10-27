import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from acquire_buildability_protection import sha, utc_now, write_json
from acquire_eligible_era5_weather import process_block
from era5_block_quarantine import quarantined_blocks


def run(root):
    selection_path = root / 'data/metadata/buildability-20261025/selection-manifest.json'
    selection = json.loads(selection_path.read_text())
    selection_sha = sha(selection_path)
    metadata_sha = sha(root / 'data/raw/era5-eligible-ecmwf-geo/.zmetadata')
    points = pd.read_parquet(selection['outputs']['points']['path']).sort_values('native_flat_index')
    points['lat_block'] = np.rint((points.latitude + 90) * 4).astype(int) // 4
    points['lon_block'] = np.rint(((points.longitude_0_360 + 180) % 360) * 4).astype(int) // 4
    blocks = [(f'lat-{lat:03d}-lon-{lon:03d}', part) for (lat, lon), part in points.groupby(['lat_block', 'lon_block'], sort=True)]
    pending = quarantined_blocks(root, selection_sha, metadata_sha, blocks)
    if 'lat-084-lon-290' not in pending:
        raise ValueError('实际隔离块与本次验证范围不同')
    checkpoint = root / 'checkpoints/era5-eligible-blocks/lat-084-lon-290.json'
    if checkpoint.exists():
        raise ValueError('隔离块被提前记为完成')
    baseline = json.loads((root / 'results/buildability-20261025/isolation/baseline.json').read_text())['blocks']
    for label, record in baseline.items():
        if sha(root / 'checkpoints/era5-eligible-blocks' / (label + '.json')) != record['sha256']:
            raise ValueError('3273份既有完成回执被重写')
    label = 'lat-084-lon-294'
    part = dict(blocks)[label]
    path = root / 'checkpoints/era5-eligible-blocks' / (label + '.json')
    before = sha(path)
    times = pd.date_range('2023-01-01', '2026-01-01', freq='h').to_numpy()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(process_block, root, label, part, selection_sha, metadata_sha, times) for _ in range(2)]
        results = [future.result() for future in futures]
    if sha(path) != before or any(result['status'] != 'reused_verified' for result in results):
        raise ValueError('并发接续未复用同一份已核验输出')
    command = [sys.executable, str(root / 'analysis/grid_job_manager.py'), '--start', 'era5-eligible', '--ensure-running']
    already_running = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    if already_running.get('status') != 'already_running':
        raise ValueError('正在运行的正式管理器没有被复用')
    monitor = json.loads(subprocess.run([sys.executable, str(root / 'analysis/monitor_grid_jobs.py'), '--root', str(root), '--check'],
                                       check=True, capture_output=True, text=True).stdout)['jobs']['era5-eligible']
    if monitor['status'] != 'running' or len(monitor['processes']['workers']) != 1 or len(monitor['processes']['managers']) != 1:
        raise ValueError('正式管理器或工作进程数量与运行状态不符')
    result = {'checked_utc': utc_now(), 'status': 'passed', 'quarantine': list(pending),
              'baseline_3273_unchanged': True, 'concurrent_reuse': results, 'idempotent_start': already_running,
              'monitor': monitor, 'scope': '真实筛选清单、冻结原文、正式输出及活跃管理器；未构造数据或替换外部系统'}
    write_json(root / 'results/buildability-20261025/isolation/behavior-validation.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root.resolve())
