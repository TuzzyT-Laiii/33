import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from acquire_buildability_protection import sha, utc_now, write_json
from acquire_eligible_era5_weather import open_source, process_block


def verify(root, label):
    path = root / 'data/metadata/buildability-20261025/weather-cds-repairs' / (label + '.json')
    report = json.loads(path.read_text())
    if report['status'] not in ('awaiting_overlap_validation', 'verified') or not all(
            item['exact_overlap_equal'] and item['verified_overlap_count'] > 0 for item in report['checks'].values()):
        raise ValueError('CDS补取与ARCO的重合时段未通过逐值一致性验证')
    for source in report['raw_sources']:
        if source['dataset'] != 'reanalysis-era5-single-levels' or sha(Path(source['path'])) != source['sha256']:
            raise ValueError('CDS补取原文来源或SHA不一致')
    selection_path = root / 'data/metadata/buildability-20261025/selection-manifest.json'
    selection = json.loads(selection_path.read_text())
    points = pd.read_parquet(selection['outputs']['points']['path']).sort_values('native_flat_index')
    lat = np.rint((points.latitude + 90) * 4).astype(int) // 4
    lon = np.rint(((points.longitude_0_360 + 180) % 360) * 4).astype(int) // 4
    labels = pd.Series([f'lat-{a:03d}-lon-{b:03d}' for a, b in zip(lat, lon)], index=points.index)
    selected = points.loc[labels.eq(label)]
    source, session, times, metadata_sha = open_source(root)
    source.close()
    session.close()
    expected_output = root / 'data/interim/era5-eligible-cds-repairs' / (label + '.parquet')
    if (report['block'] != label or report['hours'] != 26305
            or report['selection_manifest_sha256'] != sha(selection_path) or report['source_metadata_sha256'] != metadata_sha
            or report['native_flat_indices'] != selected.native_flat_index.astype(int).tolist()
            or report['output_path'] != str(expected_output) or sha(expected_output) != report['output_sha256']):
        raise ValueError('补取证明与当前合格节点或数值表不同')
    if report['status'] != 'verified':
        report.update(status='verified', verified_utc=utc_now(), acceptance='全部有效重合值完全相同，NaN位置采用同时间同节点官方ERA5原文')
        write_json(path, report)
    result = process_block(root, label, selected, sha(selection_path), metadata_sha, times)
    checkpoint = json.loads((root / 'checkpoints/era5-eligible-blocks' / (label + '.json')).read_text())
    if checkpoint['status'] != 'done' or checkpoint['hours'] != 26305 or sha(Path(checkpoint['output_path'])) != checkpoint['output_sha256']:
        raise ValueError('定向补取后的正式输出未通过完整小时与SHA检查')
    return {'formal_block': result, 'verified_overlap': report['checks'],
            'source_supplements': list(checkpoint.get('source_supplements', {}))}


def run(root, label, start_formal, group='wind', monthly=False):
    state_path = root / 'status' / f'era5-{group}-repair-controller.json'
    state = {'status': 'preparing', 'block': label, 'group': group,
             'strategy': 'monthly_sequential_with_existing_annual_reuse' if monthly else 'annual',
             'pid': os.getpid(), 'started_utc': utc_now()}
    write_json(state_path, state)
    try:
        command = [sys.executable, str(root / 'analysis/prepare_era5_wind_repair.py'),
                   '--root', str(root), '--block', label, '--group', group]
        if monthly:
            command.append('--monthly')
        subprocess.run(command, check=True)
        state.update(status='validating', updated_utc=utc_now())
        write_json(state_path, state)
        state.update(verification=verify(root, label), status='verified', updated_utc=utc_now())
        write_json(state_path, state)
        if start_formal:
            subprocess.run([sys.executable, str(root / 'analysis/grid_job_manager.py'), '--start', 'era5-eligible'], check=True)
            state.update(status='formal_acquisition_resumed', updated_utc=utc_now())
            write_json(state_path, state)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        state.update(status='failed', error_type=type(error).__name__, updated_utc=utc_now())
        write_json(state_path, state)
        raise
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--block', required=True)
    parser.add_argument('--start-formal', action='store_true')
    parser.add_argument('--group', choices=('wind', 'dewpoint'), default='wind')
    parser.add_argument('--monthly', action='store_true')
    args = parser.parse_args()
    with (args.root.resolve() / f'.era5-{args.group}-repair-controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.root.resolve(), args.block, args.start_formal, args.group, args.monthly)
