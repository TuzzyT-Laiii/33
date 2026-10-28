import argparse
import json
from pathlib import Path

from acquire_buildability_protection import utc_now, write_json


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def run(root):
    state = read(root / 'status/era5-eligible-acquisition.json')
    manifest = read(root / 'data/metadata/buildability-20261025/weather-acquisition-manifest.json')
    checkpoints = root / 'checkpoints/era5-eligible-blocks'
    outputs = root / 'data/processed/era5-eligible-2023-2025'
    completed = {path.stem for path in checkpoints.glob('*.json')}
    if any(not (outputs / (label + '.parquet')).is_file() for label in completed):
        raise ValueError('正式完成回执对应的Parquet缺失')
    quarantine = read(root / 'data/metadata/buildability-20261025/weather-block-quarantine.json')
    records = dict(quarantine['blocks']) if quarantine else {}
    for path in sorted((root / 'data/metadata/buildability-20261025/weather-block-defects').glob('*.json')):
        if path.stem in records:
            raise ValueError('隔离记录重复')
        records[path.stem] = read(path)
    pending = {}
    repaired = []
    for label, item in records.items():
        proof = read(root / 'data/metadata/buildability-20261025/weather-cds-repairs' / (label + '.json'))
        if label in completed:
            if not proof or proof['status'] != 'verified':
                raise ValueError('隔离块已有完成回执但缺少已核验补取证明')
            repaired.append(label)
            continue
        pending[label] = {**item, 'proof_status': proof['status'] if proof else 'not_available',
                          'formal_completed': False}
    result = {'checked_utc': utc_now(), 'formal_completed_blocks': len(completed),
              'total_required_blocks': manifest['blocks'], 'isolated_pending_blocks': pending,
              'other_blocks_completed': len(completed - records.keys()),
              'other_blocks_target': manifest['blocks'] - len(records),
              'repaired_isolated_blocks': repaired,
              'overall_complete': state['status'] == 'complete' and len(completed) == manifest['blocks'] and not pending,
              'formal': state, 'manager': read(root / 'status/job-era5-eligible.json'),
              'dewpoint_controller': read(root / 'status/era5-dewpoint-repair-controller.json'),
              'dewpoint_repair': read(root / 'status/era5-dewpoint-repair.json'),
              'alternatives': {name: read(root / f'status/era5-dewpoint-alternative-{name}.json')
                               for name in ('timeseries', 'ncar')},
              'count_basis': '实际完成回执及对应Parquet存在；本次不重算全量SHA，状态与完整科学验证分别报告'}
    write_json(root / 'results/buildability-20261025/formal-progress-latest.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root.resolve())
