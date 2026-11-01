import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from compute_load.model import sha256


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main(project, raw):
    output = project / 'results/trn-power-model-20261101'
    telemetry = json.loads((output / 'acme/telemetry-run.json').read_text('utf-8'))
    if not telemetry['complete'] or telemetry['files'] != 9:
        raise ValueError('监测汇总尚未完成')
    acquisition = json.loads((raw / 'acquisition.json').read_text('utf-8'))
    inventory = json.loads((raw / 'huggingface-tree.json').read_text('utf-8'))
    hf = json.loads((output / 'acme/hf-acquisition.json').read_text('utf-8'))
    expected = {row['path']: row for row in inventory if row['type'] == 'file'}
    selected = []
    for item in hf['files']:
        relative = item['url'].split('/resolve/main/', 1)[1]
        record = expected[relative]
        if item['bytes'] != record['size'] or item['sha256'] != record['lfs']['oid']:
            raise ValueError('原始监测来源清单或哈希不符')
        selected.append({'file': relative, 'url': item['url'], 'bytes': item['bytes'], 'sha256': item['sha256'], 'lfs_sha256_matched': True})
    if not hf['complete'] or len(selected) != 9 or sum(item['bytes'] for item in selected) != 21703979345:
        raise ValueError('九份监测文件未完整取得')
    save(output / 'acme/source-inventory.json', {'github_commit': acquisition['github_commit'],
         'huggingface_revision_observed': 'a484f7763a7812fc72eb49b8ba865b12fe7638e1',
         'license': 'CC BY 4.0', 'telemetry_files': selected,
         'github_files': [{'file': item['url'].split(acquisition['github_commit'] + '/', 1)[1],
                           'url': item['url'], 'bytes': item['bytes'], 'sha256': item['sha256']} for item in acquisition['files']],
         'inventory_sha256': sha256(raw / 'huggingface-tree.json'),
         'raw_server_root': '/data3/dataset/ljt/global-grid-flexibility-20261023/data/raw/trn-power-model-20261101'})
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text('utf-8'))
    manifest['scripts'].update({path.name: sha256(path) for path in (project / 'analysis').glob('*trn*.py')})
    for name in ('summarize_acme_telemetry.py', 'verify_acme_telemetry.py', 'requirements-trn-power-model.txt'):
        manifest['scripts'][name] = sha256(project / 'analysis' / name)
    manifest['evidence_sha256'] = {name: sha256(output / name) for name in
        ('literature-evidence.csv', 'literature-acquisition.json', 'specpower-evidence.csv', 'model-audit.csv', 'acme/source-inventory.json')}
    manifest['finalized_utc'] = datetime.now(timezone.utc).isoformat()
    save(manifest_path, manifest)
    save(output / 'task-status.json', {'input_commit': manifest['base_commit'], 'base_pr': 34,
         'completed': ['nine telemetry files downloaded and LFS SHA-256 checked', 'all telemetry distributions and diurnal aggregates',
                       'literature and idle audit', 'CPU smoothing', '1100 scenario recalculation', 'real-input checks', 'Chinese figures and report'],
         'pending_research_checks': [], 'utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--raw', type=Path, required=True)
    args = parser.parse_args()
    main(args.project.resolve(), args.raw.resolve())
