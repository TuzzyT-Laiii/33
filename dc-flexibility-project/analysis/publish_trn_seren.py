import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from compute_load.model import sha256
from prepare_trn_seren import RUN, save_json


def publish(args):
    source = args.project.resolve() / 'results' / RUN
    target = args.shared.resolve() / 'results' / RUN
    if source == target:
        raise ValueError('发布源目录与目标目录不能相同')
    manifest = json.loads((source / 'artifact-manifest.json').read_text('utf-8'))
    calculation = json.loads((source / 'calculation-audit.json').read_text('utf-8'))
    expected = {**manifest['artifact_sha256'], **calculation['hourly_sha256']}
    previous = {}
    if (target / 'artifact-manifest.json').exists():
        previous.update(json.loads((target / 'artifact-manifest.json').read_text('utf-8'))['artifact_sha256'])
        previous.update(json.loads((target / 'calculation-audit.json').read_text('utf-8'))['hourly_sha256'])
    for relative, digest in expected.items():
        origin, destination = source / relative, target / relative
        if sha256(origin) != digest:
            raise ValueError('发布源哈希不符：' + relative)
        if destination.exists():
            current = sha256(destination)
            if current not in (digest, previous.get(relative)):
                raise ValueError('发布目标出现未识别的改动：' + relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination)
        if sha256(destination) != digest:
            raise ValueError('发布读回失败：' + relative)
    shutil.copy2(source / 'artifact-manifest.json', target / 'artifact-manifest.json')
    record = {'utc': datetime.now(timezone.utc).isoformat(), 'shared_output': str(target),
        'summary_artifacts_sha256_verified': len(manifest['artifact_sha256']),
        'hourly_files_sha256_verified': len(calculation['hourly_sha256']), 'all_files_read_back': True,
        'artifact_manifest_sha256': sha256(source / 'artifact-manifest.json'),
        'raw_data_published_to_Git': False, 'MIT_actions': False}
    save_json(target / 'publication.json', record)
    save_json(source / 'publication.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--shared', type=Path, required=True)
    publish(parser.parse_args())
