import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from compute_load.model import sha256
from load_curve_scenarios_v3 import RUN


def publish(source, destination):
    source, destination = source.resolve(), destination.resolve()
    output = source / 'results' / RUN
    manifest = json.loads((output / 'manifest.json').read_text('utf-8'))
    validation = json.loads((output / 'validation.json').read_text('utf-8'))
    if manifest['stage'] != 'full' or validation['status'] != 'passed' or manifest['reference_computed_scenarios'] != 44:
        raise ValueError('仅发布已通过完整核验的44个情境')
    targets = [destination / 'results' / RUN, destination / 'data/processed' / RUN]
    for path in targets:
        path.relative_to(destination)
        if path.exists():
            raise FileExistsError('目标版本已存在：' + str(path))
    for path in targets:
        relative = path.relative_to(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source / relative, path)
    for relative, expected in manifest['hourly_sha256'].items():
        if sha256(destination / relative) != expected:
            raise ValueError('发布后小时文件 SHA-256 不符：' + relative)
    for path in output.rglob('*'):
        if path.is_file() and sha256(destination / 'results' / RUN / path.relative_to(output)) != sha256(path):
            raise ValueError('发布后聚合结果字节不符：' + path.name)
    record = {'published_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
              'source_project': str(source), 'destination_root': str(destination),
              'results_directory': str(targets[0]), 'processed_directory': str(targets[1]),
              'hourly_files_checked': len(manifest['hourly_sha256']),
              'hourly_rows': manifest['hourly_rows'], 'existing_versions_modified': False}
    text = json.dumps(record, ensure_ascii=False, indent=2)
    (output / 'server-publication.json').write_text(text, encoding='utf-8')
    (targets[0] / 'server-publication.json').write_text(text, encoding='utf-8')
    print(text, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-project', type=Path, required=True)
    parser.add_argument('--destination-root', type=Path, required=True)
    args = parser.parse_args()
    publish(args.source_project, args.destination_root)
