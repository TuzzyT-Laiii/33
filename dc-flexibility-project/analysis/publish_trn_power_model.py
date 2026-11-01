import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from compute_load.model import sha256


RUN = 'trn-power-model-20261101'


def publish(source, destination):
    source, destination = source.resolve(), destination.resolve()
    output = source / 'results' / RUN
    manifest = json.loads((output / 'all-locations-manifest.json').read_text('utf-8'))
    validation = json.loads((output / 'validation-all-locations.json').read_text('utf-8'))
    telemetry = json.loads((output / 'acme/validation.json').read_text('utf-8'))
    if validation['all_location_scenarios'] != 1100 or not validation['all_location_sha256_verified'] or not telemetry['all_checks_passed']:
        raise ValueError('情境或遥测验证未通过')
    targets = [destination / 'results' / RUN, destination / 'data/processed' / RUN]
    for target in targets:
        target.relative_to(destination)
        if target.exists():
            raise FileExistsError('目标结果目录已存在：' + str(target))
    for target in targets:
        relative = target.relative_to(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source / relative, target)
    for relative, expected in manifest['hourly_sha256'].items():
        if sha256(destination / relative) != expected:
            raise ValueError('发布后小时文件 SHA-256 不符：' + relative)
    metadata_count = 0
    for path in output.rglob('*'):
        if path.is_file():
            target = targets[0] / path.relative_to(output)
            if sha256(target) != sha256(path):
                raise ValueError('发布后汇总或元数据字节不符：' + str(path.relative_to(output)))
            metadata_count += 1
    record = {'published_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
              'source_project': str(source), 'destination_root': str(destination),
              'results_directory': str(targets[0]), 'processed_directory': str(targets[1]),
              'hourly_files_checked': len(manifest['hourly_sha256']), 'hourly_rows': manifest['hours'],
              'result_files_checked': metadata_count, 'existing_versions_modified': False,
              'MIT_streaming_processes_or_STOP_files_modified': False}
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
