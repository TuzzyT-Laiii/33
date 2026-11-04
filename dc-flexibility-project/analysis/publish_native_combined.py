import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from compute_load.model import sha256
from native_combined_power import RUN


def publish(project, destination, refresh=False):
    source = project / 'results' / RUN
    verification = json.loads((source / 'verification.json').read_text('utf-8'))
    if not verification['passed']:
        raise ValueError('全量核验尚未通过')
    brief = json.loads((source / 'brief-audit.json').read_text('utf-8'))
    if not brief['passed']:
        raise ValueError('简表核验尚未通过')
    for relative, expected in brief['model_input_sha256'].items():
        if sha256(project / relative) != expected:
            raise ValueError('本机与服务器模型输入不一致：' + relative)
    target = destination / 'results' / RUN
    processed = destination / 'data/processed' / RUN
    manifest = json.loads((source / 'manifest.json').read_text('utf-8'))
    if refresh:
        previous = json.loads((target / 'server-publication.json').read_text('utf-8'))
        previous_manifest = json.loads((target / 'manifest.json').read_text('utf-8'))
        for name in ('input_native_manifest_sha256', 'hourly_sha256', 'source_hourly_sha256'):
            if manifest[name] != previous_manifest[name]:
                raise ValueError('说明更新要求原小时结果保持不变：' + name)
        for relative, expected in previous['result_sha256'].items():
            if sha256(target / relative) != expected:
                raise ValueError('发布目录含后续修改，保留现状：' + relative)
        for path in source.rglob('*'):
            relative = path.relative_to(source)
            if path.is_file() and not {'hourly', 'source-hourly'}.intersection(relative.parts) and path.name != 'server-publication.json':
                (target / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target / relative)
    else:
        if target.exists() or processed.exists():
            raise FileExistsError('发布目标已存在，保留原目录：' + str(target))
        shutil.copytree(source, target)
        shutil.copytree(project / 'data/processed' / RUN, processed)
    checked = {}
    for relative, expected in {**manifest['hourly_sha256'], **manifest['source_hourly_sha256']}.items():
        actual = sha256(destination / relative)
        if actual != expected:
            raise ValueError('服务器发布读回指纹不符：' + relative)
        checked[relative] = actual
    files = {}
    for path in source.rglob('*'):
        if path.is_file() and path.name != 'server-publication.json':
            relative = path.relative_to(source)
            digest = sha256(path)
            if sha256(target / relative) != digest:
                raise ValueError('服务器聚合产物读回不符：' + str(relative))
            files[str(relative)] = digest
    audit = dict(utc=datetime.now(timezone.utc).isoformat(), verified_hourly_files=len(checked), result_files=len(files),
        result_sha256=files, publication_root=str(destination), old_directories_modified=False)
    for folder in (source, target):
        (folder / 'server-publication.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in audit.items() if key != 'result_sha256'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    publish(args.project, args.destination, args.refresh)
