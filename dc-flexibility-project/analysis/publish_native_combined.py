import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from compute_load.model import sha256
from native_combined_power import RUN


def publish(project, destination):
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
    if target.exists() or processed.exists():
        raise FileExistsError('发布目标已存在，保留原目录：' + str(target))
    shutil.copytree(source, target)
    shutil.copytree(project / 'data/processed' / RUN, processed)
    manifest = json.loads((source / 'manifest.json').read_text('utf-8'))
    checked = {}
    for relative, expected in {**manifest['hourly_sha256'], **manifest['source_hourly_sha256']}.items():
        actual = sha256(destination / relative)
        if actual != expected:
            raise ValueError('服务器发布读回指纹不符：' + relative)
        checked[relative] = actual
    files = {}
    for path in source.rglob('*'):
        if path.is_file():
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
    args = parser.parse_args()
    publish(args.project, args.destination)
