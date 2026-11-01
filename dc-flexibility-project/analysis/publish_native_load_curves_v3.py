import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from compute_load.model import sha256
from native_load_inputs_v3 import RUN, save_json


def publish(project, destination):
    project, destination = project.resolve(), destination.resolve()
    output = project / 'results' / RUN
    manifest = json.loads((output / 'manifest.json').read_text('utf-8'))
    validation = json.loads((output / 'validation.json').read_text('utf-8'))
    if validation['status'] != 'passed' or validation['reference_only']:
        raise ValueError('完整验证后才能发布')
    targets = [destination / 'results' / RUN, destination / 'data/processed' / RUN]
    if project == destination or any(target.resolve().parent not in (destination / 'results', destination / 'data/processed') for target in targets):
        raise ValueError('发布目标越出新工作包目录')
    for relative, digest in {**manifest['hourly_sha256'], **manifest['source_hourly_sha256']}.items():
        path, target = project / relative, destination / relative
        if sha256(path) != digest:
            raise ValueError('发布输入哈希不符：' + relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or sha256(target) != digest:
            shutil.copyfile(path, target)
        if sha256(target) != digest:
            raise ValueError('发布小时数据读回失败：' + relative)
    aggregate_hashes = {}
    for path in sorted(output.rglob('*')):
        if not path.is_file() or path.name == 'server-publication.json':
            continue
        target = targets[0] / path.relative_to(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256(path)
        if not target.exists() or sha256(target) != digest:
            shutil.copyfile(path, target)
        if sha256(target) != digest:
            raise ValueError('发布聚合结果读回失败')
        aggregate_hashes[path.relative_to(output).as_posix()] = digest
    code_hashes = {}
    for path in sorted((project / 'analysis').glob('*native*v3*')):
        if path.is_file():
            code_hashes[path.name] = sha256(path)
    frozen = json.loads((project / 'results/typical-load-curves-20261101-v3/manifest.json').read_text('utf-8'))
    for relative, digest in frozen['code_sha256'].items():
        if sha256(project / 'analysis' / relative) != digest:
            raise ValueError('计算目录的沿用模块与冻结代码不同：' + relative)
    record = {'published_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
        'canonical_results': str(targets[0]), 'canonical_city_hourly': str(targets[1]),
        'hourly_files_readback_checked': len(manifest['hourly_sha256']),
        'native_it_files_readback_checked': len(manifest['source_hourly_sha256']),
        'aggregate_files_readback_checked': aggregate_hashes, 'deployed_code_sha256': code_hashes,
        'calculation_project': str(project), 'calculation_analysis': str(project / 'analysis'),
        'frozen_model_source_sha256': frozen['code_sha256'],
        'historical_and_downstream_results_modified': False}
    save_json(output / 'server-publication.json', record)
    shutil.copyfile(output / 'server-publication.json', targets[0] / 'server-publication.json')
    print(json.dumps({'status': 'passed', 'hourly_files': record['hourly_files_readback_checked'], 'native_it_files': record['native_it_files_readback_checked']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    publish(args.project, args.destination)
