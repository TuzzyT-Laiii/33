import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from build_trn_diurnal import RUN
from compute_load.model import sha256


def main(args):
    project, shared = args.project.resolve(), args.shared.resolve()
    source, destination = project / 'results' / RUN, shared / 'results' / RUN
    validation = json.loads((source / 'validation.json').read_text('utf-8'))
    figures = json.loads((source / 'figure-qa.json').read_text('utf-8'))
    if validation['legacy_static_scenarios'] != 44 or len(figures) != 4:
        raise ValueError('模型或图件验证未完成')
    published = {}
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.name in ('publication.json', 'artifact-manifest.json'):
            continue
        relative = path.relative_to(source)
        target = destination / relative
        digest = sha256(path)
        if target.exists():
            if sha256(target) != digest:
                raise ValueError('已发布文件与待交付版本不同：' + str(relative))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        if sha256(target) != digest:
            raise ValueError('服务器发布读回不一致：' + str(relative))
        published[str(relative)] = digest
    parent_source = source.parent / 'README.md'
    parent_target = destination.parent / 'README.md'
    before = sha256(parent_target)
    shutil.copy2(parent_source, parent_target)
    if sha256(parent_target) != sha256(parent_source):
        raise ValueError('父目录报告入口读回不一致')
    record = {'utc': datetime.now(timezone.utc).isoformat(), 'shared_result_directory': str(destination),
        'verified_files': len(published), 'hourly_files': sum(name.startswith('hourly/') for name in published),
        'all_files_read_back_sha256_verified': True, 'parent_readme_before_sha256': before,
        'parent_readme_sha256': sha256(parent_target), 'summary_artifact_sha256': {name: digest for name, digest in published.items() if not name.startswith('hourly/')},
        'hourly_checksums': 'native-calculation-audit.json: output_hourly_sha256',
        'code_sha256': {name: sha256(project / 'analysis' / name) for name in ('aggregate_trn_acme_hourly.py',
            'prepare_trn_diurnal.py', 'build_trn_diurnal.py', 'trn_power_models.py', 'verify_trn_diurnal.py',
            'report_trn_diurnal.py', 'plot_trn_diurnal.py', 'publish_trn_diurnal.py')}}
    text = json.dumps(record, ensure_ascii=False, indent=2)
    (source / 'publication.json').write_text(text, encoding='utf-8')
    (destination / 'publication.json').write_text(text, encoding='utf-8')
    print(json.dumps({key: record[key] for key in ('utc', 'verified_files', 'hourly_files', 'all_files_read_back_sha256_verified')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--shared', type=Path, required=True)
    main(parser.parse_args())
