import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from compute_load.model import sha256
from prepare_trn_seren import RUN, save_json


def finalize(args):
    project = args.project.resolve()
    output = project / 'results' / RUN
    calculation = json.loads((output / 'calculation-audit.json').read_text('utf-8'))
    for name, digest in calculation['code_sha256'].items():
        if sha256(project / 'analysis' / name) != digest:
            raise ValueError('计算版本与本地代码不一致：' + name)
    for name in ('validation.json', 'figure-qa.json', 'README.md'):
        if not (output / name).is_file():
            raise ValueError('交付文件缺失：' + name)
    previous = subprocess.run(['git', 'diff', '--name-only', args.input_commit, '--',
        'dc-flexibility-project/results/trn-power-model-20261101', 'dc-flexibility-project/results/typical-load-curves-20261101-v3-native',
        'dc-flexibility-project/results/typical-load-curves-20261101-v3', 'dc-flexibility-project/analysis/trn_power_models.py'],
        cwd=project.parent, capture_output=True, text=True, check=True)
    if previous.stdout.strip():
        raise ValueError('旧结果或#36功率模型出现改动：' + previous.stdout)
    names = ['prepare_trn_seren.py', 'build_trn_seren.py', 'verify_trn_seren.py', 'plot_trn_seren.py',
        'report_trn_seren.py', 'finalize_trn_seren.py', 'publish_trn_seren.py', 'launch_trn_seren.sh', 'requirements-trn-seren.txt',
        'compute_load/model.py', 'compute_load/cooling.py']
    save_json(output / 'task-status.json', {'utc': datetime.now(timezone.utc).isoformat(), 'input_commit': args.input_commit,
        'branch': 'codex/trn-seren-20261101', 'dependency_prs': [34, 36], 'main_case': 'main',
        'source_dataset': 'Acme Seren', 'completed': ['source aggregation', '25 location-year power calculations',
            'real-input validation', 'Chinese figures and report'], 'old_result_bytes_changed': False,
        'MIT_source_or_process_or_STOP_actions': False, 'raw_data_transferred_from_server': False,
        'memory': 'One scenario/location/year/case at a time; observed worker VmHWM 414188kB during computation, not a final peak guarantee.',
        'code_sha256': {name: sha256(project / 'analysis' / name) for name in names}})
    paths = [path for path in output.rglob('*') if path.is_file() and 'hourly' not in path.relative_to(output).parts
        and path.name not in ('artifact-manifest.json', 'publication.json')]
    save_json(output / 'artifact-manifest.json', {'utc': datetime.now(timezone.utc).isoformat(),
        'artifact_sha256': {path.relative_to(output).as_posix(): sha256(path) for path in sorted(paths)},
        'hourly_manifest': 'calculation-audit.json:hourly_sha256', 'raw_data_in_Git': False})
    print(json.dumps({'summary_artifacts': len(paths), 'hourly_files': calculation['saved_hourly_files'], 'old_result_diff': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--input-commit', default='6de2e4cde4dbf66f055c0802364c96e511945bbe')
    finalize(parser.parse_args())
