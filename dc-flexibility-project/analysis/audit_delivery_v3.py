import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import fitz
from PIL import Image
import pandas as pd
import pyarrow.parquet as pq

from compute_load.model import sha256
from load_curve_scenarios_v3 import RUN


def audit(project, frozen_commit):
    output = project / 'results' / RUN
    manifest = json.loads((output / 'manifest.json').read_text('utf-8'))
    if manifest['stage'] != 'full' or manifest['reference_computed_scenarios'] != 44:
        raise ValueError('完整交付必须覆盖44个情境')
    summary = pd.read_csv(output / 'annual-summary.csv')
    selected = summary.loc[summary.reference_it_mw.eq(10)]
    expected_labels = selected.scenario_zh.tolist()
    figure_checks = []
    names = ['figure1-typical-day', 'figure2-typical-week', 'figure3-annual-envelope',
             'figure4-power-components', 'figure5-v2-v3-comparison', 'figure6-sai-transfer-sensitivity']
    for index, name in enumerate(names):
        pdf_path = output / (name + '.pdf')
        with fitz.open(pdf_path) as document:
            if len(document) != 1:
                raise ValueError('图件PDF必须为单页')
            page = document[0]
            text = page.get_text()
            if '\ufffd' in text or '\u25a1' in text:
                raise ValueError('图件文本含替代字符')
            for block in page.get_text('blocks'):
                box = fitz.Rect(block[:4])
                if not (page.rect + (-1, -1, 1, 1)).contains(box):
                    raise ValueError('图件文本超出PDF页面边界')
            for font in page.get_fonts():
                if not font[0] or not document.extract_font(font[0])[3]:
                    raise ValueError('图件包含未嵌入字体')
            labels = expected_labels if index < 4 else ([value for value in expected_labels if '云通用计算' not in value]
                if index == 4 else [value for value in expected_labels if '科研 AI' in value])
            compact = ''.join(text.split())
            if any(''.join(value.split()) not in compact for value in labels):
                raise ValueError('图件缺少确认情境名称：' + name)
        with Image.open(output / (name + '.png')) as image:
            image.verify()
        with Image.open(output / (name + '.png')) as image:
            dpi = image.info['dpi']
            if min(dpi) < 299 or min(image.size) < 1000:
                raise ValueError('PNG分辨率不满足交付要求')
            figure_checks.append({'figure': name, 'pixels': list(image.size), 'dpi': list(dpi),
                                  'pdf_sha256': sha256(pdf_path), 'confirmed_names_present': True,
                                  'embedded_fonts': True, 'text_within_page': True})
    banned = {'job_id', 'jobid', 'node', 'gpu_index', 'user', 'user_id'}
    for path in output.rglob('*.parquet'):
        if banned & set(pq.read_schema(path).names):
            raise ValueError('公开Parquet包含逐设备或逐作业标识')
    for path in output.glob('*.csv'):
        if banned & set(pd.read_csv(path, nrows=0).columns):
            raise ValueError('公开CSV包含逐设备或逐作业标识')
    if list(output.rglob('device-hours.parquet')) or list(output.rglob('snapshot.json')):
        raise ValueError('公开结果混入私有设备表或对象快照')
    immutable = {}
    repo = project.parent
    for directory in ('typical-load-curves-20261029-v2', 'eight-city-phase-a-20261029'):
        prefix = f'{project.name}/results/{directory}'
        names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', frozen_commit, '--', prefix], cwd=repo, text=True).splitlines()
        for name in names:
            expected = subprocess.check_output(['git', 'cat-file', 'blob', f'{frozen_commit}:{name}'], cwd=repo)
            digest = hashlib.sha256(expected).hexdigest()
            if sha256(repo / name) != digest:
                raise ValueError('历史结果字节发生变化：' + name)
            immutable[name] = digest
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed', 'figures': figure_checks,
              'visual_inspection_performed': False, 'visual_inspection_policy': '遵循用户不主动调用视觉功能的规范；执行字体、文本边界、PNG和PDF可读性检查',
              'public_schema_has_no_job_device_identifiers': True, 'frozen_commit': frozen_commit,
              'unchanged_historical_artifacts': immutable}
    (output / 'delivery-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    artifacts = {path.relative_to(output).as_posix(): sha256(path) for path in sorted(output.rglob('*'))
                 if path.is_file() and path.name not in ('artifact-manifest.json', 'progress.json', 'progress.jsonl')}
    (output / 'artifact-manifest.json').write_text(json.dumps({'sha256': artifacts,
        'code_sha256': {path.name: sha256(path) for path in sorted((project / 'analysis').glob('*_v3.py'))}},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'figure_count': len(figure_checks), 'unchanged_historical_files': len(immutable), 'artifacts': len(artifacts)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--frozen-commit', default='08e843779a472e6fc7502a50e3e0fa9b8e9cb469')
    args = parser.parse_args()
    audit(args.project.resolve(), args.frozen_commit)
