import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

import fitz
import pandas as pd
from PIL import Image
import pyarrow.parquet as pq

from compute_load.model import sha256
from native_load_inputs_v3 import RUN, PREVIOUS, save_json


def audit(project, frozen_commit):
    output = project / 'results' / RUN
    validation = json.loads((output / 'validation.json').read_text('utf-8'))
    if validation['status'] != 'passed' or validation['native_scenarios'] != 44 or validation['reference_only']:
        raise ValueError('缺少完整数值核验')
    summary = pd.read_csv(output / 'native-summary.csv')
    names = summary.loc[summary.reference_it_mw.eq(10)]
    labels = names.scenario_zh.tolist()
    figures = []
    for name in ('figure1-native-day', 'figure2-native-week', 'figure3-native-timeseries',
                 'figure4-native-components', 'figure5-native-mean-comparison', 'figure6-native-sai-sensitivity'):
        with fitz.open(output / (name + '.pdf')) as document:
            if len(document) != 1:
                raise ValueError('图件必须单页')
            page = document[0]
            text = page.get_text()
            compact = ''.join(text.split())
            expected = [label for label in labels if '科研 AI' in label] if name.startswith('figure6') else labels
            if any(''.join(label.split()) not in compact for label in expected):
                raise ValueError('图件缺少确认名称：' + name)
            if re.search(r'\bC(?:20|21|22|28)\b|R-GB200', text) or '\ufffd' in text or '\u25a1' in text:
                raise ValueError('图件名称或字符错误')
            for block in page.get_text('blocks'):
                if not (page.rect + (-1, -1, 1, 1)).contains(fitz.Rect(block[:4])):
                    raise ValueError('图件文本超出页面：' + name)
            for font in page.get_fonts():
                if not font[0] or not document.extract_font(font[0])[3]:
                    raise ValueError('图件字体未嵌入')
        with Image.open(output / (name + '.png')) as image:
            image.verify()
        with Image.open(output / (name + '.png')) as image:
            if min(image.info['dpi']) < 299 or min(image.size) < 1000:
                raise ValueError('图件分辨率不足')
            figures.append({'figure': name, 'pixels': list(image.size), 'dpi': list(image.info['dpi']),
                            'confirmed_names_present': True, 'embedded_fonts': True, 'text_within_page': True})
    banned = {'job_id', 'jobid', 'id_job', 'node', 'gpu_index', 'user', 'user_id', 'nodelist'}
    for path in output.rglob('*.parquet'):
        if banned.intersection(pq.read_schema(path).names):
            raise ValueError('公开数据包含逐作业/设备标识：' + path.name)
    for path in output.glob('*.csv'):
        if banned.intersection(pd.read_csv(path, nrows=0).columns):
            raise ValueError('公开数据包含逐作业/设备标识：' + path.name)
    repo = project.parent
    immutable = {}
    directories = ('typical-load-curves-20261029-v2', 'eight-city-phase-a-20261029', PREVIOUS)
    for directory in directories:
        prefix = f'{project.name}/results/{directory}'
        paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', frozen_commit, '--', prefix], cwd=repo, text=True).splitlines()
        for relative in paths:
            content = subprocess.check_output(['git', 'cat-file', 'blob', f'{frozen_commit}:{relative}'], cwd=repo)
            digest = hashlib.sha256(content).hexdigest()
            if sha256(repo / relative) != digest:
                raise ValueError('历史目录字节变化：' + relative)
            immutable[relative] = digest
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed', 'frozen_commit': frozen_commit,
              'figures': figures, 'visual_inspection_performed': False,
              'visual_inspection_policy': '用户禁止主动调用视觉功能；执行字体、文本边界、文件与数据检查',
              'public_schema_has_no_job_device_identifiers': True, 'unchanged_historical_artifacts': immutable}
    save_json(output / 'delivery-audit.json', result)
    hashes = {path.relative_to(output).as_posix(): sha256(path) for path in sorted(output.rglob('*'))
              if path.is_file() and path.name not in ('artifact-manifest.json', 'progress.json', 'progress.jsonl')}
    code = {path.name: sha256(path) for path in sorted((project / 'analysis').glob('*native*v3*')) if path.is_file()}
    save_json(output / 'artifact-manifest.json', {'sha256': hashes, 'code_sha256': code})
    print(json.dumps({'status': 'passed', 'figures': len(figures), 'historical_files_unchanged': len(immutable), 'artifacts': len(hashes)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--frozen-commit', default='ab07a41157442e837659535e1a33579ced5af8fa')
    args = parser.parse_args()
    audit(args.project.resolve(), args.frozen_commit)
