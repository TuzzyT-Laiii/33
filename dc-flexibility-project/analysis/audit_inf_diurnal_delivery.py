import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import fitz
from PIL import Image

from compute_load.model import sha256


def audit(project):
    output = project / 'results/inf-diurnal-20261101'
    validation = json.loads((output / 'validation.json').read_text('utf-8'))
    if validation['status'] != 'passed' or validation['files'] != 300:
        raise ValueError('正式全年核验未完成')
    manifest = json.loads((output / 'manifest.json').read_text('utf-8'))
    for relative, expected in manifest['hourly_sha256'].items():
        if '/hourly/' in relative and sha256(project / relative) != expected:
            raise ValueError('本机小时文件读回SHA不符')
    records = []
    for path in sorted(output.glob('figure*.pdf')):
        document = fitz.open(path)
        for page in document:
            fonts = page.get_fonts(full=True)
            if not fonts or any(not document.extract_font(f[0])[3] for f in fonts):
                raise ValueError('PDF字体未嵌入')
            text = page.get_text()
            if '\ufffd' in text or not any('\u4e00' <= c <= '\u9fff' for c in text):
                raise ValueError('PDF中文文本检查失败')
            for block in page.get_text('dict')['blocks']:
                if block['type'] != 0:
                    continue
                for line in block['lines']:
                    for span in line['spans']:
                        box = fitz.Rect(span['bbox'])
                        if not (page.rect + (-2, -2, 2, 2)).contains(box):
                            raise ValueError('PDF文本越界：' + span['text'])
        image = Image.open(path.with_suffix('.png'))
        if min(image.info.get('dpi', [0, 0])) < 299:
            raise ValueError('PNG分辨率不足300dpi')
        records.append(dict(file=path.name, pages=len(document), embedded_fonts=True,
                            chinese_text=True, text_bounds='passed', png_dpi=image.info['dpi']))
        document.close()
    if len(records) != 4:
        raise ValueError('正式图件数量不完整')
    tracked_v3 = subprocess.run(['git', 'diff', '--name-only', 'ab07a41157442e837659535e1a33579ced5af8fa', '--',
        'dc-flexibility-project/results/typical-load-curves-20261101-v3'], cwd=project.parent, check=True, capture_output=True, text=True)
    if tracked_v3.stdout.strip():
        raise ValueError('历史v3结果被修改')
    scripts = [p for p in (project / 'analysis').glob('*inf*diurnal*.py')]
    scripts += [project / 'analysis/inf_request_shapes.py']
    metadata = dict(checked_utc=datetime.now(timezone.utc).isoformat(), status='passed', figures=records,
        scripts_sha256={p.relative_to(project).as_posix(): sha256(p) for p in scripts},
        published_sha256={p.relative_to(output).as_posix(): sha256(p) for p in output.rglob('*')
            if p.is_file() and p.name != 'delivery-audit.json'},
        history_unchanged=True, verification_method='PDF text/font and PNG metadata checks; no image vision')
    (output / 'delivery-audit.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    print(json.dumps({'status': 'passed', 'figures': len(records), 'scripts': len(scripts)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    audit(parser.parse_args().project.resolve())
