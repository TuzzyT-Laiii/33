import argparse
from datetime import datetime, timezone
from hashlib import file_digest
import json
from pathlib import Path

from docx import Document
import numpy as np
import pandas as pd
import pyarrow as pa
from pypdf import PdfReader

RUN = 'typical-load-curves-20261101-v3-native-combined'


def sha256(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def audit(project, pdf, word_audit, visual_confirmed):
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    output = project / 'results' / RUN
    docx_path = project / 'docs/job-flexibility-brief/数据中心任务时间柔性简表.docx'
    document = Document(docx_path)
    table_data = json.loads((output / 'brief-summary.json').read_text('utf-8'))['rows']
    if len(document.tables) != 4 or len(document.tables[-1].rows) != 12:
        raise ValueError('Word表格结构不完整')
    for expected, actual in zip(table_data, document.tables[-1].rows):
        if expected != [cell.text for cell in actual.cells]:
            raise ValueError('Word原生窗口数值与结果JSON不一致')
    reader = PdfReader(pdf)
    pages = [page.extract_text() for page in reader.pages]
    if len(pages) != 6 or '各情境典型日负荷曲线' not in pages[3] or '各情境原生窗口负荷曲线' not in pages[4] or '原生窗口时序总览' not in pages[5]:
        raise ValueError('PDF章节或页数不符')
    if '\ufffd' in ''.join(pages):
        raise ValueError('PDF包含替代字符')
    layout = json.loads(word_audit.read_text(encoding='utf-8-sig'))
    if layout['pages'] != 6 or any(table['start_page'] != table['end_page'] for table in layout['tables']):
        raise ValueError('Word分页核验不通过')
    cross = pd.read_csv(output / 'chapter4-comparison.csv')
    np.testing.assert_allclose(cross.window_it_mean_absolute_difference_mw, 0, atol=1e-10)
    np.testing.assert_allclose(cross.selected_hourly_maximum_difference_mw, 0, atol=1e-10)
    source_summary = pd.read_csv(output / 'native-summary.csv')
    for record in source_summary.to_dict('records'):
        sid = record['scenario_id']
        source = pd.read_parquet(output / 'source-hourly' / (sid + '.parquet'))
        placement = 'dated' if record['has_real_dates'] else 'm01'
        full = pd.read_parquet(output / 'hourly' / f'{sid}-2025-{placement}.parquet')
        np.testing.assert_allclose(source[['it_mw', 'base_it_mw', 'cpu_it_mw', 'gpu_it_mw']], full[['it_mw', 'base_it_mw', 'cpu_it_mw', 'gpu_it_mw']], rtol=1e-12, atol=1e-10)
        values = source.it_mw.to_numpy()
        for name, expected in [('mean_it_mw', values.mean()), ('peak_it_mw', values.max()), ('minimum_it_mw', values.min()), ('window_it_mwh', values.sum())]:
            np.testing.assert_allclose(record[name], expected, rtol=1e-12)
        for label, field in [('diurnal', 'hour_of_source_day'), ('weekly', 'hour_of_source_week')]:
            if label == 'weekly' and record['use'] == 'CLD':
                continue
            groups = source[field].to_numpy(int)
            means = np.bincount(groups, weights=values) / np.bincount(groups)
            np.testing.assert_allclose(record[f'it_{label}_peak_valley'], means.max() / means.min(), rtol=1e-12)
    for path in (output / 'selected-days').glob('*-by-task.csv'):
        parts = pd.read_csv(path)
        scenario = path.name.removesuffix('-by-task.csv')
        hourly = pd.read_csv(path.with_name(scenario + '-hourly.csv'))
        np.testing.assert_allclose(parts.groupby('hour').facility_mw.sum(), hourly.facility_mw, rtol=1e-12, atol=1e-10)
    if not visual_confirmed:
        raise ValueError('需要实际逐页检查后确认视觉验收')
    model_files = ['results/server-count-scenarios-20261028-v2/server-configurations.json',
        'results/typical-load-curves-20261029-v2/mit-power-ratio-summary.json',
        'results/typical-load-curves-20261101-v3/mit-calibration.json',
        'results/typical-load-curves-20261101-v3/power-parameters.csv',
        'results/trn-power-model-20261101/cpu-shape.json',
        'results/trn-power-model-20261101/diurnal-acme/transfer-profile.csv',
        'results/inf-diurnal-20261101/request-shape-24h.csv',
        'results/inf-diurnal-20261101/power-calibration.csv',
        'analysis/compute_load/assets/cooling-20261028-v2.json']
    result = dict(utc=datetime.now(timezone.utc).isoformat(), passed=True, pages=6, table_pages=[row['start_page'] for row in layout['tables']],
        visual_review='全部六页及新增图件已查看，中文、表格、图例、边界与分页正常',
        docx_sha256=sha256(docx_path), pdf_sha256=sha256(pdf), model_input_sha256={name: sha256(project / name) for name in model_files},
        chapter4_chapter5_hourly_and_window_IT_differences_zero=True, task_layers_conserved=True, native_IT_summaries_verified=len(source_summary),
        word_native_export=layout, verification_sha256=sha256(output / 'verification.json'))
    (output / 'brief-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--word-audit', type=Path, required=True)
    parser.add_argument('--visual-qa-confirmed', action='store_true')
    args = parser.parse_args()
    audit(args.project, args.pdf, args.word_audit, args.visual_qa_confirmed)
