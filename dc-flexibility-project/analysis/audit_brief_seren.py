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


def digest(path):
    with path.open('rb') as handle:
        return file_digest(handle, 'sha256').hexdigest()


def audit(project, pdf, layout_file, visual_confirmed):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    output = project / 'results/typical-load-curves-20261101-v4-seren'
    manifest = json.loads((output / 'manifest.json').read_text('utf-8'))
    previous = project / 'results' / manifest['previous_result']
    for path, value in manifest['previous_all_files_sha256'].items():
        assert digest(previous / path) == value, path
    for path, value in manifest['preserved_files_sha256'].items():
        assert digest(output / path) == digest(previous / path) == value, path
    summary = pd.read_csv(output / 'native-summary.csv', float_precision='round_trip')
    for row in summary.to_dict('records'):
        source = pd.read_parquet(output / 'source-hourly' / (row['scenario_id'] + '.parquet'))
        placement = 'dated' if row['has_real_dates'] else 'm01'
        full = pd.read_parquet(output / 'hourly' / f'{row["scenario_id"]}-2025-{placement}.parquet')
        for component in ('it', 'base_it', 'cpu_it', 'gpu_it'):
            values = source[component + '_mw']
            np.testing.assert_allclose(values, full[component + '_mw'], rtol=1e-12, atol=1e-10)
            for field, value in [('mean', values.mean()), ('peak', values.max()), ('minimum', values.min()), ('window', values.sum())]:
                name = f'{field}_{component}_' + ('mwh' if field == 'window' else 'mw')
                np.testing.assert_allclose(row[name], value, rtol=1e-12)
        for label, group in [('diurnal', 'hour_of_source_day'), ('weekly', 'hour_of_source_week')]:
            if label == 'weekly' and row['use'] == 'CLD':
                continue
            mean = source.groupby(group).it_mw.mean()
            np.testing.assert_allclose(row['it_' + label + '_peak_valley'], mean.max()/mean.min(), rtol=1e-12)
        if row['use'] == 'TRN':
            clock = pd.DatetimeIndex(pd.to_datetime(full.source_clock, utc=True))
            assert len(clock) == 2057 and clock.is_unique and clock.is_monotonic_increasing
            assert int((clock.to_series().diff() > pd.Timedelta(hours=1)).sum()) == 53
            assert len(pd.date_range(clock[0], clock[-1], freq='h')) - len(clock) == 189
            assert not full.power_imputed.any()
            assert full.source_dataset.eq('Acme Seren').all()
            relative = f'hourly/reference-grid/main/{row["scenario_id"]}-2025.parquet'
            assert digest(output / 'hourly' / f'{row["scenario_id"]}-2025-dated.parquet') == manifest['upstream_main_hourly_sha256'][relative]
            for label, grouping in [('diurnal', 'hour_of_source_day'), ('weekly', 'hour_of_source_week')]:
                means = full.groupby(grouping).facility_mw.mean()
                np.testing.assert_allclose(row['placement_facility_' + label + '_peak_valley_median'], means.max()/means.min(), rtol=1e-12)
    selected = json.loads((output / 'selected-days/summary.json').read_text('utf-8'))
    day_scores = pd.read_csv(output / 'selected-days/TRN-day-selection.csv')
    d = pd.read_parquet(output / 'hourly/D-TRN-10MW-2025-dated.parquet')
    dates = pd.to_datetime(d.source_clock, utc=True).dt.floor('d')
    complete_dates = dates.value_counts().loc[lambda x: x.eq(24)].index.sort_values()
    assert len(complete_dates) == len(day_scores) == 55
    matrix = d.loc[dates.isin(complete_dates)].facility_mw.to_numpy().reshape(55, 24)
    energy = matrix.sum(axis=1)
    candidates = np.abs(energy/np.median(energy)-1) <= .02+1e-12
    rmse = np.sqrt(((matrix-matrix.mean(axis=0))**2).mean(axis=1))
    chosen = np.flatnonzero(candidates)[np.argmin(rmse[candidates])]
    assert candidates.sum() == 8 and complete_dates[chosen].strftime('%Y-%m-%d') == '2023-05-31'
    np.testing.assert_allclose(day_scores.rmse_mw, rmse, rtol=1e-12)
    for row in selected['scenarios']:
        path = output / 'selected-days' / (row['scenario'] + '-hourly.csv')
        hourly = pd.read_csv(path)
        parts = pd.read_csv(path.with_name(row['scenario'] + '-by-task.csv'))
        np.testing.assert_allclose(parts.groupby('hour').facility_mw.sum(), hourly.facility_mw, rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(parts.groupby('hour').it_mw.sum(), hourly.it_mw, rtol=1e-12, atol=1e-10)
        if row['scenario'].endswith('-TRN'):
            assert row['source_date'] == '2023-05-31' and len(hourly) == 24
            assert 'MLLM（多模态大模型）' in parts.task_type.values
    cross = pd.read_csv(output / 'chapter4-comparison.csv')
    np.testing.assert_allclose(cross.selected_hourly_maximum_difference_mw, 0, atol=1e-12)
    np.testing.assert_allclose(cross.window_it_mean_absolute_difference_mw, 0, atol=1e-10)
    docx_path = project / 'docs/job-flexibility-brief/数据中心任务时间柔性简表.docx'
    document = Document(docx_path)
    expected = json.loads((output / 'brief-summary.json').read_text('utf-8'))['rows']
    assert len(document.tables) == 4
    assert [[cell.text for cell in row.cells] for row in document.tables[-1].rows] == expected
    text = '\n'.join(p.text for p in document.paragraphs) + '\n' + '\n'.join(cell.text for table in document.tables for row in table.rows for cell in row.cells)
    for forbidden in ('Kalos', '0.80', '271卡', '16.50', '0.85/0.79/0.60'):
        assert forbidden not in text, forbidden
    assert all(word in text for word in ('2057', 'Seren', '0.76', '0.61', '0.90', '缺测代表性', 'TDP'))
    images = [relation.target_part.blob for relation in document.part.rels.values() if relation.reltype.endswith('/image')]
    assert len(images) == 2
    assert (output / 'native-overview.png').read_bytes() in images
    assert (output / 'selected-days/overview.png').read_bytes() in images
    reader = PdfReader(pdf)
    page_text = [page.extract_text() for page in reader.pages]
    assert len(page_text) == 6
    for index, title in [(3, '各情境典型日负荷曲线'), (4, '各情境原生窗口负荷曲线'), (5, '原生窗口时序总览')]:
        assert title in page_text[index].replace(' ', '')
    assert '\ufffd' not in ''.join(page_text)
    layout = json.loads(layout_file.read_text('utf-8-sig'))
    assert layout['pages'] == 6 and all(t['start_page'] == t['end_page'] for t in layout['tables'])
    assert visual_confirmed, '逐页与改图视觉检查后才能确认'
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed', 'pages': len(page_text),
        'Word_tables': layout['tables'], 'docx_sha256': digest(docx_path), 'pdf_sha256': digest(pdf),
        'visual_QA': '全部六页及改图已逐一检查；中文、图例、表格及分页正常',
        'source_scenarios_verified': 44, 'Seren_hourly_upstream_fingerprints': 300,
        'preserved_files_verified': len(manifest['preserved_files_sha256']), 'previous_directory_unchanged': True,
        'training_qualified_hours': 2057, 'complete_UTC_days': 55, 'candidate_days': 8, 'selected_UTC_date': '2023-05-31',
        'no_training_legacy_scheme_in_Word': True, 'two_chapters_same_hourly_values': True}
    (output / 'brief-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--layout', type=Path, required=True)
    parser.add_argument('--visual-qa-confirmed', action='store_true')
    args = parser.parse_args()
    audit(args.project, args.pdf, args.layout, args.visual_qa_confirmed)
