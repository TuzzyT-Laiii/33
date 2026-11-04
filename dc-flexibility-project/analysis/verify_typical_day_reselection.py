import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader

from compute_load.model import sha256


BASE = 'scenario-typical-day-by-task-20261101-v4'
RUN = 'scenario-typical-day-by-task-20261101-v5'


def verify(args):
    project = args.project
    output, previous = project / 'results' / RUN, project / 'results' / BASE
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    old_summary = json.loads((previous / 'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    assert len(summary['scenarios']) == 11
    assert summary['selection_parameters'] == {'primary_window': .02, 'fallback_window': .05, 'minimum_candidates': 5}
    assert sha256(previous / 'manifest.json') == manifest['v4_manifest_sha256']
    for path, digest in manifest['frozen_annual_hourly_sha256'].items():
        assert sha256(Path(path)) == digest, path
    for source in manifest['resource_inputs']:
        assert sha256(Path(source['file'])) == source['sha256']
    checks = {'v4_annual_and_classified_resource_hourly_fingerprints_unchanged': True}
    reference_checks = []
    for use, selection in summary['selections'].items():
        frame = pd.read_parquet(project / 'data/processed' / BASE / f"{selection['reference_scenario']}-10MW-2025.parquet")
        count = selection['complete_days']
        values = frame.facility_mw.to_numpy()[:count*24].reshape(count, 24)
        energy = values.sum(axis=1)
        median = np.median(energy)
        difference = 100 * (energy / median - 1)
        primary = np.abs(difference) <= 2 + 1e-10
        width = 2 if int(primary.sum()) >= 5 else 5
        candidates = np.flatnonzero(np.abs(difference) <= width + 1e-10)
        errors = np.sqrt(((values - values.mean(axis=0))**2).mean(axis=1))
        winner = min(candidates, key=lambda day: (errors[day], abs(energy[day]-median), day))
        assert winner == selection['day_index']
        assert len(candidates) == selection['candidate_count']
        assert int(primary.sum()) == selection['candidate_count_at_2pct']
        assert width == selection['applied_window_pct']
        scores = pd.read_csv(output / f'{use}-day-selection.csv')
        np.testing.assert_allclose(scores.facility_mwh, energy, rtol=1e-12)
        np.testing.assert_allclose(scores.energy_deviation_pct, difference, atol=1e-12)
        np.testing.assert_allclose(scores.rmse_mw, errors, rtol=1e-12)
        assert scores.selected.sum() == 1 and bool(scores.iloc[winner].selected)
        assert scores.candidate.to_numpy().tolist() == (np.abs(difference) <= width+1e-10).tolist()
        if use == 'HPC':
            old_day = old_summary['selections']['HPC']['day_index']
            assert primary[old_day] and np.isclose(energy[old_day], median)
            assert winner != old_day and errors[winner] < errors[old_day]
            april = scores[scores.source_day.str.startswith('2022-04-27')].iloc[0]
            assert april.candidate and errors[winner] < april.rmse_mw
            reference_checks.append({'HPC_20220427_rmse_mw': float(april.rmse_mw), 'HPC_selected_rmse_mw': float(errors[winner])})
        if use == 'CLD':
            assert int(primary.sum()) == 4 and width == 5 and len(candidates) == 5
        checks[use + '_actual_candidate_window_fallback_RMSE_and_ties'] = True

    comparison = pd.read_csv(output / 'day-selection-comparison.csv').set_index('scenario')
    diagnostics = pd.read_csv(output / 'selected-day-diagnostics.csv').fillna({'audit_flags': ''})
    annual = pd.read_csv(output / 'annual-summary.csv').set_index('scenario')
    energy_ratios = []
    for row in summary['scenarios']:
        scenario = row['scenario']
        use = scenario.split('-')[1]
        selection = summary['selections'][use]
        assert row['source_date'] == selection['source_date']
        frame = pd.read_parquet(project / 'data/processed' / BASE / f'{scenario}-10MW-2025.parquet')
        start = selection['day_index'] * 24
        selected = frame.iloc[start:start+24].reset_index(drop=True)
        hourly = pd.read_csv(output / (scenario + '-hourly.csv'))
        assert hourly.hour.tolist() == list(range(24)) and hourly.source_date.eq(row['source_date']).all()
        for column in ('it_mw', 'facility_mw', 'pue'):
            np.testing.assert_allclose(hourly[column], selected[column], rtol=1e-12)
        parts = pd.read_csv(output / (scenario + '-by-task.csv'))
        assert parts.groupby('task_type').size().eq(24).all()
        assert min(parts.it_mw.min(), parts.facility_mw.min()) >= -1e-8
        for column in ('it_mw', 'facility_mw'):
            np.testing.assert_allclose(parts.groupby('hour')[column].sum(), hourly[column], atol=1e-8)
        np.testing.assert_allclose(frame.facility_mw.sum(), annual.loc[scenario, 'annual_facility_mwh'], rtol=1e-12)
        matrix = frame.facility_mw.to_numpy()[:row['complete_days']*24].reshape(-1, 24)
        median = np.median(matrix.sum(axis=1))
        for version, index in (('old', old_summary['selections'][use]['day_index']), ('new', selection['day_index'])):
            curve = matrix[index]
            actual = comparison.loc[scenario]
            expected = {'energy_mwh': curve.sum(), 'deviation_pct': 100*(curve.sum()/median-1),
                'rmse_mw': np.sqrt(np.mean((curve-matrix.mean(axis=0))**2)), 'peak_valley_ratio': curve.max()/curve.min()}
            for name, value in expected.items():
                np.testing.assert_allclose(actual[version + '_' + name], value, rtol=1e-10, atol=1e-10)
        energy_ratios.append(float(hourly.facility_mw.sum()*365/frame.facility_mw.sum()))
        assert .5 < energy_ratios[-1] < 2
        with Image.open(output / (scenario + '.png')) as image:
            np.testing.assert_allclose(image.info['dpi'], [300, 300], atol=.02)
        ET.parse(output / (scenario + '.svg'))
        checks[scenario + '_frozen_values_shared_date_layer_balance_and_comparison_metrics'] = True
    assert len(diagnostics) == 22
    old_flags = diagnostics.query('version == "old" and audit_flags != ""')
    new_flags = diagnostics.query('version == "new" and audit_flags != ""')
    assert old_flags.scenario.tolist() == ['A-HPC'] and new_flags.empty
    assert old_flags.longest_low_IT_hours.iloc[0] >= 3 and old_flags.max_IT_step_fraction.iloc[0] > .30
    for name in ('annual-summary.csv', 'brief-facts.json', 'idle-endpoints.csv'):
        assert (output / name).read_bytes() == (previous / name).read_bytes(), name
    checks['all_22_old_new_days_screened_and_model_parameter_tables_identical'] = True

    docs = project / 'docs/job-flexibility-brief'
    readme = (docs / 'README.md').read_text(encoding='utf-8')
    assert readme.count('## 数字来源与说明') == readme.count('## 重新生成') == 1
    assert readme.count('- 第四章图与逐小时分任务汇总：') == 1
    assert '少于5天放宽至±5%' in readme and '中位电量±2%' in readme
    assert '完整日取日电量最接近中位数者' not in readme
    namespace = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    def body_text(blob):
        with ZipFile(BytesIO(blob)) as archive:
            document = ET.fromstring(archive.read('word/document.xml'))
            return ''.join(node.text or '' for node in document.iter(namespace+'t'))
    docx = docs / '数据中心任务时间柔性简表.docx'
    current_text = body_text(docx.read_bytes())
    old_blob = subprocess.check_output(['git', 'show', 'a67ebd0:dc-flexibility-project/docs/job-flexibility-brief/数据中心任务时间柔性简表.docx'], cwd=project.parent)
    assert current_text.split('四、各情境典型日负荷曲线')[0] == body_text(old_blob).split('四、各情境典型日负荷曲线')[0]
    with ZipFile(docx) as archive:
        media = [item.filename for item in archive.infolist() if item.filename.startswith('word/media/') and not item.is_dir()]
        assert len(media) == 1 and archive.read(media[0]) == (output / 'overview.png').read_bytes()
    pages = PdfReader(args.pdf).pages
    assert len(pages) == 4
    text = [page.extract_text().replace(' ', '').replace('\n', '') for page in pages]
    for index, heading in enumerate(('一、情境定义', '二、任务类型', '三、各数据源', '四、各情境')):
        assert heading in text[index]
    assert '少于5天放宽至±5%' in text[3] and '中位电量±2%' in text[3]
    checks['Word_first_three_chapters_preserved_new_rule_image_and_four_page_PDF'] = True
    unchanged = ['dc-flexibility-project/results/' + value for value in (BASE, 'scenario-typical-day-by-task-20261101-v3', 'scenario-typical-day-by-task-20261031')]
    assert not subprocess.check_output(['git', 'diff', '--name-only', 'a67ebd0', '--', *unchanged], cwd=project.parent, text=True).strip()
    checks['all_previous_version_result_bytes_preserved'] = True
    manifest['code_sha256'] = {name: sha256(project / 'analysis' / name) for name in manifest['code_sha256']}
    manifest['code_sha256'][Path(__file__).name] = sha256(Path(__file__))
    manifest['documents_sha256'] = {name: hashlib.sha256((docs / name).read_text(encoding='utf-8').encode('utf-8')).hexdigest() for name in ('build.js', 'README.md')}
    manifest['documents_sha256'][docx.name] = sha256(docx)
    manifest['document_text_hash_basis'] = 'UTF-8/LF；Word原始二进制字节'
    manifest['pdf_QA'] = {'pages': 4, 'sha256': sha256(args.pdf), 'visual_qa_confirmed': args.visual_qa_confirmed}
    manifest['output_sha256'] = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file() and path.name not in ('manifest.json', 'independent-verification.json')}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    result = {'status': 'passed', 'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'checks': checks,
        'evidence': {'new_to_annual_mean_daily_energy_range': [min(energy_ratios), max(energy_ratios)],
            'old_flagged_scenarios': old_flags.scenario.tolist(), 'new_flagged_scenarios': [], 'HPC_reference': reference_checks}}
    (output / 'independent-verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'checks_passed': len(checks), 'evidence': result['evidence']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--visual-qa-confirmed', action='store_true')
    verify(parser.parse_args())
