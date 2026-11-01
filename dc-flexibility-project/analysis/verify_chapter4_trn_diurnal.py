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
from load_curve_scenarios_v3 import catalogue, TDP
from typical_power_models import PARAMETERS
from update_chapter4_trn_diurnal import RUN, PREVIOUS, UPSTREAM, TRN


def verify(args):
    project = args.project
    output, previous = project / 'results' / RUN, project / 'results' / PREVIOUS
    summary = json.loads((output / 'summary.json').read_text('utf-8'))
    before = json.loads((previous / 'summary.json').read_text('utf-8'))
    manifest = json.loads((output / 'manifest.json').read_text('utf-8'))
    audit = json.loads((project / 'results' / UPSTREAM / 'native-calculation-audit.json').read_text('utf-8'))
    checks = {}
    for source in manifest['input_files']:
        assert sha256(Path(source['file'])) == source['sha256']
    for name, digest in manifest['preserved_v5_files_sha256'].items():
        assert sha256(output / name) == sha256(previous / name) == digest
    old_rows = {row['scenario']: row for row in before['scenarios']}
    for row in summary['scenarios']:
        if row['scenario'] not in TRN:
            assert row == old_rows[row['scenario']]
    for use in ('HPC', 'CLD', 'SAI', 'INF'):
        assert summary['selections'][use] == before['selections'][use]
    checks['all_inputs_and_eight_v5_scenarios_unchanged'] = True

    configs = catalogue(project)
    cpu_shape = json.loads((project / 'results/trn-power-model-20261101/cpu-shape.json').read_text('utf-8'))
    comparison = pd.read_csv(output / 'comparison.csv').set_index('scenario')
    upstream = pd.read_csv(project / 'results' / UPSTREAM / 'window-summary.csv').query('reference_it_mw == 10').set_index(['server_type', 'case'])
    sensitivities = pd.read_csv(output / 'sensitivity-summary.csv').set_index(['server_type', 'case'])
    period = pd.read_csv(output / 'period-summary.csv').set_index('scenario')
    profile = pd.read_csv(project / 'results' / UPSTREAM / 'transfer-profile.csv')
    resource_file = next(item['file'] for item in manifest['input_files'] if item['file'].endswith('TRN-resources.parquet'))
    resources = pd.read_parquet(resource_file)
    selected = summary['selections']['TRN']
    full_reference = pd.read_parquet(project / 'data/processed' / RUN / 'D-TRN-10MW-2025-dated.parquet')
    values = full_reference.facility_mw.to_numpy().reshape(90, 24)
    energies, mean = values.sum(axis=1), values.mean(axis=0)
    deviations = 100*(energies/np.median(energies)-1)
    rmse = np.sqrt(((values-mean)**2).mean(axis=1))
    primary = np.abs(deviations) <= 2+1e-10
    width = 2 if primary.sum() >= 5 else 5
    candidates = np.flatnonzero(np.abs(deviations) <= width+1e-10)
    day = min(candidates, key=lambda index: (rmse[index], abs(energies[index]-np.median(energies)), index))
    assert day == selected['day_index'] and width == selected['applied_window_pct']
    assert selected['candidate_count_at_2pct'] == int(primary.sum())
    assert selected['candidate_count'] == len(candidates)
    assert day == min(np.flatnonzero(primary), key=lambda index: (rmse[index], abs(deviations[index]), index))
    scores = pd.read_csv(output / 'TRN-day-selection.csv')
    for column, expected in [('facility_mwh', energies), ('rmse_mw', rmse), ('energy_deviation_pct', deviations)]:
        np.testing.assert_allclose(scores[column], expected, rtol=1e-12, atol=1e-12)
    assert scores.selected.sum() == 1 and scores.selected.iloc[day]
    assert selected['source_date'] == pd.Timestamp(full_reference.source_clock.iloc[day*24]).strftime('%Y-%m-%d')
    constant_file = project / 'data/processed' / RUN / 'upstream/constant_080/D-TRN-10MW-2025-dated.parquet'
    constant_values = pd.read_parquet(constant_file).facility_mw.to_numpy().reshape(90,24)
    constant_energy = constant_values.sum(axis=1)
    constant_errors = np.sqrt(((constant_values-constant_values.mean(axis=0))**2).mean(axis=1))
    constant_distance = abs(constant_energy/np.median(constant_energy)-1)
    constant_width = .02 if np.sum(constant_distance<=.02+1e-12)>=5 else .05
    constant_day = min(np.flatnonzero(constant_distance<=constant_width+1e-12),
        key=lambda index:(constant_errors[index],constant_distance[index],index))
    assert summary['TRN_constant_080_selection']['day_index'] == constant_day == day
    checks['independent_90_day_selection_window_RMSE_and_shared_date'] = True

    for scenario in TRN:
        key = scenario[0]
        config = configs[key]
        full_path = project / 'data/processed' / RUN / f'{scenario}-10MW-2025-dated.parquet'
        assert sha256(full_path) == audit['output_hourly_sha256'][f'hourly/reference-grid/shape_central/{full_path.name}']
        frame = pd.read_parquet(full_path)
        row = next(row for row in summary['scenarios'] if row['scenario'] == scenario)
        assert row['source_date'] == selected['source_date']
        ratio = profile.shape_central.to_numpy()[frame.hour_of_source_day.to_numpy(int)]
        np.testing.assert_allclose(np.average(ratio, weights=frame.allocated_gpu_fraction), .8, rtol=1e-12)
        units = int(frame.unit_count.iloc[0])
        gpu = config['gpu_devices_per_unit']*units/1e6*(
            frame.allocated_gpu_fraction*TDP[key]*ratio + (1-frame.allocated_gpu_fraction)*(70 if key == 'D' else 50))
        np.testing.assert_allclose(gpu, frame.gpu_it_mw, rtol=1e-12)
        base, idle_cpu, loaded_cpu, _, _ = PARAMETERS[config['configuration_id']]['central']
        u = frame.u_cpu_proxy
        c1, c2 = cpu_shape['coefficients']
        response = 3*c1*u*(1-u)**2 + 3*c2*u**2*(1-u) + u**3
        cpu = config['cpu_packages_per_unit']*units/1e6*(idle_cpu+(loaded_cpu-idle_cpu)*response)
        np.testing.assert_allclose(cpu, frame.cpu_it_mw, rtol=1e-12)
        np.testing.assert_allclose(frame.it_mw, frame.gpu_it_mw+frame.cpu_it_mw+frame.base_it_mw, rtol=1e-12)
        np.testing.assert_allclose(frame.facility_mw, frame.it_mw+frame.cooling_mw+frame.other_mw, rtol=1e-12)
        assert frame.source_clock.iloc[0].startswith('2023-05-18') and frame.source_clock.iloc[-1].startswith('2023-08-15')
        source_clock = pd.DatetimeIndex(pd.to_datetime(frame.source_clock, utc=True))
        weather_clock = pd.DatetimeIndex(frame.time_utc)
        assert source_clock.strftime('%m-%d %H').tolist() == weather_clock.strftime('%m-%d %H').tolist()
        checks[scenario + '_exact_upstream_hourly_GPU_CPU_formula_UTC_weather_and_balance'] = True
        start = day*24
        hour = pd.read_csv(output / (scenario+'-hourly.csv'))
        assert hour.hour.tolist() == list(range(24))
        current = frame.iloc[start:start+24].reset_index(drop=True)
        for column in ('it_mw', 'facility_mw', 'pue', 'allocated_gpu_power_ratio'):
            np.testing.assert_allclose(hour[column], current[column], rtol=1e-12)
        layers = pd.read_csv(output / (scenario+'-by-task.csv'))
        for column in ('it_mw', 'facility_mw'):
            np.testing.assert_allclose(layers.groupby('hour')[column].sum(), hour[column], atol=1e-10)
        assert layers.facility_mw.min() >= -1e-10 and layers.groupby('task_type').size().eq(24).all()
        selected_resources = resources.iloc[start:start+24].reset_index(drop=True)
        for label in row['composition']:
            share = np.divide(selected_resources['gpu:'+label], current.allocated_gpu_fraction,
                out=np.zeros(24), where=current.allocated_gpu_fraction.to_numpy()>0)
            expected = (current.it_mw-current.idle_it_mw)*share
            actual = layers[layers.task_type.eq(label)].sort_values('hour')
            np.testing.assert_allclose(actual.it_mw, expected, atol=1e-10)
        for version, record in [('old', old_rows[scenario]), ('new', row)]:
            for name in ('peak_mw', 'minimum_mw', 'peak_valley_ratio', 'daily_facility_mwh'):
                np.testing.assert_allclose(comparison.loc[scenario, version+'_'+name], record[name], rtol=1e-12)
        np.testing.assert_allclose(period.loc[scenario, 'mean_daily_facility_mwh'], frame.facility_mw.mean()*24, rtol=1e-12)
        assert .5 < period.loc[scenario, 'daily_to_period_mean_ratio'] < 2
        checks[scenario + '_typical_hourly_task_layers_comparison_and_90_day_energy_scale'] = True
        for case in profile.filter(regex='^shape_').columns.tolist()+['constant_080']:
            expected = upstream.loc[(key, case)]
            actual = sensitivities.loc[(key, case)]
            for name in ('mean_it_mw', 'mean_facility_mw', 'peak_it_mw', 'peak_facility_mw',
                         'minimum_it_mw', 'minimum_facility_mw', 'it_diurnal_peak_valley', 'facility_diurnal_peak_valley'):
                np.testing.assert_allclose(actual[name], expected[name], rtol=1e-12)
        checks[scenario+'_all_seven_sensitivity_cases_match_PR36'] = True
    diagnostics = pd.read_csv(output / 'selected-day-diagnostics.csv')
    assert diagnostics.audit_flags.isna().all() and diagnostics.unresolved_hours.eq(0).all()
    checks['new_TRN_days_no_quality_or_load_shape_flags'] = True
    for name in TRN+['overview']:
        with Image.open(output / (name+'.png')) as image:
            np.testing.assert_allclose(image.info['dpi'], [300, 300], atol=.02)
        ET.parse(output / (name+'.svg'))
    checks['four_changed_figures_300dpi_and_valid_SVG'] = True

    docs = project / 'docs/job-flexibility-brief'
    docx = docs / '数据中心任务时间柔性简表.docx'
    namespace = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    with ZipFile(docx) as archive:
        root = ET.fromstring(archive.read('word/document.xml'))
        media = [name for name in archive.namelist() if name.startswith('word/media/') and not name.endswith('/')]
        assert len(media) == 1 and archive.read(media[0]) == (output/'overview.png').read_bytes()
    old_blob = subprocess.check_output(['git','show','3b11b58:dc-flexibility-project/docs/job-flexibility-brief/数据中心任务时间柔性简表.docx'], cwd=project.parent)
    with ZipFile(BytesIO(old_blob)) as archive:
        old_root = ET.fromstring(archive.read('word/document.xml'))
    tables, old_tables = root.findall('.//'+namespace+'tbl'), old_root.findall('.//'+namespace+'tbl')
    def text(node):
        return ''.join(part.text or '' for part in node.iter(namespace+'t'))
    assert text(tables[0]) == text(old_tables[0]) and text(tables[1]) == text(old_tables[1])
    changed_rows = [index for index, (new, old) in enumerate(zip(tables[2], old_tables[2])) if text(new) != text(old)]
    assert len(changed_rows) == 1 and 'Acme' in text(list(tables[2])[changed_rows[0]])
    readme = (docs/'README.md').read_text('utf-8')
    assert readme.count('## 数字来源与说明') == readme.count('## 重新生成') == 1
    assert readme.count('- 第四章图与逐小时分任务汇总：') == 1
    pages = PdfReader(args.pdf).pages
    assert len(pages) == 4
    page_text = [page.extract_text().replace(' ', '').replace('\n','') for page in pages]
    for index, heading in enumerate(('一、情境定义', '二、任务类型', '三、各数据源', '四、各情境')):
        assert heading in page_text[index]
    for content in ('Kalos', '271', '反推', '0.730', '95%–99.9%', '少于5天放宽至±5%'):
        assert content in page_text[3]
    assert '中央日形状0.511–0.588' in page_text[1]
    assert '集群GPU监测（无作业映射）' in page_text[2]
    checks['Word_preserved_tables_only_Acme_row_updated_new_image_and_four_page_PDF'] = True
    paths = ['dc-flexibility-project/results/'+name for name in (
        PREVIOUS, 'scenario-typical-day-by-task-20261101-v4', 'scenario-typical-day-by-task-20261101-v3', 'scenario-typical-day-by-task-20261031')]
    assert not subprocess.check_output(['git','diff','--name-only','3b11b58','--',*paths], cwd=project.parent, text=True).strip()
    checks['all_earlier_chapter4_result_versions_preserved'] = True
    manifest['documents_sha256'] = {name: hashlib.sha256((docs/name).read_text('utf-8').encode('utf-8')).hexdigest() for name in ('build.js','README.md')}
    manifest['documents_sha256'][docx.name] = sha256(docx)
    manifest['code_sha256'][Path(__file__).name] = sha256(Path(__file__))
    manifest['pdf_QA'] = {'pages':len(pages),'sha256':sha256(args.pdf),'visual_qa_confirmed':args.visual_qa_confirmed}
    manifest['output_sha256'] = {path.name:sha256(path) for path in sorted(output.iterdir()) if path.is_file() and path.name not in ('manifest.json','independent-verification.json')}
    manifest['document_text_hash_basis'] = 'UTF-8/LF; docx original binary bytes'
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf-8')
    evidence = {'TRN_source_date':selected['source_date'], 'candidate_count_at_2pct':int(primary.sum()),
        'applied_window_pct':width,'candidate_count':len(candidates),
        'selected_reference_deviation_pct':float(deviations[day]), 'selected_reference_rmse_mw':float(rmse[day]),
        'new_to_period_mean_daily_energy_range':[float(period.daily_to_period_mean_ratio.min()),float(period.daily_to_period_mean_ratio.max())]}
    (output/'independent-verification.json').write_text(json.dumps({'status':'passed','utc':datetime.now(timezone.utc).isoformat(),
        'checks':checks,'evidence':evidence},ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps({'checks_passed':len(checks),'evidence':evidence},ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--visual-qa-confirmed', action='store_true')
    verify(parser.parse_args())
