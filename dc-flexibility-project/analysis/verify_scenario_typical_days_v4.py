import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader

from compute_load.model import sha256
from load_curve_scenarios_v3 import catalogue
from typical_power_models import PARAMETERS


RUN = 'scenario-typical-day-by-task-20261101-v4'
PREVIOUS = 'scenario-typical-day-by-task-20261101-v3'


def verify(args):
    project = args.project
    output = project / 'results' / RUN
    local = project / 'data/processed' / RUN
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    old_summary = json.loads((project / 'results' / PREVIOUS / 'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    assert len(summary['scenarios']) == 11
    inputs = {entry['role']: Path(entry['file']) for entry in manifest['input_files']}
    for entry in manifest['input_files']:
        assert sha256(Path(entry['file'])) == entry['sha256'], entry['role']
    for filename, digest in manifest['local_annual_hourly_sha256'].items():
        assert sha256(local / filename) == digest, filename
    checks = {'source_and_local_annual_hourly_SHA256': True}
    configs = catalogue(project)
    shape = json.loads(inputs['PR36_CPU_shape'].read_text(encoding='utf-8'))
    comparisons = pd.read_csv(output / 'comparison.csv').set_index('scenario')
    matched = pd.read_csv(output / 'matched-day-comparison.csv')
    annual = pd.read_csv(output / 'annual-summary.csv').set_index('scenario')
    sensitivities = pd.read_csv(output / 'sensitivity-summary.csv')
    upstream_sensitivity = pd.read_csv(project / 'results/trn-power-model-20261101/sensitivity-summary.csv')
    ratios = []
    for use, selection in summary['selections'].items():
        key = 'A' if use in ('HPC', 'CLD') else 'D'
        full = pd.read_parquet(local / f'{key}-{use}-10MW-2025.parquet')
        daily = full.facility_mw.to_numpy()[:selection['complete_days'] * 24].reshape(-1, 24)
        energy = daily.sum(axis=1)
        distance = np.abs(energy - np.median(energy))
        ties = np.flatnonzero(np.abs(distance - distance.min()) <= 1e-9)
        errors = np.sqrt(np.mean((daily - daily.mean(axis=0)) ** 2, axis=1))
        selected = sorted(ties, key=lambda index: (errors[index], index))[0]
        assert selection['day_index'] == selected
        table = pd.read_csv(output / (use + '-day-selection.csv'))
        np.testing.assert_allclose(table.facility_mwh, energy, rtol=1e-12)
        assert table.selected.sum() == 1 and table.iloc[selected].selected
        checks[use + '_median_then_RMSE_independent_selection'] = True

    maximum_formula_error = 0.
    endpoints = pd.read_csv(output / 'idle-endpoints.csv').set_index('scenario')
    mit = json.loads(inputs['MIT_calibration'].read_text(encoding='utf-8'))
    for row in summary['scenarios']:
        scenario = row['scenario']
        key, use = scenario.split('-')
        config = configs[key]
        for case in ('low', 'central', 'high'):
            base, ci, _, gi, _ = PARAMETERS[config['configuration_id']][case]
            if use == 'SAI':
                gi = (700. if key == 'D' else 350.) * mit['idle_W_per_gpu'] / mit['tdp_W']
            expected_idle = base + config['cpu_packages_per_unit'] * ci + config['gpu_devices_per_unit'] * gi
            np.testing.assert_allclose(endpoints.loc[scenario, case + '_unit_w'], expected_idle, rtol=1e-12)
        full = pd.read_parquet(local / f'{scenario}-10MW-2025.parquet')
        old = pd.read_parquet(inputs[scenario + '_v3_hourly'])
        power36 = pd.read_parquet(inputs[scenario + '_PR36_hourly'])
        u = full.u_cpu_proxy.to_numpy()
        c1, c2 = shape['coefficients']
        response = 3 * c1 * u * (1-u)**2 + 3 * c2 * u*u * (1-u) + u**3
        base_w, cpu_idle, cpu_full, _, _ = PARAMETERS[config['configuration_id']]['central']
        units = int(full.unit_count.iloc[0])
        expected_cpu = config['cpu_packages_per_unit'] * (cpu_idle + (cpu_full - cpu_idle) * response) * units / 1e6
        np.testing.assert_allclose(full.cpu_it_mw, expected_cpu, rtol=1e-12)
        np.testing.assert_array_equal(full.cpu_it_mw, power36.cpu_it_mw)
        if use == 'INF':
            request = pd.read_parquet(inputs[scenario + '_PR35_hourly'])
            for column in ('gpu_it_mw', 'online_gpu_w_each', 'request_intensity', 'source_quality_flag'):
                np.testing.assert_array_equal(full[column], request[column])
            np.testing.assert_allclose(full.it_mw.sum(), power36.it_mw.sum(), rtol=1e-12)
            np.testing.assert_allclose(full.gpu_it_mw.sum(), old.gpu_it_mw.sum(), rtol=1e-12)
        else:
            for column in ('it_mw', 'facility_mw', 'pue', 'gpu_it_mw'):
                np.testing.assert_array_equal(full[column], power36[column])
        if use == 'TRN':
            tdp, idle_each = (700., 70.) if key == 'D' else (350., 50.)
            expected_gpu = config['gpu_devices_per_unit'] * units / 1e6 * (full.allocated_gpu_fraction * tdp * .80 + (1-full.allocated_gpu_fraction) * idle_each)
            np.testing.assert_allclose(full.gpu_it_mw, expected_gpu, rtol=1e-12)
        for column in ('unit_count', 'source_hour_index', 'source_unresolved', 'u_cpu_proxy', 'allocated_gpu_fraction', 'temperature_c'):
            np.testing.assert_array_equal(full[column], old[column])
        np.testing.assert_allclose(full.it_mw, full.base_it_mw + full.cpu_it_mw + full.gpu_it_mw, atol=1e-9)
        np.testing.assert_allclose(full.facility_mw, full.it_mw + full.cooling_mw + full.other_mw, atol=1e-9)
        np.testing.assert_allclose(full.pue, full.facility_mw / full.it_mw, rtol=1e-12)
        assert full.heat_balance_residual_mw.abs().max() < 1e-8
        np.testing.assert_allclose(full.facility_mw.sum(), annual.loc[scenario, 'annual_facility_mwh'], rtol=1e-12)
        start = 24 * summary['selections'][use]['day_index']
        selected = full.iloc[start:start+24].reset_index(drop=True)
        hourly = pd.read_csv(output / (scenario + '-hourly.csv'))
        assert hourly.hour.tolist() == list(range(24))
        for column in ('it_mw', 'facility_mw', 'pue'):
            np.testing.assert_allclose(hourly[column], selected[column], rtol=1e-12)
        parts = pd.read_csv(output / (scenario + '-by-task.csv'))
        assert parts.groupby('task_type').size().eq(24).all()
        np.testing.assert_allclose(parts.groupby('hour').it_mw.sum(), hourly.it_mw, atol=1e-8)
        np.testing.assert_allclose(parts.groupby('hour').facility_mw.sum(), hourly.facility_mw, atol=1e-8)
        cooling = parts[parts.component_scope.eq('non_IT')].sort_values('hour')
        np.testing.assert_allclose(cooling.facility_mw, hourly.facility_mw-hourly.it_mw, atol=1e-8)
        resource = pd.read_parquet(inputs[use + '_local_classified_resource_hours']).iloc[start:start+24].reset_index(drop=True)
        if use == 'INF':
            cards = config['gpu_devices_per_unit'] * units
            idle_each = 70. if key == 'D' else 50.
            for label, raw_column in (('在线推理', 'online_gpu_it_mw'), ('离线推理', 'offline_gpu_it_mw')):
                share = resource['gpu:' + label] / selected.allocated_gpu_fraction
                cpu_dynamic = (selected.cpu_it_mw - cpu_idle * config['cpu_packages_per_unit'] * units / 1e6) * share
                expected = selected[raw_column] - cards / 1e6 * idle_each * resource['gpu:' + label] + cpu_dynamic
                actual = parts[parts.task_type.eq(label)].sort_values('hour').it_mw.to_numpy()
                maximum_formula_error = max(maximum_formula_error, float(np.max(np.abs(expected - actual))))
                np.testing.assert_allclose(expected, actual, atol=1e-10)
        difference = matched[matched.scenario.eq(scenario)].sort_values('hour')
        np.testing.assert_allclose(difference.old_facility_mw, old.facility_mw.iloc[start:start+24], rtol=1e-12)
        np.testing.assert_allclose(difference.new_facility_mw, selected.facility_mw, rtol=1e-12)
        comparison = comparisons.loc[scenario]
        before = next(item for item in old_summary['scenarios'] if item['scenario'] == scenario)
        np.testing.assert_allclose(comparison.old_peak_mw, before['peak_mw'], rtol=1e-12)
        np.testing.assert_allclose(comparison.old_minimum_mw, before['minimum_mw'], rtol=1e-12)
        np.testing.assert_allclose(comparison.new_peak_valley_ratio, hourly.facility_mw.max()/hourly.facility_mw.min(), rtol=1e-12)
        np.testing.assert_allclose(row['daily_facility_mwh'], hourly.facility_mw.sum(), rtol=1e-12)
        ratio = hourly.facility_mw.sum()*365/full.facility_mw.sum()
        ratios.append(float(ratio))
        assert .5 < ratio < 2
        for sensitivity in sensitivities[sensitivities.scenario.eq(scenario)].itertuples():
            case = sensitivity.case.replace('training_', 'gpu_')
            if case.startswith('inference_'):
                np.testing.assert_allclose(sensitivity.annual_mean_it_mw, full.it_mw.mean(), rtol=1e-12)
            else:
                reference = upstream_sensitivity.query('scenario == @scenario and sensitivity == @case').iloc[0]
                np.testing.assert_allclose(sensitivity.annual_mean_it_mw, reference.mean_it_mw, rtol=1e-12)
                if use != 'INF':
                    np.testing.assert_allclose(sensitivity.annual_mean_facility_mw, reference.mean_facility_mw, rtol=1e-12)
        checks[scenario + '_CPU_GPU_provenance_layers_energy_comparison_and_sensitivity'] = True

    selected = summary['selections']['SAI']
    day_begin = int(pd.Timestamp(selected['source_date'], tz='UTC').timestamp())
    jobs = pd.read_parquet(args.other_jobs, columns=['time_start', 'time_end', 'gpus', 'cpu_allocated', 'category_id'])
    groups = {'DNN 训练（含推断）': ['labelled_dnn', 'user_dnn_high', 'user_dnn_medium', 'similar_dnn'],
        '长时单任务': ['long_single'], '数组批量': ['array'], '多节点大作业': ['large'], '低 GPU 活动占用': ['idle'],
        '其他与无法判断': ['interactive', 'startup', 'short_test', 'unknown']}
    cache = pd.read_parquet(inputs['SAI_local_classified_resource_hours'])
    clocks = json.loads(inputs['source_clocks'].read_text(encoding='utf-8'))
    for resource, raw_column in (('cpu', 'cpu_allocated'), ('gpu', 'gpus')):
        original = pd.read_parquet(project / 'data/processed/scenario-typical-day-by-task-20261031/SAI-hourly-resource-fractions.parquet', columns=[resource + ':OTHER'])
        np.testing.assert_allclose(cache[[resource + ':' + label for label in groups]].sum(axis=1), original[resource + ':OTHER'], atol=1e-7, rtol=1e-8)
        factor = clocks['sources']['mit_' + resource]['normalization_denominator']
        for label, ids in groups.items():
            subset = jobs[jobs.category_id.isin(ids)]
            direct = []
            for hour in range(24):
                begin = day_begin + hour * 3600
                seconds = np.maximum(0, np.minimum(subset.time_end, begin+3600) - np.maximum(subset.time_start, begin))
                direct.append(float((seconds * subset[raw_column]).sum()/3600))
            generated = cache[resource + ':' + label].iloc[selected['day_index']*24:(selected['day_index']+1)*24] * factor
            np.testing.assert_allclose(direct, generated, atol=1e-6, rtol=1e-9)
    checks['OTHER_original_237_days_conservation_and_new_day_private_interval_check'] = True
    for path in output.glob('*.csv'):
        assert not {'id_job', 'job_id', 'user_id', 'container_id', 'machine_id'}.intersection(pd.read_csv(path, nrows=0).columns)
    for name in [row['scenario'] for row in summary['scenarios']] + ['overview']:
        with Image.open(output / (name + '.png')) as image:
            np.testing.assert_allclose(image.info['dpi'], [300, 300], atol=.02)
        assert ET.parse(output / (name + '.svg')).getroot().tag.endswith('svg')
    checks['public_fields_aggregate_only_and_12_PNG_300dpi_SVG'] = True

    docs = project / 'docs/job-flexibility-brief'
    readme = (docs / 'README.md').read_text(encoding='utf-8')
    assert readme.count('## 数字来源与说明') == readme.count('## 重新生成') == 1
    assert readme.count('- 第四章图与逐小时分任务汇总：') == 1
    for target in re.findall(r'\]\(([^)]+)\)', readme):
        if not target.startswith('http'):
            assert (docs / target).is_file(), target
    for text in ('1.3 kWh', '0.05–0.65 MWh', '10.2 kW ÷ 8', '所选日DNN层为零'):
        assert text not in readme, text
    with ZipFile(docs / '数据中心任务时间柔性简表.docx') as archive:
        media = [item.filename for item in archive.infolist() if item.filename.startswith('word/media/') and not item.is_dir()]
        assert len(media) == 1 and archive.read(media[0]) == (output / 'overview.png').read_bytes()
        text = ''.join(ET.fromstring(archive.read('word/document.xml')).itertext())
        for claim in ('BurstGPT', 'OLMo', 'SPECpower', '0.560', '0.455–0.665', '所选日接近平坦'):
            assert claim in text, claim
    pages = PdfReader(args.pdf).pages
    assert len(pages) == 4
    page_text = [page.extract_text().replace(' ', '').replace('\n', '') for page in pages]
    for index, heading in enumerate(('一、情境定义', '二、任务类型', '三、各数据源', '四、各情境')):
        assert heading in page_text[index]
    assert '其他与无法判断' in page_text[1] and 'SPECpower' in page_text[2] and '尖峰' in page_text[3]
    checks['Word_README_energy_scope_sources_embedded_plot_and_four_page_PDF'] = True
    old_dirs = [project.relative_to(project.parent).as_posix() + '/results/' + run for run in ('scenario-typical-day-by-task-20261031', PREVIOUS)]
    assert not subprocess.check_output(['git', 'diff', '--name-only', '79e12f6', '--', *old_dirs], cwd=project.parent, text=True).strip()
    checks['previous_results_and_spike_audit_unchanged'] = True
    manifest['code_sha256'].update({name: sha256(project / 'analysis' / name) for name in manifest['code_sha256']})
    manifest['code_sha256'][Path(__file__).name] = sha256(Path(__file__))
    manifest['document_files_sha256'] = {name: hashlib.sha256((docs / name).read_text(encoding='utf-8').encode('utf-8')).hexdigest() for name in ('build.js', 'README.md')}
    manifest['document_files_sha256']['数据中心任务时间柔性简表.docx'] = sha256(docs / '数据中心任务时间柔性简表.docx')
    manifest['document_hash_basis'] = '文本UTF-8/LF；Word原始二进制字节'
    manifest['pdf_QA'] = {'pages': 4, 'sha256': sha256(args.pdf), 'visual_qa_confirmed': args.visual_qa_confirmed}
    manifest['output_sha256'] = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file() and path.name not in ('manifest.json', 'independent-verification.json')}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    result = {'status': 'passed', 'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'checks': checks,
        'evidence': {'daily_to_new_annual_mean_ratio_range': [min(ratios), max(ratios)], 'maximum_inference_task_formula_error_mw': maximum_formula_error}}
    (output / 'independent-verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'checks_passed': len(checks), 'evidence': result['evidence']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--other-jobs', type=Path, default=Path('D:/mit-supercloud-gpu-summary-20261028/derived/other-refine-20261101/other-job-categories.parquet'))
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--visual-qa-confirmed', action='store_true')
    verify(parser.parse_args())
