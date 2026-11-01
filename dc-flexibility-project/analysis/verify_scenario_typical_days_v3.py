import argparse
from datetime import datetime, timezone
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


RUN = 'scenario-typical-day-by-task-20261101-v3'
UPSTREAM = 'typical-load-curves-20261101-v3'


def verify(args):
    output = args.project / 'results' / RUN
    upstream = args.project / 'results' / UPSTREAM
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    checks, evidence = {}, {}
    assert len(summary['scenarios']) == 11
    annual = pd.read_csv(upstream / 'annual-summary.csv').query('reference_it_mw == 10').set_index('scenario')
    ratios = []
    for use, selection in summary['selections'].items():
        scenario = ('A' if use in ('HPC', 'CLD') else 'D') + '-' + use
        source = pd.read_parquet(upstream / 'hourly' / f'{scenario}-10MW-2025.parquet')
        count = selection['complete_days']
        hours = source.facility_mw.to_numpy()[:24 * count].reshape(count, 24)
        energy = np.sum(hours, axis=1)
        distance = np.abs(energy - np.median(energy))
        candidates = np.flatnonzero(np.abs(distance - distance.min()) <= 1e-9)
        errors = ((hours - hours.mean(axis=0)) ** 2).mean(axis=1) ** .5
        winner = sorted(candidates, key=lambda index: (errors[index], index))[0]
        assert int(winner) == selection['day_index']
        published = pd.read_csv(output / f'{use}-day-selection.csv')
        np.testing.assert_allclose(published.facility_mwh, energy, rtol=1e-12)
        np.testing.assert_allclose(published.rmse_mw, errors, rtol=1e-12)
        assert published.selected.sum() == 1 and bool(published.iloc[winner].selected)
        checks[use + '_independent_median_then_RMSE_selection'] = True

    forbidden = {'id_job', 'user', 'user_id', 'job_id', 'machine_id', 'container_id', 'instance_id'}
    for file in output.glob('*.csv'):
        assert not forbidden.intersection(pd.read_csv(file, nrows=0).columns), file
    checks['public_CSV_aggregate_fields_only'] = True

    for row in summary['scenarios']:
        scenario = row['scenario']
        use = scenario.split('-')[1]
        selection = summary['selections'][use]
        assert row['source_date'] == selection['source_date']
        start = 24 * selection['day_index']
        source = pd.read_parquet(upstream / 'hourly' / f'{scenario}-10MW-2025.parquet')
        selected = source.iloc[start:start + 24].reset_index(drop=True)
        hourly = pd.read_csv(output / (scenario + '-hourly.csv'))
        assert hourly.hour.tolist() == list(range(24))
        assert hourly.source_date.eq(row['source_date']).all()
        assert hourly.weekday.eq(row['weekday']).all()
        for column in ('it_mw', 'facility_mw', 'pue', 'installed_reference_it_mw'):
            np.testing.assert_allclose(hourly[column], selected[column], rtol=1e-12, atol=1e-12)
        assert hourly.source_quality_flag.tolist() == selected.source_quality_flag.tolist()
        np.testing.assert_allclose(hourly.facility_mw.sum(), row['daily_facility_mwh'], rtol=1e-12)
        np.testing.assert_allclose(hourly.facility_mw.max(), row['peak_mw'], rtol=1e-12)
        np.testing.assert_allclose(hourly.facility_mw.min(), row['minimum_mw'], rtol=1e-12)
        np.testing.assert_allclose(source.facility_mw.sum(), annual.loc[scenario, 'annual_facility_mwh'], rtol=1e-12)
        ratio = hourly.facility_mw.sum() * 365 / annual.loc[scenario, 'annual_facility_mwh']
        np.testing.assert_allclose(ratio, row['daily_to_annual_mean_ratio'], rtol=1e-12)
        assert .5 < ratio < 2
        ratios.append(float(ratio))
        layers = pd.read_csv(output / (scenario + '-by-task.csv'))
        assert layers.groupby('task_type').size().eq(24).all()
        assert layers.it_mw.min() >= -1e-8 and layers.facility_mw.min() >= -1e-8
        np.testing.assert_allclose(layers.groupby('hour').it_mw.sum(), hourly.it_mw, atol=1e-8)
        np.testing.assert_allclose(layers.groupby('hour').facility_mw.sum(), hourly.facility_mw, atol=1e-8)
        cooling = layers[layers.component_scope.eq('non_IT')].sort_values('hour')
        assert cooling.task_type.eq('冷却等非 IT').all() and cooling.it_mw.eq(0).all()
        np.testing.assert_allclose(cooling.facility_mw, hourly.facility_mw - hourly.it_mw, atol=1e-8)
        np.testing.assert_allclose(sum(row['composition'].values()), 1, atol=1e-12)
        checks[scenario + '_v3_exact_values_layers_energy_and_quality_flags'] = True
    evidence['daily_to_v3_annual_mean_ratio_range'] = [min(ratios), max(ratios)]

    baseline = json.loads((args.project / 'results/typical-load-curves-20261029-v2/manifest.json').read_text(encoding='utf-8'))
    selection = summary['selections']['SAI']
    selected_start = int(pd.Timestamp(selection['source_date'], tz='UTC').timestamp())
    # 独立按每小时截断区间求和，避免重用生成脚本的事件累计方法。
    jobs = pd.read_parquet(args.other_jobs, columns=['time_start', 'time_end', 'gpus', 'cpu_allocated', 'category_id'])
    categories = {
        'DNN 训练（含推断）': ['labelled_dnn', 'user_dnn_high', 'user_dnn_medium', 'similar_dnn'],
        '长时单任务': ['long_single'], '数组批量': ['array'], '多节点大作业': ['large'],
        '低 GPU 活动占用': ['idle'], '其他与无法判断': ['interactive', 'startup', 'short_test', 'unknown']}
    full = pd.read_csv(output / 'mit-other-full-source-shares.csv').set_index('task_type')
    day = pd.read_csv(output / 'mit-other-typical-day-shares.csv').set_index('task_type')
    published = pd.read_csv(args.project / 'results/mit-other-breakdown-20261031/other-breakdown.csv')
    local = pd.read_parquet(args.project / 'data/processed' / RUN / 'SAI-resources.parquet')
    maximum_error = 0.
    direct_gpu_hours = {}
    for label, category_ids in categories.items():
        subset = jobs[jobs.category_id.isin(category_ids)]
        direct_gpu_hours[label] = 0.
        for resource, column in (('gpu', 'gpus'), ('cpu', 'cpu_allocated')):
            direct = []
            for hour in range(24):
                begin = selected_start + 3600 * hour
                overlap = np.maximum(0, np.minimum(subset.time_end, begin + 3600) - np.maximum(subset.time_start, begin))
                direct.append(float((overlap * subset[column]).sum() / 3600))
            factor = baseline['sources']['mit_' + resource]['normalization_denominator']
            generated = local[resource + ':' + label].to_numpy()[selection['day_index'] * 24:(selection['day_index'] + 1) * 24] * factor
            np.testing.assert_allclose(direct, generated, atol=1e-6, rtol=1e-9)
            maximum_error = max(maximum_error, float(np.max(np.abs(np.asarray(direct) - generated))))
            if resource == 'gpu':
                direct_gpu_hours[label] = sum(direct)
                np.testing.assert_allclose(sum(direct), day.loc[label, 'gpu_hours'], rtol=1e-9, atol=1e-6)
        official = published[published.category_id.isin(category_ids)]
        np.testing.assert_allclose(full.loc[label, 'gpu_hours'], official.gpu_hours.sum(), rtol=1e-10)
        np.testing.assert_allclose(full.loc[label, 'share_of_other'] * 100, official.pct_other_gpu_hours.sum(), rtol=1e-10)
        np.testing.assert_allclose(full.loc[label, 'share_of_mit_gpu_hours'] * 100, official.pct_mit_gpu_hours.sum(), rtol=1e-10)
    for resource in ('cpu', 'gpu'):
        prior = pd.read_parquet(args.project / 'data/processed/scenario-typical-day-by-task-20261031/SAI-hourly-resource-fractions.parquet', columns=[resource + ':OTHER'])
        np.testing.assert_allclose(local[[resource + ':' + label for label in categories]].sum(axis=1), prior[resource + ':OTHER'], rtol=1e-8, atol=1e-7)
    assert direct_gpu_hours['DNN 训练（含推断）'] == 0
    np.testing.assert_allclose(day.share_of_other.sum(), 1, atol=1e-12)
    np.testing.assert_allclose(full.share_of_other.sum(), 1, atol=1e-12)
    checks['OTHER_direct_interval_clipping_CPU_and_GPU_full_source_and_selected_day'] = True
    checks['OTHER_reconciles_original_237_day_hourly_CPU_and_GPU'] = True
    evidence['OTHER_maximum_hourly_direct_error_resource_hours'] = maximum_error
    evidence['OTHER_selected_day_GPU_hours'] = direct_gpu_hours

    for use, source_file, labels in (
        ('CLD', 'alibaba-cpu-cld-20261031/complete-component-window.csv', {'在线服务': 'online_share_of_machine', '离线批处理': 'batch_share_of_machine', '未对账部分': 'residual_share_of_machine'}),
        ('INF', 'asi-185day-workload-types-20261031/inf-hourly.csv', {'在线推理': 'online_shape_fraction', '离线推理': 'offline_shape_fraction'})):
        selection = summary['selections'][use]
        source = pd.read_csv(args.project / 'results' / source_file).iloc[selection['day_index'] * 24:(selection['day_index'] + 1) * 24]
        if use == 'CLD':
            occupied = source.occupied_core_hours / source.observed_core_hours
            np.testing.assert_allclose(occupied, source.resource_fraction, rtol=1e-12)
            totals = {label: float((source[column] * occupied).sum()) for label, column in labels.items()}
        else:
            totals = {label: float(source[column].sum()) for label, column in labels.items()}
        expected = {label: value / sum(totals.values()) for label, value in totals.items()}
        for row in (item for item in summary['scenarios'] if item['scenario'].endswith('-' + use)):
            for label in expected:
                np.testing.assert_allclose(row['composition'][label], expected[label], rtol=1e-10)
            assert row['weekday'] == '未知' and row['unresolved_selected_hours'] == 0
        checks[use + '_source_components_scope_and_unknown_calendar'] = True

    for asset in [row['scenario'] for row in summary['scenarios']] + ['overview']:
        with Image.open(output / (asset + '.png')) as image:
            np.testing.assert_allclose(image.info['dpi'], [300, 300], atol=.02)
        svg = ET.parse(output / (asset + '.svg'))
        assert svg.getroot().tag.endswith('svg') and svg.findall('.//{http://www.w3.org/2000/svg}path')
    checks['all_12_figures_PNG_300dpi_and_valid_SVG'] = True

    docs = args.project / 'docs/job-flexibility-brief'
    readme = (docs / 'README.md').read_text(encoding='utf-8')
    assert readme.count('## 数字来源与说明') == readme.count('## 重新生成') == 1
    assert readme.count('- 第四章图与逐小时分任务汇总：') == 1
    assert '正在 PR #28' not in readme and '第 109 天' not in readme and 'v2 中央' not in readme
    for target in re.findall(r'\]\(([^)]+)\)', readme):
        if not target.startswith('http'):
            assert (docs / target).is_file(), target
    with ZipFile(docs / '数据中心任务时间柔性简表.docx') as archive:
        document = ET.fromstring(archive.read('word/document.xml'))
        paragraphs = document.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t')
        word_text = '\n'.join(part.text or '' for part in paragraphs)
        for text in ('185天', '4,440小时', 'OTHER已分12类', 'SPECpower曲线50%→60%', '四、各情境典型日负荷曲线'):
            assert text in word_text, text
        image_name = next(item.filename for item in archive.infolist() if item.filename.startswith('word/media/') and not item.is_dir())
        assert archive.read(image_name) == (output / 'overview.png').read_bytes()
    pdf = PdfReader(args.pdf)
    assert len(pdf.pages) == 4
    page_text = [page.extract_text() for page in pdf.pages]
    for index, heading in enumerate(('一、情境定义', '二、任务类型', '三、各数据源', '四、各情境')):
        assert heading in page_text[index].replace(' ', ''), (index, heading)
    assert '其他与无法判断' in page_text[1].replace(' ', '')
    assert '零资源异常小时' in page_text[2].replace(' ', '').replace('\n', '')
    assert '尖峰' in page_text[3]
    checks['Word_PDF_four_pages_table_endings_and_embedded_overview'] = True
    checks['README_sources_regeneration_and_links'] = True

    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    for source in manifest['input_files']:
        assert sha256(Path(source['file'])) == source['sha256'], source['role']
    assert sha256(upstream / 'manifest.json') == manifest['upstream_manifest_sha256']
    for name in manifest['code_sha256']:
        manifest['code_sha256'][name] = sha256(args.project / 'analysis' / name)
    manifest['code_sha256'][Path(__file__).name] = sha256(Path(__file__))
    manifest['document_files_sha256'] = {name: sha256(docs / name) for name in ('build.js', 'README.md', '数据中心任务时间柔性简表.docx')}
    manifest['pdf_visual_QA'] = {'pages': 4, 'pdf_sha256': sha256(args.pdf),
        'visual_qa_confirmed': args.visual_qa_confirmed,
        'status': '四页已逐页目视确认，中文、图例与表格完整，图和表均未跨页' if args.visual_qa_confirmed else '仅完成自动检查，未记录人工逐页检查'}
    manifest['output_sha256'] = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file() and path.name not in ('manifest.json', 'independent-verification.json')}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    checks['all_input_fingerprints_unchanged'] = True

    old = 'dc-flexibility-project/results/scenario-typical-day-by-task-20261031/'
    root = args.project.parent
    previous = subprocess.check_output(['git', 'diff', '--name-only', 'd0a13b2', '--', old], cwd=root, text=True)
    assert not previous.strip(), previous
    checks['previous_typical_day_and_spike_audit_unchanged'] = True
    result = {'status': 'passed', 'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'checks': checks, 'evidence': evidence}
    (output / 'independent-verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'checks_passed': len(checks), 'evidence': evidence}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--other-jobs', type=Path, required=True)
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--visual-qa-confirmed', action='store_true', help='仅在已逐页检查所给PDF后使用')
    verify(parser.parse_args())
