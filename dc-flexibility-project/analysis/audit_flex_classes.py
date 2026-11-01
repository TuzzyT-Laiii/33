import argparse
from datetime import datetime, timezone
from hashlib import file_digest
import json
from pathlib import Path
import re

from docx import Document
import numpy as np
import pandas as pd
import pyarrow as pa
from pypdf import PdfReader


CLASSES = ['刚性', '可推迟·整体', '可推迟·可拆分', '未知']
LAYERS = ['基础与空载 IT', *CLASSES, '冷却等非 IT']


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def text_key(text):
    return re.sub(r'\s+', '', text)


def audit(project, pdf, layout_file, visual_confirmed):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    output = project / 'results/flex-class-20261101'
    manifest = json.loads((output / 'manifest.json').read_text('utf-8'))
    base = project / 'results' / manifest['source_result']
    for path, value in manifest['input_sha256'].items():
        assert digest(project / path) == value, path
    for path, value in manifest['old_result_all_files_sha256'].items():
        assert digest(base / path) == value, path
    for path, value in manifest['code_sha256'].items():
        assert digest(project / 'analysis' / path) == value, path
    mapping = pd.read_csv(output / 'task-class-mapping.csv').set_index(['use', 'source_label'])['class'].to_dict()
    # 用户明确指定的映射独立列出，以识别跨类合并错误。
    expected_groups = {
        'HPC': [('刚性', ['Jupyter']), ('可推迟·整体', ['CPU 科研分区']), ('未知', ['课程分区', '其他分区'])],
        'CLD': [('刚性', ['在线服务']), ('可推迟·整体', ['离线批处理']), ('未知', ['未对账部分'])],
        'SAI': [('刚性', ['交互式', '低 GPU 活动占用']), ('可推迟·整体', ['批处理', 'MapReduce', '数组批量']),
            ('可推迟·可拆分', ['DNN 训练（含推断）', '长时单任务', '多节点大作业']), ('未知', ['其他与无法判断'])],
        'TRN': [('刚性', ['调试']), ('可推迟·整体', ['评测']), ('可推迟·可拆分', ['预训练', 'SFT', 'MLLM（多模态大模型）']), ('未知', ['其他'])],
        'INF': [('刚性', ['在线推理']), ('可推迟·整体', ['离线推理'])],
    }
    assert mapping == {(use, label): category for use, groups in expected_groups.items() for category, labels in groups for label in labels}
    shares = pd.read_csv(output / 'shares-10MW.csv', float_precision='round_trip')
    details = pd.read_csv(output / 'native-class-summary.csv', float_precision='round_trip')
    placements = pd.read_csv(output / 'representative-placements.csv', float_precision='round_trip').set_index('scenario')
    old_places = pd.read_csv(base / 'placement-summary.csv', float_precision='round_trip')
    old_summary = json.loads((base / 'selected-days/summary.json').read_text('utf-8'))
    selected = json.loads((output / 'selected-days/summary.json').read_text('utf-8'))
    assert old_summary == selected
    max_delta, verified_hours = 0., 0
    for row in shares.to_dict('records'):
        scenario = row['scenario']
        use = scenario.split('-')[1]
        frame = pd.read_parquet(output / 'native-hourly' / (scenario + '.parquet'))
        fine = pd.read_parquet(output / 'native-by-task' / (scenario + '.parquet'))
        original = pd.read_parquet(base / 'hourly' / f'{scenario}-10MW-2025-{placements.loc[scenario, "placement"]}.parquet')
        assert len(frame) == len(fine) == len(original) == row['native_hours']
        for label in CLASSES:
            columns = [name for (kind, name), category in mapping.items() if kind == use and category == label]
            expected = fine[columns].sum(axis=1).to_numpy() if columns else np.zeros(len(fine))
            np.testing.assert_allclose(frame[label], expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(frame[LAYERS[:-1]].sum(axis=1), original.it_mw, atol=1e-8, rtol=1e-12)
        np.testing.assert_allclose(frame[LAYERS].sum(axis=1), original.facility_mw, atol=1e-8, rtol=1e-12)
        assert frame[LAYERS].min().min() >= -1e-8
        for label in LAYERS[:-1]:
            item = details.loc[details.scenario.eq(scenario) & details['class'].eq(label)].iloc[0]
            energy = frame[label].sum()
            mean = frame[label].mean()
            fraction = 100 * energy / original.it_mw.sum()
            np.testing.assert_allclose([item.it_mwh, item.mean_it_mw, item.share_of_it_energy_pct, item.share_of_mean_it_power_pct, row[label]],
                [energy, mean, fraction, fraction, fraction], atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(sum(row[label] for label in LAYERS[:-1]), 100., atol=1e-9)
        parts = pd.read_csv(output / 'selected-days' / (scenario + '-by-task.csv'), float_precision='round_trip')
        old = pd.read_csv(base / 'selected-days' / (scenario + '-by-task.csv'), float_precision='round_trip')
        old['canonical'] = old.task_type.replace({'评估': '评测'})
        assert parts.task_type.drop_duplicates().tolist() == LAYERS
        for label in CLASSES:
            sources = [name for (kind, name), category in mapping.items() if kind == use and category == label]
            expected = old.loc[old.canonical.isin(sources)].groupby('hour').facility_mw.sum().reindex(range(24), fill_value=0)
            np.testing.assert_allclose(parts.loc[parts.task_type.eq(label)].facility_mw, expected, atol=1e-12)
        hourly_path = output / 'selected-days' / (scenario + '-hourly.csv')
        assert digest(hourly_path) == digest(base / 'selected-days' / hourly_path.name)
        hourly = pd.read_csv(hourly_path, float_precision='round_trip')
        delta = float(np.abs(parts.groupby('hour').facility_mw.sum().to_numpy() - hourly.facility_mw.to_numpy()).max())
        max_delta = max(max_delta, delta)
        assert delta < 1e-8
        options = old_places.loc[old_places.scenario_id.eq(scenario + '-10MW')]
        candidates = options.loc[options.hours.eq(options.native_hours)].copy()
        chosen = candidates.assign(distance=(candidates.mean_facility_mw - options.mean_facility_mw.median()).abs()).sort_values(['distance', 'placement'], kind='stable').iloc[0]
        assert chosen.placement == placements.loc[scenario, 'placement']
        if use == 'TRN':
            clock = pd.DatetimeIndex(pd.to_datetime(frame.source_clock, utc=True))
            assert len(frame) == 2057 and clock.is_unique and clock.is_monotonic_increasing
            assert int((clock.to_series().diff() > pd.Timedelta(hours=1)).sum()) == 53
            assert len(pd.date_range(clock[0], clock[-1], freq='h')) - len(clock) == 189
        verified_hours += len(frame)
    assert len(shares) == 11 and len(details) == 55
    legend_order = [*CLASSES, '基础与空载 IT', '冷却等非 IT']
    plot_pdfs = [output / 'native-overview.pdf', output / 'shares-10MW.pdf', output / 'selected-days/overview.pdf',
        *[output / 'selected-days' / (scenario + '.pdf') for scenario in shares.scenario]]
    for path in plot_pdfs:
        text = text_key(''.join(page.extract_text() for page in PdfReader(path).pages))
        assert all(text_key(label) in text for label in legend_order), path
        assert '评估' not in text, path
    document_path = project / 'docs/job-flexibility-brief/数据中心任务时间柔性简表.docx'
    doc = Document(document_path)
    brief = json.loads((output / 'brief-summary.json').read_text('utf-8'))
    assert len(doc.tables) == 5
    for table, expected in [(doc.tables[1], brief['classification_rows']), (doc.tables[-1], brief['share_rows'])]:
        assert [[cell.text for cell in row.cells] for row in table.rows] == expected
    native_expected = json.loads((base / 'brief-summary.json').read_text('utf-8'))['rows']
    assert [[cell.text for cell in row.cells] for row in doc.tables[3].rows] == native_expected
    text = '\n'.join(p.text for p in doc.paragraphs) + '\n'.join(cell.text for table in doc.tables for row in table.rows for cell in row.cells)
    assert '评估' not in text and '0.80' not in text and 'Kalos' not in text
    assert all(word in text for word in ('未知默认按刚性', '情景', '不据此计算可调电量', '2057'))
    images = [r.target_part.blob for r in doc.part.rels.values() if r.reltype.endswith('/image')]
    assert len(images) == 3
    assert all((output / name).read_bytes() in images for name in ('native-overview.png', 'shares-10MW.png', 'selected-days/overview.png'))
    layout = json.loads(layout_file.read_text('utf-8-sig'))
    pages = [page.extract_text() for page in PdfReader(pdf).pages]
    assert len(pages) == layout['pages'] == 7
    assert [t['start_page'] for t in layout['tables']] == [1, 2, 3, 5, 6]
    assert all(t['start_page'] == t['end_page'] for t in layout['tables'])
    assert '原生窗口四类任务份额' in text_key(pages[5]) and '原生窗口时序总览' in text_key(pages[6])
    assert '\ufffd' not in ''.join(pages)
    assert visual_confirmed, '需要完成全部页面与修改图件的实际视觉检查'
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed', 'scenarios': 11, 'native_hours_verified': verified_hours,
        'user_mapping_entries_verified': len(mapping), 'energy_and_mean_power_shares_equal': True,
        'previous_result_directory_unchanged': True, 'selected_dates_and_total_power_unchanged': True,
        'maximum_selected_total_difference_mw': max_delta, 'plot_PDFs_with_six_legend_items': len(plot_pdfs),
        'Word_pages': len(pages), 'Word_tables': layout['tables'], 'Word_three_images_match_new_class_figures': True,
        'visual_QA': '七页及全部修改图件逐一检查，中文、图例、表格与分页正常',
        'docx_sha256': digest(document_path), 'pdf_sha256': digest(pdf)}
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
