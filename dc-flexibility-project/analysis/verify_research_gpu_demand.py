import argparse
import json
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
from PIL import Image

from compute_load.model import sha256


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / 'results/server-count-scenarios-20261028-v2'


def verify(render_reviewed=False):
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    checks = 0

    def require(condition, message):
        nonlocal checks
        if not condition:
            raise ValueError(message)
        checks += 1

    for name, digest in manifest['files_sha256'].items():
        require(sha256(OUT / name) == digest, '输出文件校验值不同: ' + name)
    for name, digest in manifest['source_code_sha256'].items():
        require(sha256(PROJECT / 'analysis' / name) == digest, '代码校验值不同: ' + name)
    for name, digest in manifest['previous_result_files_sha256'].items():
        require(sha256(PROJECT / name) == digest, '历史输入/结果发生改变: ' + name)
    old = PROJECT / 'results/server-count-scenarios-20261028'
    catalog = json.loads((OUT / 'server-configurations.json').read_text(encoding='utf-8'))
    require(catalog == json.loads((old / 'server-configurations.json').read_text(encoding='utf-8')), '设备参数发生改变')
    matrix = json.loads((OUT / 'scenario-matrix.json').read_text(encoding='utf-8'))
    original = json.loads((old / 'scenario-matrix.json').read_text(encoding='utf-8'))
    require(len(matrix) == 52 and len({r['scenario_id'] for r in matrix}) == 52, '单元数量/标识不唯一')
    by_id = {r['scenario_id']: r for r in matrix}
    allowed = {'C20': {'Lisa', 'Alibaba Cluster Trace 2018'},
               'C21': {'MIT Supercloud GPU', 'Acme Kalos', 'ASI'},
               'C22': {'MIT Supercloud GPU', 'Acme Kalos', 'ASI'},
               'C28': {'MIT Supercloud GPU', 'Acme Kalos', 'ASI'},
               'R-GB200': {'Acme Kalos', 'ASI'}}
    changed_fields = {'scenario_id', 'demand_source', 'demand_orientation', 'demand_readiness'}
    for row in original:
        if row['configuration_id'] in ('C21', 'C22', 'C28') and row['demand_source'] == 'Lisa':
            current = by_id[row['scenario_id'].removesuffix('_lisa') + '_mit-supercloud']
            require(all(current[k] == v for k, v in row.items() if k not in changed_fields), '替换单元的非需求字段发生改变')
            require(current['demand_key'] == 'mit_supercloud_gpu' and current['demand_observed_days'] == 237, '新科研 GPU 来源不明确')
        else:
            require(by_id[row['scenario_id']] == row, '无关单元发生变化')
    frame = pd.DataFrame(matrix)
    for size in (1, 10, 100, 500):
        for config, sources in allowed.items():
            subset = frame[frame.reference_it_mw.eq(size) & frame.configuration_id.eq(config)]
            require(set(subset.demand_source) == sources and len(subset) == len(sources), '允许配对不一致')
    require(not (frame.gpu_devices_per_unit.gt(0) & frame.demand_source.eq('Lisa')).any(), 'GPU 单元含 Lisa CPU')
    require(frame.cooling_type.value_counts().to_dict() == {'air': 44, 'direct_liquid': 8}, '设备冷却类型发生变化')
    for name, rows in [('server-configurations', catalog), ('scenario-matrix', matrix)]:
        readback = pd.read_csv(OUT / (name + '.csv')).astype(object)
        expected = pd.DataFrame(rows).astype(object)
        pd.testing.assert_frame_equal(readback.where(readback.notna(), None), expected.where(expected.notna(), None), check_dtype=False)
        checks += 1
    source = PROJECT / 'data/metadata/research-gpu-demand-20261028/source-audit.json'
    require(sha256(source) == manifest['source_audit_sha256'], '来源审计 SHA-256 不符')
    audit = json.loads(source.read_text(encoding='utf-8'))
    require(audit['lisa']['gpu_occupancy_usable'] is False and audit['lisa']['usable_gpu_occupancy_days'] == 0, 'Lisa GPU 质量标注不符')
    require(all(r['etag_verified'] for r in audit['adopted']['acquisition']['files']), 'MIT S3 校验缺失')
    require(audit['adopted']['selected_gpu_execution_rows'] == 98176, '科研 GPU 筛选计数不同')
    require(len(audit['adopted']['telemetry_samples']) == 2, 'GPU 实际监测字段未核验')
    data = PROJECT / 'data/processed/research-gpu-demand-20261028'
    require(sha256(data / 'research-gpu-observed-hourly.csv') == manifest['hourly_input_sha256'], '小时来源校验不同')
    hourly = pd.read_csv(data / 'research-gpu-observed-hourly.csv', parse_dates=['source_time'])
    replay = pd.read_csv(data / 'research-gpu-annual-replay.csv', parse_dates=['target_time'])
    columns = ['gpu_allocation_hours', 'host_cpu_allocation_hours']
    require(len(hourly) == 5688 and len(replay) == 8760, '观测/年度小时长度不同')
    require(hourly.source_time.diff().dropna().eq(pd.Timedelta(hours=1)).all(), '观测时间索引不连续')
    np.testing.assert_allclose(replay[columns], hourly[columns].to_numpy()[np.arange(8760) % len(hourly)], rtol=1e-14)
    require(replay.target_time.equals(pd.Series(pd.date_range('2025-01-01', periods=8760, freq='h'), name='target_time')), '年度目标时间不完整')
    week = pd.read_csv(data / 'research-gpu-example-week.csv', parse_dates=['source_time'])
    expected = hourly[hourly.source_time.isin(week.source_time)]
    np.testing.assert_allclose(week[columns], expected[columns], rtol=1e-14)
    require(len(week) == 168 and week.source_time.iloc[0].weekday() == 0, '示意周不完整')
    for stem in ('research-gpu-week', 'research-gpu-annual-daily'):
        with Image.open(OUT / (stem + '.png')) as picture:
            require(min(picture.info['dpi']) >= 299, '图件分辨率低于 300 dpi')
        with fitz.open(OUT / (stem + '.pdf')) as document:
            require(len(document) == 1 and 'GPU' in document[0].get_text(), '图件 PDF 文本/页数不符')
            require(all(font[1] != 'n/a' for font in document[0].get_fonts()), '图件 PDF 字体未嵌入')
    pdf = next(OUT.glob('科研GPU需求*.pdf'))
    with fitz.open(pdf) as document:
        require(len(document) == 4, '说明 PDF 页数不同')
        require(all(page.get_text().strip() for page in document), 'PDF 存在空白文本页')
        require('98,176' in document[1].get_text(), 'PDF 与实际作业数量不一致')
    validation_path = OUT / 'validation.json'
    validation = json.loads(validation_path.read_text(encoding='utf-8'))
    validation['delivery_verification_checks'] = checks
    if render_reviewed:
        validation['render_review'] = {'status': 'passed', 'report_pages': 4, 'figure_PNGs': 2,
                                     'figure_PDF_pages': 2, 'method': 'PNG and Poppler PDF rendering visually inspected'}
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest['files_sha256']['validation.json'] = sha256(validation_path)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'passed', 'checks': checks, 'render_review_recorded': render_reviewed}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render-reviewed', action='store_true')
    args = parser.parse_args()
    verify(args.render_reviewed)
