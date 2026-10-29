import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
from PIL import Image

from compute_load.model import sha256
from typical_power_models_v2 import parameters_v2, power_v2


RUN = 'typical-load-curves-20261029-v2'


def verify(project, visual_review=False):
    output = project / 'results' / RUN
    data = project / 'data/processed' / RUN
    temp = project.parent / '.codex-temp/pdf-review'
    temp.mkdir(parents=True, exist_ok=True)
    validation_path = output / 'validation.json'
    report = json.loads(validation_path.read_text(encoding='utf-8'))
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    checks = report['checks']
    table = pd.read_csv(output / 'sensitivity-summary.csv')
    for row in table.itertuples():
        prefix = '' if row.reference_it_mw == 10 else f'{int(row.reference_it_mw)}MW-'
        path = data / f'{prefix}{row.configuration_id}-{row.demand}-{row.method}-{row.cooling_case}.parquet'
        frame = pd.read_parquet(path)
        name = path.stem
        checks[name + ':parquet_hash'] = sha256(path) == manifest['processed_sha256'][path.name]
        checks[name + ':energy_readback'] = bool(np.isclose(frame.facility_mw.sum(), row.annual_facility_mwh, rtol=1e-12) and np.isclose(frame.it_mw.sum(), row.annual_it_mwh, rtol=1e-12))
        checks[name + ':pue_energy_ratio'] = bool(np.isclose(frame.facility_mw.sum() / frame.it_mw.sum(), row.annual_pue, rtol=1e-12))
        checks[name + ':peak_readback'] = bool(np.isclose(frame.facility_mw.max(), row.peak_mw, rtol=1e-12))
        checks[name + ':utc_index'] = bool(np.array_equal(frame.time_utc.to_numpy(), pd.date_range('2025-01-01', periods=8760, freq='h', tz='UTC').to_numpy()))
    image_details = {}
    for pdf_path in sorted(output.glob('figure*.pdf')):
        stem = pdf_path.stem
        with Image.open(output / (stem + '.png')) as png:
            image_details[stem] = {'png_pixels': list(png.size), 'png_dpi': list(png.info['dpi'])}
            checks[stem + ':300dpi'] = min(png.info['dpi']) >= 299.9
        with fitz.open(pdf_path) as document:
            checks[stem + ':one_pdf_page'] = len(document) == 1
            fonts = document.get_page_fonts(0, full=True)
            checks[stem + ':embedded_fonts'] = all(len(document.extract_font(font[0])[3]) > 0 for font in fonts)
            text = document[0].get_text()
            checks[stem + ':chinese_text'] = '功率' in text or '电量' in text
            document[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(temp / (stem + '.png'))
            image_details[stem]['pdf_page_size_pt'] = list(document[0].rect[2:])
            image_details[stem]['embedded_font_names'] = [font[3] for font in fonts]
    central = pd.read_csv(output / 'annual-summary.csv')
    design = pd.read_csv(output / 'design-cell-status.csv')
    checks['central_48_rows'] = len(central) == 48
    checks['sensitivity_540_rows'] = len(table) == 540
    checks['design_52_rows_and_4_pending'] = len(design) == 52 and int(design.status.eq('待数据').sum()) == 4
    checks['no_hourly_or_job_tables_in_publication'] = not any('parquet' in path.suffix or 'hourly' in path.name or 'replay' in path.name for path in output.iterdir())
    checks['v1_C20_IT_unchanged'] = bool(pd.read_csv(output / 'v1-v2-comparison.csv').loc[lambda frame: frame.configuration_id.eq('C20'), 'it_energy_change_pct'].eq(0).all())
    stats = json.loads((output / 'mit-power-ratio-summary.json').read_text(encoding='utf-8'))
    ratios = stats['transfer_distribution']
    checks['ratio_quantiles_ordered'] = all(0 <= ratios[key] <= ratios[next_key] for key, next_key in zip(['p10', 'p25', 'p50', 'p75', 'p90', 'p95'], ['p25', 'p50', 'p75', 'p90', 'p95', 'p99']))
    configurations = json.loads((project / 'results/server-count-scenarios-20261028-v2/server-configurations.json').read_text(encoding='utf-8'))
    published_parameters = pd.read_csv(output / 'power-parameters.csv')
    pd.testing.assert_frame_equal(published_parameters, parameters_v2(configurations, ratios), check_dtype=False)
    for row in published_parameters.itertuples():
        config = next(item for item in configurations if item['configuration_id'] == row.configuration_id)
        number = config['gpu_devices_per_unit']
        expected_full = number * (row.gpu_tdp_w_each * row.allocated_gpu_power_ratio if row.method.startswith('ratio_') else (row.v1_gpu_full_w_each if number else 0))
        checks[f'{row.configuration_id}:{row.method}:idle_parameter_matches_model'] = bool(np.isclose(power_v2(config, 0, 0, row.method, ratios)['gpu_w'], number * row.unallocated_gpu_idle_w_each))
        checks[f'{row.configuration_id}:{row.method}:full_parameter_matches_model'] = bool(np.isclose(power_v2(config, 0, 1, row.method, ratios)['gpu_w'], expected_full))
    approximate = stats['details'][str(stats['completed_files']) + ':gpu:volta']['instantaneous_power_w_5w_bin_midpoint']['mean']
    checks['hour_mean_and_histogram_energy_within_bin_precision'] = abs(stats['transfer_power_w']['mean'] - approximate) <= 2.5
    if not all(checks.values()):
        raise AssertionError([key for key, value in checks.items() if not value])
    stamp = datetime.now(timezone.utc).isoformat()
    report.update({'checked_at_utc': stamp, 'status': 'numerical_and_artifact_checks_passed; visual_review_pending',
        'artifact_checks': image_details, 'pdf_render_directory': str(temp),
        'visual_review': {'status': 'pending', 'scope': '五张PNG及五份PDF实际渲染；中文、图例、标注、裁切、数据编码'}})
    if visual_review:
        report['status'] = 'passed'
        report['visual_review'] = {'status': 'passed', 'reviewed_at_utc': stamp,
            'scope': '五张PNG及五份PDF渲染已逐张查看；中文完整、文字未裁切、图例与坐标清楚'}
    validation_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for name in list(manifest['code_sha256']):
        manifest['code_sha256'][name] = sha256(project / 'analysis' / name)
    manifest['code_sha256']['verify_typical_load_curves_v2.py'] = sha256(Path(__file__))
    manifest['package_versions'] = {name: version(name) for name in ['numpy', 'pandas', 'pyarrow', 'matplotlib', 'psychrolib', 'numba', 'Pillow', 'PyMuPDF']}
    manifest['processed_storage_directory'] = str(data.resolve())
    manifest['files_sha256'] = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != 'manifest.json'}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'pdf_render_directory': str(temp)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--visual-review-complete', action='store_true')
    args = parser.parse_args()
    verify(args.project.resolve(), args.visual_review_complete)
