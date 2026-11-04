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


LAYERS = ['刚性', '可推迟·整体', '可推迟·可拆分', '未知', '基础与空载 IT', '冷却等非 IT']
RUN = 'eight-city-native-flex-20261102'


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def key(text):
    return re.sub(r'\s+', '', text)


def audit(project, weather, pdf, layout_file, visual):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    output = project / 'results' / RUN
    manifest = json.loads((output / 'manifest.json').read_text('utf-8'))
    verification = json.loads((output / 'verification.json').read_text('utf-8'))
    assert digest(output / 'verification.json') == manifest['verification_sha256']
    for path, value in manifest['source_sha256'].items():
        assert digest(project / path) == value, path
    for path, value in manifest['previous_result_files_sha256'].items():
        assert digest(project / path) == value, path
    for name, value in manifest['code_sha256'].items():
        assert digest(project / 'analysis' / name) == value, name
    for name, value in verification['model_code_sha256'].items():
        assert digest(project / 'analysis/compute_load' / name) == value, name
    assert verification['hourly_files_verified'] == 1320 and verification['hour_rows_verified'] == 3799368
    assert len(verification['cooling_recomputed']) == 24 and verification['cooling_maximum_difference_mw'] < 1e-8
    read = lambda name: pd.read_csv(output / (name + '.csv'), float_precision='round_trip')
    climate, cases, years, cities = [read(name) for name in ('climate-summary', 'case-summary', 'year-summary', 'city-summary')]
    assert (len(climate), len(cases), len(years), len(cities)) == (8,1320,264,88)
    upstream = pd.read_csv(project / 'results/typical-load-curves-20261101-v4-seren/window-summary-all-locations.csv', float_precision='round_trip')
    upstream = upstream.loc[upstream.reference_it_mw.eq(10) & upstream.location_id.ne('reference-grid')].set_index(['scenario_id', 'location_id', 'weather_year', 'placement'])
    selected = cases.set_index(['scenario_id', 'location_id', 'weather_year', 'placement']).loc[upstream.index]
    pd.testing.assert_frame_equal(selected[upstream.columns], upstream, check_dtype=False, check_exact=False, rtol=1e-13, atol=1e-12)
    expected = pd.date_range('2023-01-01', '2026-01-01', freq='h', inclusive='left', tz='UTC')
    weather_index = json.loads((project / '.codex-temp/eight-city-native-inputs/weather-index.json').read_text('utf-8'))
    # 冻结清单按location_id命名，与本地缓存逐份对应。
    source_weather = {row['location_id']: row['source']['sha256'] for row in json.loads((project / 'results/trn-seren-20261101/calculation-audit.json').read_text('utf-8'))['weather'] if row['location_id'] != 'reference-grid'}
    for row in weather_index:
        path = weather / row['weather_file']
        assert digest(path) == row['weather_sha256'] == source_weather[row['city_id']]
        frame = pd.read_parquet(path)
        assert pd.DatetimeIndex(pd.to_datetime(frame.time_utc, utc=True)).equals(expected)
        t, dew = frame.t2m.to_numpy(float)-273.15, frame.d2m.to_numpy(float)-273.15
        item = climate.set_index('city_id').loc[row['city_id']]
        np.testing.assert_allclose([item.mean_temperature_c, item.mean_dew_point_c, item.hot_30c_pct, item.humid_dew20c_pct],
            [t.mean(), dew.mean(), 100*(t>=30).mean(), 100*(dew>=20).mean()], atol=1e-10, rtol=1e-12)
    fifth = pd.read_csv(project / 'results/flex-class-20261101/shares-10MW.csv', float_precision='round_trip').set_index('scenario')
    for row in cases.to_dict('records'):
        for label in LAYERS[:-1]:
            np.testing.assert_allclose(row[label], row['mean_it_mw']*fifth.loc[row['scenario_id'].removesuffix('-10MW'),label]/100, atol=1e-11)
        np.testing.assert_allclose(sum(row[label] for label in LAYERS[:-1])+row['mean_non_it_mw'], row['mean_facility_mw'], atol=1e-10)
    for row in years.to_dict('records'):
        group = cases.loc[cases.location_id.eq(row['city_id']) & cases.weather_year.eq(row['year']) & cases.scenario_id.eq(row['scenario_id'])]
        np.testing.assert_allclose([row['mean_facility_mw'],row['window_pue']], [group.mean_facility_mw.median(),group.window_pue.median()], atol=1e-12)
        np.testing.assert_allclose(sum(row[label] for label in LAYERS[:-1])+row['mean_non_it_mw'],row['mean_facility_mw'], atol=1e-10)
    for row in cities.to_dict('records'):
        group = years.loc[years.city_id.eq(row['city_id']) & years.scenario_id.eq(row['scenario_id'])]
        assert len(group) == 3
        np.testing.assert_allclose([row['mean_facility_mw'],row['window_pue']], [group.mean_facility_mw.mean(),group.window_pue.mean()], atol=1e-12)
        np.testing.assert_allclose(sum(row[label] for label in LAYERS[:-1])+row['mean_non_it_mw'],row['mean_facility_mw'], atol=1e-10)
    composition = read('facility-composition')
    assert len(composition) == 528 and set(composition.layer) == set(LAYERS)
    totals = composition.groupby(['city_id','scenario_id']).mean_facility_component_mw.sum()
    np.testing.assert_allclose(totals, cities.set_index(['city_id','scenario_id']).mean_facility_mw.reindex(totals.index), atol=1e-10)
    daily, averaged = read('typical-day-city-year'), read('typical-day-city-three-year')
    assert len(daily) == 576 and len(averaged) == 192 and daily.source_date.eq('2023-05-31').all()
    np.testing.assert_allclose(daily[LAYERS].sum(axis=1), daily.facility_mw, atol=1e-10)
    np.testing.assert_allclose(averaged[LAYERS].sum(axis=1), averaged.facility_mw, atol=1e-10)
    day_text = key(''.join(p.extract_text() for p in PdfReader(output / 'city-typical-day-comparison.pdf').pages))
    assert all(key(label) in day_text for label in LAYERS)
    docx = project / 'docs/job-flexibility-brief/数据中心任务时间柔性简表.docx'
    document = Document(docx)
    brief = json.loads((output / 'brief-summary.json').read_text('utf-8'))
    assert len(document.tables) == 7
    for table, expected_rows in [(document.tables[-2],brief['climate_rows']), (document.tables[-1],brief['facility_rows'])]:
        assert [[cell.text for cell in row.cells] for row in table.rows] == expected_rows
    for index, folder, name in [(1,'flex-class-20261101','classification_rows'),(4,'flex-class-20261101','share_rows'),(3,'typical-load-curves-20261101-v4-seren','rows')]:
        saved = json.loads((project / 'results' / folder / 'brief-summary.json').read_text('utf-8'))[name]
        assert [[cell.text for cell in row.cells] for row in document.tables[index].rows] == saved
    images = [r.target_part.blob for r in document.part.rels.values() if r.reltype.endswith('/image')]
    assert len(images) == 5
    assert all((output / name).read_bytes() in images for name in ('city-pue-heatmap.png','city-typical-day-comparison.png'))
    layout = json.loads(layout_file.read_text('utf-8-sig'))
    pages = [p.extract_text() for p in PdfReader(pdf).pages]
    assert len(pages) == layout['pages'] == 8
    assert [t['start_page'] for t in layout['tables']] == [1,2,2,4,5,7,7]
    assert all(t['start_page'] == t['end_page'] for t in layout['tables'])
    assert all(word in key(pages[1]) for word in ('二、任务时间柔性分类','三、各数据源概况'))
    assert '六、八城市气候下的负荷与柔性' in key(pages[6]) and '八城市PUE与典型日构成' in key(pages[7])
    assert '柔佛' in ''.join(pages) and '非全年PUE' in key(''.join(pages)) and '\ufffd' not in ''.join(pages)
    assert visual, '完成全部页面和新图的实际检查后才能确认'
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'status':'passed', 'weather_files_verified':8, 'weather_hours_per_city':26304,
        'reused_hourly_files_verified':1320, 'reused_hour_rows_verified':3799368, 'city_scenarios':88, 'city_year_scenarios':264,
        'six_layer_composition_records':528, 'cooling_recomputations_verified':24, 'Word_pages':8, 'new_chapter_pages':[7,8],
        'chapters_2_and_3_same_page':True, 'Word_tables':layout['tables'], 'previous_result_files_unchanged':True,
        'visual_QA':'八页和两张新图逐一检查；中文、图表、表格与分页正常', 'docx_sha256':digest(docx),'pdf_sha256':digest(pdf)}
    (output / 'brief-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--weather',type=Path)
    parser.add_argument('--pdf',type=Path,required=True)
    parser.add_argument('--layout',type=Path,required=True)
    parser.add_argument('--visual-qa-confirmed',action='store_true')
    args=parser.parse_args()
    audit(args.project,args.weather or args.project / '.codex-temp/eight-city-native-inputs/weather',args.pdf,args.layout,args.visual_qa_confirmed)
