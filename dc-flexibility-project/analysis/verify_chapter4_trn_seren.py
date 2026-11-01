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
from update_chapter4_trn_seren import RUN, PREVIOUS, UPSTREAM, TRN, LABELS, CASES


def verify(args):
    project = args.project
    output,previous,local = project/'results'/RUN,project/'results'/PREVIOUS,project/'data/processed'/RUN
    manifest = json.loads((output/'manifest.json').read_text('utf-8'))
    summary = json.loads((output/'summary.json').read_text('utf-8'))
    old = json.loads((previous/'summary.json').read_text('utf-8'))
    old_rows = {row['scenario']:row for row in old['scenarios']}
    audit = json.loads((project/'results'/UPSTREAM/'source-audit.json').read_text('utf-8'))
    calculation = json.loads((project/'results'/UPSTREAM/'calculation-audit.json').read_text('utf-8'))
    stats = json.loads((output/'seren-job-statistics.json').read_text('utf-8'))
    checks = {}
    for item in manifest['input_files']:
        assert sha256(Path(item['file'])) == item['sha256']
    for path,digest in manifest['upstream_hourly_sha256'].items():
        assert sha256(Path(path)) == digest
    for name,digest in manifest['preserved_v6_files_sha256'].items():
        assert sha256(output/name) == sha256(previous/name) == digest
    for row in summary['scenarios']:
        if row['scenario'] not in TRN:
            assert row == old_rows[row['scenario']]
    for use in ('HPC','CLD','SAI','INF'):
        assert summary['selections'][use] == old['selections'][use]
    checks['upstream_fingerprints_and_eight_v6_scenarios_unchanged'] = True
    source = pd.read_csv(project/'results'/UPSTREAM/'source-hourly.csv',parse_dates=['source_clock'])
    qualify = source.physical_device_coverage.ge(.8) & source.overlap_samples.ge(228) & source.sample_matched_allocated_gpus.gt(0) & source.inventory_excess_samples.eq(0)
    assert qualify.equals(source.power_quality_80) and qualify.sum()==2057
    qualified = source.loc[qualify]
    count = qualified.sample_matched_allocated_gpus
    inferred = (qualified.overlap_estimated_power_w-(2288-count)*60)/count/400
    np.testing.assert_allclose(inferred,qualified['inverse_ratio_idle_60.000'],rtol=1e-12)
    weights = count*qualified.overlap_samples/240
    np.testing.assert_allclose(np.average(inferred,weights=weights),audit['central_power_statistics']['gpu_hour_weighted_mean_ratio'],rtol=1e-12)
    assert int((inferred>1).sum())==8
    checks['Seren_quality_filter_hourly_inverse_formula_and_weighted_ratio'] = True
    assert stats['job_sha256']==audit['job_sha256'] and stats['source_hourly_sha256']==calculation['source_hourly_sha256']
    assert stats['raw_rows']==818327 and stats['gpu_jobs']==450590 and stats['independent_cpu_jobs']==367737
    assert stats['raw_job_size_mtime_unchanged'] and stats['duration_mismatch_rows']==0
    assert stats['script_sha256']==sha256(project/'analysis/summarize_seren_jobs_for_brief.py')
    assert sum(row['rows'] for row in stats['statistics'])==450590
    np.testing.assert_allclose(sum(row['resource_hours'] for row in stats['statistics']),audit['full_gpu_hours'],rtol=1e-12)
    shares = pd.read_csv(project/'results'/UPSTREAM/'task-composition.csv').set_index(['window','type'])
    for row in stats['statistics']:
        kind = row['class']
        assert row['negative_wait_rows']==0 and row['valid_runtime_rows']==row['rows']
        for window,column in [('full_jobs','work_fraction_of_resource_pool'),('qualified_overlap','qualified_gpu_share')]:
            np.testing.assert_allclose(row[column],shares.loc[(window,kind),'gpu_share'],rtol=1e-12)
        np.testing.assert_allclose(row['resource_hours'],source['gpu_hours_'+kind].sum(),rtol=1e-12)
        np.testing.assert_allclose(row['qualified_gpu_hours'],qualified['gpu_hours_'+kind].sum(),rtol=1e-12)
        assert row['runtime_s_p50']<=row['runtime_s_p90']<=row['runtime_s_p99']<=row['runtime_s_max']
        assert row['units_p50']<=row['units_p90']<=row['units_max']
    checks['recalculated_raw_job_statistics_counts_denominators_and_hour_integrals'] = True
    full_reference = pd.read_parquet(local/'upstream/main/D-TRN-10MW-2025.parquet')
    clock = pd.DatetimeIndex(pd.to_datetime(full_reference.source_clock,utc=True))
    groups = {date:np.flatnonzero(clock.floor('d')==date) for date in clock.floor('d').unique()}
    complete = sorted(date for date,positions in groups.items() if len(positions)==24 and clock[positions].hour.tolist()==list(range(24)))
    assert len(complete)==55 and len(full_reference)==2057
    values = np.stack([full_reference.facility_mw.to_numpy()[groups[date]] for date in complete])
    energy,average = values.sum(axis=1),values.mean(axis=0)
    deviation = 100*(energy/np.median(energy)-1)
    errors = np.sqrt(((values-average)**2).mean(axis=1))
    primary = np.abs(deviation)<=2+1e-10
    width = 2 if primary.sum()>=5 else 5
    candidates = np.flatnonzero(np.abs(deviation)<=width+1e-10)
    day = min(candidates,key=lambda index:(errors[index],abs(energy[index]-np.median(energy)),index))
    date = complete[day]
    selection = summary['selections']['TRN']
    assert day==selection['day_index'] and date.strftime('%Y-%m-%d')==selection['source_date']
    assert width==selection['applied_window_pct'] and len(candidates)==selection['candidate_count']
    assert int(primary.sum())==selection['candidate_count_at_2pct']
    scores = pd.read_csv(output/'TRN-day-selection.csv')
    for column,array in [('facility_mwh',energy),('energy_deviation_pct',deviation),('rmse_mw',errors)]:
        np.testing.assert_allclose(scores[column],array,rtol=1e-12,atol=1e-12)
    assert scores.selected.sum()==1 and scores.selected.iloc[day]
    available = pd.read_csv(output/'TRN-day-availability.csv')
    assert available.hours.sum()==2057 and available.complete.sum()==55
    assert available.loc[available.complete,'source_day'].str[:10].tolist()==[day.strftime('%Y-%m-%d') for day in complete]
    checks['independent_complete_UTC_days_gap_preservation_window_and_minimum_RMSE'] = True
    configs = catalogue(project)
    cpu_shape = json.loads((project/'results/trn-power-model-20261101/cpu-shape.json').read_text('utf-8'))
    comparisons = pd.read_csv(output/'comparison.csv').set_index('scenario')
    sensitivity = pd.read_csv(output/'sensitivity-summary.csv').set_index(['scenario','case'])
    upstream = pd.read_csv(project/'results'/UPSTREAM/'window-summary.csv').query('reference_it_mw == 10').set_index(['server_type','case'])
    period = pd.read_csv(output/'period-summary.csv').set_index('scenario')
    for scenario in TRN:
        key,config = scenario[0],configs[scenario[0]]
        full = pd.read_parquet(local/f'upstream/main/{scenario}-10MW-2025.parquet')
        pd.testing.assert_series_equal(full.source_clock,full_reference.source_clock)
        np.testing.assert_allclose(full.allocated_gpu_power_ratio,qualified['inverse_ratio_idle_60.000'],rtol=1e-12)
        assert not full.power_imputed.any() and full.source_dataset.eq('Acme Seren').all()
        units = int(full.unit_count.iloc[0])
        gpu = config['gpu_devices_per_unit']*units/1e6*(full.allocated_gpu_fraction*TDP[key]*full.allocated_gpu_power_ratio+
            (1-full.allocated_gpu_fraction)*(70 if key=='D' else 50))
        np.testing.assert_allclose(gpu,full.gpu_it_mw,rtol=1e-12)
        base,ci,cf,_,_ = PARAMETERS[config['configuration_id']]['central']
        u,c1,c2 = full.u_cpu_proxy,*cpu_shape['coefficients']
        response = 3*c1*u*(1-u)**2+3*c2*u**2*(1-u)+u**3
        cpu = config['cpu_packages_per_unit']*units/1e6*(ci+(cf-ci)*response)
        np.testing.assert_allclose(cpu,full.cpu_it_mw,rtol=1e-12)
        np.testing.assert_allclose(full.it_mw,full.gpu_it_mw+full.cpu_it_mw+full.base_it_mw,rtol=1e-12)
        np.testing.assert_allclose(full.facility_mw,full.it_mw+full.cooling_mw+full.other_mw,rtol=1e-12)
        assert pd.DatetimeIndex(full.time_utc).strftime('%m-%d %H').tolist()==clock.strftime('%m-%d %H').tolist()
        checks[scenario+'_full_main_observation_GPU_CPU_and_thermal_formulas'] = True
        current = full.iloc[groups[date]].reset_index(drop=True)
        hourly = pd.read_csv(output/(scenario+'-hourly.csv'))
        row = next(row for row in summary['scenarios'] if row['scenario']==scenario)
        assert row['source_date']==selection['source_date'] and hourly.hour.tolist()==list(range(24))
        for column in ('it_mw','facility_mw','pue','allocated_gpu_power_ratio'):
            np.testing.assert_allclose(hourly[column],current[column],rtol=1e-12)
        layers = pd.read_csv(output/(scenario+'-by-task.csv'))
        assert set(layers.task_type)==set(LABELS.values())|{'基础与空载 IT','冷却等非 IT'}
        assert layers.facility_mw.min()>=-1e-10 and layers.groupby('task_type').size().eq(24).all()
        for column in ('it_mw','facility_mw'):
            np.testing.assert_allclose(layers.groupby('hour')[column].sum(),hourly[column],atol=1e-10)
        for kind,label in LABELS.items():
            share = current['gpu_hours_'+kind]/current.allocated_gpu_hours
            np.testing.assert_allclose(layers[layers.task_type.eq(label)].sort_values('hour').it_mw,
                (current.it_mw-current.idle_it_mw)*share,atol=1e-10)
            np.testing.assert_allclose(row['composition'][label],current['gpu_hours_'+kind].sum()/current.allocated_gpu_hours.sum(),rtol=1e-12)
        for version,record in [('old',old_rows[scenario]),('new',row)]:
            for name in ('peak_mw','minimum_mw','peak_valley_ratio','daily_facility_mwh'):
                np.testing.assert_allclose(comparisons.loc[scenario,version+'_'+name],record[name],rtol=1e-12)
        assert .5<period.loc[scenario,'daily_to_period_mean_ratio']<2 and period.loc[scenario,'hours']==2057
        checks[scenario+'_24_hour_shared_date_MLLM_layer_conservation_and_energy_scale'] = True
        for case in CASES:
            np.testing.assert_allclose(sensitivity.loc[(scenario,case),['mean_it_mw','mean_facility_mw']].to_numpy(float),
                upstream.loc[(key,case),['mean_it_mw','mean_facility_mw']].to_numpy(float),rtol=1e-12)
        checks[scenario+'_six_observation_and_sensitivity_cases_match_PR38'] = True
    flags = pd.read_csv(output/'selected-day-diagnostics.csv')
    assert flags.audit_flags.isna().all() and flags.unresolved_hours.eq(0).all() and flags.above_tdp_hours.eq(0).all()
    for name in TRN+['overview']:
        with Image.open(output/(name+'.png')) as image:
            np.testing.assert_allclose(image.info['dpi'],[300,300],atol=.02)
        ET.parse(output/(name+'.svg'))
    checks['new_days_quality_flags_and_four_300dpi_SVG_figures'] = True
    docs = project/'docs/job-flexibility-brief'
    docx = docs/'数据中心任务时间柔性简表.docx'
    ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    def text(node):
        return ''.join(part.text or '' for part in node.iter(ns+'t'))
    with ZipFile(docx) as archive:
        root = ET.fromstring(archive.read('word/document.xml'))
        media = [name for name in archive.namelist() if name.startswith('word/media/') and not name.endswith('/')]
        assert len(media)==1 and archive.read(media[0])==(output/'overview.png').read_bytes()
    blob = subprocess.check_output(['git','show','c7c3df0:dc-flexibility-project/docs/job-flexibility-brief/数据中心任务时间柔性简表.docx'],cwd=project.parent)
    with ZipFile(BytesIO(blob)) as archive:
        old_root = ET.fromstring(archive.read('word/document.xml'))
    tables,old_tables = root.findall('.//'+ns+'tbl'),old_root.findall('.//'+ns+'tbl')
    matrix = [[text(cell) for cell in row.findall(ns+'tc')] for row in tables[0].findall(ns+'tr')]
    old_matrix = [[text(cell) for cell in row.findall(ns+'tc')] for row in old_tables[0].findall(ns+'tr')]
    for index,row in enumerate(matrix):
        for column,value in enumerate(row):
            assert value==('Acme Seren' if index>=2 and column==4 else old_matrix[index][column])
    table_rows = {text(row.find(ns+'tc')):text(row) for row in tables[1].findall(ns+'tr')}
    old_table_rows = {text(row.find(ns+'tc')):text(row) for row in old_tables[1].findall(ns+'tr')}
    updated = {'预训练／SFT','预训练','SFT','MLLM（多模态）','评估','交互式／开发／调试','其他与无法判断'}
    for label,value in old_table_rows.items():
        if label not in updated:
            assert table_rows[label]==value
    for kind,label in [('Pretrain','预训练'),('SFT','SFT'),('MLLM','MLLM（多模态）'),('Eval','评估'),('Debug','交互式／开发／调试'),('Other','其他与无法判断')]:
        stat = next(row for row in stats['statistics'] if row['class']==kind)
        assert f"Seren {100*stat['work_fraction_of_resource_pool']:.2f}%" in table_rows[label]
    changed = [text(new.find(ns+'tc')) for new,before in zip(tables[2].findall(ns+'tr'),old_tables[2].findall(ns+'tr')) if text(new)!=text(before)]
    assert changed==['Acme Seren','OLMo等训练功率研究']
    readme = (docs/'README.md').read_text('utf-8')
    assert readme.count('## 数字来源与说明')==readme.count('## 重新生成')==1
    assert readme.count('- 第四章图与逐小时分任务汇总：')==1
    pages = PdfReader(args.pdf).pages
    assert len(pages)==4
    page_text = [page.extract_text().replace(' ','').replace('\n','') for page in pages]
    for index,heading in enumerate(('一、情境定义','二、任务类型','三、各数据源','四、各情境')):
        assert heading in page_text[index]
    assert 'MLLM' in page_text[1] and '69.49%' in page_text[1] and '0.534' in page_text[1]
    assert 'AcmeSeren' in page_text[0] and '2057' in page_text[2] and '55' in page_text[2]
    for content in ('55个完整UTC日','2057','A100','H100','2288','60W','400W','少于5天放宽至±5%'):
        assert content in page_text[3]
    checks['Word_four_chapters_Seren_statistics_other_rows_and_four_page_PDF'] = True
    paths = ['dc-flexibility-project/results/'+name for name in (PREVIOUS,'scenario-typical-day-by-task-20261101-v5',
        'scenario-typical-day-by-task-20261101-v4','scenario-typical-day-by-task-20261101-v3','scenario-typical-day-by-task-20261031')]
    assert not subprocess.check_output(['git','diff','--name-only','c7c3df0','--',*paths],cwd=project.parent,text=True).strip()
    checks['all_earlier_chapter4_results_preserved'] = True
    manifest['code_sha256'][Path(__file__).name] = sha256(Path(__file__))
    manifest['documents_sha256'] = {name:hashlib.sha256((docs/name).read_text('utf-8').encode('utf-8')).hexdigest() for name in ('build.js','README.md')}
    manifest['documents_sha256'][docx.name] = sha256(docx)
    manifest['document_text_hash_basis'] = 'UTF-8/LF; docx original binary bytes'
    manifest['pdf_QA'] = {'pages':4,'sha256':sha256(args.pdf),'visual_qa_confirmed':args.visual_qa_confirmed}
    manifest['output_sha256'] = {path.name:sha256(path) for path in sorted(output.iterdir()) if path.is_file() and path.name not in ('manifest.json','independent-verification.json')}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf-8')
    evidence = {'source_date':selection['source_date'],'complete_days':55,'candidates':len(candidates),'window_pct':width,
        'deviation_pct':float(deviation[day]),'RMSE_mw':float(errors[day]),'raw_job_scan_peak_RSS_MiB':stats['peak_RSS_MiB'],
        'daily_to_period_mean_range':[float(period.daily_to_period_mean_ratio.min()),float(period.daily_to_period_mean_ratio.max())]}
    (output/'independent-verification.json').write_text(json.dumps({'status':'passed','utc':datetime.now(timezone.utc).isoformat(),
        'checks':checks,'evidence':evidence},ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps({'checks_passed':len(checks),'evidence':evidence},ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--pdf',type=Path,required=True)
    parser.add_argument('--visual-qa-confirmed',action='store_true')
    verify(parser.parse_args())
