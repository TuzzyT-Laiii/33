import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from compute_load.model import sha256
from eight_city_physics import cumulative_bounds, independent_hour_envelope, read_profiles, replay
from eight_city_scenarios import scenario_fields


RUN = 'eight-city-phase-a-20261029'


def validate(args):
    project = args.project.resolve()
    output, processed = project/'results'/RUN, project/'data/processed'/RUN
    checks, evidence = {}, {}
    annual = pd.read_csv(output/'annual-load-summary.csv')
    checks['annual_rows_864'] = len(annual) == 864
    checks['twelve_named_scenarios'] = annual.scenario_id.nunique() == 12
    render=json.loads((output/'figure-render-checks.json').read_text(encoding='utf-8'))
    checks['forty_one_pdf_figures_reviewed'] = (len(render['figures'])==41
        and render['visual_review']=='reviewed_by_agent' and render['all_text_inside_pages'] and render['no_replacement_glyphs'])
    sync=json.loads((output/'server-sync-validation.json').read_text(encoding='utf-8'))
    checks['server_96_files_sha256_match'] = sync['files']==96 and sync['all_sha256_and_bytes_matched']
    checks['sensitivity_energy_order'] = bool((annual.pivot_table(index=['city_id','scenario_id','year'],
        columns='cooling_case', values='annual_facility_mwh').eval('low <= central and central <= high')).all())
    times = pd.date_range('2023-01-01','2026-01-01',freq='h',inclusive='left',tz='UTC')
    profiles, sources = read_profiles(project,args.baseline)
    cities = json.loads((project/'analysis/eight_city_locations.json').read_text(encoding='utf-8'))
    hourly_records = []
    for city in cities:
        for (config,demand), profile in profiles.items():
            scenario = scenario_fields(config,demand)['scenario_scale_id']
            path = processed/f'{city["city_id"]}-{scenario}.parquet'
            frame = pd.read_parquet(path)
            key = city['city_id']+'-'+scenario
            checks[key+'-utc'] = frame.index.equals(times)
            fields = ['pv_capacity_factor','wind_capacity_factor','pue','it_mw','cooling_mw','other_mw','facility_mw']
            checks[key+'-finite'] = bool(np.isfinite(frame[fields]).all().all())
            checks[key+'-power'] = bool(np.allclose(frame.facility_mw,frame.it_mw+frame.cooling_mw+frame.other_mw,
                rtol=0,atol=1e-10) and np.allclose(frame.pue,frame.facility_mw/frame.it_mw,rtol=0,atol=1e-12))
            checks[key+'-cf'] = bool(frame.pv_capacity_factor.between(0,1/1.2+1e-12).all()
                and frame.wind_capacity_factor.between(0,.9+1e-12).all())
            baseline = pd.read_parquet(args.baseline/f'{config}-{demand}-{profile["method"]}-central.parquet')
            original = baseline.it_mw.to_numpy()
            new = frame.loc[frame.time_utc.dt.year.eq(2025),'it_mw'].to_numpy()
            checks[key+'-v2-it'] = bool(np.allclose(new,original,rtol=0,atol=1e-10))
            for year in (2023,2024,2025):
                actual=frame.loc[frame.time_utc.dt.year.eq(year)]
                saved=annual.loc[annual.city_id.eq(city['city_id']) & annual.scenario_scale_id.eq(scenario)
                    & annual.year.eq(year) & annual.cooling_case.eq('central')].iloc[0]
                checks[key+f'-{year}-annual-sum'] = bool(abs(actual.facility_mw.sum()-saved.annual_facility_mwh)<1e-6
                    and abs(actual.it_mw.sum()-saved.annual_it_mwh)<1e-6
                    and abs(actual.facility_mw.max()-saved.peak_facility_mw)<1e-9)
            hourly_records.append({'city_id':city['city_id'],'scenario_scale_id':scenario,
                'filename':path.name,'sha256':sha256(path),'hours':len(frame),'bytes':path.stat().st_size})
    pd.DataFrame(hourly_records).to_csv(output/'processed-hourly-files.csv',index=False,encoding='utf-8-sig')
    # 以真实输入的首周逐时LP独立核验单小时移出量公式。
    comparisons = []
    for demand,key in {'HPC':('C20','Lisa'),'SAI':('C21','MIT'),'TRN':('C21','Acme'),'INF':('C21','ASI')}.items():
        cpu,gpu,_ = replay(profiles[key],168)
        load = cpu if demand == 'HPC' else gpu
        n = len(load)
        diff = sparse.diags([np.ones(n),-np.ones(n-1)],[0,-1],shape=(n,n),format='csr')
        for multiple in (1.,1.5):
            share,delay = .5,6
            lower,upper = cumulative_bounds(share*load,delay)
            capacity = multiple-(1-share)*load
            envelope = independent_hour_envelope(load,share,delay,multiple)
            for hour in (0,83,167):
                objective = diff.getrow(hour).toarray().ravel()
                solution = linprog(objective,A_ub=sparse.vstack([diff,-diff]),b_ub=np.r_[capacity,np.zeros(n)],
                    bounds=np.column_stack([lower,upper]),method='highs-ds',options={'presolve':False})
                if not solution.success:
                    raise RuntimeError('真实首周独立LP核验失败：'+solution.message)
                lp_removed = share*load[hour]-solution.fun
                error = float(abs(lp_removed-envelope[hour]))
                comparisons.append({'purpose':demand,'hour':hour,'r':multiple,'absolute_error':error})
    checks['independent_hour_formula_vs_real_input_lp'] = max(c['absolute_error'] for c in comparisons)<1e-7
    evidence['real_input_lp_comparisons'] = comparisons
    # 全年满占用是方案要求的解析边界，不作为经验数据。
    checks['analytical_full_capacity_r1_zero_shift'] = bool(np.max(independent_hour_envelope(np.ones(8760),1.,48,1.))<1e-10)
    flex = pd.read_csv(output/'flexibility-resource-summary.csv')
    for axis,others in [('share',['purpose','max_extra_delay_h','capacity_multiple']),
        ('max_extra_delay_h',['purpose','share','capacity_multiple']),
        ('capacity_multiple',['purpose','share','max_extra_delay_h'])]:
        differences = flex.sort_values(axis).groupby(others).max_independent_removable_resource_h.diff().dropna()
        checks['monotonic_removability_'+axis] = bool((differences>=-1e-8).all())
    for filename in ('weather-load-validation.json','flexibility-validation.json'):
        report = json.loads((output/filename).read_text(encoding='utf-8'))
        checks[filename] = bool(report.get('all_computed_checks_passed',report.get('all_passed')))
    report = {'created_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,
        'all_passed':all(checks.values()),'evidence':evidence,'phase_b_optimization_run':False,
        'weather_hours_per_city':26304,'processed_hourly_files':len(hourly_records),
        'coverage_year_hours':{'2023':8760,'2024':8784,'2025':8760}}
    (output/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if not all(checks.values()):
        raise ValueError('核验失败：'+str([key for key,value in checks.items() if not value]))
    write_manifest(project,output,sources,hourly_records)
    print(json.dumps({'all_passed':True,'checks':len(checks),'processed_files':len(hourly_records)}))


def write_manifest(project,output,sources,hourly_records):
    versions = {name:importlib.metadata.version(name) for name in
        ('numpy','pandas','scipy','pyarrow','matplotlib','pvlib','psychrolib','rasterio','h5netcdf','xarray','Pillow','PyMuPDF')}
    manifest = {'created_utc':datetime.now(timezone.utc).isoformat(),'base_commit':'6ffda9b',
        'processed_server_directory':'/data3/dataset/ljt/eight-city-weather-20261029/processed',
        'raw_server_directory':'/data3/dataset/ljt/eight-city-weather-20261029',
        'load_sources':sources,'packages':versions,'hourly_files':hourly_records,
        'reused_inputs_sha256':{name:sha256(project/name) for name in (
            'analysis/compute_load/cooling.py','analysis/compute_load/assets/cooling-20261028-v2.json',
            'analysis/typical_power_models.py','analysis/typical_power_models_v2.py',
            'results/typical-load-curves-20261029-v2/manifest.json',
            'results/typical-load-curves-20261029-v2/mit-power-ratio-summary.json',
            'results/server-count-scenarios-20261028-v2/server-configurations.json',
            'planning/direction1/全球网格需求侧柔性收益研究方案_20261023.md',
            'analysis/requirements-eight-city-phase-a.txt')},
        'files_sha256':{p.relative_to(project).as_posix():sha256(p) for p in sorted(output.rglob('*'))
            if p.is_file() and p.name!='manifest.json'},
        'code_sha256':{p.relative_to(project).as_posix():sha256(p) for p in sorted((project/'analysis').glob('*eight_city*'))
            if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--manifest-only',action='store_true')
    args=parser.parse_args()
    if args.manifest_only:
        project=args.project.resolve()
        output=project/'results'/RUN
        previous=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        if not json.loads((output/'validation.json').read_text(encoding='utf-8'))['all_passed']:
            raise ValueError('完整核验尚未通过，不可刷新交付清单')
        write_manifest(project,output,previous['load_sources'],previous['hourly_files'])
        print('Manifest refreshed from validated inputs')
    else:
        validate(args)
