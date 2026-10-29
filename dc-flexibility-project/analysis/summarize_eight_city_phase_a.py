import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eight_city_scenarios import PURPOSES


RUN='eight-city-phase-a-20261029'


def correlation(a,b):
    if np.std(a)<1e-10 or np.std(b)<1e-10:
        return None
    return float(np.corrcoef(a,b)[0,1])


def summarize(args):
    project=args.project.resolve()
    output,processed=project/'results'/RUN,project/'data/processed'/RUN
    annual=pd.read_csv(output/'annual-load-central.csv')
    labels=['scenario_id','scenario_scale_id','scenario_name_zh','server_type','server_type_zh','purpose',
        'purpose_zh','legacy_configuration_id','demand_source','city_id','city_zh']
    records=[]
    for key,group in annual.groupby(labels,sort=False):
        records.append({**dict(zip(labels,key)),'years':len(group),
            'annual_facility_gwh_mean':float(group.annual_facility_mwh.mean()/1000),
            'annual_facility_gwh_min':float(group.annual_facility_mwh.min()/1000),
            'annual_facility_gwh_max':float(group.annual_facility_mwh.max()/1000),
            'annual_it_gwh_mean':float(group.annual_it_mwh.mean()/1000),
            'peak_facility_mw_max':float(group.peak_facility_mw.max()),
            'energy_weighted_pue':float(group.annual_facility_mwh.sum()/group.annual_it_mwh.sum()),
            'cooling_share':float(group.annual_cooling_mwh.sum()/group.annual_facility_mwh.sum()),
            'hottest_week_peak_mw_max':float(group.hottest_week_peak_mw.max())})
    summary=pd.DataFrame(records)
    summary.to_csv(output/'scenario-city-load-summary.csv',index=False,encoding='utf-8-sig')
    order=['singapore','phoenix','ashburn','frankfurt','lulea','ulanqab','guiyang','riyadh']
    names=summary.drop_duplicates('city_id').set_index('city_id').city_zh.to_dict()
    legacy=summary.drop_duplicates('scenario_id').set_index('scenario_id').legacy_configuration_id.to_dict()
    lines=['# 八城市 × 十二情境负荷模式汇总','','规模均为10 MW参考IT容量；中央冷却参数，2023–2025。',
        '电量为三个完整年度的平均值；PUE为三年设施电量除以三年IT电量。',
        '峰值、冷却占比、最热周峰值和旧编号见[96行完整汇总](scenario-city-load-summary.csv)。','']
    for field,title,digits in [('annual_facility_gwh_mean','年设施电量（GWh）',1),
        ('energy_weighted_pue','年均PUE（电量加权）',3)]:
        lines += ['## '+title,'','| 情境 | 旧配置 | '+' | '.join(names[c] for c in order)+' |',
            '| --- | --- | '+' | '.join(['---:']*len(order))+' |']
        table=summary.pivot(index='scenario_id',columns='city_id',values=field)
        for scenario in summary.scenario_id.drop_duplicates():
            lines.append('| '+scenario+' | '+legacy[scenario]+' | '+' | '.join(f'{table.loc[scenario,c]:.{digits}f}' for c in order)+' |')
        lines.append('')
    (output/'load-pattern-summary.md').write_text('\n'.join(lines),encoding='utf-8')
    climate=pd.read_csv(output/'city-climate-resource-annual.csv')
    benchmark=pd.read_csv(args.benchmarks/'city-benchmarks.csv')
    fields=['mean_temperature_c','mean_dewpoint_c','mean_pressure_pa','pv_capacity_factor',
        'pv_tracking_capacity_factor','wind_capacity_factor','wind_gross_capacity_factor','mean_wind100_m_s','ghi_kwh_m2']
    averages=climate.groupby('city_id')[[f for f in fields if f in climate]].mean().reset_index()
    averages['weather_period']='2023–2025; mean_of_three_annual_means'
    averages['total_hours']=26304
    resource=benchmark.merge(averages,on='city_id',validate='one_to_one')
    grid=pd.read_json(project/'data/raw/eight-city-weather-20261029/grid-land-check.json')
    resource=resource.merge(grid,on='city_id',validate='one_to_one')
    if 'pv_capacity_factor' in resource:
        resource['pv_difference_percentage_points']=(resource.pv_capacity_factor-resource.benchmark_pv_cf_dc_basis)*100
        resource['wind_net_difference_percentage_points']=(resource.wind_capacity_factor-resource.gwa_iec3_capacity_factor)*100
        resource['wind_gross_difference_percentage_points']=(resource.wind_gross_capacity_factor-resource.gwa_iec3_capacity_factor)*100
    resource.to_csv(output/'city-coordinates-climate-resource.csv',index=False,encoding='utf-8-sig')
    if 'pv_capacity_factor' not in resource:
        return
    cities={c['city_id']:c for c in json.loads((project/'analysis/eight_city_locations.json').read_text(encoding='utf-8'))}
    flex=pd.read_csv(output/'flexibility-resource-summary.csv')
    correlation_rows=[]
    for city in cities:
        supply_path=processed/(city+'-supply.parquet')
        if not supply_path.exists():
            continue
        supply=pd.read_parquet(supply_path)
        supply=supply.loc[pd.to_datetime(supply.time_utc,utc=True).dt.year.eq(2025)].reset_index(drop=True)
        local=pd.to_datetime(supply.time_utc,utc=True).dt.tz_convert(cities[city]['timezone'])
        for source,(purpose,_) in PURPOSES.items():
            if purpose=='CLD':
                continue
            for row in flex.loc[flex.purpose.eq(purpose)].itertuples():
                label=f'{source}-f{row.share:g}-D{row.max_extra_delay_h}-r{row.capacity_multiple:g}'
                frame=pd.read_parquet(processed/'flexibility'/(label+'.parquet'))
                frame['local_hour']=local.dt.hour.to_numpy()
                hourly=frame.groupby('local_hour')[['baseline_resource_fraction','independent_max_removable_fraction']].mean()
                for technology,field in [('光伏','pv_capacity_factor'),('风电','wind_capacity_factor')]:
                    shape=supply.groupby(local.dt.hour)[field].mean().to_numpy()
                    correlation_rows.append({'city_id':city,'city_zh':cities[city]['city_zh'],'purpose':purpose,
                        'demand_source':source,'year':2025,'share':row.share,'max_extra_delay_h':row.max_extra_delay_h,
                        'capacity_multiple':row.capacity_multiple,'technology':technology,
                        'load_shape_pearson_r':correlation(hourly.baseline_resource_fraction.to_numpy(),shape),
                        'removable_shape_pearson_r':correlation(hourly.independent_max_removable_fraction.to_numpy(),shape),
                        'shape_hours':24,'phase_convention':'source_midnight_to_UTC_midnight',
                        'interpretation':'descriptive_shape_correlation; independent_hour_bounds_not_joint_dispatch'})
    pd.DataFrame(correlation_rows).to_csv(output/'renewable-flexibility-shape-correlations.csv',index=False,encoding='utf-8-sig')
    audit={'created_utc':datetime.now(timezone.utc).isoformat(),'scenario_city_rows':len(summary),
        'resource_rows':len(resource),'correlation_rows':len(correlation_rows),
        'undefined_correlation':'constant_zero_participation_or_zero_delay_envelopes; empty_value_not_zero'}
    (output/'summary-validation.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--benchmarks',type=Path,required=True)
    summarize(parser.parse_args())
