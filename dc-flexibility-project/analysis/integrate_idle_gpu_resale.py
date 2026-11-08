import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from flex_econ_copt import CITIES, load_config
from joint_capacity_inputs import atomic_json, sha256


PRICES=(.5,1.,1.5,2.2,2.9)
SHARES=(.25,.5,.75)
SCENARIOS=tuple(f'{key}-{use}' for use in ('SAI','TRN','INF') for key in 'BCD')


def run(args):
    data=pd.read_csv(args.source)
    published=pd.read_csv(args.source_summary).set_index('scenario')
    main=pd.read_csv(args.part2/'part2-main-table.csv')
    fleet=json.loads((args.part2/'resource-manifest.json').read_text('utf-8'))
    p,_=load_config(args.project)
    if len(data)!=72 or data.duplicated(['scenario','city']).any():
        raise ValueError('来源必须包含九个情境和八个城市的唯一记录')
    if set(data.scenario)!=set(SCENARIOS):
        raise ValueError('GPU 情境集合不完整')
    rows=[]
    comparisons=[]
    imported=[]
    for scenario in SCENARIOS:
        selected=data.loc[data.scenario.eq(scenario)]
        if set(selected.city)!=set(CITIES):
            raise ValueError(f'{scenario} 城市不完整')
        for item in selected.to_dict('records'):
            city=item['city']
            current=json.loads((args.part2/'collected-cities'/f'{city}-{scenario}-economic.json').read_text('utf-8'))
            a,b,c=current['A'],current['hourly_baseline_fleet'],current['continuous_job_plan']
            np.testing.assert_allclose(item['gpus_A'],a['gpus'],rtol=0,atol=1e-8)
            np.testing.assert_allclose(item['fleet_annual_usd'],a['equipment_annual_usd'],rtol=0,atol=.01)
            np.testing.assert_allclose(item['grid_ef_t_per_mwh'],p['emission_factor'][city],rtol=0,atol=1e-8)
            if item['max_used_over_gpus']>1+1e-10:
                raise ValueError('来源可售 GPU 小时超过 A 的资源边界')
            capacity=int(current['minimum_servers'])
            used_year=a['gpus']*8760-item['idle_gpu_hours_year']
            after_C=capacity*fleet[scenario]['gpus_per_unit']*8760-used_year
            if after_C<-1e-5:
                raise ValueError('减配后 GPU 工作量超过可用总量')
            np.testing.assert_allclose(item['min_servers'],capacity,rtol=0,atol=0)
            np.testing.assert_allclose(item['idle_gpu_hours_year_after_C'],after_C,rtol=0,atol=1e-5)
            b_low=b.get('total_annual_lower_bound_usd',b['total_annual_usd'])-b['equipment_annual_usd']
            c_low=c['total_annual_lower_bound_usd']-c['equipment_annual_usd']
            equipment=a['equipment_annual_usd']-c['equipment_annual_usd']
            comparison={'scenario':scenario,'city':city,'A_servers':a['servers'],'C_servers':capacity,
                'equipment_annual_saving_usd':equipment,
                'B_vs_A_saving_usd':a['energy_annual_usd']-b['energy_annual_usd'],
                'B_vs_A_saving_upper_usd':a['energy_annual_usd']-b_low,
                'C_vs_B_saving_usd':equipment+b['energy_annual_usd']-c['energy_annual_usd'],
                'C_vs_B_saving_lower_usd':equipment+b_low-c['energy_annual_usd'],
                'C_vs_B_saving_upper_usd':equipment+b['energy_annual_usd']-c_low,
                'fleet_annual_usd':a['equipment_annual_usd']}
            for price in PRICES:
                for share in SHARES:
                    key=f'D_net_p{price}_s{int(share*100)}'
                    net=item[key]
                    expected=item['idle_gpu_hours_year']*share*(price-item['energy_cost_usd_per_sold_gpu_h'])
                    np.testing.assert_allclose(net,expected,rtol=0,atol=.01)
                    np.testing.assert_allclose(item[f'CD_extra_p{price}_s{int(share*100)}'],
                        after_C*share*(price-item['energy_cost_usd_per_sold_gpu_h']),rtol=0,atol=.01)
                    rows.append({'scenario':scenario,'city':city,'price_usd_gpu_h':price,'sold_fraction':share,
                        'D_net_annual_usd':net,'D_net_wan_usd_year':net/1e4,
                        'D_pct_of_A_fleet':100*net/a['equipment_annual_usd'],
                        'CD_extra_annual_usd':item[f'CD_extra_p{price}_s{int(share*100)}'],
                        'energy_cost_usd_per_sold_gpu_h':item['energy_cost_usd_per_sold_gpu_h'],
                        'idle_gpu_hours_year':item['idle_gpu_hours_year'],
                        'idle_gpu_hours_year_after_C':after_C})
                    if share==.5:
                        comparison[f'D_net_p{price}_s50_usd']=net
                        comparison[f'D_pct_p{price}_s50']=100*net/a['equipment_annual_usd']
            for field in ('B_vs_A_saving_usd','C_vs_B_saving_usd','equipment_annual_saving_usd'):
                comparison[field.replace('_usd','_pct')]=100*comparison[field]/a['equipment_annual_usd']
            comparisons.append(comparison)
            imported.append({key:value for key,value in item.items() if key not in (
                'flex_energy_saving_usd','equipment_saving_usd','joint_saving_usd','C_status')})
        for price in PRICES:
            for share in SHARES:
                key=f'D_net_p{price}_s{int(share*100)}'
                np.testing.assert_allclose(selected[key].median(),published.loc[scenario,key],rtol=0,atol=1e-6)
    result=pd.DataFrame(rows)
    summaries=[]
    for (scenario,price,share),group in result.groupby(['scenario','price_usd_gpu_h','sold_fraction'],sort=False):
        row={'scenario':scenario,'price_usd_gpu_h':price,'sold_fraction':share,'cities':len(group)}
        for field in ('D_net_annual_usd','D_pct_of_A_fleet','CD_extra_annual_usd','energy_cost_usd_per_sold_gpu_h'):
            row.update({field+'_median':float(group[field].median()),field+'_min':float(group[field].min()),field+'_max':float(group[field].max())})
        summaries.append(row)
    compare=pd.DataFrame(comparisons)
    compare_summary=[]
    for scenario,group in compare.groupby('scenario',sort=False):
        row={'scenario':scenario,'cities':len(group)}
        for field in group.select_dtypes(include='number').columns:
            if field=='city':
                continue
            row.update({field+'_median':float(group[field].median()),field+'_min':float(group[field].min()),field+'_max':float(group[field].max())})
        compare_summary.append(row)
    augmented=main.copy()
    medians=pd.DataFrame(compare_summary).set_index('scenario')
    for index,row in augmented.iterrows():
        if row['target']=='economic' and row['scenario'] in SCENARIOS:
            for price in (1.,2.2,2.9):
                for suffix in ('median','min','max'):
                    field=f'D_net_p{price}_s50_usd_{suffix}'
                    augmented.loc[index,field]=medians.loc[row['scenario'],field]
            augmented.loc[index,'D_scope']='A 原时序经济目标；迹线平均电耗；情景估算'
    args.output.mkdir(parents=True,exist_ok=True)
    for name,frame in (('resale-city-import',pd.DataFrame(imported)),('resale-price-share-city',result),
                       ('resale-summary',pd.DataFrame(summaries)),('B-C-D-city-comparison',compare),
                       ('B-C-D-summary',pd.DataFrame(compare_summary)),('part2-main-with-resale',augmented)):
        frame.to_csv(args.output/f'{name}.csv',index=False,encoding='utf-8-sig')
    atomic_json(args.output/'source-audit.json',{'passed':True,'source_rows':len(data),'derived_price_share_rows':len(result),
        'source_sha256':sha256(args.source),'source_summary_sha256':sha256(args.source_summary),
        'source_directory':'/data3/dataset/ljt/claude-sell-idle-20261108',
        'part2_table_sha256':sha256(args.part2/'part2-main-table.csv'),
        'generated_at':datetime.now(timezone.utc).isoformat(),
        'energy_assumption':'窗口任务动态功率除以已分配 GPU 小时，乘原时序闲置 GPU 加权 PUE 和含碳市电价格',
        'dispatch_recomputed':False,'baseline_target':'economic'})
    print('imported',len(data),'city rows;',len(result),'price-share entries; source medians verified')


def main():
    parser=argparse.ArgumentParser()
    for name in ('project','source','source-summary','part2','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__':
    main()
