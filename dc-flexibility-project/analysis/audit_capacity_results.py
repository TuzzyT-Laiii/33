import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from flex_econ_copt import SCENARIOS, CITIES
from joint_capacity_inputs import atomic_json
from joint_capacity_copt import make_resource_schedule
from minimum_server_capacity import DEADLINES, audit, resource_dispatch


def main():
    parser=argparse.ArgumentParser()
    for name in ('reference','resources','whole-jobs','capacities','cities','claude-lower','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.resources/'resource-manifest.json').read_text('utf-8'))
    claude=pd.read_csv(args.claude_lower).set_index(['scenario','whole_h','split_h'])
    checked=[]
    inference=[]
    for scenario in SCENARIOS:
        fleet=manifest[scenario]
        source=pd.read_parquet(args.reference/f'{scenario}-10MW.parquet')
        original=source[['u_cpu_proxy','allocated_gpu_fraction']].to_numpy()*fleet['reference_units']
        for whole_h,split_h in DEADLINES:
            stem=args.capacities/f'{scenario}-w{whole_h}-s{split_h}'
            result=json.loads(stem.with_suffix('.json').read_text('utf-8'))
            if result.get('gpus_per_server',fleet['gpus_per_unit'])!=fleet['gpus_per_unit'] or result['baseline_servers']!=fleet['baseline_units']:
                raise ValueError(f'{stem.name} 设备元数据不符')
            dispatch_path=stem.with_suffix('.dispatch.npz')
            if dispatch_path.is_file():
                actual=np.load(dispatch_path)['resource_dispatch']
            else:
                if fleet['use'] not in ('CLD','INF'):
                    raise ValueError(f'{stem.name} 缺少完整作业资源调度')
                args.hourly_whole=True
                schedule=make_resource_schedule(args,scenario,fleet,whole_h=whole_h,split_h=split_h)
                assignment=np.load(stem.with_suffix('.npz'))['assignment']
                if len(assignment)!=schedule.option_count:
                    raise ValueError(f'{stem.name} 原始小时工作量方案维度不一致')
                audit(schedule,assignment,result['servers'],True)
                actual=resource_dispatch(schedule,assignment)
                np.savez_compressed(dispatch_path,resource_dispatch=actual)
            if actual.shape!=original.shape or np.abs(actual.sum(axis=0)-original.sum(axis=0)).max()>1e-4:
                raise ValueError(f'{stem.name} 资源守恒不符')
            if actual.max()>result['servers']+1e-4:
                raise ValueError(f'{stem.name} 容量不足')
            if 'gpus_per_server' not in result:
                result['gpus_per_server']=fleet['gpus_per_unit']
                atomic_json(stem.with_suffix('.json'),result)
            if result['lower_servers']!=int(np.ceil(result['continuous_lower_servers']-1e-5)):
                raise ValueError(f'{stem.name} 整数下界不符')
            if 'reused_resource_model' in result and 'grouped_lp_lower_servers' in result:
                base=result['reused_resource_model']
                parent=json.loads((args.capacities/f'{base}-w{whole_h}-s{split_h}.json').read_text('utf-8'))
                factor=fleet['reference_units']/manifest[base]['reference_units']
                np.testing.assert_allclose(result['grouped_lp_lower_servers'],parent['grouped_lp_lower_servers']*factor,rtol=0,atol=1e-7)
            checked.append({'case':stem.name,'gpus_per_server':result['gpus_per_server'],
                'lower_servers':result['lower_servers'],'servers':result['servers'],
                'grouped_lp_lower_servers':result.get('grouped_lp_lower_servers'),
                'continuous_lower_servers':result['continuous_lower_servers'],
                'certified_within_one':result['gap_servers']<=1})
        if scenario.endswith('INF'):
            resources=pd.read_parquet(args.resources/f'{scenario}.parquet')
            parts=pd.read_parquet(args.reference/'INF-resources.parquet')
            actual=np.load(args.capacities/f'{scenario}-w12-s48.dispatch.npz')['resource_dispatch']
            ratios=[]
            for city in CITIES:
                for target in ('economic','alpha100'):
                    row=json.loads((args.cities/f'{city}-{scenario}-{target}.json').read_text('utf-8'))
                    ratios.append(row['continuous_job_plan']['peak_task_power_mw']/row['A']['peak_task_mw'])
            bound=float(claude.loc[(scenario,12,48),'C_min_over_K'])
            if min(ratios)<bound-.0005:
                raise ValueError('推理方案的任务功率峰值低于功率下界')
            inference.append({'scenario':scenario,'A_servers':fleet['baseline_units'],
                'gpus_per_server':fleet['gpus_per_unit'],'original_gpu_peak':float(resources.gpu_hours.max()),
                'original_gpu_mean':float(resources.gpu_hours.mean()),
                'original_gpu_server_peak':float(original[:,1].max()),'original_gpu_server_mean':float(original[:,1].mean()),
                'C_gpu_server_peak':float(actual[:,1].max()),'C_gpu_server_mean':float(actual[:,1].mean()),
                'deferrable_gpu_hours_share':float(parts['gpu:离线推理'].sum()*fleet['reference_units']*fleet['gpus_per_unit']/resources.gpu_hours.sum()),
                'C_over_A':int(np.ceil(actual.max()-1e-5))/fleet['baseline_units'],
                'claude_task_power_lower':bound,'feasible_task_power_peak_ratio_min':min(ratios),
                'feasible_task_power_peak_ratio_max':max(ratios),'whole_h':12,'split_h':48})
    atomic_json(args.output/'metadata-audit.json',{'passed':True,'cases':checked})
    pd.DataFrame(inference).to_csv(args.output/'inference-concurrency-audit.csv',index=False,encoding='utf-8-sig')
    print('metadata and resource audit',len(checked),'cases passed; certified',sum(row['certified_within_one'] for row in checked))
    print(pd.DataFrame(inference).to_string(index=False))


if __name__=='__main__':
    main()
