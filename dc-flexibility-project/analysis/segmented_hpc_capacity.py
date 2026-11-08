import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from capacity_plan import apply_whole_plan
from column_generation_capacity import WholePricing
from integer_capacity_model import run as integer_run
from joint_capacity_inputs import atomic_json, sha256
from minimum_server_capacity import DEADLINES


def prepare_case(args, mode, index, lo, hi, source, resources, jobs, fixed, warm):
    case = args.output/f'{mode}-{index:02d}'
    (case/'reference').mkdir(parents=True,exist_ok=True)
    (case/'jobs').mkdir(exist_ok=True)
    grouped = jobs.groupby('job').source_hour_index.agg(['min','max'])
    selected = grouped.index[(grouped['min'].ge(lo)&grouped['max'].lt(hi)) if mode=='upper' else
                              (grouped['min'].lt(hi)&grouped['max'].ge(lo))]
    selected_jobs = jobs.loc[jobs.job.isin(selected)].copy()
    reference = source.copy() if mode=='lower' else source.loc[source.source_hour_index.ge(lo)&source.source_hour_index.lt(hi)].copy()
    layers = resources.copy() if mode=='lower' else resources.loc[reference.index].copy()
    if mode=='lower':
        demand = np.zeros((len(source),2))
        positions = pd.Index(source.source_hour_index).get_indexer(selected_jobs.source_hour_index)
        np.add.at(demand,positions,selected_jobs[['cpu_units','gpu_units']].to_numpy()*selected_jobs.job_count.to_numpy()[:,None])
        reference[['u_cpu_proxy','allocated_gpu_fraction']] = (fixed+demand)/args.fleet['reference_units']
        layers['cpu:CPU 科研分区'] = demand[:,0]/args.fleet['reference_units']
        layers['gpu:CPU 科研分区'] = demand[:,1]/args.fleet['reference_units']
    reference.to_parquet(case/'reference/A-HPC-10MW.parquet',index=False)
    layers.to_parquet(case/'reference/HPC-resources.parquet',index=False)
    selected_jobs.to_parquet(case/'jobs/A-HPC.parquet',index=False)
    candidate = warm.loc[warm.job.isin(selected)].copy()
    whole_h = DEADLINES[args.deadline][0]
    pricing = WholePricing(selected_jobs,reference.source_hour_index.to_numpy(),whole_h)
    locations = pd.Index(pricing.labels).get_indexer(candidate.job)
    candidate.loc[~pricing.valid[locations,candidate.shift_h.to_numpy()], 'shift_h'] = 0
    candidate = candidate.groupby(['job','shift_h'],as_index=False)['count'].sum()
    original = reference[['u_cpu_proxy','allocated_gpu_fraction']].to_numpy()*args.fleet['reference_units']
    test_jobs=selected_jobs.copy()
    test_jobs['mw']=0.
    shifted=apply_whole_plan(np.column_stack((original,np.zeros(len(reference)))),pricing.hours,test_jobs,candidate)[:,:2]
    if shifted.max()>max(original.max(),args.lower)+1e-5:
        candidate=pd.DataFrame({'job':pricing.labels,'shift_h':0,'count':pricing.counts})
    candidate.to_parquet(case/'start.parquet',index=False)
    atomic_json(case/'scope.json',{'mode':mode,'lo':int(lo),'hi':int(hi),'source_reference_sha256':sha256(args.reference/'A-HPC-10MW.parquet'),
        'source_jobs_sha256':sha256(args.whole_jobs/'A-HPC.parquet'),'job_groups':len(selected),
        'boundary': '其余作业保留原时序且作业不得越过分段边界' if mode=='upper' else '省略选定时间段之外的可调度作业，保留真实固定负荷'})
    task=copy.copy(args)
    task.reference=case/'reference'
    task.whole_jobs=case/'jobs'
    task.output=case/'result'
    task.output.mkdir(exist_ok=True)
    task.publish_to=None
    task.warm_plan=case/'start.parquet'
    task.warm_assignment=None
    task.integer=task.hourly_whole=True
    task.presolve=task.cut_level=None
    return task


def solve_case(task):
    whole_h,split_h=DEADLINES[task.deadline]
    path=task.output/f'A-HPC-w{whole_h}-s{split_h}.json'
    if path.is_file():
        previous=json.loads(path.read_text('utf-8'))
        if previous['whole_jobs_sha256']!=sha256(task.whole_jobs/'A-HPC.parquet'):
            raise ValueError('分段检查点的作业输入版本发生变化')
        if previous['gap_servers']<=1 or previous.get('requested_time_limit',60)>=task.time_limit:
            return str(path)
        task.warm_plan=path.with_suffix('.whole-plan.parquet')
    integer_run(task)
    return str(path)


def global_plan(args, source, jobs, warm, cases, started):
    whole_h,split_h=DEADLINES[args.deadline]
    parts=[]
    used=set()
    for case in cases:
        path=case.output/f'A-HPC-w{whole_h}-s{split_h}.whole-plan.parquet'
        plan=pd.read_parquet(path)
        if used.intersection(plan.job.unique()):
            raise ValueError('分段作业重复')
        used.update(plan.job.unique())
        parts.append(plan)
    original_counts=jobs.groupby('job').job_count.first()
    remaining=original_counts.loc[~original_counts.index.isin(used)]
    parts.append(pd.DataFrame({'job':remaining.index,'shift_h':0,'count':remaining.to_numpy()}))
    candidate=pd.concat(parts,ignore_index=True)
    pricing=WholePricing(jobs,source.source_hour_index.to_numpy(),whole_h)
    original=source[['u_cpu_proxy','allocated_gpu_fraction']].to_numpy()*args.fleet['reference_units']
    test_jobs=jobs.copy()
    test_jobs['mw']=0.
    alternatives=[]
    for plan in (candidate,warm):
        np.testing.assert_array_equal(plan.groupby('job')['count'].sum().reindex(pricing.labels),pricing.counts)
        locations=pd.Index(pricing.labels).get_indexer(plan.job)
        if not pricing.valid[locations,plan.shift_h.to_numpy()].all():
            raise ValueError('全局整数作业越过期限或观测缺口')
        actual=apply_whole_plan(np.column_stack((original,np.zeros(len(source)))),pricing.hours,test_jobs,plan)[:,:2]
        alternatives.append((float(actual.max()),plan,actual))
    peak,plan,actual=min(alternatives,key=lambda item:item[0])
    lower=args.lower
    for path in args.output.glob(f'lower-*/result/A-HPC-w{whole_h}-s{split_h}.json'):
        lower=max(lower,json.loads(path.read_text('utf-8'))['continuous_lower_servers'])
    servers=int(np.ceil(peak-1e-5))
    lower_integer=int(np.ceil(lower-1e-5))
    if lower_integer>servers:
        raise ValueError('分段下界超过原始全局可行上界')
    stem=args.output/f'A-HPC-w{whole_h}-s{split_h}'
    plan.to_parquet(stem.with_suffix('.whole-plan.parquet'),index=False)
    np.savez_compressed(stem.with_suffix('.dispatch.npz'),resource_dispatch=actual)
    conservation=float(np.abs(actual.sum(axis=0)-original.sum(axis=0)).max())
    if conservation>1e-4:
        raise ValueError('全局拼接后资源不守恒')
    result={'scenario':'A-HPC','whole_h':whole_h,'split_h':split_h,'servers':servers,'lower_servers':lower_integer,
        'gap_servers':servers-lower_integer,'continuous_lower_servers':lower,'continuous_upper_servers':peak,
        'status':'certified_within_one' if servers-lower_integer<=1 else 'feasible_with_gap',
        'baseline_servers':args.fleet['baseline_units'],'gpus_per_server':0,'seconds':time.monotonic()-started,
        'method':'分段整数可行上界与删除窗口外作业的整数下界，原始作业全局拼接复核',
        'whole_jobs_sha256':sha256(args.whole_jobs/'A-HPC.parquet'),
        'source_manifest_sha256':sha256(args.resources/'resource-manifest.json'),
        'audit':{'capacity_servers':max(0.,peak-servers),'work_conservation':conservation}}
    atomic_json(stem.with_suffix('.json'),result)
    if args.publish_to is not None:
        import fcntl

        target=args.publish_to/stem.name
        with target.with_suffix('.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            existing=json.loads(target.with_suffix('.json').read_text('utf-8')) if target.with_suffix('.json').is_file() else None
            if existing is None or servers<existing['servers'] or (servers==existing['servers'] and lower_integer>existing['lower_servers']):
                plan.to_parquet(target.with_suffix('.whole-plan.parquet'),index=False)
                np.savez_compressed(target.with_suffix('.dispatch.npz'),resource_dispatch=actual)
                selected_keys=np.flatnonzero(pricing.valid&pricing.movable[:,None])
                assignment=np.zeros(len(selected_keys))
                locations=pd.Index(pricing.labels).get_indexer(plan.job)
                movable=pricing.movable[locations]
                key=locations[movable]*(whole_h+1)+plan.shift_h.to_numpy()[movable]
                np.add.at(assignment,np.searchsorted(selected_keys,key),plan['count'].to_numpy()[movable])
                np.savez_compressed(target.with_suffix('.npz'),assignment=assignment,pattern_choices=pricing.valid)
                atomic_json(target.with_suffix('.json'),result)
    print('GLOBAL',whole_h,split_h,lower_integer,servers,flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    for name in ('reference','resources','whole-jobs','lower-bounds','warm-plan','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--deadline',type=int,choices=(0,1,2),default=0)
    parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    parser.add_argument('--time-limit',type=int,default=60)
    parser.add_argument('--publish-to',type=Path)
    args=parser.parse_args()
    args.scenario='A-HPC'
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    args.fleet=json.loads((args.resources/'resource-manifest.json').read_text('utf-8'))['A-HPC']
    whole_h,split_h=DEADLINES[args.deadline]
    args.lower=json.loads((args.lower_bounds/f'A-HPC-w{whole_h}-s{split_h}.json').read_text('utf-8'))['resource_lower_servers']
    source=pd.read_parquet(args.reference/'A-HPC-10MW.parquet')
    resources=pd.read_parquet(args.reference/'HPC-resources.parquet')
    jobs=pd.read_parquet(args.whole_jobs/'A-HPC.parquet')
    warm=pd.read_parquet(args.warm_plan)
    matched=np.zeros((len(source),2))
    positions=pd.Index(source.source_hour_index).get_indexer(jobs.source_hour_index)
    np.add.at(matched,positions,jobs[['cpu_units','gpu_units']].to_numpy()*jobs.job_count.to_numpy()[:,None])
    fixed=source[['u_cpu_proxy','allocated_gpu_fraction']].to_numpy()*args.fleet['reference_units']-matched
    if fixed.min()<-1e-5:
        raise ValueError('原始逐作业资源超出真实任务总量')
    fixed=fixed.clip(min=0)
    cuts=[0,826,1330,1823,2342,2903,3832,4336,4840,5206,5710,6214,6718,7150,7416]
    if cuts[0]!=source.source_hour_index.min() or cuts[-1]!=source.source_hour_index.max()+1:
        raise ValueError('分段端点与冻结原生时间范围不一致')
    upper=[prepare_case(args,'upper',i,lo,hi,source,resources,jobs,fixed,warm) for i,(lo,hi) in enumerate(zip(cuts,cuts[1:]))]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(solve_case,upper))
    global_plan(args,source,jobs,warm,upper,started)
    ranked=sorted(range(len(upper)),key=lambda i:json.loads((upper[i].output/f'A-HPC-w{whole_h}-s{split_h}.json').read_text('utf-8'))['continuous_upper_servers'],reverse=True)[:3]
    lower_cases=[prepare_case(args,'lower',i,cuts[max(0,i-1)],cuts[min(len(cuts)-1,i+2)],source,resources,jobs,fixed,warm) for i in ranked]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(solve_case,lower_cases))
    global_plan(args,source,jobs,warm,upper,started)


if __name__=='__main__':
    main()
