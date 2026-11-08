import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import coptpy as cp

from capacity_plan import derive_family
from column_generation_capacity import WholePricing, save_plan
from full_capacity_relaxation import compact_model
from joint_capacity_copt import make_resource_schedule
from joint_capacity_inputs import sha256
from minimum_server_capacity import DEADLINES, audit, resource_dispatch


def run(args, deadline):
    started=time.monotonic()
    whole_h,split_h=DEADLINES[deadline]
    fleet=json.loads((args.resources/'resource-manifest.json').read_text('utf-8'))[args.scenario]
    hours=pd.read_parquet(args.reference/f'{args.scenario}-10MW.parquet').source_hour_index.to_numpy()
    pricing=WholePricing(pd.read_parquet(args.whole_jobs/f'{args.scenario}.parquet'),hours,whole_h)
    choices=np.zeros(pricing.valid.shape,dtype=bool)
    choices[:,0]=True
    base,matrix,left,right,blocks,amplitude,first=compact_model(args,pricing,fleet,whole_h,split_h,choices)
    lower=json.loads((args.lower_bounds/f'{args.scenario}-w{whole_h}-s{split_h}.json').read_text('utf-8'))['resource_lower_servers']
    target=int(np.ceil(lower-1e-5))
    lb=np.zeros(matrix.shape[1])
    ub=np.r_[np.full(first,fleet['baseline_units']),base.block_sizes[blocks]*amplitude[blocks]]
    lb[0]=ub[0]=target
    env=cp.Envr()
    model=env.createModel('stationary_whole_capacity')
    for key,value in {'Logging':1,'Threads':1,'LpMethod':2,'Crossover':1,
                      'FeasTol':1e-8,'DualTol':1e-8,'TimeLimit':args.time_limit}.items():
        model.setParam(key,value)
    model.loadMatrix(np.zeros(matrix.shape[1]),matrix,left,right,lb,ub)
    model.solve()
    if not model.getAttr(cp.COPT.Attr.HasLpSol):
        raise RuntimeError('整体作业保持原时序时，尚未找到达到资源下界的方案')
    values=np.asarray(model.getValues())
    assignment=values[first:]/amplitude[blocks]
    model.clear()
    env.close()
    del matrix,values
    args.pattern_choices=choices
    schedule=make_resource_schedule(args,args.scenario,fleet,whole_h=whole_h,split_h=split_h)
    checks=audit(schedule,assignment,target,True)
    peak=float(resource_dispatch(schedule,assignment).max())
    result={'scenario':args.scenario,'whole_h':whole_h,'split_h':split_h,
        'servers':target,'lower_servers':target,'gap_servers':0,'status':'certified_within_one',
        'continuous_lower_servers':lower,'continuous_upper_servers':peak,
        'baseline_servers':fleet['baseline_units'],'gpus_per_server':fleet['gpus_per_unit'],
        'whole_jobs_sha256':sha256(args.whole_jobs/f'{args.scenario}.parquet'),
        'source_manifest_sha256':sha256(args.resources/'resource-manifest.json'),
        'method':'整体作业全部保留原时序，连续工作量满足资源下界，原始任务完整核验',
        'seconds':time.monotonic()-started,'audit':checks}
    import fcntl

    stem=args.output/f'{args.scenario}-w{whole_h}-s{split_h}'
    with stem.with_suffix('.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        save_plan(args,args.scenario,deadline,pricing,choices,schedule,assignment,result)
    derive_family(args,args.scenario,deadline)
    print(args.scenario,whole_h,split_h,target,flush=True)


def main():
    parser=argparse.ArgumentParser()
    for name in ('reference','resources','whole-jobs','lower-bounds','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--scenario',default='B-SAI',choices=('B-SAI',))
    parser.add_argument('--deadline',type=int,choices=(0,1,2),action='append')
    parser.add_argument('--time-limit',type=int,default=300)
    args=parser.parse_args()
    args.hourly_whole=True
    args.output.mkdir(parents=True,exist_ok=True)
    for deadline in args.deadline or (0,1,2):
        run(args,deadline)


if __name__=='__main__':
    main()
