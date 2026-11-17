import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import coptpy as cp

from column_generation_capacity import WholePricing, save_plan
from full_capacity_relaxation import compact_model
from highs_capacity_solver import solve_capacity_highs
from joint_capacity_copt import make_resource_schedule
from joint_capacity_inputs import sha256
from minimum_server_capacity import DEADLINES, audit


def run(args):
    started = time.monotonic()
    whole_h, split_h = DEADLINES[args.deadline]
    name = f'{args.scenario}-w{whole_h}-s{split_h}'
    prior = json.loads((args.seed/name).with_suffix('.json').read_text('utf-8'))
    saved = np.load((args.seed/name).with_suffix('.npz'))
    choices = saved['pattern_choices']
    fleet = json.loads((args.resources/'resource-manifest.json').read_text('utf-8'))[args.scenario]
    hours = pd.read_parquet(args.reference/f'{args.scenario}-10MW.parquet').source_hour_index.to_numpy()
    pricing = WholePricing(pd.read_parquet(args.whole_jobs/f'{args.scenario}.parquet'), hours, whole_h)
    base, matrix, left, right, blocks, amplitude, first = compact_model(args, pricing, fleet, whole_h, split_h, choices)
    lower = prior['continuous_lower_servers']
    target = int(np.ceil(lower-1e-5))+1
    lb = np.zeros(matrix.shape[1])
    ub = np.r_[np.full(first, fleet['baseline_units']), base.block_sizes[blocks]*amplitude[blocks]]
    lb[0] = ub[0] = target
    cost = np.zeros(matrix.shape[1])
    groups, shifts = np.nonzero(choices & pricing.valid & pricing.movable[:,None])
    strength = np.maximum.reduce((pricing.leading.max(axis=1), pricing.interior.max(axis=1), pricing.trailing.max(axis=1)))
    fluid_count = len(blocks)-len(groups)
    cost[first+fluid_count:] = strength[groups]**args.priority_power * shifts/amplitude[blocks[fluid_count:]]
    cost /= max(1., cost.max())
    env = cp.Envr()
    model = env.createModel('fixed_capacity_waiting_refinement')
    for key, value in {'Logging': 1, 'Threads': 1, 'TimeLimit': args.time_limit,
                       'LpMethod': 2, 'Crossover': 1, 'FeasTol': 1e-8, 'DualTol': 1e-8}.items():
        model.setParam(key,value)
    model.loadMatrix(cost,matrix,left,right,lb,ub)
    model.solve()
    if not model.getAttr(cp.COPT.Attr.HasLpSol):
        raise RuntimeError('固定台数等待时间优化尚未提供可行调度')
    values = np.asarray(model.getValues())
    residual = max(np.maximum(left-matrix@values,0).max(),np.maximum(matrix@values-right,0).max())
    if residual>1e-5:
        raise ValueError(f'固定台数等待时间优化残差超限 {residual}')
    assignment = values[first:]/amplitude[blocks]
    model.clear()
    env.close()
    del matrix, values, left, right, cost, lb, ub
    args.pattern_choices = choices
    schedule = make_resource_schedule(args,args.scenario,fleet,whole_h=whole_h,split_h=split_h)
    audit(schedule,assignment,target,False)
    stem = args.output/name
    result = solve_capacity_highs(schedule,fleet['baseline_units'],stem,args.repair_limit,
        full_mip=False,fractional=np.r_[target,assignment],certified_lower=lower)
    result.update(scenario=args.scenario,whole_h=whole_h,split_h=split_h,
        baseline_servers=fleet['baseline_units'],gpus_per_server=fleet['gpus_per_unit'],
        whole_jobs_sha256=sha256(args.whole_jobs/f'{args.scenario}.parquet'),
        source_manifest_sha256=sha256(args.resources/'resource-manifest.json'),
        method='固定下界附近台数，优先减少大作业等待，再还原原始整数作业',
        seconds=time.monotonic()-started,priority_power=args.priority_power)
    save_plan(args,args.scenario,args.deadline,pricing,choices,schedule,
              np.load(stem.with_suffix('.npz'))['assignment'],result)
    print(args.scenario,whole_h,split_h,result['lower_servers'],result['servers'],flush=True)
    if result['gap_servers']<=1 and args.publish_to is not None:
        import fcntl

        target_path=args.publish_to/name
        with target_path.with_suffix('.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            for suffix in ('.whole-plan.parquet','.dispatch.npz','.npz','.json'):
                shutil.copyfile(stem.with_suffix(suffix),target_path.with_suffix(suffix))


def main():
    parser=argparse.ArgumentParser()
    for name in ('reference','resources','whole-jobs','seed','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--publish-to',type=Path)
    parser.add_argument('--scenario',required=True)
    parser.add_argument('--deadline',type=int,choices=(0,1,2),required=True)
    parser.add_argument('--time-limit',type=int,default=600)
    parser.add_argument('--repair-limit',type=int,default=300)
    parser.add_argument('--priority-power',type=int,default=2)
    args=parser.parse_args()
    args.hourly_whole=True
    args.output.mkdir(parents=True,exist_ok=True)
    run(args)


if __name__=='__main__':
    main()
