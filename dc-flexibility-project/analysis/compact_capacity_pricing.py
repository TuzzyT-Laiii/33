import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import coptpy as cp

from column_generation_capacity import WholePricing, full_dual_bound
from full_capacity_relaxation import compact_model
from joint_capacity_copt import make_resource_schedule
from joint_capacity_inputs import atomic_json, sha256
from minimum_server_capacity import DEADLINES


def run(args):
    started = time.monotonic()
    whole_h, split_h = DEADLINES[args.deadline]
    fleet = json.loads((args.resources/'resource-manifest.json').read_text('utf-8'))[args.scenario]
    hours = pd.read_parquet(args.reference/f'{args.scenario}-10MW.parquet').source_hour_index.to_numpy()
    pricing = WholePricing(pd.read_parquet(args.whole_jobs/f'{args.scenario}.parquet'), hours, whole_h)
    fine = WholePricing(pd.read_parquet(args.fine_whole_jobs/f'{args.scenario}.parquet'), hours, whole_h)
    original_args = copy.copy(args)
    original_args.whole_jobs = args.fine_whole_jobs
    original_args.pattern_choices = np.zeros(fine.valid.shape, dtype=bool)
    original_args.pattern_choices[:, 0] = True
    original_fixed = make_resource_schedule(original_args, args.scenario, fleet, whole_h=whole_h, split_h=split_h).fixed
    lower = json.loads((args.lower_bounds/f'{args.scenario}-w{whole_h}-s{split_h}.json').read_text('utf-8'))['resource_lower_servers']
    best_lower = lower
    choices = np.zeros(pricing.valid.shape, dtype=bool)
    choices[:, ::max(1, whole_h//12)] = True
    choices[:, whole_h] = True
    choices &= pricing.valid
    history = []
    stem = args.output/f'{args.scenario}-w{whole_h}-s{split_h}'
    for iteration in range(args.iterations):
        base, matrix, left, right, blocks, amplitude, first = compact_model(args, pricing, fleet, whole_h, split_h, choices)
        lb = np.zeros(matrix.shape[1])
        lb[0] = lower
        ub = np.r_[np.full(first, fleet['baseline_units']), base.block_sizes[blocks]*amplitude[blocks]]
        env = cp.Envr()
        model = env.createModel('compact_capacity_pricing')
        for key, value in {'Logging': 1, 'Threads': 1, 'LpMethod': 2, 'Crossover': 1,
                           'FeasTol': 1e-8, 'DualTol': 1e-8, 'TimeLimit': args.time_limit}.items():
            model.setParam(key, value)
        model.loadMatrix(np.r_[1., np.zeros(matrix.shape[1]-1)], matrix, left, right, lb, ub)
        model.solve()
        if not model.getAttr(cp.COPT.Attr.HasLpSol):
            raise RuntimeError('容量价格迭代没有可行 LP 方案')
        values = np.asarray(model.getValues())
        error = max(np.maximum(left-matrix@values, 0).max(), np.maximum(matrix@values-right, 0).max())
        if error > 1e-5:
            raise ValueError(f'容量价格迭代原始残差超限 {error}')
        n = len(hours)
        dual = np.minimum(np.asarray(model.getDuals())[2*n:4*n].reshape(2,n).T, 0)
        model.clear()
        env.close()
        assignment = values[first:]/amplitude[blocks]
        peak = float(values[0])
        del matrix, left, right, lb, ub, values
        prices, best, _ = pricing.price(dual, choices)
        original_prices = fine.price(dual, fine.valid)[0]
        bound = full_dual_bound(base, dual, original_prices, fine, lower,
                                fleet['baseline_units'], fixed_override=original_fixed)
        best_lower = max(best_lower, bound)
        if best_lower > peak+1e-4:
            raise ValueError('完整候选下界超过受限 LP 可行上界')
        history.append({'iteration': iteration, 'global_lower': best_lower, 'restricted_lp': peak,
                        'options': len(assignment), 'seconds': time.monotonic()-started})
        atomic_json(stem.with_suffix('.progress.json'), {'scenario': args.scenario, 'history': history})
        print(args.scenario, whole_h, split_h, iteration, best_lower, peak, len(assignment), flush=True)
        if np.ceil(peak-1e-5) <= np.ceil(best_lower-1e-5) or iteration == args.iterations-1:
            np.savez_compressed(stem.with_suffix('.npz'), assignment=assignment, pattern_choices=choices)
            atomic_json(stem.with_suffix('.json'), {'scenario': args.scenario, 'whole_h': whole_h,
                'split_h': split_h, 'status': 'relaxation_complete', 'relaxation_peak': peak,
                'continuous_lower_servers': best_lower, 'seconds': time.monotonic()-started,
                'pricing_history': history, 'direct_pricing_check': fine.checked_prices,
                'fine_sha256': sha256(args.fine_whole_jobs/f'{args.scenario}.parquet')})
            return
        groups, shifts = np.nonzero(choices & pricing.valid & pricing.movable[:,None])
        fluid_count = len(assignment)-len(groups)
        positive = assignment[fluid_count:] > 1e-8
        following = np.zeros_like(choices)
        following[:,0] = True
        following[groups[positive],shifts[positive]] = True
        for offset in (-1,0,1):
            selected = (best+offset).clip(0,whole_h)
            following[np.arange(len(best)),selected] |= pricing.valid[np.arange(len(best)),selected]
        if np.array_equal(following,choices):
            raise RuntimeError('候选不再变化但容量上下界尚未达到目标精度')
        choices = following


def main():
    parser = argparse.ArgumentParser()
    for name in ('reference','resources','whole-jobs','fine-whole-jobs','lower-bounds','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--scenario',required=True)
    parser.add_argument('--deadline',type=int,choices=(0,1,2),required=True)
    parser.add_argument('--iterations',type=int,default=8)
    parser.add_argument('--time-limit',type=int,default=600)
    args = parser.parse_args()
    args.hourly_whole = True
    args.output.mkdir(parents=True,exist_ok=True)
    run(args)


if __name__=='__main__':
    main()
