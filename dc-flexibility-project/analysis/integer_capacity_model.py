import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import coptpy as cp

from capacity_plan import apply_whole_plan
from column_generation_capacity import WholePricing
from full_capacity_relaxation import compact_model
from joint_capacity_copt import resource_layers
from joint_capacity_inputs import atomic_json, sha256
from minimum_server_capacity import DEADLINES


def dispatch(args, fleet, pricing, base, values, first, whole_h, split_h):
    fluid_count = int((~base.integer).sum())
    groups, shifts = np.nonzero(pricing.valid & pricing.movable[:, None])
    whole_values = np.rint(values[first + fluid_count:]).astype(np.int64)
    selected = whole_values > 0
    plan = pd.DataFrame({'job': pricing.labels[groups[selected]], 'shift_h': shifts[selected],
                         'count': whole_values[selected]})
    unchanged = pd.DataFrame({'job': pricing.labels[~pricing.movable], 'shift_h': 0,
                              'count': pricing.counts[~pricing.movable]})
    plan = pd.concat([plan, unchanged], ignore_index=True)
    np.testing.assert_array_equal(plan.groupby('job')['count'].sum().reindex(pricing.labels), pricing.counts)
    hours, rigid, whole, split = resource_layers(args, args.scenario, fleet)
    actual = np.column_stack((rigid + whole, np.zeros(len(hours))))
    jobs = pricing.jobs.copy()
    jobs['mw'] = 0.
    actual = apply_whole_plan(actual, hours, jobs, plan)[:, :2]
    entries = base.columns < fluid_count
    np.add.at(actual, base.slots[entries], base.values[entries, :2] * values[first + base.columns[entries], None])
    original = rigid + whole + split
    error = float(np.abs(actual.sum(axis=0) - original.sum(axis=0)).max())
    if error > 1e-4:
        raise ValueError(f'整数容量模型的工作量不守恒 {error}')
    return plan, actual, error


def run(args):
    started = time.monotonic()
    whole_h, split_h = DEADLINES[args.deadline]
    fleet = json.loads((args.resources / 'resource-manifest.json').read_text('utf-8'))[args.scenario]
    source = pd.read_parquet(args.reference / f'{args.scenario}-10MW.parquet')
    pricing = WholePricing(pd.read_parquet(args.whole_jobs / f'{args.scenario}.parquet'),
                           source.source_hour_index.to_numpy(), whole_h)
    base, matrix, left, right, blocks, amplitude, first = compact_model(args, pricing, fleet, whole_h, split_h)
    fluid_count = int((~base.integer).sum())
    lower = json.loads((args.lower_bounds / f'{args.scenario}-w{whole_h}-s{split_h}.json').read_text('utf-8'))['resource_lower_servers']
    lb = np.zeros(matrix.shape[1])
    lb[0] = np.ceil(lower - 1e-5)
    ub = np.r_[np.full(first, fleet['baseline_units']), base.block_sizes[blocks]]
    kinds = np.full(matrix.shape[1], cp.COPT.CONTINUOUS)
    kinds[0] = cp.COPT.INTEGER
    kinds[first + fluid_count:] = cp.COPT.INTEGER
    start = np.zeros(matrix.shape[1])
    start[0] = fleet['baseline_units']
    start[1:first] = base.original[:, :2].T.ravel()
    first_choice = np.unique(blocks, return_index=True)[1]
    start[first + first_choice] = base.block_sizes
    if args.warm_assignment is not None:
        prior = np.load(args.warm_assignment)['assignment']
        start[first:first + fluid_count] = prior[:fluid_count]
    if args.warm_plan is not None:
        plan = pd.read_parquet(args.warm_plan)
        np.testing.assert_array_equal(plan.groupby('job')['count'].sum().reindex(pricing.labels), pricing.counts)
        group, shift = np.nonzero(pricing.valid & pricing.movable[:, None])
        keys = group * (whole_h + 1) + shift
        locations = pd.Index(pricing.labels).get_indexer(plan.job)
        active = pricing.movable[locations]
        chosen = np.searchsorted(keys, locations[active] * (whole_h + 1) + plan.shift_h.to_numpy()[active])
        start[first + fluid_count:] = 0.
        np.add.at(start, first + fluid_count + chosen, plan['count'].to_numpy()[active])
        _, actual, _ = dispatch(args, fleet, pricing, base, start, first, whole_h, split_h)
        start[0] = max(np.ceil(lower - 1e-5), np.ceil(actual.max() - 1e-5))
        start[1:first] = actual.T.ravel()
        ub[0] = start[0]
    residual = max(np.maximum(left - matrix @ start, 0).max(), np.maximum(matrix @ start - right, 0).max(),
                   np.maximum(lb - start, 0).max(), np.maximum(start - ub, 0).max())
    if residual > 1e-5:
        raise ValueError(f'整数容量初始调度的差分行残差超限 {residual}')
    print('integer model', matrix.shape, matrix.nnz, 'seconds', time.monotonic() - started, flush=True)
    env = cp.Envr()
    model = env.createModel('exact_integer_minimum_capacity')
    for key, value in {'Logging': 1, 'Threads': 1, 'TimeLimit': args.time_limit,
                       'LpMethod': 2, 'RelGap': 0., 'AbsGap': 1., 'IntTol': 1e-9,
                       'FeasTol': 1e-8, 'DualTol': 1e-8}.items():
        model.setParam(key, value)
    if args.presolve is not None:
        model.setParam('Presolve', args.presolve)
    if args.cut_level is not None:
        model.setParam('CutLevel', args.cut_level)
        model.setParam('RootCutLevel', args.cut_level)
    model.loadMatrix(np.r_[1., np.zeros(matrix.shape[1] - 1)], matrix, left, right, lb, ub, kinds)
    model.setMipStart(model.getVars(), start.tolist())
    model.loadMipStart()
    model.solve()
    status = model.status
    if not model.getAttr(cp.COPT.Attr.HasMipSol):
        raise RuntimeError(f'整数容量模型未提供可行解，状态 {status}')
    values = np.asarray(model.getValues())
    bound = float(model.getAttr(cp.COPT.Attr.BestBnd))
    if not np.isfinite(bound) or bound > values[0] + 1e-4:
        raise ValueError('整数容量模型的求解器下界无效')
    bound = max(lower, bound)
    model.clear()
    env.close()
    plan, actual, conservation = dispatch(args, fleet, pricing, base, values, first, whole_h, split_h)
    servers = int(round(values[0]))
    error = max(0., float(actual.max() - servers))
    if error > 1e-4:
        raise ValueError(f'原始整数作业的实际资源超过整数台数 {error}')
    stem = args.output / f'{args.scenario}-w{whole_h}-s{split_h}'
    plan.to_parquet(stem.with_suffix('.whole-plan.parquet'), index=False)
    np.savez_compressed(stem.with_suffix('.dispatch.npz'), resource_dispatch=actual)
    np.savez_compressed(stem.with_suffix('.npz'), assignment=values[first:], pattern_choices=pricing.valid)
    lower_integer = int(np.ceil(bound - 1e-5))
    result = {'scenario': args.scenario, 'whole_h': whole_h, 'split_h': split_h,
        'status': 'certified_within_one' if servers - lower_integer <= 1 else 'feasible_with_gap',
        'servers': servers, 'lower_servers': lower_integer, 'gap_servers': servers - lower_integer,
        'continuous_lower_servers': bound, 'continuous_upper_servers': float(actual.max()),
        'baseline_servers': fleet['baseline_units'], 'gpus_per_server': fleet['gpus_per_unit'],
        'seconds': time.monotonic() - started, 'lp_solver_status': int(status),
        'requested_time_limit': args.time_limit,
        'method': '原始同类作业整数数量与资源差分模型，MIP 全局下界及原始调度上界',
        'whole_jobs_sha256': sha256(args.whole_jobs / f'{args.scenario}.parquet'),
        'reference_sha256': sha256(args.reference / f'{args.scenario}-10MW.parquet'),
        'source_manifest_sha256': sha256(args.resources / 'resource-manifest.json'),
        'audit': {'capacity_servers': error, 'work_conservation': conservation}}
    atomic_json(stem.with_suffix('.json'), result)
    print(args.scenario, whole_h, split_h, lower_integer, servers, flush=True)
    if args.publish_to is not None and result['gap_servers'] <= 1:
        import fcntl

        target = args.publish_to / stem.name
        with target.with_suffix('.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            for suffix in ('.whole-plan.parquet', '.dispatch.npz', '.npz'):
                shutil.copyfile(stem.with_suffix(suffix), target.with_suffix(suffix))
            atomic_json(target.with_suffix('.json'), result)


def main():
    parser = argparse.ArgumentParser()
    for name in ('reference', 'resources', 'whole-jobs', 'lower-bounds', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--publish-to', type=Path)
    parser.add_argument('--warm-plan', type=Path)
    parser.add_argument('--warm-assignment', type=Path)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--deadline', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--time-limit', type=int, default=1800)
    parser.add_argument('--presolve', type=int, choices=(0,1,2,3))
    parser.add_argument('--cut-level', type=int, choices=(0,1,2,3))
    args = parser.parse_args()
    args.integer = args.hourly_whole = True
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == '__main__':
    main()
