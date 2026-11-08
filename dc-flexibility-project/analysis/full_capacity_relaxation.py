import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
import coptpy as cp

from column_generation_capacity import WholePricing, full_dual_bound
from joint_capacity_copt import make_resource_schedule
from joint_capacity_inputs import atomic_json, sha256
from minimum_server_capacity import DEADLINES


def compact_model(args, pricing, fleet, whole_h, split_h, choices=None):
    args.pattern_choices = np.zeros(pricing.valid.shape, dtype=bool)
    args.pattern_choices[:, 0] = True
    base = make_resource_schedule(args, args.scenario, fleet, whole_h=whole_h, split_h=split_h)
    n = len(base.hours)
    fluid_count = int((~base.integer).sum())
    fluid_blocks = base.block_count - int(pricing.movable.sum())
    selections = pricing.valid if choices is None else choices & pricing.valid
    groups, shifts = np.nonzero(selections & pricing.movable[:, None])
    whole_block = np.cumsum(pricing.movable) - 1 + fluid_blocks
    blocks = np.r_[base.blocks[:fluid_count], whole_block[groups]]
    amplitude = np.zeros(base.block_count)
    fluid_entries = base.columns < fluid_count
    np.maximum.at(amplitude, base.blocks[base.columns[fluid_entries]], base.values[fluid_entries, :2].max(axis=1))
    amplitude[fluid_blocks:] = np.maximum.reduce((pricing.leading[pricing.movable].max(axis=1),
        pricing.trailing[pricing.movable].max(axis=1), pricing.interior[pricing.movable].max(axis=1)))
    amplitude[amplitude == 0] = 1.
    if getattr(args, 'integer', False):
        amplitude[:] = 1.
    first_option = 1 + 2 * n
    rows, cols, values = [], [], []

    def add(r, c, v):
        r, c, v = np.broadcast_arrays(r, c, v)
        valid = (r >= 0) & (r < 4 * n + base.block_count) & (v != 0)
        rows.append(r[valid].astype(np.int32))
        cols.append(c[valid].astype(np.int32))
        values.append(v[valid])

    def delta(slots, columns, resource):
        for r in range(2):
            valid = slots < n
            add(r * n + slots[valid], first_option + columns[valid], -resource[valid, r])

    slot = base.slots[fluid_entries]
    col = base.columns[fluid_entries]
    resource = base.values[fluid_entries, :2] / amplitude[base.blocks[col], None]
    delta(slot, col, resource)
    delta(slot + 1, col, -resource)
    col = fluid_count + np.arange(len(groups))
    lead = pricing.leading[groups] / amplitude[blocks[col], None]
    tail = pricing.trailing[groups] / amplitude[blocks[col], None]
    middle = pricing.interior[groups] / amplitude[blocks[col], None]
    left = np.searchsorted(base.hours, pricing.release[groups] + shifts)
    right = np.searchsorted(base.hours, pricing.finish[groups] + shifts)
    width = pricing.widths[groups]
    delta(left, col, lead)
    one = width == 1
    delta(left[one] + 1, col[one], -lead[one])
    short = width == 2
    delta(right[short], col[short], tail[short] - lead[short])
    long = width > 2
    delta(left[long] + 1, col[long], middle[long] - lead[long])
    delta(right[long], col[long], tail[long] - middle[long])
    many = width > 1
    delta(right[many] + 1, col[many], -tail[many])
    t = np.arange(n)
    for r in range(2):
        add(r * n + t, 1 + r * n + t, 1.)
        add(r * n + t[1:], 1 + r * n + t[:-1], -1.)
        add(2 * n + r * n + t, 1 + r * n + t, 1.)
        add(2 * n + r * n + t, 0, -1.)
    add(4 * n + blocks, first_option + np.arange(len(blocks)), 1.)
    matrix = sparse.csc_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(4 * n + base.block_count, first_option + len(blocks)))
    fixed_delta = np.diff(base.fixed[:, :2], axis=0, prepend=np.zeros((1, 2))).T.ravel()
    left_bound = np.r_[fixed_delta, np.full(2 * n, -cp.COPT.INFINITY), base.block_sizes * amplitude]
    right_bound = np.r_[fixed_delta, np.zeros(2 * n), base.block_sizes * amplitude]
    return base, matrix, left_bound, right_bound, blocks, amplitude, first_option


def run(args):
    started = time.monotonic()
    whole_h, split_h = DEADLINES[args.deadline]
    fleet = json.loads((args.resources / 'resource-manifest.json').read_text('utf-8'))[args.scenario]
    frame = pd.read_parquet(args.reference / f'{args.scenario}-10MW.parquet')
    pricing = WholePricing(pd.read_parquet(args.whole_jobs / f'{args.scenario}.parquet'),
                            frame.source_hour_index.to_numpy(), whole_h)
    base, matrix, left, right, blocks, amplitude, first = compact_model(args, pricing, fleet, whole_h, split_h)
    resource = json.loads((args.lower_bounds / f'{args.scenario}-w{whole_h}-s{split_h}.json').read_text('utf-8'))
    lower = resource['resource_lower_servers']
    lb = np.zeros(matrix.shape[1])
    lb[0] = lower
    ub = np.r_[np.full(first, fleet['baseline_units']), base.block_sizes[blocks] * amplitude[blocks]]
    print('compact model', matrix.shape, matrix.nnz, 'seconds', time.monotonic() - started, flush=True)
    env = cp.Envr()
    model = env.createModel('compact_capacity_relaxation')
    for key, value in {'Logging': 1, 'Threads': 1, 'LpMethod': 2, 'Crossover': 1,
                       'FeasTol': 1e-8, 'DualTol': 1e-8, 'TimeLimit': args.time_limit}.items():
        model.setParam(key, value)
    model.loadMatrix(np.r_[1., np.zeros(matrix.shape[1] - 1)], matrix, left, right, lb, ub)
    model.solve()
    status = model.status
    if not model.getAttr(cp.COPT.Attr.HasLpSol):
        raise RuntimeError(f'完整容量 LP 尚未提供可核验方案，状态 {status}')
    values = np.asarray(model.getValues())
    residual = float(max(np.maximum(left - matrix @ values, 0).max(), np.maximum(matrix @ values - right, 0).max()))
    if residual > 1e-5:
        raise ValueError(f'差分容量 LP 残差超限 {residual}')
    n = len(frame)
    dual = np.minimum(np.asarray(model.getDuals())[2 * n:4 * n].reshape(2, n).T, 0)
    model.clear()
    env.close()
    assignment = values[first:] / amplitude[blocks]
    fine = WholePricing(pd.read_parquet(args.fine_whole_jobs / f'{args.scenario}.parquet'), pricing.hours, whole_h)
    original_args = copy.copy(args)
    original_args.whole_jobs = args.fine_whole_jobs
    original_args.pattern_choices = np.zeros(fine.valid.shape, dtype=bool)
    original_args.pattern_choices[:, 0] = True
    fixed = make_resource_schedule(original_args, args.scenario, fleet,
                                    whole_h=whole_h, split_h=split_h).fixed
    prices = fine.price(dual, fine.valid)[0]
    bound = max(lower, full_dual_bound(base, dual, prices, fine, lower,
                                     fleet['baseline_units'], fixed_override=fixed))
    if bound > values[0] + 1e-4:
        raise ValueError('原始作业价格下界超过可行 LP 上界')
    stem = args.output / f'{args.scenario}-w{whole_h}-s{split_h}'
    np.savez_compressed(stem.with_suffix('.npz'), assignment=assignment, pattern_choices=pricing.valid)
    atomic_json(stem.with_suffix('.json'), {
        'scenario': args.scenario, 'whole_h': whole_h, 'split_h': split_h,
        'status': 'relaxation_complete', 'relaxation_peak': float(values[0]),
        'lp_solver_status': int(status), 'lp_solver_optimal': status == cp.COPT.OPTIMAL,
        'continuous_lower_servers': float(bound), 'seconds': time.monotonic() - started,
        'audit': {'compact_rows': residual}, 'direct_pricing_check': fine.checked_prices,
        'coarse_sha256': sha256(args.whole_jobs / f'{args.scenario}.parquet'),
        'fine_sha256': sha256(args.fine_whole_jobs / f'{args.scenario}.parquet'),
        'reference_sha256': sha256(args.reference / f'{args.scenario}-10MW.parquet'),
        'source_manifest_sha256': sha256(args.resources / 'resource-manifest.json')})
    print(args.scenario, whole_h, split_h, bound, values[0], flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('reference', 'resources', 'whole-jobs', 'fine-whole-jobs', 'lower-bounds', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--deadline', type=int, choices=(0, 1, 2), required=True)
    parser.add_argument('--time-limit', type=int, default=600)
    args = parser.parse_args()
    args.hourly_whole = True
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl

    whole_h, split_h = DEADLINES[args.deadline]
    stem = args.output / f'{args.scenario}-w{whole_h}-s{split_h}'
    with stem.with_suffix('.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if stem.with_suffix('.json').is_file():
            existing = json.loads(stem.with_suffix('.json').read_text('utf-8'))
            hashes = {'fine_sha256': args.fine_whole_jobs / f'{args.scenario}.parquet',
                      'coarse_sha256': args.whole_jobs / f'{args.scenario}.parquet',
                      'reference_sha256': args.reference / f'{args.scenario}-10MW.parquet',
                      'source_manifest_sha256': args.resources / 'resource-manifest.json'}
            if all(existing.get(key) == sha256(path) for key, path in hashes.items()):
                return
        run(args)


if __name__ == '__main__':
    main()
