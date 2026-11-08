import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from joint_capacity_copt import whole_choices
from joint_capacity_inputs import atomic_json, sha256
from minimum_server_capacity import DEADLINES


def decode(schedule, assignment, jobs, whole_h):
    if jobs.empty:
        return []
    jobs = jobs.sort_values(['job', 'source_hour_index'])
    codes, labels = pd.factorize(jobs.job, sort=False)
    starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    valid = whole_choices(jobs.source_hour_index.to_numpy(), codes, schedule.hours, whole_h)
    movable = valid.sum(axis=1) > 1
    group, shift = np.nonzero(valid[movable])
    choices = assignment[schedule.integer]
    expected = jobs.job_count.to_numpy()[starts]
    np.testing.assert_allclose(np.bincount(group, weights=choices, minlength=movable.sum()),
                               expected[movable], rtol=0, atol=1e-5)
    used = choices > .5
    moved = pd.DataFrame({'job': labels[movable][group[used]], 'shift_h': shift[used],
                          'count': np.rint(choices[used]).astype(int)})
    unchanged = pd.DataFrame({'job': labels[~movable], 'shift_h': 0, 'count': expected[~movable]})
    return list(pd.concat([moved, unchanged], ignore_index=True).itertuples(index=False, name=None))


def apply_whole_plan(whole, hours, jobs, plan):
    shifted = plan.loc[plan.shift_h.gt(0)].merge(jobs, on='job', validate='many_to_many')
    if not len(shifted):
        return whole
    index = pd.Index(hours)
    origin = index.get_indexer(shifted.source_hour_index)
    target = index.get_indexer(shifted.source_hour_index + shifted.shift_h)
    if (origin < 0).any() or (target < 0).any():
        raise ValueError('整作业平移跨过缺失观测')
    values = shifted[['cpu_units', 'gpu_units', 'mw']].to_numpy() * shifted['count'].to_numpy()[:, None]
    np.add.at(whole, origin, -values)
    np.add.at(whole, target, values)
    if whole.min() < -1e-5:
        raise ValueError('整体作业调度产生负的资源或用电')
    return whole.clip(min=0)


def derive_family(args, base, deadline):
    whole_h, split_h = DEADLINES[deadline]
    manifest = json.loads((args.resources / 'resource-manifest.json').read_text('utf-8'))
    stem = args.output / f'{base}-w{whole_h}-s{split_h}'
    result = json.loads(stem.with_suffix('.json').read_text('utf-8'))
    data = np.load(stem.with_suffix('.dispatch.npz'))
    plan = pd.read_parquet(stem.with_suffix('.whole-plan.parquet'))
    base_source = pd.read_parquet(args.reference / f'{base}-10MW.parquet')
    base_jobs = pd.read_parquet(args.whole_jobs / f'{base}.parquet')
    for prefix in ('C', 'D'):
        scenario = prefix + base[1:]
        fleet = manifest[scenario]
        factor = fleet['reference_units'] / manifest[base]['reference_units']
        source_path = args.reference / f'{scenario}-10MW.parquet'
        jobs_path = args.whole_jobs / f'{scenario}.parquet'
        source, jobs = pd.read_parquet(source_path), pd.read_parquet(jobs_path)
        np.testing.assert_array_equal(source.source_hour_index, base_source.source_hour_index)
        np.testing.assert_allclose(source[['u_cpu_proxy', 'allocated_gpu_fraction']],
                                   base_source[['u_cpu_proxy', 'allocated_gpu_fraction']], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(jobs[['job', 'source_hour_index', 'job_count']],
                                      base_jobs[['job', 'source_hour_index', 'job_count']])
        np.testing.assert_allclose(jobs[['cpu_units', 'gpu_units']],
                                   base_jobs[['cpu_units', 'gpu_units']] * factor, rtol=1e-12, atol=1e-8)
        dispatch = data['resource_dispatch'] * factor
        original = source[['u_cpu_proxy', 'allocated_gpu_fraction']].to_numpy() * fleet['reference_units']
        conservation_error = float(np.abs(dispatch.sum(axis=0) - original.sum(axis=0)).max())
        if conservation_error > 1e-4:
            raise ValueError('硬件投影后的工作量不守恒')
        lower = int(np.ceil(result['continuous_lower_servers'] * factor - 1e-5))
        upper = int(np.ceil(dispatch.max() - 1e-5))
        derived = {**result, 'scenario': scenario, 'baseline_servers': fleet['baseline_units'],
                   'gpus_per_server': fleet['gpus_per_unit'], 'lower_servers': lower, 'servers': upper,
                   'gap_servers': upper - lower,
                   'continuous_lower_servers': result['continuous_lower_servers'] * factor,
                   'continuous_upper_servers': float(dispatch.max()),
                   'task_power_peak_mw': None, 'task_power_peak_ratio': None,
                   'resource_scale': factor, 'reused_resource_model': base,
                   'whole_jobs_sha256': sha256(jobs_path), 'reference_sha256': sha256(source_path),
                   'base_capacity_sha256': sha256(stem.with_suffix('.json')),
                   'base_model_audit': result['audit'],
                   'audit': {'capacity_servers': max(0., float(dispatch.max() - upper)),
                             'work_conservation': conservation_error},
                   'status': 'certified_within_one' if upper - lower <= 1 else 'feasible_with_gap',
                   'seconds': 0., 'solve_seconds': 0.}
        for field in ('resource_fluid_lower_servers', 'grouped_lp_lower_servers'):
            if field in result:
                derived[field] = result[field] * factor
        if 'pricing_history' in derived:
            derived['source_pricing_history'] = derived.pop('pricing_history')
        output = args.output / f'{scenario}-w{whole_h}-s{split_h}'
        np.savez_compressed(output.with_suffix('.dispatch.npz'), resource_dispatch=dispatch)
        plan.to_parquet(output.with_suffix('.whole-plan.parquet'), index=False)
        atomic_json(output.with_suffix('.json'), derived)
        print(scenario, whole_h, split_h, lower, upper, flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('reference', 'resources', 'whole-jobs', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--base', choices=('B-SAI', 'B-TRN'), required=True)
    parser.add_argument('--deadline', type=int, choices=(0, 1, 2), action='append')
    args = parser.parse_args()
    for deadline in args.deadline or (0, 1, 2):
        derive_family(args, args.base, deadline)


if __name__ == '__main__':
    main()
