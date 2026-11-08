import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from column_generation_capacity import WholePricing, save_plan
from highs_capacity_solver import solve_capacity_highs
from joint_capacity_copt import make_resource_schedule
from joint_capacity_inputs import sha256
from minimum_server_capacity import DEADLINES, resource_dispatch


def run(args, deadline):
    started = time.monotonic()
    whole_h, split_h = DEADLINES[deadline]
    scenario = args.scenario
    fleet = json.loads((args.resources / 'resource-manifest.json').read_text('utf-8'))[scenario]
    stem = args.relaxations / f'{scenario}-w{whole_h}-s{split_h}'
    certificate = json.loads(stem.with_suffix('.json').read_text('utf-8'))
    data = np.load(stem.with_suffix('.npz'))
    source = pd.read_parquet(args.reference / f'{scenario}-10MW.parquet')
    coarse_jobs = pd.read_parquet(args.aggregate_jobs / f'{scenario}.parquet')
    fine_jobs = pd.read_parquet(args.whole_jobs / f'{scenario}.parquet')
    coarse = WholePricing(coarse_jobs, source.source_hour_index.to_numpy(), whole_h)
    fine = WholePricing(fine_jobs, coarse.hours, whole_h)
    mapping = pd.read_parquet(args.aggregate_jobs / f'{scenario}.mapping.parquet')
    np.testing.assert_array_equal(mapping.job, fine.labels)
    coarse_choices = data['pattern_choices'] & coarse.valid & coarse.movable[:, None]
    group, shift = np.nonzero(coarse_choices)
    fluid_count = len(data['assignment']) - len(group)
    y = np.zeros(coarse.valid.shape)
    y[group, shift] = data['assignment'][fluid_count:]
    y[~coarse.movable, 0] = 1.
    total_weight = np.bincount(mapping.cohort, weights=mapping.amplitude * mapping.job_count,
                                minlength=len(coarse.labels))
    order = np.lexsort((-mapping.amplitude.to_numpy(), mapping.cohort.to_numpy()))
    members = np.bincount(mapping.cohort, minlength=len(coarse.labels))
    offsets = np.r_[0, np.cumsum(members)]
    keys, quantities = [], []
    for cohort in range(len(coarse.labels)):
        destinations = np.flatnonzero(y[cohort] > 1e-12)
        if not len(destinations):
            raise ValueError('LP 聚合作业没有执行位置')
        fractions = y[cohort, destinations]
        fractions = fractions / fractions.sum()
        remaining = fractions * total_weight[cohort]
        destination = 0
        for member in order[offsets[cohort]:offsets[cohort + 1]]:
            amount = float(mapping.job_count.iloc[member])
            amplitude = float(mapping.amplitude.iloc[member])
            if amplitude == 0 or len(destinations) == 1:
                keys.append(member * (whole_h + 1) + int(destinations[0]))
                quantities.append(amount)
                continue
            while amount > 0:
                while destination < len(destinations) - 1 and remaining[destination] < 1e-10:
                    destination += 1
                allocation = (amount if destination == len(destinations) - 1 else
                              min(amount, max(0., remaining[destination]) / amplitude))
                if allocation <= 0:
                    raise ValueError('原始作业还原未能守恒')
                keys.append(member * (whole_h + 1) + int(destinations[destination]))
                quantities.append(allocation)
                amount -= allocation
                remaining[destination] -= allocation * amplitude
    keys, quantities = np.asarray(keys, dtype=np.int64), np.asarray(quantities)
    choices = np.zeros(fine.valid.shape, dtype=bool)
    choices[:, 0] = True
    choices.ravel()[keys] = True
    fractional_groups = np.unique((keys[np.abs(quantities - np.rint(quantities)) > 1e-6] // (whole_h + 1)))
    choices[fractional_groups] = fine.valid[fractional_groups]
    args.pattern_choices = choices
    schedule = make_resource_schedule(args, scenario, fleet, whole_h=whole_h, split_h=split_h)
    selected_keys = np.flatnonzero(choices & fine.valid & fine.movable[:, None])
    whole_values = np.zeros(len(selected_keys))
    movable = fine.movable[keys // (whole_h + 1)]
    np.add.at(whole_values, np.searchsorted(selected_keys, keys[movable]), quantities[movable])
    assignment = np.r_[data['assignment'][:fluid_count], whole_values]
    peak = float(resource_dispatch(schedule, assignment).max())
    prefix = args.output / f'{scenario}-w{whole_h}-s{split_h}'
    result = solve_capacity_highs(schedule, fleet['baseline_units'], prefix,
        args.time_limit, full_mip=False, fractional=np.r_[peak, assignment],
        certified_lower=certificate['continuous_lower_servers'])
    result.update(scenario=scenario, whole_h=whole_h, split_h=split_h,
        baseline_servers=fleet['baseline_units'], gpus_per_server=fleet['gpus_per_unit'],
        method='归一化占用曲线合并 LP、原始作业完整价格下界与原始整数作业修复',
        original_whole_groups=len(fine.labels), lp_cohorts=len(coarse.labels),
        continuous_lower_servers=certificate['continuous_lower_servers'],
        whole_jobs_sha256=sha256(args.whole_jobs / f'{scenario}.parquet'),
        source_manifest_sha256=sha256(args.resources / 'resource-manifest.json'),
        direct_pricing_check=certificate['direct_pricing_check'],
        seconds=certificate['seconds'] + time.monotonic() - started)
    save_plan(args, scenario, deadline, fine, choices, schedule,
              np.load(prefix.with_suffix('.npz'))['assignment'], result)
    print(scenario, whole_h, split_h, result['lower_servers'], result['servers'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('reference', 'resources', 'whole-jobs', 'aggregate-jobs', 'relaxations', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--deadline', type=int, choices=(0, 1, 2), action='append')
    parser.add_argument('--time-limit', type=int, default=300)
    args = parser.parse_args()
    args.hourly_whole = True
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl

    for deadline in args.deadline or (0, 1, 2):
        whole_h, split_h = DEADLINES[deadline]
        stem = args.output / f'{args.scenario}-w{whole_h}-s{split_h}'
        with stem.with_suffix('.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            if stem.with_suffix('.json').is_file():
                existing = json.loads(stem.with_suffix('.json').read_text('utf-8'))
                if (existing['gap_servers'] <= 1 and existing.get('whole_jobs_sha256') ==
                        sha256(args.whole_jobs / f'{args.scenario}.parquet')):
                    continue
            run(args, deadline)


if __name__ == '__main__':
    main()
