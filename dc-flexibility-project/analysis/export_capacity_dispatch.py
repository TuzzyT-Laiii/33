import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from joint_capacity_copt import make_schedule
from joint_capacity_inputs import atomic_json
from minimum_server_capacity import DEADLINES, audit, resource_dispatch
from resource_capacity_lower_bound import resource_layers
from capacity_plan import decode, derive_family


def main():
    parser = argparse.ArgumentParser()
    for name in ('reference', 'resources', 'whole-jobs', 'city-inputs', 'results', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--deadline', type=int, choices=(0, 1, 2), action='append')
    parser.add_argument('--derive-family', action='store_true')
    args = parser.parse_args()
    args.hourly_whole = True
    args.output.mkdir(parents=True, exist_ok=True)
    fleet = json.loads((args.resources / 'resource-manifest.json').read_text('utf-8'))[args.scenario]
    jobs = pd.read_parquet(args.whole_jobs / f'{args.scenario}.parquet')
    for deadline in args.deadline or (0, 1, 2):
        whole_h, split_h = DEADLINES[deadline]
        name = f'{args.scenario}-w{whole_h}-s{split_h}'
        result = json.loads((args.results / f'{name}.json').read_text('utf-8'))
        if result['gap_servers'] > 1:
            raise ValueError(f'{name} 仍缺少台数证明')
        schedule = make_schedule(args, args.scenario, fleet, whole_h=whole_h, split_h=split_h)
        assignment = np.load(args.results / f'{name}.npz')['assignment']
        audit(schedule, assignment, result['servers'], True)
        _, rigid_resources, _, _ = resource_layers(args, args.scenario, fleet)
        frame = pd.read_parquet(args.city_inputs / f'frankfurt-{args.scenario}.parquet')
        whole = schedule.fixed - np.column_stack((rigid_resources, frame.rigid + frame.unknown))
        chosen = schedule.integer[schedule.columns]
        np.add.at(whole, schedule.slots[chosen], schedule.values[chosen] * assignment[schedule.columns[chosen], None])
        if whole.min() < -1e-6:
            raise ValueError('整体作业的固定调度出现负资源量')
        plan = pd.DataFrame(decode(schedule, assignment, jobs, whole_h), columns=['job', 'shift_h', 'count'])
        expected = jobs.groupby('job').job_count.first()
        np.testing.assert_array_equal(plan.groupby('job')['count'].sum().reindex(expected.index), expected)
        plan.to_parquet(args.output / f'{name}.whole-plan.parquet', index=False)
        np.savez_compressed(args.output / f'{name}.npz', assignment=assignment)
        np.savez_compressed(args.output / f'{name}.dispatch.npz',
                            resource_dispatch=resource_dispatch(schedule, assignment), whole_dispatch=whole.clip(min=0))
        atomic_json(args.output / f'{name}.json', {**result, 'method': '完整作业模式 LP 下界与整数修复可行上界'})
        if args.derive_family:
            derive_family(args, args.scenario, deadline)
        print(name, result['lower_servers'], result['servers'], flush=True)


if __name__ == '__main__':
    main()
