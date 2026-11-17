import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from column_generation_capacity import WholePricing
from joint_capacity_inputs import atomic_json, sha256
from minimum_server_capacity import DEADLINES


def main():
    parser=argparse.ArgumentParser()
    for name in ('reference','resources','whole-jobs','capacities','lower-bounds'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-deadline',type=int,choices=(0,1),default=0)
    parser.add_argument('--target-deadline',type=int,choices=(1,2),required=True)
    args=parser.parse_args()
    started=time.monotonic()
    old_whole,old_split=DEADLINES[args.source_deadline]
    whole_h,split_h=DEADLINES[args.target_deadline]
    if whole_h<old_whole or split_h<old_split:
        raise ValueError('目标期限必须包含来源期限的全部可行调度')
    stem=args.capacities/f'A-HPC-w{old_whole}-s{old_split}'
    target=args.capacities/f'A-HPC-w{whole_h}-s{split_h}'
    source=json.loads(stem.with_suffix('.json').read_text('utf-8'))
    if target.with_suffix('.json').is_file():
        existing=json.loads(target.with_suffix('.json').read_text('utf-8'))
        if existing['servers']<=source['servers']:
            return
    plan=pd.read_parquet(stem.with_suffix('.whole-plan.parquet'))
    hours=pd.read_parquet(args.reference/'A-HPC-10MW.parquet').source_hour_index.to_numpy()
    jobs=pd.read_parquet(args.whole_jobs/'A-HPC.parquet')
    pricing=WholePricing(jobs,hours,whole_h)
    groups=pd.Index(pricing.labels).get_indexer(plan.job)
    if not pricing.valid[groups,plan.shift_h.to_numpy()].all():
        raise ValueError('来源完整作业方案未通过新期限核验')
    np.testing.assert_array_equal(plan.groupby('job')['count'].sum().reindex(pricing.labels),pricing.counts)
    keys=np.flatnonzero(pricing.valid&pricing.movable[:,None])
    assignment=np.zeros(len(keys))
    active=pricing.movable[groups]
    selected=groups[active]*(whole_h+1)+plan.shift_h.to_numpy()[active]
    np.add.at(assignment,np.searchsorted(keys,selected),plan['count'].to_numpy()[active])
    lower=json.loads((args.lower_bounds/f'A-HPC-w{whole_h}-s{split_h}.json').read_text('utf-8'))['resource_lower_servers']
    lower_integer=int(np.ceil(lower-1e-5))
    result={**source,'whole_h':whole_h,'split_h':split_h,'lower_servers':lower_integer,
        'continuous_lower_servers':lower,'gap_servers':source['servers']-lower_integer,
        'status':'certified_within_one' if source['servers']-lower_integer<=1 else 'feasible_with_gap',
        'method':'可行上界沿用已核验的更严格期限方案，下界按本期限单独计算',
        'inherited_from_capacity_sha256':sha256(stem.with_suffix('.json')),'seconds':time.monotonic()-started}
    plan.to_parquet(target.with_suffix('.whole-plan.parquet'),index=False)
    resource=np.load(stem.with_suffix('.dispatch.npz'))['resource_dispatch']
    np.savez_compressed(target.with_suffix('.dispatch.npz'),resource_dispatch=resource)
    np.savez_compressed(target.with_suffix('.npz'),assignment=assignment,pattern_choices=pricing.valid)
    atomic_json(target.with_suffix('.json'),result)


if __name__=='__main__':
    main()
