import argparse
import json
from pathlib import Path
import time

import numpy as np

from joint_capacity_copt import make_resource_schedule
from joint_capacity_inputs import atomic_json, sha256
from minimum_server_capacity import DEADLINES, compulsory_resources


def main():
    parser=argparse.ArgumentParser()
    for name in ('reference','resources','whole-jobs','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--scenario',default='A-HPC')
    parser.add_argument('--deadline',type=int,choices=(0,1,2),default=0)
    args=parser.parse_args()
    args.hourly_whole=True
    fleet=json.loads((args.resources/'resource-manifest.json').read_text('utf-8'))[args.scenario]
    whole_h,split_h=DEADLINES[args.deadline]
    started=time.monotonic()
    schedule=make_resource_schedule(args,args.scenario,fleet,whole_h=whole_h,split_h=split_h)
    mandatory,_,lower=compulsory_resources(schedule)
    result={'scenario':args.scenario,'whole_h':whole_h,'split_h':split_h,
        'integer_job_lower_servers':lower,'lower_servers':int(np.ceil(lower-1e-5)),
        'compulsory_profile_peak':float(mandatory.max()),'seconds':time.monotonic()-started,
        'method':'所有作业必占资源叠加，再检查每个完整作业的全部起始候选',
        'whole_jobs_sha256':sha256(args.whole_jobs/f'{args.scenario}.parquet'),
        'source_manifest_sha256':sha256(args.resources/'resource-manifest.json')}
    atomic_json(args.output,result)
    print(result,flush=True)


if __name__=='__main__':
    main()
