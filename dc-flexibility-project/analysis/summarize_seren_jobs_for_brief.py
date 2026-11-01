import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys

import numpy as np
import pandas as pd

from compute_load.model import sha256
from task_trace_inputs import interval_resource_hours


TYPES = ('Pretrain', 'SFT', 'MLLM', 'Eval', 'Debug', 'Other')


def summarize(args):
    before_stat = args.jobs.stat()
    digest = sha256(args.jobs)
    source = pd.read_csv(args.source_hourly, parse_dates=['source_clock'])
    clocks = pd.DatetimeIndex(source.source_clock)
    edges = clocks.append(pd.DatetimeIndex([clocks[-1]+pd.Timedelta(hours=1)]))
    assert clocks.is_unique and clocks.to_series().diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
    arrays = {kind: {name: [] for name in ('runtime_s','wait_s','units','nodes')} for kind in TYPES}
    hourly = {kind: np.zeros(len(source)) for kind in TYPES}
    counts = {kind: {'rows':0,'valid_runtime_rows':0,'negative_wait_rows':0,'resource_hours':0.} for kind in TYPES}
    total, cpu, gpu, duration_mismatch = 0, 0, 0, 0
    fields = ['gpu_num','node_num','type','submit_time','start_time','end_time','duration']
    for chunk in pd.read_csv(args.jobs, usecols=fields, chunksize=65536):
        total += len(chunk)
        start, end, submit = [pd.to_datetime(chunk[name], utc=True, format='mixed') for name in ('start_time','end_time','submit_time')]
        runtime, wait = (end-start).dt.total_seconds(), (start-submit).dt.total_seconds()
        assert start.notna().all() and end.notna().all() and submit.notna().all()
        assert runtime.ge(0).all() and chunk.gpu_num.ge(0).all()
        duration_mismatch += int(runtime.ne(chunk.duration).sum())
        gpu_mask = chunk.gpu_num.gt(0)
        gpu += int(gpu_mask.sum())
        cpu += int((~gpu_mask).sum())
        assert set(chunk.loc[gpu_mask,'type']).issubset(TYPES)
        for kind in TYPES:
            mask = gpu_mask & chunk.type.eq(kind)
            valid = mask & runtime.ge(0) & wait.ge(0)
            record = counts[kind]
            record['rows'] += int(mask.sum())
            record['valid_runtime_rows'] += int(valid.sum())
            record['negative_wait_rows'] += int((mask & wait.lt(0)).sum())
            record['resource_hours'] += float((runtime[valid]*chunk.loc[valid,'gpu_num']/3600).sum())
            for name, values in [('runtime_s',runtime[valid]),('wait_s',wait[mask & wait.ge(0)]),
                                 ('units',chunk.loc[mask,'gpu_num']),('nodes',chunk.loc[mask,'node_num'])]:
                arrays[kind][name].append(values.to_numpy(float))
            hourly[kind] += interval_resource_hours(start[mask].astype('int64').to_numpy(),
                end[mask].astype('int64').to_numpy(),chunk.loc[mask,'gpu_num'].to_numpy(float),edges.asi8)
    denominator = sum(record['resource_hours'] for record in counts.values())
    rows = []
    qualified = source.power_quality_80.to_numpy(bool)
    for kind in TYPES:
        record = {'dataset':'Acme_Seren','resource_pool':'gpu_hours','class':kind,**counts[kind]}
        for name, parts in arrays[kind].items():
            values = np.concatenate(parts)
            for suffix, quantile in [('p50',.5),('p90',.9),('p99',.99)]:
                record[name+'_'+suffix] = float(np.quantile(values,quantile)) if len(values) else None
            record[name+'_max'] = float(np.max(values)) if len(values) else None
        record['work_fraction_of_resource_pool'] = record['resource_hours']/denominator
        record['row_fraction_of_resource_pool'] = record['rows']/gpu
        record['qualified_gpu_hours'] = float(hourly[kind][qualified].sum())
        record['qualified_gpu_share'] = record['qualified_gpu_hours']/source.loc[qualified,'allocated_gpu_hours'].sum()
        np.testing.assert_allclose(hourly[kind],source['gpu_hours_'+kind],rtol=1e-10,atol=1e-6)
        np.testing.assert_allclose(record['resource_hours'],hourly[kind].sum(),rtol=1e-10,atol=1e-6)
        rows.append(record)
    after_stat = args.jobs.stat()
    assert before_stat.st_size == after_stat.st_size and before_stat.st_mtime_ns == after_stat.st_mtime_ns
    result = {'utc':datetime.now(timezone.utc).isoformat(),'job_sha256':digest,
        'source_hourly_sha256':sha256(args.source_hourly),'script_sha256':args.script_sha256,
        'raw_rows':total,'gpu_jobs':gpu,'independent_cpu_jobs':cpu,'duration_mismatch_rows':duration_mismatch,
        'full_gpu_hours':denominator,'qualified_hours':int(qualified.sum()),'statistics':rows,
        'definition':'GPU作业完整表；有效运行需end>=start且start>=submit；GPU小时为gpu_num*(end-start)/3600；排队start-submit非负；分位数按作业数线性插值。正文占比另按合格小时区间积分。',
        'raw_job_size_mtime_unchanged':True,'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'privacy':'仅汇总，逐作业数据及作业标识未输出'}
    json.dump(result,sys.stdout,ensure_ascii=False,indent=2,allow_nan=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs',type=Path,required=True)
    parser.add_argument('--source-hourly',type=Path,required=True)
    parser.add_argument('--script-sha256',required=True)
    summarize(parser.parse_args())
