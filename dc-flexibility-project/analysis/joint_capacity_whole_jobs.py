import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from joint_capacity_inputs import atomic_json, sha256
from task_trace_inputs import interval_resource_hours


HOUR_NS = 3_600_000_000_000


def whole_profile(jobs, source, power, expected):
    hours = pd.Index(source.source_hour_index)
    utc = pd.DatetimeIndex(pd.to_datetime(source.source_clock, utc=True))
    timestamps = utc.as_unit('ns').asi8
    epoch = timestamps[0] - int(hours[0]) * HOUR_NS
    np.testing.assert_array_equal((timestamps - epoch) // HOUR_NS, hours.to_numpy())
    starts = pd.DatetimeIndex(jobs.start_time).as_unit('ns').asi8 - epoch
    ends = pd.DatetimeIndex(jobs.end_time).as_unit('ns').asi8 - epoch
    counts = jobs[['cpu', 'gpu']].to_numpy(float)
    start_hours = starts // HOUR_NS
    end_hours = (ends - 1) // HOUR_NS
    complete = (start_hours >= hours[0]) & (end_hours <= hours[-1])
    full_hours = np.arange(hours[0], hours[-1] + 1)
    edges = np.arange(hours[0], hours[-1] + 2) * HOUR_NS
    traced = np.column_stack([interval_resource_hours(starts, ends, counts[:, i], edges)[hours - hours[0]]
                              for i in range(2)])
    totals = traced.sum(axis=0)
    factors = np.divide(expected.sum(axis=0), totals, out=np.zeros(2), where=totals > 0)
    np.testing.assert_allclose(traced * factors, expected, rtol=1e-7, atol=1e-7)
    unavailable = np.r_[0, (~np.isin(full_hours, hours)).cumsum()]
    ids = np.flatnonzero(complete)
    ids = ids[unavailable[end_hours[ids] - hours[0] + 1] == unavailable[start_hours[ids] - hours[0]]]
    widths = end_hours[ids] - start_hours[ids] + 1
    job_ids = np.repeat(ids, widths)
    positions = np.arange(len(job_ids)) - np.repeat(np.cumsum(widths) - widths, widths)
    slots = start_hours[job_ids] + positions
    start_in_slot = np.maximum(starts[job_ids], slots * HOUR_NS)
    end_in_slot = np.minimum(ends[job_ids], (slots + 1) * HOUR_NS)
    work = ((end_in_slot - start_in_slot) / HOUR_NS)[:, None] * counts[job_ids] * factors
    # 同一起止小时的作业采用相同平移量，保留组内作业原有连续运行区间。
    groups, _ = pd.factorize(pd.MultiIndex.from_arrays([start_hours[job_ids], end_hours[job_ids]]))
    grouped = pd.DataFrame({'job': groups, 'source_hour_index': slots, 'cpu_units': work[:, 0], 'gpu_units': work[:, 1]})
    grouped = grouped.groupby(['job', 'source_hour_index'], sort=True, as_index=False).sum()
    index = hours.get_indexer(grouped.source_hour_index)
    denominator = expected[index].sum(axis=1)
    grouped['mw'] = grouped[['cpu_units', 'gpu_units']].sum(axis=1).to_numpy() / denominator * power[index]
    eligible = grouped[['cpu_units', 'gpu_units']].sum(axis=0)
    details = {'source_jobs': len(jobs), 'eligible_jobs': len(ids), 'groups': int(grouped.job.nunique()),
               'eligible_cpu_unit_hours': float(eligible.iloc[0]), 'eligible_gpu_unit_hours': float(eligible.iloc[1]),
               'total_cpu_unit_hours': float(expected[:, 0].sum()), 'total_gpu_unit_hours': float(expected[:, 1].sum()),
               'fixed_when_incomplete': True, 'common_shift_for_same_start_and_end_hour': True}
    return grouped, details


def seren_jobs(path):
    jobs = pd.read_csv(path, usecols=['job_id', 'gpu_num', 'type', 'start_time', 'end_time'])
    jobs = jobs.loc[jobs.type.eq('Eval') & jobs.gpu_num.gt(0)].copy()
    jobs['gpu'] = jobs.gpu_num
    jobs['cpu'] = .25 * jobs.gpu
    return jobs


def lisa_jobs(path):
    jobs = pd.read_parquet(path, columns=['id', 'start_date', 'end_date', 'nodetypes', 'numcores', 'hardware_group'])
    queues = ['normal', 'shared', 'fat_soil_shared', 'sw', 'short', 'fat', 'shared_52c_384g']
    jobs = jobs.loc[jobs.hardware_group.eq('cpu_partition') & jobs.nodetypes.isin(queues)].copy()
    return jobs.rename(columns={'id': 'job_id', 'start_date': 'start_time', 'end_date': 'end_time', 'numcores': 'cpu'}).assign(gpu=0)


def mit_jobs(path, categories):
    from audit_research_gpu_demand import tres_count

    fields = ['id_job', 'time_start', 'time_end', 'time_suspended', 'tres_req', 'tres_alloc', 'job_type']
    jobs = pd.read_csv(path, usecols=fields, dtype={'tres_req': str, 'tres_alloc': str})
    arrays = pd.read_parquet(categories, columns=['id_job', 'category_id'])
    arrays = arrays.loc[arrays.category_id.eq('array'), 'id_job']
    eligible = jobs.job_type.isin(['LLSUB:BATCH', 'LLMAPREDUCE:MAP']) | (jobs.job_type.eq('OTHER') & jobs.id_job.isin(arrays))
    jobs = jobs.loc[eligible & jobs.time_start.gt(0) & jobs.time_end.gt(jobs.time_start) & jobs.time_suspended.eq(0)].copy()
    requested = sum(tres_count(jobs.tres_req, key).fillna(0) for key in (1001, 1002))
    jobs['gpu'] = sum(tres_count(jobs.tres_alloc, key).fillna(0) for key in (1001, 1002))
    jobs['cpu'] = tres_count(jobs.tres_alloc, 1).fillna(0)
    jobs = jobs.loc[requested.gt(0) & jobs.gpu.gt(0)].copy()
    jobs['start_time'] = pd.to_datetime(jobs.time_start, unit='s', utc=True)
    jobs['end_time'] = pd.to_datetime(jobs.time_end, unit='s', utc=True)
    return jobs.rename(columns={'id_job': 'job_id'})


def expected_units(source, reference, scenario):
    units = int(source.unit_count.iloc[0])
    use = scenario.split('-')[1]
    if use == 'TRN':
        gpu = source.gpu_hours_Eval.to_numpy() * units / 2032
        return np.column_stack((.25 * gpu, gpu))
    resources = pd.read_parquet(reference / f'{use}-resources.parquet')
    labels = ('CPU 科研分区',) if use == 'HPC' else ('批处理', 'MapReduce', '数组批量')
    return units * np.column_stack([resources[[resource + ':' + label for label in labels]].sum(axis=1)
                                    for resource in ('cpu', 'gpu')])


def main():
    parser = argparse.ArgumentParser()
    for name in ('jobs', 'source-audit', 'reference', 'resources', 'layers', 'output', 'resource-audit',
                 'lisa-jobs', 'mit-slurm', 'mit-categories'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.source_audit.read_text('utf-8'))
    if sha256(args.jobs) != audit['job_sha256']:
        raise ValueError('Seren 作业输入与原研究窗口的来源哈希不同')
    inputs = {row['role']: row['sha256'] for row in json.loads(args.resource_audit.read_text('utf-8'))['input_files']}
    for role, path in (('HPC_jobs_local_only', args.lisa_jobs), ('MIT_slurm_readonly', args.mit_slurm),
                       ('MIT_private_classification_local_only', args.mit_categories)):
        if sha256(path) != inputs[role]:
            raise ValueError(f'{role} 与资源输入的来源哈希不同')
    sources = (
        (seren_jobs(args.jobs), ('B-TRN', 'C-TRN', 'D-TRN'), [audit['job_sha256']]),
        (lisa_jobs(args.lisa_jobs), ('A-HPC',), [inputs['HPC_jobs_local_only']]),
        (mit_jobs(args.mit_slurm, args.mit_categories), ('B-SAI', 'C-SAI', 'D-SAI'),
         [inputs['MIT_slurm_readonly'], inputs['MIT_private_classification_local_only']]),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for jobs, scenarios, hashes in sources:
        for field in ('start_time', 'end_time'):
            jobs[field] = pd.to_datetime(jobs[field], format='mixed', utc=True)
        if jobs[['start_time', 'end_time']].isna().any().any():
            raise ValueError(f'{scenarios[0]} 作业时段无效')
        jobs = jobs.loc[jobs.end_time.gt(jobs.start_time)]
        for scenario in scenarios:
            source = pd.read_parquet(args.reference / f'{scenario}-10MW.parquet')
            layers = pd.read_parquet(args.layers / f'{scenario}.parquet')
            np.testing.assert_array_equal(source.source_hour_index, layers.source_hour_index)
            power = layers['可推迟·整体'].to_numpy()
            expected = expected_units(source, args.resources, scenario)
            frame, details = whole_profile(jobs, source, power, expected)
            path = args.output / f'{scenario}.parquet'
            frame.to_parquet(path, index=False)
            manifest[scenario] = {**details, 'input_sha256': hashes, 'output_sha256': sha256(path)}
            print(scenario, details, flush=True)
    atomic_json(args.output / 'whole-job-manifest.json', manifest)


if __name__ == '__main__':
    main()
