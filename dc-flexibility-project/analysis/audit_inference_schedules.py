import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

import resource_inference_whole as whole
import resource_inference_capacity as hourly


def inspect(schedule, fleet, state, requests, row, assignment, execution, scale):
    if assignment.shape != (schedule.option_count,) or execution.shape != schedule.hours.shape:
        raise ValueError('服务方案数组形状与原始任务不一致')
    if not np.isfinite(np.r_[assignment, execution, scale]).all():
        raise ValueError('服务方案包含非有限值')
    counts = np.bincount(schedule.blocks, weights=assignment, minlength=schedule.block_count)
    original = schedule.original / fleet['baseline_units']
    dispatch = schedule.fixed.copy() / fleet['baseline_units']
    np.add.at(dispatch, schedule.slots,
              schedule.values * assignment[schedule.columns, None] / fleet['baseline_units'])
    positions = schedule.hours - schedule.hours[0] - row['phase_hours']
    mask = (positions >= 0) & (positions < len(requests))
    shape = np.zeros(len(positions))
    shape[mask] = requests[positions[mask].astype(int)]
    idle = float((1. - original[mask, 1]).sum())
    arrivals = shape / shape.sum() * idle * scale
    offline = arrivals * row['offline_share']
    online = arrivals - offline
    backlog = np.cumsum(offline - execution)
    next_hours = np.r_[schedule.hours[1:], schedule.hours[-1] + 1]
    cumulative = np.r_[0., np.cumsum(offline)]
    due_before_next = np.searchsorted(schedule.hours, next_hours - 24, side='left')
    allowance = cumulative[1:] - cumulative[due_before_next]
    allowance[-1] = 0.
    occupied = dispatch + (online + execution)[:, None] * np.array([state['host_ratio'], 1.])
    residuals = dict(
        block_workload=float(np.abs(counts - schedule.block_sizes).max()),
        whole_integrality=float(np.abs(assignment[schedule.integer] - np.rint(assignment[schedule.integer])).max()),
        nonnegative=float(max(0., -assignment.min(), -execution.min(), -scale)),
        option_upper=float(np.maximum(assignment - schedule.block_sizes[schedule.blocks], 0.).max()),
        own_resource_conservation=float(np.abs(dispatch.sum(axis=0) - original.sum(axis=0)).max()),
        resource_capacity=float(np.maximum(occupied - 1., 0.).max()),
        external_early_execution=float(np.maximum(-backlog, 0.).max()),
        external_deadline=float(np.maximum(backlog - allowance, 0.).max()),
        external_completion=float(abs(backlog[-1])))
    if max(residuals.values()) > 1e-6:
        raise ValueError(f'服务方案独立核验失败 {row["scenario"]} {row["phase_hours"]} {residuals}')
    return residuals


def main():
    parser = argparse.ArgumentParser()
    for name in ['project', 'joint', 'capacity', 'residual-code', 'requests', 'prior', 'input', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    whole.prepare(args)
    from joint_capacity_copt import make_resource_schedule

    expected = [(s, phase, share) for s in whole.CONTEXT['manifest']
                if s.endswith(('SAI', 'TRN')) for phase in [0, 24, 168] for share in [0., .28]]
    if len(expected) != 36 or not (args.input / 'gpu-service-capacity.csv').is_file():
        raise ValueError('完整作业服务计算尚未完成')
    args.output.mkdir(parents=True, exist_ok=True)
    records, solutions, audits = {}, {}, []
    for scenario in dict.fromkeys(key[0] for key in expected):
        fleet = whole.CONTEXT['manifest'][scenario]
        state = hourly.CONTEXT['scenarios'][scenario]
        schedule = make_resource_schedule(whole.CONTEXT['model_args'], scenario, fleet)
        np.testing.assert_array_equal(schedule.hours, state['hours'])
        for key in [item for item in expected if item[0] == scenario]:
            _, phase, share = key
            path = args.input / f'{scenario}-{phase}-{share}.json'
            row = json.loads(path.read_text())
            if (row['scenario'], row['phase_hours'], row['offline_share']) != key:
                raise ValueError('完整作业服务键不一致')
            with np.load(path.with_suffix('.npz')) as saved:
                assignment = saved['assignment']
                execution = saved['external_offline']
                scale = float(saved['external_scale'])
            if abs(scale - row['feasible_share_of_idle']) > 1e-6:
                raise ValueError('保存的服务方案与可行服务量不一致')
            residuals = inspect(schedule, fleet, state, hourly.CONTEXT['requests'],
                                row, assignment, execution, scale)
            records[key] = row.copy()
            solutions[key] = (assignment, execution, scale)
            audits.append(dict(scenario=scenario, phase_hours=phase, offline_share=share,
                               source_sha256=sha256(path.read_bytes()).hexdigest(),
                               solution_sha256=sha256(path.with_suffix('.npz').read_bytes()).hexdigest(),
                               residuals=residuals))
        for phase in [0, 24, 168]:
            key = (scenario, phase, 0.)
            row = records[key]
            assignment, _, offline_scale = solutions[(scenario, phase, .28)]
            scale = offline_scale * .72
            if scale > row['feasible_share_of_idle'] + 1e-9:
                execution = np.zeros(len(schedule.hours))
                residuals = inspect(schedule, fleet, state, hourly.CONTEXT['requests'],
                                    row, assignment, execution, scale)
                if scale > row['upper_share_of_idle'] + 1e-6:
                    raise ValueError('构造服务方案超过已核验最优性界限')
                row['feasible_share_of_idle'] = scale
                row['upper_share_of_idle'] = max(scale, row['upper_share_of_idle'])
                row['capacity_proven'] = bool(row['upper_share_of_idle'] - scale < 1e-6)
                row['construction'] = 'online component of verified 28% offline schedule'
                row['max_residual'] = max(row['max_residual'], max(residuals.values()))
                solutions[key] = (assignment, execution, scale)
                audits.append(dict(scenario=scenario, phase_hours=phase, offline_share=0.,
                                   constructed_from_offline_share=.28, residuals=residuals))
    result = pd.read_csv(args.input / 'gpu-service-capacity.csv')
    for key, row in records.items():
        scenario, phase, share = key
        mask = (result.scenario == scenario) & (result.phase_hours == phase) & (result.offline_share == share)
        if mask.sum() != 3:
            raise ValueError('吞吐量敏感性结果配对不完整')
        ratio = row['feasible_share_of_idle'] / result.loc[mask, 'feasible_share_of_idle']
        for field in ['feasible_standard_requests', 'feasible_trace_multiplier']:
            result.loc[mask, field] *= ratio
        result.loc[mask, 'upper_standard_requests'] *= row['upper_share_of_idle'] / result.loc[mask, 'upper_share_of_idle']
        for field in ['feasible_share_of_idle', 'upper_share_of_idle', 'capacity_proven']:
            result.loc[mask, field] = row[field]
        result.loc[mask, 'max_residual'] = np.maximum(result.loc[mask, 'max_residual'], row['max_residual'])
        path = args.output / f'{scenario}-{phase}-{share}.json'
        path.write_text(json.dumps(row, indent=2, allow_nan=False))
        assignment, execution, scale = solutions[key]
        np.savez_compressed(path.with_suffix('.npz'), assignment=assignment, external_offline=execution, external_scale=scale)
    destination = args.output / 'gpu-service-capacity.csv'
    result.to_csv(destination, index=False)
    maximum = max(max(item['residuals'].values()) for item in audits)
    report = dict(whole_groups=36, constructed_groups=len(audits) - 36,
                  proven_rows=int(result.capacity_proven.sum()), total_rows=len(result),
                  max_independent_residual=maximum, cases=audits,
                  result_sha256=sha256(destination.read_bytes()).hexdigest())
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}))


if __name__ == '__main__':
    main()
