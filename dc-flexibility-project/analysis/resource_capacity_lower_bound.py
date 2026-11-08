import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from flex_econ_copt import SCENARIOS, backlog_limit
from joint_capacity_copt import resource_layers
from joint_capacity_inputs import atomic_json
from minimum_server_capacity import DEADLINES


def solve(hours, fixed, whole, split, whole_h, split_h):
    started = time.monotonic()
    n = len(hours)
    active = np.flatnonzero((fixed + whole + split).sum(axis=0) > 0)
    count = 1 + len(active) * 4 * n
    lower, upper = np.zeros(count), np.full(count, np.inf)
    objective = np.zeros(count)
    objective[0] = 1
    rows, columns, values, lhs, rhs = [], [], [], [], []
    row = 0
    for domain, resource in enumerate(active):
        starts = [1 + (domain * 4 + k) * n for k in range(4)]
        xw, bw, xs, bs = [start + np.arange(n) for start in starts]
        for execution, backlog, arrival, deadline in ((xw, bw, whole[:, resource], whole_h),
                                                       (xs, bs, split[:, resource], split_h)):
            upper[backlog] = backlog_limit(arrival, deadline, hours)
            r = row + np.arange(n)
            rows.extend((r, r, r[1:]))
            columns.extend((execution, backlog, backlog[:-1]))
            values.extend((np.ones(n), np.ones(n), -np.ones(n - 1)))
            lhs.append(arrival)
            rhs.append(arrival)
            row += n
        r = row + np.arange(n)
        rows.extend((r, r, r))
        columns.extend((xw, xs, np.zeros(n, dtype=int)))
        values.extend((np.ones(n), np.ones(n), -np.ones(n)))
        lhs.append(np.full(n, -np.inf))
        rhs.append(-fixed[:, resource])
        row += n
    matrix = sparse.csc_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                               shape=(row, count))
    left, right = np.concatenate(lhs), np.concatenate(rhs)
    equality = np.isfinite(left) & (left == right)
    result = linprog(objective, A_ub=matrix[~equality], b_ub=right[~equality],
                     A_eq=matrix[equality], b_eq=right[equality],
                     bounds=np.column_stack((lower, upper)), method='highs',
                     options={'primal_feasibility_tolerance': 1e-9,
                              'dual_feasibility_tolerance': 1e-9, 'time_limit': 300, 'threads': 1})
    if not result.success:
        raise RuntimeError(f'独立资源松弛未求至最优 {result.message}')
    return float(result.fun), time.monotonic() - started


def main():
    parser = argparse.ArgumentParser()
    for name in ('reference', 'resources', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.resources / 'resource-manifest.json').read_text('utf-8'))
    records = []
    for scenario in SCENARIOS:
        data = resource_layers(args, scenario, manifest[scenario])
        for whole_h, split_h in DEADLINES:
            lower, elapsed = solve(*data, whole_h, split_h)
            row = {'scenario': scenario, 'whole_h': whole_h, 'split_h': split_h,
                   'resource_lower_servers': lower, 'integer_lower_servers': int(np.ceil(lower - 1e-5)),
                   'resource_lower_over_A': lower / manifest[scenario]['baseline_units'], 'seconds': elapsed}
            records.append(row)
            atomic_json(args.output / f'{scenario}-w{whole_h}-s{split_h}.json', row)
            print(row, flush=True)
    pd.DataFrame(records).to_csv(args.output / 'resource-lower-bounds.csv', index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
