import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--independent-groups', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    columns = ['grid_id', 'scenario', 'boundary', 'alpha', 'flex', 'status', 'total_cost_usd',
               'extra_task_mw', 'grid_mwh', 'pv_mw', 'wind_mw', 'battery_mw', 'battery_mwh',
               'grid_cap_mw', 'max_residual']
    frame = pd.read_parquet(args.source, columns=columns)
    frame = frame[(frame.boundary == 'planning') & frame.alpha.isna() &
                  (frame.status == 'optimal') & frame.flex.isin(['none', 'both'])].copy()
    if len(frame) != 2 * 4842 * 11 or frame.extra_task_mw.max() > 1e-8:
        raise RuntimeError('原价基线不满足完整覆盖和零加卡条件')
    frame['mode'] = frame.flex.map({'none': 'baseline', 'both': 'coordination'})
    if frame.duplicated(['grid_id', 'scenario', 'mode']).any():
        raise RuntimeError('原价基线配对键重复')
    groups = [json.loads(line) for line in args.independent_groups.read_text().splitlines()]
    independent = pd.DataFrame([row for group in groups for row in group['rows']
                                if row['mode'] != 'sell_all' and row['multiplier'] == 1.])
    paired = independent.merge(frame, on=['grid_id', 'scenario', 'mode'], validate='one_to_one')
    if len(paired) != len(independent) or paired.empty:
        raise RuntimeError('独立重算结果未与原价基线完整配对')
    differences = np.abs(paired.total_cost_usd_x - paired.total_cost_usd_y)
    if differences.max() > .02:
        raise RuntimeError(f'原价基线与独立重算费用不一致: {differences.max()}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    audit = dict(source_sha256=sha256(args.source.read_bytes()).hexdigest(),
                 independent_groups_sha256=sha256(args.independent_groups.read_bytes()).hexdigest(),
                 reference_rows=len(frame), independent_pairs=len(paired),
                 max_difference_usd=float(differences.max()),
                 maximum_extra_capacity_mw=float(frame.extra_task_mw.max()))
    args.output.with_suffix('.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
