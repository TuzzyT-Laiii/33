import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
import os
from pathlib import Path

import pandas as pd

from rerun_training_calendar import calculate, identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text('utf-8'))
    failures = {row['hourly_path'] for row in audit['cases'] if not row['passed']}
    if not failures:
        raise ValueError('没有待提高数值精度的案例')
    out = args.output
    out.mkdir(exist_ok=False)
    (out / 'hourly').mkdir()
    (out / 'inputs').symlink_to((args.source / 'inputs').resolve(), target_is_directory=True)
    index = pd.read_csv(args.source / 'hourly-index.csv', dtype={'card_price': str}, float_precision='round_trip')
    selected = index[index.hourly_path.isin(failures)]
    if len(selected) != len(failures):
        raise ValueError('失败案例与结果索引不唯一对应')
    old_tables = {}
    for name in ('runs-cards.csv', 'runs-fixed.csv'):
        frame = pd.read_csv(args.source / name, dtype={'card_price': str}, float_precision='round_trip')
        frame['hourly_path'] = frame.hourly_path.map(lambda value: os.path.relpath((args.source / value).resolve(), out.resolve()))
        frame.to_csv(out / name, index=False)
        old_tables[name] = frame.to_dict('records')
    tasks = []
    for row in selected.to_dict('records'):
        details = {name: row[name] for name in ('city', 'scenario', 'flex', 'card_price', 'supply')}
        details['alpha'] = None if pd.isna(row['alpha']) else float(row['alpha'])
        assets = [row[key] for key in ('pv_mw', 'wind_mw', 'battery_mw', 'battery_mwh', 'grid_cap_mw')] if row['supply'] == 'fixed_noflex_opt' else None
        tasks.append((str(args.project), str(out), details, assets))
    new_rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(calculate, task) for task in tasks]):
            row = future.result()
            if row['status'] != 'optimal':
                raise RuntimeError('提高精度后尚未取得最优解：' + identity(row))
            new_rows.append(row)
    replacement = {identity(row): row for row in new_rows}
    merged_tables = []
    for name, old_rows in old_tables.items():
        merged = [replacement.get(identity(row), row) for row in old_rows]
        frame = pd.DataFrame(merged)
        frame.to_csv(out / name, index=False)
        merged_tables.append(frame)
    pd.concat(merged_tables, ignore_index=True).to_csv(out / 'hourly-index.csv', index=False)
    pd.DataFrame(new_rows).to_csv(out / 'refined-index.csv', index=False)
    (out / 'precision-provenance.json').write_text(json.dumps({'source': str(args.source), 'audit_sha256': sha256(args.audit.read_bytes()).hexdigest(),
        'refined_case_count': len(new_rows), 'model_sha256': sha256((args.project / 'analysis/flex_econ_copt.py').read_bytes()).hexdigest(),
        'precision': {'FeasTol': 1e-9, 'DualTol': 1e-9, 'Crossover': 1},
        'unchanged_case_count': len(index) - len(new_rows)}, indent=2), encoding='utf-8')
    print(json.dumps({'refined': len(new_rows), 'retained': len(index) - len(new_rows)}))


if __name__ == '__main__':
    main()
