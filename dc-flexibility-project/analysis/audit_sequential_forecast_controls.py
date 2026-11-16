import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np

import rolling_forecast_validation as rolling

CONTEXT = {}


def verify_one(task):
    import coptpy as cp
    from residual_flex_pilot import solve
    name, forecast = task
    started = time.monotonic()
    root = CONTEXT['root']
    reference_path = root / 'forecast' / name
    reference = json.loads(reference_path.read_text())
    scenario, city = reference['scenario'], reference['city']
    g = rolling.CONTEXT['g']
    grid = rolling.CONTEXT['selected'][city]
    frame, on_raw, off_raw, _, _ = g.align(g.build_frame(grid, scenario), scenario)
    own_task = frame[['rigid', 'unknown', 'whole', 'split']].sum(axis=1).to_numpy()
    intensity = reference['intensity_of_max'] * g.CONTEXT['max_intensity'][scenario]
    factor = intensity * (own_task.max() - own_task.mean()) / (on_raw.mean() + off_raw.mean())
    frame['on'], frame['off'] = factor * on_raw, factor * off_raw
    frame.attrs['capacity'] = float(own_task.max())
    np.testing.assert_allclose(frame[list(rolling.CLASSES)].sum(), reference['flex_arrival_mwh'], rtol=0, atol=1e-8)
    e0 = float(np.dot(frame.pue, frame.base + own_task))
    if grid != reference['grid_id'] or len(frame) != reference['hours'] or abs(e0 - reference['E0_mwh']) > 1e-8:
        raise ValueError(f'顺序对照输入不一致 {name}')
    p = {**g.CONTEXT['parameters'], 'emission_factor': {grid: g.CONTEXT['meta'][grid]['carbon_t_mwh']}}
    p['ef'] = p['emission_factor'][grid]
    first, schedule = solve(frame, p, grid, frame.on.to_numpy(), frame.off.to_numpy(), fee=120., mode='asap', solver='copt')
    if first['status'] != 'optimal' or schedule is None:
        raise ValueError(f'顺序对照任务方案不可行 {name}')
    assets, _ = solve(frame, p, grid, frame.on.to_numpy(), frame.off.to_numpy(), fee=120., mode='cost',
                      fixed_x=schedule, solver='copt')
    if assets['status'] != 'optimal':
        raise ValueError(f'顺序对照供能方案不可行 {name}')
    env = cp.Envr()
    result = rolling.replay(frame, p, assets, 'seq', forecast == 'perfect', env)
    env.dispose()
    previous = next(row for row in reference['cases'] if row['mode'] == 'seq' and row['forecast'] == forecast)
    difference = abs(result['cost_usd'] - previous['cost_usd'])
    if difference > .02 or sum(result['deadline_missed_mwh']) > 1e-5 or sum(result['unserved_mwh']) > 1e-5:
        raise ValueError(f'顺序对照重放未通过 {name} {forecast} {difference}')
    record = dict(name=name, forecast=forecast, reference_sha256=sha256(reference_path.read_bytes()).hexdigest(),
                  terminal_result_sha256=sha256((root / 'forecast-terminal-verified' / name).read_bytes()).hexdigest(),
                  difference_usd=difference, case=result, seconds=time.monotonic() - started)
    destination = CONTEXT['output'] / f'{Path(name).stem}-{forecast}.json'
    destination.write_text(json.dumps(record, indent=2, allow_nan=False))
    print(name, forecast, difference, record['seconds'], flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    for name in ('project', 'config', 'asi', 'max-table', 'inputs', 'residual-code', 'root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=49)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rolling.prepare(args)
    CONTEXT.update(root=args.root, output=args.output)
    tasks, differences = [], []
    for path in sorted((args.root / 'forecast-terminal-verified').glob('*.json')):
        current = json.loads(path.read_text())
        original = json.loads((args.root / 'forecast' / path.name).read_text())
        for case in current['cases']:
            if case['mode'] != 'seq':
                continue
            previous = next(row for row in original['cases'] if row['mode'] == 'seq' and row['forecast'] == case['forecast'])
            difference = abs(case['cost_usd'] - previous['cost_usd'])
            differences.append(difference)
            if difference > .02:
                tasks.append((path.name, case['forecast']))
    if len(differences) != 352:
        raise ValueError('顺序对照覆盖不完整')
    with ProcessPoolExecutor(args.workers) as pool:
        records = list(pool.map(verify_one, tasks))
    audit = dict(total_controls=352, fresh_replays=len(records),
                 max_difference_usd=max((record['difference_usd'] for record in records), default=0.),
                 max_residual=max((record['case']['max_residual'] for record in records), default=0.),
                 case_hashes={path.name: sha256(path.read_bytes()).hexdigest()
                              for path in sorted(args.output.glob('*.json')) if path.name != 'audit.json'})
    (args.output / 'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in audit.items() if key != 'case_hashes'}), flush=True)


if __name__ == '__main__':
    main()
