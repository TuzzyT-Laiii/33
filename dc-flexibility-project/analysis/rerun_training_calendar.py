import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pandas as pd

import flex_econ_copt as flex


def identity(row):
    alpha = 'economic' if row['alpha'] is None or pd.isna(row['alpha']) else str(float(row['alpha']))
    return '|'.join(str(row[key]) for key in ('city', 'scenario', 'flex', 'card_price', 'supply')) + '|' + alpha


def calculate(task):
    project, out, row, assets = task
    project, out = Path(project), Path(out)
    p, own = flex.load_config(project)
    input_path = Path('inputs') / (row['city'] + '-' + row['scenario'] + '.parquet')
    frame = pd.read_parquet(out / input_path)
    frame.attrs['scenario'] = row['scenario']
    noflex = row['flex'] == 'none'
    price = row['card_price']
    multiplier = 0. if price in ('forced0', 'noflex') else float(price)
    spec = flex.card_spec(p, own, frame, row['scenario'], multiplier)
    if price in ('forced0', 'noflex'):
        spec['extra_max_mw'] = 0.
    elif assets is not None:
        spec['extra_max_mw'] *= 2
    whole_h, split_h = (0, 0) if noflex else (own['flex_cases']['both']['whole_h'], own['flex_cases']['both']['split_h'])
    if noflex and assets is not None:
        # 将规划最优资产固定后，原最优解仍可行且达到同一下界，可直接复用。
        planning = pd.read_csv(out / 'runs-cards.csv', dtype={'card_price': str})
        selected = planning.city.eq(row['city']) & planning.scenario.eq(row['scenario']) & planning.flex.eq('none')
        selected &= planning.alpha.isna() if row['alpha'] is None else planning.alpha.eq(row['alpha'])
        baseline = planning.loc[selected]
        if len(baseline) != 1 or baseline.iloc[0].status != 'optimal':
            raise ValueError('固定资产基线缺少唯一的规划最优解')
        summary = baseline.iloc[0].to_dict()
        if assets != [summary[key] for key in ('pv_mw', 'wind_mw', 'battery_mw', 'battery_mwh', 'grid_cap_mw')]:
            raise ValueError('固定资产与规划最优资产不完全相同')
        source_hourly = out / summary['hourly_path']
        hourly = pd.read_parquet(source_hourly)
        summary['fixed_baseline_reused_from_planning'] = True
        summary['planning_baseline_hourly_sha256'] = flex.digest(source_hourly)
        summary['fixed_asset_match_max_abs'] = 0.
    else:
        summary, hourly = flex.solve(frame, p, row['city'], alpha=row['alpha'], design='opt', whole_h=whole_h,
                                     split_h=split_h, cards=spec, fixed_assets=assets, keep_hourly=True)
    summary.update(row)
    summary.update(idle_per_mw=spec['idle_per_mw'], baseline_fleet_usd_year=spec['baseline_fleet_usd_year'],
                   servers_per_mw=spec['servers_per_mw'], input_path=input_path.as_posix(), input_sha256=flex.digest(out / input_path))
    name = 'card_market_usd_per_mw_year' if assets is not None else 'card_usd_per_mw_year'
    summary[name] = flex.card_spec(p, own, frame, row['scenario'], 1.)['usd_per_mw_year'] if assets is not None else spec['usd_per_mw_year']
    if hourly is not None:
        filename = identity(row).replace('|', '_') + '.parquet'
        target = Path('hourly') / filename
        hourly.to_parquet(out / target, index=False)
        summary['hourly_path'] = target.as_posix()
        summary['hourly_sha256'] = flex.digest(out / target)
    return summary


def write_results(out, name, rows):
    table = pd.DataFrame(rows).sort_values(['city', 'scenario', 'alpha', 'flex', 'card_price'], na_position='first')
    pending = out / (name + '.pending.csv')
    table.to_csv(pending, index=False)
    pending.replace(out / name)
    tables = [pd.read_csv(out / file, dtype={'card_price': str}) for file in ('runs-cards.csv', 'runs-fixed.csv') if (out / file).exists()]
    pd.concat(tables, ignore_index=True).to_csv(out / 'hourly-index.csv', index=False)


def execute(tasks, out, name, workers):
    rows = pd.read_csv(out / name, dtype={'card_price': str}).to_dict('records') if (out / name).exists() else []
    failed = [identity(row) for row in rows if row['status'] != 'optimal']
    if failed:
        print('retry non-optimal rows', failed, flush=True)
        rows = [row for row in rows if row['status'] == 'optimal']
    completed = {identity(row) for row in rows}
    tasks = [task for task in tasks if identity(task[2]) not in completed]
    print(name, 'pending', len(tasks), 'completed', len(rows), flush=True)
    last = time.monotonic()
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(calculate, task) for task in tasks]
        for k, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            if row['status'] != 'optimal':
                write_results(out, name, rows)
                raise RuntimeError('未取得最优解：' + identity(row) + ' status=' + row['status'])
            if time.monotonic() - last >= 20 or k == len(futures):
                write_results(out, name, rows)
                last = time.monotonic()
                (out / 'progress.json').write_text(json.dumps({'stage': name, 'done': len(rows), 'remaining': len(tasks) - k,
                    'utc': datetime.now(timezone.utc).isoformat()}), encoding='utf-8')
                print(name, 'finished', len(rows), 'pending', len(tasks) - k, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--stage', choices=['smoke', 'cards', 'fixed', 'all'], default='all')
    args = parser.parse_args()
    out = args.output
    (out / 'hourly').mkdir(parents=True, exist_ok=True)
    fingerprints = {path.name: flex.digest(path) for path in [Path(__file__), args.project / 'analysis/flex_econ_copt.py',
        args.project / 'analysis/flex-econ-copt-20261102.json', args.project / 'analysis/eight-city-flex-value-20261029.json']}
    manifest = out / 'run-inputs.json'
    if manifest.exists():
        if json.loads(manifest.read_text('utf-8'))['code_and_parameters_sha256'] != fingerprints:
            raise ValueError('续跑源码或参数指纹改变，请使用独立输出目录')
    else:
        manifest.write_text(json.dumps({'code_and_parameters_sha256': fingerprints, 'prior': str(args.prior),
            'scope': '8 cities x B/C/D-TRN; native elapsed-hour deadlines; no execution variables in missing hours',
            'utc': datetime.now(timezone.utc).isoformat()}, indent=2), encoding='utf-8')
    tasks = []
    prior_cards = pd.read_csv(args.prior / 'runs-cards.csv', dtype={'card_price': str})
    for row in prior_cards.loc[prior_cards.scenario.str.endswith('-TRN')].to_dict('records'):
        details = {key: row[key] for key in ('city', 'scenario', 'flex', 'card_price')}
        details['alpha'] = None if pd.isna(row['alpha']) else float(row['alpha'])
        details['supply'] = 'reoptimized'
        if args.stage == 'smoke' and not (details['city'] == 'frankfurt' and details['scenario'] == 'D-TRN'
                and details['alpha'] in (None, 1.) and (details['flex'] == 'none' or details['card_price'] == '1.0')):
            continue
        tasks.append((str(args.project), str(out), details, None))
    if args.stage in ('smoke', 'cards', 'all'):
        execute(tasks, out, 'runs-cards.csv', args.workers)
    if args.stage in ('fixed', 'all'):
        cards = pd.read_csv(out / 'runs-cards.csv', dtype={'card_price': str})
        bases = cards.loc[cards.flex.eq('none')]
        if len(bases) != 144 or not bases.status.eq('optimal').all():
            raise ValueError('固定供给侧须取得全部144个训练无柔性最优配置')
        tasks = []
        for row in bases.to_dict('records'):
            assets = [row[key] for key in ('pv_mw', 'wind_mw', 'battery_mw', 'battery_mwh', 'grid_cap_mw')]
            for multiplier in ('noflex', '0.0', '1.0'):
                details = {key: row[key] for key in ('city', 'scenario')}
                details.update(alpha=None if pd.isna(row['alpha']) else float(row['alpha']), flex='none' if multiplier == 'noflex' else 'both',
                               card_price=multiplier, supply='fixed_noflex_opt')
                tasks.append((str(args.project), str(out), details, assets))
        execute(tasks, out, 'runs-fixed.csv', args.workers)


if __name__ == '__main__':
    main()
