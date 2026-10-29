import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

from eight_city_onsite_solver import sha256, solve_onsite, storage_bounds
from run_eight_city_flex_value import DEFAULT_CONFIG, PROJECT, load_input, write_json


def validate_group(arguments):
    config_path, directory, output, city, scenario, storage = arguments
    config = json.loads(config_path.read_text(encoding='utf-8'))
    loaded = load_input(directory, city, scenario, config)
    if loaded is None:
        raise FileNotFoundError(f'全年物理核验输入未就绪：{city}/{scenario}')
    frame, idle, design, provenance = loaded
    bounds = storage_bounds(frame, config, storage)
    reference = config['sweep']['physical_reference']
    assets = {key: reference[key] for key in ('pv_mw', 'wind_mw')}
    assets.update(battery_power_mw=bounds[0][0], battery_energy_mwh=bounds[1][0])
    summaries = []
    for delay in (0, 168):
        result = solve_onsite(frame, config, deadline=delay, storage=storage, idle_mw=idle, design_mw=design,
            emission_factor=config['cities'][city]['emission_factor'], fixed_assets=assets, physical=True)
        if result['summary']['status'] != 'optimal':
            raise AssertionError(result['summary'])
        summary = {**result['summary'], 'city': city, 'scenario': scenario, 'input_provenance': provenance,
                   'input_sha256': provenance['sha256'], 'config_sha256': sha256(config_path),
                   'physical_reference': reference}
        folder = output / 'physical-validation-cases' / city / scenario / f'S{storage}-D{delay}'
        write_json(folder / 'summary.json', summary)
        result['hourly'].to_parquet(folder / 'hourly.parquet', index=False, compression='zstd')
        summaries.append(summary)
    if summaries[1]['grid_mwh'] > summaries[0]['grid_mwh'] + .05:
        raise AssertionError('D增加后固定装机最小购电量增加')
    return {'scenario': scenario, 'S': storage, 'status': 'passed',
        'grid_reduction_mwh': summaries[0]['grid_mwh'] - summaries[1]['grid_mwh'],
        'cfe_hourly_gain_pp': 100 * (summaries[1]['local_cfe_hourly_mean'] - summaries[0]['local_cfe_hourly_mean']),
        'maximum_conservation_residual': max(max(item['diagnostics'].values()) for item in summaries)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--city', default='phoenix')
    parser.add_argument('--scenario', nargs='+', default=['A-HPC', 'D-TRN'])
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--details-dir', type=Path, default=PROJECT / 'data/processed/eight-city-flex-value-20261029')
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'results/eight-city-flex-value-20261029')
    args = parser.parse_args()
    tasks = [(args.config, args.input_dir, args.details_dir, args.city, scenario, storage)
             for scenario in args.scenario for storage in (0, 4)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        checks = list(pool.map(validate_group, tasks))
    report = {'status': 'passed', 'utc': datetime.now(timezone.utc).isoformat(), 'real_annual_cases': len(tasks) * 2,
              'config_sha256': sha256(args.config), 'solver_sha256': sha256(Path(__file__).with_name('eight_city_onsite_solver.py')),
              'checks': checks}
    write_json(args.output_dir / 'physical-validation.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
