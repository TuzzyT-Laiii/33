import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eight_city_onsite_solver import sha256
from run_eight_city_flex_value import DEFAULT_CONFIG, PROJECT, write_json


GROUP = ['city', 'scenario', 'storage', 'alpha_key', 'carbon_price', 'phi', 'rmax', 'input_sha256']


def record_key(item):
    storage = item['storage']
    storage_key = json.dumps(storage, sort_keys=True) if isinstance(storage, dict) else ('opt' if storage == 'opt' else float(storage))
    return (item['city'], item['scenario'], storage_key,
            None if item['alpha'] is None else float(item['alpha']),
            float(item['carbon_price']), float(item['phi']), float(item['rmax']), int(item['D']),
            item['input_provenance']['sha256'])


def summary_paths(base):
    return sorted(set(base.glob('*/*/summary.json')) | set(base.glob('*/*/*/summary.json')))


def assemble(records):
    rows = []
    for item in records:
        row = {key: val for key, val in item.items() if not isinstance(val, (dict, list))}
        row.setdefault('infeasibility_reason', None)
        row['alpha_key'] = 'price_only' if item['alpha'] is None else str(item['alpha'])
        row['storage'] = str(item['storage'])
        row.update({f'cost_{key}_usd': amount for key, amount in item.get('cost_components_usd', {}).items()})
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    for _, part in frame.groupby(GROUP, dropna=False):
        part = part.sort_values('D')
        base = part.loc[part.D.eq(0)]
        if len(base) != 1 or base.iloc[0].status != 'optimal':
            frame.loc[part.index, 'comparison_status'] = 'fixed_reference_unavailable'
            frame.loc[part.loc[part.status.eq('optimal')].index, 'comparison_status'] = 'flexibility_enables_feasibility'
            continue
        fixed = base.iloc[0]
        good = part.loc[part.status.eq('optimal')]
        frame.loc[good.index, 'comparison_status'] = 'paired_optimal'
        frame.loc[good.index, 'fixed_cost_usd'] = fixed.annual_cost_usd
        frame.loc[good.index, 'value_usd_year'] = fixed.annual_cost_usd - good.annual_cost_usd
        frame.loc[good.index, 'value_usd_mwh_it'] = (fixed.annual_cost_usd - good.annual_cost_usd) / fixed.baseline_it_mwh
        frame.loc[good.index, 'value_percent_cost'] = 100 * (fixed.annual_cost_usd - good.annual_cost_usd) / fixed.annual_cost_usd
        frame.loc[good.index, 'grid_reduction_mwh'] = fixed.grid_mwh - good.grid_mwh
        frame.loc[good.index, 'curtailment_reduction_mwh'] = fixed.curtailment_mwh - good.curtailment_mwh
        frame.loc[good.index, 'cfe_energy_gain_pp'] = 100 * (good.local_cfe_energy - fixed.local_cfe_energy)
        frame.loc[good.index, 'cfe_hourly_gain_pp'] = 100 * (good.local_cfe_hourly_mean - fixed.local_cfe_hourly_mean)
        frame.loc[good.index, 'battery_power_reduction_mw'] = fixed.battery_power_mw - good.battery_power_mw
        frame.loc[good.index, 'battery_energy_reduction_mwh'] = fixed.battery_energy_mwh - good.battery_energy_mwh
        moved = good.moved_dynamic_it_mwh.where(good.moved_dynamic_it_mwh.gt(1e-6))
        frame.loc[good.index, 'value_per_moved_mwh'] = (fixed.annual_cost_usd - good.annual_cost_usd) / moved
        for previous, current in zip(good.itertuples(), list(good.itertuples())[1:]):
            expected = [0, 1, 2, 4, 6, 12, 24, 48, 72, 168]
            if expected.index(current.D) == expected.index(previous.D) + 1:
                frame.loc[current.Index, 'marginal_value_usd_year_h'] = (previous.annual_cost_usd - current.annual_cost_usd) / (current.D - previous.D)
        frame.loc[part.loc[~part.status.eq('optimal')].index, 'comparison_status'] = 'flexible_solution_unavailable'
    if 'value_usd_year' in frame:
        keys = ['city', 'scenario', 'alpha_key', 'carbon_price', 'phi', 'rmax', 'D', 'input_sha256']
        zero = frame.loc[frame.storage.eq('0'), keys + ['value_usd_year']].rename(columns={'value_usd_year':'value_S0'})
        frame = frame.merge(zero, on=keys, how='left', validate='many_to_one')
        frame['storage_interaction_usd_year'] = frame.value_usd_year - frame.value_S0
    return frame


def thresholds(frame, expected_D):
    rows = []
    if frame.empty:
        return pd.DataFrame()
    for group, part in frame.groupby(GROUP, dropna=False):
        row = dict(zip(GROUP, group))
        good = part.loc[part.status.eq('optimal')].sort_values('D')
        row['observed_D_count'] = len(good)
        terminal = good.loc[good.D.eq(168)]
        fixed = part.loc[part.D.eq(0)]
        if len(fixed) == 1 and fixed.iloc[0].status == 'infeasible':
            row['threshold_status'] = 'fixed_reference_infeasible'
        elif len(good) != len(expected_D) or terminal.empty:
            row['threshold_status'] = 'incomplete_curve'
        elif terminal.iloc[0].value_usd_year <= .2:
            row['threshold_status'] = 'no_positive_terminal_value'
        else:
            terminal_value = terminal.iloc[0].value_usd_year
            row.update(threshold_status='complete_discrete_grid', V168_usd_year=terminal_value)
            for target in (.5, .9):
                row[f'D{int(target*100)}_h'] = int(good.loc[good.value_usd_year.ge(target * terminal_value), 'D'].min())
        rows.append(row)
    return pd.DataFrame(rows)


def figures(frame, threshold, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 300, 'legend.frameon': False})
    if frame.empty or 'value_usd_year' not in frame:
        return []
    sample = frame.loc[frame.status.eq('optimal') & frame.alpha_key.eq('0.95') & frame.phi.eq(1) & frame.carbon_price.eq(100)]
    if sample.empty:
        return []
    city = sample.city.iloc[0]
    sample = sample.loc[sample.city.eq(city)]
    colors = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout='constrained')
    curve_sample = sample.loc[sample.storage.eq('0') & sample.value_usd_year.notna()]
    if curve_sample.empty:
        curve_sample = frame.loc[frame.city.eq(city) & frame.storage.eq('0') & frame.alpha_key.eq('price_only') & frame.phi.eq(1) & frame.carbon_price.eq(100) & frame.rmax.eq(2) & frame.value_usd_year.notna()]
    for number, (scenario, part) in enumerate(curve_sample.groupby('scenario')):
        part = part.sort_values('D')
        ax.plot(part.D, part.value_usd_year / 1e6, marker='o', ms=3, label=scenario, color=colors(number % 10))
    ax.set(xlabel='Maximum additional delay (h)', ylabel='Net flexibility value (million USD/year)',
           title=f'{city}, 10 MW, 2024 (Feb 29 removed), S=0; {curve_sample.alpha_key.iloc[0] if len(curve_sample) else "reference unavailable"}')
    ax.axhline(0, color='0.6', lw=.6)
    if len(curve_sample):
        ax.legend(ncol=3, fontsize=8)
    export(fig, output / 'value-vs-delay')
    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout='constrained')
    valid = threshold.loc[threshold.city.eq(city) & threshold.storage.eq('0') & threshold.alpha_key.eq('0.95') & threshold.phi.eq(1) & threshold.carbon_price.eq(100)]
    valid = valid.loc[valid.threshold_status.eq('complete_discrete_grid')]
    threshold_label = 'alpha=95%'
    if valid.empty:
        valid = threshold.loc[threshold.city.eq(city) & threshold.storage.eq('0') & threshold.alpha_key.eq('price_only') & threshold.phi.eq(1) & threshold.carbon_price.eq(100) & threshold.threshold_status.eq('complete_discrete_grid')]
        threshold_label = 'price only'
    if len(valid):
        ax.scatter(valid.scenario, valid.D50_h, color='#277DA1', label='D50')
        ax.scatter(valid.scenario, valid.D90_h, color='#F8961E', marker='s', label='D90')
        ax.tick_params(axis='x', rotation=40)
        ax.legend()
    else:
        ax.text(.5, .5, 'No complete positive-value curves', transform=ax.transAxes, ha='center')
    ax.set(ylabel='Delay on the tested grid (h)', title=f'{city}, S=0, {threshold_label}')
    export(fig, output / 'delay-thresholds')
    scenario = sample.scenario.iloc[0]
    matrix = sample.loc[sample.scenario.eq(scenario)].pivot(index='storage', columns='D', values='value_usd_year')
    matrix = matrix.reindex(index=['0', '1', '2', '4', '8', 'opt'], columns=[0, 1, 2, 4, 6, 12, 24, 48, 72, 168]) / 1e6
    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout='constrained')
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#E8E8E8')
    plot = ax.imshow(np.ma.masked_invalid(matrix.to_numpy()), cmap=cmap, aspect='auto', vmin=0)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(6), ['0', '1 h', '2 h', '4 h', '8 h', 'Optimized'])
    ax.set(xlabel='Maximum additional delay (h)', ylabel='Storage at mean IT power',
           title=f'{city}, {scenario}, alpha=95%; gray = unavailable')
    fig.colorbar(plot, ax=ax, label='Net value (million USD/year)')
    export(fig, output / 'storage-delay-heatmap')
    return ['value-vs-delay', 'delay-thresholds', 'storage-delay-heatmap']


def export(fig, base):
    fig.savefig(base.with_suffix('.png'), bbox_inches='tight', pad_inches=.2)
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight', pad_inches=.2)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--details-dir', type=Path, default=PROJECT / 'data/processed/eight-city-flex-value-20261029')
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'results/eight-city-flex-value-20261029')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, physical = [], []
    for subdir, target in [('cases', records), ('annual-validation-cases', records), ('physical', physical), ('physical-validation-cases', physical)]:
        for path in summary_paths(args.details_dir / subdir):
            record = json.loads(path.read_text(encoding='utf-8'))
            if record.get('config_sha256') == sha256(args.config):
                target.append(record)
    # 全年核验与正式批次存在相同算例时只保留一次。
    unique = {}
    for item in records:
        item['input_sha256'] = item['input_provenance']['sha256']
        key = record_key(item)
        unique[key] = item
    frame = assemble(list(unique.values()))
    physical_unique = {}
    for item in physical:
        item['input_sha256'] = item['input_provenance']['sha256']
        key = record_key(item)
        physical_unique[key] = item
    physical_frame = assemble(list(physical_unique.values()))
    if not physical_frame.empty:
        economic_columns = [column for column in physical_frame if column.startswith(('value_', 'cost_', 'fixed_cost_', 'marginal_value_', 'storage_interaction_')) or column in ('annual_cost_usd', 'value_S0')]
        physical_frame = physical_frame.drop(columns=economic_columns)
    for name, data in [('main-results.csv', frame), ('physical-results.csv', physical_frame)]:
        if data.empty:
            data = pd.DataFrame(columns=['city','scenario','D','storage','alpha','status','value_usd_year'])
        data.to_csv(args.output_dir / name, index=False, encoding='utf-8-sig')
    threshold = thresholds(frame, config['sweep']['D'])
    if threshold.empty:
        threshold = pd.DataFrame(columns=GROUP + ['threshold_status','D50_h','D90_h'])
    threshold.to_csv(args.output_dir / 'delay-thresholds.csv', index=False, encoding='utf-8-sig')
    statuses = pd.DataFrame(columns=['status','infeasibility_reason','cases']) if frame.empty else frame.groupby(['status','infeasibility_reason'], dropna=False).size().rename('cases').reset_index()
    statuses.to_csv(args.output_dir / 'infeasible-summary.csv', index=False, encoding='utf-8-sig')
    plotted = figures(frame, threshold, args.output_dir)
    inputs = {item['input_provenance']['sha256']: item['input_provenance'] for item in records + physical}
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'config_sha256': sha256(args.config),
        'model_sha256': sha256(Path(__file__).with_name('eight_city_onsite_solver.py')),
        'economic_cases_completed': int(frame.status.isin(['optimal','infeasible']).sum()) if len(frame) else 0,
        'economic_cases_not_solved': int(frame.status.eq('not_solved').sum()) if len(frame) else 0,
        'economic_status_counts': frame.status.value_counts().to_dict() if len(frame) else {},
        'physical_cases_completed': len(physical_frame),
        'planned_main_economic_cases': 8 * 12 * 2 * 6 * 10,
        'planned_all_economic_cases': 8 * 12 * 2 * 4 * 6 * 6 * 10,
        'planned_physical_cases': 8 * 12 * 2 * 6 * 10,
        'generated_diagnostic_figures': plotted, 'inputs': list(inputs.values()),
        'outputs_sha256': {p.name: sha256(p) for p in args.output_dir.iterdir() if p.suffix in ('.csv','.png','.pdf')},
        'scope': '已完成且通过逐小时守恒检查的算例；空白表示尚未完成或不可行；未完成曲线不推断D50/D90'}
    write_json(args.output_dir / 'manifest.json', manifest)
    print(json.dumps({key: manifest[key] for key in ('economic_cases_completed','physical_cases_completed','generated_diagnostic_figures')}))


if __name__ == '__main__':
    main()
