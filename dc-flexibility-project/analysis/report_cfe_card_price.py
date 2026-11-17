import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SERVER_MULTIPLIERS = (0.1, 0.25, 0.5, 0.75, 1., 1.5, 2., 3.)
GPU_MULTIPLIERS = (0., 0.25, 0.5, 0.75, 1., 1.5, 2., 3.)
KEYS = ['scenario', 'city', 'target']
ADD_TOLERANCE_MW = 1e-6


def city_costs(folder):
    rows = []
    for path in sorted(folder.glob('*-*.json')):
        data = json.loads(path.read_text('utf-8'))
        if data['scenario'].startswith('A-'):
            continue
        baseline, flat, reduced = (data[name] for name in ('A', 'hourly_baseline_fleet', 'continuous_job_plan'))
        reference_unit = baseline['equipment_annual_usd'] / baseline['servers']
        avoided = baseline['servers'] - reduced['servers']
        if not (reduced['servers'] == data['minimum_servers'] == data['minimum_lower_servers']) or avoided <= 0:
            raise ValueError(f'{path.name} 的减配口径不一致')
        np.testing.assert_allclose(data['equipment_annual_saving_usd'], avoided * reference_unit, atol=.01)
        rows.append({**{key: data[key] for key in KEYS}, 'original_servers': baseline['servers'],
                     'capacity_min_servers': reduced['servers'], 'reference_annual_usd_per_server': reference_unit,
                     'rigid_energy_usd': baseline['energy_annual_usd'],
                     'flat_energy_upper_usd': flat['energy_annual_usd'],
                     'flat_energy_lower_usd': flat['total_annual_lower_bound_usd'] - flat['equipment_annual_usd'],
                     'reduced_energy_upper_usd': reduced['energy_annual_usd'],
                     'reduced_energy_lower_usd': reduced['total_annual_lower_bound_usd'] - reduced['equipment_annual_usd']})
    table = pd.DataFrame(rows)
    if len(table) != 144 or table.duplicated(KEYS).any():
        raise ValueError('八城市参考费用组合不完整')
    return table


def thresholds(costs, energy):
    base_energy = energy.loc[energy.reserve_pct.eq(0), KEYS + ['energy_relaxed_original_usd']]
    joined = costs.merge(base_energy, on=KEYS, validate='one_to_one')
    if len(joined) != 144 or (joined.energy_relaxed_original_usd > joined.flat_energy_lower_usd + 1).any():
        raise ValueError('加卡与平卡的共同供能松弛下界无效')
    avoided = joined.original_servers - joined.capacity_min_servers
    capital = avoided * joined.reference_annual_usd_per_server
    joined['crossover_flat_low'] = (joined.reduced_energy_lower_usd - joined.flat_energy_upper_usd) / capital
    joined['crossover_flat_high'] = (joined.reduced_energy_upper_usd - joined.flat_energy_lower_usd) / capital
    joined['sufficient_price_multiplier_against_added'] = (
        (joined.reduced_energy_upper_usd - joined.energy_relaxed_original_usd) /
        ((avoided + 1) * joined.reference_annual_usd_per_server)).clip(lower=0)
    joined['sufficient_price_multiplier_against_flat_or_added'] = joined[
        ['crossover_flat_high', 'sufficient_price_multiplier_against_added']].max(axis=1).clip(lower=0)
    if (joined.crossover_flat_low > joined.crossover_flat_high + 1e-6).any():
        raise ValueError('减卡与平卡的交点区间无效')
    return joined


def price_grid(table):
    rows = []
    for row in table.itertuples(index=False):
        avoided = row.original_servers - row.capacity_min_servers
        for mode, fraction, multiplier in [
                *[('whole_server', 1., m) for m in SERVER_MULTIPLIERS],
                *[('gpu_only', share, m) for share in (0.50, 0.65, 0.80) for m in GPU_MULTIPLIERS]]:
            effective = 1 - fraction + fraction * multiplier
            savings = avoided * row.reference_annual_usd_per_server * effective
            rows.append(dict(scenario=row.scenario, city=row.city, target=row.target, mode=mode,
                             variable_equipment_share=fraction, price_multiplier=multiplier,
                             effective_server_multiplier=effective,
                             reduced_saving_vs_flat_low_usd=savings + row.flat_energy_lower_usd - row.reduced_energy_upper_usd,
                             reduced_saving_vs_flat_high_usd=savings + row.flat_energy_upper_usd - row.reduced_energy_lower_usd,
                             proven_cheaper_than_flat_or_added=effective > row.sufficient_price_multiplier_against_flat_or_added))
    return pd.DataFrame(rows)


def reserve_grid(costs, energy):
    cases = energy.dropna(subset=['reserve_energy_usd']).merge(
        costs, on=KEYS + ['original_servers', 'capacity_min_servers'], validate='many_to_one')
    zero_reserve = cases.loc[cases.reserve_pct.eq(0), KEYS + ['reserve_energy_usd']].rename(
        columns={'reserve_energy_usd': 'zero_reserve_energy_usd'})
    cases = cases.merge(zero_reserve, on=KEYS, validate='many_to_one')
    cases['added_energy_usd'] = cases.reserve_energy_usd - cases.zero_reserve_energy_usd
    cases['avoided_servers'] = cases.original_servers - cases.provisioned_servers
    if (len(cases) != 144 or (cases.avoided_servers < 0).any() or (cases.added_energy_usd < -1.).any() or
            (cases.reserve_energy_usd < cases.reduced_energy_lower_usd - 1.).any()):
        raise ValueError('冗余情境的样本或台数有误')
    rows = []
    for row in cases.itertuples(index=False):
        for multiplier in SERVER_MULTIPLIERS:
            savings = row.avoided_servers * row.reference_annual_usd_per_server * multiplier
            rows.append(dict(scenario=row.scenario, city=row.city, target=row.target, reserve_pct=row.reserve_pct,
                             price_multiplier=multiplier, provisioned_servers=row.provisioned_servers,
                             avoided_servers=row.avoided_servers, equipment_saving_usd=savings,
                             reserve_supply_cost_usd=row.reserve_energy_usd,
                             incremental_energy_usd=row.added_energy_usd,
                             saving_vs_rigid_original_usd=savings + row.rigid_energy_usd - row.reserve_energy_usd,
                             saving_vs_flat_low_usd=savings + row.flat_energy_lower_usd - row.reserve_energy_usd,
                             saving_vs_flat_high_usd=savings + row.flat_energy_upper_usd - row.reserve_energy_usd))
    return pd.DataFrame(rows)


def historical_add_scan(path):
    rows = pd.read_csv(path, dtype={'card_price': str})
    rows = rows.loc[rows.flex.eq('both') & rows.design.eq('opt') & ~rows.scenario.str.startswith('A-') &
                    (rows.alpha.isna() | rows.alpha.eq(1.)) & rows.card_price.isin(['0', '0.1', '0.3', '1.0'])].copy()
    rows['target'] = np.where(rows.alpha.isna(), 'economic', 'alpha100')
    if len(rows) != 576 or not rows.status.eq('optimal').all() or rows.duplicated(KEYS + ['card_price']).any():
        raise ValueError('历史加卡扫描缺少合格算例')
    rows['added'] = rows.extra_task_mw.gt(ADD_TOLERANCE_MW)
    return rows.groupby(['target', 'card_price'], as_index=False).agg(
        cases=('scenario', 'size'), added_cases=('added', 'sum'), max_extra_task_mw=('extra_task_mw', 'max'))


def main():
    parser = argparse.ArgumentParser()
    for name in ('collected', 'historic', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    energy = pd.read_csv(args.output / 'energy-cases.csv')
    if len(energy) != 240 or energy.duplicated(KEYS + ['reserve_pct']).any():
        raise ValueError('新的供能结果未完整生成')
    if energy[['relaxed_residual', 'reserve_energy_residual']].max().max() > 1e-5:
        raise ValueError('新的供能结果超出残差阈值')
    cfe = energy.loc[energy.target.eq('alpha100'), 'reserve_grid_mwh'].dropna()
    if not np.allclose(cfe, 0., rtol=0, atol=1e-4):
        raise ValueError('100% 本地无碳情境的储备机群使用了市电')
    costs = city_costs(args.collected)
    limits = thresholds(costs, energy)
    price = price_grid(limits)
    reserve = reserve_grid(costs, energy)
    historical = historical_add_scan(args.historic)
    for name, table in [('crossover-by-city', limits), ('price-sensitivity', price),
                        ('price-reserve-sensitivity', reserve), ('historical-add-scan', historical)]:
        table.to_csv(args.output / f'{name}.csv', index=False, float_format='%.9g')


if __name__ == '__main__':
    main()
