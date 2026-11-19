"""算电协同供能侧节省的分项拆分（全球 4,842 网格 × 11 类机群，结论一至四）。

节省 = 顺序调度（不做协同，任务按原时间执行）− 联合调度（算电协同），两者的风光储与市电接入
都按成本最优重新规划；逐项相减后除以基准用电 E0（同网格、同情境、推理强度 0 时顺序调度的设施用电）。
供能成本为等年值口径：风光储资本费 CRF(8%, 寿命) + 固定运维，接入容量费 120 USD/kW·年，
市电 100 USD/MWh，碳价 × Ember 2024 国家电网强度（IPCC AR5 生命周期因子）。

另做两项检验：
1. 结论二的供能侧：减配少耗的供能成本、卖推理新增负荷的边际供能成本（分项）；
2. 排放口径一致性：给自建风光储补制造—建设隐含排放（低/中/高三档），在现有最优解上事后补算，
   不重新优化，得到协同节省与协同减排的变化。

输入为服务器全量结果的本地压缩表（路径见参数），输出写入 results/supply-cost-decomposition-20261119。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/supply-cost-decomposition-20261119'
COST = ['cost_pv', 'cost_wind', 'cost_battery_power', 'cost_battery_energy', 'cost_grid_capacity',
        'cost_grid_energy', 'cost_carbon']
PHYS = ['pv_mw', 'wind_mw', 'battery_mw', 'battery_mwh', 'grid_cap_mw', 'grid_mwh', 'emissions_t',
        'facility_mwh', 'facility_peak_mw']
GROUPS = {'renewables': ['cost_pv', 'cost_wind'], 'storage': ['cost_battery_power', 'cost_battery_energy'],
          'grid_capacity': ['cost_grid_capacity'], 'grid_energy': ['cost_grid_energy'], 'carbon': ['cost_carbon']}
KEYS = ['grid_id', 'scenario']
# 隐含排放：光伏、风电 t CO2e/MW，储能 t CO2e/MWh；寿命与成本年化一致（25/25/15 年）
EMBODIED = {'low': (800, 600, 50), 'mid': (1300, 800, 70), 'high': (1800, 1100, 100)}
LIFE = (25, 25, 15)


def load(path, extra):
    d = pd.read_parquet(path, columns=KEYS + extra + ['mode', 'status', 'hours'] + COST + PHYS)
    return d[d.status.eq('optimal')]


def e0_table(section3):
    d = pd.read_parquet(section3, columns=KEYS + ['mode', 'status', 'intensity_of_max', 'facility_mwh'])
    d = d[d.status.eq('optimal') & d['mode'].eq('seq') & d.intensity_of_max.eq(0)]
    return d.set_index(KEYS).facility_mwh.rename('E0')


def paired(d, extra, e0):
    keys = KEYS + extra
    s = d[d['mode'].eq('seq')].set_index(keys)
    j = d[d['mode'].eq('joint')].set_index(keys)
    s, j = s.align(j, join='inner')
    e = e0.reindex(s.index.droplevel(extra)).to_numpy()
    if np.isnan(e).any():
        raise ValueError('部分配对缺少基准用电 E0')
    p = pd.DataFrame(index=s.index)
    p['E0'] = e
    p['V'] = (s[COST].sum(axis=1) - j[COST].sum(axis=1)) / e
    p['base_total'] = s[COST].sum(axis=1) / e
    for g, cols in GROUPS.items():
        p['d_' + g] = (s[cols].sum(axis=1) - j[cols].sum(axis=1)) / e     # 正值 = 协同省下的钱
        p['base_' + g] = s[cols].sum(axis=1) / e                         # 不做协同时的成本结构
    gap = (p[['d_' + g for g in GROUPS]].sum(axis=1) - p.V).abs().max()
    if gap > 1e-6:
        raise ValueError(f'分项之和与净节省不符：{gap}')
    load_mw = e / s.hours
    for mode, f in (('seq', s), ('joint', j)):
        p[f'{mode}_grid_share'] = f.grid_mwh / f.facility_mwh
        p[f'{mode}_pv_per_mw'] = f.pv_mw / load_mw
        p[f'{mode}_wind_per_mw'] = f.wind_mw / load_mw
        p[f'{mode}_storage_h'] = f.battery_mwh / load_mw
        p[f'{mode}_grid_cap_over_peak'] = f.grid_cap_mw / s.facility_peak_mw
    p['dEm_t_mwh'] = (s.emissions_t - j.emissions_t) / e
    for k, (pv, wd, bt) in EMBODIED.items():
        # 协同多出（+）或少出（−）的隐含排放，按寿命摊到每年再除以年用电
        p[f'emb_{k}_t_mwh'] = ((j.pv_mw - s.pv_mw) * pv / LIFE[0] + (j.wind_mw - s.wind_mw) * wd / LIFE[1]
                               + (j.battery_mwh - s.battery_mwh) * bt / LIFE[2]) / load_mw / 8760
    p = p.reset_index()
    p['use'] = p.scenario.str.split('-').str[1]
    return p


def summarize(p, by):
    g = p.groupby(by)
    comp = ['d_' + k for k in GROUPS]
    mean = g[['V'] + comp + ['base_' + k for k in GROUPS] + ['base_total']].mean()
    out = pd.DataFrame({'pairs': g.size(), 'saving_median_usd_mwh': g.V.median(),
                        'saving_mean_usd_mwh': mean.V, 'supply_cost_mean_usd_mwh': mean.base_total})
    for k in GROUPS:
        out[f'{k}_saving_mean_usd_mwh'] = mean['d_' + k]
    for k in GROUPS:
        out[f'{k}_share_of_net_pct'] = mean['d_' + k] / mean.V * 100   # 均值可加，正负相抵后合计 100%
    for k in GROUPS:
        out[f'{k}_baseline_cost_mean_usd_mwh'] = mean['base_' + k]
    phys = [c for c in p.columns if c.startswith(('seq_', 'joint_'))]
    out = out.join(g[phys + ['dEm_t_mwh']].median().add_suffix('_median'))
    return out.reset_index()


def lifecycle(p, price, label):
    rows = []
    for (u,), q in p.groupby(['use']):
        row = {'case': label, 'carbon_price_usd_t': price, 'use': u, 'pairs': len(q),
               'saving_median_usd_mwh': q.V.median(), 'operational_abatement_median_kg_mwh': q.dEm_t_mwh.median() * 1e3}
        for k in EMBODIED:
            emb = q[f'emb_{k}_t_mwh']
            row[f'embodied_change_{k}_median_kg_mwh'] = emb.median() * 1e3
            row[f'saving_change_{k}_median_usd_mwh'] = (-emb * price).median()
            row[f'saving_change_{k}_median_pct'] = (-emb * price / q.V).median() * 100
            row[f'net_abatement_{k}_median_kg_mwh'] = (q.dEm_t_mwh - emb).median() * 1e3
        rows.append(row)
    return rows


def section2(capacity, section3):
    cap = pd.read_parquet(capacity)
    cap['supply_saving_usd_mwh'] = (cap.baseline_energy_usd_year - cap.feasible_energy_usd_year) / cap.E0_mwh_year
    cap['equipment_saving_usd_mwh'] = cap.equipment_saving_feasible_usd_year / cap.E0_mwh_year
    cap['supply_share_of_total_pct'] = cap.supply_saving_usd_mwh / (
        cap.supply_saving_usd_mwh + cap.equipment_saving_usd_mwh) * 100
    cap['idle_energy_reduction_pct_of_E0'] = cap.idle_energy_reduction_mwh_year / cap.E0_mwh_year * 100
    down = cap.groupby('scenario')[['supply_saving_usd_mwh', 'equipment_saving_usd_mwh', 'supply_share_of_total_pct',
                                    'idle_energy_reduction_pct_of_E0']].median().add_suffix('_median')
    down.insert(0, 'grids', cap.groupby('scenario').size())
    d = pd.read_parquet(section3, columns=KEYS + ['mode', 'status', 'intensity_of_max'] + COST + ['facility_mwh'])
    d = d[d.status.eq('optimal') & d['mode'].eq('seq')].set_index(KEYS + ['intensity_of_max'])
    base = d.xs(0.0, level='intensity_of_max')
    rows = []
    for x in (0.25, 0.5, 0.75):
        s, b = d.xs(x, level='intensity_of_max').align(base, join='inner')
        added = s.facility_mwh - b.facility_mwh
        r = pd.DataFrame({'scenario': s.index.get_level_values('scenario'), 'intensity_of_max': x})
        for k, cols in GROUPS.items():
            r[k] = ((s[cols].sum(axis=1) - b[cols].sum(axis=1)) / added).to_numpy()
        r['total'] = ((s[COST].sum(axis=1) - b[COST].sum(axis=1)) / added).to_numpy()
        r['baseline_average'] = (b[COST].sum(axis=1) / b.facility_mwh).to_numpy()
        rows.append(r)
    r = pd.concat(rows)
    r['use'] = r.scenario.str.split('-').str[1]
    marginal = r.groupby(['intensity_of_max', 'use']).mean(numeric_only=True).add_suffix('_mean_usd_mwh')
    return down.reset_index(), marginal.reset_index()


def main():
    runs = Path('D:/codex-runs')
    ap = argparse.ArgumentParser()
    ap.add_argument('--section3', default=runs / 'residual-flex-pilot-20261114/global/runs-compact.parquet')
    ap.add_argument('--carbon', default=runs / 'sec4-20261116/carbon-compact.parquet')
    ap.add_argument('--sec4-data', default=runs / 'sec4-card-inference-20261118/data')
    ap.add_argument('--capacity', default=ROOT / 'results/pending-results-20261116/global-capacity.parquet')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    e0 = e0_table(a.section3)
    write = lambda f, name: f.to_csv(OUT / name, index=False, float_format='%.6g', encoding='utf-8-sig')

    p13 = paired(load(a.section3, ['intensity_of_max']), ['intensity_of_max'], e0)
    write(summarize(p13, ['intensity_of_max', 'scenario']), 'sec1-3-by-scenario.csv')
    write(summarize(p13, ['intensity_of_max', 'use']), 'sec1-3-by-use.csv')
    write(summarize(p13, ['intensity_of_max']), 'sec1-3-all.csv')
    life = lifecycle(p13[p13.intensity_of_max.eq(0)], 100, 'carbon100_I0')
    life += lifecycle(p13[p13.intensity_of_max.eq(.75)], 100, 'carbon100_I75')

    pc = paired(load(a.carbon, ['carbon_price_usd_t', 'intensity_of_max']), ['carbon_price_usd_t', 'intensity_of_max'], e0)
    pc = pd.concat([pc, p13.assign(carbon_price_usd_t=100.)])
    write(summarize(pc, ['carbon_price_usd_t', 'intensity_of_max', 'use']), 'sec4-carbon-by-use.csv')
    write(summarize(pc, ['carbon_price_usd_t', 'intensity_of_max']), 'sec4-carbon-all.csv')
    life += lifecycle(pc[pc.carbon_price_usd_t.eq(200) & pc.intensity_of_max.eq(0)], 200, 'carbon200_I0')
    del pc

    lc = pd.concat([load(Path(a.sec4_data) / f'{f}.parquet', ['cost_scenario', 'intensity_of_max'])
                    for f in ('full-lc', 'full-lc2')])
    pl = paired(lc, ['cost_scenario', 'intensity_of_max'], e0)
    pl = pd.concat([pl, p13.assign(cost_scenario='2024')])
    write(summarize(pl, ['cost_scenario', 'intensity_of_max', 'use']), 'sec4-learning-by-use.csv')
    write(summarize(pl, ['cost_scenario', 'intensity_of_max']), 'sec4-learning-all.csv')
    for path in ('2035-advanced', '2035-moderate'):
        life += lifecycle(pl[pl.cost_scenario.eq(path) & pl.intensity_of_max.eq(0)], 100, f'{path}_I0')
    del lc, pl

    cross = load(Path(a.sec4_data) / 'full-cross.parquet', ['cost_scenario', 'carbon_price_usd_t', 'intensity_of_max'])
    px = paired(cross, ['cost_scenario', 'carbon_price_usd_t', 'intensity_of_max'], e0)
    write(summarize(px, ['cost_scenario', 'carbon_price_usd_t', 'intensity_of_max']), 'sec4-cross-all.csv')
    del cross, px

    write(pd.DataFrame(life), 'lifecycle-emissions-check.csv')
    down, marginal = section2(a.capacity, a.section3)
    write(down, 'sec2-downsizing-supply.csv')
    write(marginal, 'sec2-inference-marginal-supply-cost.csv')


if __name__ == '__main__':
    main()
