"""第二节卖推理三层（全球）：在第三节全球网格上合并收入与额外供能成本。

- 售出量：按空闲 GPU 计（八城市 sell_inference_tiers.py，与地点无关）。任务功率与占用 GPU 不成固定比例
  （法兰克福 D-TRN 逐时每卡功率 P5–P95 为 0.34–0.59 kW，功率峰值与 GPU 峰值不在同一小时），按功率口径折算会多计 20–50%。
- 额外供能成本：同一网格顺序调度（不协同）下，该层与 I=0 的供能总成本之差，供能资产按规划边界重新优化；
  再按 GPU 口径售出电量（GPU 小时 × 平均每卡任务功率）与模型中推理任务电量之比等比例折算。
- 收入：2.9 美元/GPU·时全天同价（DeepInfra H100）；峰谷敏感性乘以八城市算得的时段因子（平均价不变）。
- 对照：第三节同一强度下联合调度比顺序调度多省的供能成本（ideal 与 max 两档无此项）。
全部按窗口小时年化到每年。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PRICE = 2.9
LEVEL_NAME = {0.25: 'real_25', 0.5: 'real_50', 0.75: 'real_75'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True, help='第三节全量 runs-compact.parquet')
    ap.add_argument('--tiers', required=True, help='ideal/max 补算 runs-compact.parquet')
    ap.add_argument('--city', required=True, help='八城市 sell-inference-tiers-median.csv')
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    city = pd.read_csv(args.city, index_col=0)
    gpu_scen = list(city.index)
    runs = pd.read_parquet(args.runs)
    runs = runs[(runs.status == 'optimal') & runs.scenario.isin(gpu_scen)]
    key = ['grid_id', 'scenario']
    seq = runs[runs['mode'] == 'seq']
    base = seq[seq.intensity_of_max == 0].set_index(key)
    joint = runs[runs['mode'] == 'joint'].set_index(key + ['intensity_of_max']).total_cost_usd
    parts = []
    for level, name in LEVEL_NAME.items():
        x = seq[seq.intensity_of_max == level].set_index(key)
        x = x.assign(tier=name, coord_saving_usd=x.total_cost_usd.to_numpy()
                     - joint.reindex(pd.MultiIndex.from_arrays([x.index.get_level_values(0), x.index.get_level_values(1),
                                                                np.full(len(x), level)])).to_numpy())
        parts.append(x)
    tiers = pd.read_parquet(args.tiers)
    bad = tiers[tiers.status != 'optimal']
    tiers = tiers[tiers.status == 'optimal'].set_index(key)
    tiers['tier'] = tiers.tier.map({'ideal': 'ideal', 'max': 'real_max'})
    tiers['coord_saving_usd'] = np.nan
    t = pd.concat(parts + [tiers]).reset_index()
    b = base.reindex(pd.MultiIndex.from_frame(t[key]))
    year = 8760 / t.hours.to_numpy()
    t['extra_supply_cost_usd_year'] = (t.total_cost_usd.to_numpy() - b.total_cost_usd.to_numpy()) * year
    t['model_task_mwh_year'] = (t.inf_online_mwh + t.inf_offline_mwh) * year
    level = t.tier.map({'ideal': np.nan, 'real_max': 1.0, 'real_75': 0.75, 'real_50': 0.5, 'real_25': 0.25})
    gpu_h = np.where(t.tier == 'ideal', t.scenario.map(city.ideal_gpu_hours_year), level * t.scenario.map(city.real_max_gpu_hours_year))
    t['sold_gpu_hours_year'] = gpu_h
    t['energy_scale'] = gpu_h * t.scenario.map(city.mw_per_gpu) / t.model_task_mwh_year
    t['extra_supply_cost_usd_year'] = t.extra_supply_cost_usd_year * t.energy_scale
    factor = np.where(t.tier == 'ideal', t.scenario.map(city.ideal_tou_factor), t.scenario.map(city.real_max_tou_factor))
    t['revenue_usd_year'] = PRICE * t.sold_gpu_hours_year
    t['net_usd_year'] = t.revenue_usd_year - t.extra_supply_cost_usd_year
    t['net_tou_usd_year'] = PRICE * factor * t.sold_gpu_hours_year - t.extra_supply_cost_usd_year
    t['coord_saving_usd_year'] = t.coord_saving_usd * year
    keep = ['grid_id', 'scenario', 'tier', 'lat', 'lon', 'iso3', 'continent', 'hours', 'sold_gpu_hours_year', 'energy_scale', 'revenue_usd_year',
            'extra_supply_cost_usd_year', 'net_usd_year', 'net_tou_usd_year', 'coord_saving_usd_year']
    t[keep].to_parquet(args.out / 'sell-inference-global.parquet', index=False)
    order = ['ideal', 'real_max', 'real_75', 'real_50', 'real_25']
    g = t.groupby(['scenario', 'tier'])
    summary = pd.DataFrame({
        'grids': g.size(),
        'net_median': g.net_usd_year.median(), 'net_p10': g.net_usd_year.quantile(0.1), 'net_p90': g.net_usd_year.quantile(0.9),
        'net_tou_median': g.net_tou_usd_year.median(),
        'extra_supply_cost_median': g.extra_supply_cost_usd_year.median(),
        'energy_scale_median': g.energy_scale.median(),
        'coord_saving_median': g.coord_saving_usd_year.median()}).reset_index()
    summary['tier'] = pd.Categorical(summary.tier, order)
    summary = summary.sort_values(['scenario', 'tier'])
    summary.to_csv(args.out / 'sell-inference-global-summary.csv', index=False)
    pd.DataFrame({'nonoptimal_rows': [len(bad)]}).to_csv(args.out / 'tiers-nonoptimal.csv', index=False)
    view = summary.set_index(['scenario', 'tier'])[['net_median', 'net_p10', 'net_p90', 'net_tou_median', 'coord_saving_median']] / 1e4
    print(view.round(0).to_string())


if __name__ == '__main__':
    main()
