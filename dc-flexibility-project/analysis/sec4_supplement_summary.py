"""第四节局限补算汇总：碳价 × 学习曲线交叉、学习曲线补推理强度 25/75%。

输入（服务器 /data3/dataset/ljt/residual-flex-global-20261114 各目录 runs-compact.parquet 的本地副本）：
  full        第三节全量（2024 成本、碳价 100、强度上限 0/25/50/75%），同时给出基准用电 E0
  full-carbon 碳价 0/200 × 2024 成本 × 四档强度
  full-lc     ATB 六路径 × 碳价 100 × 强度 0/50%
  full-lc2    ATB 六路径 × 碳价 100 × 强度 25/75%（补算）
  full-cross  ATB 六路径 × 碳价 0/200 × 强度 0/50%（补算）
口径同 results/sec4-carbon-learning-20261116/README.md：V = 顺序调度 − 联合调度供能成本，按年折算后 ÷ E0
（不卖推理、碳价 100、2024 成本下顺序调度年用电），单位美元/MWh。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ['grid_id', 'scenario', 'cost_scenario', 'carbon_price_usd_t', 'intensity_of_max']


def paired(path, cost=None, carbon=None):
    r = pd.read_parquet(path, columns=['grid_id', 'scenario', 'mode', 'status', 'hours', 'total_cost_usd', 'intensity_of_max', 'continent']
                        + (['cost_scenario'] if cost is None else []) + (['carbon_price_usd_t'] if carbon is None else []))
    r = r[r.status == 'optimal']
    if cost is not None:
        r['cost_scenario'] = cost
    if carbon is not None:
        r['carbon_price_usd_t'] = carbon
    r['intensity_of_max'] = r.intensity_of_max.round(4)
    r['annual'] = r.total_cost_usd * 8760 / r.hours
    w = r.pivot_table(index=KEY, columns='mode', values='annual', aggfunc='first').dropna(subset=['seq', 'joint'])
    return w.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True, help='含 full.parquet、full-carbon.parquet、full-lc.parquet、full-lc2.parquet、full-cross.parquet')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(a.data / 'full.parquet', columns=['grid_id', 'scenario', 'mode', 'status', 'hours', 'facility_mwh', 'intensity_of_max'])
    e0 = base[(base['mode'] == 'seq') & (base.intensity_of_max == 0) & (base.status == 'optimal')]
    e0 = (e0.facility_mwh * 8760 / e0.hours).set_axis(pd.MultiIndex.from_frame(e0[['grid_id', 'scenario']])).rename('E0')
    d = pd.concat([paired(a.data / 'full.parquet', 'base-2024', 100.), paired(a.data / 'full-carbon.parquet', 'base-2024'),
                   paired(a.data / 'full-lc.parquet'), paired(a.data / 'full-lc2.parquet'), paired(a.data / 'full-cross.parquet')])
    assert not d.duplicated(KEY).any()
    d = d.join(e0, on=['grid_id', 'scenario'])
    d['V'] = (d.seq - d.joint) / d.E0
    d['V_pct'] = 100 * (d.seq - d.joint) / d.seq
    d['supply'] = d.seq / d.E0
    cover = d.groupby(['cost_scenario', 'carbon_price_usd_t', 'intensity_of_max']).size().rename('pairs').reset_index()
    cover.to_csv(a.out / 'coverage.csv', index=False)

    q = lambda s, p: s.quantile(p)
    g = d.groupby(['scenario', 'cost_scenario', 'carbon_price_usd_t', 'intensity_of_max'])
    summary = pd.DataFrame({'pairs': g.size(), 'V_usd_per_mwh_p10': g.V.quantile(.1), 'V_usd_per_mwh_median': g.V.median(),
                            'V_usd_per_mwh_p90': g.V.quantile(.9), 'supply_cost_usd_per_mwh_median': g.supply.median(),
                            'V_pct_of_supply_median': g.V_pct.median()}).reset_index()
    summary.to_csv(a.out / 'cross-summary.csv', index=False)

    # 碳价 0→200 放大倍数（逐对比值，按情境取中位数），各成本路径
    v = d.set_index(KEY).V.unstack('carbon_price_usd_t').dropna(subset=[0., 100., 200.])  # 只取三档碳价都有的组合
    amp = (v[200.] / v[0.]).rename('amp').reset_index()
    mono = ((v[100.] >= v[0.] - 1e-9) & (v[200.] >= v[100.] - 1e-9)).rename('mono').reset_index()
    ampt = amp.merge(mono, on=['grid_id', 'scenario', 'cost_scenario', 'intensity_of_max'])
    ampt = ampt.groupby(['scenario', 'cost_scenario', 'intensity_of_max']).agg(amp_median=('amp', 'median'), monotone_share=('mono', 'mean'),
                                                                                 pairs=('amp', 'size')).reset_index()
    ampt.to_csv(a.out / 'carbon-amplification-by-cost.csv', index=False)

    # 学习曲线倍数（相对同碳价、同强度的 2024），各碳价、各强度
    w = d.set_index(KEY).V.unstack('cost_scenario')
    ratio = w.drop(columns='base-2024').div(w['base-2024'], axis=0).stack().rename('ratio').reset_index()
    rt = ratio.groupby(['scenario', 'cost_scenario', 'carbon_price_usd_t', 'intensity_of_max']).ratio.agg(
        ratio_median='median', ratio_p25=lambda s: q(s, .25), ratio_p75=lambda s: q(s, .75), share_above_1=lambda s: (s > 1).mean(),
        pairs='size').reset_index()
    rt.to_csv(a.out / 'learning-ratio-by-carbon-intensity.csv', index=False)
    allr = ratio.groupby(['cost_scenario', 'carbon_price_usd_t', 'intensity_of_max']).ratio.agg(
        ratio_median='median', share_above_1=lambda s: (s > 1).mean(), pairs='size').reset_index()
    allr.to_csv(a.out / 'learning-ratio-all.csv', index=False)
    alla = amp.merge(mono, on=['grid_id', 'scenario', 'cost_scenario', 'intensity_of_max']).groupby(['cost_scenario', 'intensity_of_max']).agg(
        amp_median=('amp', 'median'), monotone_share=('mono', 'mean'), pairs=('amp', 'size')).reset_index()
    alla.to_csv(a.out / 'carbon-amplification-all.csv', index=False)
    print(cover.to_string())
    print(alla.to_string())
    print(allr.to_string())


if __name__ == '__main__':
    main()
