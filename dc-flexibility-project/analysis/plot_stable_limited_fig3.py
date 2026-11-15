"""图3：算电协同在真实推理需求下仍能产生稳定收益提升（第三节）。

a 保留率随推理需求强度变化（可承接上限的 0/25/50/75%）；
b 双变量地图（Mollweide）：训练机群强度 75% 保留率 × 不接推理时风电装机占比（各三分位）；
c 箭头图：转移电量（原任务 + 离线推理）与每 MWh 转移电量价值相对 I=0 的变化；
d 热图（三城市试算）：离线推理可推迟带来的补充、需求波动放大对保留率的影响；
e 强度 75% 时协同节省的累积分布（对数轴），全部为正；
f 协同节省占卖推理净收入（25/50/75%）。
"""
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stable_limited_figs_common import (COLORS, GPU_SCENARIOS, GROUP, INPUT, REPRESENTATIVES, ROOT, SCENARIOS, csv,
                                        draw_map, export, guides, label, paired, panel, scenario_axis, sell, setup,
                                        shade, use)

LEVELS = [0, .25, .5, .75]
# 3×3 双变量色板（行：风电占比低→高；列：保留率低→高）
BIVAR = [['#F4E3A1', '#A8D5A0', '#4AB39C'],
         ['#E7A9B6', '#A7A6BD', '#5891A8'],
         ['#C9609A', '#8B5C9E', '#3B4994']]
PILOT = ROOT / 'results/residual-flex-pilot-20261114'
CITY = {'riyadh': 'Riyadh', 'frankfurt': 'Frankfurt', 'singapore': 'Singapore'}


def pilot_tables():
    p = pd.read_csv(PILOT / 'paired.csv')
    p = p[p.grid_fee_usd_kw_year.eq(120)]
    real = p[p['shape'].eq('real')].set_index(['city', 'scenario', 'intensity'])
    off = pd.read_csv(PILOT / 'offline-not-deferrable.csv').set_index(['city', 'scenario', 'intensity'])
    off_v = off.seq - off.joint
    contrib = ((real.V.reindex(off_v.index) - off_v) / real.V.reindex(off_v.index) * 100).unstack()
    r = p.pivot_table(index=['city', 'scenario', 'intensity'], columns='shape', values='R')
    vol = ((r.amp2 - r.flat) * 100).unstack()
    vol_real = ((r.amp2 - r.real) * 100).unstack()
    return contrib, vol, vol_real


def color_of(sc):
    return COLORS[use(sc)]


def main():
    setup()
    p = paired()
    g = p.groupby(['scenario', 'intensity_of_max']).retention
    ret = pd.DataFrame({'p10': g.quantile(.1), 'median': g.median(), 'p90': g.quantile(.9)}).reset_index()

    fig = plt.figure(figsize=(180 / 25.4, 215 / 25.4))

    # ---------------- a 保留率曲线
    ax_a = fig.add_axes([.075, .7, .21, .265])
    x = np.array(LEVELS) * 100
    for sc in SCENARIOS:
        r = ret[ret.scenario.eq(sc)].set_index('intensity_of_max').loc[LEVELS]
        rep = sc in REPRESENTATIVES
        if rep:
            ax_a.fill_between(x, r.p10, r.p90, color=color_of(sc), alpha=.13, lw=0)
        ax_a.plot(x, r['median'], color=shade(sc), lw=1.1 if rep else .6, ls='-' if rep else (0, (2, 1.5)),
                  marker='o' if rep else None, markersize=2.2, zorder=3 if rep else 2)
    ax_a.axhline(1, color='#999999', lw=.4, ls=(0, (3, 2)))
    ax_a.set_xlim(-3, 78)
    ax_a.set_ylim(.2, 1.12)
    ax_a.set_xticks(x)
    ax_a.set_xlabel('External inference demand\n(% of absorbable cap)')
    ax_a.set_ylabel('Retention of coordination saving')
    guides(ax_a, 'y')
    handles = [Line2D([], [], color=COLORS[use(sc)], lw=1.1,
                      label=GROUP[use(sc)] + ('' if sc[0] == 'A' else ' (GPU-heavy)')) for sc in REPRESENTATIVES]
    handles += [Line2D([], [], color='#999999', lw=.6, ls=(0, (2, 1.5)), label='CPU-heavy, balanced'),
                Patch(facecolor='#999999', alpha=.25, label='P10–P90')]
    ax_a.legend(handles=handles, loc='lower left', fontsize=5.1, handlelength=1.4, labelspacing=.25)
    panel(fig, 'a', .012, .99)

    # ---------------- b 双变量地图
    q75 = p[p.intensity_of_max.eq(.75)]
    dtrn = q75[q75.scenario.eq('D-TRN')].set_index('grid_id')
    rc = pd.qcut(dtrn.retention, 3, labels=False)
    wc = pd.qcut(dtrn.wind_share, 3, labels=False)
    colors = pd.Series([BIVAR[w][r] for r, w in zip(rc, wc)], index=dtrn.index)
    draw_map(fig, [.34, .69, .64, .285], dtrn.retention, colors=colors, projection=ccrs.Mollweide(),
             lon_labels=False)
    lg = fig.add_axes([.375, .705, .06, .06])
    for i in range(3):
        for j in range(3):
            lg.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=BIVAR[i][j], edgecolor='white', lw=.4))
    lg.set_xlim(0, 3)
    lg.set_ylim(0, 3)
    lg.set_aspect('equal')
    lg.set_xticks([])
    lg.set_yticks([])
    for side in lg.spines.values():
        side.set_visible(False)
    lg.set_xlabel('Retention →', fontsize=5.3, labelpad=1)
    lg.set_ylabel('Wind share →', fontsize=5.3, labelpad=1)
    rq = dtrn.retention.quantile([1 / 3, 2 / 3]).to_numpy()
    wq = dtrn.wind_share.quantile([1 / 3, 2 / 3]).to_numpy() * 100
    rho = spearmanr(dtrn.wind_share, dtrn.retention)[0]
    panel(fig, 'b', .33, .99)

    # ---------------- c 箭头图：转移电量 vs 每 MWh 价值
    mv = pd.read_parquet(INPUT / 'moved-energy.parquet')
    mv['moved'] = mv.moved_task_mwh + mv.moved_off_mwh
    mv = mv.merge(p[['grid_id', 'scenario', 'intensity_of_max', 'V_usd']], on=['grid_id', 'scenario', 'intensity_of_max'])
    base = mv[mv.intensity_of_max.eq(0)].set_index(['grid_id', 'scenario'])[['moved', 'V_usd']]
    mv = mv.join(base, on=['grid_id', 'scenario'], rsuffix='0')
    mv['moved_ratio'] = mv.moved / mv.moved0
    mv['value_ratio'] = (mv.V_usd / mv.moved) / (mv.V_usd0 / mv.moved0)
    path = mv.groupby(['scenario', 'intensity_of_max'])[['moved_ratio', 'value_ratio']].median().reset_index()
    ax_c = fig.add_axes([.085, .385, .24, .225])
    ax_c.axhline(1, color='#BBBBBB', lw=.5)
    ax_c.axvline(1, color='#BBBBBB', lw=.5)
    for sc in SCENARIOS:
        q = path[path.scenario.eq(sc)].set_index('intensity_of_max').loc[LEVELS]
        c = shade(sc)
        ax_c.plot(q.moved_ratio.iloc[:-1], q.value_ratio.iloc[:-1], color=c, lw=.8, marker='o', markersize=1.8)
        ax_c.annotate('', xy=(q.moved_ratio.iloc[-1], q.value_ratio.iloc[-1]),
                      xytext=(q.moved_ratio.iloc[-2], q.value_ratio.iloc[-2]),
                      arrowprops=dict(arrowstyle='-|>,head_length=.35,head_width=.18', color=c, lw=.8,
                                      shrinkA=0, shrinkB=0))
    ax_c.set_xlim(.85, 1.15)
    ax_c.set_ylim(.45, 1.15)
    ax_c.set_xlabel('Energy shifted by coordination\n(relative to no inference)')
    ax_c.set_ylabel('Value per shifted MWh\n(relative to no inference)')
    ax_c.legend(handles=[Line2D([], [], color=COLORS[u], lw=1, label=GROUP[u]) for u in GROUP],
                loc='lower left', fontsize=5, handlelength=1.2, labelspacing=.25)
    guides(ax_c, 'both')
    panel(fig, 'c', .012, .65)

    # ---------------- d 热图：离线推理可推迟、需求波动（三城市试算）
    contrib, vol, vol_real = pilot_tables()
    order = [(c, s) for c in CITY for s in ('B-TRN', 'D-TRN')]
    contrib = contrib.loc[order, [.25, .5, .75]]
    vol = vol.loc[order, [.25, .5]]
    ax_d1 = fig.add_axes([.555, .385, .15, .2])
    ax_d2 = fig.add_axes([.765, .385, .1, .2])
    cm1 = mpl.colormaps['Purples']
    cm2 = mpl.colormaps['Oranges']
    ax_d1.imshow(contrib.to_numpy(), cmap=cm1, vmin=0, vmax=15, aspect='auto')
    ax_d2.imshow(vol.to_numpy(), cmap=cm2, vmin=0, vmax=15, aspect='auto')
    for axh, tab in [(ax_d1, contrib), (ax_d2, vol)]:
        for i in range(tab.shape[0]):
            for j in range(tab.shape[1]):
                v = tab.iat[i, j]
                axh.text(j, i, '–' if pd.isna(v) else f'{v:.1f}', ha='center', va='center', fontsize=5.3,
                         color='white' if (not pd.isna(v) and v > 9) else '#222222')
        axh.set_xticks(range(tab.shape[1]), [f'{c:.2f}' for c in tab.columns], fontsize=5.5)
        axh.tick_params(length=0, pad=2)
        for side in axh.spines.values():
            side.set_visible(False)
        axh.set_xlabel('Inference demand\n(× mean idle power)')
    ax_d1.set_yticks(range(len(order)), [('CPU-heavy' if s[0] == 'B' else 'GPU-heavy') for _, s in order],
                     fontsize=5.4)
    ax_d2.set_yticks([])
    for k, c in enumerate(CITY.values()):
        ax_d1.text(-1.75, 2 * k + .5, c, ha='right', va='center', fontsize=5.8, fontweight='bold')
        if k:
            for axh in (ax_d1, ax_d2):
                axh.axhline(2 * k - .5, color='white', lw=1.5)
    ax_d1.set_title('Saving lost if offline inference\ncannot be deferred (%)', fontsize=5.6, pad=3)
    ax_d2.set_title('Retention gain when\nvolatility doubles (pp)', fontsize=5.6, pad=3)
    fig.text(.43, .385 + .1, 'LLM training', rotation=90, ha='center', va='center', fontsize=5.8,
             color=COLORS['TRN'], fontweight='bold')
    panel(fig, 'd', .4, .65)

    # ---------------- e 累积分布：强度 75% 全部为正
    ax_e = fig.add_axes([.085, .07, .29, .2])
    for sc in SCENARIOS:
        v = np.sort(q75.loc[q75.scenario.eq(sc), 'saving_usd_mwh'].to_numpy())
        ax_e.plot(v, np.arange(1, len(v) + 1) / len(v), color=shade(sc), lw=.9 if sc in REPRESENTATIVES else .6)
    ax_e.set_xscale('log')
    ax_e.set_xlim(.02, 40)
    ax_e.set_xticks([.1, 1, 10], ['0.1', '1', '10'])
    ax_e.set_ylim(0, 1)
    ax_e.axhline(.1, color='#8B253E', lw=.5, ls=(0, (3, 2)))
    ax_e.set_xlabel('Coordination saving at 75% of cap (USD/MWh)')
    ax_e.set_ylabel('Cumulative share of grid cells')
    ax_e.text(.025, .12, 'P10', color='#8B253E', fontsize=5.3, va='bottom')
    ax_e.legend(handles=[Line2D([], [], color=COLORS[u], lw=1, label=GROUP[u]) for u in GROUP],
                loc='upper left', fontsize=5, handlelength=1.2, labelspacing=.25)
    guides(ax_e, 'both')
    panel(fig, 'e', .012, .31)

    # ---------------- f 协同节省 / 卖推理净收入
    sl = sell()
    sl = sl[sl.tier.isin(['real_25', 'real_50', 'real_75'])]
    m = sl.groupby(['scenario', 'tier'])[['coord_usd_mwh', 'net_usd_mwh']].median()
    share = (m.coord_usd_mwh / m.net_usd_mwh * 100).unstack().loc[GPU_SCENARIOS]
    ax_f = fig.add_axes([.53, .115, .45, .16])
    xs = np.arange(len(GPU_SCENARIOS))
    for k, (t, alpha) in enumerate([('real_25', .35), ('real_50', .65), ('real_75', 1.)]):
        ax_f.bar(xs + (k - 1) * .27, share[t], width=.25, color=[shade(sc) for sc in GPU_SCENARIOS], alpha=alpha,
                 lw=0)
    ax_f.set_xlim(-.6, len(xs) - .4)
    ax_f.set_ylabel('Coordination saving /\nnet inference revenue (%)')
    scenario_axis(ax_f, GPU_SCENARIOS, xs, axis='x', outer=-.42)
    ax_f.legend(handles=[Patch(facecolor='#555555', alpha=a, label=t) for a, t in
                         [(.35, 'Sell 25% of cap'), (.65, '50%'), (1., '75%')]], loc='upper right', ncol=3,
                fontsize=5.2, handlelength=1, columnspacing=.8)
    guides(ax_f, 'y')
    panel(fig, 'f', .45, .31)

    export(fig, 'fig3')

    csv(ret.assign(label=ret.scenario.map(label)), 'fig3a')
    csv(dtrn.reset_index()[['grid_id', 'lat', 'lon', 'retention', 'wind_share']].assign(
        retention_class=rc.values, wind_class=wc.values), 'fig3b')
    csv(path, 'fig3c')
    csv(contrib.stack().rename('offline_contrib_pct').reset_index().merge(
        vol.stack().rename('amp2_minus_flat_pp').reset_index(), how='left'), 'fig3d')
    csv(q75.groupby('scenario').saving_usd_mwh.describe(percentiles=[.1, .5]).reset_index(), 'fig3e')
    csv(share.reset_index(), 'fig3f')
    return path, contrib, vol, vol_real, rho, share


if __name__ == '__main__':
    path, contrib, vol, vol_real, rho, share = main()
    print(path[path.intensity_of_max.eq(.75)].round(2).to_string())
    print(contrib.round(1).to_string()); print(vol.round(1).to_string()); print(vol_real.round(1).to_string())
    print(rho)
