"""图3：算电协同在真实推理需求下仍能产生稳定收益提升（第三节）。

a 保留率随推理需求强度（可承接上限的 0/25/50/75%）变化，11 情境中位数，代表情境加 P10–P90 带；
b 训练机群（GPU-heavy）强度 75% 时保留率全球地图 + 纬度统计条；
c 训练机群保留率与无推理时最优风电装机占比的关系（六边形密度，Spearman ρ）；
d 强度 75% 时协同节省 USD/MWh 分布：213,048 对配对算例全部为正；
e 协同节省占卖推理净收入的比例（25/50/75%）。
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stable_limited_figs_common import (GPU_SCENARIOS, NOTE, REPRESENTATIVES, SCENARIOS, colorbar, color, csv,
                                        draw_map, export, guides, label, latitude_strip, paired, panel, sell, setup,
                                        shade)

LEVELS = [0, .25, .5, .75]


def main():
    setup()
    p = paired()
    g = p.groupby(['scenario', 'intensity_of_max']).retention
    ret = pd.DataFrame({'p10': g.quantile(.1), 'median': g.median(), 'p90': g.quantile(.9)}).reset_index()

    fig = plt.figure(figsize=(180 / 25.4, 160 / 25.4))

    # ---------------- a 保留率曲线
    ax_a = fig.add_axes([.065, .655, .2, .28])
    x = np.array(LEVELS) * 100
    for sc in SCENARIOS:
        r = ret[ret.scenario.eq(sc)].set_index('intensity_of_max').loc[LEVELS]
        rep = sc in REPRESENTATIVES
        if rep:
            ax_a.fill_between(x, r.p10, r.p90, color=color(sc), alpha=.13, lw=0)
        ax_a.plot(x, r['median'], color=shade(sc) if not rep else color(sc), lw=1.1 if rep else .6,
                  ls='-' if rep else (0, (2, 1.5)), marker='o' if rep else None, markersize=2.2, zorder=3 if rep else 2)
    ax_a.axhline(1, color='#999999', lw=.4, ls=(0, (3, 2)))
    ax_a.set_xlim(-3, 78)
    ax_a.set_ylim(.2, 1.12)
    ax_a.set_xticks(x)
    ax_a.set_xlabel('External inference demand\n(% of absorbable cap)')
    ax_a.set_ylabel('Retention of coordination saving')
    guides(ax_a, 'y')
    handles = [Line2D([], [], color=color(sc), lw=1.1, label=label(sc, short=True)) for sc in REPRESENTATIVES]
    handles += [Line2D([], [], color='#999999', lw=.6, ls=(0, (2, 1.5)), label='Other fleets'),
                Patch(facecolor='#999999', alpha=.25, label='P10–P90')]
    ax_a.legend(handles=handles, loc='lower left', fontsize=5.2, handlelength=1.4, labelspacing=.25, ncol=1)
    panel(fig, 'a', .012, .985, 'Retention falls with demand')

    # ---------------- b 地图 + 纬度条
    q75 = p[p.intensity_of_max.eq(.75)]
    dtrn = q75[q75.scenario.eq('D-TRN')].set_index('grid_id')
    cmap = mpl.colormaps['RdYlBu']
    norm = mpl.colors.Normalize(.2, 1.0)
    ax_map, mapped = draw_map(fig, [.335, .6, .5, .345], dtrn.retention, cmap, norm)
    frames = [(label(sc, short=True), color(sc), q75.loc[q75.scenario.eq(sc), ['lat', 'retention']]
               .rename(columns={'retention': 'value'}), sc == 'D-TRN') for sc in REPRESENTATIVES]
    ax_lat, lat_rows = latitude_strip(fig, ax_map, .07, frames, 'Retention', xlim=(.2, 1.1))
    ax_lat.set_xticks([.25, .5, .75, 1])
    ax_lat.set_xticklabels(['.25', '.5', '.75', '1'])
    colorbar(fig, [.42, .655, .3, .01], cmap, norm, [.2, .4, .6, .8, 1.0],
             'Retention, LLM training · GPU-heavy, demand at 75% of cap', extend='both')
    panel(fig, 'b', .315, .985, 'About half of the training saving remains even at 75% of cap')

    # ---------------- c 风电占比 vs 保留率
    ax_c = fig.add_axes([.065, .1, .2, .38])
    hb = ax_c.hexbin(dtrn.wind_share * 100, dtrn.retention, gridsize=28, cmap='Greens', mincnt=1, lw=.1, bins='log',
                     extent=(0, 100, .1, 1.0), rasterized=True)
    rho = {sc: spearmanr(q75.loc[q75.scenario.eq(sc), 'wind_share'], q75.loc[q75.scenario.eq(sc), 'retention'])[0]
           for sc in SCENARIOS}
    bins = pd.cut(dtrn.wind_share * 100, np.arange(0, 101, 10))
    binned = dtrn.groupby(bins, observed=True).retention.median()
    ax_c.plot([b.mid for b in binned.index], binned.values, color='#1B5E20', lw=1, marker='o', markersize=2)
    ax_c.set_xlim(0, 100)
    ax_c.set_ylim(.1, 1.0)
    ax_c.set_xlabel('Wind share of optimal wind + PV capacity\nwithout inference (%)')
    ax_c.set_ylabel('Retention (training · D, 75% of cap)')
    ax_c.text(.04, .96, f'Spearman ρ = {rho["D-TRN"]:.2f}\n{len(dtrn):,} grid cells', transform=ax_c.transAxes,
              va='top', fontsize=5.6)
    cb = fig.colorbar(hb, cax=fig.add_axes([.272, .1, .006, .13]))
    cb.ax.tick_params(labelsize=5, length=1.5, width=.4, pad=1)
    cb.outline.set_linewidth(.3)
    cb.set_label('Cells', fontsize=5.2, labelpad=1)
    panel(fig, 'c', .012, .555, 'Wind–solar complementarity sets the remainder')

    # ---------------- d 强度 75% 时 USD/MWh 全部为正
    ax_d = fig.add_axes([.43, .1, .21, .38])
    y = np.arange(len(SCENARIOS))[::-1]
    stat = q75.groupby('scenario').saving_usd_mwh.describe(percentiles=[.1, .25, .5, .75, .9]).loc[SCENARIOS]
    i0 = p[p.intensity_of_max.eq(0)].groupby('scenario').saving_usd_mwh.median()
    for yi, sc in zip(y, SCENARIOS):
        r = stat.loc[sc]
        c = shade(sc)
        ax_d.plot([r['10%'], r['90%']], [yi, yi], color=c, lw=.7, zorder=2)
        ax_d.add_patch(mpl.patches.Rectangle((r['25%'], yi - .27), r['75%'] - r['25%'], .54, facecolor=c,
                                             edgecolor=c, lw=.5, zorder=3))
        ax_d.plot([r['50%']] * 2, [yi - .27, yi + .27], color='#1A1A1A', lw=.8, zorder=4)
        ax_d.scatter(r['min'], yi, marker='|', s=14, color='#8B253E', lw=.8, zorder=4)
        ax_d.scatter(i0[sc], yi, marker='o', s=9, facecolors='white', edgecolors='#333333', lw=.5, zorder=5)
    ax_d.axvline(0, color='#555555', lw=.5)
    ax_d.set_xlim(-.5, 20)
    ax_d.set_ylim(-.7, len(y) - .3)
    ax_d.set_yticks(y, [label(sc, short=True) for sc in SCENARIOS])
    ax_d.tick_params(axis='y', length=0, labelsize=5.5)
    ax_d.spines['left'].set_visible(False)
    ax_d.set_xlabel('Coordination saving at 75% of cap (USD/MWh)')
    guides(ax_d)
    ax_d.legend(handles=[Patch(facecolor='#8A8A8A', label='IQR; whiskers P10–P90'),
                         Line2D([], [], marker='|', color='none', markeredgecolor='#8B253E', markersize=5,
                                label='Minimum across cells'),
                         Line2D([], [], marker='o', color='none', markerfacecolor='white', markeredgecolor='#333333',
                                markersize=3, label='Median without inference')],
                loc='lower right', bbox_to_anchor=(1.02, 0), ncol=1, fontsize=5.1, handlelength=1, borderaxespad=.2)
    n_all = len(p)
    panel(fig, 'd', .37, .555, f'All {n_all:,} paired cases remain positive')
    ax_d.text(.99, 1.01, f'Global minimum {p.saving_usd_mwh.min():.2f} USD/MWh', transform=ax_d.transAxes,
              ha='right', va='bottom', fontsize=5.3, color=NOTE)

    # ---------------- e 协同节省 / 卖推理净收入
    sl = sell()
    sl = sl[sl.tier.isin(['real_25', 'real_50', 'real_75'])]
    m = sl.groupby(['scenario', 'tier'])[['coord_usd_mwh', 'net_usd_mwh']].median()
    share = (m.coord_usd_mwh / m.net_usd_mwh * 100).unstack()
    ax_e = fig.add_axes([.775, .1, .2, .38])
    ye = np.arange(len(GPU_SCENARIOS))[::-1]
    for yi, sc in zip(ye, GPU_SCENARIOS):
        c = shade(sc)
        vals = share.loc[sc, ['real_25', 'real_50', 'real_75']].to_numpy()
        ax_e.plot([vals.min(), vals.max()], [yi, yi], color='#D5D5D5', lw=.8, zorder=1)
        for v, mk, face in zip(vals, ['o', 'o', 'o'], ['white', mpl.colors.to_hex(np.array(c) * .5 + .5), c]):
            ax_e.scatter(v, yi, s=12, marker=mk, facecolors=face, edgecolors=c, lw=.6, zorder=3)
    ax_e.set_xscale('log')
    ax_e.set_xlim(.5, 15)
    ax_e.set_xticks([.5, 1, 2, 5, 10], ['0.5', '1', '2', '5', '10'])
    ax_e.minorticks_off()
    ax_e.set_ylim(-.6, len(ye) - .4)
    ax_e.set_yticks(ye, [label(sc, short=True) for sc in GPU_SCENARIOS])
    ax_e.tick_params(axis='y', length=0, labelsize=5.5)
    ax_e.spines['left'].set_visible(False)
    ax_e.set_xlabel('Coordination saving / net inference revenue\n(%, log scale)')
    guides(ax_e)
    ax_e.legend(handles=[Line2D([], [], marker='o', color='none', markerfacecolor=f, markeredgecolor='#666666',
                                markersize=3.2, label=t)
                         for f, t in [('white', '25% of cap'), ('#B3B3B3', '50%'), ('#666666', '75%')]],
                loc='lower center', bbox_to_anchor=(.45, 1.0), ncol=3, fontsize=5.3, handletextpad=.2,
                columnspacing=.6)
    panel(fig, 'e', .705, .555, 'Small next to inference revenue')

    export(fig, 'fig3')

    csv(ret.assign(label=ret.scenario.map(label)), 'fig3a')
    csv(dtrn.reset_index()[['grid_id', 'lat', 'lon', 'iso3', 'continent', 'retention', 'wind_share',
                            'cfe_hourly_joint_I0', 'saving_usd_mwh']], 'fig3b_map_D-TRN_75')
    csv(lat_rows, 'fig3b_latitude')
    csv(pd.DataFrame({'scenario': list(rho), 'spearman_retention_wind_share_75': list(rho.values())}), 'fig3c')
    csv(stat.reset_index().assign(median_without_inference=lambda d: d.scenario.map(i0)), 'fig3d')
    csv(share.reset_index(), 'fig3e')
    return ret, rho, stat, share


if __name__ == '__main__':
    ret, rho, stat, share = main()
    print(ret[ret.intensity_of_max.eq(.75)].set_index('scenario')['median'].round(2).to_dict())
    print({k: round(v, 2) for k, v in rho.items()})
    print(stat[['min', '10%', '50%']].round(2).T.to_string())
    print(share.round(2).to_string())
