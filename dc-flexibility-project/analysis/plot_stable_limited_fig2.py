"""图2：闲置算力变现的收益远超电力调度（第二节）。

a 9 个 GPU 情境的量级阶梯（对数轴，USD/MWh）：协同节省（I=0）→ 上限 25/50/75% → 真实上限 → 理想全卖；
b 真实需求形状下能卖出的比例（真实上限 / 理想，半环）与可承接上限；
c 训练机群（GPU-heavy）真实上限净收入相对全球中位数的偏离（%）+ 纬度统计条；
d 峰谷计价与地点对净收入的影响。
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Wedge
import numpy as np
import pandas as pd

from stable_limited_figs_common import (COLORS, GPU_SCENARIOS, NOTE, RESIDUAL, USES, colorbar, color, csv, draw_map,
                                        export, guides, label, latitude_strip, paired, panel, setup, shade)

TIERS = ['real_25', 'real_50', 'real_75', 'real_max', 'ideal']
TIER_NAME = {'real_25': '25% of cap', 'real_50': '50% of cap', 'real_75': '75% of cap',
             'real_max': 'Realistic cap', 'ideal': 'Ideal (all idle sold)'}


def main():
    setup()
    s = paired()
    coord = s[s.intensity_of_max.eq(0) & s.scenario.isin(GPU_SCENARIOS)].groupby('scenario').saving_usd_mwh
    from stable_limited_figs_common import sell
    sl = sell()
    med = sl.groupby(['scenario', 'tier']).net_usd_mwh.median().unstack()[TIERS]
    p10 = sl.groupby(['scenario', 'tier']).net_usd_mwh.quantile(.1).unstack()[TIERS]
    p90 = sl.groupby(['scenario', 'tier']).net_usd_mwh.quantile(.9).unstack()[TIERS]
    cap = pd.read_csv(RESIDUAL / 'max-intensity.csv').set_index('scenario')

    fig = plt.figure(figsize=(180 / 25.4, 160 / 25.4))

    # ---------------- a 量级阶梯
    ax = fig.add_axes([.175, .565, .47, .375])
    y = np.arange(len(GPU_SCENARIOS))[::-1]
    for yi, sc in zip(y, GPU_SCENARIOS):
        c = shade(sc)
        cm = coord.median()[sc]
        ax.plot([cm, med.loc[sc, 'ideal']], [yi, yi], color='#DADADA', lw=.8, zorder=1)
        ax.plot([coord.quantile(.1)[sc], coord.quantile(.9)[sc]], [yi, yi], color='#555555', lw=.7, zorder=2)
        ax.scatter(cm, yi, marker='D', s=13, color='#555555', zorder=3)
        for t, size in zip(['real_25', 'real_50', 'real_75'], [7, 9, 11]):
            ax.scatter(med.loc[sc, t], yi, s=size, facecolors='white', edgecolors=c, linewidths=.7, zorder=3)
        ax.scatter(med.loc[sc, 'real_max'], yi, s=20, color=c, zorder=4)
        ax.scatter(med.loc[sc, 'ideal'], yi, s=26, marker='s', facecolors='white', edgecolors=c, linewidths=.9,
                   zorder=4)
        ax.text(med.loc[sc, 'ideal'] * 1.35, yi, f'{med.loc[sc, "ideal"] / cm:,.0f}×', va='center', fontsize=5.3,
                color=NOTE)
    ax.set_xscale('log')
    ax.set_xlim(1, 2.5e4)
    ax.set_xticks([1, 10, 100, 1000, 1e4], ['1', '10', '100', '1,000', '10,000'])
    ax.set_ylim(-.6, len(y) - .4)
    ax.set_yticks(y, [label(sc) for sc in GPU_SCENARIOS])
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('USD per MWh of data-centre electricity (log scale)')
    guides(ax)
    ax.axvspan(1, 20, color='#F3F3F3', zorder=0, lw=0)
    ax.text(4.5, len(y) - .55, 'Power-side\ncoordination', ha='center', va='top', fontsize=5.5, color=NOTE)
    ax.text(2.2e4, len(y) - .55, 'Ideal ÷\ncoordination', ha='right', va='top', fontsize=5.3, color=NOTE)
    handles = [Line2D([], [], marker='D', color='#555555', lw=.7, markersize=3, label='Coordination saving (P10–P90)'),
               Line2D([], [], marker='o', color='none', markerfacecolor='white', markeredgecolor='#666666',
                      markersize=3, label='Sell 25/50/75% of cap'),
               Line2D([], [], marker='o', color='none', markerfacecolor='#666666', markeredgecolor='#666666',
                      markersize=3.5, label='Sell realistic cap'),
               Line2D([], [], marker='s', color='none', markerfacecolor='white', markeredgecolor='#666666',
                      markersize=3.8, label='Sell all idle GPUs (ideal)')]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.45, -.13), ncol=4, handletextpad=.3,
              columnspacing=.9)
    panel(fig, 'a', .012, .985, 'Selling idle GPU-hours at 2.9 USD/GPU-h earns about two orders of magnitude more '
                                'than coordination (net revenue, global median)')

    # ---------------- b 半环：真实上限/理想
    ratio = (med.real_max / med.ideal * 100)
    groups = [('TRN', ['B-TRN', 'C-TRN', 'D-TRN']), ('SAI', ['B-SAI', 'C-SAI', 'D-SAI']),
              ('INF', ['B-INF', 'C-INF', 'D-INF'])]
    ax_b = fig.add_axes([.665, .55, .33, .375])
    ax_b.set_xlim(0, 3.2)
    ax_b.set_ylim(-.15, 3.25)
    ax_b.set_aspect('equal')
    ax_b.axis('off')
    for i, (u, scs) in enumerate(groups):
        cy = 2.62 - i * 1.18
        cx = .5
        r = ratio[scs]
        mid = r.median()
        ax_b.add_patch(Wedge((cx, cy), .5, 0, 360, width=.16, facecolor='#EDEDED', edgecolor='none'))
        ax_b.add_patch(Wedge((cx, cy), .5, 90 - 360 * mid / 100, 90, width=.16, facecolor=COLORS[u], edgecolor='none'))
        ax_b.text(cx, cy, f'{mid:.0f}%', ha='center', va='center', fontsize=7, fontweight='bold', color=COLORS[u])
        k = cap.loc[scs, 'max_intensity']
        rng = f'{r.min():.0f}–{r.max():.0f}%' if r.max() - r.min() >= 1 else f'{r.min():.0f}%'
        kr = f'{k.min():.2f}–{k.max():.2f}' if k.max() - k.min() >= .005 else f'{k.min():.2f}'
        ax_b.text(1.1, cy + .22, USES[u], fontsize=6.5, fontweight='bold', va='center')
        ax_b.text(1.1, cy - .02, f'Realistic / ideal: {rng}', fontsize=5.7, va='center')
        ax_b.text(1.1, cy - .24, f'Cap: {kr}× mean idle power', fontsize=5.7, va='center', color=NOTE)
    panel(fig, 'b', .675, .962, 'Share of idle compute that can be sold')
    ax_b.text(.0, -.1, 'Ring: median of CPU-heavy, balanced and GPU-heavy fleets.\nCap: set by hourly real inference '
                       'demand and\nthe deadlines of existing jobs.', fontsize=5.2, color=NOTE, va='top')

    # ---------------- c 地点敏感性地图
    d = sl[sl.scenario.eq('D-TRN') & sl.tier.eq('real_max')].set_index('grid_id')
    dev = (d.net_usd_year / d.net_usd_year.median() - 1) * 100
    cmap = mpl.colormaps['RdBu_r']
    norm = mpl.colors.TwoSlopeNorm(0, -1.5, 1.5)
    ax_map, mapped = draw_map(fig, [.02, .1, .56, .33], dev, cmap, norm)
    frames = []
    for sc in ['D-TRN', 'D-SAI', 'D-INF']:
        x = sl[sl.scenario.eq(sc) & sl.tier.eq('real_max')]
        frames.append((label(sc, short=True), color(sc),
                       pd.DataFrame({'lat': x.lat, 'value': (x.net_usd_year / x.net_usd_year.median() - 1) * 100}),
                       sc == 'D-TRN'))
    ax_lat, lat_rows = latitude_strip(fig, ax_map, .07, frames, 'Deviation (%)', xlim=(-1.5, 1.5))
    ax_lat.set_xticks([-1, 0, 1])
    ax_lat.axvline(0, color='#999999', lw=.4)
    ax_lat.legend(loc='upper left', bbox_to_anchor=(-.1, -.17), fontsize=5.2, handlelength=1, labelspacing=.25)
    colorbar(fig, [.12, .112, .34, .01], cmap, norm, [-1.5, -1, -.5, 0, .5, 1, 1.5],
             'Net revenue vs global median, LLM training · GPU-heavy, realistic cap (%)', extend='both')
    panel(fig, 'c', .012, .465, 'Location changes net revenue by about ±1% (only the extra energy cost varies)')

    # ---------------- d 峰谷计价 + 地点离散度
    ax_d = fig.add_axes([.8, .15, .18, .23])
    tou = sl.groupby(['scenario', 'tier'])[['net_usd_year', 'net_tou_usd_year']].median()
    tou = (tou.net_tou_usd_year / tou.net_usd_year - 1).unstack() * 100
    g = sl.groupby(['scenario', 'tier']).net_usd_year
    spread = ((g.quantile(.9) - g.quantile(.1)) / g.median() * 100).unstack()
    yy = np.arange(len(GPU_SCENARIOS))[::-1]
    for yi, sc in zip(yy, GPU_SCENARIOS):
        c = shade(sc)
        ax_d.barh(yi + .17, tou.loc[sc, 'real_max'], height=.3, color=c, lw=0)
        ax_d.barh(yi - .17, spread.loc[sc, 'real_max'], height=.3, facecolor='white', edgecolor=c, lw=.6)
    ax_d.axvline(0, color='#555555', lw=.5)
    ax_d.set_xlim(-3.2, 2)
    ax_d.set_xticks([-3, -2, -1, 0, 1, 2])
    ax_d.set_ylim(-.6, len(yy) - .4)
    ax_d.set_yticks(yy, [label(sc, short=True) for sc in GPU_SCENARIOS])
    ax_d.tick_params(axis='y', length=0, labelsize=5.5)
    ax_d.spines['left'].set_visible(False)
    ax_d.set_xlabel('Change in net revenue (%)')
    guides(ax_d)
    ax_d.legend(handles=[Patch(facecolor='#888888', label='Peak/off-peak pricing\n(same mean price)'),
                         Patch(facecolor='white', edgecolor='#888888', label='P10–P90 spread\nacross locations')],
                loc='lower center', bbox_to_anchor=(.35, 1.0), ncol=1, handlelength=1, fontsize=5.3)
    panel(fig, 'd', .74, .465, 'Pricing and location')

    export(fig, 'fig2')

    out = med.add_prefix('median_').join(p10.add_prefix('p10_')).join(p90.add_prefix('p90_'))
    out = out.join(coord.median().rename('coord_saving_median')).join(coord.quantile(.1).rename('coord_saving_p10'))
    out = out.join(coord.quantile(.9).rename('coord_saving_p90'))
    out['ideal_over_coord'] = out.median_ideal / out.coord_saving_median
    csv(out.reset_index(), 'fig2a')
    csv(pd.DataFrame({'scenario': ratio.index, 'realistic_over_ideal_pct': ratio.values,
                      'max_intensity': cap.loc[ratio.index, 'max_intensity'].values}), 'fig2b')
    csv(d.reset_index()[['grid_id', 'lat', 'lon', 'net_usd_year']].assign(deviation_pct=dev.values), 'fig2c_map_D-TRN')
    csv(lat_rows, 'fig2c_latitude')
    csv(pd.DataFrame({'tou_change_pct': tou.real_max, 'location_spread_p10_p90_pct': spread.real_max}).reset_index(),
        'fig2d')
    return out, ratio, tou, spread


if __name__ == '__main__':
    out, ratio, tou, spread = main()
    print(out[['coord_saving_median', 'median_real_max', 'median_ideal', 'ideal_over_coord']].round(1).to_string())
    print(ratio.round(1).to_dict())
