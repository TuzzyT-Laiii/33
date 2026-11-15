"""图1：算力中心存在可观的调节余量，但单靠电力调度收益有限（第一节）。

a 训练机群（GPU-heavy）协同节省全球地图（USD/MWh，I=0，规划边界）+ 纬度统计条（五个代表情境）；
b 11 情境峰值余量；c 协同节省 USD/MWh 网格分布；d 两种分母下的节省占比（供能成本 vs 机群+供能总成本）。
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from stable_limited_figs_common import (LAND, NOTE, truncated, REPRESENTATIVES, SCENARIOS, SECTION1, colorbar, color, csv, draw_map,
                                        export, guides, label, latitude_strip, paired, panel, part1_summary, setup,
                                        shade)


def section1_shares():
    """第一节原生窗口（PR #45 grid-summary，规划边界、经济最优）：节省占供能成本与占总成本。"""
    g = pd.read_parquet(SECTION1 / 'grid-summary.parquet',
                        columns=['grid_id', 'scenario', 'alpha', 'boundary', 'baseline_status', 'flex_status',
                                 'baseline_cost_usd', 'flex_cost_usd', 'savings_pct'])
    g = g[g.alpha.isna() & g.boundary.eq('planning') & g.baseline_status.eq('optimal') & g.flex_status.eq('optimal')]
    assert g.groupby('scenario').size().eq(4842).all()
    s = part1_summary()
    year = 8760 / g.scenario.map(s.hours)
    saving = (g.baseline_cost_usd - g.flex_cost_usd) * year
    total = g.scenario.map(s.fleet_annual_musd) * 1e6 + g.baseline_cost_usd * year
    return g.assign(share_supply_pct=g.savings_pct, share_total_pct=100 * saving / total)[
        ['grid_id', 'scenario', 'share_supply_pct', 'share_total_pct']]


def quantiles(frame, col):
    g = frame.groupby('scenario')[col]
    return pd.DataFrame({'p10': g.quantile(.1), 'q25': g.quantile(.25), 'median': g.median(),
                         'q75': g.quantile(.75), 'p90': g.quantile(.9), 'min': g.min(), 'max': g.max(),
                         'n': g.size()}).loc[SCENARIOS]


def rows(ax, y, labels):
    ax.set_ylim(-.7, len(y) - .3)
    ax.set_yticks(y, labels if labels is not None else [])
    ax.tick_params(axis='y', length=0, pad=2)
    ax.spines['left'].set_visible(False)


def main():
    setup()
    p = paired()
    p0 = p[p.intensity_of_max.eq(0)].copy()
    shares = section1_shares()
    head = part1_summary().loc[SCENARIOS].reset_index()

    fig = plt.figure(figsize=(180 / 25.4, 158 / 25.4))

    # ---------------- a 地图 + 纬度条
    cmap = truncated('YlGnBu', .1)
    norm = mpl.colors.Normalize(0, 30)
    dtrn = p0[p0.scenario.eq('D-TRN')].set_index('grid_id')
    ax_map, mapped = draw_map(fig, [.03, .5, .74, .455], dtrn.saving_usd_mwh, cmap, norm)
    frames = [(label(s, short=True), color(s), p0.loc[p0.scenario.eq(s), ['lat', 'saving_usd_mwh']]
               .rename(columns={'saving_usd_mwh': 'value'}), s == 'D-TRN') for s in REPRESENTATIVES]
    ax_lat, lat_rows = latitude_strip(fig, ax_map, .085, frames, 'Saving (USD/MWh)', xlim=(0, 30))
    ax_lat.set_xticks([0, 10, 20, 30])
    handles, names = ax_lat.get_legend_handles_labels()
    handles.append(Patch(facecolor=LAND, edgecolor='none'))
    names.append('No eligible\ngrid cell (map)')
    ax_lat.legend(handles, names, loc='upper left', bbox_to_anchor=(1.32, 1.0), handlelength=1.2, labelspacing=.35, fontsize=5.5,
                  title='Latitude median\n(4° bands)', title_fontsize=5.5, alignment='left')
    colorbar(fig, [.2, .54, .36, .011], cmap, norm, [0, 5, 10, 15, 20, 25, 30],
             'Coordination saving, LLM training · GPU-heavy (USD per MWh of data-centre electricity)', extend='max')
    panel(fig, 'a', .012, .985, 'Joint compute–power scheduling saves 3.5–15.4 USD/MWh (planning boundary, '
                                 'no external inference)')
    fig.text(.035, .555, f'{len(dtrn):,} grid cells\n{mapped:,} land-clipped 2° blocks', fontsize=5.3, color=NOTE,
             va='bottom')

    # ---------------- b 峰值余量
    y = np.arange(len(SCENARIOS))[::-1]
    ax_b = fig.add_axes([.165, .105, .17, .3])
    ax_c = fig.add_axes([.415, .105, .225, .3])
    ax_d = fig.add_axes([.725, .105, .255, .3])
    for ax in (ax_b, ax_c, ax_d):
        rows(ax, y, [label(s) for s in SCENARIOS] if ax is ax_b else None)
    for yi, s, u in zip(y, SCENARIOS, head.mean_over_K_pct):
        ax_b.barh(yi, u, height=.62, color=shade(s), lw=0)
        ax_b.barh(yi, 100 - u, left=u, height=.62, color='#F2F2F2', edgecolor='#A0A0A0', lw=.35)
        ax_b.text((100 + u) / 2, yi, f'{100 - u:.0f}', ha='center', va='center', fontsize=5.5)
    ax_b.set_xlim(0, 100)
    ax_b.set_xticks([0, 25, 50, 75, 100])
    ax_b.set_xlabel('Share of task-power peak K (%)')
    ax_b.legend(handles=[Patch(facecolor='#8A8A8A', label='Mean task power'),
                         Patch(facecolor='#F2F2F2', edgecolor='#A0A0A0', label='Headroom')],
                loc='upper center', bbox_to_anchor=(.45, -.16), ncol=2, handlelength=.9, columnspacing=.8)
    panel(fig, 'b', .012, .455, 'Headroom below the observed peak: 27–52%')

    # ---------------- c USD/MWh 分布
    q = quantiles(p0, 'saving_usd_mwh')
    for yi, s in zip(y, SCENARIOS):
        r = q.loc[s]
        c = shade(s)
        ax_c.plot([r.p10, r.p90], [yi, yi], color=c, lw=.7, solid_capstyle='butt', zorder=2)
        ax_c.add_patch(mpl.patches.Rectangle((r.q25, yi - .27), r.q75 - r.q25, .54, facecolor=c, edgecolor=c,
                                             lw=.5, zorder=3))
        ax_c.plot([r['median']] * 2, [yi - .27, yi + .27], color='#1A1A1A', lw=.8, zorder=4)
        ax_c.text(r.p90 + .6, yi, f'{r["median"]:.1f}', va='center', fontsize=5.3, color='#333333')
    ax_c.set_xlim(0, 24)
    ax_c.set_xticks([0, 5, 10, 15, 20])
    ax_c.set_xlabel('Coordination saving (USD/MWh)')
    guides(ax_c)
    supply = (p0.supply_cost0_usd / p0.facility_mwh0).groupby(p0.scenario).median()
    ax_c.text(.98, 1.03, f'Supply cost without coordination: {supply.min():.0f}–{supply.max():.0f} USD/MWh',
              transform=ax_c.transAxes, ha='right', va='bottom', fontsize=5.3, color=NOTE)
    ax_c.legend(handles=[Patch(facecolor='#8A8A8A', label='Interquartile range'),
                         Line2D([], [], color='#8A8A8A', lw=.7, label='P10–P90'),
                         Line2D([], [], color='#1A1A1A', lw=.8, label='Median')],
                loc='upper center', bbox_to_anchor=(.5, -.16), ncol=3, handlelength=1, columnspacing=.7)
    panel(fig, 'c', .37, .455, 'Saving per MWh across 4,842 grid cells')

    # ---------------- d 两种分母
    qs = quantiles(shares, 'share_supply_pct')
    qt = quantiles(shares, 'share_total_pct')
    for yi, s in zip(y, SCENARIOS):
        c = shade(s)
        ax_d.plot([qt.loc[s, 'median'], qs.loc[s, 'median']], [yi, yi], color='#D0D0D0', lw=.8, zorder=1)
        for qq, off, face in [(qs, .14, c), (qt, -.14, 'white')]:
            ax_d.plot([qq.loc[s, 'p10'], qq.loc[s, 'p90']], [yi + off] * 2, color=c, lw=.6, zorder=2)
            ax_d.scatter(qq.loc[s, 'median'], yi + off, s=13, marker='o', facecolors=face, edgecolors=c,
                         linewidths=.7, zorder=3)
    ax_d.axvline(1, color='#8B253E', lw=.6, ls=(0, (3, 2)), zorder=0)
    ax_d.text(1.07, -.55, '1%', color='#8B253E', fontsize=5.5, va='center')
    ax_d.set_xscale('log')
    ax_d.set_xlim(.04, 40)
    ax_d.set_xticks([.1, 1, 10], ['0.1', '1', '10'])
    ax_d.set_xlabel('Saving as share of annual cost (%, log scale)')
    guides(ax_d)
    ax_d.legend(handles=[Line2D([], [], marker='o', color='none', markerfacecolor='#777777', markeredgecolor='#777777',
                                markersize=3.5, label='Energy supply'),
                         Line2D([], [], marker='o', color='none', markerfacecolor='white', markeredgecolor='#777777',
                                markersize=3.5, label='Fleet + energy supply')],
                loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2, handletextpad=.3, columnspacing=.8,
                borderaxespad=.2)
    panel(fig, 'd', .69, .455, 'Below 1% of total cost once the fleet is counted')
    ax_d.text(.5, -.17, f'Medians {qs["median"].min():.1f}–{qs["median"].max():.1f}% (energy supply) vs '
              f'{qt["median"].min():.2f}–{qt["median"].max():.2f}% (total);\nmax cell {shares.share_total_pct.max():.2f}%. '
              'Bars: P10–P90 across grid cells', transform=ax_d.transAxes, fontsize=5.3, color=NOTE, va='top', ha='center')

    export(fig, 'fig1')

    csv(dtrn.reset_index()[['grid_id', 'lat', 'lon', 'iso3', 'continent', 'saving_usd_mwh', 'V_usd', 'facility_mwh0',
                            'supply_cost0_usd']], 'fig1a_map_D-TRN')
    csv(lat_rows, 'fig1a_latitude')
    csv(head[['scenario', 'hours', 'K_mw', 'mean_over_K_pct']].assign(headroom_pct=100 - head.mean_over_K_pct,
                                                                       label=head.scenario.map(label)), 'fig1b')
    csv(q.reset_index().assign(supply_usd_mwh=lambda d: d.scenario.map(supply)), 'fig1c')
    csv(pd.concat([qs.reset_index().assign(denominator='energy_supply'),
                   qt.reset_index().assign(denominator='fleet_plus_energy_supply')]), 'fig1d')
    return q, qs, qt


if __name__ == '__main__':
    q, qs, qt = main()
    print(q[['median', 'p10']].round(2).T.to_string())
    print(qs['median'].round(2).to_dict(), qt['median'].round(2).to_dict())
