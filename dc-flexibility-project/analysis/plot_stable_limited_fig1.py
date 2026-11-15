"""图1：算力中心存在可观的调节余量，但单靠电力调度收益有限（第一节）。

a 训练机群（GPU-heavy）协同节省全球地图（USD/MWh，I=0，规划边界）+ 经度、纬度统计条；
b 11 情境峰值余量（环形柱图）；c 协同节省 USD/MWh 网格分布（山脊图）；
d 两种分母下的节省占比（斜率图：占供能成本 → 占机群+供能总成本）；
e 地区差异归因：不做协同时的市电占比 × 风电装机占比，颜色为训练机群节省中位数（六边形分箱）。
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr

from stable_limited_figs_common import (COLORS, GROUP, INNER, LAND, REPRESENTATIVES, SCENARIOS, SECTION1, baseline,
                                        colorbar, color, csv, draw_map, export, guides, label, latitude_strip,
                                        longitude_strip, paired, panel, part1_summary, scenario_axis, setup, shade,
                                        truncated, use)


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


def main():
    setup()
    p = paired()
    p0 = p[p.intensity_of_max.eq(0)].copy()
    shares = section1_shares()
    head = part1_summary().loc[SCENARIOS].reset_index()
    b0 = baseline()
    p0 = p0.merge(b0[['grid_id', 'scenario', 'grid_mwh0']], on=['grid_id', 'scenario'])
    p0['grid_share'] = 100 * p0.grid_mwh0 / p0.facility_mwh0

    fig = plt.figure(figsize=(180 / 25.4, 205 / 25.4))

    # ---------------- a 地图 + 经度、纬度条
    cmap = truncated('YlGnBu', .1)
    norm = mpl.colors.Normalize(0, 30)
    dtrn = p0[p0.scenario.eq('D-TRN')].set_index('grid_id')
    ax_map, _ = draw_map(fig, [.035, .585, .74, .33], dtrn.saving_usd_mwh, cmap, norm)
    frames = [(label(s, short=True), color(s), p0.loc[p0.scenario.eq(s), ['lat', 'saving_usd_mwh']]
               .rename(columns={'saving_usd_mwh': 'value'}), s == 'D-TRN') for s in REPRESENTATIVES]
    ax_lat, lat_rows = latitude_strip(fig, ax_map, .075, frames, 'USD/MWh', xlim=(0, 30))
    ax_lat.set_xticks([0, 10, 20, 30])
    lon_frames = [(n, c, p0.loc[p0.scenario.eq(s), ['lon', 'saving_usd_mwh']].rename(columns={'saving_usd_mwh': 'value'}),
                   s == 'D-TRN') for (n, c, _, _), s in zip(frames, REPRESENTATIVES)]
    ax_lon = longitude_strip(fig, ax_map, .045, lon_frames, 'USD/MWh', ylim=(0, 30))
    ax_lon.set_yticks([0, 15, 30])
    handles = [Line2D([], [], color=COLORS[use(s)], lw=1, label=GROUP[use(s)] + ('' if s[0] == 'A' else '\n(GPU-heavy)'))
               for s in REPRESENTATIVES] + [Patch(facecolor=LAND, label='No eligible\ngrid cell')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.875, .99), fontsize=5.4, handlelength=1.2,
               labelspacing=.45, borderaxespad=0)
    colorbar(fig, [.2, .607, .38, .009], cmap, norm, [0, 5, 10, 15, 20, 25, 30],
             'Coordination saving, LLM training · GPU-heavy (USD/MWh)', extend='max')
    panel(fig, 'a', .012, .99)

    # ---------------- b 环形柱图：峰值余量
    ax_b = fig.add_axes([.07, .3, .22, .2], projection='polar')
    order = SCENARIOS
    gap = .5
    theta, t = [], 0.
    for i, s in enumerate(order):
        if i and use(s) != use(order[i - 1]):
            t += gap
        theta.append(t)
        t += 1
    theta = np.array(theta) / (t + gap) * 2 * np.pi
    width = 2 * np.pi / (t + gap) * .82
    headroom = (100 - head.set_index('scenario').mean_over_K_pct).loc[order]
    ax_b.bar(theta, headroom.to_numpy(), width=width, bottom=18, color=[shade(s) for s in order], lw=0)
    for th, s, v in zip(theta, order, headroom):
        # 柱外沿径向写“配置 + 数值”，左半圈翻转 180° 保证可读
        disp = (90 - np.degrees(th)) % 360
        left = 90 < disp < 270
        name = INNER[s[0]] if s[0] != 'A' else GROUP[use(s)]
        ax_b.text(th, 18 + v + 3, f'{name}  {v:.0f}' if not left else f'{v:.0f}  {name}',
                  ha='right' if left else 'left', va='center', rotation=disp + (180 if left else 0),
                  rotation_mode='anchor', fontsize=5.2, color=COLORS[use(s)])
    ax_b.set_theta_zero_location('N')
    ax_b.set_theta_direction(-1)
    ax_b.set_ylim(0, 72)
    ax_b.set_yticks([18 + 20, 18 + 40], [])
    ax_b.legend(handles=[Patch(facecolor=COLORS[u], label=GROUP[u]) for u in GROUP], loc='upper center',
                bbox_to_anchor=(.5, -.13), ncol=3, fontsize=5.3, handlelength=.9, columnspacing=.6)
    ax_b.set_xticks([])
    ax_b.grid(color='#DDDDDD', lw=.4)
    ax_b.spines['polar'].set_visible(False)
    ax_b.text(0, 0, 'Headroom\n(%)', ha='center', va='center', fontsize=5.6)
    panel(fig, 'b', .012, .555)

    # ---------------- c 山脊图：USD/MWh
    ax_c = fig.add_axes([.555, .29, .425, .245])
    xs = np.linspace(0, 32, 400)
    step = 1.0
    ys = np.arange(len(SCENARIOS))[::-1] * step
    ridge = []
    for yi, s in zip(ys, SCENARIOS):
        v = p0.loc[p0.scenario.eq(s), 'saving_usd_mwh'].to_numpy()
        dens = gaussian_kde(v, bw_method=.25)(xs)
        dens = dens / dens.max() * 1.55
        c = shade(s)
        ax_c.fill_between(xs, yi, yi + dens, color=c, alpha=.85, lw=0, zorder=20 - yi)
        ax_c.plot(xs, yi + dens, color='white', lw=.5, zorder=20 - yi)
        med = np.median(v)
        ax_c.plot([med, med], [yi, yi + np.interp(med, xs, dens)], color='#1A1A1A', lw=.7, zorder=21 - yi)
        ridge.append(pd.DataFrame({'scenario': s, 'x_usd_mwh': xs, 'density_scaled': dens}))
    ax_c.set_xlim(0, 32)
    ax_c.set_ylim(-.2, ys.max() + 1.8)
    ax_c.spines['left'].set_visible(False)
    scenario_axis(ax_c, SCENARIOS, ys + .25, outer=-.27)
    ax_c.set_xlabel('Coordination saving (USD per MWh of data-centre electricity)')
    supply = (p0.supply_cost0_usd / p0.facility_mwh0).groupby(p0.scenario).median()
    guides(ax_c)
    panel(fig, 'c', .36, .555)

    # ---------------- d 斜率图：两种分母
    ax_d = fig.add_axes([.09, .05, .33, .155])
    qs = shares.groupby('scenario').share_supply_pct.median().loc[SCENARIOS]
    qt = shares.groupby('scenario').share_total_pct.median().loc[SCENARIOS]
    for s in SCENARIOS:
        ax_d.plot([0, 1], [qs[s], qt[s]], color=shade(s), lw=1, marker='o', markersize=2.6, solid_capstyle='round')
    ax_d.set_yscale('log')
    ax_d.set_ylim(.07, 20)
    ax_d.set_yticks([.1, .3, 1, 3, 10], ['0.1', '0.3', '1', '3', '10'])
    ax_d.minorticks_off()
    ax_d.axhline(1, color='#8B253E', lw=.6, ls=(0, (3, 2)), zorder=0)
    ax_d.set_xlim(-.15, 1.15)
    ax_d.set_xticks([0, 1], ['Share of energy-\nsupply cost', 'Share of fleet + energy-\nsupply cost'])
    ax_d.set_ylabel('Saving / annual cost (%, median)')
    ax_d.tick_params(axis='x', length=0)
    ax_d.legend(handles=[Line2D([], [], color=COLORS[u], lw=1, label=GROUP[u]) for u in GROUP],
                loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=5.4, handlelength=1.2, labelspacing=.35)
    guides(ax_d, 'y')
    panel(fig, 'd', .012, .225)

    # ---------------- e 归因：市电占比 × 风电占比
    ax_e = fig.add_axes([.69, .05, .22, .155])
    d = p0[p0.scenario.eq('D-TRN')]
    hb = ax_e.hexbin(d.grid_share, d.wind_share * 100, C=d.saving_usd_mwh, reduce_C_function=np.median,
                     gridsize=(18, 10), cmap=cmap, norm=norm, mincnt=3, lw=.15, edgecolors='white',
                     extent=(0, 100, 0, 100))
    ax_e.set_xlim(0, 100)
    ax_e.set_ylim(0, 100)
    ax_e.set_xlabel('Grid electricity share without\ncoordination (%)')
    ax_e.set_ylabel('Wind share of optimal\nwind + PV capacity (%)')
    rho_grid = spearmanr(d.grid_share, d.saving_usd_mwh)[0]
    rho_wind = spearmanr(d.wind_share, d.saving_usd_mwh)[0]
    cax = fig.add_axes([.925, .05, .008, .155])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend='max')
    cb.outline.set_linewidth(.3)
    cax.tick_params(labelsize=5, length=1.5, width=.4, pad=1)
    cb.set_label('Median saving (USD/MWh)', fontsize=5.4, labelpad=2)
    ax_e.text(.97, .96, f'ρ(grid) = {rho_grid:.2f}\nρ(wind) = {rho_wind:.2f}', transform=ax_e.transAxes,
              ha='right', va='top', fontsize=5.6)
    panel(fig, 'e', .6, .225)

    export(fig, 'fig1')

    csv(dtrn.reset_index()[['grid_id', 'lat', 'lon', 'iso3', 'continent', 'saving_usd_mwh', 'V_usd', 'facility_mwh0',
                            'supply_cost0_usd']], 'fig1a_map_D-TRN')
    csv(lat_rows, 'fig1a_latitude')
    csv(pd.DataFrame({'scenario': order, 'headroom_pct': headroom.to_numpy()}), 'fig1b')
    csv(pd.concat(ridge).assign(median=lambda x: x.scenario.map(p0.groupby('scenario').saving_usd_mwh.median()),
                                supply_usd_mwh=lambda x: x.scenario.map(supply)), 'fig1c')
    csv(pd.DataFrame({'scenario': SCENARIOS, 'share_supply_pct_median': qs.to_numpy(),
                      'share_total_pct_median': qt.to_numpy()}), 'fig1d')
    csv(d[['grid_id', 'lat', 'lon', 'grid_share', 'wind_share', 'saving_usd_mwh']], 'fig1e')
    return qs, qt, rho_grid, rho_wind


if __name__ == '__main__':
    print(main())
