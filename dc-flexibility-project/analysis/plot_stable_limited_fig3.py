"""图3：真实推理需求下，算电协同仍带来稳定收益（第三节）。

a 推理加到上限 75% 时，每个网格 11 类机群中最低的协同节省（六边形格网地图，USD/MWh，0.5 一档离散色），
  海域内嵌直方图兼色标，下方为 14 个合并分区的箱线图；
b 实施成本存活曲线：实施成本为横轴时仍有净收益的网格–机群组合占比；
c 争用与互补：被推理占用的闲置比例 vs 协同保留率，虚线为完全争用（保留率 = 1 − 占用比例）；
d 保留率分解：转移电量比 × 每 MWh 转移电量价值比，背景为等保留率线；
e 剩余收益的地区差异：训练机群保留率 vs 风电装机占比（点色为不做协同时的市电占比），右侧为各机群的秩相关系数。
"""
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stable_limited_figs_common import (COLORS, DARK, GROUP, INNER, INPUT, RESIDUAL, SCENARIOS, SEQ, baseline, csv,
                                        export, guides, hex_mosaic, land_axes, map_layers, paired, panel, setup,
                                        shade, use)

LEVELS = [0, .25, .5, .75]
USES = ['HPC', 'CLD', 'SAI', 'TRN', 'INF']
MARKS = {'A': 'D', 'B': 'o', 'C': 's', 'D': '^'}

H = 212                      # 画布高度（mm）；c–e 保持原 200 mm 版的绝对尺寸
K = 200 / H
STEP, VMAX = .5, 6           # a 的离散色级：0–6 USD/MWh，每 0.5 一档（最大值 5.8）
BOUNDS = np.arange(0, VMAX + STEP / 2, STEP)
CLASS = mpl.colors.ListedColormap(SEQ(np.linspace(0, 1, len(BOUNDS) - 1)))
CNORM = mpl.colors.BoundaryNorm(BOUNDS, CLASS.N)
# 联合国子区域合并为 14 个区（a 的分区箱线图）
MERGE = {
    'Northern America': 'NAM', 'Central America': 'CAC', 'Caribbean': 'CAC', 'South America': 'SAM',
    'Northern Europe': 'WCE', 'Western Europe': 'WCE', 'Southern Europe': 'WCE', 'Eastern Europe': 'EER',
    'Northern Africa': 'MNA', 'Western Asia': 'MNA', 'Western Africa': 'WAF', 'Eastern Africa': 'CEA',
    'Middle Africa': 'CEA', 'Southern Africa': 'SAF', 'Central Asia': 'CAS', 'Southern Asia': 'SAS',
    'Eastern Asia': 'EAS', 'South-Eastern Asia': 'SEA', 'Australia and New Zealand': 'OCE', 'Melanesia': 'OCE',
    'Micronesia': 'OCE', 'Polynesia': 'OCE',
}


def low(x, y, w, h):
    """c–e 的坐标沿用 200 mm 画布的比例，换算到加高后的画布。"""
    return [x, y * K, w, h * K]


def regions(loc):
    """每个网格按最近的国家归入合并分区（海岛与南极等未归类的为 NaN）。"""
    _, countries = map_layers()
    pts = gpd.GeoDataFrame(index=loc.index, geometry=gpd.points_from_xy(loc.lon, loc.lat), crs=4326).to_crs(8857)
    near = gpd.sjoin_nearest(pts, countries[['SUBREGION', 'geometry']].to_crs(8857), how='left')
    return near[~near.index.duplicated()].SUBREGION.map(MERGE).reindex(loc.index)


def main():
    setup()
    p = paired()
    q75 = p[p.intensity_of_max.eq(.75)]
    fig = plt.figure(figsize=(180 / 25.4, H / 25.4))

    # ---------------- a 最不利机群地图（六边形格网，0.5 一档离散色）+ 内嵌直方图色标 + 分区箱线图
    wide = q75.pivot_table(index='grid_id', columns='scenario', values='saving_usd_mwh')
    worst = wide.min(axis=1)
    loc = q75.drop_duplicates('grid_id').set_index('grid_id').loc[worst.index, ['lat', 'lon']]
    region = regions(loc)
    ax_a = land_axes(fig, [.012, (H - 53) / H, .65, 50 / H])
    hex_mosaic(ax_a, loc.lon, loc.lat, worst.to_numpy(), CLASS, CNORM, width_km=210, max_km=300)
    _, countries = map_layers()
    outline = (countries.assign(code=countries.SUBREGION.map(MERGE)).dropna(subset=['code'])
               .assign(geometry=lambda d: d.geometry.buffer(0)).dissolve('code'))
    pc = ccrs.PlateCarree()
    ax_a.add_geometries(outline.geometry, crs=pc, facecolor='none', edgecolor='#6B6B6B', linewidth=.35, zorder=4)
    for code, sub in loc.groupby(region):
        ax_a.text(sub.lon.median(), sub.lat.median(), code, transform=pc, fontsize=5, fontweight='bold',
                  ha='center', va='center', color='#1E1E1E', zorder=6,
                  path_effects=[pe.withStroke(linewidth=1.2, foreground='white')])

    # 色标即直方图，放在南美以西的海域
    fig.canvas.draw()
    to_fig = ax_a.transData + fig.transFigure.inverted()
    (hx0, hy0), (hx1, hy1) = [to_fig.transform(ax_a.projection.transform_point(lo, la, pc))
                              for lo, la in ((-158, -33), (-92, -4))]
    axh = fig.add_axes([hx0, hy0, hx1 - hx0, hy1 - hy0])
    cnt, _ = np.histogram(worst, BOUNDS)
    axh.bar(BOUNDS[:-1], cnt / cnt.sum() * 100, width=STEP * .9, align='edge', color=CLASS(np.arange(CLASS.N)), lw=0)
    top = cnt.max() / cnt.sum() * 100
    p10, med = worst.quantile(.1), worst.median()
    for v, lab, ha in ((p10, f'P10 {p10:.1f}', 'right'), (med, f'Median {med:.1f}', 'left')):
        axh.plot([v, v], [0, top * 1.12], color=DARK, lw=.5, ls=(0, (2, 1.5)))
        axh.text(v + (.12 if ha == 'left' else -.12), top * 1.12, lab, fontsize=4.8, ha=ha, va='top')
    axh.set_xlim(0, VMAX)
    axh.set_ylim(0, top * 1.15)
    axh.set_xticks(range(VMAX + 1))
    axh.tick_params(labelsize=5, length=1.5, width=.4, pad=1)
    axh.set_yticks([])
    axh.spines['left'].set_visible(False)
    axh.patch.set_alpha(0)
    axh.set_xlabel('Lowest saving (USD/MWh)', fontsize=5.2, labelpad=1)

    # 分区箱线图：箱为四分位，须为 10–90 百分位，箱色为中位数所在色级，虚线为全球中位数
    stats = (worst.groupby(region).describe(percentiles=[.1, .25, .5, .75, .9]).sort_values('50%'))
    axr = fig.add_axes([.06, 131 / H, .585, 23 / H])
    for i, (code, s) in enumerate(stats.iterrows()):
        axr.plot([i, i], [s['10%'], s['90%']], color='#4A4A4A', lw=.5, zorder=1)
        axr.add_patch(mpl.patches.Rectangle((i - .3, s['25%']), .6, s['75%'] - s['25%'], lw=.4, zorder=2,
                                            facecolor=CLASS(CNORM(s['50%'])), edgecolor='#4A4A4A'))
        axr.plot([i - .3, i + .3], [s['50%']] * 2, color='white' if s['50%'] >= 3.5 else DARK, lw=.8, zorder=3)
    axr.axhline(med, color=DARK, lw=.5, ls=(0, (3, 2)), zorder=0)
    axr.set_xticks(range(len(stats)), stats.index, fontsize=5.2)
    axr.set_xlim(-.6, len(stats) - .4)
    axr.set_ylim(0, VMAX)
    axr.set_yticks(range(0, VMAX + 1, 2))
    axr.tick_params(axis='y', labelsize=5.2)
    axr.tick_params(axis='x', length=0)
    axr.set_ylabel('USD/MWh', fontsize=5.6, labelpad=1)
    guides(axr, 'y')
    panel(fig, 'a', .012, 1 - 3 / H)

    # ---------------- b 实施成本存活曲线
    ax_b = fig.add_axes([.735, 128 / H, .18, 78 / H])
    XMAX = 8
    cost = np.linspace(0, XMAX, 161)
    rows_b = []
    allv = q75.saving_usd_mwh.to_numpy()
    surv_all = (allv[None, :] > cost[:, None]).mean(axis=1) * 100
    ax_b.fill_between(cost, 0, surv_all, color='#EDEDED', lw=0, zorder=0)
    ends = []
    for u in USES:
        v = q75.loc[q75.scenario.str.endswith(u), 'saving_usd_mwh'].to_numpy()
        surv = (v[None, :] > cost[:, None]).mean(axis=1) * 100
        ax_b.plot(cost, surv, color=COLORS[u], lw=.9, zorder=2)
        ends.append([surv[-1], GROUP[u], COLORS[u], 'normal'])
        rows_b.append(pd.DataFrame({'group': u, 'cost_usd_mwh': cost, 'share_net_positive_pct': surv}))
    ax_b.plot(cost, surv_all, color='black', lw=1.6, zorder=3)
    ends.append([surv_all[-1], 'All fleets', 'black', 'bold'])
    rows_b.append(pd.DataFrame({'group': 'all', 'cost_usd_mwh': cost, 'share_net_positive_pct': surv_all}))
    ends.sort(key=lambda e: -e[0])
    for i in range(1, len(ends)):
        ends[i][0] = min(ends[i][0], ends[i - 1][0] - 7.5)
    for yv, lab, col, wt in ends:
        ax_b.text(XMAX + .2, yv, lab, color=col, fontsize=5.3, fontweight=wt, va='center', ha='left', clip_on=False)
    lines_b = ['All fleets still net positive']
    for c0 in (2, 3):
        s0 = (allv > c0).mean() * 100
        ax_b.scatter(c0, s0, s=16, color='black', edgecolors='white', lw=.5, zorder=5)
        lines_b.append(f'{s0:.0f}% at {c0} USD/MWh')
    ax_b.text(.2, 8, '\n'.join(lines_b), fontsize=5.4, ha='left', va='bottom', linespacing=1.3)
    ax_b.set_xlim(0, XMAX)
    ax_b.set_ylim(0, 103)
    ax_b.set_yticks([0, 25, 50, 75, 100])
    ax_b.tick_params(labelsize=5.4)
    ax_b.set_xlabel('Implementation cost (USD/MWh)', fontsize=6)
    ax_b.set_ylabel('Cases still net positive (%)', fontsize=6)
    guides(ax_b, 'y')
    panel(fig, 'b', .68, 1 - 3 / H)

    # ---------------- c 争用与互补（线尾直接标注）
    cap = pd.read_csv(RESIDUAL / 'max-intensity.csv').set_index('scenario').max_intensity
    ret = p.groupby(['scenario', 'intensity_of_max']).retention.median().unstack()
    ax_c = fig.add_axes(low(.085, .355, .38, .225))
    xx = np.linspace(0, .75, 50)
    ax_c.fill_between(xx, 1 - xx, 1.12, color='#EEF6F4', lw=0, zorder=0)
    ax_c.plot(xx, 1 - xx, color=DARK, lw=.8, ls=(0, (3, 2)))
    rows_c = []
    ends = {}
    for sc in SCENARIOS:
        occ = np.array(LEVELS) * cap[sc]
        r = ret.loc[sc, LEVELS].to_numpy()
        ax_c.plot(occ, r, color=shade(sc), lw=.8, zorder=3)
        ax_c.scatter(occ[-1], r[-1], marker=MARKS[sc[0]], s=22, color=COLORS[use(sc)], edgecolors='white', lw=.5,
                     zorder=4)
        ends.setdefault(use(sc), []).append((occ[-1], r[-1]))
        rows_c.append(pd.DataFrame({'scenario': sc, 'intensity_of_cap': LEVELS, 'idle_occupied': occ, 'retention': r}))
    offsets = {'INF': (0, .045, 'center', 'bottom'), 'SAI': (-.01, -.05, 'right', 'top'),
               'HPC': (.025, .015, 'left', 'center'), 'TRN': (0, -.045, 'center', 'top'),
               'CLD': (.0, .05, 'center', 'bottom')}
    for u, pts in ends.items():
        x_, y_ = np.mean(pts, axis=0)
        dx, dy, ha, va = offsets[u]
        ax_c.text(x_ + dx, y_ + dy, GROUP[u], color=COLORS[u], fontsize=5.8, fontweight='bold', ha=ha, va=va)
    ax_c.text(.36, 1 - .36 - .035, 'Full competition', rotation=-31, fontsize=5.4, color=DARK, ha='center', va='top',
              rotation_mode='anchor')
    ax_c.text(.72, 1.08, 'Complementarity', fontsize=5.6, color='#3F7F72', ha='right', va='top', style='italic')
    ax_c.set_xlim(0, .75)
    ax_c.set_ylim(.3, 1.12)
    ax_c.tick_params(labelsize=5.4)
    ax_c.set_xlabel('Idle capacity occupied by inference (× mean idle)', fontsize=6)
    ax_c.set_ylabel('Retention of coordination saving', fontsize=6)
    guides(ax_c, 'both')
    ax_c.legend(handles=[Line2D([], [], marker=mk, ls='none', color='#777777', markersize=3.5, label=INNER[k])
                         for k, mk in MARKS.items()],
                loc='lower left', fontsize=5, handletextpad=.2, labelspacing=.3, borderaxespad=.3, frameon=True,
                framealpha=.85, edgecolor='none')
    panel(fig, 'c', .012, .605 * K)

    # ---------------- d 保留率分解：转移电量比 × 每 MWh 价值比
    mv = pd.read_parquet(INPUT / 'moved-energy.parquet')
    mv['moved'] = mv.moved_task_mwh + mv.moved_off_mwh
    mv = mv.merge(p[['grid_id', 'scenario', 'intensity_of_max', 'V_usd']], on=['grid_id', 'scenario', 'intensity_of_max'])
    base = mv[mv.intensity_of_max.eq(0)].set_index(['grid_id', 'scenario'])[['moved', 'V_usd']]
    mv = mv.join(base, on=['grid_id', 'scenario'], rsuffix='0')
    mv = mv[mv.intensity_of_max.eq(.75)]
    mv['r_energy'] = mv.moved / mv.moved0
    mv['r_value'] = (mv.V_usd / mv.moved) / (mv.V_usd0 / mv.moved0)
    g = mv.groupby('scenario')
    dm = pd.DataFrame({'r_energy': g.r_energy.median(), 'r_value': g.r_value.median(),
                       'e25': g.r_energy.quantile(.25), 'e75': g.r_energy.quantile(.75),
                       'v25': g.r_value.quantile(.25), 'v75': g.r_value.quantile(.75)}).loc[SCENARIOS]
    ax_d = fig.add_axes(low(.61, .355, .26, .225))
    X, Y = np.meshgrid(np.linspace(.8, 1.25, 200), np.linspace(.3, 1.15, 200))
    lv = [.4, .5, .6, .7, .8, .9, 1.0, 1.1]
    cs = ax_d.contour(X, Y, X * Y, levels=lv, colors='#BDBDBD', linewidths=.5, linestyles='-')
    ax_d.clabel(cs, fmt=lambda v: f'{v:.1f}', fontsize=4.8, inline=True,
                manual=[(1.215, v / 1.215) for v in lv if .32 < v / 1.215 < 1.13])
    ax_d.axvline(1, color=DARK, lw=.6, ls=(0, (3, 2)))
    for sc in SCENARIOS:
        c = COLORS[use(sc)]
        r = dm.loc[sc]
        ax_d.plot([r.e25, r.e75], [r.r_value] * 2, color=shade(sc), lw=.6, zorder=3)
        ax_d.plot([r.r_energy] * 2, [r.v25, r.v75], color=shade(sc), lw=.6, zorder=3)
        ax_d.scatter(r.r_energy, r.r_value, marker=MARKS[sc[0]], s=22, color=c, edgecolors='white', lw=.5, zorder=4)
    ax_d.set_xlim(.8, 1.25)
    ax_d.set_ylim(.3, 1.15)
    ax_d.tick_params(labelsize=5.4)
    ax_d.set_xlabel('Energy shifted, 75% of cap ÷ no inference', fontsize=6)
    ax_d.set_ylabel('Value per shifted MWh,\n75% of cap ÷ no inference', fontsize=6)
    ax_d.legend(handles=[Line2D([], [], marker='o', ls='none', color=COLORS[u], markersize=3.5, label=GROUP[u])
                         for u in USES],
                loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=5, handletextpad=.2, labelspacing=.35,
                borderaxespad=0)
    panel(fig, 'd', .53, .605 * K)

    # ---------------- e 保留率 vs 风电占比（训练 GPU-heavy）+ 各机群秩相关
    dtrn = q75[q75.scenario.eq('D-TRN')]
    b0 = baseline()[['grid_id', 'scenario', 'grid_mwh0']]
    dd = dtrn.merge(b0[b0.scenario.eq('D-TRN')].drop(columns='scenario'), on='grid_id')
    gshare = 100 * dd.grid_mwh0 / dd.facility_mwh0
    ax_e = fig.add_axes(low(.085, .065, .42, .2))
    gnorm = mpl.colors.Normalize(0, 80)
    order = np.argsort(-gshare.to_numpy())
    ax_e.scatter(dd.wind_share.to_numpy()[order] * 100, dd.retention.to_numpy()[order], s=1.6, lw=0,
                 c=gshare.to_numpy()[order], cmap=SEQ.reversed(), norm=gnorm, rasterized=True)
    cut = pd.cut(dd.wind_share * 100, np.arange(0, 101, 5))
    bm = dd.groupby(cut, observed=True).retention.agg(['median', 'size'])
    bm = bm[bm['size'] >= 60]
    ax_e.plot([i.mid for i in bm.index], bm['median'], color=DARK, lw=1.1)
    ax_e.set_xlim(0, 100)
    ax_e.set_ylim(.1, 1.0)
    ax_e.tick_params(labelsize=5.4)
    ax_e.set_xlabel('Wind share of optimal wind + PV (%)', fontsize=6)
    ax_e.set_ylabel('Retention, LLM training · GPU-heavy', fontsize=6)
    guides(ax_e, 'both')
    cax = fig.add_axes(low(.515, .065, .006, .2))
    cbd = fig.colorbar(mpl.cm.ScalarMappable(norm=gnorm, cmap=SEQ.reversed()), cax=cax)
    cbd.outline.set_linewidth(.3)
    cax.tick_params(labelsize=5, length=1.5, width=.4, pad=1)
    cbd.set_label('Grid share without coordination (%)', fontsize=5.5, labelpad=2)

    rho = {sc: spearmanr(gq.wind_share, gq.retention)[0] for sc, gq in q75.groupby('scenario')}
    ax_r = fig.add_axes(low(.7, .065, .27, .2))
    order_u = sorted(USES, key=lambda u: -max(v for k, v in rho.items() if k.endswith(u)))
    rows_e = []
    for i, u in enumerate(order_u):
        vals = [rho[s] for s in SCENARIOS if s.endswith(u)]
        y0 = len(order_u) - 1 - i
        ax_r.barh(y0, max(vals), height=.55, color=COLORS[u], alpha=.35, lw=0)
        for s in [s for s in SCENARIOS if s.endswith(u)]:
            ax_r.scatter(rho[s], y0, marker=MARKS[s[0]], s=18, color=COLORS[u], edgecolors='white', lw=.4, zorder=3)
            rows_e.append({'scenario': s, 'rho_wind_retention': rho[s]})
        lab = f'{min(vals):.2f}' if len(vals) == 1 else f'{min(vals):.2f}–{max(vals):.2f}'
        ax_r.text(max(vals) + .03, y0, lab, va='center', fontsize=5.3)
    ax_r.set_yticks(range(len(order_u))[::-1], [GROUP[u] for u in order_u], fontsize=5.6)
    for t, u in zip(ax_r.get_yticklabels(), order_u):
        t.set_color(COLORS[u])
    ax_r.set_xlim(0, 1)
    ax_r.set_ylim(-.6, len(order_u) - .4)
    ax_r.tick_params(axis='x', labelsize=5.4)
    ax_r.tick_params(axis='y', length=0)
    ax_r.spines['left'].set_visible(False)
    ax_r.set_xlabel('Spearman ρ, wind share vs retention', fontsize=6)
    guides(ax_r)
    panel(fig, 'e', .012, .295 * K)

    export(fig, 'fig3')

    csv(pd.DataFrame({'grid_id': worst.index, 'lat': loc.lat.values, 'lon': loc.lon.values, 'region': region.values,
                      'lowest_saving_usd_mwh': worst.values, 'lowest_fleet': wide.idxmin(axis=1).values}), 'fig3a')
    csv(stats.rename_axis('region').reset_index(), 'fig3a-regions')
    csv(pd.concat(rows_b), 'fig3b')
    csv(pd.concat(rows_c), 'fig3c')
    csv(dm.reset_index(), 'fig3d')
    csv(pd.concat([dd[['grid_id', 'wind_share', 'retention']].assign(grid_share_pct=gshare.values, scenario='D-TRN'),
                   pd.DataFrame(rows_e)]), 'fig3e')
    return worst, stats, dm, rho


if __name__ == '__main__':
    worst, stats, dm, rho = main()
    print('worst min %.2f' % worst.min())
    print(stats[['count', '10%', '50%', '90%']].round(2).to_string())
    print(dm.round(3).to_string())
    print({k: round(v, 2) for k, v in rho.items()})
