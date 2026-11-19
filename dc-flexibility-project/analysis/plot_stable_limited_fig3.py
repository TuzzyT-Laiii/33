"""图3：真实推理需求下，算电协同仍带来稳定收益（第三节）。

a 推理加到上限 75% 时，每个网格 11 类机群中最低的协同节省（六边形格网地图，USD/MWh，0.5 一档离散色），
  海域内嵌直方图兼色标，下方为 14 个合并分区的箱线图；
b 实施成本存活曲线：实施成本为横轴时仍有净收益的网格–机群组合占比；下方为节省占供能成本与占总成本（机群设备 + 供能）的比例；
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
GRAYS = ['#C4C4C4', '#8F8F8F', '#555555', '#000000']   # b 中推理 0、25%、50%、75% 四档

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

    # ---------------- b 四档推理强度（灰阶由浅到深为 0→75%）。上：实施成本下仍有净收益的比例；下：节省占供能成本的比例
    ax_b = fig.add_axes([.735, 172 / H, .215, 34 / H])
    XMAX = 12
    cost = np.linspace(0, XMAX, 241)
    ax_b.axvspan(2, 3, color='#F0F0F0', lw=0, zorder=0)
    rows_b = []
    at = {2: [], 3: []}
    ends = []
    for lv, gc in zip(LEVELS, GRAYS):
        v = p.loc[p.intensity_of_max.eq(lv), 'saving_usd_mwh'].to_numpy()
        surv = (v[None, :] > cost[:, None]).mean(axis=1) * 100
        ax_b.plot(cost, surv, color=gc, lw=1.2, zorder=3)
        ends.append([surv[-1], f'{lv:.0%}', gc])
        for c0 in at:
            s0 = (v > c0).mean() * 100
            at[c0].append(s0)
            ax_b.scatter(c0, s0, s=8, color=gc, edgecolors='white', lw=.3, zorder=4)
        rows_b.append(pd.DataFrame({'kind': 'net_positive', 'intensity_of_cap': lv, 'cost_usd_mwh': cost,
                                    'share_net_positive_pct': surv}))
    for c0, ss in at.items():
        lab = f'{min(ss):.0f}–{max(ss):.0f}%'
        if c0 == 2:   # 2 的标注放在灰带左下（曲线仍在 100% 附近，下方空白）
            ax_b.text(c0 - .15, 90, lab, fontsize=5.2, ha='right', va='top')
        else:         # 3 的标注放在灰带右上（曲线已降到 94% 以下）
            ax_b.text(c0 + .15, 101, lab, fontsize=5.2, ha='left', va='top')
    for i in range(1, len(ends)):
        ends[i][0] = min(ends[i][0], ends[i - 1][0] - 7)
    for yv, lab, gc in ends:
        ax_b.text(XMAX + .25, yv, lab, color=gc, fontsize=5.2, va='center', ha='left', clip_on=False,
                  fontweight='bold')
    ax_b.text(XMAX + .25, ends[0][0] + 9, 'Inference\n(% of cap)', fontsize=4.8, va='bottom', ha='left',
              color=DARK, linespacing=1.05, clip_on=False)
    ax_b.set_xlim(0, XMAX)
    ax_b.set_ylim(0, 103)
    ax_b.set_yticks([0, 50, 100])
    ax_b.tick_params(labelsize=5.4)
    ax_b.set_xlabel('Implementation cost (USD/MWh)', fontsize=6, labelpad=1)
    ax_b.set_ylabel('Still net positive (%)', fontsize=6)
    guides(ax_b, 'y')

    # 下：各用途四档的节省占不做协同时供能成本的比例（中位数；点色同上）
    ax_p = fig.add_axes([.8, 131 / H, .08, 28 / H])
    groups = ['all'] + USES
    for i, u in enumerate(groups):
        g = p if u == 'all' else p[p.scenario.str.endswith(u)]
        md = g.groupby('intensity_of_max').V_pct_of_seq.median().loc[LEVELS].to_numpy()
        ax_p.plot([md.min(), md.max()], [i, i], color='#D0D0D0', lw=2.2, solid_capstyle='round', zorder=1)
        ax_p.scatter(md, [i] * len(md), s=14, color=GRAYS, edgecolors='white', lw=.4, zorder=3)
        rows_b.append(pd.DataFrame({'kind': 'pct_of_supply_cost_median', 'group': u, 'intensity_of_cap': LEVELS,
                                    'median_pct': md}))
    ax_p.set_yticks(range(len(groups)), ['All fleets'] + [GROUP[u] for u in USES], fontsize=5.4)
    for t, u in zip(ax_p.get_yticklabels(), groups):
        t.set_color('black' if u == 'all' else COLORS[u])
        t.set_fontweight('bold' if u == 'all' else 'normal')
    ax_p.set_ylim(len(groups) - .5, -.5)
    ax_p.set_xlim(0, 14)
    ax_p.set_xticks([0, 5, 10])
    ax_p.tick_params(axis='x', labelsize=5.4)
    ax_p.tick_params(axis='y', length=0, pad=2)
    ax_p.spines['left'].set_visible(False)
    ax_p.set_xlabel('% of supply\ncost', fontsize=6, labelpad=1)
    guides(ax_p, 'x')

    # 右：节省占总成本（机群年化设备成本 + 供能成本）的比例（逐格比值的中位数），浅蓝底突出
    dec = RESIDUAL.parent / 'supply-cost-decomposition-20261119'
    tu = pd.read_csv(dec / 'sec1-3-by-use.csv').set_index(['use', 'intensity_of_max'])
    ta = pd.read_csv(dec / 'sec1-3-all.csv').set_index('intensity_of_max')
    ax_q = fig.add_axes([.895, 131 / H, .065, 28 / H])
    ax_q.set_facecolor('#E3EEF6')
    for i, u in enumerate(groups):
        md = (ta if u == 'all' else tu.loc[u]).loc[LEVELS, 'saving_share_of_total_cost_median_pct'].to_numpy()
        ax_q.plot([md.min(), md.max()], [i, i], color='#B9CCDD', lw=2.2, solid_capstyle='round', zorder=1)
        ax_q.scatter(md, [i] * len(md), s=14, color=GRAYS, edgecolors='white', lw=.4, zorder=3)
        ax_q.text(1.02, i, f'{md.max():.2f}', transform=ax_q.get_yaxis_transform(), ha='left', va='center',
                  fontsize=5, fontweight='bold', color='#1B3F70', clip_on=False)
        rows_b.append(pd.DataFrame({'kind': 'pct_of_total_cost_median', 'group': u, 'intensity_of_cap': LEVELS,
                                    'median_pct': md}))
    ax_q.set_ylim(len(groups) - .5, -.5)
    ax_q.set_yticks([])
    ax_q.set_xlim(0, 1)
    ax_q.set_xticks([0, .5, 1], ['0', '0.5', '1'])
    ax_q.tick_params(axis='x', labelsize=5.4)
    ax_q.spines['left'].set_visible(False)
    ax_q.set_xlabel('% of total\ncost', fontsize=6, labelpad=1, fontweight='bold', color='#1B3F70')
    panel(fig, 'b', .68, 1 - 3 / H)

    # ---------------- c 争用与互补：每条线在 0/25/50/75% 打点，左上拥挤处另放局部放大
    cap = pd.read_csv(RESIDUAL / 'max-intensity.csv').set_index('scenario').max_intensity
    ret = p.groupby(['scenario', 'intensity_of_max']).retention.median().unstack()
    ax_c = fig.add_axes(low(.085, .355, .38, .225))
    ZOOM = (0, .3, .77, 1.035)
    ax_z = ax_c.inset_axes([.45, .63, .36, .35])
    xx = np.linspace(0, .75, 50)
    rows_c = []
    for ax in (ax_c, ax_z):
        ax.fill_between(xx, 1 - xx, 1.12, color='#EEF6F4', lw=0, zorder=0)
        ax.plot(xx, 1 - xx, color=DARK, lw=.8 if ax is ax_c else .6, ls=(0, (3, 2)))
    for sc in SCENARIOS:
        occ = np.array(LEVELS) * cap[sc]
        r = ret.loc[sc, LEVELS].to_numpy()
        for ax, s_mid, s_end in ((ax_c, 5, 22), (ax_z, 4, 16)):
            ax.plot(occ, r, color=shade(sc), lw=.8, zorder=3)
            ax.scatter(occ[1:-1], r[1:-1], s=s_mid, color=shade(sc), lw=0, zorder=4)
            ax.scatter(occ[-1], r[-1], marker=MARKS[sc[0]], s=s_end, color=COLORS[use(sc)], edgecolors='white',
                       lw=.5, zorder=5)
        rows_c.append(pd.DataFrame({'scenario': sc, 'intensity_of_cap': LEVELS, 'idle_occupied': occ, 'retention': r}))
    # 用云计算一条线标出四档强度
    occ = np.array(LEVELS) * cap['A-CLD']
    rc = ret.loc['A-CLD', LEVELS].to_numpy()
    for x_, y_, lab in zip(occ[1:], rc[1:], ['25%', '50%', '75% of cap']):
        ax_c.text(x_ + .012, y_ + .022, lab, fontsize=5, color=COLORS['CLD'], ha='left', va='bottom')
    ax_c.text(.555, .675, 'Cloud', color=COLORS['CLD'], fontsize=5.8, fontweight='bold', ha='left', va='top')
    ax_c.text(.47, .495, 'LLM training', color=COLORS['TRN'], fontsize=5.8, fontweight='bold', ha='left', va='top')
    for u, (x_, y_, ha) in {'INF': (.155, 1.02, 'left'), 'HPC': (.295, .80, 'right'),
                            'SAI': (.135, .805, 'right')}.items():
        ax_z.text(x_, y_, GROUP[u], color=COLORS[u], fontsize=5, fontweight='bold', ha=ha, va='center')
    ax_c.add_patch(mpl.patches.Rectangle((ZOOM[0], ZOOM[2]), ZOOM[1] - ZOOM[0], ZOOM[3] - ZOOM[2], fill=False,
                                         ec='#8C8C8C', lw=.5, ls=(0, (2, 1.5)), zorder=6))
    ax_z.set_xlim(*ZOOM[:2])
    ax_z.set_ylim(*ZOOM[2:])
    ax_z.set_xticks([0, .1, .2, .3])
    ax_z.set_yticks([.8, .9, 1.0])
    ax_z.tick_params(labelsize=4.6, length=1.5, pad=1)
    for s_ in ax_z.spines.values():
        s_.set_visible(True)
        s_.set_edgecolor('#8C8C8C')
        s_.set_linewidth(.5)
    ax_c.text(.36, 1 - .36 - .035, 'Full competition', rotation=-31, fontsize=5.4, color=DARK, ha='center', va='top',
              rotation_mode='anchor')
    ax_c.text(.74, .7, 'Complementarity', fontsize=5.6, color='#3F7F72', ha='right', va='bottom', style='italic')
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

    # ---------------- d 保留率分解：转移电量比 × 每 MWh 价值比（按用途直接标注）
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
    ax_d = fig.add_axes(low(.6, .355, .37, .225))
    XL, YL = (.85, 1.13), (.38, 1.08)
    X, Y = np.meshgrid(np.linspace(*XL, 200), np.linspace(*YL, 200))
    lv = [.4, .5, .6, .7, .8, .9, 1.0, 1.1]
    cs = ax_d.contour(X, Y, X * Y, levels=lv, colors='#BDBDBD', linewidths=.5, linestyles='-')
    ax_d.clabel(cs, fmt=lambda v: f'{v:.1f}', fontsize=4.8, inline=True,
                manual=[(1.11, v / 1.11) for v in lv if YL[0] + .02 < v / 1.11 < YL[1] - .02])
    ax_d.axvline(1, color=DARK, lw=.6, ls=(0, (3, 2)))
    for sc in SCENARIOS:
        c = COLORS[use(sc)]
        r = dm.loc[sc]
        ax_d.plot([r.e25, r.e75], [r.r_value] * 2, color=shade(sc), lw=.6, zorder=3)
        ax_d.plot([r.r_energy] * 2, [r.v25, r.v75], color=shade(sc), lw=.6, zorder=3)
        ax_d.scatter(r.r_energy, r.r_value, marker=MARKS[sc[0]], s=22, color=c, edgecolors='white', lw=.5, zorder=4)
    lab_d = {'INF': (1.0, 1.035, 'center'), 'HPC': (.935, .93, 'right'), 'SAI': (1.09, .86, 'left'),
             'CLD': (.955, .67, 'right'), 'TRN': (.995, .43, 'right')}
    for u, (x_, y_, ha) in lab_d.items():
        ax_d.text(x_, y_, GROUP[u], color=COLORS[u], fontsize=5.8, fontweight='bold', ha=ha, va='center')
    ax_d.set_xlim(*XL)
    ax_d.set_ylim(*YL)
    ax_d.tick_params(labelsize=5.4)
    ax_d.set_xlabel('Energy shifted, 75% of cap ÷ no inference', fontsize=6)
    ax_d.set_ylabel('Value per shifted MWh,\n75% of cap ÷ no inference', fontsize=6)
    panel(fig, 'd', .53, .605 * K)

    # ---------------- e 保留率随风电占比的变化：五个用途的分档中位数线 + 四分位带，背景为算例分布（参照 NCC 2022 Rainfall gradients）
    ax_e = fig.add_axes(low(.085, .065, .7, .2))
    edges = np.arange(0, 101, 5)
    mids = (edges[:-1] + edges[1:]) / 2
    ws = q75.wind_share * 100
    ax_h = ax_e.twinx()
    cnt, _ = np.histogram(ws, edges)
    ax_h.bar(mids, cnt / cnt.sum() * 100, width=4.4, color='#EDEDED', lw=0, zorder=0)
    ax_h.set_ylim(0, 60)
    ax_h.set_yticks([0, 10, 20])
    ax_h.tick_params(labelsize=5, colors='#8C8C8C')
    ax_h.spines['right'].set_visible(True)
    ax_h.spines['right'].set_color('#BDBDBD')
    ax_h.set_ylabel('Cases (%)', fontsize=5.6, color='#8C8C8C', labelpad=2)
    ax_h.yaxis.set_label_coords(1.05, .17)
    ax_e.set_zorder(ax_h.get_zorder() + 1)
    ax_e.patch.set_visible(False)
    rho = {sc: spearmanr(gq.wind_share, gq.retention)[0] for sc, gq in q75.groupby('scenario')}
    rows_e = []
    ends = []
    for u in USES:
        gu = q75[q75.scenario.str.endswith(u)]
        cut = pd.cut(gu.wind_share * 100, edges, labels=mids)
        st = gu.groupby(cut, observed=True).retention.agg(q25=lambda s: s.quantile(.25), median='median',
                                                         q75=lambda s: s.quantile(.75), n='size')
        st = st[st.n >= 60]
        xs = st.index.astype(float)
        ax_e.fill_between(xs, st.q25, st.q75, color=COLORS[u], alpha=.15, lw=0, zorder=2)
        ax_e.plot(xs, st['median'], color=COLORS[u], lw=1.2, zorder=3)
        vals = [rho[s] for s in SCENARIOS if s.endswith(u)]
        lab = f'{min(vals):.2f}' if len(vals) == 1 else f'{min(vals):.2f}–{max(vals):.2f}'
        ends.append([st['median'].iloc[-1], xs[-1], f'{GROUP[u]}  ρ = {lab}', COLORS[u]])
        rows_e.append(st.reset_index(names='wind_share_mid_pct').assign(group=u))
    ends.sort(key=lambda e: -e[0])
    for i in range(1, len(ends)):
        ends[i][0] = min(ends[i][0], ends[i - 1][0] - .065)
    for yv, xv, lab, col in ends:
        ax_e.text(101.5, yv, lab, color=col, fontsize=5.4, fontweight='bold', va='center', ha='left', clip_on=False)
    ax_e.set_xlim(0, 100)
    ax_e.set_ylim(.3, 1.05)
    ax_e.tick_params(labelsize=5.4)
    ax_e.set_xlabel('Wind share of optimal wind + PV (%)', fontsize=6)
    ax_e.set_ylabel('Retention of coordination saving', fontsize=6)
    guides(ax_e, 'y')
    panel(fig, 'e', .012, .295 * K)

    export(fig, 'fig3')

    csv(pd.DataFrame({'grid_id': worst.index, 'lat': loc.lat.values, 'lon': loc.lon.values, 'region': region.values,
                      'lowest_saving_usd_mwh': worst.values, 'lowest_fleet': wide.idxmin(axis=1).values}), 'fig3a')
    csv(stats.rename_axis('region').reset_index(), 'fig3a-regions')
    csv(pd.concat(rows_b), 'fig3b')
    csv(pd.concat(rows_c), 'fig3c')
    csv(dm.reset_index(), 'fig3d')
    csv(pd.concat(rows_e + [pd.DataFrame({'scenario': list(rho), 'rho_wind_retention': list(rho.values())})]),
        'fig3e')
    return worst, stats, dm, rho


if __name__ == '__main__':
    worst, stats, dm, rho = main()
    print('worst min %.2f' % worst.min())
    print(stats[['count', '10%', '50%', '90%']].round(2).to_string())
    print(dm.round(3).to_string())
    print({k: round(v, 2) for k, v in rho.items()})
