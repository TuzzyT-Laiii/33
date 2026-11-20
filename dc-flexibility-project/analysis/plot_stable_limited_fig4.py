"""图4：碳价和技术成本变化不改变“协同收益稳定但有限”的判断（第四节）。按子论点排布 8 个子图：

4.1 碳价提高会稳定放大协同收益（放大的是经济价值而不是减排）
  a 碳价 200 美元/吨比不收碳价多省的钱（USD/MWh，各网格取 11 类机群中位数，不接推理）：
    等经纬度地图 + 上方经度柱 + 右侧纬度柱（中位数，误差线为四分位）+ 海域直方图色标（参照 GeoSciPlot 21b97afa0d）；
  b 全球与六大洲：右为多省的钱，左为协同减排的变化（左右分向条形，参照 GeoSciPlot 1a4badb023 d–e）；
4.2 碳价越高，接入推理后剩余的协同收益越稳
  c 各用途协同节省下十分位从碳价 0 到 200 的变化，不接推理与推理 75% 两档（区间点线图）；
4.3 卡价和待机功耗下降不会让为柔性加卡变得划算
  d 允许增减卡时的减卡比例随推理强度的变化（八城市，PR #83；只许加卡时 0/29,568 加卡，见图注）；
4.4 风光成本下降影响有限，储能大幅降价会部分替代协同
  e 2035 乐观路径下协同节省相对 2024 的倍数：版式同 a，柱以 1 为基线；
  f 三条路径 2024→2030→2035 的轨迹：协同节省倍数 × 占供能成本比例，横向误差线为碳价 0–200（PR #83 交叉）；
  g 不做协同时新建储能时长 vs 协同节省倍数（二维直方图）；
  h 各成本情景的供能成本中位数 vs 碳价 0→200 放大倍数（气泡大小为单调上升比例，PR #83 交叉）。
逐网格结果来自 PR #80 的服务器结果（本地 D:/codex-runs/sec4-20261116），先提取为 INPUT/sec4/ 下的小表。
"""
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from stable_limited_figs_common import (COLORS, DARK, DIV, GROUP, INPUT, ROOT, SCENARIOS, SEQ, csv, export, guides,
                                        map_layers, paired, panel, setup, use)

SEC4 = INPUT / 'sec4'
SUPP = ROOT / 'results/sec4-supplement-20261118'
RAW = Path('D:/codex-runs/sec4-20261116')
USES = ['HPC', 'CLD', 'SAI', 'TRN', 'INF']
PATHS = {'conservative': '#93CDB9', 'moderate': '#3F93A8', 'advanced': '#1B3F70'}
LEVELS = [0, .25, .5, .75]
EXTRA = 50            # 为底部面板 i 加高的 mm
H = 252 + EXTRA
EXT = (-180, 180, -58, 84)
REGIONS = ['Asia', 'Africa', 'Oceania', 'Europe', 'North America', 'South America']
TEAL, TERRA = '#2E8FA3', '#C0623F'


def prepare():
    """从逐网格结果提取图 4 所需的小表（只在缺失时运行）。"""
    SEC4.mkdir(parents=True, exist_ok=True)
    if not (SEC4 / 'fig4-grid-carbon.parquet').exists():
        c = pd.read_parquet(RAW / 'carbon-paired.parquet').reset_index()
        c[['grid_id', 'scenario', 'carbon_price_usd_t', 'intensity_of_max', 'ef', 'V', 'dEm']].to_parquet(
            SEC4 / 'fig4-grid-carbon.parquet', index=False)
    if not (SEC4 / 'fig4-grid-lc.parquet').exists():
        lc = pd.read_parquet(RAW / 'lc-paired.parquet').reset_index()
        lc = lc[lc.intensity_of_max.eq(0)]
        lc[['grid_id', 'scenario', 'cost_scenario', 'Vmwh', 'batt_seq', 'batt_joint', 'E0']].to_parquet(
            SEC4 / 'fig4-grid-lc.parquet', index=False)
    if not (SEC4 / 'learning-pct-all.csv').exists():
        d = pd.read_parquet(RAW / 'lc-paired.parquet').reset_index()
        d['pct'] = d.V / d.seq * 100
        d.groupby(['cost_scenario', 'intensity_of_max']).agg(
            pct_median=('pct', 'median'), supply_cost_median=('seq_mwh', 'median'),
            saving_median=('Vmwh', 'median'), pairs=('V', 'size')).reset_index().to_csv(
            SEC4 / 'learning-pct-all.csv', index=False)


def lonlat_fmt(ax):
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: f'{abs(v):.0f}°{"W" if v < 0 else "E" if v > 0 else ""}'))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: f'{abs(v):.0f}°{"S" if v < 0 else "N" if v > 0 else ""}'))


def marginal_map(fig, rect, values, cmap, norm, bar_lim, bar_ticks, unit, hist_label, base=0., hist_ticks=None,
                 med_fmt='{:.1f}', top_h=.07, side_w=.11):
    """等经纬度格网地图 + 上方经度柱（10°）+ 右侧纬度柱（4°），柱为中位数、误差线为四分位，柱从 base 起；
    南太平洋海域内嵌直方图兼作色标。返回 (地图轴, 地图位置, 经度统计, 纬度统计)。"""
    cells, countries = map_layers()
    cells = cells.merge(values.rename('v'), left_on='grid_id', right_index=True)
    cells = cells.loc[~cells.geometry.is_empty]
    cen = cells.geometry.representative_point()
    cells['lon'], cells['lat'] = cen.x, cen.y
    pc = ccrs.PlateCarree()
    ax = fig.add_axes(rect, projection=pc)
    ax.set_extent(EXT, crs=pc)
    ax.add_geometries(countries.geometry, crs=pc, facecolor='#E6E6E6', edgecolor='none', zorder=0, rasterized=True)
    ax.add_geometries(cells.geometry, crs=pc, facecolor=cmap(norm(cells.v.to_numpy())), edgecolor='none',
                      antialiased=False, zorder=1, rasterized=True)
    ax.add_geometries(countries.geometry, crs=pc, facecolor='none', edgecolor='#9A9A9A', linewidth=.15, zorder=2,
                      rasterized=True)
    ax.spines['geo'].set_linewidth(.5)
    ax.set_xticks(range(-150, 181, 60), crs=pc)
    ax.set_yticks([-40, -20, 0, 20, 40, 60, 80], crs=pc)
    lonlat_fmt(ax)
    ax.tick_params(labelsize=5.2, length=1.8, width=.4, pad=1.5)
    fig.canvas.draw()
    pos = ax.get_position()

    def binned(key, edges):
        g = cells.groupby(pd.cut(cells[key], edges), observed=False).v
        q = pd.DataFrame({'q25': g.quantile(.25), 'median': g.median(), 'q75': g.quantile(.75), 'n': g.size()})
        q['mid'] = (edges[:-1] + edges[1:]) / 2
        return q[q.n >= 5]

    lon = binned('lon', np.arange(-180, 181, 10))
    axt = fig.add_axes([pos.x0, pos.y1 + .008, pos.width, top_h])
    axt.bar(lon.mid, lon['median'] - base, bottom=base, width=8.6, color=cmap(norm(lon['median'].to_numpy())), lw=0)
    axt.errorbar(lon.mid, lon['median'], yerr=[lon['median'] - lon.q25, lon.q75 - lon['median']], fmt='none',
                 ecolor=DARK, elinewidth=.45, capsize=0)
    axt.set_xlim(EXT[0], EXT[1])
    axt.set_ylim(*bar_lim)
    axt.set_yticks(bar_ticks)
    axt.set_xticks([])
    axt.tick_params(labelsize=5, length=1.5, pad=1)
    axt.set_ylabel(unit, fontsize=5.4, labelpad=1)
    axt.spines['bottom'].set_visible(False)
    axt.grid(axis='y', color='#ECECEC', lw=.4)
    axt.set_axisbelow(True)

    lat = binned('lat', np.arange(-58, 85, 4))
    axr = fig.add_axes([pos.x1 + .008, pos.y0, side_w, pos.height])
    axr.barh(lat.mid, lat['median'] - base, left=base, height=3.4, color=cmap(norm(lat['median'].to_numpy())), lw=0)
    axr.errorbar(lat['median'], lat.mid, xerr=[lat['median'] - lat.q25, lat.q75 - lat['median']], fmt='none',
                 ecolor=DARK, elinewidth=.45, capsize=0)
    axr.set_ylim(EXT[2], EXT[3])
    axr.set_xlim(*bar_lim)
    axr.set_xticks(bar_ticks)
    axr.set_yticks([])
    axr.tick_params(labelsize=5, length=1.5, pad=1)
    axr.set_xlabel(unit, fontsize=5.4, labelpad=1)
    axr.spines['left'].set_visible(False)
    axr.grid(axis='x', color='#ECECEC', lw=.4)
    axr.set_axisbelow(True)
    if base:
        axt.axhline(base, color='#8C8C8C', lw=.5)
        axr.axvline(base, color='#8C8C8C', lw=.5)

    to_fig = ax.transData + fig.transFigure.inverted()
    (x0, y0), (x1, y1) = to_fig.transform([(-170, -40), (-100, -6)])
    axh = fig.add_axes([x0, y0, x1 - x0, y1 - y0])
    bounds = norm.boundaries
    cnt, _ = np.histogram(np.clip(values, bounds[0], bounds[-1] - 1e-9), bounds)
    width = np.diff(bounds)
    axh.bar(bounds[:-1], cnt / cnt.sum() * 100, width=width * .88, align='edge', color=cmap(np.arange(cmap.N)), lw=0)
    med = float(np.median(values))
    top = cnt.max() / cnt.sum() * 100
    axh.plot([med, med], [0, top * 1.1], color=DARK, lw=.5, ls=(0, (2, 1.5)))
    axh.text(med + width.mean() * .15, top * 1.1, 'Median ' + med_fmt.format(med), fontsize=4.8, ha='left', va='top')
    axh.set_xlim(bounds[0], bounds[-1])
    axh.set_ylim(0, top * 1.15)
    axh.set_xticks(hist_ticks if hist_ticks is not None else bounds[::2])
    axh.set_yticks([])
    axh.spines['left'].set_visible(False)
    axh.patch.set_alpha(0)
    axh.tick_params(labelsize=4.8, length=1.5, pad=1)
    axh.set_xlabel(hist_label, fontsize=5, labelpad=1)
    return ax, pos, lon, lat


DECOMP = ROOT / 'results/supply-cost-decomposition-20261119'
COMPS = [('grid_energy', 'Grid energy', '#1B3F70'), ('grid_capacity', 'Grid capacity', '#46A6B5'),
         ('carbon', 'Carbon cost', '#7FC3AE'), ('storage', 'Storage', '#C9A646')]


def panel_i(fig):
    """i：碳价与成本路径下协同省钱方式的转变（毛节省构成，不接推理）与协同节省占供能成本、占总成本的比例。"""
    ca = pd.read_csv(DECOMP / 'sec4-carbon-all.csv')
    ca = ca[ca.intensity_of_max.eq(0)].set_index('carbon_price_usd_t')
    la = pd.read_csv(DECOMP / 'sec4-learning-all.csv')
    la = la[la.intensity_of_max.eq(0)].set_index('cost_scenario')
    cu = pd.read_csv(DECOMP / 'sec4-carbon-by-use.csv')
    cu = cu[cu.intensity_of_max.eq(0)]
    lu = pd.read_csv(DECOMP / 'sec4-learning-by-use.csv')
    lu = lu[lu.intensity_of_max.eq(0)]
    cases = [(f'Carbon {c} USD/t', ca.loc[c], cu[cu.carbon_price_usd_t.eq(c)], None) for c in (0, 100, 200)]
    for yr in (2030, 2035):
        for path in ('conservative', 'moderate', 'advanced'):
            k = f'{yr}-{path}'
            cases.append((f'{yr} {path}', la.loc[k], lu[lu.cost_scenario.eq(k)], k))
    y = np.arange(len(cases))[::-1].astype(float)
    y[:3] += 1.0
    y[3:6] += .5
    ax1 = fig.add_axes([.17, 7 / H, .37, (EXTRA - 15) / H])
    axs = fig.add_axes([.63, 7 / H, .15, (EXTRA - 15) / H])
    ax2 = fig.add_axes([.83, 7 / H, .14, (EXTRA - 15) / H])
    rows = []
    for yi, (name, a, u, key) in zip(y, cases):
        v = {k: max(a[f'{k}_saving_mean_usd_mwh'], 0) for k, _, _ in COMPS}
        gross = sum(v.values())
        left = 0
        for k, lab, col in COMPS:
            w = v[k] / gross * 100
            ax1.barh(yi, w, left=left, height=.62, color=col, lw=0, zorder=2)
            if k in ('grid_energy', 'storage') and w >= 9:
                ax1.text(left + w / 2, yi, f'{w:.0f}%', ha='center', va='center', fontsize=5.8, fontweight='bold',
                         color='white' if k == 'grid_energy' else '#2B2B2B', zorder=3)
            left += w
            rows.append({'case': name, 'component': k, 'share_of_gross_pct': w, 'usd_mwh': v[k]})
        ax1.text(101.5, yi, f'{a.saving_mean_usd_mwh:.1f}', ha='left', va='center', fontsize=5.6, fontweight='bold',
                 color='#2B2B2B')
        for _, r in u.iterrows():
            c = COLORS[r.use]
            axs.scatter(r.saving_share_of_supply_cost_median_pct, yi, s=13, color=c, edgecolors='white', lw=.4,
                        zorder=3)
            ax2.scatter(r.saving_share_of_total_cost_median_pct, yi + (.13 if key else 0), s=13, color=c,
                        edgecolors='white', lw=.4, zorder=3)
            if key:
                ax2.scatter(r.saving_share_of_total_cost_cardpath_median_pct, yi - .13, s=12, facecolor='white',
                            edgecolors=c, lw=.8, zorder=3)
        top = max(u.saving_share_of_total_cost_median_pct.max(),
                  u.saving_share_of_total_cost_cardpath_median_pct.max() if key else 0)
        ax2.text(top + .1, yi, f'≤ {top:.1f}%', ha='left', va='center', fontsize=5.6, fontweight='bold',
                 color='#1B3F70')
        sup = u.saving_share_of_supply_cost_median_pct
        axs.text(sup.max() + .4, yi, f'{sup.min():.1f}–{sup.max():.1f}%', ha='left', va='center', fontsize=5.4,
                 fontweight='bold', color='#2B2B2B')
        for _, r in u.iterrows():
            rows.append({'case': name, 'component': 'pct_' + r.use, 'share_of_supply_cost_median_pct':
                         r.saving_share_of_supply_cost_median_pct, 'share_of_total_cost_median_pct':
                         r.saving_share_of_total_cost_median_pct, 'share_of_total_cost_cardpath_median_pct':
                         r.get('saving_share_of_total_cost_cardpath_median_pct', np.nan)})
    ax1.set_yticks(y, [c[0] for c in cases], fontsize=5.8)
    ax1.tick_params(axis='y', length=0)
    ax1.spines['left'].set_visible(False)
    ax1.set_xlim(0, 100)
    ax1.set_xticks([0, 25, 50, 75, 100])
    ax1.set_xlabel('Share of gross coordination saving (%)', fontsize=6, labelpad=1)
    ax1.text(101.5, y[0] + .75, 'Net\nUSD/MWh', ha='left', va='bottom', fontsize=5.2)
    ax1.legend(handles=[Patch(color=col, label=lab) for _, lab, col in COMPS], loc='lower center',
               bbox_to_anchor=(.5, 1.0), ncol=4, fontsize=5.6, handlelength=1, columnspacing=1.2, borderaxespad=.2)
    axs.set_ylim(ax1.get_ylim())
    axs.set_yticks([])
    axs.spines['left'].set_visible(False)
    axs.set_xlim(0, 24)
    axs.set_xticks([0, 5, 10, 15])
    axs.set_xlabel('Saving (% of supply cost)', fontsize=6, labelpad=1, fontweight='bold')
    guides(axs)
    axs.legend(handles=[Line2D([], [], marker='o', ls='none', color=COLORS[u_], ms=3.2, label=GROUP[u_])
                        for u_ in ('HPC', 'CLD', 'SAI', 'TRN', 'INF')], loc='lower center', bbox_to_anchor=(.5, 1.0),
               ncol=3, fontsize=5, handletextpad=.1, columnspacing=.5, borderaxespad=.2)
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks([])
    ax2.spines['left'].set_visible(False)
    ax2.set_facecolor('#E3EEF6')
    ax2.set_xlim(0, 3.6)
    ax2.set_xlabel('Saving (% of total cost)', fontsize=6, labelpad=1, fontweight='bold', color='#1B3F70')
    guides(ax2)
    ax2.legend(handles=[Line2D([], [], marker='o', ls='none', color='#666666', ms=3.4, label='Fleet at 2024 prices'),
                        Line2D([], [], marker='o', ls='none', mfc='white', mec='#666666', ms=3.4,
                               label='Fleet on card-price path')],
               loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=1, fontsize=5.4, handletextpad=.2, borderaxespad=.2)
    return pd.DataFrame(rows)


def main():
    setup()
    prepare()
    cg = pd.read_parquet(SEC4 / 'fig4-grid-carbon.parquet')
    cg['use'] = cg.scenario.str[2:]
    lg = pd.read_parquet(SEC4 / 'fig4-grid-lc.parquet')
    cont = paired().drop_duplicates('grid_id').set_index('grid_id').continent
    fig = plt.figure(figsize=(180 / 25.4, H / 25.4))
    out = {}

    # ================= 4.1 =================
    z0 = cg[cg.intensity_of_max.eq(0)]
    wv = z0.pivot_table(index=['grid_id', 'scenario'], columns='carbon_price_usd_t', values='V')
    we = z0.pivot_table(index=['grid_id', 'scenario'], columns='carbon_price_usd_t', values='dEm') * 1000
    dv = (wv[200] - wv[0]).groupby(level=0).median()
    bounds_a = np.arange(0, 12.01, 1.5)
    cmap_a = mpl.colors.ListedColormap(SEQ(np.linspace(0, 1, len(bounds_a) - 1)))
    cmap_a.set_over(SEQ(1.0))
    norm_a = mpl.colors.BoundaryNorm(bounds_a, cmap_a.N)
    ax_a, pos_a, lon_a, lat_a = marginal_map(fig, [.07, (188 + EXTRA) / H, .68, 48 / H], dv, cmap_a, norm_a, (0, 12),
                                             [0, 6, 12], 'USD/MWh', 'Extra saving at 200 USD/t (USD/MWh)',
                                             hist_ticks=range(0, 13, 3), top_h=.052, side_w=.12)
    panel(fig, 'a', .012, 1 - 2 / H)

    # b 全球与六大洲：右为多省的钱，左为协同减排的变化（碳价 200 相对 0）
    # 与 a 同口径：先在每个网格取 11 类机群的中位数，再取区域内网格的中位数
    pair = pd.DataFrame({'dv': dv, 'de': (we[200] - we[0]).groupby(level=0).median()})
    pair['region'] = pair.index.map(cont)
    rows_b = [('Global', pair)] + [(r, pair[pair.region.eq(r)]) for r in REGIONS]
    ys = np.arange(len(rows_b))[::-1]
    axl = fig.add_axes([.15, (132 + EXTRA) / H, .16, 36 / H])     # 左：协同减排的变化（kg/MWh，向左为减少）
    axr = fig.add_axes([.315, (132 + EXTRA) / H, .17, 36 / H])    # 右：多省的钱（USD/MWh）
    tab_b = []
    for y, (name, g) in zip(ys, rows_b):
        v, e = g.dv.median(), g.de.median()
        tab_b.append({'region': name, 'extra_saving_usd_mwh': v, 'emission_cut_change_kg_mwh': e, 'grids': len(g)})
        axr.barh(y, v, height=.62, color=TEAL, lw=0)
        axl.barh(y, -e, height=.62, color=TERRA, lw=0)
        axr.text(v + .3, y, f'+{v:.1f}', fontsize=5, va='center', ha='left', color=TEAL)
        axl.text(-e + .6, y, f'{e:.1f}', fontsize=5, va='center', ha='right', color=TERRA)
    axl.set_xlim(32, 0)
    axr.set_xlim(0, 12)
    axl.set_xticks([30, 20, 10])
    axr.set_xticks([0, 4, 8, 12])
    axl.set_yticks(ys, [n for n, _ in rows_b], fontsize=5.4)
    axl.get_yticklabels()[0].set_fontweight('bold')
    axl.tick_params(axis='y', length=0, pad=2)
    axr.set_yticks([])
    for ax in (axl, axr):
        ax.set_ylim(-.6, len(rows_b) - .4)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='x', labelsize=5.2)
        guides(ax, 'x')
    axr.axvline(0, color=DARK, lw=.6)
    axl.set_xlabel('Emission cut lost (kg CO$_2$/MWh)', fontsize=5.8, labelpad=1, color=TERRA)
    axr.set_xlabel('Extra saving (USD/MWh)', fontsize=5.8, labelpad=1, color=TEAL)
    out['b'] = pd.DataFrame(tab_b)
    panel(fig, 'b', .012, (177 + EXTRA) / H)

    # ================= 4.2 =================
    # c 各用途下十分位：碳价 0（空心）→ 200（实心），上为不接推理，下为推理 75%
    axc = fig.add_axes([.64, (132 + EXTRA) / H, .33, 36 / H])
    rows_c = []
    yl = []
    groups = ['All'] + USES
    for k, u in enumerate(groups):
        for off, lv, alpha in ((.17, 0, .45), (-.17, .75, 1)):
            g = cg[cg.intensity_of_max.eq(lv)] if u == 'All' else cg[cg.intensity_of_max.eq(lv) & cg.use.eq(u)]
            p10 = g.groupby('carbon_price_usd_t').V.quantile(.1)
            y = len(groups) - 1 - k + off
            colr = 'black' if u == 'All' else COLORS[u]
            axc.plot([p10[0], p10[200]], [y, y], color=colr, lw=1.6, alpha=alpha, solid_capstyle='butt')
            axc.scatter(p10[0], y, s=13, facecolor='white', edgecolor=colr, lw=.7, zorder=3)
            axc.scatter(p10[200], y, s=15, color=colr, edgecolor='white', lw=.4, zorder=3, alpha=max(alpha, .6))
            axc.text(p10[200] + .12, y, f'×{p10[200] / p10[0]:.2f}', fontsize=4.6, va='center', ha='left',
                     color=colr, alpha=max(alpha, .7))
            rows_c.append({'group': u, 'intensity_of_cap': lv, 'p10_c0': p10[0], 'p10_c100': p10[100],
                           'p10_c200': p10[200]})
        yl.append(len(groups) - 1 - k)
    axc.set_yticks(yl, ['All fleets'] + [GROUP[u] for u in USES], fontsize=5.4)
    for t, u in zip(axc.get_yticklabels(), groups):
        t.set_color('black' if u == 'All' else COLORS[u])
        t.set_fontweight('bold' if u == 'All' else 'normal')
    axc.tick_params(axis='y', length=0, pad=2)
    axc.spines['left'].set_visible(False)
    axc.set_xlim(0, 12.5)
    axc.set_ylim(-.6, len(groups) - .4)
    axc.tick_params(axis='x', labelsize=5.2)
    axc.set_xlabel('Lower decile of saving (USD/MWh)', fontsize=6, labelpad=1)
    guides(axc, 'x')
    axc.legend(handles=[Line2D([], [], marker='o', ls='none', mfc='white', mec=DARK, ms=3.2, label='0 USD/t'),
                        Line2D([], [], marker='o', ls='none', color=DARK, ms=3.2, label='200 USD/t'),
                        Line2D([], [], color=DARK, lw=1.6, alpha=.45, label='No inference'),
                        Line2D([], [], color=DARK, lw=1.6, label='75% of cap')],
               loc='lower right', fontsize=4.8, ncol=2, handlelength=1.1, columnspacing=.6, labelspacing=.25,
               borderaxespad=.2, frameon=True, framealpha=.9, edgecolor='none')
    out['c'] = pd.DataFrame(rows_c)
    panel(fig, 'c', .53, (177 + EXTRA) / H)

    # ================= 4.3 =================
    # d 允许增减卡时的减卡比例：每个用途一条中位数线 + 配置范围色带，线头直接标注
    cut = pd.read_csv(SUPP / 'card-city/free-cut-share.csv')
    cs = cut.groupby(['scenario', 'intensity_of_max']).cut_share_median.median().unstack() * 100
    axd = fig.add_axes([.12, (74 + EXTRA) / H, .2, 38 / H])
    xx = np.array(LEVELS) * 100
    rows_d = []
    for u in USES:
        sub = cs.loc[[s for s in SCENARIOS if use(s) == u], LEVELS]
        lo, md, hi = sub.min().to_numpy(), sub.median().to_numpy(), sub.max().to_numpy()
        if len(sub) > 1:
            axd.fill_between(xx, lo, hi, color=COLORS[u], alpha=.18, lw=0)
        axd.plot(xx, md, color=COLORS[u], lw=1.3, marker='o', ms=2.8, mec='white', mew=.4)
        axd.text(-3, md[0] + {'HPC': -1.2, 'SAI': 1.0}.get(u, 0), GROUP[u], color=COLORS[u], fontsize=5,
                 fontweight='bold', ha='right', va='center')
        rows_d.append(pd.DataFrame({'use': u, 'intensity_of_cap': LEVELS, 'cut_share_min_pct': lo,
                                    'cut_share_median_pct': md, 'cut_share_max_pct': hi}))
    axd.set_xlim(-2, 77)
    axd.set_ylim(0, 35)
    axd.set_xticks(xx)
    axd.set_yticks([0, 10, 20, 30])
    axd.tick_params(labelsize=5.2)
    axd.yaxis.tick_right()
    axd.yaxis.set_label_position('right')
    axd.spines['left'].set_visible(False)
    axd.spines['right'].set_visible(True)
    axd.set_xlabel('Inference (% of cap)', fontsize=6, labelpad=1)
    axd.set_ylabel('Cards cut (% of fleet)', fontsize=6, labelpad=2)
    guides(axd, 'y')
    panel(fig, 'd', .012, (122 + EXTRA) / H)

    # ================= 4.4 =================
    # e 2035 乐观路径下协同节省相对 2024 的倍数（版式同 a，柱以 1 为基线）
    w = lg.pivot_table(index=['grid_id', 'scenario'], columns='cost_scenario', values='Vmwh')
    ratio = w.div(w['base-2024'], axis=0)
    rmap = ratio['2035-advanced'].groupby(level=0).median()
    bounds_e = np.array([.4, .5, .6, .7, .8, .9, 1.0, 1.1, 1.2])
    cmap_e = mpl.colors.ListedColormap(list(DIV(np.linspace(0, .42, 6))) + list(DIV(np.linspace(.66, .95, 2))))
    cmap_e.set_over(DIV(1.0))
    cmap_e.set_under(DIV(0.))
    norm_e = mpl.colors.BoundaryNorm(bounds_e, cmap_e.N)
    ax_e, pos_e, lon_e, lat_e = marginal_map(fig, [.41, (74 + EXTRA) / H, .46, 34 / H], rmap, cmap_e, norm_e, (.5, 1.2),
                                             [.6, .8, 1.0, 1.2], '× 2024', '× 2024', base=1.,
                                             hist_ticks=[.4, .8, 1.2], med_fmt='{:.2f}', top_h=.04,
                                             side_w=.09)
    panel(fig, 'e', .36, (122 + EXTRA) / H)

    # f 三条路径的轨迹：横为协同节省倍数（误差线为碳价 0–200），纵为占供能成本的比例（不接推理）
    lr = pd.read_csv(SUPP / 'learning-ratio-all.csv')
    lr = lr[lr.intensity_of_max.eq(0)].set_index(['cost_scenario', 'carbon_price_usd_t']).ratio_median
    pct = pd.read_csv(SEC4 / 'learning-pct-all.csv')
    pct0 = pct[pct.intensity_of_max.eq(0)].set_index('cost_scenario')
    axf = fig.add_axes([.07, (9 + EXTRA) / H, .25, 40 / H])
    rows_f = []
    for path, pc in PATHS.items():
        cs_ = ['base-2024', f'2030-{path}', f'2035-{path}']
        xs = [1.0] + [lr[(c, 100.)] for c in cs_[1:]]
        lo = [1.0] + [min(lr[(c, 0.)], lr[(c, 200.)]) for c in cs_[1:]]
        hi = [1.0] + [max(lr[(c, 0.)], lr[(c, 200.)]) for c in cs_[1:]]
        ys_ = [pct0.loc[c, 'pct_median'] for c in cs_]
        for k in range(2):
            axf.annotate('', xy=(xs[k + 1], ys_[k + 1]), xytext=(xs[k], ys_[k]),
                         arrowprops=dict(arrowstyle='-|>', color=pc, lw=1.1, mutation_scale=6, shrinkA=2.5,
                                         shrinkB=2.5))
        axf.errorbar(xs[1:], ys_[1:], xerr=[np.subtract(xs, lo)[1:], np.subtract(hi, xs)[1:]], fmt='none',
                     ecolor=pc, elinewidth=.6, capsize=1.2, alpha=.8)
        axf.scatter(xs[1:], ys_[1:], s=[14, 22], color=pc, edgecolors='white', lw=.5, zorder=3)
        dx, dy, ha = {'advanced': (0, .3, 'center'), 'moderate': (0, .3, 'center'),
                      'conservative': (.025, 0, 'left')}[path]
        sep = '\n' if path == 'conservative' else ' '
        axf.text(xs[-1] + dx, ys_[-1] + dy, f'{path.capitalize()}{sep}2035',
                 fontsize=4.8, color=pc, ha=ha, linespacing=1.05,
                 va='bottom' if dy > 0 else 'center', fontweight='bold')
        rows_f.append(pd.DataFrame({'path': path, 'cost_scenario': cs_, 'ratio_c100': xs, 'ratio_low': lo,
                                    'ratio_high': hi, 'pct_of_supply': ys_}))
    axf.scatter(1, pct0.loc['base-2024', 'pct_median'], marker='*', s=40, color='black', edgecolors='white', lw=.4,
                zorder=4)
    axf.text(1.01, pct0.loc['base-2024', 'pct_median'] - .25, '2024', fontsize=4.8, ha='left', va='top')
    axf.axvline(1, color='#9A9A9A', lw=.5, zorder=0)
    axf.set_xlim(.55, 1.2)
    axf.set_ylim(8, 12)
    axf.tick_params(labelsize=5.2)
    axf.set_xlabel('Coordination saving (× 2024)', fontsize=6, labelpad=1)
    axf.set_ylabel('Saving (% of supply cost)', fontsize=6, labelpad=1)
    guides(axf, 'both')
    panel(fig, 'f', .012, (58 + EXTRA) / H)

    # g 二维直方图：不做协同时的储能时长 vs 协同节省倍数（六条路径合并）
    lg['store_h'] = lg.batt_seq / (lg.E0 / 8760)
    brat = lg.pivot_table(index=['grid_id', 'scenario'], columns='cost_scenario', values='store_h')
    paths6 = [c for c in ratio.columns if c != 'base-2024']
    xs = brat[paths6].to_numpy().ravel()
    ys = ratio[paths6].to_numpy().ravel()
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    axg = fig.add_axes([.41, (9 + EXTRA) / H, .24, 40 / H])
    hh = axg.hist2d(xs, ys, bins=[np.linspace(0, 16, 33), np.linspace(0, 1.6, 33)], cmap=SEQ,
                    norm=mpl.colors.LogNorm(1, 3e4), cmin=1, rasterized=True)
    edges = np.linspace(0, 16, 17)
    mid = (edges[:-1] + edges[1:]) / 2
    gb = pd.Series(ys).groupby(pd.cut(xs, edges), observed=False)
    med = gb.median().to_numpy()
    okm = gb.size().to_numpy() >= 200
    axg.plot(mid[okm], med[okm], color='black', lw=1.3, zorder=3)
    axg.axhline(1, color='#9A9A9A', lw=.5, zorder=2)
    rows_g = []
    for cs_ in paths6:
        yr, path = cs_.split('-')
        bx, by = brat[cs_].median(), ratio[cs_].median()
        axg.scatter(bx, by, marker='o' if yr == '2030' else '^', s=18, color=PATHS[path], edgecolors='white',
                    lw=.5, zorder=4)
        rows_g.append({'cost_scenario': cs_, 'storage_hours_median': bx, 'saving_ratio_median': by})
    axg.set_xlim(0, 16)
    axg.set_ylim(0, 1.6)
    axg.tick_params(labelsize=5.2)
    axg.set_xlabel('Storage without coordination (h)', fontsize=6, labelpad=1)
    axg.set_ylabel('Coordination saving (× 2024)', fontsize=6, labelpad=1)
    axg.legend(handles=[Line2D([], [], marker='o', ls='none', color='#777777', markersize=3, label='2030'),
                        Line2D([], [], marker='^', ls='none', color='#777777', markersize=3.2, label='2035'),
                        Line2D([], [], color='black', lw=1.3, label='Binned median')],
               loc='lower left', fontsize=4.8, handletextpad=.3, labelspacing=.25, borderaxespad=.3, frameon=True,
               framealpha=.85, edgecolor='none')
    cax = fig.add_axes([.655, (9 + EXTRA) / H, .005, 40 / H])
    cbh = fig.colorbar(hh[3], cax=cax)
    cbh.ax.tick_params(labelsize=4.6, length=1.5, pad=1)
    cbh.set_label('Cases', fontsize=5.2, labelpad=1)
    panel(fig, 'g', .35, (58 + EXTRA) / H)

    # h 气泡图：各成本情景的供能成本中位数 vs 碳价 0→200 放大倍数（气泡大小 = 单调上升比例）
    ca = pd.read_csv(SUPP / 'carbon-amplification-all.csv')
    ca = ca[ca.intensity_of_max.eq(0)].merge(pct0.reset_index(), on='cost_scenario')
    axh = fig.add_axes([.76, (9 + EXTRA) / H, .22, 40 / H])
    for _, r in ca.iterrows():
        cs_ = r.cost_scenario
        colr = 'black' if cs_ == 'base-2024' else PATHS[cs_.split('-')[1]]
        mk = '*' if cs_ == 'base-2024' else ('o' if cs_.startswith('2030') else '^')
        axh.scatter(r.supply_cost_median, r.amp_median, s=10 + (r.monotone_share - .75) * 300, marker=mk,
                    color=colr, alpha=.85, edgecolors='white', lw=.5, zorder=3)
        lab = '2024' if cs_ == 'base-2024' else cs_.split('-')[0]
        left = cs_ == '2035-conservative'
        axh.text(r.supply_cost_median + (-1.6 if left else 1.6), r.amp_median - (.04 if left else 0), lab,
                 fontsize=4.6, va='center', ha='right' if left else 'left', color=colr)
    axh.set_xlim(70, 118)
    axh.set_ylim(1, 1.8)
    axh.tick_params(labelsize=5.2)
    axh.set_xlabel('Supply cost (USD/MWh)', fontsize=6, labelpad=1)
    axh.set_ylabel('Amplification, 0→200 USD/t', fontsize=6, labelpad=1)
    guides(axh, 'both')
    axh.legend(handles=[Line2D([], [], color=pc, lw=0, marker='o', ms=3.4, label=p.capitalize())
                        for p, pc in PATHS.items()],
               loc='upper left', fontsize=4.8, handletextpad=.2, labelspacing=.25, borderaxespad=.3)
    panel(fig, 'h', .7, (58 + EXTRA) / H)

    rows_i = panel_i(fig)
    panel(fig, 'i', .012, (EXTRA - 1) / H)

    export(fig, 'fig4')
    csv(pd.DataFrame({'grid_id': dv.index, 'extra_saving_c200_vs_c0_usd_mwh': dv.values}), 'fig4a')
    csv(out['b'], 'fig4b')
    csv(out['c'], 'fig4c')
    csv(pd.concat(rows_d), 'fig4d')
    csv(pd.DataFrame({'grid_id': rmap.index, 'saving_ratio_2035_advanced': rmap.values}), 'fig4e')
    csv(pd.concat(rows_f), 'fig4f')
    csv(pd.DataFrame(rows_g), 'fig4g')
    csv(ca[['cost_scenario', 'supply_cost_median', 'amp_median', 'monotone_share']], 'fig4h')
    csv(rows_i, 'fig4i')
    out['a'] = dv.describe()
    out['e'] = rmap.describe()
    return out


if __name__ == '__main__':
    o = main()
    for k, v in o.items():
        print('==', k)
        print(v.round(3) if hasattr(v, 'round') else v)
