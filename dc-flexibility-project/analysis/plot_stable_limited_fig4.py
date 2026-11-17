"""图4：碳价和技术成本变化不改变“协同收益稳定但有限”的判断（第四节）。

a 双变量地图：当地电网碳因子 × 碳价 0→200 的协同节省放大倍数（各网格取 11 类机群中位数），3×3 分位色块；
b 云雨图：推理加到上限 75% 时，五类用途在碳价 0/100/200 下的协同节省分布（半边密度 + 区间 + 散点）；
c 地图 + 经度统计条：2035 乐观供能成本路径下协同节省相对 2024 的倍数；经度条为三条路径 2035 年的中位数；
d 六边形密度：不做协同时新建储能相对 2024 的倍数 vs 协同节省倍数（六条路径合并），黑线为分档中位数。
逐网格结果来自 PR #80 的服务器结果（本地 D:/codex-runs/sec4-20261116），先提取为 INPUT/sec4/fig4-grid-*.parquet。
"""
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr

from stable_limited_figs_common import (COLORS, DARK, DIV, GROUP, INPUT, SEQ, colorbar, csv, draw_map, export,
                                        guides, longitude_strip, paired, panel, setup)

SEC4 = INPUT / 'sec4'
RAW = Path('D:/codex-runs/sec4-20261116')
USES = ['HPC', 'CLD', 'SAI', 'TRN', 'INF']
PATHS = {'conservative': '#93CDB9', 'moderate': '#3F93A8', 'advanced': '#1B3F70'}
# 3×3 双变量色（行：放大倍数低→高；列：碳因子低→高）
BIV = [['#F3EEE2', '#E7BFA6', '#D58A62'],
       ['#B9DCD6', '#B7A99A', '#A86A50'],
       ['#5BA5AA', '#5E8A86', '#6E4E44']]


def lighten(c, f):
    return tuple(np.array(mpl.colors.to_rgb(c)) * (1 - f) + f)


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


def biv_legend(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-.1, 3.1)
    ax.set_aspect('equal')
    ax.axis('off')
    s = .5
    for i in range(3):          # 放大倍数
        for j in range(3):      # 碳因子
            cx, cy = (j - i) * s, (i + j) * s + s
            ax.add_patch(Polygon([(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)], closed=True,
                                 facecolor=BIV[i][j], edgecolor='white', lw=.6))
    ax.text(1.62, 1.5, 'Grid carbon\nintensity', fontsize=5.2, ha='left', va='center', linespacing=.95)
    ax.text(-1.62, 1.5, 'Amplification\n0→200 USD/t', fontsize=5.2, ha='right', va='center', linespacing=.95)


def main():
    setup()
    prepare()
    cg = pd.read_parquet(SEC4 / 'fig4-grid-carbon.parquet')
    lg = pd.read_parquet(SEC4 / 'fig4-grid-lc.parquet')
    loc = paired().drop_duplicates('grid_id').set_index('grid_id')[['lat', 'lon']]
    fig = plt.figure(figsize=(180 / 25.4, 185 / 25.4))

    # ---------------- a 双变量地图
    c0 = cg[cg.intensity_of_max.eq(0)].pivot_table(index=['grid_id', 'scenario'], columns='carbon_price_usd_t',
                                                    values='V')
    amp = (c0[200] / c0[0]).groupby(level=0).median()
    ef = cg.groupby('grid_id').ef.first().loc[amp.index]
    qa = pd.qcut(amp, 3, labels=False)
    qe = pd.qcut(ef.rank(method='first'), 3, labels=False)
    col = pd.Series([BIV[i][j] for i, j in zip(qa, qe)], index=amp.index)
    ax_a, _ = draw_map(fig, [.005, .615, .62, .355], amp, colors=col, projection=ccrs.EqualEarth(), lon_labels=False)
    y_lo = ax_a.projection.transform_point(0, -58, ccrs.PlateCarree())[1]
    ax_a.set_ylim(y_lo, ax_a.get_ylim()[1])
    biv_legend(fig, [.05, .56, .2, .115])
    rho_a = spearmanr(amp, ef)[0]
    panel(fig, 'a', .012, .985)

    # ---------------- b 云雨图：推理 75%，碳价 0/100/200
    q75 = cg[cg.intensity_of_max.eq(.75)].copy()
    q75['use'] = q75.scenario.str[2:]
    ax_b = fig.add_axes([.74, .58, .245, .39])
    rng = np.random.default_rng(7)
    xmax = 30
    grid_x = np.linspace(0, xmax, 300)
    ticks, labels, rows_b = [], [], []
    y = 0.0
    for u in USES[::-1]:
        y0 = y
        for k, cp in enumerate([0, 100, 200]):
            v = q75.loc[q75.use.eq(u) & q75.carbon_price_usd_t.eq(cp), 'V'].to_numpy()
            face = lighten(COLORS[u], [.62, .32, 0][k])
            dens = gaussian_kde(np.clip(v, 0, xmax), bw_method=.18)(grid_x)
            dens = dens / dens.max() * .32
            ax_b.fill_between(grid_x, y, y + dens, color=face, lw=0, zorder=2)
            p10, p50, p90 = np.percentile(v, [10, 50, 90])
            ax_b.plot([p10, p90], [y - .06] * 2, color=DARK, lw=.7, zorder=3)
            ax_b.scatter(p50, y - .06, s=7, color='white', edgecolors=DARK, lw=.5, zorder=4)
            ax_b.plot([p10, p10], [y - .1, y - .02], color=DARK, lw=.9, zorder=4)
            pts = rng.choice(v, size=min(160, len(v)), replace=False)
            ax_b.scatter(np.clip(pts, 0, xmax), y - .16 + rng.uniform(-.035, .035, len(pts)), s=.6, color=face,
                         lw=0, zorder=1, rasterized=True)
            rows_b.append({'use': u, 'carbon_price': cp, 'p10': p10, 'median': p50, 'p90': p90, 'n': len(v)})
            y += .48
        ticks.append((y0 + y - .48) / 2 + .08)
        labels.append(u)
        y += .32
    ax_b.set_yticks(ticks, [GROUP[u] for u in labels], fontsize=5.6)
    for t, u in zip(ax_b.get_yticklabels(), labels):
        t.set_color(COLORS[u])
        t.set_fontweight('bold')
    ax_b.tick_params(axis='y', length=0)
    ax_b.spines['left'].set_visible(False)
    ax_b.set_xlim(0, xmax)
    ax_b.set_ylim(-.32, y - .2)
    ax_b.tick_params(axis='x', labelsize=5.4)
    ax_b.set_xlabel('Saving at 75% inference (USD/MWh)', fontsize=6)
    guides(ax_b)
    ax_b.legend(handles=[Patch(facecolor=lighten('#5A5A5A', f), label=f'{cp} USD/t')
                         for cp, f in zip([0, 100, 200], [.62, .32, 0])] +
                [Line2D([], [], color=DARK, lw=.9, marker='|', markersize=4, label='P10')],
                loc='upper right', fontsize=5, handlelength=1.1, labelspacing=.3, borderaxespad=.2, frameon=True,
                framealpha=.9, edgecolor='none')
    panel(fig, 'b', .66, .985)

    # ---------------- c 地图 + 经度统计条：2035 乐观路径的协同节省倍数
    w = lg.pivot_table(index=['grid_id', 'scenario'], columns='cost_scenario', values='Vmwh')
    ratio = w.div(w['base-2024'], axis=0)
    rmap = ratio['2035-advanced'].groupby(level=0).median()
    cnorm = mpl.colors.TwoSlopeNorm(1, .4, 1.6)
    ax_c, _ = draw_map(fig, [.075, .1, .57, .29], rmap, DIV, cnorm, lon_labels=False)
    frames = []
    for path, pc in PATHS.items():
        g = ratio[f'2035-{path}'].groupby(level=0).median()
        frames.append((path, pc, pd.DataFrame({'lon': loc.loc[g.index, 'lon'].to_numpy(), 'value': g.to_numpy()}),
                       path == 'advanced'))
    ax_s = longitude_strip(fig, ax_c, .07, frames, 'Saving\n(× 2024)', ylim=(.4, 1.4), band=8)
    ax_s.axhline(1, color='#9A9A9A', lw=.5)
    ax_s.legend(handles=[Line2D([], [], color=pc, lw=1, label=f'2035 {p}') for p, pc in PATHS.items()],
                loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, fontsize=4.9, handlelength=1.2, columnspacing=.8,
                borderaxespad=0)
    colorbar(fig, [.18, .085, .3, .007], DIV, cnorm, [.4, .7, 1, 1.3, 1.6],
             'Coordination saving, 2035 advanced costs ÷ 2024', extend='both')
    panel(fig, 'c', .012, .52)

    # ---------------- d 六边形密度：储能倍数 vs 协同节省倍数
    lg['store_h'] = lg.batt_seq / (lg.E0 / 8760)
    brat = lg.pivot_table(index=['grid_id', 'scenario'], columns='cost_scenario', values='store_h')
    paths6 = [c for c in ratio.columns if c != 'base-2024']
    xs = brat[paths6].to_numpy().ravel()
    ys = ratio[paths6].to_numpy().ravel()
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    ax_d = fig.add_axes([.74, .1, .23, .34])
    hb = ax_d.hexbin(xs, ys, gridsize=(34, 22), extent=(0, 16, 0, 1.6), cmap=SEQ, bins='log', mincnt=1, lw=.1,
                     edgecolors='white', zorder=1)
    edges = np.linspace(0, 16, 17)
    mid = (edges[:-1] + edges[1:]) / 2
    g = pd.Series(ys).groupby(pd.cut(xs, edges), observed=False)
    med = g.median().to_numpy()
    okm = g.size().to_numpy() >= 200
    ax_d.plot(mid[okm], med[okm], color='black', lw=1.4, zorder=3)
    ax_d.axhline(1, color='#9A9A9A', lw=.5, zorder=2)
    rows_d = []
    for cs_ in paths6:
        yr, path = cs_.split('-')
        bx = brat[cs_].median()
        by = ratio[cs_].median()
        ax_d.scatter(bx, by, marker='o' if yr == '2030' else '^', s=22, color=PATHS[path], edgecolors='white',
                     lw=.5, zorder=4)
        rows_d.append({'cost_scenario': cs_, 'storage_hours_median': bx, 'saving_ratio_median': by})
    ax_d.set_xlim(0, 16)
    ax_d.set_ylim(0, 1.6)
    ax_d.tick_params(labelsize=5.4)
    ax_d.set_xlabel('Storage built without coordination (h of mean load)', fontsize=6)
    ax_d.set_ylabel('Coordination saving (× 2024)', fontsize=6)
    ax_d.legend(handles=[Line2D([], [], marker='s', ls='none', color=pc, markersize=3.5, label=p.capitalize())
                         for p, pc in PATHS.items()] +
                [Line2D([], [], marker='o', ls='none', color='#777777', markersize=3.2, label='2030'),
                 Line2D([], [], marker='^', ls='none', color='#777777', markersize=3.4, label='2035'),
                 Line2D([], [], color='black', lw=1.4, label='Binned median')],
                loc='lower left', fontsize=4.9, handletextpad=.3, labelspacing=.25, borderaxespad=.3, frameon=True,
                framealpha=.85, edgecolor='none', ncol=2, columnspacing=.6)
    cax = fig.add_axes([.975, .1, .005, .34])
    cbd = fig.colorbar(hb, cax=cax)
    cbd.ax.tick_params(labelsize=4.8, length=1.5, pad=1)
    cbd.set_label('Cases', fontsize=5.2, labelpad=1)
    panel(fig, 'd', .66, .52)

    export(fig, 'fig4')
    csv(pd.DataFrame({'grid_id': amp.index, 'amplification_200_over_0': amp.values, 'grid_ef_t_mwh': ef.values,
                      'amp_tercile': qa.values, 'ef_tercile': qe.values}), 'fig4a')
    csv(pd.DataFrame(rows_b), 'fig4b')
    csv(pd.DataFrame({'grid_id': rmap.index, 'saving_ratio_2035_advanced': rmap.values}), 'fig4c')
    csv(pd.DataFrame(rows_d), 'fig4d')
    return rho_a, pd.DataFrame(rows_b), pd.DataFrame(rows_d), med[okm]


if __name__ == '__main__':
    rho_a, b, d, med = main()
    print('rho amp~ef %.2f' % rho_a)
    print(b.round(2).to_string())
    print(d.round(2).to_string())
    print(np.round(med, 2))
