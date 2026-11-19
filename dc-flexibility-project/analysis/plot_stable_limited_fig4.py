"""图4：碳价和技术成本变化不改变“协同收益稳定但有限”的判断（第四节）。按子论点排布：

4.1 碳价提高会稳定放大协同收益（但不是减排）
  a 双变量地图：当地电网碳因子 × 碳价 0→200 的协同节省放大倍数（各网格取 11 类机群中位数）；
  b 碳价（纵轴）下各用途协同节省与协同减排的中位数（不接推理）；
4.2 碳价越高，接入推理后剩余的协同收益越稳
  c 碳价（纵轴）下四档推理强度的协同节省下十分位（全部网格与机群）；
4.3 卡价和待机功耗下降不会让为柔性加卡变得划算
  d 卡价倍数 × 新增卡待机比例的检验网格（格内为加卡算例数），叠加学习曲线路径；
  e 允许增减卡时的减卡比例随推理强度的变化（八城市，PR #83）；
4.4 风光成本下降影响有限，储能大幅降价会部分替代协同
  f 地图 + 经度统计条：2035 乐观路径下协同节省相对 2024 的倍数；
  g 三条成本路径下协同节省倍数（带碳价 0–200 范围，PR #83 交叉）与占供能成本的比例随年份变化；
  h 不做协同时新建储能时长 vs 协同节省倍数（二维直方图）；
  i 各成本情景的供能成本中位数 vs 碳价 0→200 的放大倍数（气泡大小为单调上升比例，PR #83 交叉）。
逐网格结果来自 PR #80 的服务器结果（本地 D:/codex-runs/sec4-20261116），先提取为 INPUT/sec4/ 下的小表。
"""
import json
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stable_limited_figs_common import (COLORS, DARK, DIV, GROUP, INNER, INPUT, ROOT, SCENARIOS, SEQ, csv, draw_map,
                                        export, guides, longitude_strip, paired, panel, setup, shade, use)

SEC4 = INPUT / 'sec4'
SUPP = ROOT / 'results/sec4-supplement-20261118'
RAW = Path('D:/codex-runs/sec4-20261116')
USES = ['HPC', 'CLD', 'SAI', 'TRN', 'INF']
PATHS = {'conservative': '#93CDB9', 'moderate': '#3F93A8', 'advanced': '#1B3F70'}
LEVELS = [0, .25, .5, .75]
GRAYS = ['#C4C4C4', '#8F8F8F', '#555555', '#000000']
MARKS = {'A': 'D', 'B': 'o', 'C': 's', 'D': '^'}
H = 250
# 3×3 双变量色（行：放大倍数低→高；列：碳因子低→高）
BIV = [['#F3EEE2', '#E7BFA6', '#D58A62'],
       ['#B9DCD6', '#B7A99A', '#A86A50'],
       ['#5BA5AA', '#5E8A86', '#6E4E44']]


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


def biv_legend(ax, x0, y0, s):
    """在地图轴的投影坐标中画 3×3 菱形图例（左下角 (x0, y0)，菱形半对角线 s）。"""
    for i in range(3):          # 放大倍数
        for j in range(3):      # 碳因子
            cx, cy = x0 + (j - i) * s, y0 + (i + j) * s + s
            ax.add_patch(Polygon([(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)], closed=True,
                                 facecolor=BIV[i][j], edgecolor='white', lw=.5, zorder=8))
    ax.text(x0 + 1.75 * s, y0 + 2.6 * s, 'Grid carbon\nintensity →', fontsize=5, ha='left', va='center',
            linespacing=.95, zorder=8)
    ax.text(x0 - 1.75 * s, y0 + 2.6 * s, '← Amplification\n0→200 USD/t', fontsize=5, ha='right', va='center',
            linespacing=.95, zorder=8)


def carbon_axis(ax, label=True):
    ax.set_ylim(-25, 225)
    ax.set_yticks([0, 100, 200])
    if label:
        ax.set_ylabel('Carbon price (USD/t)', fontsize=6, labelpad=1)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=5.4)
    guides(ax, 'x')


def main():
    setup()
    prepare()
    cg = pd.read_parquet(SEC4 / 'fig4-grid-carbon.parquet')
    cg['use'] = cg.scenario.str[2:]
    lg = pd.read_parquet(SEC4 / 'fig4-grid-lc.parquet')
    loc = paired().drop_duplicates('grid_id').set_index('grid_id')[['lat', 'lon']]
    fig = plt.figure(figsize=(180 / 25.4, H / 25.4))
    out = {}

    # ================= 4.1 =================
    # a 双变量地图（图例放进南太平洋海域）
    c0 = cg[cg.intensity_of_max.eq(0)].pivot_table(index=['grid_id', 'scenario'], columns='carbon_price_usd_t',
                                                    values='V')
    amp = (c0[200] / c0[0]).groupby(level=0).median()
    ef = cg.groupby('grid_id').ef.first().loc[amp.index]
    qa = pd.qcut(amp, 3, labels=False)
    qe = pd.qcut(ef.rank(method='first'), 3, labels=False)
    col = pd.Series([BIV[i][j] for i, j in zip(qa, qe)], index=amp.index)
    ax_a, _ = draw_map(fig, [.005, 186 / H, .6, 60 / H], amp, colors=col,
                       projection=ccrs.EqualEarth(), lon_labels=False)
    y_lo = ax_a.projection.transform_point(0, -58, ccrs.PlateCarree())[1]
    ax_a.set_ylim(y_lo, ax_a.get_ylim()[1])
    x0, y0 = ax_a.projection.transform_point(-150, -52, ccrs.PlateCarree())
    biv_legend(ax_a, x0, y0, 5.2e5)
    out['rho_amp_ef'] = spearmanr(amp, ef)[0]
    panel(fig, 'a', .012, 1 - 3 / H)

    # b 碳价下各用途节省与减排的中位数（不接推理）：左节省、右减排
    z = cg[cg.intensity_of_max.eq(0)]
    rows_b = []
    axb1 = fig.add_axes([.69, 200 / H, .13, 42 / H])
    axb2 = fig.add_axes([.855, 200 / H, .13, 42 / H])
    for u in USES:
        g = z[z.use.eq(u)].groupby('carbon_price_usd_t')
        v, e = g.V.median(), g.dEm.median() * 1000
        axb1.plot(v.values, v.index, color=COLORS[u], lw=1, marker='o', ms=2.6, mec='white', mew=.4)
        axb2.plot(e.values, e.index, color=COLORS[u], lw=1, marker='o', ms=2.6, mec='white', mew=.4)
        rows_b.append(pd.DataFrame({'use': u, 'carbon_price_usd_t': v.index, 'saving_median_usd_mwh': v.values,
                                    'emission_cut_median_kg_mwh': e.values}))
    allv = z.groupby('carbon_price_usd_t').V.median()
    alle = z.groupby('carbon_price_usd_t').dEm.median() * 1000
    axb1.plot(allv.values, allv.index, color='black', lw=1.4, ls=(0, (2, 1)))
    axb2.plot(alle.values, alle.index, color='black', lw=1.4, ls=(0, (2, 1)))
    out['b_all'] = pd.DataFrame({'saving': allv, 'emission_cut_kg': alle})
    carbon_axis(axb1)
    carbon_axis(axb2, label=False)
    axb1.set_xlim(0, 20)
    axb2.set_xlim(0, 50)
    axb1.set_xlabel('Saving (USD/MWh)', fontsize=6, labelpad=1)
    axb2.set_xlabel('Emission cut (kg CO$_2$/MWh)', fontsize=6, labelpad=1)
    axb1.legend(handles=[Line2D([], [], color=COLORS[u], lw=1, label=GROUP[u]) for u in USES] +
                [Line2D([], [], color='black', lw=1.4, ls=(0, (2, 1)), label='All fleets')],
                loc='upper center', bbox_to_anchor=(1.13, -.3), ncol=3, fontsize=4.9, handlelength=1.2,
                columnspacing=.7, labelspacing=.25, borderaxespad=0)
    panel(fig, 'b', .64, 1 - 3 / H)

    # ================= 4.2 =================
    # c 碳价下四档推理强度的下十分位（全部网格与机群）
    ax_c = fig.add_axes([.075, 133 / H, .22, 38 / H])
    rows_c = []
    for lv, gc in zip(LEVELS, GRAYS):
        p10 = cg[cg.intensity_of_max.eq(lv)].groupby('carbon_price_usd_t').V.quantile(.1)
        ax_c.plot(p10.values, p10.index, color=gc, lw=1.2, marker='o', ms=2.8, mec='white', mew=.4)
        rows_c.append(pd.DataFrame({'intensity_of_cap': lv, 'carbon_price_usd_t': p10.index, 'p10_usd_mwh': p10.values}))
    rc = pd.concat(rows_c)
    w = rc.pivot(index='intensity_of_cap', columns='carbon_price_usd_t', values='p10_usd_mwh')
    out['c_amp'] = (w[200] / w[0])
    ax_c.legend(handles=[Line2D([], [], color=gc, lw=1.2, label=f'{lv:.0%}') for lv, gc in zip(LEVELS, GRAYS)],
                title='Inference (% of cap)', loc='lower right', fontsize=4.9, title_fontsize=4.9, handlelength=1.2,
                labelspacing=.25, borderaxespad=.2, frameon=True, framealpha=.9, edgecolor='none')
    carbon_axis(ax_c)
    ax_c.set_xlim(1.5, 4.5)
    ax_c.set_xlabel('Lower decile of saving (USD/MWh)', fontsize=6, labelpad=1)
    panel(fig, 'c', .012, 178 / H)

    # ================= 4.3 =================
    # d 卡价倍数 × 待机比例的检验网格，格内为加卡算例数；叠加学习曲线路径
    add = pd.read_csv(SUPP / 'card-city/add-cases.csv')
    cnt = add.groupby(['m', 'idle_scale'])[['cases', 'cases_adding_cards']].sum().reset_index()
    ax_d = fig.add_axes([.42, 133 / H, .2, 38 / H])
    ms_ = sorted(cnt.m.unique())
    ss_ = sorted(cnt.idle_scale.unique())
    xe = np.exp(np.r_[np.log(ms_[0]) - .45, (np.log(ms_[:-1]) + np.log(ms_[1:])) / 2, np.log(ms_[-1]) + .3])
    ye = np.r_[-.25, .25, .75, 1.25]
    for _, r in cnt.iterrows():
        i, j = ms_.index(r.m), ss_.index(r.idle_scale)
        ax_d.add_patch(mpl.patches.Rectangle((xe[i], ye[j]), xe[i + 1] - xe[i], ye[j + 1] - ye[j],
                                             facecolor='#E3EFEC' if r.cases_adding_cards == 0 else '#D58A62',
                                             edgecolor='white', lw=1.2, zorder=1))
        ax_d.text(np.sqrt(xe[i] * xe[i + 1]), ye[j] + .05,
                  f'{int(r.cases_adding_cards)}/{int(r.cases):,}', fontsize=4.3, ha='center', va='bottom',
                  color='#3F6F66', zorder=2)
    cp = json.load(open(ROOT / 'analysis/learning-curves-atb2025.json', encoding='utf-8'))['card_paths']
    for path, pc in PATHS.items():
        xs = [1] + [cp['card_price_per_mw'][path][y] for y in ('2030', '2035')]
        ys = [1] + [cp['standby_scale'][path][y] for y in ('2030', '2035')]
        ax_d.plot(xs, ys, color=pc, lw=1, zorder=3)
        ax_d.scatter(xs[1:], ys[1:], marker='o', s=[12, 18], color=pc, edgecolors='white', lw=.5, zorder=4)
    ax_d.scatter([1], [1], marker='*', s=40, color='black', edgecolors='white', lw=.5, zorder=5)
    ax_d.text(1.06, 1.12, '2024', fontsize=5, ha='left', va='bottom')
    ax_d.set_xscale('log')
    ax_d.set_xlim(xe[0], xe[-1])
    ax_d.set_ylim(ye[0], ye[-1])
    ax_d.set_xticks(ms_, [f'{m:g}×' for m in ms_])
    ax_d.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax_d.set_yticks(ss_, [f'{s:g}' for s in ss_])
    ax_d.tick_params(labelsize=5.4)
    ax_d.set_xlabel('Card price per MW (× 2024)', fontsize=6, labelpad=1)
    ax_d.set_ylabel('Standby power of added cards (× today)', fontsize=6, labelpad=1)
    ax_d.legend(handles=[Line2D([], [], color=pc, lw=1, marker='o', ms=2.6, label=p.capitalize())
                         for p, pc in PATHS.items()],
                loc='upper left', bbox_to_anchor=(0, -.2), ncol=3, fontsize=4.9, handlelength=1.1,
                columnspacing=.6, borderaxespad=0)
    out['d_total'] = (int(cnt.cases_adding_cards.sum()), int(cnt.cases.sum()))
    panel(fig, 'd', .335, 178 / H)

    # e 允许增减卡时的减卡比例随推理强度的变化（各卡价、待机下相同，取中位数）
    cut = pd.read_csv(SUPP / 'card-city/free-cut-share.csv')
    cs = cut.groupby(['scenario', 'intensity_of_max']).cut_share_median.median().unstack() * 100
    ax_e = fig.add_axes([.745, 133 / H, .2, 38 / H])
    xx = np.array(LEVELS) * 100
    for sc in SCENARIOS:
        yv = cs.loc[sc, LEVELS].to_numpy()
        ax_e.plot(xx, yv[0] * (1 - np.array(LEVELS)), color=shade(sc), lw=.5, ls=(0, (2, 1.5)), zorder=2)
        ax_e.plot(xx, yv, color=shade(sc), lw=.9, zorder=3)
        ax_e.scatter(xx, yv, marker=MARKS[sc[0]], s=9, color=COLORS[use(sc)], edgecolors='white', lw=.3, zorder=4)
    for u, yv in {'CLD': 33.5, 'TRN': 27.5, 'SAI': 21.8, 'HPC': 13.2, 'INF': 5.0}.items():
        ax_e.text(-2, yv, GROUP[u], color=COLORS[u], fontsize=5, fontweight='bold', ha='right', va='center')
    ax_e.set_xlim(-1, 78)
    ax_e.set_ylim(0, 36)
    ax_e.set_xticks(xx)
    ax_e.tick_params(labelsize=5.4)
    ax_e.set_yticks([0, 10, 20, 30])
    ax_e.yaxis.tick_right()
    ax_e.yaxis.set_label_position('right')
    ax_e.spines['left'].set_visible(False)
    ax_e.spines['right'].set_visible(True)
    ax_e.set_xlabel('Inference (% of cap)', fontsize=6, labelpad=1)
    ax_e.set_ylabel('Cards cut when allowed (% of fleet)', fontsize=6, labelpad=2)
    guides(ax_e, 'y')
    panel(fig, 'e', .64, 178 / H)

    # ================= 4.4 =================
    # f 地图 + 经度统计条：2035 乐观路径的协同节省倍数（离散分级，1 以下为赤陶色、以上为青色）
    w = lg.pivot_table(index=['grid_id', 'scenario'], columns='cost_scenario', values='Vmwh')
    ratio = w.div(w['base-2024'], axis=0)
    rmap = ratio['2035-advanced'].groupby(level=0).median()
    bounds = [.4, .5, .6, .7, .8, .9, 1.0, 1.1, 1.2, 1.4]
    fcols = list(DIV(np.linspace(0, .42, 6))) + list(DIV(np.linspace(.62, 1, 3)))
    fcmap = mpl.colors.ListedColormap(fcols)
    fnorm = mpl.colors.BoundaryNorm(bounds, fcmap.N)
    ax_f, _ = draw_map(fig, [.06, 60 / H, .55, 44 / H], rmap, fcmap, fnorm, lon_labels=False)
    frames = []
    for path, pc in PATHS.items():
        gg = ratio[f'2035-{path}'].groupby(level=0).median()
        frames.append((path, pc, pd.DataFrame({'lon': loc.loc[gg.index, 'lon'].to_numpy(), 'value': gg.to_numpy()}),
                       path == 'advanced'))
    ax_fs = longitude_strip(fig, ax_f, 13 / H, frames, 'Saving\n(× 2024)', ylim=(.5, 1.3), band=8)
    ax_fs.axhline(1, color='#9A9A9A', lw=.5)
    cax = fig.add_axes([.17, 57 / H, .33, 1.6 / H])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=fnorm, cmap=fcmap), cax=cax, orientation='horizontal',
                      ticks=bounds, spacing='uniform')
    cb.outline.set_linewidth(.3)
    cax.tick_params(labelsize=5, length=1.5, width=.4, pad=1)
    cb.set_label('Coordination saving, 2035 advanced costs ÷ 2024', fontsize=5.8, labelpad=1.5)
    panel(fig, 'f', .012, 122 / H)

    # g 三条路径随年份：上为协同节省倍数（线为碳价 100，色带为碳价 0–200），下为占供能成本的比例
    lr = pd.read_csv(SUPP / 'learning-ratio-all.csv')
    lr = lr[lr.intensity_of_max.eq(0)]
    pct = pd.read_csv(SEC4 / 'learning-pct-all.csv')
    axg1 = fig.add_axes([.71, 90 / H, .2, 24 / H])
    axg2 = fig.add_axes([.71, 61 / H, .2, 24 / H])
    years = [2024, 2030, 2035]
    rows_g = []
    for path, pc in PATHS.items():
        cs_ = ['base-2024', f'2030-{path}', f'2035-{path}']
        rr = {cp_: [1.0] + [lr[(lr.cost_scenario == c) & lr.carbon_price_usd_t.eq(cp_)].ratio_median.iloc[0]
                            for c in cs_[1:]] for cp_ in (0, 100, 200)}
        axg1.fill_between(years, rr[0], rr[200], color=pc, alpha=.18, lw=0)
        axg1.plot(years, rr[100], color=pc, lw=1.2, marker='o', ms=2.6, mec='white', mew=.4)
        for iv, ls in ((0, '-'), (.5, (0, (2, 1)))):
            pv = [pct[(pct.cost_scenario == c) & pct.intensity_of_max.eq(iv)].pct_median.iloc[0] for c in cs_]
            axg2.plot(years, pv, color=pc, lw=1.2 if iv == 0 else .8, ls=ls, marker='o' if iv == 0 else None,
                      ms=2.6, mec='white', mew=.4)
            rows_g.append(pd.DataFrame({'path': path, 'year': years, 'intensity_of_cap': iv, 'pct_of_supply': pv}))
        rows_g.append(pd.DataFrame({'path': path, 'year': years, 'ratio_c0': rr[0], 'ratio_c100': rr[100],
                                    'ratio_c200': rr[200]}))
        axg1.text(2035.6, rr[100][-1] + {'conservative': .05, 'moderate': -.04, 'advanced': 0}[path], path.capitalize(), color=pc, fontsize=5, fontweight='bold', va='center',
                  ha='left')
    axg1.axhline(1, color='#9A9A9A', lw=.5, zorder=0)
    axg1.set_ylim(.55, 1.2)
    axg1.set_ylabel('Saving\n(× 2024)', fontsize=5.8, labelpad=1)
    axg2.set_ylim(5, 12)
    axg2.set_ylabel('Saving (% of\nsupply cost)', fontsize=5.8, labelpad=1)
    axg2.legend(handles=[Line2D([], [], color=DARK, lw=1.2, label='No inference'),
                         Line2D([], [], color=DARK, lw=.8, ls=(0, (2, 1)), label='50% of cap')],
                loc='lower left', fontsize=4.8, handlelength=1.6, labelspacing=.2, borderaxespad=.2, ncol=2,
                columnspacing=.6)
    for ax in (axg1, axg2):
        ax.set_xlim(2022.5, 2035.5)
        ax.set_xticks(years)
        ax.tick_params(labelsize=5.4)
        guides(ax, 'y')
    axg1.set_xticklabels([])
    panel(fig, 'g', .64, 122 / H)

    # h 二维直方图：不做协同时的储能时长 vs 协同节省倍数（六条路径合并）
    lg['store_h'] = lg.batt_seq / (lg.E0 / 8760)
    brat = lg.pivot_table(index=['grid_id', 'scenario'], columns='cost_scenario', values='store_h')
    paths6 = [c for c in ratio.columns if c != 'base-2024']
    xs = brat[paths6].to_numpy().ravel()
    ys = ratio[paths6].to_numpy().ravel()
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    ax_h = fig.add_axes([.075, 9 / H, .37, 34 / H])
    hh = ax_h.hist2d(xs, ys, bins=[np.linspace(0, 16, 41), np.linspace(0, 1.6, 33)], cmap=SEQ,
                     norm=mpl.colors.LogNorm(1, 3e4), cmin=1, rasterized=True)
    edges = np.linspace(0, 16, 17)
    mid = (edges[:-1] + edges[1:]) / 2
    gb = pd.Series(ys).groupby(pd.cut(xs, edges), observed=False)
    med = gb.median().to_numpy()
    okm = gb.size().to_numpy() >= 200
    ax_h.plot(mid[okm], med[okm], color='black', lw=1.3, zorder=3)
    ax_h.axhline(1, color='#9A9A9A', lw=.5, zorder=2)
    ax_h.axvline(4, color=DARK, lw=.5, ls=(0, (2, 1.5)), zorder=2)
    rows_h = []
    for cs_ in paths6:
        yr, path = cs_.split('-')
        bx, by = brat[cs_].median(), ratio[cs_].median()
        ax_h.scatter(bx, by, marker='o' if yr == '2030' else '^', s=20, color=PATHS[path], edgecolors='white',
                     lw=.5, zorder=4)
        rows_h.append({'cost_scenario': cs_, 'storage_hours_median': bx, 'saving_ratio_median': by})
    ax_h.set_xlim(0, 16)
    ax_h.set_ylim(0, 1.6)
    ax_h.tick_params(labelsize=5.4)
    ax_h.set_xlabel('Storage built without coordination (h of mean load)', fontsize=6, labelpad=1)
    ax_h.set_ylabel('Saving (× 2024)', fontsize=6, labelpad=1)
    ax_h.legend(handles=[Line2D([], [], marker='o', ls='none', color='#777777', markersize=3, label='2030'),
                         Line2D([], [], marker='^', ls='none', color='#777777', markersize=3.2, label='2035'),
                         Line2D([], [], color='black', lw=1.3, label='Binned median')],
                loc='lower left', fontsize=4.8, handletextpad=.3, labelspacing=.25, borderaxespad=.3, frameon=True,
                framealpha=.85, edgecolor='none')
    cax = fig.add_axes([.455, 9 / H, .006, 34 / H])
    cbh = fig.colorbar(hh[3], cax=cax)
    cbh.ax.tick_params(labelsize=4.8, length=1.5, pad=1)
    cbh.set_label('Grid–fleet cases', fontsize=5.4, labelpad=1)
    panel(fig, 'h', .012, 48 / H)

    # i 气泡图：各成本情景的供能成本中位数 vs 碳价 0→200 的放大倍数（气泡大小 = 单调上升比例）
    ca = pd.read_csv(SUPP / 'carbon-amplification-all.csv')
    ca = ca[ca.intensity_of_max.eq(0)].merge(pct[pct.intensity_of_max.eq(0)], on='cost_scenario')
    ax_i = fig.add_axes([.62, 9 / H, .35, 34 / H])
    for _, r in ca.iterrows():
        cs_ = r.cost_scenario
        colr = 'black' if cs_ == 'base-2024' else PATHS[cs_.split('-')[1]]
        mk = '*' if cs_ == 'base-2024' else ('o' if cs_.startswith('2030') else '^')
        ax_i.scatter(r.supply_cost_median, r.amp_median, s=12 + (r.monotone_share - .75) * 360, marker=mk,
                     color=colr, alpha=.85, edgecolors='white', lw=.5, zorder=3)
        yr_, path_ = ('2024', '') if cs_ == 'base-2024' else cs_.split('-')
        lab = '2024' if not path_ else f'{path_.capitalize()} {yr_}'
        left = cs_ in ('2035-conservative',)
        ax_i.text(r.supply_cost_median + (-1.4 if left else 1.4), r.amp_median - (.035 if left else 0), lab,
                  fontsize=4.8, va='center', ha='right' if left else 'left', color=colr)
    ax_i.set_xlim(70, 118)
    ax_i.set_ylim(1, 1.8)
    ax_i.tick_params(labelsize=5.4)
    ax_i.set_xlabel('Median supply cost without coordination (USD/MWh)', fontsize=6, labelpad=1)
    ax_i.set_ylabel('Amplification, carbon 0→200', fontsize=6, labelpad=1)
    guides(ax_i, 'both')
    ax_i.legend(handles=[Line2D([], [], marker='o', ls='none', color='#9A9A9A', markersize=ms,
                                label=f'{lab} rise monotonically') for ms, lab in ((2.8, '77%'), (6.5, '98%'))],
                loc='upper left', fontsize=4.8, handletextpad=.3, labelspacing=.4, borderaxespad=.3)
    panel(fig, 'i', .53, 48 / H)

    export(fig, 'fig4')
    csv(pd.DataFrame({'grid_id': amp.index, 'amplification_200_over_0': amp.values, 'grid_ef_t_mwh': ef.values,
                      'amp_tercile': qa.values, 'ef_tercile': qe.values}), 'fig4a')
    csv(pd.concat(rows_b), 'fig4b')
    csv(rc, 'fig4c')
    csv(cnt, 'fig4d')
    csv(cs.reset_index(), 'fig4e')
    csv(pd.DataFrame({'grid_id': rmap.index, 'saving_ratio_2035_advanced': rmap.values}), 'fig4f')
    csv(pd.concat(rows_g), 'fig4g')
    csv(pd.DataFrame(rows_h), 'fig4h')
    csv(ca[['cost_scenario', 'supply_cost_median', 'amp_median', 'monotone_share']], 'fig4i')
    out['h_med'] = med[okm]
    out['i'] = ca[['cost_scenario', 'supply_cost_median', 'amp_median', 'monotone_share']]
    return out


if __name__ == '__main__':
    o = main()
    for k, v in o.items():
        print(k, v if not hasattr(v, 'round') else v.round(3))
