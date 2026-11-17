"""图4：碳价和技术成本变化不改变“协同收益稳定但有限”的判断（第四节）。

a 碳价 0/100/200 美元/吨下各机群协同节省中位数（不接推理），下方为协同带来的购电减排；
b 碳价 0→200 的节省放大倍数：机群 × 推理强度热图；
c 卡价倍数 × 新增卡待机比例的加卡检验：全部 7,392 个算例不加卡，叠加学习曲线路径；
d 供能学习曲线：供能成本 vs 协同节省（各路径），射线为协同节省占供能成本的比例。
数据来自第四节汇总（PR #80，results/sec4-carbon-learning-20261116），复制于 INPUT/sec4。
"""
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from stable_limited_figs_common import (COLORS, DARK, GROUP, INNER, INPUT, SCENARIOS, SEQ, csv, export, guides,
                                        panel, scenario_axis, setup, shade, use)

SEC4 = INPUT / 'sec4'
MARKS = {'A': 'D', 'B': 'o', 'C': 's', 'D': '^'}
PATHS = {'conservative': '#93CDB9', 'moderate': '#3F93A8', 'advanced': '#1B3F70'}


def main():
    setup()
    carbon = pd.read_csv(SEC4 / 'carbon-summary.csv')
    learn = pd.read_csv(SEC4 / 'learning-curve-summary.csv')
    cards = pd.read_csv(SEC4 / 'card-learning-city-summary.csv')
    lc = json.load(open(SEC4 / 'learning-curves-atb2025.json', encoding='utf-8'))
    fig = plt.figure(figsize=(180 / 25.4, 170 / 25.4))

    # ---------------- a 碳价斜率图 + 减排条
    c0 = carbon[carbon.intensity_of_max.eq(0)]
    med = c0.pivot(index='scenario', columns='carbon_price_usd_t', values='V_usd_per_mwh_median').loc[SCENARIOS]
    cut = c0.pivot(index='scenario', columns='carbon_price_usd_t', values='emission_cut_t_per_mwh_median').loc[SCENARIOS]
    prices = [0, 100, 200]
    ax_a = fig.add_axes([.085, .62, .26, .33])
    for sc in SCENARIOS:
        ax_a.plot(prices, med.loc[sc, prices], color=shade(sc), lw=.9, zorder=2)
        ax_a.scatter(prices, med.loc[sc, prices], marker=MARKS[sc[0]], s=14, color=COLORS[use(sc)],
                     edgecolors='white', lw=.4, zorder=3)
    for u in ('HPC', 'CLD', 'SAI', 'TRN', 'INF'):
        v = med.loc[[s for s in SCENARIOS if s.endswith(u)], 200].mean()
        ax_a.text(212, v, GROUP[u], color=COLORS[u], fontsize=5.4, fontweight='bold', va='center', ha='left')
    ax_a.set_xlim(-12, 212)
    ax_a.set_xticks(prices, [])
    ax_a.set_ylim(0, 21)
    ax_a.tick_params(labelsize=5.4)
    ax_a.set_ylabel('Coordination saving (USD/MWh)', fontsize=6)
    guides(ax_a, 'y')
    ax_a2 = fig.add_axes([.085, .52, .26, .085])
    q = cut.agg(['min', 'median', 'max'])
    ax_a2.fill_between(prices, q.loc['min'], q.loc['max'], color='#C9C9C9', lw=0)
    ax_a2.plot(prices, q.loc['median'], color=DARK, lw=1.1, marker='o', markersize=2.5)
    ax_a2.set_xlim(-12, 212)
    ax_a2.set_xticks(prices)
    ax_a2.set_ylim(0, .05)
    ax_a2.set_yticks([0, .02, .04])
    ax_a2.tick_params(labelsize=5.4)
    ax_a2.set_ylabel('Emission cut\n(t/MWh)', fontsize=5.6)
    ax_a2.set_xlabel('Carbon price (USD/t CO$_2$)', fontsize=6)
    guides(ax_a2, 'y')
    ax_a.legend(handles=[Line2D([], [], marker=mk, ls='none', color='#777777', markersize=3.2, label=INNER[k])
                         for k, mk in MARKS.items()],
                loc='upper left', fontsize=5, handletextpad=.2, labelspacing=.3, borderaxespad=.3, frameon=True,
                framealpha=.85, edgecolor='none')
    panel(fig, 'a', .012, .985)

    # ---------------- b 放大倍数热图：机群 × 推理强度
    m_all = carbon.pivot_table(index=['scenario', 'intensity_of_max'], columns='carbon_price_usd_t',
                               values='V_usd_per_mwh_median')
    ratio = (m_all[200] / m_all[0]).unstack().loc[SCENARIOS]
    ax_b = fig.add_axes([.71, .52, .21, .43])
    bnorm = mpl.colors.Normalize(1.3, 2.1)
    ax_b.imshow(ratio.to_numpy(), cmap=SEQ, norm=bnorm, aspect='auto', interpolation='nearest')
    for i in range(ratio.shape[0]):
        for j in range(ratio.shape[1]):
            v = ratio.iat[i, j]
            ax_b.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5,
                      color='white' if bnorm(v) > .6 else DARK)
    ax_b.set_xticks(range(4), ['0', '25', '50', '75'])
    ax_b.xaxis.set_ticks_position('top')
    ax_b.xaxis.set_label_position('top')
    ax_b.set_xlabel('External inference (% of cap)', fontsize=6, labelpad=3)
    ax_b.tick_params(length=0, labelsize=5.4)
    for sp in ax_b.spines.values():
        sp.set_visible(False)
    scenario_axis(ax_b, SCENARIOS, list(range(len(SCENARIOS))), outer=-.85, fontsize=5.3)
    cax = fig.add_axes([.935, .52, .008, .43])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=bnorm, cmap=SEQ), cax=cax)
    cb.outline.set_linewidth(.3)
    cax.tick_params(labelsize=5, length=1.5, pad=1)
    cb.set_label('Saving at 200 ÷ saving at 0 USD/t', fontsize=5.6, labelpad=2)
    panel(fig, 'b', .47, .985)

    # ---------------- c 加卡检验：卡价倍数 × 待机比例，叠加学习曲线路径
    ax_c = fig.add_axes([.085, .075, .33, .33])
    grid = cards.groupby(['m', 'idle_scale'])[['cases', 'cases_adding_cards']].sum().reset_index()
    ax_c.axvspan(.17, 2.4, ymin=0, ymax=1, color='#EEF6F4', lw=0, zorder=0)
    ax_c.scatter(grid.m, grid.idle_scale, s=46, marker='s', facecolors='white', edgecolors='#7FB5A8', lw=.7, zorder=2)
    for _, r in grid.iterrows():
        ax_c.text(r.m, r.idle_scale, f'{int(r.cases_adding_cards)}', ha='center', va='center', fontsize=5.2,
                  color='#2F6F62', zorder=3)
    cp = lc['card_paths']
    rows_c = []
    for path, col in PATHS.items():
        xs = [1.0] + [cp['card_price_per_mw'][path][y] for y in ('2030', '2035')]
        ys = [1.0] + [cp['standby_scale'][path][y] for y in ('2030', '2035')]
        ax_c.plot(xs, ys, color=col, lw=1.3, zorder=4)
        ax_c.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                      arrowprops=dict(arrowstyle='-|>', color=col, lw=1.3, mutation_scale=7), zorder=4)
        ax_c.scatter(xs[1:], ys[1:], s=14, color=col, zorder=5)
        rows_c.append(pd.DataFrame({'path': path, 'year': [2024, 2030, 2035], 'card_price_multiple': xs,
                                    'standby_scale': ys}))
    ax_c.scatter(1, 1, s=24, color=DARK, zorder=6)
    ax_c.text(1, 1.07, '2024', ha='center', va='bottom', fontsize=5.2)
    ax_c.text(.2, -.16, '0 of 7,392 cases add cards for flexibility', fontsize=5.6, fontweight='bold',
              color='#2F6F62', ha='left', va='center')
    ax_c.set_xscale('log')
    ax_c.set_xlim(.17, 2.4)
    ax_c.set_xticks([.2, .5, 1, 2], ['0.2', '0.5', '1', '2'])
    ax_c.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax_c.set_ylim(-.25, 1.2)
    ax_c.set_yticks([0, .5, 1])
    ax_c.tick_params(labelsize=5.4)
    ax_c.set_xlabel('Card price per MW (× 2024 market)', fontsize=6)
    ax_c.set_ylabel('Standby power of added cards (× today)', fontsize=6)
    ax_c.legend(handles=[Line2D([], [], color=c, lw=1.3, label=f'{p.capitalize()} path')
                         for p, c in PATHS.items()] +
                [Line2D([], [], marker='s', ls='none', markerfacecolor='white', markeredgecolor='#7FB5A8',
                        markersize=4, label='Tested grid (cases adding cards)')],
                loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=5, handlelength=1.4, labelspacing=.35,
                borderaxespad=0)
    panel(fig, 'c', .012, .44)

    # ---------------- d 供能学习曲线：相对 2024 的供能成本倍数 vs 协同节省倍数（各机群中位数的中位数，须线为机群范围）
    l0 = learn[learn.intensity_of_max.eq(0)]
    sup = l0.pivot(index='scenario', columns='cost_scenario', values='supply_cost_usd_per_mwh_median')
    sav = l0.pivot(index='scenario', columns='cost_scenario', values='V_usd_per_mwh_median')
    rs = sup.div(sup['base-2024'], axis=0)
    rv = sav.div(sav['base-2024'], axis=0)
    ax_d = fig.add_axes([.69, .075, .29, .33])
    xx = np.linspace(.5, 1.2, 10)
    ax_d.fill_between(xx, xx, 1.25, color='#EEF6F4', lw=0, zorder=0)
    ax_d.plot(xx, xx, color=DARK, lw=.7, ls=(0, (3, 2)), zorder=1)
    ax_d.axhline(1, color='#9A9A9A', lw=.5, zorder=1)
    ax_d.text(.66, 1.2, 'Saving share of supply cost rises', fontsize=5.3, color='#3F7F72', style='italic',
              ha='left', va='top')
    ax_d.text(.83, .8, 'Share unchanged', fontsize=5.2, color=DARK, rotation=38, ha='center', va='top',
              rotation_mode='anchor')
    rows_d = []
    for path, col in PATHS.items():
        xs, ys = [1.0], [1.0]
        for yr in ('2030', '2035'):
            k = f'{yr}-{path}'
            x0, y0 = rs[k].median(), rv[k].median()
            ax_d.plot([rs[k].min(), rs[k].max()], [y0, y0], color=col, lw=.5, alpha=.8, zorder=2)
            ax_d.plot([x0, x0], [rv[k].min(), rv[k].max()], color=col, lw=.5, alpha=.8, zorder=2)
            xs.append(x0)
            ys.append(y0)
            rows_d.append({'path': path, 'year': int(yr), 'supply_ratio_median': x0, 'saving_ratio_median': y0,
                           'saving_ratio_min': rv[k].min(), 'saving_ratio_max': rv[k].max()})
        ax_d.plot(xs, ys, color=col, lw=1.3, zorder=3)
        ax_d.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                      arrowprops=dict(arrowstyle='-|>', color=col, lw=1.3, mutation_scale=7), zorder=3)
        ax_d.scatter(xs[1:], ys[1:], s=16, color=col, edgecolors='white', lw=.4, zorder=4)
    ax_d.scatter(1, 1, s=26, color=DARK, zorder=5)
    ax_d.text(1.01, 1.015, '2024', fontsize=5.2, ha='left', va='bottom')
    ax_d.set_xlim(.6, 1.08)
    ax_d.set_ylim(.5, 1.22)
    ax_d.tick_params(labelsize=5.4)
    ax_d.set_xlabel('Energy-supply cost (× 2024)', fontsize=6)
    ax_d.set_ylabel('Coordination saving (× 2024)', fontsize=6)
    guides(ax_d, 'both')
    panel(fig, 'd', .6, .44)

    export(fig, 'fig4')
    csv(med.join(cut, lsuffix='_saving', rsuffix='_emission_cut').reset_index(), 'fig4a')
    csv(ratio.reset_index(), 'fig4b')
    csv(pd.concat([grid.assign(path='tested'), *rows_c]), 'fig4c')
    csv(pd.DataFrame(rows_d), 'fig4d')
    return med, ratio, pd.DataFrame(rows_d)


if __name__ == '__main__':
    med, ratio, agg = main()
    print(ratio.round(2).to_string())
    print(agg.round(2).to_string())
