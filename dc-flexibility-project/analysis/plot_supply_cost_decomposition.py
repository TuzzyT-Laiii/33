"""附录图：算电协同供能侧节省的分项构成（结论一至四）与排放口径检验。

输入为 analysis/supply_cost_decomposition.py 的输出（results/supply-cost-decomposition-20261119）。
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import stable_limited_figs_common as c

RES = c.ROOT / 'results/supply-cost-decomposition-20261119'
COMP = ['grid_energy', 'grid_capacity', 'carbon', 'storage', 'renewables']
NAME = {'grid_energy': 'Grid energy', 'grid_capacity': 'Grid connection capacity', 'carbon': 'Carbon cost',
        'storage': 'Storage', 'renewables': 'Wind and solar'}
CC = {'grid_energy': '#1B3F70', 'grid_capacity': '#46A6B5', 'carbon': '#93CDB9', 'storage': '#C9A646',
      'renewables': '#C0623F'}
USE_ORDER = ['HPC', 'CLD', 'SAI', 'TRN', 'INF']
MM = 1 / 25.4


def panel_a(ax, d):
    """各情境的分项节省（均值，可加）；黑点为净节省。"""
    d = d[d.intensity_of_max.eq(0)].set_index('scenario').loc[c.SCENARIOS[::-1]]
    y = np.arange(len(d))
    pos, neg = np.zeros(len(d)), np.zeros(len(d))
    for k in COMP:
        v = d[f'{k}_saving_mean_usd_mwh'].to_numpy()
        left = np.where(v >= 0, pos, neg)
        ax.barh(y, v, left=left, height=.68, color=CC[k], lw=0, label=NAME[k], zorder=2)
        pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
    ax.scatter(d.saving_mean_usd_mwh, y, s=9, color='black', zorder=3, label='Net saving')
    ax.axvline(0, color='black', lw=.5, zorder=2)
    ax.set_xlim(-7, 23)
    ax.set_xlabel('Supply-cost saving from coordination (USD/MWh)')
    c.scenario_axis(ax, list(d.index), y, outer=-.33)
    ax.set_ylim(-.6, len(d) - .4)
    c.guides(ax)
    c.csv(d.reset_index(), 'figS1a_components_by_scenario')


def panel_b(fig, rect, d):
    """协同前后供能结构（中位数）：空心为不做协同，实心为协同。"""
    d = d[d.intensity_of_max.eq(0)].set_index('use').loc[USE_ORDER]
    metrics = [('grid_share', 'Grid share of\nelectricity (%)', 100), ('pv_per_mw', 'Solar capacity\n(MW per MW load)', 1),
               ('storage_h', 'Storage\n(hours of load)', 1), ('grid_cap_over_peak', 'Grid capacity /\npeak load (%)', 100)]
    x0, y0, w, h = rect
    h_ = h
    gap = .012
    aw = (w - gap * 3) / 4
    y = np.arange(len(USE_ORDER))[::-1]
    axes = []
    for i, (m, lab, f) in enumerate(metrics):
        ax = fig.add_axes([x0 + i * (aw + gap), y0, aw, h])
        s, j = d[f'seq_{m}_median'] * f, d[f'joint_{m}_median'] * f
        for yy, u, a, b in zip(y, USE_ORDER, s, j):
            ax.annotate('', xy=(b, yy), xytext=(a, yy),
                        arrowprops=dict(arrowstyle='-|>', color=c.COLORS[u], lw=.8, mutation_scale=5,
                                        shrinkA=2.5, shrinkB=2.5))
            ax.scatter(a, yy, s=14, facecolor='white', edgecolor=c.COLORS[u], lw=.8, zorder=3)
            ax.scatter(b, yy, s=14, color=c.COLORS[u], zorder=3)
        lo, hi = min(s.min(), j.min()), max(s.max(), j.max())
        pad = (hi - lo) * .18 + 1e-3
        ax.set_xlim(max(lo - pad, 0) if m != 'storage_h' else -.05, hi + pad)
        ax.set_xlabel(lab, fontsize=5.8)
        ax.set_ylim(-.6, len(USE_ORDER) - .4)
        if i == 0:
            ax.set_yticks(y, [c.GROUP[u] for u in USE_ORDER])
            for t, u in zip(ax.get_yticklabels(), USE_ORDER):
                t.set_color(c.COLORS[u]); t.set_fontweight('bold')
            ax.tick_params(axis='y', length=0)
        else:
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
        c.guides(ax)
        axes.append(ax)
    h = [axes[0].scatter([], [], s=14, facecolor='white', edgecolor='black', lw=.8),
         axes[0].scatter([], [], s=14, color='black')]
    fig.legend(h, ['No coordination', 'Coordination'], loc='lower right', bbox_to_anchor=(x0 + w, y0 + h_ + .005),
               ncol=2, fontsize=5.8, handletextpad=.2, columnspacing=1, borderaxespad=0)
    c.csv(d.reset_index(), 'figS1b_supply_structure')
    return axes


def panel_c(ax, d):
    """推理强度 0–75%：分项（均值）的有符号堆叠面积，黑线为净节省。"""
    d = d.sort_values('intensity_of_max')
    x = d.intensity_of_max.to_numpy() * 100
    pos, neg = np.zeros(len(d)), np.zeros(len(d))
    for k in COMP:
        v = d[f'{k}_saving_mean_usd_mwh'].to_numpy()
        if (v >= 0).all():
            ax.fill_between(x, pos, pos + v, color=CC[k], lw=0, zorder=2); pos = pos + v
        else:
            ax.fill_between(x, neg, neg + v, color=CC[k], lw=0, zorder=2); neg = neg + v
    ax.plot(x, d.saving_mean_usd_mwh, color='black', lw=1.1, marker='o', ms=2.5, zorder=3)
    ax.axhline(0, color='black', lw=.5)
    ax.set_xticks(x, [f'{v:.0f}%' for v in x])
    ax.set_xlim(x.min(), x.max())
    ax.set_xlabel('Inference demand (% of absorbable maximum)')
    ax.set_ylabel('Saving by component (USD/MWh)')
    c.guides(ax, 'y')
    c.csv(d, 'figS1c_components_by_inference')


def panel_d(ax, carbon, learning):
    """碳价与学习曲线情景下的分项（均值，不接推理）：碳价 0/100/200 与 2030、2035 三条成本路径。"""
    cb = carbon[carbon.intensity_of_max.eq(0)].sort_values('carbon_price_usd_t')
    lc = learning[learning.intensity_of_max.eq(0) & learning.cost_scenario.ne('2024')].copy()
    order = [f'{y}-{p}' for y in (2030, 2035) for p in ('conservative', 'moderate', 'advanced')]
    lc = lc.set_index('cost_scenario').loc[order].reset_index()
    rows = pd.concat([cb.assign(case=[f'{v:.0f}' for v in cb.carbon_price_usd_t]), lc.assign(case=order)])
    x = np.r_[0:3, 3.6:6.6, 7.2:10.2]
    for k in COMP:
        ax.plot(x, rows[f'{k}_saving_mean_usd_mwh'], color=CC[k], lw=1, marker='o', ms=2.5, zorder=2)
    ax.plot(x, rows.saving_mean_usd_mwh, color='black', lw=1.3, marker='o', ms=3, zorder=3)
    ax.axhline(0, color='black', lw=.5)
    labels = ['0', '100', '200'] + ['Cons.', 'Mod.', 'Adv.'] * 2
    ax.set_xticks(x, labels, fontsize=5.6)
    import matplotlib.transforms as mt
    tr = mt.blended_transform_factory(ax.transData, ax.transAxes)
    for mid, lab in ((1, 'Carbon price (USD/t), 2024 costs'), (4.6, '2030 cost path'), (8.2, '2035 cost path')):
        ax.text(mid, -.17, lab, transform=tr, ha='center', va='top', fontsize=5.8)
    for s in (3.0, 6.6):
        ax.axvline(s + .3, color='#CFCFCF', lw=.5)
    ax.set_xlim(-.4, 9.6)
    ax.set_ylabel('Saving by component (USD/MWh)')
    c.guides(ax, 'y')
    c.csv(rows, 'figS1d_components_future')


def panel_e(ax, by_scenario, down):
    """结论二：减配少耗的供能成本 vs 现有协同节省；右列为减配总节省中供能部分的占比。"""
    a = by_scenario[by_scenario.intensity_of_max.eq(0)].set_index('scenario').loc[c.SCENARIOS[::-1]]
    b = down.set_index('scenario').loc[c.SCENARIOS[::-1]]
    y = np.arange(len(a))
    for yy, s in zip(y, a.index):
        ax.plot([a.loc[s, 'saving_median_usd_mwh'], b.loc[s, 'supply_saving_usd_mwh_median']], [yy, yy],
                color='#D0D0D0', lw=.8, zorder=1)
    ax.scatter(a.saving_median_usd_mwh, y, s=13, marker='o', color=[c.color(s) for s in a.index], zorder=3,
               label='Coordination')
    ax.scatter(b.supply_saving_usd_mwh_median, y, s=15, marker='D', facecolor='white',
               edgecolor=[c.color(s) for s in a.index], lw=.8, zorder=3, label='Downsizing, supply side')
    for yy, s in zip(y, a.index):
        ax.text(1.02, yy, f'{b.loc[s, "supply_share_of_total_pct_median"]:.1f}%', transform=ax.get_yaxis_transform(),
                ha='left', va='center', fontsize=5.6, color=c.color(s))
    ax.text(1.02, len(a) - .2, 'Share of\ndownsizing', transform=ax.get_yaxis_transform(), ha='left', va='bottom',
            fontsize=5.6)
    ax.set_xlim(0, 17)
    ax.set_ylim(-.6, len(a) - .4)
    ax.set_xlabel('Supply-cost saving (USD/MWh, median)')
    c.scenario_axis(ax, list(a.index), y, outer=-.40)
    c.guides(ax)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], ls='', marker='o', ms=3.5, color='black'),
               Line2D([], [], ls='', marker='D', ms=3.5, mfc='white', mec='black', mew=.8)],
              ['Coordination', 'Downsizing, supply side'], loc='upper left', fontsize=5.6, handletextpad=.2,
              borderaxespad=.2)
    c.csv(a[['saving_median_usd_mwh']].join(b).reset_index(), 'figS1e_downsizing_supply')


def panel_f(ax, life):
    """排放口径检验：补隐含排放后协同节省的变化（中位数，点为中档，线为低–高档）。"""
    cases = [('carbon100_I0', 'Carbon 100, no inference'), ('carbon100_I75', 'Carbon 100, inference 75%'),
             ('carbon200_I0', 'Carbon 200, no inference'), ('2035-moderate_I0', '2035 moderate path'),
             ('2035-advanced_I0', '2035 advanced path')]
    off = dict(zip(USE_ORDER, np.linspace(-.28, .28, 5)))
    for i, (k, lab) in enumerate(cases[::-1]):
        q = life[life.case.eq(k)].set_index('use')
        for u in USE_ORDER:
            yy = i + off[u]
            ax.plot([q.loc[u, 'saving_change_low_median_pct'], q.loc[u, 'saving_change_high_median_pct']], [yy] * 2,
                    color=c.COLORS[u], lw=.8, zorder=2)
            ax.scatter(q.loc[u, 'saving_change_mid_median_pct'], yy, s=9, color=c.COLORS[u], zorder=3,
                       label=c.GROUP[u] if i == 0 else None)
    ax.axvline(0, color='black', lw=.5)
    ax.set_yticks(range(len(cases)), [lab for _, lab in cases[::-1]])
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-.6, len(cases) - .4)
    ax.set_xlabel('Change in coordination saving with embodied emissions (%)')
    c.guides(ax)
    ax.legend(loc='upper right', fontsize=5.6, ncol=1, handletextpad=.1, borderaxespad=.2)
    c.csv(life, 'figS1f_lifecycle_check')


def main():
    c.setup()
    by_s = pd.read_csv(RES / 'sec1-3-by-scenario.csv')
    by_u = pd.read_csv(RES / 'sec1-3-by-use.csv')
    allc = pd.read_csv(RES / 'sec1-3-all.csv')
    carbon = pd.read_csv(RES / 'sec4-carbon-all.csv')
    learning = pd.read_csv(RES / 'sec4-learning-all.csv')
    down = pd.read_csv(RES / 'sec2-downsizing-supply.csv')
    life = pd.read_csv(RES / 'lifecycle-emissions-check.csv')
    fig = plt.figure(figsize=(180 * MM, 200 * MM))
    L, R = .20, .64
    ax = fig.add_axes([L, .70, .30, .28]); panel_a(ax, by_s); c.panel(fig, 'a', .01, .99)
    panel_b(fig, [R, .735, .35, .215], by_u); c.panel(fig, 'b', .53, .99)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(color=CC[k], lw=0) for k in COMP] + [
        Line2D([], [], color='black', marker='o', ms=3, lw=1.1)]
    fig.legend(handles, [NAME[k] for k in COMP] + ['Net saving'], loc='center', bbox_to_anchor=(.5, .641),
               ncol=6, fontsize=6, handlelength=1.4, columnspacing=1.4, borderaxespad=0)
    ax = fig.add_axes([.075, .395, .34, .205]); panel_c(ax, allc); c.panel(fig, 'c', .01, .625)
    ax = fig.add_axes([.555, .395, .43, .205]); panel_d(ax, carbon, learning); c.panel(fig, 'd', .48, .625)
    ax = fig.add_axes([L, .05, .25, .27]); panel_e(ax, by_s, down); c.panel(fig, 'e', .01, .345)
    ax = fig.add_axes([.68, .05, .30, .27]); panel_f(ax, life); c.panel(fig, 'f', .545, .345)
    c.export(fig, 'figS1-supply-cost')


if __name__ == '__main__':
    main()
