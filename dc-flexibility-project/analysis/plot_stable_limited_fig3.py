"""图3：真实推理需求下，算电协同仍带来稳定收益（第三节）。

a 推理加到上限 75% 时，每个网格 11 类机群中最低的协同节省（六边形格网地图，USD/MWh），色标上方为网格直方图；
b 实施成本存活曲线：实施成本为横轴时仍有净收益的网格–机群组合占比；
c 争用与互补：被推理占用的闲置比例 vs 协同保留率，虚线为完全争用（保留率 = 1 − 占用比例）；
d 保留率分解：转移电量比 × 每 MWh 转移电量价值比，背景为等保留率线；
e 剩余收益的地区差异：训练机群保留率 vs 风电装机占比（点色为不做协同时的市电占比），右侧为各机群的秩相关系数。
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stable_limited_figs_common import (COLORS, DARK, GROUP, INNER, INPUT, RESIDUAL, SCENARIOS, SEQ, baseline, csv,
                                        export, guides, hex_mosaic, land_axes, paired, panel, setup, shade, use)

LEVELS = [0, .25, .5, .75]
USES = ['HPC', 'CLD', 'SAI', 'TRN', 'INF']
MARKS = {'A': 'D', 'B': 'o', 'C': 's', 'D': '^'}


def main():
    setup()
    p = paired()
    q75 = p[p.intensity_of_max.eq(.75)]
    fig = plt.figure(figsize=(180 / 25.4, 200 / 25.4))

    # ---------------- a 最不利机群地图（六边形格网）+ 直方图色标
    wide = q75.pivot_table(index='grid_id', columns='scenario', values='saving_usd_mwh')
    worst = wide.min(axis=1)
    loc = q75.drop_duplicates('grid_id').set_index('grid_id').loc[worst.index, ['lat', 'lon']]
    norm = mpl.colors.Normalize(0, 8)
    ax_a = land_axes(fig, [.012, .675, .65, .305])
    hex_mosaic(ax_a, loc.lon, loc.lat, worst.clip(upper=8).to_numpy(), SEQ, norm, width_km=210, max_km=300)
    axh = fig.add_axes([.17, .655, .32, .03])
    bins = np.linspace(0, 8, 41)
    cnt, edges = np.histogram(worst.clip(upper=8), bins)
    axh.bar(edges[:-1], cnt, width=np.diff(edges), align='edge', color=SEQ(norm(edges[:-1] + .1)), lw=0)
    axh.set_xlim(0, 8)
    axh.axis('off')
    vmin = worst.min()
    axh.annotate(f'min {vmin:.2f}', (vmin, 0), xytext=(.25, cnt.max() * .85), fontsize=5.3,
                 arrowprops=dict(arrowstyle='-', lw=.4, color=DARK))
    cax = fig.add_axes([.17, .65, .32, .006])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=SEQ), cax=cax, orientation='horizontal',
                      ticks=range(9), extend='max', extendfrac=.04)
    cb.outline.set_linewidth(.3)
    cax.tick_params(labelsize=5.5, length=1.5, width=.4, pad=1)
    cb.set_label('Lowest coordination saving across 11 fleets (USD/MWh)', fontsize=6, labelpad=2)
    panel(fig, 'a', .012, .99)

    # ---------------- b 实施成本存活曲线
    ax_b = fig.add_axes([.74, .695, .24, .27])
    cost = np.linspace(0, 8, 161)
    rows_b = []
    for u in USES:
        v = q75.loc[q75.scenario.str.endswith(u), 'saving_usd_mwh'].to_numpy()
        surv = (v[None, :] > cost[:, None]).mean(axis=1) * 100
        ax_b.plot(cost, surv, color=COLORS[u], lw=1.0)
        rows_b.append(pd.DataFrame({'group': u, 'cost_usd_mwh': cost, 'share_net_positive_pct': surv}))
    allv = q75.saving_usd_mwh.to_numpy()
    surv_all = (allv[None, :] > cost[:, None]).mean(axis=1) * 100
    ax_b.plot(cost, surv_all, color='black', lw=1.6)
    rows_b.append(pd.DataFrame({'group': 'all', 'cost_usd_mwh': cost, 'share_net_positive_pct': surv_all}))
    for c0 in (2, 3):
        s0 = (allv > c0).mean() * 100
        ax_b.plot([c0, c0], [0, s0], color='#8C8C8C', lw=.5, ls=(0, (2, 2)))
        ax_b.scatter(c0, s0, s=14, color='black', zorder=5)
        ax_b.text(c0 - .2, s0 - 3, f'{s0:.0f}%', fontsize=5.6, fontweight='bold', ha='right', va='top')
    ax_b.set_xlim(0, 8)
    ax_b.set_ylim(0, 103)
    ax_b.set_yticks([0, 25, 50, 75, 100])
    ax_b.tick_params(labelsize=5.4)
    ax_b.set_xlabel('Implementation cost (USD/MWh)', fontsize=6)
    ax_b.set_ylabel('Cases still net positive (%)', fontsize=6)
    guides(ax_b, 'both')
    ax_b.legend(handles=[Line2D([], [], color='black', lw=1.6, label='All fleets')] +
                [Line2D([], [], color=COLORS[u], lw=1.0, label=GROUP[u]) for u in USES],
                loc='lower left', fontsize=5, handlelength=1.3, labelspacing=.25, borderaxespad=.3,
                frameon=True, framealpha=.85, edgecolor='none')
    panel(fig, 'b', .68, .99)

    # ---------------- c 争用与互补（线尾直接标注）
    cap = pd.read_csv(RESIDUAL / 'max-intensity.csv').set_index('scenario').max_intensity
    ret = p.groupby(['scenario', 'intensity_of_max']).retention.median().unstack()
    ax_c = fig.add_axes([.085, .355, .38, .225])
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
    panel(fig, 'c', .012, .605)

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
    ax_d = fig.add_axes([.61, .355, .26, .225])
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
    panel(fig, 'd', .53, .605)

    # ---------------- e 保留率 vs 风电占比（训练 GPU-heavy）+ 各机群秩相关
    dtrn = q75[q75.scenario.eq('D-TRN')]
    b0 = baseline()[['grid_id', 'scenario', 'grid_mwh0']]
    dd = dtrn.merge(b0[b0.scenario.eq('D-TRN')].drop(columns='scenario'), on='grid_id')
    gshare = 100 * dd.grid_mwh0 / dd.facility_mwh0
    ax_e = fig.add_axes([.085, .065, .42, .2])
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
    cax = fig.add_axes([.515, .065, .006, .2])
    cbd = fig.colorbar(mpl.cm.ScalarMappable(norm=gnorm, cmap=SEQ.reversed()), cax=cax)
    cbd.outline.set_linewidth(.3)
    cax.tick_params(labelsize=5, length=1.5, width=.4, pad=1)
    cbd.set_label('Grid share without coordination (%)', fontsize=5.5, labelpad=2)

    rho = {sc: spearmanr(gq.wind_share, gq.retention)[0] for sc, gq in q75.groupby('scenario')}
    ax_r = fig.add_axes([.7, .065, .27, .2])
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
    panel(fig, 'e', .012, .295)

    export(fig, 'fig3')

    csv(pd.DataFrame({'grid_id': worst.index, 'lat': loc.lat.values, 'lon': loc.lon.values,
                      'lowest_saving_usd_mwh': worst.values, 'lowest_fleet': wide.idxmin(axis=1).values}), 'fig3a')
    csv(pd.concat(rows_b), 'fig3b')
    csv(pd.concat(rows_c), 'fig3c')
    csv(dm.reset_index(), 'fig3d')
    csv(pd.concat([dd[['grid_id', 'wind_share', 'retention']].assign(grid_share_pct=gshare.values, scenario='D-TRN'),
                   pd.DataFrame(rows_e)]), 'fig3e')
    return worst, dm, rho


if __name__ == '__main__':
    worst, dm, rho = main()
    print('worst min %.2f' % worst.min())
    print(dm.round(3).to_string())
    print({k: round(v, 2) for k, v in rho.items()})
