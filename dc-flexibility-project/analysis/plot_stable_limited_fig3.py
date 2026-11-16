"""图3：真实推理需求下，算电协同仍带来稳定收益（第三节）。

a 推理加到上限 75% 时训练机群（GPU-heavy）协同节省地图（USD/MWh，配色同图 1a），色标上方为网格直方图；
b 环形图：上限 75% 时“卖推理净收入 + 协同节省”中协同节省所占比例；
c 争用与互补：被推理占用的闲置比例 vs 协同保留率，虚线为完全争用（保留率 = 1 − 占用比例）；
d 剩余收益的地区差异：训练机群保留率 vs 风电装机占比，点色为不做协同时的市电占比；
e 上限 75% 时相对不接推理：协同转移电量变化与每 MWh 转移电量价值变化（发散柱）。
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Wedge
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stable_limited_figs_common import (COLORS, DARK, GPU_SCENARIOS, GROUP, INNER, INPUT, LAND, NEUTRAL, RESIDUAL,
                                        SCENARIOS, SEQ, baseline, csv, draw_map, export, guides, paired, panel,
                                        scenario_axis, sell, setup, shade, use)

LEVELS = [0, .25, .5, .75]


def lighten(c, f):
    return tuple(np.array(mpl.colors.to_rgb(c)) * (1 - f) + f)


def main():
    setup()
    p = paired()
    q75 = p[p.intensity_of_max.eq(.75)]
    fig = plt.figure(figsize=(180 / 25.4, 195 / 25.4))

    # ---------------- a 地图 + 直方图色标
    dtrn = q75[q75.scenario.eq('D-TRN')].set_index('grid_id')
    norm = mpl.colors.Normalize(0, 30)
    draw_map(fig, [.02, .655, .64, .325], dtrn.saving_usd_mwh, SEQ, norm, lon_labels=False)
    axh = fig.add_axes([.17, .64, .34, .035])
    bins = np.linspace(0, 30, 61)
    cnt, edges = np.histogram(dtrn.saving_usd_mwh.clip(upper=30), bins)
    axh.bar(edges[:-1], cnt, width=np.diff(edges), align='edge', color=SEQ(norm(edges[:-1] + .25)), lw=0)
    axh.set_xlim(0, 30)
    axh.axis('off')
    vmin = dtrn.saving_usd_mwh.min()
    axh.annotate(f'min {vmin:.2f}', (vmin, 0), xytext=(1.0, cnt.max() * .8), fontsize=5.3,
                 arrowprops=dict(arrowstyle='-', lw=.4, color=DARK))
    cax = fig.add_axes([.17, .635, .34, .006])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=SEQ), cax=cax, orientation='horizontal',
                      ticks=[0, 5, 10, 15, 20, 25, 30], extend='max', extendfrac=.04)
    cb.outline.set_linewidth(.3)
    cax.tick_params(labelsize=5.5, length=1.5, width=.4, pad=1)
    cb.set_label('Coordination saving with inference at 75% of cap, LLM training · GPU-heavy (USD/MWh)',
                 fontsize=6, labelpad=2)
    panel(fig, 'a', .012, .99)

    # ---------------- b 环形图：协同节省占净收入
    sl = sell()
    m = sl[sl.tier.eq('real_75')].groupby('scenario')[['coord_usd_mwh', 'net_usd_mwh']].median()
    share = m.coord_usd_mwh / (m.coord_usd_mwh + m.net_usd_mwh) * 100
    ax_b = fig.add_axes([.7, .62, .28, .36])
    ax_b.set_xlim(-1.3, 1.3)
    ax_b.set_ylim(-.5, 3.6)
    ax_b.set_aspect('equal')
    ax_b.axis('off')
    groups = [('SAI', ['B-SAI', 'C-SAI', 'D-SAI']), ('TRN', ['B-TRN', 'C-TRN', 'D-TRN']), ('INF', ['B-INF', 'C-INF', 'D-INF'])]
    for i, (u, scs) in enumerate(groups):
        cy = 2.95 - i * 1.25
        cx = -.55
        sv = share[scs]
        mid = sv.loc[scs[2]]
        ax_b.add_patch(Wedge((cx, cy), .5, 90, 90 + 360 * (1 - mid / 100), width=.17, facecolor=lighten(COLORS[u], .35),
                             edgecolor='none'))
        ax_b.add_patch(Wedge((cx, cy), .5, 90 - 360 * mid / 100, 90, width=.17, facecolor=DARK, edgecolor='none'))
        ax_b.text(cx, cy, f'{mid:.1f}%', ha='center', va='center', fontsize=7, fontweight='bold', color=DARK)
        ax_b.text(.1, cy + .15, GROUP[u], ha='left', va='center', fontsize=6, fontweight='bold', color=COLORS[u])
        ax_b.text(.1, cy - .14, f'all configs {sv.min():.1f}–{sv.max():.1f}%', ha='left', va='center', fontsize=5.3)
    ax_b.legend(handles=[Patch(facecolor=DARK, label='Coordination saving'),
                         Patch(facecolor='#BFBFBF', label='Net inference revenue')],
                loc='lower left', bbox_to_anchor=(0, -.02), fontsize=5.2, handlelength=.9)
    panel(fig, 'b', .67, .99)

    # ---------------- c 争用与互补
    cap = pd.read_csv(RESIDUAL / 'max-intensity.csv').set_index('scenario').max_intensity
    ret = p.groupby(['scenario', 'intensity_of_max']).retention.median().unstack()
    ax_c = fig.add_axes([.075, .335, .36, .23])
    xx = np.linspace(0, .7, 50)
    ax_c.fill_between(xx, 1 - xx, 1.1, color='#EEF6F4', lw=0, zorder=0)
    ax_c.plot(xx, 1 - xx, color=DARK, lw=.8, ls=(0, (3, 2)))
    marks = {'A': 'D', 'B': 'o', 'C': 's', 'D': '^'}
    rows_c = []
    for sc in SCENARIOS:
        occ = np.array(LEVELS) * cap[sc]
        r = ret.loc[sc, LEVELS].to_numpy()
        c = COLORS[use(sc)]
        ax_c.plot(occ, r, color=shade(sc), lw=.8, zorder=3)
        ax_c.scatter(occ[-1], r[-1], marker=marks[sc[0]], s=22, color=c, edgecolors='white', lw=.5, zorder=4)
        rows_c.append(pd.DataFrame({'scenario': sc, 'intensity_of_cap': LEVELS, 'idle_occupied': occ, 'retention': r}))
    ax_c.set_xlim(0, .7)
    ax_c.set_ylim(.3, 1.1)
    ax_c.set_xlabel('Idle capacity occupied by inference (× mean idle)', fontsize=6)
    ax_c.set_ylabel('Retention of coordination saving', fontsize=6)
    guides(ax_c, 'both')
    h1 = [Line2D([], [], marker='o', ls='none', color=COLORS[u], markersize=3.5, label=GROUP[u]) for u in GROUP]
    h2 = [Line2D([], [], marker=mk, ls='none', color='#777777', markersize=3.5, label=INNER[k]) for k, mk in marks.items()]
    h3 = [Line2D([], [], color=DARK, lw=.8, ls=(0, (3, 2)), label='Full competition'),
          Patch(facecolor='#EEF6F4', label='Complementarity')]
    ax_c.legend(handles=h1 + h2 + h3, loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=5, handletextpad=.2,
                labelspacing=.35, borderaxespad=0)
    panel(fig, 'c', .012, .6)

    # ---------------- d 保留率 vs 风电占比
    b0 = baseline()[['grid_id', 'scenario', 'grid_mwh0']]
    dd = dtrn.reset_index().merge(b0[b0.scenario.eq('D-TRN')].drop(columns='scenario'), on='grid_id')
    gshare = 100 * dd.grid_mwh0 / dd.facility_mwh0
    ax_d = fig.add_axes([.64, .335, .24, .23])
    gnorm = mpl.colors.Normalize(0, 80)
    order = np.argsort(-gshare.to_numpy())
    ax_d.scatter(dd.wind_share.to_numpy()[order] * 100, dd.retention.to_numpy()[order], s=1.6, lw=0,
                 c=gshare.to_numpy()[order], cmap=SEQ.reversed(), norm=gnorm, rasterized=True)
    cut = pd.cut(dd.wind_share * 100, np.arange(0, 101, 5))
    bm = dd.groupby(cut, observed=True).retention.agg(['median', 'size'])
    bm = bm[bm['size'] >= 60]
    ax_d.plot([i.mid for i in bm.index], bm['median'], color=DARK, lw=1.1)
    rho = spearmanr(dd.wind_share, dd.retention)[0]
    ax_d.text(.04, .95, f'ρ = {rho:.2f}', transform=ax_d.transAxes, va='top', fontsize=6)
    ax_d.set_xlim(0, 100)
    ax_d.set_ylim(.1, 1.0)
    ax_d.set_xlabel('Wind share of optimal wind + PV (%)', fontsize=6)
    ax_d.set_ylabel('Retention, training · GPU-heavy', fontsize=6)
    guides(ax_d, 'both')
    cax = fig.add_axes([.89, .335, .006, .23])
    cbd = fig.colorbar(mpl.cm.ScalarMappable(norm=gnorm, cmap=SEQ.reversed()), cax=cax)
    cbd.outline.set_linewidth(.3)
    cax.tick_params(labelsize=5, length=1.5, width=.4, pad=1)
    cbd.set_label('Grid share (%)', fontsize=5.5, labelpad=2)
    panel(fig, 'd', .57, .6)

    # ---------------- e 发散柱：转移电量 vs 每 MWh 价值
    mv = pd.read_parquet(INPUT / 'moved-energy.parquet')
    mv['moved'] = mv.moved_task_mwh + mv.moved_off_mwh
    mv = mv.merge(p[['grid_id', 'scenario', 'intensity_of_max', 'V_usd']], on=['grid_id', 'scenario', 'intensity_of_max'])
    base = mv[mv.intensity_of_max.eq(0)].set_index(['grid_id', 'scenario'])[['moved', 'V_usd']]
    mv = mv.join(base, on=['grid_id', 'scenario'], rsuffix='0')
    mv = mv[mv.intensity_of_max.eq(.75)]
    mv['d_energy'] = (mv.moved / mv.moved0 - 1) * 100
    mv['d_value'] = ((mv.V_usd / mv.moved) / (mv.V_usd0 / mv.moved0) - 1) * 100
    dm = mv.groupby('scenario')[['d_energy', 'd_value']].median().loc[SCENARIOS]
    ax_e = fig.add_axes([.12, .08, .85, .14])
    xs = np.arange(len(SCENARIOS)).astype(float)
    for x0, sc in zip(xs, SCENARIOS):
        ax_e.bar(x0 - .19, dm.loc[sc, 'd_energy'], width=.36, color=NEUTRAL, lw=0)
        ax_e.bar(x0 + .19, dm.loc[sc, 'd_value'], width=.36, color=shade(sc), lw=0)
    ax_e.axhline(0, color=DARK, lw=.6)
    ax_e.set_xlim(-.6, len(xs) - .4)
    ax_e.set_ylim(-55, 15)
    ax_e.set_yticks([-50, -25, 0])
    ax_e.set_ylabel('Change at 75% of cap\nvs no inference (%)', fontsize=6)
    scenario_axis(ax_e, SCENARIOS, xs, axis='x', outer=-.3)
    for lab in ax_e.get_xticklabels():
        lab.set_rotation(0)
    ax_e.legend(handles=[Patch(facecolor=NEUTRAL, label='Energy shifted by coordination'),
                         Patch(facecolor='#5E9E8C', label='Value per shifted MWh')],
                loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2, fontsize=5.4, handlelength=.9)
    guides(ax_e, 'y')
    panel(fig, 'e', .012, .27)

    export(fig, 'fig3')

    csv(dtrn.reset_index()[['grid_id', 'lat', 'lon', 'saving_usd_mwh', 'retention']], 'fig3a')
    csv(pd.DataFrame({'scenario': share.index, 'coord_usd_mwh': m.coord_usd_mwh.values,
                      'net_inference_usd_mwh': m.net_usd_mwh.values, 'coord_share_pct': share.values}), 'fig3b')
    csv(pd.concat(rows_c), 'fig3c')
    csv(dd[['grid_id', 'wind_share', 'retention']].assign(grid_share_pct=gshare.values), 'fig3d')
    csv(dm.reset_index(), 'fig3e')
    return share, pd.concat(rows_c), dm, rho


if __name__ == '__main__':
    share, c, dm, rho = main()
    print(share.round(2).to_dict())
    print(c[c.intensity_of_cap.eq(.75)].round(3).to_string())
