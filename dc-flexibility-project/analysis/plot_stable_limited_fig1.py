"""图1：算力中心存在可观的调节余量，但单靠电力调度收益有限（第一节）。

a 训练机群（GPU-heavy）协同节省全球地图（USD/MWh，I=0，规划边界）+ 右侧纬度统计条；
b 11 情境平均任务功率与峰值余量（堆叠条）；c 同一行的协同节省网格分布（山脊图）；
d 5 个代表情境的成本嵌套方块：机群+供能总成本 ⊃ 供能成本 ⊃ 协同节省（面积按比例）；
e 训练机群逐网格节省 vs 不做协同时的市电占比，点色为风电装机占比。
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr

from stable_limited_figs_common import (COLORS, DARK, GROUP, INNER, LAND, NEUTRAL, REPRESENTATIVES, SCENARIOS, SECTION1,
                                        SEQ, INPUT, baseline, colorbar, downsizing_usd_mwh, task_layers, csv, draw_map, export, guides, latitude_strip,
                                        paired, panel, part1_summary, scenario_axis, setup, shade, use)


def section1_shares():
    """第一节原生窗口（PR #45 grid-summary，规划边界、经济最优）：节省占供能成本与占总成本。"""
    g = pd.read_parquet(SECTION1 / 'grid-summary.parquet',
                        columns=['grid_id', 'scenario', 'alpha', 'boundary', 'baseline_status', 'flex_status',
                                 'baseline_cost_usd', 'flex_cost_usd', 'savings_pct'])
    g = g[g.alpha.isna() & g.boundary.eq('planning') & g.baseline_status.eq('optimal') & g.flex_status.eq('optimal')]
    assert g.groupby('scenario').size().eq(4842).all()
    s = part1_summary()
    year = 8760 / g.scenario.map(s.hours)
    fleet = g.scenario.map(s.fleet_annual_musd) * 1e6
    energy = g.baseline_cost_usd * year
    saving = (g.baseline_cost_usd - g.flex_cost_usd) * year
    return g.assign(share_supply_pct=g.savings_pct, share_total_pct=100 * saving / (fleet + energy),
                    supply_of_total_pct=100 * energy / (fleet + energy))[
        ['grid_id', 'scenario', 'share_supply_pct', 'share_total_pct', 'supply_of_total_pct']]


def main():
    setup()
    p = paired()
    p0 = p[p.intensity_of_max.eq(0)].merge(baseline()[['grid_id', 'scenario', 'grid_mwh0']], on=['grid_id', 'scenario'])
    p0['grid_share'] = 100 * p0.grid_mwh0 / p0.facility_mwh0
    shares = section1_shares()
    head = part1_summary().loc[SCENARIOS]

    fig = plt.figure(figsize=(180 / 25.4, 215 / 25.4))
    norm = mpl.colors.Normalize(0, 30)

    # ---------------- a 地图 + 右侧纬度条
    dtrn = p0[p0.scenario.eq('D-TRN')].set_index('grid_id')
    ax_map, _ = draw_map(fig, [.03, .695, .74, .285], dtrn.saving_usd_mwh, SEQ, norm, lon_labels=False)
    frames = [(GROUP[use(s)], COLORS[use(s)], p0.loc[p0.scenario.eq(s), ['lat', 'saving_usd_mwh']]
               .rename(columns={'saving_usd_mwh': 'value'}), s == 'D-TRN') for s in REPRESENTATIVES]
    ax_lat, lat_rows = latitude_strip(fig, ax_map, .08, frames, 'USD/MWh', xlim=(0, 30))
    ax_lat.set_xticks([0, 10, 20, 30])
    handles = [Line2D([], [], color=COLORS[use(s)], lw=1.1, label=GROUP[use(s)] + ('' if s[0] == 'A' else '\n(GPU-heavy)'))
               for s in REPRESENTATIVES] + [Patch(facecolor=LAND, label='No eligible\ngrid cell')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.885, .975), fontsize=5.4, handlelength=1.2,
               labelspacing=.5, borderaxespad=0)
    colorbar(fig, [.2, .69, .38, .008], SEQ, norm, [0, 5, 10, 15, 20, 25, 30],
             'Coordination saving, LLM training · GPU-heavy (USD per MWh of electricity)', extend='max')
    panel(fig, 'a', .012, .99)

    # ---------------- b 堆叠条 + c 山脊图（共用行）
    y = np.arange(len(SCENARIOS))[::-1].astype(float)
    ax_b = fig.add_axes([.21, .395, .2, .235])
    potential = []
    ax_c = fig.add_axes([.46, .395, .26, .235])
    for yi, s in zip(y, SCENARIOS):
        rigid, defer, k = task_layers(s)
        r_pct, d_pct = rigid.mean() / k * 100, defer.mean() / k * 100
        u = r_pct + d_pct
        ax_b.barh(yi, r_pct, height=.66, color=tuple(np.array(mpl.colors.to_rgb(shade(s))) * .7), lw=0)
        ax_b.barh(yi, d_pct, left=r_pct, height=.66, color=shade(s), lw=0)
        ax_b.barh(yi, 100 - u, left=u, height=.66, color='#F1F1F1', edgecolor='#B5B5B5', lw=.35)
        ax_b.text(r_pct + d_pct / 2, yi, f'{d_pct:.0f}', ha='center', va='center', fontsize=5.4, color='white')
        ax_b.text((100 + u) / 2, yi, f'{100 - u:.0f}', ha='center', va='center', fontsize=5.4)
        potential.append({'scenario': s, 'non_deferrable_pct_K': r_pct, 'deferrable_pct_K': d_pct,
                          'headroom_pct_K': 100 - u})
    ax_b.set_xlim(0, 100)
    ax_b.set_xticks([0, 25, 50, 75, 100])
    ax_b.set_ylim(-.6, len(y) + .5)
    ax_b.spines['left'].set_visible(False)
    scenario_axis(ax_b, SCENARIOS, y, outer=-.36)
    ax_b.set_xlabel('Share of task-power peak (%)')
    ax_b.legend(handles=[Patch(facecolor='#5A5A5A', label='Non-deferrable'),
                         Patch(facecolor='#8C8C8C', label='Deferrable'),
                         Patch(facecolor='#F1F1F1', edgecolor='#B5B5B5', label='Headroom')],
                loc='upper center', bbox_to_anchor=(.42, -.16), ncol=3, handlelength=.9, columnspacing=.6, fontsize=5.4)
    panel(fig, 'b', .012, .655)

    xs = np.linspace(0, 32, 400)
    ridge = []
    for yi, s in zip(y, SCENARIOS):
        v = p0.loc[p0.scenario.eq(s), 'saving_usd_mwh'].to_numpy()
        dens = gaussian_kde(v, bw_method=.25)(xs)
        dens = dens / dens.max() * 1.45
        base = yi - .33
        ax_c.fill_between(xs, base, base + dens, color=shade(s), alpha=.9, lw=0, zorder=20 - yi)
        ax_c.plot(xs, base + dens, color='white', lw=.5, zorder=20 - yi)
        med = np.median(v)
        ax_c.plot([med, med], [base, base + np.interp(med, xs, dens)], color=DARK, lw=.7, zorder=21 - yi)
        ridge.append(pd.DataFrame({'scenario': s, 'x_usd_mwh': xs, 'density_scaled': dens, 'median': med}))
    ax_c.set_xlim(0, 32)
    ax_c.set_ylim(-.6, len(y) + .5)
    ax_c.set_yticks([])
    ax_c.spines['left'].set_visible(False)
    ax_c.set_xlabel('Coordination saving (USD/MWh)')
    guides(ax_c)
    panel(fig, 'c', .44, .655)

    # ---------------- f 现有协同 vs 减配上限（与 b、c 共用行）
    up = downsizing_usd_mwh().loc[SCENARIOS]
    now = p0.groupby('scenario').saving_usd_mwh.median().loc[SCENARIOS]
    ax_f = fig.add_axes([.765, .395, .19, .235])
    for yi, s in zip(y, SCENARIOS):
        ax_f.barh(yi, up[s], height=.66, color=tuple(np.array(mpl.colors.to_rgb(shade(s))) * .45 + .55), lw=0)
        ax_f.barh(yi, now[s], height=.66, color=DARK, lw=0)
        ax_f.text(up[s] + 12, yi, f'×{up[s] / now[s]:.0f}', va='center', fontsize=5.4, color=COLORS[use(s)])
    ax_f.set_xlim(0, 720)
    ax_f.set_xticks([0, 200, 400, 600])
    ax_f.set_ylim(-.6, len(y) + .5)
    ax_f.set_yticks([])
    ax_f.spines['left'].set_visible(False)
    ax_f.set_xlabel('USD/MWh')
    guides(ax_f)
    ax_f.legend(handles=[Patch(facecolor=DARK, label='Current coordination'),
                         Patch(facecolor='#C8C8C8', label='Upper bound: downsizing')],
                loc='upper center', bbox_to_anchor=(.45, -.16), ncol=1, handlelength=.9, fontsize=5.4)
    panel(fig, 'f', .74, .655)

    # ---------------- d 嵌套方块
    med = shares.groupby('scenario')[['share_supply_pct', 'share_total_pct', 'supply_of_total_pct']].median()
    ax_d = fig.add_axes([.015, .195, .53, .105])
    ax_d.set_anchor('N')
    ax_d.set_xlim(0, 4 * 1.15 + 1)
    ax_d.set_ylim(-.3, 1.32)
    ax_d.set_aspect('equal')
    ax_d.axis('off')
    rows = []
    for k, s in enumerate(REPRESENTATIVES):
        x0 = k * 1.15
        sup = med.loc[s, 'supply_of_total_pct'] / 100
        sav = med.loc[s, 'share_total_pct'] / 100
        ax_d.add_patch(Rectangle((x0, 0), 1, 1, facecolor='#E6E6E6', edgecolor='white', lw=.5))
        ax_d.add_patch(Rectangle((x0, 0), np.sqrt(sup), np.sqrt(sup), facecolor=shade(s), edgecolor='white', lw=.5))
        ax_d.add_patch(Rectangle((x0, 0), np.sqrt(sav), np.sqrt(sav), facecolor=DARK, edgecolor='none'))
        first = k == 0
        if first:
            ax_d.text(x0 + .97, .95, 'Fleet + energy\n= 100%', ha='right', va='top', fontsize=5, color='#777777')
        ax_d.text(x0 + .02, np.sqrt(sup) + .03, ('Energy ' if first else '') + f'{sup * 100:.1f}%', ha='left',
                  va='bottom', fontsize=5.4, color=shade(s))
        ax_d.text(x0, -.04, ('Saving ' if first else '') + f'{sav * 100:.2f}%', ha='left', va='top', fontsize=5.7,
                  fontweight='bold', color=DARK)
        ax_d.text(x0 + .5, 1.04, GROUP[use(s)] + ('' if s[0] == 'A' else '\n(GPU-heavy)'), ha='center',
                  va='bottom', fontsize=5.6, color=COLORS[use(s)], fontweight='bold')
        rows.append({'scenario': s, 'supply_of_total_pct': sup * 100, 'saving_of_total_pct': sav * 100,
                     'saving_of_supply_pct': med.loc[s, 'share_supply_pct']})
    panel(fig, 'd', .012, .3)

    # ---------------- e 散点：节省 vs 市电占比，点色为风电占比
    d = p0[p0.scenario.eq('D-TRN')]
    ax_e = fig.add_axes([.64, .07, .25, .2])
    wnorm = mpl.colors.Normalize(0, 100)
    order = np.argsort(d.wind_share.to_numpy())
    ax_e.scatter(d.grid_share.to_numpy()[order], d.saving_usd_mwh.to_numpy()[order], s=1.6, lw=0,
                 c=d.wind_share.to_numpy()[order] * 100, cmap=SEQ, norm=wnorm, rasterized=True)
    bins = np.arange(0, 101, 5)
    b = d.groupby(pd.cut(d.grid_share, bins), observed=True).saving_usd_mwh.agg(['median', 'size'])
    b = b[b['size'] >= 60]
    ax_e.plot([i.mid for i in b.index], b['median'], color=DARK, lw=1.1)
    ax_e.set_xlim(0, 100)
    ax_e.set_ylim(0, 36)
    ax_e.set_xlabel('Grid electricity share without coordination (%)')
    ax_e.set_ylabel('Saving (USD/MWh)')
    rho_g = spearmanr(d.grid_share, d.saving_usd_mwh)[0]
    rho_w = spearmanr(d.wind_share, d.saving_usd_mwh)[0]
    ax_e.text(.97, .95, f'ρ = {rho_g:.2f}', transform=ax_e.transAxes, ha='right', va='top', fontsize=6)
    guides(ax_e, 'both')
    cax = fig.add_axes([.9, .07, .007, .2])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=wnorm, cmap=SEQ), cax=cax)
    cb.outline.set_linewidth(.3)
    cax.tick_params(labelsize=5, length=1.5, width=.4, pad=1)
    cb.set_label('Wind share of optimal wind + PV (%)', fontsize=5.5, labelpad=2)
    panel(fig, 'e', .57, .3)

    # ---------------- g 机制曲线：减配随可推迟时长（上限）与加卡随卡价（完全协同）
    cap = pd.read_csv(INPUT / 'min-fleet-vs-deadline.csv')
    ax_g1 = fig.add_axes([.07, .045, .2, .1])
    xpos = [0, 1, 2, 3]
    rows_g = []
    for sc in REPRESENTATIVES:
        q = cap[cap.scenario.eq(sc)].sort_values('whole_h')
        yv = [100] + list(np.minimum.accumulate(q.C_over_A.to_numpy()) * 100)
        ax_g1.plot(xpos, yv, color=COLORS[use(sc)], lw=1.1, marker='o', markersize=2.2)
        rows_g.append(pd.DataFrame({'panel': 'fleet', 'scenario': sc, 'deadline': ['none', '12/48 h', '24/96 h', '48/168 h'],
                                    'value_pct': yv}))
    ax_g1.set_xticks(xpos, ['None', '12/48', '24/96', '48/168'], fontsize=5.2)
    ax_g1.set_xlim(-.2, 3.2)
    ax_g1.set_ylim(60, 103)
    ax_g1.set_yticks([60, 70, 80, 90, 100])
    ax_g1.tick_params(labelsize=5.2)
    ax_g1.set_xlabel('Deferral window, whole/split jobs (h)', fontsize=5.6)
    ax_g1.set_ylabel('Fleet needed (%)', fontsize=5.6)
    guides(ax_g1, 'y')

    ac = pd.read_csv(INPUT / 'add-cards-global-sample.csv')
    ax_g2 = fig.add_axes([.35, .045, .19, .1])
    for idle, ls in [(0.0, '-'), (0.1, (0, (3, 1.5)))]:
        q = ac[ac.idle_scale.eq(idle)].sort_values('m')
        xv = list(q.m * 100) + [100]
        yv = list(q.share_adding_cards * 100) + [0]
        ax_g2.plot(xv[:-1], yv[:-1], color=DARK, lw=1.1, ls=ls, marker='o', markersize=2.2)
        ax_g2.plot(xv[-2:], yv[-2:], color=DARK, lw=.6, ls=(0, (1, 1.5)))
        rows_g.append(pd.DataFrame({'panel': 'add_cards', 'scenario': f'idle_standby_{idle:g}', 'deadline': xv,
                                    'value_pct': yv}))
    ax_g2.axvline(100, color='#BBBBBB', lw=.6)
    ax_g2.set_xscale('log')
    ax_g2.set_xlim(.4, 130)
    ax_g2.set_xticks([.5, 1, 3, 10, 100], ['0.5', '1', '3', '10', '100'])
    ax_g2.minorticks_off()
    ax_g2.set_ylim(-3, 100)
    ax_g2.tick_params(labelsize=5.2)
    ax_g2.set_xlabel('Card price (% of market)', fontsize=5.6)
    ax_g2.set_ylabel('Cases adding cards (%)', fontsize=5.6)
    ax_g2.legend(handles=[Line2D([], [], color=DARK, lw=1.1, label='Idle cards off'),
                          Line2D([], [], color=DARK, lw=1.1, ls=(0, (3, 1.5)), label='10% standby')],
                 loc='upper right', fontsize=5, handlelength=1.6, borderaxespad=.2)
    guides(ax_g2, 'y')
    panel(fig, 'g', .012, .175)


    export(fig, 'fig1')

    csv(dtrn.reset_index()[['grid_id', 'lat', 'lon', 'iso3', 'continent', 'saving_usd_mwh']], 'fig1a')
    csv(lat_rows, 'fig1a_latitude')
    csv(pd.DataFrame(potential), 'fig1b')
    csv(pd.concat(ridge), 'fig1c')
    csv(pd.DataFrame(rows), 'fig1d')
    csv(d[['grid_id', 'lat', 'lon', 'grid_share', 'wind_share', 'saving_usd_mwh']], 'fig1e')
    csv(pd.DataFrame({'scenario': SCENARIOS, 'current_coordination_usd_mwh': now.values, 'downsizing_usd_mwh': up.values, 'ratio': (up / now).values}), 'fig1f')
    csv(pd.concat(rows_g), 'fig1g')
    return pd.DataFrame(rows), rho_g, rho_w


if __name__ == '__main__':
    print(main())
