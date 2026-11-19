"""图1：算力中心存在可观的调节余量，但单靠电力调度收益有限（第一节）。

a 训练机群（GPU-heavy）协同节省全球地图（USD/MWh，I=0，规划边界）+ 右侧纬度统计条；
b 11 情境平均任务功率与峰值余量（堆叠条）；c 同一行的协同节省网格分布（山脊图）；
d 5 个代表情境的成本嵌套方块：机群+供能总成本 ⊃ 供能成本 ⊃ 协同节省（面积按比例）；
e 训练机群逐网格节省 vs 不做协同时的市电占比，点色为风电装机占比；
h 能源侧节省从哪里来（桑基图，全部机群均值，不接推理、碳价 100）：省下的各项供能成本 → 毛节省 → 净节省与多建风光。
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


EXTRA = 52            # 为底部桑基图加高的 mm
H = 215 + EXTRA
DECOMP = SECTION1.parent / 'supply-cost-decomposition-20261119'
FLOW = [('grid_energy', 'Grid energy', '#1B3F70'), ('grid_capacity', 'Grid connection capacity', '#46A6B5'),
        ('carbon', 'Carbon cost', '#7FC3AE'), ('storage', 'Storage', '#C9A646')]
RE_COLOR = '#C0623F'


def R(x, y, w, h):
    """原 215 mm 画布上的位置换算到加高后的画布（原内容整体上移，底部留给桑基图）。"""
    return [x, (y * 215 + EXTRA) / H, w, h * 215 / H]


def FY(y):
    return (y * 215 + EXTRA) / H


def band(ax, x0, x1, a0, b0, a1, b1, color, alpha=.55):
    t = np.linspace(0, 1, 80)
    s = (1 - np.cos(np.pi * t)) / 2
    x = x0 + (x1 - x0) * t
    ax.fill_between(x, a0 + (a1 - a0) * s, b0 + (b1 - b0) * s, color=color, alpha=alpha, lw=0, zorder=1)


def sankey(ax):
    """省下的各项供能成本（左）→ 毛节省（中）→ 净节省与多建风光的花销（右），单位 USD/MWh（均值可加）。"""
    a = pd.read_csv(DECOMP / 'sec1-3-all.csv').set_index('intensity_of_max').loc[0]
    sc = pd.read_csv(DECOMP / 'sec1-3-by-scenario.csv')
    sc = sc[sc.intensity_of_max.eq(0)]
    v = {k: a[f'{k}_saving_mean_usd_mwh'] for k, _, _ in FLOW}
    pct = {k: a[f'{k}_share_of_net_pct'] for k, _, _ in FLOW}
    re, net = -a.renewables_saving_mean_usd_mwh, a.saving_mean_usd_mwh
    gross = sum(v.values())
    med = lambda c: a[c + '_median']
    notes = {'grid_energy': f'Grid share of electricity {med("seq_grid_share") * 100:.0f}% → {med("joint_grid_share") * 100:.0f}%',
             'grid_capacity': f'Grid capacity {med("seq_grid_cap_over_peak") * 100:.0f}% → {med("joint_grid_cap_over_peak") * 100:.0f}% of peak load',
             'carbon': 'Follows the smaller grid purchase',
             'storage': f'Storage {med("seq_storage_h"):.2f} → {med("joint_storage_h"):.2f} h of load'}
    gap, w = .75, .035
    xl, xm, xr = 0., .5, 1.
    top = gross + gap * (len(FLOW) - 1)
    m_top = top - (top - gross) / 2
    y_left, y_mid = top, m_top
    for k, name, col in FLOW:
        h = v[k]
        ax.add_patch(Rectangle((xl - w, y_left - h), w, h, color=col, lw=0, zorder=3))
        band(ax, xl, xm - w / 2, y_left - h, y_left, y_mid - h, y_mid, col)
        ax.text(xl - w - .015, y_left - h / 2 + .18, f'{name}  {h:.1f}  ({pct[k]:.0f}%)', ha='right', va='bottom',
                fontsize=6.6, fontweight='bold', color=col)
        ax.text(xl - w - .015, y_left - h / 2 + .1, notes[k], ha='right', va='top', fontsize=5.8, color=col)
        y_left -= h + gap
        y_mid -= h
    ax.add_patch(Rectangle((xm - w / 2, m_top - gross), w, gross, color='#5A5A5A', lw=0, zorder=3))
    ax.text(xm, m_top + .35, f'Gross saving {gross:.1f}', ha='center', va='bottom', fontsize=6.6, fontweight='bold',
            color='#3A3A3A')
    r_top = m_top
    ax.add_patch(Rectangle((xr, r_top - net), w, net, color=DARK, lw=0, zorder=3))
    band(ax, xm + w / 2, xr, m_top - net, m_top, r_top - net, r_top, '#4A4A4A', alpha=.45)
    re_top = r_top - net - 1.4
    ax.add_patch(Rectangle((xr, re_top - re), w, re, color=RE_COLOR, lw=0, zorder=3))
    band(ax, xm + w / 2, xr, m_top - gross, m_top - net, re_top - re, re_top, RE_COLOR, alpha=.6)
    lo, hi = sc.saving_share_of_total_cost_median_pct.min(), sc.saving_share_of_total_cost_median_pct.max()
    ax.text(xr + w + .015, r_top - net / 2 + .2, f'Net saving {net:.1f} USD/MWh', ha='left', va='bottom',
            fontsize=7.2, fontweight='bold', color=DARK)
    ax.text(xr + w + .015, r_top - net / 2 + .05,
            f'= {net / a.supply_cost_mean_usd_mwh * 100:.1f}% of supply cost ({a.supply_cost_mean_usd_mwh:.0f} USD/MWh)\n'
            f'= {lo:.2f}–{hi:.2f}% of fleet + supply cost', ha='left', va='top', fontsize=6, fontweight='bold',
            color='#9C3F22')
    ax.text(xr + w + .015, re_top - re / 2, f'Extra wind and solar  −{re:.1f}  ({-a.renewables_share_of_net_pct:.0f}%)\n'
            f'Solar {med("seq_pv_per_mw"):.1f} → {med("joint_pv_per_mw"):.1f} MW per MW load', ha='left', va='center',
            fontsize=6.2, fontweight='bold', color=RE_COLOR)
    ax.set_xlim(-.62, 1.62)
    ax.set_ylim(-.3, top + 1.3)
    ax.axis('off')
    rows = [{'node': k, 'usd_mwh': v[k], 'share_of_net_pct': pct[k]} for k, _, _ in FLOW]
    rows += [{'node': 'gross', 'usd_mwh': gross}, {'node': 'renewables', 'usd_mwh': -re,
             'share_of_net_pct': a.renewables_share_of_net_pct}, {'node': 'net', 'usd_mwh': net, 'share_of_net_pct': 100}]
    return pd.DataFrame(rows)


def main():
    setup()
    p = paired()
    p0 = p[p.intensity_of_max.eq(0)].merge(baseline()[['grid_id', 'scenario', 'grid_mwh0']], on=['grid_id', 'scenario'])
    p0['grid_share'] = 100 * p0.grid_mwh0 / p0.facility_mwh0
    shares = section1_shares()
    head = part1_summary().loc[SCENARIOS]

    fig = plt.figure(figsize=(180 / 25.4, H / 25.4))
    norm = mpl.colors.Normalize(0, 30)

    # ---------------- a 地图 + 右侧纬度条
    dtrn = p0[p0.scenario.eq('D-TRN')].set_index('grid_id')
    ax_map, _ = draw_map(fig, R(.03, .695, .74, .285), dtrn.saving_usd_mwh, SEQ, norm, lon_labels=False)
    frames = [(GROUP[use(s)], COLORS[use(s)], p0.loc[p0.scenario.eq(s), ['lat', 'saving_usd_mwh']]
               .rename(columns={'saving_usd_mwh': 'value'}), s == 'D-TRN') for s in REPRESENTATIVES]
    ax_lat, lat_rows = latitude_strip(fig, ax_map, .08, frames, 'USD/MWh', xlim=(0, 30))
    ax_lat.set_xticks([0, 10, 20, 30])
    handles = [Line2D([], [], color=COLORS[use(s)], lw=1.1, label=GROUP[use(s)] + ('' if s[0] == 'A' else '\n(GPU-heavy)'))
               for s in REPRESENTATIVES] + [Patch(facecolor=LAND, label='No eligible\ngrid cell')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.885, FY(.975)), fontsize=5.4, handlelength=1.2,
               labelspacing=.5, borderaxespad=0)
    colorbar(fig, R(.2, .69, .38, .008), SEQ, norm, [0, 5, 10, 15, 20, 25, 30],
             'Coordination saving, LLM training · GPU-heavy (USD per MWh of electricity)', extend='max')
    panel(fig, 'a', .012, FY(.99))

    # ---------------- b 堆叠条 + c 山脊图（共用行）
    y = np.arange(len(SCENARIOS))[::-1].astype(float)
    ax_b = fig.add_axes(R(.21, .395, .2, .235))
    potential = []
    ax_c = fig.add_axes(R(.46, .395, .26, .235))
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
    panel(fig, 'b', .012, FY(.655))

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
    panel(fig, 'c', .44, FY(.655))

    # ---------------- f 现有协同 vs 减配上限（与 b、c 共用行）
    up = downsizing_usd_mwh().loc[SCENARIOS]
    now = p0.groupby('scenario').saving_usd_mwh.median().loc[SCENARIOS]
    ax_f = fig.add_axes(R(.765, .395, .19, .235))
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
    panel(fig, 'd', .74, FY(.655))

    # ---------------- d 嵌套方块
    med = shares.groupby('scenario')[['share_supply_pct', 'share_total_pct', 'supply_of_total_pct']].median()
    ax_d = fig.add_axes(R(.015, .195, .53, .105))
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
    panel(fig, 'e', .012, FY(.3))

    # ---------------- e 散点：节省 vs 市电占比，点色为风电占比
    d = p0[p0.scenario.eq('D-TRN')]
    ax_e = fig.add_axes(R(.64, .07, .25, .2))
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
    cax = fig.add_axes(R(.9, .07, .007, .2))
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=wnorm, cmap=SEQ), cax=cax)
    cb.outline.set_linewidth(.3)
    cax.tick_params(labelsize=5, length=1.5, width=.4, pad=1)
    cb.set_label('Wind share of optimal wind + PV (%)', fontsize=5.5, labelpad=2)
    panel(fig, 'f', .57, FY(.3))

    # ---------------- g 上下限从何而来：节省比例 × 成本基数（背景为等节省额线）
    # 每条曲线从现有协同（只作用于供能成本）走到减配上限，每点多少买 2.5% 的服务器；被优化的成本基数随减配程度
    # 从供能成本线性扩大到“供能 + 机群设备成本”，横坐标 = 总节省 ÷ 成本基数，因此点所在等值线即该点的总节省额。
    b1 = baseline()
    annual_mwh = (b1.facility_mwh0 * 8760 / b1.hours0).groupby(b1.scenario).median()
    energy_base = (b1.supply_cost0_usd / b1.facility_mwh0).groupby(b1.scenario).median()
    p0b = p[p.intensity_of_max.eq(0)]
    energy_share = (p0b.V_usd / p0b.seq_cost_usd * 100).groupby(p0b.scenario).median()
    dn = pd.read_csv(INPUT / 'downsizing-part2-main.csv', encoding='utf-8-sig')
    dn = dn[dn.target.eq('economic')].set_index('scenario')
    fleet_base = dn.annual_fleet_equipment_cost_usd / annual_mwh
    fleet_share = dn.equipment_annual_saving_usd / dn.annual_fleet_equipment_cost_usd * 100
    ax_g = fig.add_axes(R(.085, .045, .33, .125))
    X, Y = np.meshgrid(np.linspace(0, 36, 200), np.logspace(np.log10(60), np.log10(6000), 200))
    Z = X / 100 * Y
    levels = [1, 3, 10, 30, 100, 300, 1000]
    ax_g.contourf(X, Y, Z, levels=levels, cmap=SEQ, norm=mpl.colors.LogNorm(1, 1000), alpha=.45, extend='both')
    cs = ax_g.contour(X, Y, Z, levels=levels, colors='white', linewidths=.5)
    ax_g.clabel(cs, fmt=lambda v: f'{v:g}', fontsize=4.6, inline=True)
    rows_g = []
    for sc in SCENARIOS:
        c = COLORS[use(sc)]
        lower = energy_share[sc] / 100 * energy_base[sc]
        r_max = fleet_share[sc]
        r = np.unique(np.r_[np.arange(0, r_max, 2.5), r_max])
        w = r / r_max
        base_r = energy_base[sc] + w * fleet_base[sc]
        save_r = lower + r / 100 * fleet_base[sc]
        x_r = save_r / base_r * 100
        ax_g.plot(x_r, base_r, color=shade(sc), lw=.8, zorder=3)
        ax_g.scatter(x_r[1:-1], base_r[1:-1], s=5, color=shade(sc), lw=0, zorder=4)
        ax_g.scatter(x_r[0], base_r[0], marker='o', s=12, facecolors='white', edgecolors=c, lw=.7, zorder=5)
        ax_g.scatter(x_r[-1], base_r[-1], marker='s', s=13, color=shade(sc), edgecolors='white', lw=.4, zorder=5)
        rows_g.append(pd.DataFrame({'scenario': sc, 'servers_avoided_pct': r, 'cost_base_usd_mwh': base_r,
                                    'saving_usd_mwh': save_r, 'share_of_base_pct': x_r}))
    ax_g.set_yscale('log')
    ax_g.set_xlim(0, 36)
    ax_g.set_ylim(60, 6000)
    ax_g.set_yticks([100, 1000], ['100', '1,000'])
    ax_g.minorticks_off()
    ax_g.tick_params(labelsize=5.2)
    ax_g.set_xlabel('Share of the cost base saved (%)', fontsize=5.8)
    ax_g.set_ylabel('Cost base (USD/MWh)', fontsize=5.8)
    ax_g.legend(handles=[Line2D([], [], marker='o', ls='none', markerfacecolor='white', markeredgecolor='#666666',
                                markersize=3.2, label='Current coordination\n(energy supply)'),
                         Line2D([], [], marker='o', ls='-', lw=.8, color='#888888', markersize=1.8,
                                label='Each dot: 2.5% more\nservers avoided'),
                         Line2D([], [], marker='s', ls='none', color='#888888', markersize=3.2,
                                label='Upper bound: downsizing\n(fleet + energy)')],
                loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=5, handletextpad=.3, borderaxespad=0)
    panel(fig, 'g', .012, FY(.175))

    # ---------------- h 桑基图：能源侧节省从哪里来
    ax_h = fig.add_axes([.03, 4 / H, .94, (EXTRA - 10) / H])
    flows = sankey(ax_h)
    panel(fig, 'h', .012, (EXTRA - 3) / H)

    export(fig, 'fig1')

    csv(dtrn.reset_index()[['grid_id', 'lat', 'lon', 'iso3', 'continent', 'saving_usd_mwh']], 'fig1a')
    csv(lat_rows, 'fig1a_latitude')
    csv(pd.DataFrame(potential), 'fig1b')
    csv(pd.concat(ridge), 'fig1c')
    csv(pd.DataFrame(rows), 'fig1e')
    csv(d[['grid_id', 'lat', 'lon', 'grid_share', 'wind_share', 'saving_usd_mwh']], 'fig1f')
    csv(pd.DataFrame({'scenario': SCENARIOS, 'current_coordination_usd_mwh': now.values, 'downsizing_usd_mwh': up.values, 'ratio': (up / now).values}), 'fig1d')
    csv(pd.concat(rows_g), 'fig1g')
    csv(flows, 'fig1h')
    return pd.DataFrame(rows), rho_g, rho_w


if __name__ == '__main__':
    print(main())
