"""图2：闲置算力变现的收益远超算电协同（第二节）。

a 9 个 GPU 情境：现有协同、减配上限、真实需求下卖推理、闲置全卖（对数轴分组横条，USD/MWh）；
b 盈亏平衡：全卖净收入随售价变化，与现有协同、减配上限的交点；
c 日内热条：外部推理需求与三类机群闲置（相对日均偏离），解释真实需求下收入为何收窄；
d 归因（仿 Whittaker 图）：闲置 GPU 比例 × 单位用电对应的 GPU 小时，背景为等收益分区；
e 稳健性：各网格相对全球中位数的偏离，左半为协同节省，右半为全卖收入（半边小提琴）。
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from residual_flex_pilot import asi_inference
from stable_limited_figs_common import (COLORS, DARK, DIV, GPU_SCENARIOS, GROUP, INNER, INPUT, NEUTRAL, ROOT, SELL,
                                        SEQ, baseline, colorbar, csv, downsizing_usd_mwh, export, guides, paired, panel, scenario_axis,
                                        sell, setup, shade, use)

PRICE = 2.9
MARKET = (2.2, 2.9)
TOKEN = 12.9
HOURLY = ROOT / 'results/flex-class-20261101/native-hourly'
TASKS = ['刚性', '未知', '可推迟·整体', '可推迟·可拆分']


def lighten(c, f):
    return tuple(np.array(mpl.colors.to_rgb(c)) * (1 - f) + f)


def main():
    setup()
    p = paired()
    p0 = p[p.intensity_of_max.eq(0) & p.scenario.isin(GPU_SCENARIOS)]
    coord = p0.groupby('scenario').saving_usd_mwh.median()
    sl = sell()
    sl['annual_mwh'] = sl.facility_mwh0 * 8760 / sl.hours0
    ideal = sl[sl.tier.eq('ideal')].copy()
    ideal['slope'] = ideal.revenue_usd_year / ideal.annual_mwh / PRICE
    ideal['cost'] = ideal.extra_supply_cost_usd_year / ideal.annual_mwh
    m = ideal.groupby('scenario')[['net_usd_mwh', 'slope', 'cost']].median().loc[GPU_SCENARIOS]
    m['real'] = sl[sl.tier.eq('real_max')].groupby('scenario').net_usd_mwh.median()
    m['coord'] = coord
    b0 = baseline()
    annual = (b0.facility_mwh0 * 8760 / b0.hours0).groupby(b0.scenario).median()
    m['down'] = downsizing_usd_mwh().loc[GPU_SCENARIOS]
    m['p_coord'] = (m.coord + m.cost) / m.slope
    m['p_down'] = (m.down + m.cost) / m.slope

    fig = plt.figure(figsize=(180 / 25.4, 220 / 25.4))

    # ---------------- a 四档量级（仿环形分组柱状图）：三个用途各占一段扇区，每个情境四根径向柱，半径取对数
    ax = fig.add_axes([.0, .585, .56, .405], projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    R0, RMAX = 1.25, np.log10(3500)
    top_gap, grp_gap, sc_gap = np.deg2rad(30), np.deg2rad(9), np.deg2rad(3.5)
    per_grp = (2 * np.pi - top_gap - 2 * grp_gap) / 3
    bw = (per_grp - 2 * sc_gap) / 12

    def rot(t, radial=False):
        d = (90 - np.rad2deg(t)) if radial else -np.rad2deg(t)
        d = (d + 180) % 360 - 180
        return d + 180 if abs(d) > 90 else d

    th = np.linspace(top_gap / 2, 2 * np.pi - top_gap / 2, 400)
    for k, lab in enumerate(['1', '10', '100', '1,000']):
        ax.plot(th, np.full_like(th, R0 + k), color='#D9D9D9', lw=.4, ls=(0, (2, 2)) if k else '-', zorder=1)
        ax.text(0, R0 + k, lab, ha='center', va='center', fontsize=4.8, color=DARK)
    ax.text(0, R0 + RMAX + .35, 'USD/MWh', ha='center', va='center', fontsize=5)
    rows_a = []
    for gi, u in enumerate(['SAI', 'TRN', 'INF']):
        g0 = top_gap / 2 + gi * (per_grp + grp_gap)
        ax.bar(g0 + per_grp / 2, RMAX + .1, width=per_grp, bottom=R0, color=COLORS[u], alpha=.07, lw=0, zorder=0)
        arc = np.linspace(g0, g0 + per_grp, 80)
        ax.plot(arc, np.full_like(arc, R0 + RMAX + .85), color=COLORS[u], lw=.9)
        gm = g0 + per_grp / 2
        ax.text(gm, R0 + RMAX + 1.3, GROUP[u], ha='center', va='center', rotation=rot(gm), fontsize=6.2,
                fontweight='bold', color=COLORS[u])
        for si, cfg in enumerate('BCD'):
            sc = f'{cfg}-{u}'
            c = shade(sc)
            s0 = g0 + si * (4 * bw + sc_gap)
            vals = [('coord', DARK, DARK), ('down', NEUTRAL, NEUTRAL), ('real', c, c),
                    ('net_usd_mwh', lighten(c, .6), c)]
            for k, (key, face, edge) in enumerate(vals):
                v = m.loc[sc, key]
                ax.bar(s0 + (k + .5) * bw, np.log10(v), width=bw * .86, bottom=R0, color=face, edgecolor=edge,
                       lw=.3, zorder=3)
                rows_a.append({'scenario': sc, 'metric': key, 'usd_mwh': v})
            tc = s0 + 2 * bw
            ax.text(tc, R0 + RMAX + .45, INNER[cfg], ha='center', va='center', rotation=rot(tc), fontsize=5)
    ax.text(0, 0, 'Value per MWh\nof electricity', ha='center', va='center', fontsize=5.2, transform=ax.transData)
    ax.set_ylim(0, R0 + RMAX + 1.6)
    ax.set_axis_off()
    fig.legend(handles=[Patch(facecolor=DARK, label='Current coordination'),
                        Patch(facecolor=NEUTRAL, label='Upper bound: downsizing'),
                        Patch(facecolor='#8C8C8C', label='Sell inference, real demand'),
                        Patch(facecolor='#DDDDDD', edgecolor='#8C8C8C', lw=.35, label='Sell all idle GPU-hours')],
               loc='upper center', bbox_to_anchor=(.28, .585), ncol=2, fontsize=5.4, handlelength=.9,
               columnspacing=1.2, frameon=False)
    panel(fig, 'a', .012, .995)

    # ---------------- b 盈亏平衡售价（仿悬浮阈值区间条）：区间从胜过现有协同到胜过减配上限，右侧为市场价相对盈亏平衡价的倍数
    ax_b = fig.add_axes([.71, .6, .27, .36])
    y = np.arange(len(GPU_SCENARIOS))[::-1].astype(float)
    MKT_FACE, MKT_EDGE = '#F2E2B3', '#B08A2E'
    ax_b.axvspan(*MARKET, color=MKT_FACE, lw=0, zorder=0)
    for xv in MARKET:
        ax_b.axvline(xv, color=MKT_EDGE, lw=.5, zorder=1)
    ax_b.axvline(TOKEN, color='#8C8C8C', lw=.6, ls=(0, (3, 2)), zorder=0)
    ytop = len(y) + .3
    ax_b.text(np.sqrt(MARKET[0] * MARKET[1]), ytop, 'Market\nrental\n2.2–2.9', ha='center', va='bottom', fontsize=5,
              color=MKT_EDGE, fontweight='bold', linespacing=.95)
    ax_b.text(TOKEN, ytop, 'Token-\nequivalent\n12.9', ha='center', va='bottom', fontsize=5, color='#6E6E6E',
              linespacing=.95)
    hb = .5
    for yi, sc in zip(y, GPU_SCENARIOS):
        lo, hi = m.loc[sc, 'p_coord'], m.loc[sc, 'p_down']
        ax_b.plot([hi, MARKET[0]], [yi, yi], color=MKT_EDGE, lw=.5, ls=(0, (1, 1.5)), zorder=1)
        ax_b.barh(yi, hi - lo, left=lo, height=hb, color=lighten(shade(sc), .55), edgecolor='none', zorder=2)
        ax_b.plot([lo, lo], [yi - hb / 2, yi + hb / 2], color=DARK, lw=1.6, solid_capstyle='butt', zorder=3)
        ax_b.plot([hi, hi], [yi - hb / 2, yi + hb / 2], color=NEUTRAL, lw=1.6, solid_capstyle='butt', zorder=3)
        ax_b.text(lo / 1.12, yi, f'{lo:.2f}', ha='right', va='center', fontsize=4.6)
        ax_b.text(hi * 1.1, yi, f'{hi:.2f}', ha='left', va='center', fontsize=4.6, zorder=4,
                  bbox=dict(facecolor='white', edgecolor='none', pad=.4))
        ax_b.text(np.sqrt(MARKET[1] * TOKEN), yi, f'×{MARKET[1] / hi:.1f}', ha='center', va='center', fontsize=5,
                  color=MKT_EDGE, fontweight='bold')
    rows_b = [m.loc[GPU_SCENARIOS, ['p_coord', 'p_down']].rename_axis('scenario').reset_index()]
    ax_b.set_xscale('log')
    ax_b.set_xlim(.008, 25)
    ax_b.set_xticks([.01, .1, 1, 10], ['0.01', '0.1', '1', '10'])
    ax_b.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax_b.set_ylim(-.6, len(y) + .25)
    ax_b.spines['left'].set_visible(False)
    scenario_axis(ax_b, GPU_SCENARIOS, y, outer=-.5, fontsize=5.4)
    ax_b.set_xlabel('Break-even selling price (USD per GPU-hour)', fontsize=6)
    guides(ax_b)
    y0 = y[0] + hb / 2 + .12
    lo0, hi0 = m.loc[GPU_SCENARIOS[0], 'p_coord'], m.loc[GPU_SCENARIOS[0], 'p_down']
    ax_b.text(lo0, y0, 'Beats\ncoordination', ha='center', va='bottom', fontsize=4.8, linespacing=.95)
    ax_b.text(hi0, y0, 'Beats\ndownsizing', ha='center', va='bottom', fontsize=4.8, linespacing=.95)
    ax_b.text(np.sqrt(MARKET[1] * TOKEN), y0, 'Market ÷\nbreak-even', ha='center', va='bottom', fontsize=4.8,
              color=MKT_EDGE, linespacing=.95)
    panel(fig, 'b', .56, .995)

    # ---------------- c 日 × 时日历热图：外部推理需求与三类机群闲置，取三者共同覆盖的整段窗口（训练平移 14 天，与第二节口径一致）
    asi = asi_inference(HOURLY / 'D-INF.parquet').set_index('time_utc')
    demand = asi.on + asi.off
    shift = {'D-SAI': 0, 'D-TRN': 14, 'D-INF': 0}
    idle_s = {}
    for sc in shift:
        fr = pd.read_parquet(HOURLY / f'{sc}.parquet')
        task = fr[TASKS].sum(axis=1).to_numpy()
        t = pd.to_datetime(fr.time_utc, utc=True) - pd.Timedelta(days=shift[sc])
        idle_s[sc] = pd.Series(1 - task / task.max(), index=t.to_numpy())
    start = max(v.index.min() for v in idle_s.values()).ceil('D')
    end = min(min(v.index.max() for v in idle_s.values()), demand.index.max()).floor('D')
    hours = pd.date_range(start, end, freq='h', inclusive='left')
    days = len(hours) // 24
    hours = hours[:days * 24]
    tiers = pd.read_csv(SELL / 'sell-inference-tiers-median.csv', encoding='utf-8-sig').set_index('scenario')
    d_full = demand.reindex(hours)
    win = pd.date_range(pd.Timestamp('2025-06-02', tz='UTC'), periods=14 * 24, freq='h')
    tt = np.arange(len(win)) / 24
    prof_rows = []
    for k, sc in enumerate(shift):
        c = COLORS[use(sc)]
        idle_full = idle_s[sc].reindex(hours)
        idle_w = idle_s[sc].reindex(win).to_numpy()
        dem_w = (d_full.reindex(win) / d_full.mean() * idle_full.mean()).to_numpy()
        axk = fig.add_axes([.165, .418 - k * .062, .37, .052])
        axk.fill_between(tt, 0, idle_w, color=lighten(c, .55), lw=0, zorder=1)
        axk.plot(tt, idle_w, color=c, lw=.6, zorder=2)
        over = np.nan_to_num(dem_w) > np.nan_to_num(idle_w, nan=np.inf)
        axk.fill_between(tt, idle_w, dem_w, where=over, color='#9C3F22', alpha=.55, lw=0, zorder=3, interpolate=True)
        axk.plot(tt, dem_w, color=DARK, lw=.7, zorder=4)
        axk.set_xlim(0, 14)
        axk.set_ylim(0, 1)
        axk.set_yticks([0, .5, 1], ['0', '50', '100'])
        axk.set_xticks(np.arange(0, 15, 2))
        axk.tick_params(labelsize=5.2, length=1.5, pad=1.5)
        if k < 2:
            axk.set_xticklabels([])
        else:
            axk.set_xticklabels([(win[0] + pd.Timedelta(days=int(d))).strftime('%d %b') for d in np.arange(0, 15, 2)])
        axk.text(-.13, .5, f'{GROUP[use(sc)]}\n{tiers.loc[sc, "max_intensity"] * 100:.0f}% sellable', transform=axk.transAxes,
                 ha='right', va='center', fontsize=5.6, color=c, fontweight='bold', linespacing=1.0)
        if k == 1:
            axk.set_ylabel('Share of fleet (%)', fontsize=5.8, labelpad=2)
        guides(axk, 'y')
        prof_rows.append(pd.DataFrame({'scenario': sc, 'time_utc': win, 'idle_share': idle_w, 'demand_scaled': dem_w}))
    axk.set_xlabel('Hour-by-hour, 2–15 June 2025 (UTC)', fontsize=6, labelpad=2)
    fig.legend(handles=[Patch(facecolor='#C9C9C9', edgecolor='#7A7A7A', lw=.5, label='Idle GPUs'),
                        Line2D([], [], color=DARK, lw=.7, label='External inference demand, same mean'),
                        Patch(facecolor='#9C3F22', alpha=.55, label='Demand above idle')],
               loc='lower center', bbox_to_anchor=(.33, .472), ncol=3, fontsize=5.3, handlelength=1.1,
               columnspacing=.9, frameon=False)
    panel(fig, 'c', .012, 0.503)

    # ---------------- d 归因：闲置比例 × 单位用电 GPU 小时（等收益分区）
    city = pd.read_csv(SELL / 'sell-inference-tiers-median.csv').set_index('scenario').loc[GPU_SCENARIOS]
    idle_share = city.mean_idle_share * 100
    gpu_h = city.gpus_A * 8760 / annual.loc[GPU_SCENARIOS]
    ax_d = fig.add_axes([0.66, 0.307, 0.25, 0.177])
    X, Y = np.meshgrid(np.linspace(20, 50, 200), np.linspace(1000, 2800, 200))
    Z = PRICE * X / 100 * Y
    levels = [500, 1000, 1500, 2000, 2500, 3000, 3500]
    znorm = mpl.colors.Normalize(500, 3500)
    ax_d.contourf(X, Y, Z, levels=levels, cmap=SEQ, norm=znorm, alpha=.55, extend='both')
    cs = ax_d.contour(X, Y, Z, levels=levels, colors='white', linewidths=.5)
    ax_d.clabel(cs, fmt=lambda v: f'{v:,.0f}', fontsize=4.6, inline=True)
    marks = {'B': 'o', 'C': 's', 'D': '^'}
    for sc in GPU_SCENARIOS:
        ax_d.scatter(idle_share[sc], gpu_h[sc], marker=marks[sc[0]], s=26, color=COLORS[use(sc)], edgecolors='white',
                     lw=.6, zorder=5)
    ax_d.set_xlim(20, 50)
    ax_d.set_ylim(1000, 2800)
    ax_d.set_xlabel('Idle GPU share (%)', fontsize=6)
    ax_d.set_ylabel('GPU-hours per MWh', fontsize=6)
    ax_d.set_title('Iso-lines: gross revenue at 2.9 USD/GPU-h (USD/MWh)', fontsize=5.4, pad=3)
    h1 = [Line2D([], [], marker='o', ls='none', color=COLORS[u], markersize=3.5, label=GROUP[u]) for u in ('SAI', 'TRN', 'INF')]
    h2 = [Line2D([], [], marker=mk, ls='none', color='#777777', markersize=3.5, label=INNER[k]) for k, mk in marks.items()]
    ax_d.legend(handles=h1 + h2, loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=5, handletextpad=.2,
                labelspacing=.35, borderaxespad=0)
    panel(fig, 'd', .6, 0.503)

    # ---------------- e 电价梯度：横轴为当地市电价格（100 美元/MWh + 碳价 100 美元/t × 当地碳因子），纵轴为相对全球中位数的偏离
    p0y = p0.assign(vy=p0.V_usd * 8760 / p0.hours, price=100 + 100 * p0.carbon_t_mwh)
    p0y['dev'] = p0y.groupby('scenario').vy.transform(lambda v: (v / v.median() - 1) * 100)
    se = ideal[['grid_id', 'scenario', 'net_usd_year']].merge(p0y[['grid_id', 'scenario', 'price']],
                                                            on=['grid_id', 'scenario'])
    se['dev'] = se.groupby('scenario').net_usd_year.transform(lambda v: (v / v.median() - 1) * 100)
    pr_lo, pr_hi = np.percentile(p0y.price, [.5, 99.5])
    edges = np.unique(np.quantile(p0y.price, np.linspace(0, 1, 16)))
    mids = (edges[:-1] + edges[1:]) / 2

    def binned(frame):
        g = frame.groupby(pd.cut(frame.price, edges, include_lowest=True), observed=False).dev
        return g.median().to_numpy(), g.quantile(.1).to_numpy(), g.quantile(.9).to_numpy()

    ax_e = fig.add_axes([0.075, 0.07, 0.6, 0.15])
    hb_ = ax_e.hexbin(p0y.price, p0y.dev, gridsize=(26, 11), extent=(pr_lo, pr_hi, -70, 110), cmap=SEQ, bins='log',
                      mincnt=1, lw=.1, edgecolors='white', zorder=1)
    rows_e = []
    for u in ('SAI', 'TRN', 'INF'):
        q = p0y[p0y.scenario.str.endswith(u)]
        med, _, _ = binned(q)
        ax_e.plot(mids, med, color=COLORS[u], lw=1.3, zorder=3)
        qs = se[se.scenario.str.endswith(u)]
        smed, s10, s90 = binned(qs)
        rows_e.append(pd.DataFrame({'use': u, 'price_mid': mids, 'coord_median': med, 'sell_median': smed,
                                    'sell_p10': s10, 'sell_p90': s90}))
    ax_e.plot(mids, binned(se)[0], color='black', lw=1.6, zorder=4)
    ax_e.axhline(0, color='#888888', lw=.4, zorder=2)
    ax_e.set_xlim(pr_lo, pr_hi)
    ax_e.set_ylim(-70, 110)
    ax_e.set_yticks([-50, 0, 50, 100])
    ax_e.tick_params(labelsize=5.4)
    ax_e.set_xlabel('Local grid electricity price incl. carbon cost (USD/MWh)', fontsize=6)
    ax_e.set_ylabel('Deviation from global\nmedian (%)', fontsize=6)
    ax_e.legend(handles=[Patch(facecolor=SEQ(.55), label='Current coordination, grid cells'),
                         *[Line2D([], [], color=COLORS[u], lw=1.3, label=f'Coordination median, {GROUP[u]}')
                           for u in ('SAI', 'TRN', 'INF')],
                         Line2D([], [], color='black', lw=1.6, label='Sell all idle GPU-hours')],
                loc='upper left', fontsize=5.1, ncol=1, handlelength=1.3, labelspacing=.25, borderaxespad=.3,
                frameon=True, framealpha=.85, edgecolor='none')
    cb_ax = fig.add_axes([0.685, 0.08, 0.006, 0.13])
    cb = fig.colorbar(hb_, cax=cb_ax)
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
    cb.set_label('Grid cells', fontsize=5.4, labelpad=1)

    ax_z = fig.add_axes([0.8, 0.07, 0.18, 0.15])
    for u, frame in zip(('SAI', 'TRN', 'INF'), rows_e):
        ax_z.fill_between(mids, frame.sell_p10, frame.sell_p90, color=COLORS[u], alpha=.18, lw=0)
        ax_z.plot(mids, frame.sell_median, color=COLORS[u], lw=1.1)
    ax_z.axhline(0, color='#888888', lw=.4)
    ax_z.set_xlim(pr_lo, pr_hi)
    ax_z.set_ylim(-1.5, 1.5)
    ax_z.set_yticks([-1, 0, 1])
    ax_z.tick_params(labelsize=5.4)
    ax_z.set_xlabel('Grid price (USD/MWh)', fontsize=6)
    ax_z.set_ylabel('Sell all idle,\ndeviation (%)', fontsize=6)
    guides(ax_z, 'y')
    panel(fig, 'e', .012, 0.233)

    export(fig, 'fig2')

    csv(m.reset_index(), 'fig2a_b')
    csv(pd.concat(rows_b), 'fig2b')
    csv(pd.concat(prof_rows), 'fig2c')
    csv(pd.DataFrame({'scenario': GPU_SCENARIOS, 'idle_gpu_share_pct': idle_share.values,
                      'gpu_hours_per_mwh': gpu_h.values, 'sell_all_usd_mwh': m.net_usd_mwh.values}), 'fig2d')
    csv(pd.concat(rows_e), 'fig2e')
    return m, pd.concat(rows_e)


if __name__ == '__main__':
    m, e = main()
    print(m[['coord', 'down', 'real', 'net_usd_mwh', 'p_coord', 'p_down']].round(2).to_string())
    print(e.round(1).to_string())
