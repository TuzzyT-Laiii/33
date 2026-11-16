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
                                        SEQ, baseline, colorbar, csv, export, guides, paired, panel, scenario_axis,
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
    down = pd.read_csv(INPUT / 'downsizing-part2-main.csv', encoding='utf-8-sig')
    down = down[down.target.eq('economic')].set_index('scenario')
    m['down'] = (down.net_annual_saving_vs_A_usd_median / annual).loc[GPU_SCENARIOS]
    m['p_coord'] = (m.coord + m.cost) / m.slope
    m['p_down'] = (m.down + m.cost) / m.slope

    fig = plt.figure(figsize=(180 / 25.4, 205 / 25.4))

    # ---------------- a 四档量级
    ax = fig.add_axes([.215, .6, .44, .38])
    y = np.arange(len(GPU_SCENARIOS))[::-1].astype(float)
    h = .2
    for yi, sc in zip(y, GPU_SCENARIOS):
        c = shade(sc)
        bars = [(1.5 * h, m.loc[sc, 'coord'], DARK, DARK), (.5 * h, m.loc[sc, 'down'], NEUTRAL, NEUTRAL),
                (-.5 * h, m.loc[sc, 'real'], c, c), (-1.5 * h, m.loc[sc, 'net_usd_mwh'], lighten(c, .6), c)]
        for off, v, face, edge in bars:
            ax.barh(yi + off, v - 1, left=1, height=h * .9, color=face, edgecolor=edge, lw=.35)
            ax.text(v * 1.08, yi + off, f'{v:,.0f}' if v >= 10 else f'{v:.1f}', va='center', fontsize=4.6, zorder=6)
    ax.set_xscale('log')
    ax.set_xlim(1, 6000)
    ax.set_xticks([1, 10, 100, 1000], ['1', '10', '100', '1,000'])
    ax.set_ylim(-.6, len(y) - .4)
    ax.spines['left'].set_visible(False)
    scenario_axis(ax, GPU_SCENARIOS, y, outer=-.3, fontsize=5.6)
    ax.set_xlabel('USD per MWh of data-centre electricity (log scale)', fontsize=6)
    guides(ax)
    ax.legend(handles=[Patch(facecolor=DARK, label='Current coordination'),
                       Patch(facecolor=NEUTRAL, label='Upper bound: downsizing'),
                       Patch(facecolor='#8C8C8C', label='Sell inference, real demand'),
                       Patch(facecolor='#DDDDDD', edgecolor='#8C8C8C', lw=.35, label='Sell all idle GPU-hours')],
              loc='upper center', bbox_to_anchor=(.42, -.08), ncol=2, fontsize=5.4, handlelength=.9, columnspacing=1.2)
    panel(fig, 'a', .012, .995)

    # ---------------- b 盈亏平衡
    ax_b = fig.add_axes([.75, .64, .23, .33])
    prices = np.logspace(-2, np.log10(20), 300)
    ax_b.axvspan(*MARKET, color='#E9E9E9', lw=0, zorder=0)
    ax_b.axvline(TOKEN, color='#BBBBBB', lw=.6, ls=(0, (3, 2)), zorder=0)
    rows_b = []
    for sc in ['D-SAI', 'D-TRN', 'D-INF']:
        c = COLORS[use(sc)]
        net = m.loc[sc, 'slope'] * prices - m.loc[sc, 'cost']
        ok = net > 0
        ax_b.plot(prices[ok], net[ok], color=c, lw=1.2)
        ax_b.axhline(m.loc[sc, 'coord'], color=c, lw=.6, ls=(0, (2, 1.5)))
        ax_b.axhline(m.loc[sc, 'down'], color=c, lw=.6, ls=(0, (5, 1.5, 1, 1.5)))
        for lvl, key in [('coord', 'p_coord'), ('down', 'p_down')]:
            ax_b.scatter(m.loc[sc, key], m.loc[sc, lvl], s=12, color=c, edgecolors='white', lw=.5, zorder=5)
        rows_b.append(pd.DataFrame({'scenario': sc, 'price_usd_gpu_h': prices, 'net_usd_mwh': net}))
    ax_b.set_xscale('log')
    ax_b.set_yscale('log')
    ax_b.set_xlim(.01, 20)
    ax_b.set_ylim(1, 1e4)
    ax_b.set_xticks([.01, .1, 1, 10], ['0.01', '0.1', '1', '10'])
    ax_b.set_yticks([1, 10, 100, 1000, 1e4], ['1', '10', '100', '1,000', '10,000'])
    ax_b.minorticks_off()
    ax_b.set_xlabel('Selling price (USD per GPU-hour)', fontsize=6)
    ax_b.set_ylabel('USD/MWh', fontsize=6)
    ax_b.text(MARKET[0] / 1.05, 1.3, 'Market rental', ha='right', va='bottom', fontsize=5, rotation=90)
    ax_b.text(TOKEN / 1.05, 1.3, 'Token-equivalent', ha='right', va='bottom', fontsize=5, rotation=90)
    guides(ax_b, 'both')
    ax_b.legend(handles=[Line2D([], [], color='#666666', lw=1.2, label='Sell all idle'),
                         Line2D([], [], color='#666666', lw=.6, ls=(0, (2, 1.5)), label='Current coordination'),
                         Line2D([], [], color='#666666', lw=.6, ls=(0, (5, 1.5, 1, 1.5)), label='Downsizing'),
                         Line2D([], [], marker='o', ls='none', color='#666666', markersize=3, label='Break-even')] +
                [Line2D([], [], color=COLORS[u], lw=1.2, label=GROUP[u]) for u in ('SAI', 'TRN', 'INF')],
                loc='upper left', fontsize=4.8, handlelength=1.6, labelspacing=.25, borderaxespad=.3)
    panel(fig, 'b', .69, .995)

    # ---------------- c 日内热条
    asi = asi_inference(HOURLY / 'D-INF.parquet')
    dem = asi.groupby(asi.time_utc.dt.hour).on.mean().to_numpy()
    strips = [('External inference demand', dem / dem.mean() - 1, '#555555', None)]
    prof_rows = []
    for sc in ['D-SAI', 'D-TRN', 'D-INF']:
        f = pd.read_parquet(HOURLY / f'{sc}.parquet')
        task = f[TASKS].sum(axis=1)
        idle = pd.Series(1 - task / task.max()).groupby(pd.to_datetime(f.time_utc, utc=True).dt.hour.to_numpy()).mean().to_numpy()
        r = np.corrcoef(idle, dem)[0, 1]
        defer = (f['可推迟·整体'] + f['可推迟·可拆分']).sum() / task.sum()
        strips.append((f'{GROUP[use(sc)]} idle ({defer * 100:.0f}% deferrable)', idle / idle.mean() - 1,
                       COLORS[use(sc)], r))
        prof_rows.append(pd.DataFrame({'scenario': sc, 'hour_utc': np.arange(24), 'idle_over_K': idle,
                                       'demand_over_mean': dem / dem.mean(), 'corr_with_demand': r,
                                       'deferrable_share': defer}))
    mat = np.vstack([v for _, v, _, _ in strips]) * 100
    ax_c = fig.add_axes([.215, .37, .31, .13])
    hnorm = mpl.colors.TwoSlopeNorm(0, -50, 50)
    ax_c.imshow(mat, cmap=DIV, norm=hnorm, aspect='auto', interpolation='nearest')
    for i in range(1, len(strips)):
        ax_c.axhline(i - .5, color='white', lw=1.5)
    ax_c.set_yticks(range(len(strips)), [n for n, _, _, _ in strips], fontsize=5.3)
    for lab, (_, _, c, _) in zip(ax_c.get_yticklabels(), strips):
        lab.set_color(c)
    ax_c.set_xticks([0, 6, 12, 18, 23])
    ax_c.tick_params(length=0, pad=2, labelsize=5.3)
    ax_c.set_xlabel('Hour of day (UTC)', fontsize=6)
    for sp in ax_c.spines.values():
        sp.set_visible(False)
    for i, (_, _, c, r) in enumerate(strips):
        if r is not None:
            ax_c.text(24.0, i, f'r = {r:+.2f}', va='center', fontsize=5.3, color=c)
    colorbar(fig, [.215, .315, .2, .007], DIV, hnorm, [-50, -25, 0, 25, 50], 'Deviation from daily mean (%)',
             extend='both')
    panel(fig, 'c', .012, .54)

    # ---------------- d 归因：闲置比例 × 单位用电 GPU 小时（等收益分区）
    city = pd.read_csv(SELL / 'sell-inference-tiers-median.csv').set_index('scenario').loc[GPU_SCENARIOS]
    idle_share = city.mean_idle_share * 100
    gpu_h = city.gpus_A * 8760 / annual.loc[GPU_SCENARIOS]
    ax_d = fig.add_axes([.66, .33, .25, .19])
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
    panel(fig, 'd', .6, .54)

    # ---------------- e 半边小提琴：地点偏离
    p0y = p0.assign(vy=p0.V_usd * 8760 / p0.hours)
    ax_e = fig.add_axes([.215, .085, .76, .14])
    xs = np.arange(len(GPU_SCENARIOS)).astype(float)
    grid = np.linspace(-70, 110, 500)
    rows_e = []
    for x0, sc in zip(xs, GPU_SCENARIOS):
        cv = p0y.loc[p0y.scenario.eq(sc), 'vy']
        cdev = (cv / cv.median() - 1).to_numpy() * 100
        sv = ideal.loc[ideal.scenario.eq(sc), 'net_usd_year']
        sdev = (sv / sv.median() - 1).to_numpy() * 100
        for dev, side, col in [(cdev, -1, NEUTRAL), (sdev, 1, shade(sc))]:
            lo, hi = np.percentile(dev, [.5, 99.5])
            gg = grid[(grid >= lo) & (grid <= hi)] if side < 0 else np.linspace(lo, hi, 50)
            k = gaussian_kde(dev, bw_method=.15 if side < 0 else .5)(gg)
            k = k / k.max() * .42
            ax_e.fill_betweenx(gg, x0, x0 + side * k, color=col, lw=0, alpha=.9)
            q10, q90 = np.percentile(dev, [10, 90])
            ax_e.plot([x0 + side * .05] * 2, [q10, q90], color=DARK, lw=.7)
        rows_e.append({'scenario': sc, 'coord_p10': np.percentile(cdev, 10), 'coord_p90': np.percentile(cdev, 90),
                       'sell_p10': np.percentile(sdev, 10), 'sell_p90': np.percentile(sdev, 90)})
    ax_e.axhline(0, color='#AAAAAA', lw=.5)
    ax_e.set_xlim(-.6, len(xs) - .4)
    ax_e.set_ylim(-60, 90)
    ax_e.set_yticks([-50, 0, 50])
    ax_e.set_ylabel('Deviation across\nlocations (%)', fontsize=6)
    scenario_axis(ax_e, GPU_SCENARIOS, xs, axis='x', outer=-.28)
    ax_e.tick_params(axis='x', labelrotation=0)
    for lab in ax_e.get_xticklabels():
        lab.set_rotation(0)
    ax_e.legend(handles=[Patch(facecolor=NEUTRAL, label='Current coordination (left)'),
                         Patch(facecolor='#8C8C8C', label='Sell all idle (right; P10–P90 within ±0.7%)'),
                         Line2D([], [], color=DARK, lw=.7, label='P10–P90')],
                loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=3, fontsize=5.2, handlelength=1, columnspacing=1)
    guides(ax_e, 'y')
    panel(fig, 'e', .012, .25)

    export(fig, 'fig2')

    csv(m.reset_index(), 'fig2a_b')
    csv(pd.concat(rows_b), 'fig2b')
    csv(pd.concat(prof_rows), 'fig2c')
    csv(pd.DataFrame({'scenario': GPU_SCENARIOS, 'idle_gpu_share_pct': idle_share.values,
                      'gpu_hours_per_mwh': gpu_h.values, 'sell_all_usd_mwh': m.net_usd_mwh.values}), 'fig2d')
    csv(pd.DataFrame(rows_e), 'fig2e')
    return m, pd.DataFrame(rows_e)


if __name__ == '__main__':
    m, e = main()
    print(m[['coord', 'down', 'real', 'net_usd_mwh', 'p_coord', 'p_down']].round(2).to_string())
    print(e.round(1).to_string())
