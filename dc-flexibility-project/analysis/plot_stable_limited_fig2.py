"""图2：闲置算力变现的收益远超电力调度（第二节）。

a 气泡矩阵：9 个 GPU 情境 × 6 档（协同节省、上限 25/50/75%、真实上限、理想全卖），面积 ∝ USD/MWh；
b 半环：真实上限 / 理想，环下为可承接上限；
c 日历热图：三类 GPU-heavy 机群原任务下逐时闲置任务功率（÷ 平均闲置），说明闲置的时间分布；
d 六边形格网地图：训练机群真实上限年净收入相对全球中位数的偏离；
e 散点：峰谷计价带来的变化 vs 地点离散度（P10–P90）。
"""
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Wedge
import numpy as np
import pandas as pd

from stable_limited_figs_common import (COLORS, GPU_SCENARIOS, GROUP, INNER, RESIDUAL, ROOT, colorbar, csv, export,
                                        land_axes, paired, panel, scenario_axis, sell, setup, shade, use)

TIERS = ['coord', 'real_25', 'real_50', 'real_75', 'real_max', 'ideal']
TIER_LABEL = ['Coordi-\nnation', 'Sell 25%\nof cap', 'Sell 50%\nof cap', 'Sell 75%\nof cap', 'Sell\nrealistic\ncap',
              'Sell all\nidle (ideal)']
TASKS = ['刚性', '未知', '可推迟·整体', '可推迟·可拆分']


def idle_calendar(scenario):
    f = pd.read_parquet(ROOT / f'results/flex-class-20261101/native-hourly/{scenario}.parquet')
    task = f[TASKS].sum(axis=1)
    idle = task.max() - task
    t = pd.to_datetime(f.time_utc, utc=True)
    day = ((t - t.iloc[0].normalize()).dt.total_seconds() // 86400).astype(int)
    grid = pd.DataFrame({'day': day, 'hour': t.dt.hour, 'ratio': idle / idle.mean()})
    deferrable = (f['可推迟·整体'] + f['可推迟·可拆分']).sum() / task.sum()
    return grid.pivot_table(index='hour', columns='day', values='ratio'), deferrable, grid


def main():
    setup()
    p = paired()
    coord = p[p.intensity_of_max.eq(0) & p.scenario.isin(GPU_SCENARIOS)].groupby('scenario').saving_usd_mwh.median()
    sl = sell()
    med = sl.groupby(['scenario', 'tier']).net_usd_mwh.median().unstack()
    med['coord'] = coord
    med = med.loc[GPU_SCENARIOS, TIERS]
    cap = pd.read_csv(RESIDUAL / 'max-intensity.csv').set_index('scenario')

    fig = plt.figure(figsize=(180 / 25.4, 200 / 25.4))

    # ---------------- a 气泡矩阵
    ax = fig.add_axes([.2, .6, .45, .345])
    y = np.arange(len(GPU_SCENARIOS))[::-1]
    smax = 170.
    vmax = med.to_numpy().max()
    for j, t in enumerate(TIERS):
        for yi, sc in zip(y, GPU_SCENARIOS):
            v = med.loc[sc, t]
            c = shade(sc)
            if t == 'coord':
                ax.scatter(j, yi, s=max(smax * v / vmax, 1.2), color='#333333', zorder=3)
            else:
                ax.scatter(j, yi, s=smax * v / vmax, facecolors=c if t != 'ideal' else 'white', edgecolors=c,
                           lw=.8, alpha=.9 if t != 'ideal' else 1, zorder=3)
            r_pt = np.sqrt(max(smax * v / vmax, 1.2)) / 2
            ax.annotate(f'{v:,.0f}' if v >= 10 else f'{v:.1f}', (j, yi), xytext=(r_pt + 1.5, 0),
                        textcoords='offset points', ha='left', va='center', fontsize=5, color='#333333')
    ax.axvline(.5, color='#BBBBBB', lw=.6, ls=(0, (3, 2)))
    ax.set_xlim(-.5, len(TIERS) - .4)
    ax.set_ylim(-.8, len(y) - .4)
    ax.set_xticks(range(len(TIERS)), TIER_LABEL, fontsize=5.6)
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', length=0, pad=2)
    for side in ('left', 'bottom', 'right', 'top'):
        ax.spines[side].set_visible(False)
    scenario_axis(ax, GPU_SCENARIOS, y, outer=-.2)
    for v in (10, 100, 1000):
        pass
    leg = [ax.scatter([], [], s=smax * v / vmax, facecolors='#BBBBBB', edgecolors='none', label=f'{v:,}')
           for v in (100, 500, 2000)]
    ax.legend(handles=leg, title='USD/MWh (circle area)', title_fontsize=5.6, loc='upper center',
              bbox_to_anchor=(.45, -.01), ncol=3, fontsize=5.5, columnspacing=1.5, handletextpad=.4)
    panel(fig, 'a', .012, .99)

    # ---------------- b 半环
    ratio = med.real_max / med.ideal * 100
    groups = [('TRN', ['B-TRN', 'C-TRN', 'D-TRN']), ('SAI', ['B-SAI', 'C-SAI', 'D-SAI']),
              ('INF', ['B-INF', 'C-INF', 'D-INF'])]
    ax_b = fig.add_axes([.71, .6, .27, .36])
    ax_b.set_xlim(-1.15, 1.15)
    ax_b.set_ylim(-.2, 3.55)
    ax_b.set_aspect('equal')
    ax_b.axis('off')
    for i, (u, scs) in enumerate(groups):
        cy = 2.45 - i * 1.17
        r = ratio[scs]
        mid = r.median()
        ax_b.add_patch(Wedge((0, cy), .82, 0, 180, width=.24, facecolor='#ECECEC', edgecolor='none'))
        ax_b.add_patch(Wedge((0, cy), .82, 180 - 180 * mid / 100, 180, width=.24, facecolor=COLORS[u],
                             edgecolor='none'))
        ax_b.text(0, cy + .12, f'{mid:.0f}%', ha='center', va='bottom', fontsize=7.5, fontweight='bold',
                  color=COLORS[u])
        ax_b.text(0, cy - .02, GROUP[u], ha='center', va='top', fontsize=5.8, color=COLORS[u])
        k = cap.loc[scs, 'max_intensity']
        ax_b.text(0, cy - .26, f'cap {k.min():.2f}–{k.max():.2f}× mean idle' if k.max() - k.min() >= .005
                  else f'cap {k.min():.2f}× mean idle', ha='center', va='top', fontsize=5.2)
    ax_b.text(0, 3.5, 'Realistic cap ÷ ideal', ha='center', va='center', fontsize=5.8)
    panel(fig, 'b', .68, .99)

    # ---------------- c 日历热图
    cmap = mpl.colormaps['magma_r']
    norm = mpl.colors.Normalize(0, 2)
    cal = {}
    rows = []
    days_max = 0
    for sc in ['D-TRN', 'D-SAI', 'D-INF']:
        cal[sc] = idle_calendar(sc)
        days_max = max(days_max, cal[sc][0].shape[1])
    x0, w_full, h = .2, .68, .055
    for i, sc in enumerate(['D-TRN', 'D-SAI', 'D-INF']):
        piv, deferrable, grid = cal[sc]
        w = w_full * piv.shape[1] / days_max
        axc = fig.add_axes([x0, .465 - i * (h + .016), w, h])
        axc.set_facecolor('#D9D9D9')
        axc.pcolormesh(piv.columns, piv.index, piv.to_numpy(), cmap=cmap, norm=norm, shading='nearest',
                       rasterized=True)
        axc.set_ylim(23.5, -.5)
        axc.set_yticks([0, 12, 23], ['00 UTC', '12', '23'], fontsize=5)
        axc.tick_params(axis='both', length=1.5, pad=1)
        if i < 2:
            axc.set_xticks([])
        else:
            axc.set_xlabel('Day of the observed window', fontsize=6)
        axc.tick_params(axis='x', labelsize=5)
        for side in ('top', 'right'):
            axc.spines[side].set_visible(True)
        fig.text(x0 - .03, .465 - i * (h + .016) + h / 2, f'{GROUP[use(sc)]}\n{deferrable * 100:.0f}% of task\nenergy deferrable',
                 ha='right', va='center', fontsize=5.6, color=COLORS[use(sc)])

        rows.append(grid.assign(scenario=sc))
    cax = fig.add_axes([.915, .465 - 2 * (h + .016), .008, 3 * h + 2 * .016])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend='max', ticks=[0, .5, 1, 1.5, 2])
    cb.outline.set_linewidth(.3)
    cax.tick_params(labelsize=5, length=1.5, width=.4, pad=1)
    cb.set_label('Hourly idle task power ÷ mean idle', fontsize=5.6, labelpad=2)
    panel(fig, 'c', .012, .545)

    # ---------------- d 六边形格网地图
    d = sl[sl.scenario.eq('D-TRN') & sl.tier.eq('real_max')]
    dev = (d.net_usd_year / d.net_usd_year.median() - 1) * 100
    ax_d = land_axes(fig, [.03, .045, .56, .215])
    xy = ax_d.projection.transform_points(ccrs.PlateCarree(), d.lon.to_numpy(), d.lat.to_numpy())
    dmap = mpl.colormaps['RdBu_r']
    dnorm = mpl.colors.TwoSlopeNorm(0, -1.5, 1.5)
    x_lo, x_hi = ax_d.get_xlim()
    y_lo, y_hi = ax_d.get_ylim()
    ax_d.hexbin(xy[:, 0], xy[:, 1], C=dev.to_numpy(), reduce_C_function=np.mean, gridsize=(64, 18), cmap=dmap,
                norm=dnorm, mincnt=1, lw=.1, edgecolors='white', extent=(x_lo, x_hi, y_lo, y_hi), zorder=2,
                rasterized=True)
    colorbar(fig, [.12, .035, .38, .008], dmap, dnorm, [-1.5, -1, -.5, 0, .5, 1, 1.5],
             'Net revenue vs global median, LLM training · GPU-heavy, realistic cap (%)', extend='both')
    panel(fig, 'd', .012, .275)

    # ---------------- e 定价 vs 地点
    ax_e = fig.add_axes([.73, .07, .25, .18])
    tou = sl.groupby(['scenario', 'tier'])[['net_usd_year', 'net_tou_usd_year']].median()
    tou = (tou.net_tou_usd_year / tou.net_usd_year - 1).unstack().real_max * 100
    g = sl[sl.tier.eq('real_max')].groupby('scenario').net_usd_year
    spread = (g.quantile(.9) - g.quantile(.1)) / g.median() * 100
    marks = {'B': 'o', 'C': 's', 'D': '^'}
    for sc in GPU_SCENARIOS:
        ax_e.scatter(tou[sc], spread[sc], marker=marks[sc[0]], s=18, color=COLORS[use(sc)], edgecolors='white',
                     lw=.4, zorder=3)
    ax_e.set_xlim(-3.5, .5)
    ax_e.set_ylim(0, 3.2)
    ax_e.axvline(0, color='#888888', lw=.5)
    ax_e.set_xlabel('Change with peak/off-peak pricing (%)')
    ax_e.set_ylabel('Spread across\nlocations, P10–P90 (%)')
    h1 = [Line2D([], [], marker='o', ls='none', color=COLORS[u], markersize=3.5, label=GROUP[u]) for u in ('TRN', 'SAI', 'INF')]
    h2 = [Line2D([], [], marker=m, ls='none', color='#777777', markersize=3.5, label=INNER[k]) for k, m in marks.items()]
    ax_e.legend(handles=h1 + h2, loc='upper left', fontsize=5, ncol=2, handletextpad=.2, columnspacing=.5,
                borderaxespad=.2)
    panel(fig, 'e', .64, .275)

    export(fig, 'fig2')

    csv(med.reset_index(), 'fig2a')
    csv(pd.DataFrame({'scenario': ratio.index, 'realistic_over_ideal_pct': ratio.values,
                      'max_intensity': cap.loc[ratio.index, 'max_intensity'].values}), 'fig2b')
    csv(pd.concat(rows), 'fig2c')
    csv(d[['grid_id', 'lat', 'lon', 'net_usd_year']].assign(deviation_pct=dev.values), 'fig2d')
    csv(pd.DataFrame({'scenario': GPU_SCENARIOS, 'tou_change_pct': tou[GPU_SCENARIOS].values,
                      'location_spread_p10_p90_pct': spread[GPU_SCENARIOS].values}), 'fig2e')
    return med, ratio, {sc: cal[sc][1] for sc in cal}


if __name__ == '__main__':
    med, ratio, dfr = main()
    print(med.round(1).to_string())
    print(dfr)
