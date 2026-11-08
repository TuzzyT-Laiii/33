import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from paper_figure_zh_common import (ENGLISH, SCENARIOS, color, csv, export,
                                    guides, label, panel, read_panel, rows, setup, shade)


def main():
    setup()
    train = read_panel('fig1a')
    continents = read_panel('fig1a_continents')
    head = read_panel('fig1b').set_index('scenario').loc[SCENARIOS].reset_index()
    head['label'] = head.scenario.map(label)
    costs = read_panel('fig1c')
    distribution = read_panel('fig1d')
    stats = read_panel('fig1d_summary')
    for frame, name in [(train, 'fig1a'), (continents, 'fig1a_continents'), (head, 'fig1b'),
                        (costs, 'fig1c'), (distribution, 'fig1d'), (stats, 'fig1d_summary')]:
        csv(frame, name)
    assert len(train) == 4842
    assert distribution.groupby(['scenario', 'boundary']).size().eq(4842).all()

    fig = plt.figure(figsize=(180 / 25.4, 162 / 25.4))
    ax_map = fig.add_axes([.015, .615, .715, .33], projection=ccrs.Robinson())
    cells = gpd.read_parquet(ENGLISH / 'map-assets/map-block-land-cells.parquet')
    countries = gpd.read_file('zip://' + str(ENGLISH / 'map-assets/countries-10m.zip'))
    cells = cells.merge(train[['grid_id', 'savings_pct']], on='grid_id', validate='one_to_one')
    assert len(cells) == 4842 and cells.crs.to_epsg() == 4326
    mapped = cells.loc[~cells.geometry.is_empty]
    cmap = mpl.colormaps['viridis']
    norm = mpl.colors.Normalize(0, 40)
    ax_map.set_global()
    for geometries, face, edge, lw, order in [
        (countries.geometry.simplify(.04), '#ECECEC', 'none', 0, 0),
        (mapped.geometry, cmap(norm(mapped.savings_pct.to_numpy())), 'none', 0, 1),
        (countries.geometry.simplify(.04), 'none', '#999999', .12, 2),
    ]:
        ax_map.add_geometries(geometries, crs=ccrs.PlateCarree(), facecolor=face,
                             edgecolor=edge, linewidth=lw, antialiased=False,
                             zorder=order, rasterized=True)
    ax_map.spines['geo'].set_visible(False)
    cax = fig.add_axes([.17, .586, .37, .012])
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                       orientation='horizontal', ticks=[0, 10, 20, 30, 40])
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor('face')
    bar.set_label('能源系统成本节省（%）', labelpad=2)
    bar.outline.set_linewidth(.4)
    panel(fig, 'a', .012, .985, 'GPU主导型·大模型训练｜规划阶段｜经济最优', .065)
    cont_names = {'Africa': '非洲', 'Asia': '亚洲', 'Europe': '欧洲', 'North America': '北美洲',
                  'South America': '南美洲', 'Oceania': '大洋洲'}
    ax_cont = fig.add_axes([.825, .665, .15, .215])
    yy = np.arange(len(continents))
    ax_cont.barh(yy, continents['median'], color=color('D-TRN'), height=.55)
    ax_cont.set_yticks(yy, continents.continent.map(cont_names))
    assert continents.continent.map(cont_names).notna().all()
    ax_cont.set_xlim(0, 25)
    ax_cont.set_xticks([0, 10, 20])
    ax_cont.set_xlabel('节省中位数（%）')
    ax_cont.tick_params(axis='y', length=0)
    for y, v in zip(yy, continents['median']):
        ax_cont.text(v + .5, y, f'{v:.1f}', va='center', fontsize=6)
    fig.text(.78, .925, '各大洲中位数', fontsize=6.5)
    fig.text(.76, .555, f'4,842 个有效网格\n{len(mapped):,} 个可绘区块\n2° 区块裁切至陆地', fontsize=6, color='#555555')

    ax_head = fig.add_axes([.255, .145, .19, .335])
    ax_cost = fig.add_axes([.505, .145, .215, .335])
    ax_dist = fig.add_axes([.785, .145, .19, .335])
    for ax in [ax_head, ax_cost, ax_dist]:
        rows(ax, labels=ax is ax_head)
    y = np.arange(11)[::-1]
    for yi, u, s in zip(y, head.mean_over_K_pct, SCENARIOS):
        ax_head.barh(yi, u, height=.64, color=shade(s))
        ax_head.barh(yi, 100 - u, left=u, height=.64, color='#F1F1F1', edgecolor='#999999', linewidth=.4)
        ax_head.text((100 + u) / 2, yi, f'{100 - u:.0f}', ha='center', va='center', fontsize=6)
    ax_head.set_xlim(0, 100)
    ax_head.set_xticks([0, 50, 100])
    ax_head.set_xlabel('占任务峰值 K 的比例（%）')
    ax_head.legend(handles=[Patch(facecolor='#777777', label='平均负荷'),
                            Patch(facecolor='#F1F1F1', edgecolor='#999999', label='峰值余量')],
                   loc='upper center', bbox_to_anchor=(.5, -.18), ncol=2, handlelength=.9, columnspacing=.6)
    panel(fig, 'b', .012, .525, '任务峰值余量', .255)

    indexed = costs.set_index(['scenario', 'boundary'])
    markers = [('fleet_annual_usd', 'planning', 's', '机群', '#606060'),
               ('energy_annual_usd', 'planning', 'D', '无柔性能源', '#303030'),
               ('saving_annual_usd', 'planning', 'o', '规划节省', None),
               ('saving_annual_usd', 'fixed', 'o', '已建成节省', None)]
    for s, yi in zip(SCENARIOS, y):
        ax_cost.plot([indexed.loc[(s, 'fixed'), 'saving_annual_usd'] / 1e6,
                      indexed.loc[(s, 'planning'), 'fleet_annual_usd'] / 1e6],
                     [yi, yi], color='#D7D7D7', lw=.55, zorder=0)
    for column, boundary, m, text, neutral in markers:
        values = np.array([indexed.loc[(s, boundary), column] for s in SCENARIOS]) / 1e6
        edge = neutral or [shade(s) for s in SCENARIOS]
        ax_cost.scatter(values, y, s=14, marker=m, facecolors='white' if boundary == 'fixed' else edge,
                        edgecolors=edge, linewidths=.7, zorder=3)
    ax_cost.set_xscale('log')
    ax_cost.set_xlim(.075, 300)
    ax_cost.set_xticks([.1, 1, 10, 100], ['0.1', '1', '10', '100'])
    ax_cost.minorticks_off()
    ax_cost.set_xlabel('年化成本（百万美元/年）\n10 MW 参考规模；对数轴')
    guides(ax_cost, 'x')
    handles = [Line2D([], [], marker=m, color='none', markerfacecolor='white' if b == 'fixed' else n or '#555555',
                      markeredgecolor=n or '#555555', markersize=3, label=t) for _, b, m, t, n in markers]
    ax_cost.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.18), ncol=2,
                   handletextpad=.3, columnspacing=.7)
    panel(fig, 'c', .505, .525, '年化成本量级')

    for boundary, offset in [('planning', .19), ('fixed', -.19)]:
        for s, yi in zip(SCENARIOS, y):
            q = stats.loc[stats.scenario.eq(s) & stats.boundary.eq(boundary)].iloc[0]
            box = dict(whislo=q['min'], q1=q.q25, med=q['median'], q3=q.q75, whishi=q['max'], fliers=[])
            c = shade(s)
            ax_dist.bxp([box], positions=[yi + offset], orientation='horizontal', widths=.29,
                        showfliers=False, patch_artist=True,
                        boxprops={'facecolor': c if boundary == 'planning' else 'white', 'edgecolor': c, 'linewidth': .65},
                        whiskerprops={'color': c, 'linewidth': .55}, capprops={'color': c, 'linewidth': .55},
                        medianprops={'color': '#222222', 'linewidth': .65})
    ax_dist.axvline(3, color='#8B253E', ls=(0, (3, 2)), lw=.8)
    ax_dist.set_xlim(0, 3.15)
    ax_dist.set_yticks(y, [])
    ax_dist.set_xticks([0, 1, 2, 3])
    ax_dist.set_xlabel('节省占机群年化成本（%）')
    guides(ax_dist, 'x')
    ax_dist.text(2.91, 10.85, '3% 参考线', ha='right', va='bottom', fontsize=6, color='#8B253E')
    ax_dist.legend(handles=[Patch(facecolor='#777777', label='规划阶段'),
                            Patch(facecolor='white', edgecolor='#777777', label='已建成')],
                   loc='upper center', bbox_to_anchor=(.5, -.18), ncol=2, handlelength=.9, columnspacing=.7)
    panel(fig, 'd', .785, .525, '相对机群成本的价值')
    ax_dist.text(.67, .70, f'全部网格最大值\n{distribution.fleet_savings_pct.max():.2f}%',
                 transform=ax_dist.transAxes, ha='center', va='center', fontsize=6.5)
    fig.text(.065, .956, '11 情境：峰值余量 27–52%；能源侧节省中位数：规划 3.3–13.8%，已建成 2.2–7.3%。',
             fontsize=6, color='#555555')
    fig.text(.255, .018, '经济最优；成本按各情境原生观测窗口等效年化；箱体为网格四分位区间，须线为全范围。',
             fontsize=6, color='#555555')
    export(fig, 'fig1', {'grids_per_scenario_boundary': 4842, 'mapped_blocks': len(mapped),
                        'map': 'Robinson；WGS84；2°陆地相交区块；未作空间插值',
                        'max_fleet_savings_pct': float(distribution.fleet_savings_pct.max()),
                        'input': 'PR #63 面板源数据；科学内容保持一致'})


if __name__ == '__main__':
    main()
