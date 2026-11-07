import json

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from paper_figure_common import (OUT, RUN, SCENARIOS, SOURCE, color, csv, export,
                                 load_economics, panel, setup, shade)


def main():
    font = setup()
    head, valid, factor = load_economics()
    train = valid.loc[valid.scenario.eq('D-TRN') & valid.boundary.eq('planning')].copy()
    csv(train[['grid_id', 'block_id', 'lat', 'lon', 'block_lat', 'block_lon', 'country', 'continent',
               'scenario', 'boundary', 'savings_pct']], 'fig1a')
    continents = train.groupby('continent').savings_pct.agg(n='size', median='median').sort_values('median')
    csv(continents.reset_index(), 'fig1a_continents')
    csv(head.reset_index(), 'fig1b')
    costs = (valid.groupby(['scenario', 'boundary'])
             .agg(n=('grid_id', 'size'), fleet_annual_usd=('fleet_annual_usd', 'median'),
                  energy_annual_usd=('energy_annual_usd', 'median'), saving_annual_usd=('saving_annual_usd', 'median'),
                  energy_savings_pct=('savings_pct', 'median'), fleet_savings_pct=('fleet_savings_pct', 'median'))
             .reset_index())
    csv(costs, 'fig1c')
    csv(valid[['grid_id', 'lat', 'lon', 'scenario', 'boundary', 'window_hours', 'baseline_cost_usd',
               'flex_cost_usd', 'saving_annual_usd', 'fleet_annual_usd', 'fleet_savings_pct']], 'fig1d')

    fig = plt.figure(figsize=(180 / 25.4, 160 / 25.4))
    projection = ccrs.Robinson()
    ax_map = fig.add_axes([.015, .605, .715, .345], projection=projection)
    cells = gpd.read_parquet(OUT / 'map-assets/map-block-land-cells.parquet')
    countries = gpd.read_file('zip://' + str(OUT / 'map-assets/countries-10m.zip'))
    cells = cells.merge(train[['grid_id', 'savings_pct']], on='grid_id', validate='one_to_one')
    assert len(cells) == len(train)
    assert cells.crs.to_epsg() == 4326
    cells['map_drawn'] = ~cells.geometry.is_empty
    csv(cells.drop(columns='geometry'), 'fig1a_geometry_audit')
    mapped = cells.loc[cells.map_drawn]
    cmap = mpl.colormaps['viridis']
    norm = mpl.colors.Normalize(0, 40)
    ax_map.set_global()
    ax_map.set_facecolor('#FFFFFF')
    ax_map.add_geometries(countries.geometry.simplify(.04), crs=ccrs.PlateCarree(),
                          facecolor='#ECECEC', edgecolor='none', linewidth=0, zorder=0, rasterized=True)
    ax_map.add_geometries(mapped.geometry, crs=ccrs.PlateCarree(),
                          facecolor=cmap(norm(mapped.savings_pct.to_numpy())), edgecolor='none',
                          linewidth=0, antialiased=False, zorder=1, rasterized=True)
    ax_map.add_geometries(countries.geometry.simplify(.04), crs=ccrs.PlateCarree(),
                          facecolor='none', edgecolor='#999999', linewidth=.12, zorder=2, rasterized=True)
    ax_map.spines['geo'].set_visible(False)
    cax = fig.add_axes([.175, .58, .37, .012])
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', ticks=[0, 10, 20, 30, 40])
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor('face')
    bar.set_label('Energy-system cost savings (%)', labelpad=2)
    bar.outline.set_linewidth(.4)
    fig.text(.095, .982, 'GPU-heavy LLM training · Planning · Economic optimum', fontsize=6.5, va='top')
    panel(fig, 'a', .012, .985)
    ax_cont = fig.add_axes([.825, .655, .15, .225])
    yy = np.arange(len(continents))
    ax_cont.barh(yy, continents['median'], color=color('D-TRN'), height=.55)
    ax_cont.set_yticks(yy, continents.index, fontsize=6)
    ax_cont.set_xlim(0, 24)
    ax_cont.set_xticks([0, 10, 20])
    ax_cont.set_xlabel('Median savings (%)')
    ax_cont.tick_params(axis='y', length=0)
    for y, value in zip(yy, continents['median']):
        ax_cont.text(value + .5, y, f'{value:.1f}', va='center', fontsize=6)
    fig.text(.75, .917, 'Continental medians', fontsize=6.5)
    fig.text(.75, .55, f'4,842 valid grids; {len(mapped):,} mapped\n2° blocks clipped to land', fontsize=6, color='#555555')

    y = np.arange(len(SCENARIOS))[::-1]
    ax_head = fig.add_axes([.255, .135, .195, .345])
    ax_cost = fig.add_axes([.50, .135, .22, .345])
    ax_dist = fig.add_axes([.77, .135, .21, .345])
    for ax in (ax_head, ax_cost, ax_dist):
        ax.set_ylim(-.65, 10.65)
        for divider in [8.5, 5.5, 2.5]:
            ax.axhline(divider, color='#E6E6E6', linewidth=.45, zorder=0)
        ax.tick_params(axis='y', length=0)
    used = head.loc[SCENARIOS, 'mean_over_K_pct'].to_numpy()
    for yi, u, scenario in zip(y, used, SCENARIOS):
        ax_head.barh(yi, u, height=.64, color=shade(scenario), linewidth=0)
        ax_head.barh(yi, 100 - u, left=u, height=.64, color='#F1F1F1', edgecolor='#999999', linewidth=.4)
        ax_head.text(u + (100 - u) / 2, yi, f'{100 - u:.0f}', ha='center', va='center', fontsize=6)
    ax_head.set_yticks(y, head.loc[SCENARIOS, 'label'], fontsize=6)
    for text, scenario in zip(ax_head.get_yticklabels(), SCENARIOS):
        text.set_color(color(scenario))
    ax_head.set_xlim(0, 100)
    ax_head.set_xticks([0, 50, 100])
    ax_head.set_xlabel('Share of task peak K (%)')
    ax_head.legend(handles=[Patch(facecolor='#777777', label='Mean load'), Patch(facecolor='#F1F1F1', edgecolor='#999999', label='Headroom')],
                   loc='upper center', bbox_to_anchor=(.50, -.22), ncol=2, handlelength=1, columnspacing=.7)
    panel(fig, 'b', .012, .515)
    fig.text(.255, .515, 'Native task headroom', va='top', fontsize=6.5)

    indexed = costs.set_index(['scenario', 'boundary'])
    markers = [('fleet_annual_usd', 'planning', 's', 'Fleet', '#606060'),
               ('energy_annual_usd', 'planning', 'D', 'Energy baseline', '#303030'),
               ('saving_annual_usd', 'planning', 'o', 'Savings: planning', None),
               ('saving_annual_usd', 'fixed', 'o', 'Savings: as-built', None)]
    for scenario, yi in zip(SCENARIOS, y):
        ax_cost.plot([indexed.loc[(scenario, 'fixed'), 'saving_annual_usd'] / 1e6,
                      indexed.loc[(scenario, 'planning'), 'fleet_annual_usd'] / 1e6], [yi, yi], color='#D7D7D7', lw=.55, zorder=0)
    for column, boundary, marker, label, neutral in markers:
        values = np.array([indexed.loc[(s, boundary), column] for s in SCENARIOS]) / 1e6
        edge = neutral or [shade(s) for s in SCENARIOS]
        face = 'white' if boundary == 'fixed' else edge
        ax_cost.scatter(values, y, s=14, marker=marker, facecolors=face, edgecolors=edge, linewidths=.7, zorder=3)
    ax_cost.set_xscale('log')
    ax_cost.set_xlim(.075, 300)
    ax_cost.set_xticks([.1, 1, 10, 100], ['0.1', '1', '10', '100'])
    ax_cost.minorticks_off()
    ax_cost.set_yticks(y, [])
    ax_cost.set_xlabel('Annual cost (US$ million)\n10 MW reference · Log scale')
    ax_cost.grid(axis='x', color='#E6E6E6', lw=.45)
    handles = [Line2D([], [], marker=m, color='none', markerfacecolor=('white' if b == 'fixed' else (n or '#555555')),
                      markeredgecolor=n or '#555555', markersize=3, label=label) for _, b, m, label, n in markers]
    ax_cost.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.22), ncol=2, handletextpad=.3, columnspacing=.6, fontsize=6)
    panel(fig, 'c', .50, .515)
    fig.text(.53, .515, 'Cost magnitudes', va='top', fontsize=6.5)

    stats = []
    for boundary, offset in [('planning', .19), ('fixed', -.19)]:
        for scenario, yi in zip(SCENARIOS, y):
            values = valid.loc[valid.scenario.eq(scenario) & valid.boundary.eq(boundary), 'fleet_savings_pct']
            q = values.quantile([0, .25, .5, .75, 1]).to_numpy()
            box = dict(whislo=q[0], q1=q[1], med=q[2], q3=q[3], whishi=q[4], fliers=[])
            c = shade(scenario)
            ax_dist.bxp([box], positions=[yi + offset], vert=False, widths=.29, showfliers=False, patch_artist=True,
                        boxprops={'facecolor': c if boundary == 'planning' else 'white', 'edgecolor': c, 'linewidth': .65},
                        whiskerprops={'color': c, 'linewidth': .55}, capprops={'color': c, 'linewidth': .55},
                        medianprops={'color': '#222222', 'linewidth': .65})
            stats.append({'scenario': scenario, 'boundary': boundary, 'n': len(values),
                          **dict(zip(['min', 'q25', 'median', 'q75', 'max'], q))})
    csv(pd.DataFrame(stats), 'fig1d_summary')
    ax_dist.axvline(3, color='#8B253E', ls=(0, (3, 2)), lw=.8)
    ax_dist.text(2.94, 10.9, '3%', ha='right', color='#8B253E', fontsize=6)
    ax_dist.set_xlim(0, 3.15)
    ax_dist.set_yticks(y, [])
    ax_dist.set_xticks([0, 1, 2, 3])
    ax_dist.set_xlabel('Savings / annual fleet cost (%)')
    ax_dist.grid(axis='x', color='#E6E6E6', lw=.45)
    ax_dist.text(.03, 1.025, f'Maximum {valid.fleet_savings_pct.max():.2f}%', transform=ax_dist.transAxes,
                 ha='left', va='bottom', fontsize=6)
    ax_dist.legend(handles=[Patch(facecolor='#777777', label='Planning'), Patch(facecolor='white', edgecolor='#777777', label='As-built')],
                   loc='upper center', bbox_to_anchor=(.5, -.22), ncol=2, handlelength=1, columnspacing=.6)
    panel(fig, 'd', .77, .515)
    fig.text(.80, .515, 'Value relative to fleet cost', va='top', fontsize=6.5)
    export(fig, 'fig1')
    (SOURCE / 'fig1_metadata.json').write_text(json.dumps({'font': font, 'annualization_factor': factor,
        'annualization': 'Native-window energy costs and savings multiplied by 8760 / window hours',
        'map': 'Robinson; WGS84 cached 2-degree land intersections; no spatial interpolation',
        'box_whiskers': 'minimum and maximum; boxes: 25th to 75th percentiles; line: median',
        'grids_per_scenario_boundary': 4842, 'mapped_blocks': len(mapped), 'empty_land_intersections': int((~cells.map_drawn).sum()),
        'economic_optimum': 'alpha is null; both solves optimal'}, indent=2) + '\n', 'utf-8')
    print(f'Fig. 1 exported; font={font}; n={len(valid)}; maximum={valid.fleet_savings_pct.max():.6f}%')


if __name__ == '__main__':
    main()
