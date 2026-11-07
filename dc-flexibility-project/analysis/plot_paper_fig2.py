import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from paper_figure_common import (OUT, REPRESENTATIVES, RUN, SOURCE, USES, color,
                                 csv, export, panel, setup)

VARIABLES = ['cf_wind', 'cf_pv', 'carbon', 'abs_lat', 'wind_daily_cv', 'pv_season', 'temp_mean', 'pv_wind_corr']
LABELS = ['Wind capacity factor', 'Solar capacity factor', 'Grid carbon intensity', 'Absolute latitude',
          'Daily wind variability', 'Solar seasonality', 'Mean temperature', 'Wind–solar correlation']
EDGES = [0, .05, .10, .15, .20, .25, .30, .35, .45, 1.]


def main():
    font = setup()
    table = pd.read_parquet(RUN / 'attribution/attribution-table.parquet')
    regression = pd.read_csv(RUN / 'attribution/regression.csv')
    fits = pd.read_csv(RUN / 'attribution/fit.csv')
    assert (table.groupby(['scenario', 'boundary']).size() == 4842).all()
    assert not table.duplicated(['grid_id', 'scenario', 'boundary']).any()
    plan = table.loc[table.boundary.eq('planning') & table.scenario.isin(REPRESENTATIVES)].copy()
    plan['wind_bin_index'] = pd.cut(plan.cf_wind, EDGES, labels=False, include_lowest=True)
    assert plan.wind_bin_index.notna().all()
    bins = (plan.groupby(['scenario', 'wind_bin_index']).agg(n=('grid_id', 'size'),
            median=('savings_pct', 'median'), q25=('savings_pct', lambda v: v.quantile(.25)),
            q75=('savings_pct', lambda v: v.quantile(.75)), cf_wind_median=('cf_wind', 'median')).reset_index())
    bins['lower_edge'] = bins.wind_bin_index.map(dict(enumerate(EDGES[:-1])))
    bins['upper_edge'] = bins.wind_bin_index.map(dict(enumerate(EDGES[1:])))
    csv(bins, 'fig2a')
    csv(plan[['grid_id', 'scenario', 'boundary', 'cf_wind', 'savings_pct', 'wind_bin_index']], 'fig2a_gridvalues')
    train = plan.loc[plan.scenario.eq('D-TRN')].copy()
    train['grid_share_pct'] = train.grid_share * 100
    correlation = train.grid_share.corr(train.savings_pct)
    csv(train[['grid_id', 'lat', 'lon', 'scenario', 'boundary', 'grid_share', 'grid_share_pct', 'cf_wind', 'savings_pct']], 'fig2b')
    exo = regression.loc[regression.model.eq('exogenous') & regression.scenario.isin(REPRESENTATIVES)].copy()
    exo = exo.merge(fits[['scenario', 'boundary', 'r2_exogenous']], on=['scenario', 'boundary'], validate='many_to_one')
    exo['important_delta_r2_ge_0_02'] = exo.delta_r2.ge(.02)
    exo['variable_label'] = exo.variable.map(dict(zip(VARIABLES, LABELS)))
    csv(exo.drop(columns=['label']), 'fig2c')

    fig = plt.figure(figsize=(180 / 25.4, 154 / 25.4))
    ax_wind = fig.add_axes([.082, .635, .37, .315])
    ax_grid = fig.add_axes([.60, .635, .295, .315])
    lines = {'A-HPC': ('s', (0, (4, 2))), 'A-CLD': ('D', (0, (2, 2))), 'D-SAI': ('^', '-'),
             'D-TRN': ('o', '-'), 'D-INF': ('v', (0, (5, 2, 1, 2)))}
    for scenario in REPRESENTATIVES:
        values = bins.loc[bins.scenario.eq(scenario)].sort_values('wind_bin_index')
        x = values.wind_bin_index.to_numpy()
        marker, style = lines[scenario]
        ax_wind.plot(x, values['median'], marker=marker, ls=style, color=color(scenario),
                     ms=2.5, lw=1.15 if scenario == 'D-TRN' else .85, label=USES[scenario.split('-')[1]])
        if scenario == 'D-TRN':
            ax_wind.fill_between(x, values.q25, values.q75, color=color(scenario), alpha=.16, linewidth=0)
    ax_wind.set_ylim(0, 26)
    ax_wind.set_yticks([0, 5, 10, 15, 20, 25])
    ax_wind.set_xlim(-.25, 8.25)
    labels = ['≤0.05', '0.05–\n0.10', '0.10–\n0.15', '0.15–\n0.20', '0.20–\n0.25', '0.25–\n0.30', '0.30–\n0.35', '0.35–\n0.45', '>0.45']
    ax_wind.set_xticks(range(9), labels, fontsize=6)
    ax_wind.set_xlabel('Annual mean wind capacity factor (bins)', labelpad=15)
    ax_wind.set_ylabel('Median energy-system cost savings (%)')
    ax_wind.grid(axis='y', color='#E6E6E6', lw=.45)
    ax_wind.legend(loc='upper left', frameon=False, ncol=1, fontsize=6, borderaxespad=.2,
                   handlelength=2, labelspacing=.3)
    counts = bins.loc[bins.scenario.eq('D-TRN')].sort_values('wind_bin_index').n
    for x, n in enumerate(counts):
        ax_wind.text(x, -4.1, str(n), ha='center', fontsize=6, color='#666666', clip_on=False)
    ax_wind.text(-.55, -4.1, 'n', ha='right', fontsize=6, color='#666666', clip_on=False)
    panel(fig, 'a', .012, .985)
    fig.text(.082, .985, 'Planning · Economic optimum', va='top', fontsize=6.5)

    scatter = ax_grid.scatter(train.grid_share_pct, train.savings_pct, c=train.cf_wind, cmap='viridis',
                              norm=mpl.colors.Normalize(0, .8), s=3.2, alpha=.60, linewidths=0)
    ax_grid.set_xlim(0, 100)
    ax_grid.set_ylim(0, 40)
    ax_grid.set_xticks([0, 25, 50, 75, 100])
    ax_grid.set_yticks([0, 10, 20, 30, 40])
    ax_grid.set_xlabel('Grid electricity / baseline facility demand (%)')
    ax_grid.set_ylabel('Energy-system cost savings (%)')
    ax_grid.text(.97, .96, f'Pearson r = {correlation:.2f}\nn = {len(train):,}', transform=ax_grid.transAxes,
                 ha='right', va='top', fontsize=6.5)
    cax = fig.add_axes([.913, .635, .012, .315])
    # 色标保持不透明，使颜色数值的含义独立于散点叠加密度。
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=scatter.norm, cmap=scatter.cmap), cax=cax, ticks=[0, .2, .4, .6, .8])
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor('face')
    bar.set_label('Annual mean wind capacity factor', fontsize=6, labelpad=3)
    bar.outline.set_linewidth(.4)
    panel(fig, 'b', .53, .985)
    fig.text(.60, .985, 'GPU-heavy LLM training · Planning', va='top', fontsize=6.5)

    columns = [(boundary, scenario) for boundary in ['planning', 'fixed'] for scenario in REPRESENTATIVES]
    beta = exo.pivot(index='variable', columns=['boundary', 'scenario'], values='beta_pp_per_sd').reindex(index=VARIABLES, columns=columns)
    delta = exo.pivot(index='variable', columns=['boundary', 'scenario'], values='delta_r2').reindex(index=VARIABLES, columns=columns)
    assert beta.notna().all().all()
    ax_heat = fig.add_axes([.255, .17, .65, .30])
    limit = 4.5
    assert beta.abs().max().max() <= limit
    image = ax_heat.pcolormesh(np.arange(11) - .5, np.arange(9) - .5, beta.to_numpy(),
                               cmap='RdBu_r', norm=TwoSlopeNorm(0, -limit, limit),
                               shading='flat', rasterized=False)
    ax_heat.set_xlim(-.5, 9.5)
    ax_heat.set_ylim(7.5, -.5)
    for row in range(len(VARIABLES)):
        for column in range(len(columns)):
            value = beta.iat[row, column]
            label = '0.0' if abs(value) < .05 else f'{value:+.1f}'
            ax_heat.text(column, row, label, ha='center', va='center', fontsize=6.5,
                         weight='bold' if delta.iat[row, column] >= .02 else 'normal',
                         color='white' if abs(value) > .56 * limit else '#202020')
    ax_heat.set_yticks(range(8), LABELS, fontsize=6.5)
    short_labels = ['Scientific\ncomputing\n(CPU)', 'Cloud\ncomputing\n(CPU)', 'Scientific\nAI', 'LLM\ntraining', 'Inference']
    ax_heat.set_xticks(range(10), short_labels * 2, fontsize=6)
    for label, scenario in zip(ax_heat.get_xticklabels(), REPRESENTATIVES * 2):
        label.set_color(color(scenario))
    ax_heat.tick_params(length=0, pad=4)
    ax_heat.axvline(4.5, color='white', lw=2)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.set_xticks(np.arange(-.5, 10, 1), minor=True)
    ax_heat.set_yticks(np.arange(-.5, 8, 1), minor=True)
    ax_heat.grid(which='minor', color='white', linewidth=.4)
    ax_heat.tick_params(which='minor', length=0)
    for x, label in [(2, 'Planning'), (7, 'As-built')]:
        ax_heat.text(x, -.87, label, ha='center', va='bottom', fontsize=7, clip_on=False)
    fit_index = fits.set_index(['boundary', 'scenario'])
    for i, key in enumerate(columns):
        ax_heat.text((i + .5) / 10, -.255, f'{fit_index.loc[key, "r2_exogenous"]:.2f}',
                     transform=ax_heat.transAxes, ha='center', va='top', fontsize=6, clip_on=False)
    ax_heat.text(-.025, -.255, 'R²', transform=ax_heat.transAxes, ha='right', va='top', fontsize=6.5)
    heat_cax = fig.add_axes([.925, .17, .012, .30])
    bar = fig.colorbar(image, cax=heat_cax, ticks=[-4, -2, 0, 2, 4])
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor('face')
    bar.set_label('Coefficient (percentage points per SD)', fontsize=6, labelpad=3)
    bar.outline.set_linewidth(.4)
    panel(fig, 'c', .012, .52)
    fig.text(.255, .52, 'Exogenous regression · Bold: ΔR² ≥ 0.02', va='top', fontsize=6.5)
    fig.text(.255, .035, 'GPU scenarios: GPU-heavy · All panels: economic optimum', fontsize=6, color='#555555')
    export(fig, 'fig2')
    (SOURCE / 'fig2_metadata.json').write_text(json.dumps({'font': font, 'n_per_fit': 4842,
        'correlation': float(correlation), 'regression': 'OLS with eight z-scored predictors; outcome is not standardized',
        'bold': 'R-squared loss after dropping predictor >= 0.02; no significance test',
        'feature_year': 2025, 'spatial_autocorrelation_adjustment': False,
        'wind_bins': 'Right-closed; first bin includes zero; equal-spaced categorical bin positions',
        'shading': '25th to 75th grid percentiles for GPU-heavy LLM training only'}, indent=2) + '\n', 'utf-8')
    print(f'Fig. 2 exported; font={font}; r={correlation:.8f}; fits={len(columns)}')


if __name__ == '__main__':
    main()
