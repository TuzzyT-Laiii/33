import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from paper_figure_zh_common import (REPRESENTATIVES, color, csv, export, guides,
                                    marker, panel, read_panel, setup, use)

VARIABLES = ['cf_wind', 'cf_pv', 'carbon', 'abs_lat', 'wind_daily_cv', 'pv_season', 'temp_mean', 'pv_wind_corr']
LABELS = ['风电出力系数', '光伏出力系数', '市电碳强度', '纬度绝对值',
          '风电日际变异系数', '光伏季节性', '平均气温', '风光相关系数']


def main():
    setup()
    bins = read_panel('fig2a')
    grids = read_panel('fig2a_gridvalues')
    train = read_panel('fig2b')
    exo = read_panel('fig2c')
    exo['variable_label'] = exo.variable.map(dict(zip(VARIABLES, LABELS)))
    for df, name in [(bins, 'fig2a'), (grids, 'fig2a_gridvalues'), (train, 'fig2b'), (exo, 'fig2c')]:
        csv(df, name)
    correlation = train.grid_share.corr(train.savings_pct)
    assert len(train) == 4842
    fig = plt.figure(figsize=(180 / 25.4, 150 / 25.4))
    ax_wind = fig.add_axes([.083, .65, .37, .295])
    ax_grid = fig.add_axes([.60, .65, .295, .295])
    styles = ['--', ':', '-', '-', '-.']
    for scenario, style in zip(REPRESENTATIVES, styles):
        v = bins.loc[bins.scenario.eq(scenario)].sort_values('wind_bin_index')
        ax_wind.plot(v.wind_bin_index, v['median'], marker=marker(scenario), ls=style, color=color(scenario),
                     ms=2.7, lw=1.1 if scenario == 'D-TRN' else .85, label=use(scenario))
        if scenario == 'D-TRN':
            ax_wind.fill_between(v.wind_bin_index, v.q25, v.q75, color=color(scenario), alpha=.16, linewidth=0)
    ax_wind.set_ylim(0, 26)
    ax_wind.set_yticks([0, 5, 10, 15, 20, 25])
    ax_wind.set_xlim(-.25, 8.25)
    tick_labels = ['≤0.05', '0.05–\n0.10', '0.10–\n0.15', '0.15–\n0.20', '0.20–\n0.25',
                   '0.25–\n0.30', '0.30–\n0.35', '0.35–\n0.45', '>0.45']
    ax_wind.set_xticks(range(9), tick_labels)
    ax_wind.set_xlabel('风电年平均出力系数（分档）', labelpad=17)
    ax_wind.set_ylabel('能源系统成本节省中位数（%）')
    guides(ax_wind)
    ax_wind.legend(loc='upper left', borderaxespad=.2, handlelength=2, labelspacing=.3)
    counts = bins.loc[bins.scenario.eq('D-TRN')].sort_values('wind_bin_index').n
    for x, n in enumerate(counts):
        ax_wind.text(x, -5.1, str(n), ha='center', fontsize=6, color='#666666', clip_on=False)
    ax_wind.text(-.55, -5.1, 'n', ha='right', fontsize=6, color='#666666', clip_on=False)
    training = bins.loc[bins.scenario.eq('D-TRN')].sort_values('wind_bin_index')
    for idx, dx, dy in [(0, 6, 7), (8, -5, 7)]:
        v = training.iloc[idx]
        ax_wind.annotate(f'{v["median"]:.1f}%', (v.wind_bin_index, v['median']),
                         xytext=(dx, dy), textcoords='offset points', fontsize=6,
                         ha='left' if idx == 0 else 'right', color=color('D-TRN'))
    panel(fig, 'a', .012, .985, '规划阶段｜经济最优', .083)

    norm = mpl.colors.Normalize(0, .8)
    scatter = ax_grid.scatter(train.grid_share_pct, train.savings_pct, c=train.cf_wind,
                              cmap='viridis', norm=norm, s=3.2, alpha=.6, linewidths=0)
    ax_grid.set(xlim=(0, 100), ylim=(0, 40), xticks=[0, 25, 50, 75, 100], yticks=[0, 10, 20, 30, 40])
    ax_grid.set_xlabel('无柔性市电占设施用电比例（%）')
    ax_grid.set_ylabel('能源系统成本节省（%）')
    ax_grid.text(.97, .96, f'Pearson r = {correlation:.2f}\nn = 4,842', transform=ax_grid.transAxes,
                 ha='right', va='top', fontsize=6.5)
    cax = fig.add_axes([.913, .65, .012, .295])
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=scatter.cmap), cax=cax, ticks=[0, .2, .4, .6, .8])
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor('face')
    bar.set_label('风电年平均出力系数', fontsize=6.5, labelpad=3)
    bar.outline.set_linewidth(.4)
    panel(fig, 'b', .53, .985, 'GPU主导型·大模型训练', .60)

    columns = [(b, s) for b in ['planning', 'fixed'] for s in REPRESENTATIVES]
    beta = exo.pivot(index='variable', columns=['boundary', 'scenario'], values='beta_pp_per_sd').reindex(index=VARIABLES, columns=columns)
    delta = exo.pivot(index='variable', columns=['boundary', 'scenario'], values='delta_r2').reindex(index=VARIABLES, columns=columns)
    ax_heat = fig.add_axes([.205, .18, .69, .29])
    limit = 4.5
    assert beta.notna().all().all() and beta.abs().max().max() <= limit
    image = ax_heat.pcolormesh(np.arange(11) - .5, np.arange(9) - .5, beta.to_numpy(),
                               cmap='RdBu_r', norm=TwoSlopeNorm(0, -limit, limit), shading='flat', rasterized=False)
    ax_heat.set(xlim=(-.5, 9.5), ylim=(7.5, -.5))
    for r in range(8):
        for c in range(10):
            v = beta.iat[r, c]
            text = '0.0' if abs(v) < .05 else f'{v:+.1f}'
            ax_heat.text(c, r, text, ha='center', va='center', fontsize=6.5,
                         weight='bold' if delta.iat[r, c] >= .02 else 'normal',
                         color='white' if abs(v) > .56 * limit else '#202020')
    ax_heat.set_yticks(range(8), LABELS)
    ax_heat.set_xticks(range(10), ['科研\n计算', '云通用\n计算', '科研AI', '大模型\n训练', '推理\n服务'] * 2)
    for t, s in zip(ax_heat.get_xticklabels(), REPRESENTATIVES * 2):
        t.set_color(color(s))
    ax_heat.tick_params(length=0, pad=4)
    ax_heat.axvline(4.5, color='white', lw=2)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.set_xticks(np.arange(-.5, 10, 1), minor=True)
    ax_heat.set_yticks(np.arange(-.5, 8, 1), minor=True)
    ax_heat.grid(which='minor', color='white', lw=.4)
    ax_heat.tick_params(which='minor', length=0)
    for x, text in [(2, '规划阶段'), (7, '已建成')]:
        ax_heat.text(x, -.87, text, ha='center', va='bottom', fontsize=7, clip_on=False)
    fits = exo[['boundary', 'scenario', 'r2_exogenous']].drop_duplicates().set_index(['boundary', 'scenario'])
    for i, key in enumerate(columns):
        ax_heat.text((i + .5) / 10, -.25, f'{fits.loc[key, "r2_exogenous"]:.2f}',
                     transform=ax_heat.transAxes, ha='center', va='top', fontsize=6, clip_on=False)
    ax_heat.text(-.025, -.25, 'R²', transform=ax_heat.transAxes, ha='right', va='top', fontsize=6.5)
    cax = fig.add_axes([.918, .18, .012, .29])
    bar = fig.colorbar(image, cax=cax, ticks=[-4, -2, 0, 2, 4])
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor('face')
    bar.set_label('系数（百分点/自变量标准差）', fontsize=6.5, labelpad=3)
    bar.outline.set_linewidth(.4)
    panel(fig, 'c', .012, .53, '标准化自变量回归｜粗体：移除变量后 ΔR² ≥ 0.02', .205)
    fig.text(.205, .035, 'GPU 用途取 GPU主导型；各面板均为经济最优；色标对应响应变量百分点，响应变量未标准化。',
             fontsize=6, color='#555555')
    export(fig, 'fig2', {'n_per_fit': 4842, 'correlation': float(correlation),
                        'regression': '8 个标准化自变量的 OLS；响应变量未标准化；未校正空间自相关',
                        'bold': '移除自变量后 R² 降幅 ≥ 0.02；不表示显著性检验',
                        'shading': 'GPU主导型训练在各风电档内的网格第25–75百分位区间',
                        'input': 'PR #63 面板源数据；科学内容保持一致'})


if __name__ == '__main__':
    main()
