import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from paper_figure_zh_common import (CAPACITY, RESALE, ROOT, SCENARIOS, capacity_main,
                                    color, csv, export, guides, marker, panel, rows,
                                    setup, shade, use)


def main():
    setup()
    main_table = capacity_main()
    part1 = pd.read_csv(ROOT / 'results/global-flex-econ-20261103/part1/part1-summary.csv')
    a = main_table.merge(part1[['scenario', 'fleet_pct_planning', 'fleet_pct_fixed']], on='scenario', validate='one_to_one')
    a['equipment_pct'] = a.equipment_annual_saving_usd / a.annual_fleet_equipment_cost_usd * 100
    a['energy_pct'] = a.energy_annual_saving_vs_A_usd_median / a.annual_fleet_equipment_cost_usd * 100
    np.testing.assert_allclose(a.equipment_pct + a.energy_pct, a.net_total_pct_of_fleet, atol=1e-10)
    csv(a, 'fig4a')
    price = pd.read_csv(CAPACITY / 'equipment-price-sensitivity.csv')
    price = price.loc[price.target.eq('economic') & price.scenario.isin(['D-TRN', 'D-INF'])].copy()
    price = price.merge(part1[['scenario', 'fleet_pct_planning', 'fleet_pct_fixed']], on='scenario', validate='many_to_one')
    price['planning_flex_pct_at_price'] = price.fleet_pct_planning / price.price_multiplier
    price['fixed_flex_pct_at_price'] = price.fleet_pct_fixed / price.price_multiplier
    csv(price, 'fig4b')
    resale = pd.read_csv(RESALE / 'resale-price-share-city.csv')
    csv(resale, 'fig4c_all_scenarios')
    selected = ['D-TRN', 'D-SAI', 'D-INF']
    resale = resale.loc[resale.scenario.isin(selected)].copy()
    csv(resale, 'fig4c_cityvalues')
    curves = resale.groupby(['scenario', 'price_usd_gpu_h', 'sold_fraction']).agg(
        net_annual_usd=('D_net_annual_usd', 'median'), n_cities=('city', 'nunique'),
        cd_extra_annual_usd=('CD_extra_annual_usd', 'median')).reset_index()
    csv(curves, 'fig4c')
    breaks = []
    for scenario in selected:
        v = curves.loc[curves.scenario.eq(scenario) & curves.sold_fraction.eq(.5)].sort_values('price_usd_gpu_h')
        slope = np.diff(v.net_annual_usd) / np.diff(v.price_usd_gpu_h)
        np.testing.assert_allclose(slope, slope[0], rtol=1e-12)
        intercept = v.net_annual_usd.iloc[0] - slope[0] * v.price_usd_gpu_h.iloc[0]
        equipment = main_table.set_index('scenario').loc[scenario, 'equipment_annual_saving_usd']
        breaks.append({'scenario': scenario, 'equipment_annual_saving_usd': equipment,
                       'break_even_usd_gpu_h_s50': (equipment - intercept) / slope[0]})
    breaks = pd.DataFrame(breaks)
    csv(breaks, 'fig4c_break_even')
    references = pd.DataFrame([
        {'price_usd_gpu_h': 2.2, 'label': '2026-07 第三方历史零售价',
         'url': 'https://www.usagepricing.com/blueprint/activity/deepinfra-2026-07-14-gpu-hour-token-hike'},
        {'price_usd_gpu_h': 2.9, 'label': '2026-11-08 官方当前零售价', 'url': 'https://deepinfra.com/pricing'},
    ])
    csv(references, 'fig4c_price_references')

    fig = plt.figure(figsize=(180 / 25.4, 188 / 25.4))
    ax_joint = fig.add_axes([.255, .59, .34, .335])
    y = rows(ax_joint, labels=True)
    for yi, (_, r) in zip(y, a.iterrows()):
        ax_joint.barh(yi, r.equipment_pct, height=.62, color=shade(r.scenario), zorder=2)
        ax_joint.barh(yi, r.energy_pct, left=r.equipment_pct, height=.62,
                      facecolor='white', edgecolor=shade(r.scenario), hatch='////', lw=.5, zorder=2)
        ax_joint.text(r.net_total_pct_of_fleet + .45, yi, f'{r.net_total_pct_of_fleet:.1f}', va='center', fontsize=6)
    ax_joint.scatter(a.fleet_pct_planning, y, marker='o', s=13, color='#222222', edgecolors='white', lw=.3, zorder=4)
    ax_joint.scatter(a.fleet_pct_fixed, y, marker='o', s=13, facecolors='white', edgecolors='#222222', lw=.7, zorder=4)
    ax_joint.set(xlim=(0, 37), xticks=[0, 10, 20, 30])
    ax_joint.set_xlabel('年净节省占原机群年化成本（%）')
    guides(ax_joint, 'x')
    handles = [Patch(facecolor='#777777', label='设备节省'), Patch(facecolor='white', edgecolor='#555555', hatch='////', label='能源节省'),
               Line2D([], [], marker='o', ms=3, color='none', markerfacecolor='#222222', label='仅调度·规划'),
               Line2D([], [], marker='o', ms=3, color='none', markerfacecolor='white', markeredgecolor='#222222', label='仅调度·已建成')]
    ax_joint.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.17), ncol=2,
                     handlelength=1.1, handletextpad=.5, columnspacing=1)
    energy_values = a.energy_annual_saving_vs_A_usd_median / 1e6
    fig.text(.255, .535, f'能源部分：{energy_values.min():.2f}–{energy_values.max():.2f} 百万美元/年',
             fontsize=6, color='#555555')
    panel(fig, 'a', .012, .975, '原时序规划 → 机群与能源联合规划', .255)
    fig.text(.255, .943, '仅调度约 0.1–1%；联合规划 9.45–33.0%', fontsize=6.5, color='#333333')

    ax_price_net = fig.add_axes([.765, .765, .205, .155])
    ax_price_ratio = fig.add_axes([.765, .59, .205, .13])
    for scenario in ['D-TRN', 'D-INF']:
        v = price.loc[price.scenario.eq(scenario)].sort_values('price_multiplier')
        ax_price_net.plot(v.price_multiplier, v.net_annual_saving_vs_A_usd / 1e6,
                          color=color(scenario), marker=marker(scenario), ms=3, label=use(scenario))
        ax_price_ratio.plot(v.price_multiplier, v.planning_flex_pct_at_price,
                            color=color(scenario), marker=marker(scenario), ms=3)
        ax_price_ratio.plot(v.price_multiplier, v.fixed_flex_pct_at_price,
                            color=color(scenario), ls='--', lw=.65)
    ax_price_net.set(xlim=(.4, 3.1), ylim=(0, 100), yticks=[0, 50, 100])
    ax_price_ratio.set(xlim=(.4, 3.1), ylim=(0, 1.85), yticks=[0, .5, 1, 1.5])
    for ax in [ax_price_net, ax_price_ratio]:
        ax.set_xticks([.5, 1, 2, 3], ['0.5', '1', '2', '3'])
        guides(ax)
    ax_price_net.set_xticklabels([])
    ax_price_net.set_ylabel('联合规划净节省\n（百万美元/年）', labelpad=2)
    ax_price_ratio.set_ylabel('仅调度节省占机群成本\n（%）', labelpad=2)
    ax_price_ratio.set_xlabel('设备价格倍数')
    ax_price_net.legend(loc='upper left', fontsize=6, handlelength=1.3, borderaxespad=.2)
    price_labels = price.loc[price.scenario.eq('D-TRN') & price.price_multiplier.isin([1, 2])]
    for _, r in price_labels.iterrows():
        value = r.net_annual_saving_vs_A_usd / 1e6
        ax_price_net.annotate(f'{value:.2f}', (r.price_multiplier, value), xytext=(3, -10),
                              textcoords='offset points', fontsize=6, color=color('D-TRN'))
    ax_price_ratio.text(.97, .98, '实线：规划\n虚线：已建成', transform=ax_price_ratio.transAxes,
                        ha='right', va='top', fontsize=6)
    panel(fig, 'b', .665, .975, '设备价格敏感性', .765)

    ax_resale = fig.add_axes([.105, .13, .865, .285])
    for scenario in selected:
        v = curves.loc[curves.scenario.eq(scenario)].pivot(index='price_usd_gpu_h', columns='sold_fraction', values='net_annual_usd') / 1e6
        ax_resale.fill_between(v.index, v[.25], v[.75], color=color(scenario), alpha=.12, linewidth=0)
        ax_resale.plot(v.index, v[.5], color=color(scenario), marker=marker(scenario), ms=3,
                       lw=1.1, label=use(scenario))
        b = breaks.loc[breaks.scenario.eq(scenario)].iloc[0]
        yy = b.equipment_annual_saving_usd / 1e6
        ax_resale.axhline(yy, color=color(scenario), ls=(0, (4, 2)), lw=.8)
        ax_resale.scatter(b.break_even_usd_gpu_h_s50, yy, marker=marker(scenario), s=24,
                          facecolors='white', edgecolors=color(scenario), lw=.9, zorder=5)
        offsets = {'D-TRN': (-6, 7), 'D-SAI': (6, 6), 'D-INF': (6, -12)}
        ax_resale.annotate(f'{use(scenario)}持平 {b.break_even_usd_gpu_h_s50:.2f}',
                           (b.break_even_usd_gpu_h_s50, yy), xytext=offsets[scenario], textcoords='offset points',
                           ha='right' if scenario == 'D-TRN' else 'left', fontsize=6, color=color(scenario))
    for x in [2.2, 2.9]:
        ax_resale.axvline(x, color='#777777', ls=(0, (2, 2)), lw=.8, zorder=1)
    ax_resale.text(2.2, 77, '2.2｜2026-07\n第三方历史记录', ha='right', va='top', fontsize=6, color='#555555')
    ax_resale.text(2.9, 77, '2.9｜官方当前价\n2026-11-08', ha='right', va='top', fontsize=6, color='#555555')
    ax_resale.set(xlim=(.45, 3.02), ylim=(0, 80), xticks=[.5, 1, 1.5, 2.2, 2.9], yticks=[0, 20, 40, 60, 80])
    ax_resale.set_xlabel('出售价格（美元/GPU·小时）')
    ax_resale.set_ylabel('出售空闲算力年净收入（百万美元/年）')
    guides(ax_resale)
    ax_resale.legend(loc='upper left', ncol=3, handlelength=1.7, columnspacing=1.4, borderaxespad=.5)
    panel(fig, 'c', .012, .48, 'GPU主导型｜可售比例 50% 为实线，25–75% 为阴影；水平虚线为设备年化节省', .105)
    fig.text(.105, .453, '收入为八城市中位数；持平价格按 50% 可售比例计算；竖线为 DeepInfra H100 80GB 公开零售价。',
             fontsize=6, color='#555555')
    fig.text(.105, .066, 'a、b：经济最优目标，八城市可行方案中位数；仅调度比例来自全球网格中位数；价格敏感性保持供能方案。', fontsize=6, color='#555555')
    fig.text(.105, .043, 'c：保留原机群与供能，扣除新增市电及碳费用；可售比例为情景参数，未模拟市场需求、逐卡碎片和加载开销。', fontsize=6, color='#555555')
    fig.text(.105, .020, '三种用法分别比较，收益不可直接相加；金额按原生观测窗口等效年化；科研计算金额采用 13,620 台可行方案。', fontsize=6, color='#555555')
    export(fig, 'fig4', {'part2_scope': '八城市经济最优目标下的容量可行策略；整体作业能源费用最优性未证明',
                        'part1_scope': '4842个全球有效网格中位数；未与八城市作配对因果比较',
                        'price_scope': '仅调整设备费用与机群分母；未重新优化容量和供能',
                        'resale_scope': '保留A机群与A供能；扣除新增能源费用；三种用法收益不能直接相加',
                        'break_even': breaks.to_dict('records'),
                        'price_references': references.to_dict('records'),
                        'input': 'PR #60、#65、#67；未重新运行科研求解'})


if __name__ == '__main__':
    main()
