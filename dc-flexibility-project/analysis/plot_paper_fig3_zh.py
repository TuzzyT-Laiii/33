import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from paper_figure_zh_common import (CAPACITY, REPRESENTATIVES, SCENARIOS, capacity_main,
                                    color, csv, export, guides, marker, panel, rows,
                                    setup, shade, use)


def main():
    setup()
    main_table = capacity_main()
    main_table['reduction_upper_pct'] = 100 * (1 - main_table.lower_servers / main_table.A_servers)
    csv(main_table, 'fig3a')
    sensitivity = pd.read_csv(CAPACITY / 'capacity-sensitivity.csv')
    sensitivity = sensitivity.loc[sensitivity.scenario.isin(REPRESENTATIVES)].copy()
    sensitivity['reduction_lower_pct'] = 100 * (1 - sensitivity.servers / sensitivity.A_servers)
    sensitivity['reduction_upper_pct'] = 100 * (1 - sensitivity.lower_servers / sensitivity.A_servers)
    csv(sensitivity, 'fig3b')
    waiting = pd.read_csv(CAPACITY / 'waiting-and-utilization.csv')
    waiting = waiting.loc[waiting.target.eq('economic') & waiting.stage.eq('C') & waiting.scenario.isin(REPRESENTATIVES)]
    fields = ['flex_energy_mean_delay_h', 'mean_bottleneck_fraction']
    points = waiting.groupby('scenario')[fields].median().reindex(REPRESENTATIVES).reset_index()
    points['city_count'] = waiting.groupby('scenario').city.nunique().reindex(REPRESENTATIVES).to_numpy()
    points = points.merge(main_table[['scenario', 'equipment_annual_saving_usd']], on='scenario', validate='one_to_one')
    points['occupancy_pct'] = 100 * points.mean_bottleneck_fraction
    csv(points, 'fig3c')
    csv(waiting, 'fig3c_cityvalues')
    reserve = pd.read_csv(CAPACITY / 'equipment-reserve-sensitivity.csv')
    selected = ['D-TRN', 'D-SAI', 'D-INF', 'A-HPC']
    base = main_table[['scenario', 'servers', 'equipment_annual_saving_usd']].rename(
        columns={'servers': 'minimum_servers', 'equipment_annual_saving_usd': 'annual_equipment_saving_usd'})
    base['reserve_fraction'] = 0
    base['provisioned_servers'] = base.minimum_servers
    reserve = pd.concat([reserve, base], ignore_index=True)
    reserve = reserve.loc[reserve.scenario.isin(selected)].sort_values(['scenario', 'reserve_fraction'])
    csv(reserve, 'fig3d')

    fig = plt.figure(figsize=(180 / 25.4, 174 / 25.4))
    ax_bar = fig.add_axes([.255, .51, .25, .415])
    y = rows(ax_bar, labels=True)
    for yi, (_, r) in zip(y, main_table.iterrows()):
        ax_bar.barh(yi, r.reduction_pct, color=shade(r.scenario), height=.61, zorder=2)
        if not r.capacity_proof_complete:
            ax_bar.errorbar(r.reduction_pct, yi, xerr=[[0], [r.reduction_upper_pct - r.reduction_pct]],
                            color='#333333', capsize=2, lw=.8, fmt='none', zorder=4)
        count = int(r.servers_avoided if r.scenario.startswith('A') else r.gpus_avoided)
        unit = '台' if r.scenario.startswith('A') else 'GPU'
        rate = f'{r.reduction_pct:.1f}–{r.reduction_upper_pct:.1f}%' if not r.capacity_proof_complete else f'{r.reduction_pct:.1f}%'
        ax_bar.text(max(r.reduction_upper_pct, r.reduction_pct) + .7, yi,
                     f'{rate}｜{count:,} {unit}', va='center', fontsize=6, color='#333333')
    ax_bar.set(xlim=(0, 58), xticks=[0, 10, 20, 30, 40, 50])
    ax_bar.set_xlabel('服务器减配率（%）')
    guides(ax_bar, 'x')
    panel(fig, 'a', .012, .973, '12/48 小时期限；条旁为减配率与少配数量', .255)
    fig.text(.255, .45, '科研计算：13,195–13,620 台；金额及数量取 13,620 台可行方案。', fontsize=6, color='#555555')

    ax_deadline = fig.add_axes([.66, .585, .315, .34])
    for scenario in REPRESENTATIVES:
        v = sensitivity.loc[sensitivity.scenario.eq(scenario)].sort_values('whole_h')
        xx = np.arange(3)
        if scenario == 'A-HPC':
            mid = (v.reduction_upper_pct.to_numpy() + v.reduction_lower_pct.to_numpy()) / 2
            err = (v.reduction_upper_pct.to_numpy() - v.reduction_lower_pct.to_numpy()) / 2
            ax_deadline.errorbar(xx, mid, yerr=err, color=color(scenario), marker='_',
                                 capsize=3, ls='none', label='科研计算（求解区间）', lw=1)
        else:
            ax_deadline.plot(xx, v.reduction_lower_pct, marker=marker(scenario), color=color(scenario),
                             ls='--' if scenario in ['A-CLD', 'D-INF'] else '-', ms=3, label=use(scenario))
    ax_deadline.set(xlim=(-.15, 2.15), ylim=(0, 36), yticks=[0, 10, 20, 30])
    ax_deadline.set_xticks(range(3), ['12/48', '24/96', '48/168'])
    ax_deadline.set_xlabel('整体作业/可拆分工作量的推迟期限（小时）')
    ax_deadline.set_ylabel('服务器减配率（%）')
    guides(ax_deadline)
    handles, texts = ax_deadline.get_legend_handles_labels()
    order = [texts.index(use(s)) if s != 'A-HPC' else texts.index('科研计算（求解区间）') for s in REPRESENTATIVES]
    ax_deadline.legend([handles[i] for i in order], [texts[i] for i in order],
                       loc='lower center', bbox_to_anchor=(.5, -.46), ncol=2,
                       handlelength=1.5, columnspacing=.8, labelspacing=.4)
    panel(fig, 'b', .59, .973, '延长期限的收益', .66)
    ax_deadline.text(2.07, 27.56, '27.6%', fontsize=6, va='bottom', ha='right', color=color('D-TRN'))
    ax_deadline.text(2.07, 22.37, '22.4%', fontsize=6, va='bottom', ha='right', color=color('D-SAI'))

    ax_wait = fig.add_axes([.105, .13, .36, .225])
    offsets = {'A-HPC': (5, -10), 'D-SAI': (5, -9), 'D-TRN': (-3, 7), 'D-INF': (5, 5)}
    for _, r in points.dropna(subset=['flex_energy_mean_delay_h']).iterrows():
        size = r.equipment_annual_saving_usd / 1e6 * 2.4
        ax_wait.scatter(r.flex_energy_mean_delay_h, r.occupancy_pct, s=size, marker=marker(r.scenario),
                        color=color(r.scenario), edgecolors='white', linewidths=.5, zorder=3)
        text = f'{use(r.scenario)}\n{r.flex_energy_mean_delay_h:.1f} 小时；{r.occupancy_pct:.1f}%' if r.scenario == 'D-TRN' else use(r.scenario)
        ax_wait.annotate(text, (r.flex_energy_mean_delay_h, r.occupancy_pct),
                         xytext=offsets[r.scenario], textcoords='offset points', fontsize=6,
                         ha='right' if r.scenario == 'D-TRN' else 'left', color=color(r.scenario))
    ax_wait.set(xlim=(0, 22), ylim=(65, 100), yticks=[70, 80, 90, 100])
    ax_wait.set_xlabel('可推迟电量加权平均推迟（小时）')
    ax_wait.set_ylabel('缩减后机群资源平均占用率（%）')
    guides(ax_wait)
    size_handles = [Line2D([], [], marker='o', color='none', markerfacecolor='#888888',
                           markeredgewidth=0, markersize=np.sqrt(v * 2.4), label=str(v)) for v in [10, 20, 30]]
    ax_wait.legend(handles=size_handles, title='设备年化节省（百万美元）', title_fontsize=6,
                   loc='lower left', bbox_to_anchor=(-.025, 1.02), ncol=3,
                   handlelength=1, columnspacing=.9, borderaxespad=0)
    panel(fig, 'c', .012, .42, '减配后的等待与占用率', .105)
    cloud = points.loc[points.scenario.eq('A-CLD')].iloc[0]
    fig.text(.105, .060, f'云通用计算：占用率 {cloud.occupancy_pct:.1f}%；平均推迟未记录。', fontsize=6, color='#555555')

    ax_reserve = fig.add_axes([.645, .13, .325, .225])
    for scenario in selected:
        v = reserve.loc[reserve.scenario.eq(scenario)].sort_values('reserve_fraction')
        ax_reserve.plot(v.reserve_fraction * 100, v.annual_equipment_saving_usd / 1e6,
                        marker=marker(scenario), color=color(scenario), ms=3, label=use(scenario))
        if scenario == 'D-TRN':
            for x, val in zip(v.reserve_fraction * 100, v.annual_equipment_saving_usd / 1e6):
                ax_reserve.annotate(f'{val:.2f}', (x, val), xytext=(0, 5), textcoords='offset points',
                                    ha='center', fontsize=6, color=color(scenario))
    ax_reserve.set(xlim=(-.9, 10.7), ylim=(0, 35), xticks=[0, 5, 10], yticks=[0, 10, 20, 30])
    ax_reserve.set_xlabel('在最少台数上增加的冗余（%）')
    ax_reserve.set_ylabel('设备年化节省（百万美元/年）')
    guides(ax_reserve)
    ax_reserve.legend(loc='lower left', bbox_to_anchor=(-.015, 1.015), ncol=2,
                      handlelength=1.4, columnspacing=.8, borderaxespad=0)
    panel(fig, 'd', .555, .42, '保留设备冗余后的节省', .645)
    fig.text(.105, .018, 'GPU 用途在 b–d 取 GPU主导型；c 为容量可行调度，按八城市中位数汇总；d 仅改变设备配置费用。',
             fontsize=6, color='#555555')
    export(fig, 'fig3', {'capacity_scope': '主期限12/48小时；11情境；科研计算未证明唯一最优台数',
                        'hpc_interval_servers': [13195, 13620], 'hpc_money_servers': 13620,
                        'delay': '可推迟电量加权，包含零推迟工作量；非逐作业平均等待',
                        'occupancy': 'mean_bottleneck_fraction；CPU/GPU瓶颈资源平均占用比例',
                        'reserve': 'ceil(最少台数×(1+冗余比例))；能源费用未重新优化',
                        'input': 'PR #65；未重新运行科研求解'})


if __name__ == '__main__':
    main()
