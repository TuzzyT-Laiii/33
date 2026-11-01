import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from native_load_inputs_v3 import RUN
from plot_typical_load_curves_v3 import save, setup


def panel(output, names, profiles, weekly=False):
    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharey=True, layout='constrained')
    axis = 'hour_of_source_week' if weekly else 'hour_of_source_day'
    for index, (ax, row) in enumerate(zip(axes.flat, names.to_dict('records'))):
        scenario = row['scenario_id']
        ax.set_title(f'({chr(97 + index)}) {row["scenario_zh"]}', fontsize=9)
        if weekly and row['use'] == 'CLD':
            ax.text(.5, .5, '仅 5 天观测，不估计周内形状', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            continue
        source = pd.read_parquet(output / 'source-hourly' / (scenario + '.parquet'))
        grouped = source.groupby(axis).it_mw.mean()
        ax.plot(grouped.index, grouped, color='#2878B5', lw=1.2, label='原生 IT')
        cases = profiles.loc[profiles.scenario_id.eq(scenario)]
        for column, color, label in [('facility_mw', '#A05799', '设施：放置中位数'), ('cooling_mw', '#268878', '制冷：放置中位数')]:
            grouped = cases.groupby(axis)[column].agg(['median', 'min', 'max'])
            ax.fill_between(grouped.index, grouped['min'], grouped['max'], color=color, alpha=.17, linewidth=0)
            ax.plot(grouped.index, grouped['median'], color=color, lw=1.1, label=label)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.15)
        ax.set_xlabel(('源星期×小时；无日期为相对七日' if weekly else '源小时（0–23）'), fontsize=8)
        if index % 3 == 0:
            ax.set_ylabel('平均功率（MW）')
    axes.flat[-1].axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc='center', fontsize=10)
    axes.flat[-1].text(.5, .16, '阴影：12 次放置的最小–最大\n有日期来源只有一次对应\n年末截断使放置窗口长度不同', ha='center', transform=axes.flat[-1].transAxes, fontsize=9)
    save(fig, output, 'figure2-native-week' if weekly else 'figure1-native-day')


def plot(project):
    setup()
    output = project / 'results' / RUN
    summary = pd.read_csv(output / 'native-summary.csv')
    names = summary.loc[summary.reference_it_mw.eq(10)]
    day = pd.read_csv(output / 'typical-day-placements.csv')
    week = pd.read_csv(output / 'typical-week-placements.csv')
    panel(output, names, day)
    panel(output, names, week, weekly=True)
    series = pd.read_parquet(output / 'native-timeseries-10MW.parquet')
    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharey=True, layout='constrained')
    for index, (ax, row) in enumerate(zip(axes.flat, names.to_dict('records'))):
        source = pd.read_parquet(output / 'source-hourly' / (row['scenario_id'] + '.parquet'))
        data = series.loc[series.scenario_id.eq(row['scenario_id'])].sort_values('source_hour_index')
        x = data.source_hour_index / 24
        ax.fill_between(x, data.facility_mw_min, data.facility_mw_max, color='#A05799', alpha=.2, linewidth=0)
        ax.plot(x, data.facility_mw_median, color='#A05799', lw=.65, label='设施：可用放置中位数')
        ax.plot(source.source_hour_index / 24, source.it_mw, color='#2878B5', lw=.65, label='原生 IT')
        ax.set_title(f'({chr(97 + index)}) {row["scenario_zh"]}', fontsize=9)
        ax.set_xlabel(f'源窗口经过天数（共 {int(row["native_days"])} 天）', fontsize=8)
        ax.set_xlim(0, row['native_days'])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.15)
        if index % 3 == 0:
            ax.set_ylabel('小时功率（MW）')
        if not row['has_real_dates']:
            ax.text(.98, .95, f'逐小时可用放置数：{data.available_placements.min()}–{data.available_placements.max()}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=7)
    axes.flat[-1].axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc='center', fontsize=10)
    axes.flat[-1].text(.5, .15, '阴影：可用放置最小–最大\n逐小时聚合曲线不对应单次设施运行', ha='center', transform=axes.flat[-1].transAxes, fontsize=9)
    save(fig, output, 'figure3-native-timeseries')
    cases = pd.read_csv(output / 'placement-summary.csv')
    components = cases.loc[cases.reference_it_mw.eq(10)].groupby('scenario_id').mean(numeric_only=True).reindex(names.scenario_id)
    fig, ax = plt.subplots(figsize=(12, 6.5), layout='constrained')
    positions, left = np.arange(len(names)), np.zeros(len(names))
    for column, label, color in [('base_it', '基础 IT', '#A4ACAF'), ('cpu_it', 'CPU', '#D1A238'),
                                 ('gpu_it', 'GPU', '#2878B5'), ('cooling', '制冷', '#268878'), ('other', '配电及其他', '#9B6BB5')]:
        values = components['mean_' + column + '_mw'].to_numpy()
        ax.barh(positions, values, left=left, label=label, color=color, height=.65)
        left += values
    ax.set_yticks(positions, names.scenario_zh)
    ax.invert_yaxis()
    ax.set_xlabel('窗口平均功率分项（MW）；无日期来源为 12 次放置窗口均值的算术平均')
    ax.set_xlim(left=0)
    ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(.5, 1.12))
    save(fig, output, 'figure4-native-components')
    comparison = pd.read_csv(output / 'mean-power-comparison.csv')
    selected = comparison.loc[comparison.reference_it_mw.eq(10)].set_index('scenario_id').reindex(names.scenario_id)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True, layout='constrained')
    for ax, component, title in zip(axes, ['it', 'facility'], ['IT 平均功率变化（%）', '设施平均功率变化（%）']):
        ax.axvline(0, color='#888888', lw=.8)
        for label, offset, color, text in [('v2_annual', -.12, '#8D743A', '相对 v2 全年均值'), ('previous_v3_annual', .12, '#2878B5', '相对之前 v3 全年均值')]:
            values = selected[f'{label}_{component}_change_pct'].to_numpy()
            ax.scatter(values, np.arange(len(values)) + offset, color=color, s=30, label=text)
        ax.set_xlabel(title)
        ax.grid(axis='y', alpha=.15)
        ax.margins(x=.2)
    axes[0].set_yticks(np.arange(len(names)), names.scenario_zh)
    axes[0].invert_yaxis()
    axes[1].legend(loc='best', fontsize=8)
    fig.supxlabel('不同窗口与气象对应下的平均功率描述性对比；A-CLD 无 v2 数值', fontsize=9)
    save(fig, output, 'figure5-native-mean-comparison')
    sensitivity = pd.read_csv(output / 'sensitivity-summary.csv')
    selected = sensitivity.loc[sensitivity.cooling_case.eq('central')]
    labels = {'central': '六月均值＋比例空载', 'ratio_p10': '设备小时 P10', 'ratio_p90': '设备小时 P90',
              'legacy_idle': '目标原空载', 'idle_half': '比例空载 ×0.5', 'idle_double': '比例空载 ×2'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
    sai_names = names.loc[names.use.eq('SAI')]
    for ax, column, xlabel in zip(axes, ['mean_it_mw', 'mean_facility_mw'], ['原生窗口平均 IT（MW）', '原生窗口平均设施（MW）']):
        for offset, (method, label) in enumerate(labels.items()):
            rows = selected.loc[selected.power_sensitivity.eq(method)].set_index('scenario_id').reindex(sai_names.scenario_id)
            ax.scatter(rows[column], np.arange(3) + (offset - 2.5) * .08, label=label, s=35)
        ax.set_yticks(np.arange(3), sai_names.scenario_zh)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.grid(axis='y', alpha=.2)
    axes[1].legend(fontsize=8, loc='best')
    fig.supxlabel('237 天原生分配；功率分位数为描述性端点，空载倍数为工程敏感性', fontsize=9)
    save(fig, output, 'figure6-native-sai-sensitivity')
    print('6 组原生窗口 PNG/PDF 已生成', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    plot(parser.parse_args().project.resolve())
