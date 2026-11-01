import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from load_curve_scenarios_v3 import RUN


def setup():
    font = 'WenQuanYi Zen Hei'
    font_manager.findfont(font, fallback_to_default=False)
    plt.rcParams.update({'font.family': font, 'font.size': 9, 'axes.unicode_minus': False,
        'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'savefig.dpi': 300})


def save(fig, output, name):
    fig.savefig(output / (name + '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(output / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def panel_curves(output, data, x, name, xlabel, envelope=False):
    ids = data.scenario_id.drop_duplicates().tolist()
    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharey=True, layout='constrained')
    for index, (ax, scenario) in enumerate(zip(axes.flat, ids)):
        rows = data.loc[data.scenario_id.eq(scenario)]
        axis = rows[x].to_numpy()
        if envelope:
            axis = pd.to_datetime(axis)
            ax.fill_between(axis, rows.facility_mw_min.to_numpy(), rows.facility_mw_max.to_numpy(), color='#2878B5', alpha=.20)
            ax.plot(axis, rows.facility_mw_mean, color='#2878B5', lw=1, label='设施日均与日内范围')
            ax.plot(axis, rows.it_mw_mean, color='#D66B27', lw=.9, label='IT 日均')
            ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m月'))
        else:
            for column, label, color in [('it_mw', 'IT', '#D66B27'), ('cooling_mw', '制冷', '#268878'), ('facility_mw', '设施', '#2878B5')]:
                ax.plot(axis, rows[column], color=color, lw=1.2, label=label)
            ax.set_xlim(0, 23 if x == 'hour_utc' else 167)
            ax.set_xticks([0, 6, 12, 18, 23] if x == 'hour_utc' else [0, 24, 48, 72, 96, 120, 144, 167])
        ax.set_ylim(bottom=0)
        ax.set_title(f'({chr(97 + index)}) {rows.scenario_zh.iloc[0]}', fontsize=9)
        ax.grid(axis='y', alpha=.15)
        ax.set_ylabel('功率（MW）')
        ax.set_xlabel(xlabel)
    axes.flat[-1].axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc='center', fontsize=11)
    save(fig, output, name)


def plot(project):
    output = project / 'results' / RUN
    setup()
    day = pd.read_csv(output / 'typical-day.csv')
    week = pd.read_csv(output / 'typical-week.csv')
    envelope = pd.read_csv(output / 'annual-envelope.csv')
    panel_curves(output, day.loc[day.reference_it_mw.eq(10)], 'hour_utc', 'figure1-typical-day', '目标 UTC 小时')
    panel_curves(output, week.loc[week.reference_it_mw.eq(10)], 'hour_of_target_week_utc', 'figure2-typical-week', '目标周内小时（周一 00 时起）')
    panel_curves(output, envelope.loc[envelope.reference_it_mw.eq(10)], 'date_utc', 'figure3-annual-envelope', '2025 年目标 UTC 日期', envelope=True)
    summary = pd.read_csv(output / 'annual-summary.csv')
    selected = summary.loc[summary.reference_it_mw.eq(10)]
    fig, ax = plt.subplots(figsize=(12, 6.5), layout='constrained')
    positions, left = np.arange(len(selected)), np.zeros(len(selected))
    for column, label, color in [('base_it', '基础 IT', '#A4ACAF'), ('cpu_it', 'CPU', '#D1A238'),
                                 ('gpu_it', 'GPU', '#2878B5'), ('cooling', '制冷', '#268878'), ('other', '配电及其他', '#9B6BB5')]:
        values = selected['annual_' + column + '_mwh'].to_numpy() / 1000
        ax.barh(positions, values, left=left, label=label, color=color, height=.65)
        left += values
    ax.set_yticks(positions, selected.scenario_zh)
    ax.invert_yaxis()
    ax.set_xlabel('年电量（GWh）')
    ax.set_xlim(left=0)
    ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(.5, 1.12))
    save(fig, output, 'figure4-power-components')
    comparison = pd.read_csv(output / 'v2-v3-comparison.csv')
    selected = comparison.loc[comparison.reference_it_mw.eq(10) & comparison.v2_available]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5), sharey=True, layout='constrained')
    for ax, column, label in zip(axes, ['mean_it_change_pct', 'mean_facility_change_pct', 'peak_facility_change_pct'],
                                ['年均 IT 功率变化（%）', '年均设施功率变化（%）', '峰值设施功率变化（%）']):
        values = selected[column].to_numpy()
        ax.axvline(0, color='#888888', lw=.8)
        ax.scatter(values, np.arange(len(values)), color='#2878B5', s=40, zorder=3)
        for index, value in enumerate(values):
            ax.annotate(f'{value:+.1f}', (value, index), xytext=(6, 0), textcoords='offset points', va='center', fontsize=8)
        ax.set_xlabel(label)
        ax.grid(axis='y', alpha=.15)
        ax.margins(x=.25)
    axes[0].set_yticks(np.arange(len(selected)), selected.scenario_zh)
    axes[0].invert_yaxis()
    save(fig, output, 'figure5-v2-v3-comparison')
    sensitivity = pd.read_csv(output / 'sensitivity-summary.csv')
    selected = sensitivity.loc[sensitivity.use.eq('SAI') & sensitivity.cooling_case.eq('central')]
    if len(selected):
        labels = {'central': '六月均值＋比例空载', 'ratio_p10': '六月设备小时 P10', 'ratio_p90': '六月设备小时 P90',
                  'legacy_idle': '六月均值＋原空载假设', 'idle_half': '六月均值＋空载×0.5', 'idle_double': '六月均值＋空载×2'}
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True, layout='constrained')
        for ax, key in zip(axes, ['B', 'C', 'D']):
            rows = selected.loc[selected.server_type.eq(key)]
            ax.scatter(rows.mean_it_mw, np.arange(len(rows)), color='#D66B27', label='IT', s=35)
            ax.scatter(rows.mean_facility_mw, np.arange(len(rows)), color='#2878B5', label='设施', s=35)
            ax.set_yticks(np.arange(len(rows)), [labels[value] for value in rows.power_sensitivity])
            ax.set_xlabel('年平均功率（MW）')
            ax.set_title(rows.scenario_zh.iloc[0])
            ax.grid(axis='y', alpha=.15)
            ax.set_xlim(left=0)
        axes[0].invert_yaxis()
        axes[-1].legend()
        save(fig, output, 'figure6-sai-transfer-sensitivity')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    plot(parser.parse_args().project.resolve())
