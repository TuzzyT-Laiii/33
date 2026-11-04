import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_trn_power_model import BLUE, ORANGE, GREEN, GRAY, setup, save


def finish(fig, output, name, font_path, audit):
    for ax in fig.axes:
        low, high = ax.get_ylim()
        ax.set_yticks([value for value in ax.get_yticks() if low <= value <= high])
        low, high = ax.get_xlim()
        ax.set_xticks([value for value in ax.get_xticks() if low <= value <= high])
        ax.grid(axis='y', alpha=.15)
    save(fig, output, name, font_path, audit)


def plot(args):
    output = args.project / 'results/trn-power-model-20261101/diurnal-acme'
    font_path, audit = setup(args.font), []
    hourly = pd.read_csv(output / 'kalos-hourly-screened.csv')
    part = hourly.loc[hourly.overlap_samples.gt(0)].reset_index(drop=True)
    x = np.arange(len(part))
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), layout='constrained', sharex=True)
    axes[0].plot(x, part.allocated_gpu_count_at_samples, color=BLUE, label='全群已分配GPU（含右删失）')
    axes[0].plot(x, part.mean_reporting_gpu_count, color=ORANGE, label='功率报送设备')
    axes[0].axhline(2416, color=GRAY, ls=':', label='论文物理库存2416张')
    axes[0].set(ylabel='设备数量（张）', ylim=(0, 2600), title='Kalos作业与监测重叠：设备总体及反推诊断')
    axes[0].legend(ncol=3, fontsize=9)
    axes[1].plot(x, 100 * part.device_coverage_physical, color=BLUE)
    axes[1].axhline(80, color=GRAY, ls='--', label='反推覆盖筛选阈值80%')
    axes[1].set(ylabel='物理库存报送覆盖（%）', ylim=(0, 100))
    axes[1].legend()
    for idle, color in [(50., ORANGE), (60., BLUE), (70., GREEN)]:
        axes[2].plot(x, part[f'inverse_ratio_idle_{idle:.3f}'], '.-', color=color, label=f'空闲{idle:.0f}W')
    axes[2].axhline(1, color=GRAY, ls='--', label='A100 TDP比例1.0')
    axes[2].set(ylabel='代表性假设下反推功率/TDP', xlabel='监测时刻（UTC；不能据此标定训练功率）', ylim=(0, 4))
    axes[2].legend(ncol=4)
    ticks = [0, 8, 16, 24, 32]
    axes[2].set_xticks(ticks, [pd.Timestamp(part.time_utc.iloc[i]).strftime('%m-%d %H时') for i in ticks])
    finish(fig, output, 'figure1-monitoring-overlap', font_path, audit)

    fixed = pd.read_csv(output / 'kalos-fixed-panel-diurnal.csv')
    profile = pd.read_csv(output / 'transfer-profile.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for cutoff, color, style in [(.95, GREEN, ':'), (.99, BLUE, '-'), (.999, ORANGE, '--')]:
        panel = fixed.loc[fixed.device_coverage_threshold.eq(cutoff)].sort_values('hour_utc')
        axes[0].plot(panel.hour_utc, panel.mean_power_w, color=color, ls=style,
            label=f'覆盖至少{100 * cutoff:g}%：{int(panel.devices.iloc[0])}张')
    axes[0].set(xlim=(0, 23), xticks=[0, 6, 12, 18, 23], xlabel='源UTC小时', ylabel='固定设备平均功率（W）', title='全部发布监测的固定设备样本')
    axes[0].legend(fontsize=9)
    axes[1].plot(profile.hour_utc, profile.shape_central, color=BLUE, label='形状迁移：GPU小时加权均值0.80')
    axes[1].axhline(.8, color=GRAY, ls='--', label='恒定0.80')
    axes[1].set(xlim=(0, 23), xticks=[0, 6, 12, 18, 23], ylim=(0, 1), xlabel='源UTC小时', ylabel='目标已分配GPU功率/TDP', title='H100情境；已分配专属源形状未知')
    axes[1].legend(fontsize=9)
    finish(fig, output, 'figure2-source-shape', font_path, audit)

    days = pd.read_csv(output / 'typical-day.csv')
    weeks = pd.read_csv(output / 'typical-week.csv')
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), layout='constrained')
    for col, key in enumerate(('B', 'C', 'D')):
        for row, (table, group, metric) in enumerate([(days, 'hour_of_source_day', 'it_mw'),
                (days, 'hour_of_source_day', 'facility_mw'), (weeks, 'hour_of_source_week', 'it_mw')]):
            ax = axes[row, col]
            for case, color, style, label in [('constant_080', GRAY, '--', '恒定0.80'), ('shape_central', BLUE, '-', '固定样本形状迁移')]:
                part = table.loc[table.server_type.eq(key) & table.reference_it_mw.eq(10) & table.case.eq(case)].sort_values(group)
                ax.plot(part[group], part[metric], color=color, ls=style, lw=1.7, label=label)
            ax.set(ylabel=('设施' if row == 1 else 'IT') + '功率（MW）')
            ax.margins(y=.2)
            if row == 0:
                ax.set_title(f'{key}-TRN · 10MW参考规模')
            if row < 2:
                ax.set(xlim=(0, 23), xticks=[0, 6, 12, 18, 23], xlabel='源UTC小时')
            else:
                ax.set(xlim=(0, 167), xticks=[0, 24, 48, 72, 96, 120, 144],
                       xticklabels=['一', '二', '三', '四', '五', '六', '日'], xlabel='源星期（UTC小时）')
            ax.legend(fontsize=8, loc='lower right')
    finish(fig, output, 'figure3-trn-day-week', font_path, audit)

    summary = pd.read_csv(output / 'window-summary.csv')
    selected = summary.loc[summary.scenario_id.eq('D-TRN-10MW')].set_index('case')
    cases = ['constant_080', 'shape_half_amplitude', 'shape_central', 'shape_onehalf_amplitude', 'shape_panel095', 'shape_panel0999']
    labels = ['恒定0.80', '幅度0.5倍', '中央形状', '幅度1.5倍', '设备覆盖95%', '设备覆盖99.9%']
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), layout='constrained')
    for ax, metric, ylabel in zip(axes, ['it_diurnal_peak_valley', 'peak_it_mw'], ['IT日内最高/最低', '90天窗口IT峰值（MW）']):
        ax.plot(np.arange(len(cases)), selected.loc[cases, metric], 'o', color=BLUE, ms=7)
        ax.set(xlim=(-.5, 5.5), xticks=np.arange(6), xticklabels=labels, ylabel=ylabel, title='D-TRN-10MW：形状及设备集合敏感性')
        ax.tick_params(axis='x', labelrotation=22)
    finish(fig, output, 'figure4-sensitivity', font_path, audit)
    (output / 'figure-qa.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--font', default='Microsoft YaHei')
    plot(parser.parse_args())
