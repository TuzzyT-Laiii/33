import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_trn_job_power_evidence import finish
from plot_trn_power_model import BLUE, GREEN, ORANGE, GRAY, setup
from prepare_trn_seren import RUN, TYPES
from report_trn_seren import LABELS


def plot(args):
    project = args.project
    output = project / 'results' / RUN
    font, audit = setup(args.font), []
    source = pd.read_csv(output / 'source-hourly.csv', parse_dates=['source_clock'])
    x = source.source_clock
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, layout='constrained')
    axes[0].plot(x, source.allocated_gpu_hours, color=BLUE, lw=.8, label='全部GPU作业分配')
    axes[0].plot(x, source.gpu_hours_Pretrain, color=GREEN, lw=.7, label='其中预训练')
    axes[0].plot(x, source.gpu_hours_MLLM, color=ORANGE, lw=.7, label='其中MLLM')
    axes[0].set(ylabel='小时平均分配GPU数（张）', ylim=(0, 2200), title='a  Seren原生需求：4192个小时分组')
    axes[0].legend(ncol=3)
    axes[1].plot(x, source['inverse_ratio_idle_60.000'], '.', color=GRAY, markersize=2, label='全部有监测的小时')
    axes[1].plot(x, source['inverse_ratio_idle_60.000'].where(source.power_quality_80), color=BLUE, lw=.8, label='正文合格小时；保留时间缺口')
    axes[1].axhline(1, color=ORANGE, ls='--', lw=1, label='源设备额定TDP')
    axes[1].set(ylabel='已分配GPU估计功率 / 400W', title='b  同期集群条件反推；默认空闲60W')
    axes[1].legend(ncol=3, fontsize=9)
    axes[2].plot(x, source.physical_device_coverage * 100, color=GREEN, lw=.7, label='功率报送设备 / 2288张物理库存')
    axes[2].axhline(80, color=ORANGE, ls='--', label='正文覆盖阈值80%')
    axes[2].set(ylabel='报送覆盖率（%）', xlabel='源UTC日期；真实间隔保留', ylim=(70, 90), title='c  报送数量不等同于在线数量')
    axes[2].legend(ncol=2, fontsize=9)
    axes[2].xaxis.set_major_locator(mdates.MonthLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    finish(fig, output, 'figure1-source', font, audit)

    day = pd.read_csv(output / 'typical-day.csv')
    old = pd.read_csv(project / 'results/trn-power-model-20261101/diurnal-acme/typical-day.csv')
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
    for col, key in enumerate(('B', 'C', 'D')):
        new_part = day.loc[day.server_type.eq(key) & day.reference_it_mw.eq(10)]
        old_part = old.loc[old.server_type.eq(key) & old.reference_it_mw.eq(10) & old.case.eq('shape_central')]
        for row, (component, label) in enumerate([('it_mw', 'IT功率（MW）'), ('facility_mw', '设施功率（MW）')]):
            ax = axes[row, col]
            for case, color, text in [('main', BLUE, 'Seren同期反推'), ('seren_constant_080', GRAY, '相同Seren需求，恒定0.80')]:
                part = new_part.loc[new_part.case.eq(case)]
                ax.plot(part.hour_of_source_day, part[component], color=color, lw=2, label=text)
            ax.plot(old_part.hour_of_source_day, old_part[component], color=ORANGE, ls='--', label='PR #36：Kalos需求及形状')
            ax.set(xlim=(0, 23), xticks=[0, 6, 12, 18, 23], xlabel='源UTC小时', ylabel=label, title=f'{chr(97 + row*3 + col)}  {key}-TRN-10MW')
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    finish(fig, output, 'figure2-comparison', font, audit)

    profiles = pd.read_csv(output / 'power-profiles.csv')
    summary = pd.read_csv(output / 'window-summary.csv')
    composition = pd.read_csv(output / 'task-composition.csv')
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), layout='constrained')
    for ax, grouping, title, ticks, xlabel in [
        (axes[0, 0], 'hour_of_source_day', 'a  合格窗口的日内条件均值', [0, 6, 12, 18, 23], '源UTC小时'),
        (axes[0, 1], 'hour_of_source_week', 'b  合格窗口的周内条件均值', [0, 24, 48, 72, 96, 120, 144, 167], '源星期小时；0为周一00时')]:
        part = profiles.loc[profiles.grouping.eq(grouping)]
        ax.plot(part.position, part.gpu_hour_weighted_mean_ratio, color=BLUE)
        ax.fill_between(part.position, part.p10, part.p90, color=BLUE, alpha=.15, label='观测P10–P90；不表示置信区间')
        ax.set(xlim=(part.position.min(), part.position.max()), xticks=ticks, xlabel=xlabel,
               ylabel='已分配GPU估计功率 / 400W', title=title)
        ax.legend(fontsize=8)
    part = summary.loc[summary.server_type.eq('D') & summary.reference_it_mw.eq(10)].copy()
    y = np.arange(len(part))
    labels = [LABELS[row.case] + f'（{int(row.hours)}h）' for row in part.itertuples()]
    axes[1, 0].scatter(part.mean_it_mw, y, color=BLUE, label='窗口平均IT', zorder=3)
    axes[1, 0].scatter(part.peak_it_mw, y, color=ORANGE, marker='|', s=100, label='窗口峰值IT', zorder=3)
    axes[1, 0].set(yticks=y, yticklabels=labels, xlabel='IT功率（MW）', title='c  D-TRN-10MW：功率与筛选敏感性')
    axes[1, 0].tick_params(axis='y', labelsize=8)
    axes[1, 0].invert_yaxis()
    axes[1, 0].legend(fontsize=8)
    colors = [GREEN, '#8B6BB1', GRAY, BLUE, ORANGE, '#C5954E']
    left = np.zeros(2)
    for kind, color in zip(TYPES, colors):
        values = composition.loc[composition.type.eq(kind)].set_index('window').reindex(['full_jobs', 'qualified_overlap']).gpu_share.to_numpy() * 100
        axes[1, 1].barh([0, 1], values, left=left, color=color, label=kind, height=.5)
        left += values
    axes[1, 1].set(yticks=[0, 1], yticklabels=['完整作业窗口', '正文同期合格窗口'], xlim=(0, 100),
                   xlabel='分配GPU小时占比（%）', title='d  任务类型构成；MLLM独立保留')
    axes[1, 1].legend(ncol=3, fontsize=9, loc='lower center', bbox_to_anchor=(.5, .85))
    finish(fig, output, 'figure3-shapes-sensitivity', font, audit)
    (output / 'figure-qa.json').write_text(json.dumps({'figures': audit, 'visual_inspection': False,
        'checks': 'Chinese glyph coverage, text bounds, PDF font embedding; no visual tool used'}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--font', default='Microsoft YaHei')
    plot(parser.parse_args())
