import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_trn_power_model import BLUE, ORANGE, GREEN, GRAY, setup, save


def finish(fig, output, name, font, audit):
    for ax in fig.axes:
        low, high = ax.get_ylim()
        ax.set_yticks([x for x in ax.get_yticks() if low <= x <= high])
        low, high = ax.get_xlim()
        ax.set_xticks([x for x in ax.get_xticks() if low <= x <= high])
        ax.grid(axis='y', alpha=.15)
    save(fig, output, name, font, audit)


def plot(args):
    output = args.project / 'results/trn-power-model-20261101/job-power-evidence'
    font, audit = setup(args.font), []
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), layout='constrained')
    histogram = pd.read_csv(output / 'mit/mit-power-histogram.csv')
    for cohort, color, label in [('all_observed_V100', GRAY, '全部已观测V100'), ('official_DNN', BLUE, '官方DNN训练标签')]:
        part = histogram.loc[histogram.cohort.eq(cohort)].sort_values('power_bin_lower_w')
        axes[0, 0].plot((part.power_bin_lower_w + 2.5) / 250, part.gpu_hours.cumsum() / part.gpu_hours.sum(), color=color, label=label)
    axes[0, 0].set(xlabel='GPU功率 / 250W（5W分箱中点）', ylabel='设备时间累计比例', title='MIT：分配内实测分布', ylim=(0, 1))
    axes[0, 0].legend()
    day = pd.read_csv(output / 'seren/seren-diurnal.csv').query('case == "coverage_qualified_all_ratios"')
    axes[0, 1].plot(day.hour_utc, day.gpu_hour_weighted_mean_ratio, 'o-', color=GREEN, label='Seren混合负载条件反推')
    axes[0, 1].axhline(.8, color=GRAY, ls='--', label='当前TRN中央情境0.80')
    axes[0, 1].set(xlim=(0, 23), xticks=[0, 6, 12, 18, 23], xlabel='源UTC小时', ylabel='已分配GPU估计功率 / 400W', title='Seren：2057个合格小时')
    axes[0, 1].legend(fontsize=9)
    sessions = pd.read_csv(output / 'h100-experiments/h100-llm-session-summary.csv')
    labels = []
    for row in sessions.itertuples():
        batch = row.experiment.split('BatchSize')[1].split('.')[0]
        zero = row.experiment.split('Deepspeedds_')[1].split('_')[0]
        length = row.experiment.split('SeqLength')[1].split('_')[0]
        labels.append(f'{row.model_billion_parameters}B / batch{batch} / {zero} / 长度{length}')
    axes[1, 0].barh(np.arange(len(sessions)), sessions.mean_gpu_tdp_ratio, color=BLUE, height=.65)
    axes[1, 0].axvline(.8, color=GRAY, ls='--')
    axes[1, 0].set(yticks=np.arange(len(sessions)), yticklabels=labels, xlim=(0, 1),
                   xlabel='八卡记录窗口平均GPU功率 / 700W', title='H100：9个独立15分钟LLaMA训练窗口')
    axes[1, 0].tick_params(axis='y', labelsize=8)
    progress = pd.read_csv(output / 'mit/mit-within-job-progress.csv').query('cohort == "official_DNN"')
    x = (progress.progress_bin + .5) * 10
    axes[1, 1].plot(x, progress.relative_power_mean, color=BLUE, label='官方DNN：至少4个小时分组的作业')
    axes[1, 1].fill_between(x, progress.relative_power_p10, progress.relative_power_p90, color=BLUE, alpha=.15, label='观察到的P10—P90范围')
    axes[1, 1].set(xlim=(0, 100), xticks=[0, 25, 50, 75, 100], xlabel='作业内相对进度（%，按小时均值近似）',
                   ylabel='功率 / 对应作业均值', title='MIT：长训练作业的小时级变化')
    axes[1, 1].legend(fontsize=8)
    finish(fig, output, 'figure-evidence', font, audit)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    curves = pd.read_csv(output / 'h100-experiments/h100-llm-session-progress.csv')
    for row in sessions.itertuples():
        part = curves.loc[curves.experiment.eq(row.experiment)]
        batch = row.experiment.split('BatchSize')[1].split('.')[0]
        zero = row.experiment.split('Deepspeedds_')[1].split('_')[0]
        length = row.experiment.split('SeqLength')[1].split('_')[0]
        axes[0].plot((part.progress_bin + .5) * 5, part.mean_power_tdp_ratio, lw=1.2,
            label=f'{row.model_billion_parameters}B/b{batch}/{zero}/L{length}')
    axes[0].set(xlim=(0, 100), xticks=[0, 25, 50, 75, 100], ylim=(0, 1), xlabel='15分钟记录窗口进度（%）',
        ylabel='平均GPU功率 / 700W', title='受控训练窗口：每段约45秒汇总')
    axes[0].legend(fontsize=7, ncol=2, loc='lower left')
    proposals = pd.read_csv(output / 'proposal-typical-day.csv').query('scenario_id == "D-TRN-10MW"')
    for case, color, label, style in [('current_080_Kalos', BLUE, '当前Kalos形状', '-'),
        ('recommended_080_constant', GRAY, '建议正文：分配内比例恒定0.80', '--'),
        ('sensitivity_080_Seren', GREEN, 'Seren形状敏感性', ':')]:
        part = proposals.loc[proposals.case.eq(case)]
        axes[1].plot(part.hour_of_source_day, part.it_mw, color=color, ls=style, label=label)
    axes[1].set(xlim=(0, 23), xticks=[0, 6, 12, 18, 23], xlabel='原生需求UTC小时', ylabel='IT功率（MW）', title='D-TRN-10MW：报告内方案比较')
    axes[1].legend(fontsize=9)
    finish(fig, output, 'figure-proposal-comparison', font, audit)
    (output / 'figure-qa.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--font', default='Microsoft YaHei')
    plot(parser.parse_args())
