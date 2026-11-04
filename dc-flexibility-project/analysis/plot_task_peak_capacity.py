"""柔性经济模型中算力容量 K 的来源可视化。

K = 原生窗口内“任务 IT 功率”（刚性 + 未知 + 可推迟·整体 + 可推迟·可拆分，按原到达时刻）的最大值，
不含基础与空载 IT 和冷却等非 IT。左栏为原生窗口时序（缺测小时断开），右栏为任务功率持续曲线。
输入 results/flex-class-20261101/native-hourly/<情境>.parquet，输出 results/flex-class-20261101/task-peak-capacity.{png,pdf,csv}。
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from plot_scenario_typical_days import setup

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/flex-class-20261101/native-hourly'
OUT = ROOT / 'results/flex-class-20261101/task-peak-capacity'
SCENARIOS = ['A-HPC', 'A-CLD', 'D-SAI', 'D-TRN', 'D-INF']
CLASSES = ['刚性', '未知', '可推迟·整体', '可推迟·可拆分']
COLORS = {'刚性': '#CB6B53', '可推迟·整体': '#4C8BA5', '可推迟·可拆分': '#69A682', '未知': '#D8B36B'}
K_COLOR = '#B2182B'


def load(scenario):
    frame = pd.read_parquet(SOURCE / f'{scenario}.parquet').sort_values('source_hour_index').reset_index(drop=True)
    missing = [label for label in CLASSES if label not in frame.columns]
    if missing:
        raise KeyError(f'{scenario} 缺少任务层：{missing}')
    frame['task_mw'] = frame[CLASSES].sum(axis=1)
    return frame


def with_gaps(frame):
    """在来源小时不连续处插入空行，使缺测时段在图中断开。"""
    hours = frame.source_hour_index.to_numpy()
    breaks = np.flatnonzero(np.diff(hours) > 1)
    rows = []
    for index in breaks:
        gap = {column: np.nan for column in frame.columns}
        gap['source_hour_index'] = hours[index] + 1
        rows.append(gap)
    if rows:
        frame = pd.concat([frame, pd.DataFrame(rows, columns=frame.columns)]).sort_values('source_hour_index').reset_index(drop=True)
    return frame


def main():
    setup()
    fig, axes = plt.subplots(len(SCENARIOS), 2, figsize=(13.5, 2.15 * len(SCENARIOS)),
                             gridspec_kw={'width_ratios': [5.2, 1], 'wspace': .06, 'hspace': .62})
    records = []
    for row, scenario in enumerate(SCENARIOS):
        frame = load(scenario)
        peak_k = float(frame.task_mw.max())
        at = frame.loc[frame.task_mw.idxmax()]
        start = frame.source_hour_index.min()
        dated = scenario not in ('A-CLD', 'D-INF')
        name = str(frame.scenario_name.iloc[0]).replace('·10MW', '')
        mean_pct = frame.task_mw.mean() / peak_k * 100
        near95 = int((frame.task_mw > .95 * peak_k).sum())
        near90 = int((frame.task_mw > .90 * peak_k).sum())
        peak_label = pd.Timestamp(at.source_clock).strftime('%Y-%m-%d %H时') if dated else f'窗口第 {(at.source_hour_index - start) / 24:.1f} 天'
        records.append({'scenario': scenario, 'scenario_name': name, 'hours': len(frame), 'K_mw': round(peak_k, 4),
                        'peak_time': peak_label, 'mean_task_mw': round(float(frame.task_mw.mean()), 4), 'mean_pct_of_K': round(mean_pct, 2),
                        'hours_above_90pct_K': near90, 'hours_above_95pct_K': near95,
                        'base_idle_it_mean_mw': round(float(frame['基础与空载 IT'].mean()), 4) if '基础与空载 IT' in frame else None})

        ax = axes[row, 0]
        plot = with_gaps(frame)
        x = (plot.source_hour_index - start) / 24
        ax.stackplot(x, *[plot[label].to_numpy() for label in CLASSES], colors=[COLORS[label] for label in CLASSES], linewidth=0)
        ax.axhline(peak_k, color=K_COLOR, lw=1.1, ls='--')
        px = (at.source_hour_index - start) / 24
        ax.plot([px], [peak_k], 'o', ms=5, mfc='white', mec=K_COLOR, mew=1.3, zorder=5)
        span = x.max()
        ax.text(px + span * .008 if px < span * .8 else px - span * .008, peak_k * 1.04, f'K = {peak_k:.2f} MW（{peak_label}）',
                ha='left' if px < span * .8 else 'right', va='bottom', fontsize=8.5, color=K_COLOR)
        ax.set(xlim=(0, span), ylim=(0, peak_k * 1.3))
        ax.set_ylabel('任务 IT（MW）', fontsize=9)
        ax.set_title(f'{name}：原生窗口 {len(frame):,} 小时', loc='left', fontsize=10, pad=4)
        ax.text(1, 1.02, f'平均 = K 的 {mean_pct:.0f}%    >95%K：{near95} 小时    >90%K：{near90} 小时',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5, color='#35414D')
        ax.grid(axis='y', alpha=.2, lw=.4)
        ax.tick_params(labelsize=8, pad=2)
        if row == len(SCENARIOS) - 1:
            ax.set_xlabel('自窗口起点的天数（缺测时段断开）', fontsize=9)

        dur = axes[row, 1]
        ranked = np.sort(frame.task_mw.to_numpy())[::-1] / peak_k * 100
        share = np.arange(1, len(ranked) + 1) / len(ranked) * 100
        dur.fill_between(share, ranked, color='#8C96A0', alpha=.55, lw=0)
        dur.plot(share, ranked, color='#35414D', lw=.8)
        dur.axhline(100, color=K_COLOR, lw=1.1, ls='--')
        dur.axhline(95, color=K_COLOR, lw=.6, ls=':')
        dur.set(xlim=(0, 100), ylim=(0, 125), xticks=[0, 50, 100], yticks=[0, 50, 100])
        dur.yaxis.tick_right()
        dur.tick_params(labelsize=8, pad=2)
        dur.grid(alpha=.2, lw=.4)
        dur.set_title('持续曲线（% K）', fontsize=9, pad=4)
        if row == len(SCENARIOS) - 1:
            dur.set_xlabel('小时占比（%）', fontsize=9)

    fig.legend(handles=[Patch(facecolor=COLORS[label], label=label) for label in CLASSES]
               + [plt.Line2D([], [], color=K_COLOR, ls='--', label='K：任务 IT 窗口峰值')],
               loc='upper center', ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(.5, .995))
    fig.text(.01, .005, 'K 只计四类任务的原到达功率，不含基础与空载 IT、冷却等非 IT，因此低于原生窗口时序总览中的设施峰值。'
             '调度时每小时可推迟任务最多执行到 K 减去当时的刚性与未知负荷；风光储与市电接入容量另行优化。'
             'A-CLD、D-INF 无真实日期，按代表放置显示相对天数。', fontsize=8, color='#535B63')
    fig.subplots_adjust(left=.055, right=.965, top=.93, bottom=.075)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix('.png'), dpi=300, facecolor='white')
    fig.savefig(OUT.with_suffix('.pdf'), facecolor='white')
    pd.DataFrame(records).to_csv(OUT.with_suffix('.csv'), index=False, encoding='utf-8-sig')
    print(pd.DataFrame(records).to_string())


if __name__ == '__main__':
    main()
