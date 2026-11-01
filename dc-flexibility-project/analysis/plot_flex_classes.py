import json

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from plot_native_combined import GRID, save
from plot_scenario_typical_days import setup


CLASSES = ['刚性', '可推迟·整体', '可推迟·可拆分', '未知']
LEGEND = [*CLASSES, '基础与空载 IT', '冷却等非 IT']
LAYERS = ['基础与空载 IT', *CLASSES, '冷却等非 IT']
COLORS = {'刚性': '#CB6B53', '可推迟·整体': '#4C8BA5', '可推迟·可拆分': '#69A682',
    '未知': '#D8B36B', '基础与空载 IT': '#D9DEE3', '冷却等非 IT': '#9CABC2'}


def handles():
    return [Patch(facecolor=COLORS[label], label=label) for label in LEGEND]


def day_panel(ax, typical, row, upper, small):
    parts = pd.read_csv(typical / (row['scenario'] + '-by-task.csv'))
    values = [parts.loc[parts.task_type.eq(label)].sort_values('hour').facility_mw.to_numpy() for label in LAYERS]
    ax.stackplot(np.arange(25), *[np.r_[value, value[-1]] for value in values], step='post',
        colors=[COLORS[label] for label in LAYERS], linewidth=0)
    total = sum(values)
    ax.step(np.arange(25), np.r_[total, total[-1]], where='post', color='#35414D', lw=.7)
    ax.set(xlim=(0, 24), ylim=(0, upper), xticks=[0, 6, 12, 18, 24])
    ax.set_xlabel('来源UTC小时' if row['scenario'].endswith('-TRN') else '来源小时', fontsize=9 if small else 11)
    ax.set_ylabel('设施功率（MW）', fontsize=9 if small else 11)
    title_size = 9 if small and row['scenario'].endswith('-TRN') else 10 if small else 15
    ax.set_title(row['scenario_name'], loc='left', fontsize=title_size, pad=21 if small else 28)
    ax.text(0, 1.025, row['source_date'] + '  ' + row['weekday'], transform=ax.transAxes, fontsize=8.5 if small else 10, color='#535B63')
    ax.text(.99, .94, f'最高/最低 {row["peak_valley_ratio"]:.3f}', transform=ax.transAxes, ha='right', va='top', fontsize=7.6 if small else 10)
    ax.grid(axis='y', alpha=.2, lw=.4)
    ax.tick_params(labelsize=8 if small else 10, pad=2)


def typical_days(output, rows):
    typical = output / 'selected-days'
    upper = float(np.ceil(max(row['peak_mw'] for row in rows.values()) * 1.08))
    fig, axes = plt.subplots(4, 3, figsize=(16, 10))
    fig.subplots_adjust(left=.065, right=.985, top=.885, bottom=.09, hspace=.92, wspace=.26)
    fig.suptitle('按时间柔性分类的典型日负荷', x=.065, y=.985, ha='left', fontsize=20)
    fig.text(.065, .944, '参考IT 10 MW  ·  原有典型日与功率保持  ·  全部情境采用相同四类与配色', fontsize=11, color='#535B63')
    for r, scenarios in enumerate(GRID):
        for c, scenario in enumerate(scenarios):
            ax = axes[r, c]
            if scenario is None:
                ax.axis('off')
                ax.legend(handles=handles(), loc='upper left', fontsize=9.5, ncol=2, columnspacing=1)
                ax.text(.02, .14, '未知默认按刚性处理。\n按任务类型的资源份额分摊功率；\n推迟时长和分段长度尚缺数据支撑。', transform=ax.transAxes, fontsize=9, va='top', color='#535B63', linespacing=1.5)
            else:
                day_panel(ax, typical, rows[scenario], upper, True)
    fig.text(.065, .025, '原有选日：完整日中位电量±2%候选，少于5天放宽±5%，再按最小RMSE；类别份额不等于可调电量。', fontsize=10, color='#535B63')
    save(fig, typical, 'overview')
    for scenario, row in rows.items():
        fig, ax = plt.subplots(figsize=(10.2, 5.4))
        fig.subplots_adjust(left=.09, right=.98, top=.80, bottom=.29)
        day_panel(ax, typical, row, upper, False)
        fig.legend(handles=handles(), loc='lower left', bbox_to_anchor=(.085, .11), ncol=2, fontsize=9)
        fig.text(.09, .065, '分类采用用户确认映射；未知默认按刚性，基础与空载IT、冷却等非IT分别保留。', fontsize=9)
        fig.text(.09, .025, '原有来源日期、气象放置及逐小时功率保持；推迟时长、分段长度和恢复成本另作为情景假设。', fontsize=9)
        save(fig, typical, scenario)


def native_overview(output, rows):
    placements = pd.read_csv(output / 'representative-placements.csv').set_index('scenario')
    frames = {scenario: pd.read_parquet(output / 'native-hourly' / (scenario + '.parquet')) for scenario in rows}
    upper = float(np.ceil(max(float(frame.facility_mw.max()) for frame in frames.values()) * 1.04))
    fig, axes = plt.subplots(4, 3, figsize=(16, 10), sharey=True)
    fig.subplots_adjust(left=.06, right=.985, top=.89, bottom=.09, hspace=.70, wspace=.23)
    fig.suptitle('按时间柔性分类的原生窗口负荷', x=.06, y=.98, ha='left', fontsize=20)
    fig.text(.06, .935, '参考IT 10 MW  ·  各来源保留原生时长  ·  设施功率按六层堆叠  ·  全部面板共用纵轴', fontsize=11, color='#535B63')
    for r, scenarios in enumerate(GRID):
        for c, scenario in enumerate(scenarios):
            ax = axes[r, c]
            if scenario is None:
                ax.axis('off')
                ax.legend(handles=handles(), loc='upper left', fontsize=9.5, ncol=2, columnspacing=1)
                ax.text(.02, .14, '无日期来源选择完整窗口的代表放置，\n最接近12次放置设施均值的中位数。\n范围同时含天气和截断源前缀差异。', transform=ax.transAxes, fontsize=9, va='top', color='#535B63', linespacing=1.5)
                continue
            frame = frames[scenario]
            row = placements.loc[scenario]
            if scenario.endswith('-TRN'):
                clock = pd.DatetimeIndex(pd.to_datetime(frame.source_clock, utc=True))
                calendar = pd.date_range(clock[0], clock[-1], freq='h')
                values = frame[LAYERS].set_axis(clock).reindex(calendar)
                x = (calendar - calendar[0]).total_seconds() / 86400
                ax.set_xlim(0, x[-1])
                ax.set_xticks([0, 30, 60, 90])
                caption = '自05-16 10时UTC起经过天数（2057小时，有缺口）'
            else:
                values = frame[LAYERS]
                x = np.arange(len(frame)) / 24
                days = len(frame) / 24
                ax.set_xlim(0, days)
                step = int(np.ceil(days / 4))
                ax.set_xticks(np.unique(np.r_[np.arange(0, days, step), days]))
                caption = f'来源经过天数（共{int(days)}天）'
            # NaN仅断开真实时间缺口，不插入功率值或改变积分。
            ax.stackplot(x, *[values[label].to_numpy() for label in LAYERS], colors=[COLORS[label] for label in LAYERS], linewidth=0)
            ax.set_title(row.scenario_name, loc='left', fontsize=10)
            if row.placement != 'dated':
                text = f'{int(row.placement[1:])}月初代表放置；设施均值范围{row.facility_mean_range_min_mw:.3f}–{row.facility_mean_range_max_mw:.3f} MW'
                ax.text(.01, .96, text, transform=ax.transAxes, va='top', fontsize=7.2, color='#535B63')
            ax.set_xlabel(caption, fontsize=8 if scenario.endswith('-TRN') else 9)
            ax.set_ylim(0, upper)
            ax.set_yticks(np.arange(0, upper+.001, 2))
            ax.tick_params(labelsize=8, pad=2)
            ax.grid(axis='y', alpha=.15, lw=.4)
            if c == 0:
                ax.set_ylabel('功率（MW）', fontsize=9)
    fig.text(.06, .025, '未知默认按刚性；Seren缺口断开。无日期来源的范围来自全部12次实际放置，代表堆叠曲线采用一次完整窗口放置。', fontsize=10, color='#535B63')
    save(fig, output, 'native-overview')


def shares(output):
    frame = pd.read_csv(output / 'shares-10MW.csv')
    fig, ax = plt.subplots(figsize=(16, 5.1))
    fig.subplots_adjust(left=.37, right=.985, top=.79, bottom=.15)
    left = np.zeros(len(frame))
    for label in ['基础与空载 IT', *CLASSES]:
        values = frame[label].to_numpy()
        ax.barh(np.arange(len(frame)), values, left=left, color=COLORS[label], height=.7)
        left += values
    ax.set_yticks(np.arange(len(frame)), frame.scenario_name, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.tick_params(axis='x', labelsize=11)
    ax.set_xlabel('占原生窗口总IT电量的比例（%）；亦为平均IT功率比例', fontsize=12)
    ax.grid(axis='x', alpha=.15, lw=.4)
    fig.legend(handles=handles(), loc='upper center', bbox_to_anchor=(.65, 1.01), ncol=2, fontsize=11, columnspacing=1.2)
    fig.text(.37, .018, '基础与空载IT补足至100%；冷却等非IT不计入IT份额；未知按刚性处理。', fontsize=11, color='#535B63')
    save(fig, output, 'shares-10MW')


def plot_all(output):
    setup()
    plt.rcParams['pdf.fonttype'] = 42
    summary = json.loads((output / 'selected-days/summary.json').read_text('utf-8'))
    rows = {row['scenario']: row for row in summary['scenarios']}
    typical_days(output, rows)
    native_overview(output, rows)
    shares(output)
