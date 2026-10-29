from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIGS = ('C20', 'C21', 'C22', 'C28', 'R-GB200')
TITLES = {'C20': 'C20 · 双路 CPU', 'C21': 'C21 · 双路 CPU + 1 张 H100 PCIe',
    'C22': 'C22 · 双路 CPU + 2 张 H100 PCIe', 'C28': 'C28 · DGX H100 八卡整机',
    'R-GB200': 'GB200 NVL72 · 直接液冷整柜'}
COLORS = {'Lisa': '#167D8D', 'MIT': '#167D8D', 'Acme': '#4B60AE', 'ASI': '#C27A24', 'Alibaba': '#8A579D'}


def setup():
    font = Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.unicode_minus': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.frameon': False,
        'savefig.dpi': 300, 'xtick.labelsize': 9, 'ytick.labelsize': 9})
    return str(font)


def save(fig, output, stem, footnote):
    fig.text(.075, .035, footnote, fontsize=9, va='bottom', color='#444444', linespacing=1.6)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for artist in fig.findobj(Text):
        if artist.get_visible() and artist.get_text().strip():
            box = artist.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
                outside.append(artist.get_text())
    fig.savefig(output / (stem + '.png'), dpi=300, facecolor='white')
    fig.savefig(output / (stem + '.pdf'), facecolor='white')
    plt.close(fig)
    return outside


def grid(title, subtitle, upper):
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    fig.subplots_adjust(left=.075, right=.98, bottom=.13, top=.855, wspace=.20, hspace=.48)
    fig.suptitle(title, y=.975, fontsize=18)
    fig.text(.5, .937, subtitle, ha='center', fontsize=10, color='#444444')
    handles = [Line2D([0], [0], color=COLORS[d], lw=2, label=label) for d, label in
        [('MIT', '科研：C20 为 Lisa CPU，其余为 MIT GPU'), ('Acme', 'Acme · 90 日循环'), ('ASI', 'ASI · 单日重复演示')]]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .916), ncol=3, fontsize=10)
    for index, key in enumerate(CONFIGS):
        ax = axes.flat[index]
        ax.set_title(chr(97 + index) + '  ' + TITLES[key], loc='left', pad=10)
        ax.set_ylim(0, upper)
        ax.set_ylabel('设施功率（MW）')
        ax.grid(axis='y', color='#e5e5e5', lw=.6)
    ax = axes.flat[5]
    ax.axis('off')
    ax.text(.02, .96, '\n\n'.join([
        '10 MW 为安装参考 IT 预算，台数/柜数向下取整。',
        '主曲线：实测功率比例中位 × 目标 GPU TDP。',
        '风冷与直接液冷均采用 v2 中央参数。',
        'MIT：1,574 文件随机前缀，功率分布尚未稳定。',
        'CPU、未分配 GPU 空载及跨代迁移含假设。',
        'C20 × Alibaba：待数据，跳过计算。']),
        transform=ax.transAxes, va='top', fontsize=10, linespacing=1.3)
    return fig, axes


def plot_all_v2(frames, comparison, output, week_start):
    font = setup()
    outside = {}
    upper = np.ceil(max(frame.facility_mw.max() for frame in frames.values()) * 1.06)
    common = '功率比例的 P10/P90 与冷却高/低档为敏感性情境，见汇总表；曲线未建立跨设备吞吐等价。'
    fig, axes = grid('典型工作日 · v2 设施负荷', '2025 年周一至周五按 UTC 小时求均值 · 10 MW IT 参考规模', upper)
    for index, key in enumerate(CONFIGS):
        ax = axes.flat[index]
        for (config, demand), frame in frames.items():
            if config == key:
                selected = frame.loc[frame.time_utc.dt.dayofweek.lt(5)]
                mean = selected.groupby(selected.time_utc.dt.hour).facility_mw.mean()
                ax.plot(mean.index, mean, color=COLORS[demand], lw=1.9)
        ax.set_xlim(0, 23)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlabel('UTC 小时')
    outside['figure1'] = save(fig, output, 'figure1-typical-weekday', common)
    first = next(iter(frames.values())).time_utc.iloc[week_start]
    fig, axes = grid('典型一周 · v2 设施负荷', f'共同代表周：{first:%Y-%m-%d} 至 {(first + pd.Timedelta(hours=167)):%Y-%m-%d} · UTC', upper)
    for index, key in enumerate(CONFIGS):
        ax = axes.flat[index]
        for (config, demand), frame in frames.items():
            if config == key:
                ax.plot(np.arange(168), frame.facility_mw.iloc[week_start:week_start + 168], color=COLORS[demand], lw=1.5)
        ax.set_xlim(0, 167)
        ax.set_xticks(np.arange(0, 168, 24), ['周一', '周二', '周三', '周四', '周五', '周六', '周日'])
        ax.set_xlabel('UTC 日期；每日起点 00:00')
    outside['figure2'] = save(fig, output, 'figure2-typical-week', common)
    fig, axes = grid('全年包络 · v2 日均与日内范围', '实线为每日均值，浅色包络为同日逐小时最小至最大值 · 2025 年', upper)
    for index, key in enumerate(CONFIGS):
        ax = axes.flat[index]
        for (config, demand), frame in frames.items():
            if config == key:
                daily = frame.groupby(frame.time_utc.dt.floor('D')).facility_mw.agg(['mean', 'min', 'max'])
                ax.fill_between(daily.index, daily['min'], daily['max'], color=COLORS[demand], alpha=.12, linewidth=0)
                ax.plot(daily.index, daily['mean'], color=COLORS[demand], lw=1.)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m月'))
        ax.set_xlim(pd.Timestamp('2025-01-01', tz='UTC'), pd.Timestamp('2025-12-31', tz='UTC'))
        ax.set_xlabel('目标月份（UTC）')
    outside['figure3'] = save(fig, output, 'figure3-annual-envelope',
        'Acme 的 90 日循环可形成反复出现的需求方块；这些结构不代表真实季节性。ASI 仅为单日重复演示。\n' + common)
    fig, ax = plt.subplots(figsize=(14, 7.8))
    fig.subplots_adjust(left=.075, right=.98, bottom=.21, top=.83)
    fig.suptitle('IT / 冷却构成 · v2 全年设施电量', y=.97, fontsize=18)
    keys = list(frames)
    x = np.arange(len(keys))
    bottom = np.zeros(len(keys))
    for column, label, color in [('base_it_mw', '基础 IT', '#9EAAB2'), ('cpu_it_mw', 'CPU', '#167D8D'),
        ('gpu_it_mw', 'GPU', '#4B60AE'), ('cooling_mw', '冷却', '#C27A24'), ('other_mw', 'UPS / 配电与其他', '#886B9E')]:
        values = np.array([frames[key][column].sum() / 1000 for key in keys])
        ax.bar(x, values, bottom=bottom, color=color, label=label, width=.66)
        bottom += values
    ax.set_xticks(x, [key + '\n' + demand for key, demand in keys], fontsize=9)
    ax.set_ylabel('年电量（GWh）')
    ax.set_ylim(0, max(bottom) * 1.12)
    ax.grid(axis='y', color='#eeeeee', linewidth=.6)
    ax.set_axisbelow(True)
    fig.legend(loc='upper center', bbox_to_anchor=(.5, .91), ncol=5)
    outside['figure4'] = save(fig, output, 'figure4-power-components',
        '10 MW IT 参考规模；GPU 功率比例中位与冷却中央参数。GB200 已包含直接液冷、残余风冷和设施电气损耗。\n' + common)
    fig, axes = plt.subplots(1, 3, figsize=(14, 8.8), sharey=True)
    fig.subplots_adjust(left=.18, right=.98, bottom=.21, top=.83, wspace=.18)
    fig.suptitle('v1 → v2 · 年电量、峰值与 PUE', y=.975, fontsize=18)
    y = np.arange(len(comparison))
    labels = [row.configuration_id + ' × ' + ('Lisa→MIT' if row.demand_changed else row.demand_v2) for row in comparison.itertuples()]
    for ax, (before, after, scale, title) in zip(axes, [
        ('v1_reported_annual_mwh', 'v2_annual_facility_mwh', 1000, '年电量（GWh）'),
        ('v1_reported_peak_mw', 'v2_facility_peak_mw', 1, '峰值功率（MW）'),
        ('v1_annual_pue', 'v2_annual_pue', 1, '全年电量比 PUE')]):
        for index, row in comparison.iterrows():
            a, b = row[before] / scale, row[after] / scale
            if np.isfinite(a):
                ax.plot([a, b], [index, index], color='#C7CCD0', lw=1.2)
                ax.scatter(a, index, color='#7F8A91', marker='^' if row.configuration_id == 'R-GB200' else 's', s=33, zorder=3)
            ax.scatter(b, index, color='#167D8D', marker='o', s=38, zorder=4)
        ax.set_title(title, pad=15)
        ax.grid(axis='x', color='#eeeeee', linewidth=.6)
        ax.tick_params(axis='y', length=0)
        if scale == 1000 or before.endswith('peak_mw'):
            ax.set_xlim(left=0)
        else:
            ax.set_xlim(1.05, 1.38)
            ax.set_xticks([1.1, 1.2, 1.3])
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    handles = [Line2D([0], [0], color='#7F8A91', marker='s', linestyle='', label='v1'),
        Line2D([0], [0], color='#167D8D', marker='o', linestyle='', label='v2'),
        Line2D([0], [0], color='#7F8A91', marker='^', linestyle='', label='GB200 的 v1：仅 IT')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.58, .91), ncol=3)
    outside['figure5'] = save(fig, output, 'figure5-v1-v2-comparison',
        '科研 GPU 组合由 v1 的 Lisa CPU 需求改为 v2 的 MIT GPU 作业，变化同时包含需求与功率方法。\n'
        'GB200 的 v1 仅有 IT 电量/峰值，v2 为设施；原版没有 PUE，故不计算 GB200 的设施增幅。')
    return {'font': font, 'dpi': 300, 'outside_canvas': outside,
        'checks': {'five_figures_inside_canvas': not any(outside.values()),
                   'five_png_and_pdf': all((output / (stem + suffix)).is_file() for stem in
                       ['figure1-typical-weekday', 'figure2-typical-week', 'figure3-annual-envelope', 'figure4-power-components', 'figure5-v1-v2-comparison'] for suffix in ('.png', '.pdf'))}}
