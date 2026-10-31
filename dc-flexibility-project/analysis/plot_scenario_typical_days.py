import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.text import Text
import matplotlib.pyplot as plt
from lxml import etree
import numpy as np
import pandas as pd


COLORS = {'基础与空载 IT': '#D9DEE3', '冷却等非 IT': '#9CABC2',
    'CPU 科研分区': '#418A9B', 'Jupyter': '#B184B5', '课程分区': '#E8B65B', '其他分区': '#7F8C8D',
    '业务未分解': '#418A9B', '批处理': '#418A9B', 'MapReduce': '#DCA24D', '交互式': '#B184B5', 'OTHER': '#788486',
    '预训练': '#418A9B', 'SFT': '#81BAA5', '评估': '#DCA24D', '调试': '#B184B5', '其他': '#788486',
    '训练': '#418A9B', '在线推理': '#DB9673', '离线推理': '#DCA24D', '开发': '#B184B5', '其他/未知': '#788486'}


def setup():
    path = Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(path)).get_name(),
        'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.unicode_minus': False, 'legend.frameon': False, 'svg.fonttype': 'path',
        'savefig.dpi': 300, 'axes.titleweight': 'normal'})


def components(output, scenario):
    frame = pd.read_csv(output / (scenario + '-by-task.csv'))
    labels = frame.task_type.drop_duplicates().tolist()
    values = [frame[frame.task_type.eq(label)].sort_values('hour').facility_mw.to_numpy() for label in labels]
    return labels, values


def draw(ax, output, row, upper, small=False):
    labels, values = components(output, row['scenario'])
    ax.stackplot(np.arange(25), *[np.r_[value, value[-1]] for value in values],
        step='post', colors=[COLORS[label] for label in labels], linewidth=0)
    total = sum(values)
    ax.step(np.arange(25), np.r_[total, total[-1]], where='post', color='#35414D', lw=.8)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, upper)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel('小时', fontsize=9 if small else 11, labelpad=1)
    ax.set_ylabel('设施功率（MW）', fontsize=9 if small else 11, labelpad=2)
    ax.set_title(row['scenario_name'], loc='left', fontsize=10 if small else 15, pad=21 if small else 28)
    ax.text(0, 1.025, row['source_date'] + '  ' + row['weekday'], transform=ax.transAxes,
        fontsize=8.5 if small else 10, color='#535B63')
    ax.grid(axis='y', color='#AAB2BA', lw=.4, alpha=.3)
    ax.set_axisbelow(False)
    ax.tick_params(labelsize=8 if small else 10, pad=2)
    return labels


def handles(labels):
    return [Patch(facecolor=COLORS[label], label=label) for label in labels]


def save(fig, output, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for artist in fig.findobj(Text):
        if artist.get_visible() and artist.get_text().strip():
            box = artist.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
                outside.append(artist.get_text())
    if outside:
        raise AssertionError({'figure': stem, 'outside_text': outside})
    fig.savefig(output / (stem + '.png'), dpi=300, facecolor='white')
    fig.savefig(output / (stem + '.svg'), facecolor='white')
    svg = output / (stem + '.svg')
    # 统一 XML 序列化，避免路径属性产生行尾空格。
    etree.parse(str(svg)).write(str(svg), encoding='utf-8', xml_declaration=True, pretty_print=True)
    plt.close(fig)
    return True


def plot_all(output):
    output = Path(output)
    setup()
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    rows = {row['scenario']: row for row in summary['scenarios']}
    upper = float(np.ceil(max(row['peak_mw'] for row in rows.values()) * 1.08))
    checks = {}
    for scenario, row in rows.items():
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        fig.subplots_adjust(left=.10, right=.97, top=.78, bottom=.29)
        labels = draw(ax, output, row, upper)
        fig.legend(handles=handles(labels), loc='lower left', bbox_to_anchor=(.095, .12), ncol=4, fontsize=9)
        use = scenario.split('-')[1]
        selected = summary['selections'][use]
        rule = f"{selected['complete_days']} 个完整日：日电量最接近中位数；并列时日内 RMSE 最小。" if selected['complete_days'] > 1 else '唯一完整日（第 109 天）；日历日期与星期未知，单日演示。'
        fig.text(.10, .065, rule, fontsize=9, color='#4C535A')
        fig.text(.10, .025, '参考 IT 10 MW；基础与空载独立列示；冷却等非 IT ＝ 设施功率 − IT 功率。', fontsize=9, color='#4C535A')
        checks[scenario + '_figure_bounds'] = save(fig, output, scenario)
    fig, axes = plt.subplots(4, 3, figsize=(16, 10.0))
    fig.subplots_adjust(left=.065, right=.985, top=.885, bottom=.09, hspace=1.15, wspace=.26)
    fig.suptitle('各情境典型日 24 小时负荷曲线', x=.065, y=.985, ha='left', fontsize=20)
    fig.text(.065, .944, '参考 IT 10 MW  ·  同一需求共用日期与资源构成  ·  全部面板共用纵轴', fontsize=11, color='#535B63')
    grid = [['A-HPC', 'A-CLD', None], ['B-SAI', 'C-SAI', 'D-SAI'],
            ['B-TRN', 'C-TRN', 'D-TRN'], ['B-INF', 'C-INF', 'D-INF']]
    for r, scenarios in enumerate(grid):
        for c, scenario in enumerate(scenarios):
            ax = axes[r, c]
            if scenario is None:
                ax.axis('off')
                ax.legend(handles=handles(['基础与空载 IT', '冷却等非 IT']), loc='upper left', fontsize=10)
                ax.text(.02, .43, '逐小时平均功率，阶梯面积表示小时区间。\n源日期与气象模型日期分别记录。\nAlibaba 的在线／离线实际用量待分解。',
                    va='top', transform=ax.transAxes, fontsize=9, color='#535B63', linespacing=1.6)
                continue
            if scenario not in rows:
                ax.set_title('纯 CPU 型·云通用计算', loc='left', fontsize=10)
                ax.text(.5, .5, '待数据', ha='center', va='center', transform=ax.transAxes)
                ax.set_xlim(0, 24)
                ax.set_ylim(0, upper)
                continue
            labels = draw(ax, output, rows[scenario], upper, small=True)
            if r == 0:
                ax.legend(handles=handles(labels[1:-1]), loc='upper left', bbox_to_anchor=(0, -.32),
                    ncol=4, fontsize=8, handlelength=1.0, columnspacing=.8)
            elif c == 0:
                ax.legend(handles=handles(labels[1:-1]), loc='upper left', bbox_to_anchor=(0, -.32),
                    ncol=5, fontsize=9, handlelength=1.0, columnspacing=1.4)
    fig.text(.065, .018, '任务层为资源驱动的 IT 增量；非线性 CPU 增量按小时资源份额分摊。ASI 为单日演示，所有曲线均含功率与冷却模型假设。', fontsize=10, color='#535B63')
    checks['overview_figure_bounds'] = save(fig, output, 'overview')
    return checks
