import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
import pandas as pd


COLORS = {'burst': '#333333', 'azure-2024-code': '#BB5929', 'azure-2024-conv': '#267C9E'}
NAMES = {'burst': 'BurstGPT（119日）', 'azure-2024-code': 'Azure 2024 代码', 'azure-2024-conv': 'Azure 2024 对话'}


def setup():
    font = 'Microsoft YaHei' if Path('C:/Windows/Fonts/msyh.ttc').exists() else 'WenQuanYi Zen Hei'
    path = font_manager.findfont(font, fallback_to_default=False)
    plt.rcParams.update({'font.family': font, 'font.size': 10, 'axes.unicode_minus': False,
        'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'savefig.dpi': 300})
    return path


def save(fig, output, name, audit, font):
    # 按用户约定采用字体覆盖和边界检查，不调用图像视觉接口。
    fig.canvas.draw()
    glyphs = FT2Font(font).get_charmap()
    hidden_ticks = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lower, upper = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lower <= tick.get_loc() <= upper:
                    hidden_ticks.update((tick.label1, tick.label2))
    # 轴范围外的刻度不参与实际绘制，导出边界检查只覆盖可见文字。
    texts = [text for text in fig.findobj(matplotlib.text.Text) if text not in hidden_ticks and text.get_visible()]
    labels = [text.get_text() for text in texts]
    missing = sorted({c for label in labels for c in label if not c.isspace() and ord(c) not in glyphs})
    if missing:
        raise ValueError('字体缺字：' + ''.join(missing))
    renderer = fig.canvas.get_renderer()
    box = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
    for text in texts:
        if text.get_visible() and text.get_text():
            extent = text.get_window_extent(renderer)
            if extent.x0 < box.x0 - 2 or extent.x1 > box.x1 + 2 or extent.y0 < box.y0 - 2 or extent.y1 > box.y1 + 2:
                raise ValueError('文本超出导出边界：' + text.get_text())
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'{name}.{suffix}', bbox_inches='tight', dpi=300)
    audit.append(dict(figure=name, font=font, missing_glyphs=missing, text_bounds='passed', png_dpi=300,
                      validation_method='font cmap and renderer text bounds; no image vision'))
    plt.close(fig)


def plot(output):
    font, audit = setup(), []
    day = pd.read_csv(output / 'typical-day.csv').query('reference_it_mw == 10')
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), layout='constrained', sharex=True)
    for col, (key, data) in enumerate(day.groupby('server_type', sort=True)):
        for row, (metric, label) in enumerate([('it_mw', 'IT 功率（MW）'), ('facility_mw', '设施功率（MW）')]):
            ax = axes[row, col]
            for version, color, style, legend in [('v3', '#777777', '--', 'v3'), ('diurnal', '#177E78', '-', '请求形状映射')]:
                values = data.loc[data.version.eq(version)]
                ax.plot(values.hour, values[metric], color=color, ls=style, lw=2, label=legend)
            ax.set_title(data.scenario_zh.iloc[0], fontsize=10)
            ax.set_ylabel(label)
            ax.set_xlabel('目标 UTC 小时（请求时钟相位为情境设定）')
            ax.set_xticks([0, 6, 12, 18, 23])
            ax.grid(alpha=.15)
            if col == 0:
                ax.legend()
    save(fig, output, 'figure1-typical-day', audit, font)

    shapes = pd.read_csv(output / 'request-shape-24h.csv').query("zero_policy == 'retained_zero'")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout='constrained')
    for ax, metric, title in zip(axes, ('requests', 'input_tokens', 'output_tokens'), ('请求数量', '输入 token', '输出 token')):
        for source in COLORS:
            values = shapes.loc[shapes.source.eq(source) & shapes.metric.eq(metric)]
            ax.plot(values.hour, values.normalized, label=NAMES[source], color=COLORS[source], lw=1.8)
        ax.set_title(title)
        ax.set_xlabel('数据源小时\nBurstGPT：当地相对时钟；Azure：UTC', fontsize=9)
        ax.set_ylabel('负载 / 本来源小时均值')
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=9)
    save(fig, output, 'figure2-request-shapes', audit, font)

    sensitivity = pd.read_csv(output / 'sensitivity-typical-day.csv')
    stats = pd.read_csv(output / 'sensitivity-summary.csv')
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), layout='constrained')
    for col, (key, data) in enumerate(sensitivity.groupby('server_type', sort=True)):
        ax = axes[0, col]
        for variant, label, color in [('power-low', '低波动', '#6B91B2'), ('central', '中央', '#177E78'), ('power-high', '高波动', '#BB5929')]:
            values = data.loc[data.sensitivity.eq(variant)]
            ax.plot(values.hour, values.it_mw, color=color, label=label)
        ax.set_title(data.scenario_zh.iloc[0], fontsize=10)
        ax.set_xlabel('目标 UTC 小时')
        ax.set_ylabel('IT 功率（MW）')
        ax.set_xticks([0, 6, 12, 18, 23])
        if col == 0:
            ax.legend()
        ax = axes[1, col]
        selected = stats.loc[stats.server_type.eq(key)]
        central = selected.loc[selected.sensitivity.eq('central')].iloc[0]
        names = [('功率端点与曲率', ['power-low', 'central', 'power-high']),
                 ('输入贡献20–80%', ['input-share-20pct', 'central', 'input-share-80pct']),
                 ('Azure两类日形状', ['azure-2024-code-daily', 'azure-2024-conv-daily']),
                 ('Burst七个周相位', [f'burst-weekly-day0-{i}' for i in range(7)]),
                 ('排除含零观测日', ['positive-days'])]
        for index, (_, nameset) in enumerate(names):
            values = selected.loc[selected.sensitivity.isin(nameset)].it_diurnal_peak_valley
            ax.plot([values.min(), values.max()], [index, index], lw=4, color='#177E78')
            ax.scatter(values, [index] * len(values), s=14, color='#333333')
        ax.axvline(central.it_diurnal_peak_valley, color='#777777', ls='--')
        ax.set_yticks(range(len(names)), [x[0] for x in names], fontsize=9)
        ax.set_xlabel('IT 典型日峰谷比')
        ax.grid(axis='x', alpha=.15)
    save(fig, output, 'figure3-sensitivity', audit, font)

    weeks = pd.read_csv(output / 'request-weekday-weekend.csv').query("zero_policy == 'retained_zero'")
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), layout='constrained')
    for col, source in enumerate(COLORS):
        for row, metric in enumerate(('requests', 'output_tokens')):
            ax = axes[row, col]
            data = weeks.loc[weeks.source.eq(source) & weeks.metric.eq(metric)]
            for label, zh, color in [('weekday', '工作日', '#267C9E'), ('weekend', '周末', '#BB5929')]:
                rows = data.loc[data.day_type.eq(label)]
                if source == 'burst':
                    grouped = rows.groupby('hour').normalized
                    ax.fill_between(grouped.mean().index, grouped.min(), grouped.max(), color=color, alpha=.2)
                    ax.plot(grouped.mean(), color=color, label=zh + '：七个相位范围')
                else:
                    ax.plot(rows.hour, rows.normalized, color=color, label=zh)
            ax.set_title(NAMES[source] + ('：星期未知' if source == 'burst' else '：UTC日历'), fontsize=10)
            ax.set_ylabel(('请求数量' if metric == 'requests' else '输出 token') + ' / 全窗口均值')
            ax.set_xlabel('数据源小时')
            ax.set_xticks([0, 6, 12, 18, 23])
            ax.set_ylim(bottom=0)
            if row == 0:
                ax.legend(fontsize=8)
    save(fig, output, 'figure4-weekday-weekend', audit, font)
    (output / 'figure-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    plot(parser.parse_args().output)
