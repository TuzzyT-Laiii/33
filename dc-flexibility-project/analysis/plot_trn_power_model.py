import argparse
import json
from pathlib import Path

import fitz
from fontTools.ttLib import TTFont
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trn_power_models import cpu_response


BLUE, ORANGE, GREEN, GRAY = '#2878B5', '#D66B27', '#268878', '#92999D'


def setup(font):
    path = font_manager.findfont(font, fallback_to_default=False)
    plt.rcParams.update({'font.family': font, 'font.size': 10, 'axes.unicode_minus': False,
        'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'savefig.dpi': 300})
    return path


def save(fig, output, name, font_path, audit):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    chars, outside = set(), []
    for item in fig.findobj(matplotlib.text.Text):
        if not item.get_visible() or not item.get_text():
            continue
        chars.update(item.get_text())
        box = item.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1):
            outside.append(item.get_text())
    font = TTFont(font_path, fontNumber=0)
    supported = set().union(*(table.cmap.keys() for table in font['cmap'].tables))
    missing = [char for char in chars if not char.isspace() and ord(char) not in supported]
    font.close()
    if missing or outside:
        raise ValueError(f'图件字体或边界检查失败：{name}，缺字={missing}，越界={outside}')
    fig.savefig(output / f'{name}.png', dpi=300)
    pdf = output / f'{name}.pdf'
    fig.savefig(pdf)
    with fitz.open(pdf) as doc:
        fonts = {row[0] for page in doc for row in page.get_fonts(full=True)}
        if not fonts or any(not doc.extract_font(xref)[3] for xref in fonts):
            raise ValueError('PDF 字体未嵌入')
        if not ''.join(page.get_text() for page in doc).strip():
            raise ValueError('PDF 文本为空')
    audit.append({'figure': name, 'missing_characters': [], 'text_outside_canvas': [],
                  'embedded_pdf_fonts': len(fonts), 'png_dpi': 300, 'verification': 'programmatic font coverage, text bounds and PDF font embedding; no visual inspection'})
    plt.close(fig)


def plot(project, font):
    output = project / 'results/trn-power-model-20261101'
    font_path, audit = setup(font), []
    days = pd.read_csv(output / 'typical-day.csv')
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), layout='constrained')
    for col, key in enumerate(('B', 'C', 'D')):
        rows = days.loc[days.server_type.eq(key) & days.use.eq('TRN') & days.reference_it_mw.eq(10)]
        for row, metric in enumerate(('it_mw', 'facility_mw')):
            ax = axes[row, col]
            for version, color, label, style in [('v3', GRAY, 'v3', '--'), ('new', BLUE, '修正模型', '-')]:
                part = rows.loc[rows.version.eq(version)].sort_values('hour_utc')
                ax.plot(part.hour_utc, part[metric], color=color, ls=style, lw=1.8, label=label)
            if row == 0:
                part = rows.loc[rows.version.eq('new')].sort_values('hour_utc')
                ax.plot(part.hour_utc, part.idle_it_mw, color=ORANGE, lw=1, ls=':', label='基础与空载 IT')
            ax.set(xlim=(0, 23), ylim=(0, 8), xticks=[0, 6, 12, 18, 23], xlabel='目标 UTC 小时',
                   ylabel=('IT' if row == 0 else '设施') + ' 功率（MW）', title=f'{key}-TRN · 10MW 参考规模')
            ax.grid(axis='y', alpha=.15)
            ax.legend(loc='lower right', fontsize=9)
    save(fig, output, 'figure1-trn-typical-day', font_path, audit)

    ratios = json.loads((project / 'results/typical-load-curves-20261029-v2/mit-power-ratio-summary.json').read_text('utf-8'))['transfer_distribution']
    mit = json.loads((project / 'results/typical-load-curves-20261101-v3/mit-calibration.json').read_text('utf-8'))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), layout='constrained')
    labels = ['v3 TRN 迁移假设', 'SAI：V100 实测均值', 'OLMo：H100 活跃训练', 'OLMo：H100 检查点', '本次 TRN 情境中央值']
    values = [ratios['p50'], mit['measured_power_to_tdp_ratio'], 600 / 700, 100 / 700, .80]
    colors = [GRAY, GREEN, ORANGE, ORANGE, BLUE]
    ax = axes[0]
    ax.barh(np.arange(5), values, color=colors, height=.6)
    ax.errorbar(.8, 4, xerr=[[.15], [.15]], fmt='none', ecolor=BLUE, capsize=5, lw=2)
    for y, value in enumerate(values):
        prefix = '>' if y == 2 else '约 ' if y == 3 else ''
        ax.text(value + .025, y, prefix + f'{value:.2f}', va='center', fontsize=9)
    ax.set(yticks=np.arange(5), yticklabels=labels, xlim=(0, 1.1), xlabel='GPU 功率 / 对应 TDP', title='(a) GPU 功率口径；横线为情境范围')
    ax.invert_yaxis()
    ax = axes[1]
    labels = ['HGX H100：实测空闲', 'HGX H100：微调中位数', 'MLPerf GPT-3：节点 DC', 'DGX H100：厂商最大值']
    values = [1.86, 7.92, 6.747, 10.2]
    ax.barh(np.arange(4), values, color=[GREEN, ORANGE, BLUE, GRAY], height=.6)
    for y, value in enumerate(values):
        ax.text(value + .15, y, f'{value:.2f}', va='center', fontsize=9)
    ax.set(yticks=np.arange(4), yticklabels=labels, xlim=(0, 11.7), xticks=np.arange(0, 11, 2), xlabel='八卡整机功率（kW）', title='(b) 整机参照；配置与计量边界不同')
    ax.invert_yaxis()
    save(fig, output, 'figure2-power-evidence', font_path, audit)

    sensitivity = pd.read_csv(output / 'sensitivity-summary.csv')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True, layout='constrained')
    order = ['gpu_low', 'central', 'gpu_high', 'idle_low', 'idle_high']
    labels = ['GPU 比例 0.65', 'GPU 比例 0.80', 'GPU 比例 0.95', '空闲功率低值', '空闲功率高值']
    for ax, key in zip(axes, ('B', 'C', 'D')):
        rows = sensitivity.loc[sensitivity.server_type.eq(key) & sensitivity.use.eq('TRN')].set_index('sensitivity').loc[order]
        ax.scatter(rows.mean_it_mw, np.arange(5), color=ORANGE, label='IT', s=36)
        ax.scatter(rows.mean_facility_mw, np.arange(5), color=BLUE, label='设施', s=36)
        ax.set(yticks=np.arange(5), yticklabels=labels, xlim=(0, 10), xlabel='年平均功率（MW）', title=f'{key}-TRN · 10MW')
        ax.grid(axis='y', alpha=.15)
        ax.legend(loc='lower right')
    axes[0].invert_yaxis()
    save(fig, output, 'figure3-sensitivity', font_path, audit)

    spec = pd.read_csv(output / 'specpower-evidence.csv')
    shape = json.loads((output / 'cpu-shape.json').read_text('utf-8'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for key, rows in spec.groupby('source_id'):
        axes[0].plot(rows.load * 100, rows.normalized_dynamic, 'o--', ms=3, lw=.9, alpha=.65, label=key)
    u = np.linspace(0, 1, 501)
    axes[0].plot(u * 100, cpu_response(u, shape), lw=2.2, color=BLUE, label='三份均值的单调平滑曲线')
    axes[0].set(xlim=(0, 100), ylim=(0, 1.05), yticks=np.arange(0, 1.01, .2), xlabel='负载（%）', ylabel='归一化动态功率', title='(a) 同型号或同代 SPECpower 整机响应')
    axes[0].legend(fontsize=8)
    old = spec.loc[spec.source_id.eq('spec-01377')]
    axes[1].plot(old.load * 100, old.normalized_dynamic * 410, 'o--', color=GRAY, label='v3：单份 SPEC 曲线')
    axes[1].plot(u * 100, cpu_response(u, shape) * 410, color=BLUE, label='修正：三份曲线的平滑结果')
    axes[1].axvspan(50, 60, color=ORANGE, alpha=.10)
    axes[1].set(xlim=(0, 100), ylim=(0, 435), xlabel='CPU 利用率代理（%）', ylabel='双路动态增量（W）', title='(b) A/B/C 型中央参数的曲线迁移')
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(axis='y', alpha=.15)
    save(fig, output, 'figure4-cpu-response', font_path, audit)

    acme = output / 'acme'
    if (acme / 'telemetry-run.json').exists() and json.loads((acme / 'telemetry-run.json').read_text('utf-8'))['complete']:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout='constrained')
        for ax, metric, label in zip(axes, ('POWER_USAGE', 'GPU_UTIL', 'FB_USED'), ('GPU 功率（W）', 'GPU 利用率（%）', '已用显存（MiB）')):
            for cluster, color in [('kalos', BLUE), ('seren', ORANGE)]:
                rows = pd.read_csv(acme / f'{cluster}-{metric}-diurnal.csv').sort_values('diurnal')
                ax.plot(rows.diurnal, rows['mean'], color=color, label=cluster, lw=1.8)
            ax.set(xlim=(0, 23), ylim=(0, None), xticks=[0, 6, 12, 18, 23], xlabel='源数据北京时间小时', ylabel=label)
            ax.set_yticks([tick for tick in ax.get_yticks() if 0 <= tick <= ax.get_ylim()[1]])
            ax.grid(axis='y', alpha=.15)
            ax.legend()
        fig.suptitle('Acme 原始监测的集群样本均值；未区分已分配与未分配 GPU')
        save(fig, output, 'figure5-acme-diurnal', font_path, audit)
    (output / 'figure-qa.json').write_text(json.dumps({'font': font, 'font_path': font_path, 'figures': audit}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--font', default='Microsoft YaHei')
    args = parser.parse_args()
    plot(args.project.resolve(), args.font)
