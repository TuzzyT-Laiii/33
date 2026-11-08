import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'papers/demand-flexibility/figures-zh'
SOURCE = OUT / 'source-data'
ENGLISH = ROOT / 'papers/demand-flexibility/figures'
CAPACITY = ROOT / 'results/minimum-server-capacity-20261108'
RESALE = ROOT / 'results/idle-gpu-resale-import-20261108'
TEMP = ROOT / '.codex-temp/paper-figures-zh'
SCENARIOS = ['A-HPC', 'A-CLD', 'B-SAI', 'C-SAI', 'D-SAI', 'B-TRN', 'C-TRN', 'D-TRN', 'B-INF', 'C-INF', 'D-INF']
REPRESENTATIVES = ['A-HPC', 'A-CLD', 'D-SAI', 'D-TRN', 'D-INF']
USES = {'HPC': '科研计算', 'CLD': '云通用计算', 'SAI': '科研AI', 'TRN': '大模型训练', 'INF': '推理服务'}
SPECS = {'A': '纯CPU型', 'B': 'CPU主导型', 'C': '均衡型', 'D': 'GPU主导型'}
COLORS = {'HPC': '#7B61A8', 'CLD': '#C18B22', 'SAI': '#0072B2', 'TRN': '#009E73', 'INF': '#D55E00'}
MARKERS = {'HPC': 's', 'CLD': 'D', 'SAI': '^', 'TRN': 'o', 'INF': 'v'}


def label(scenario):
    spec, use = scenario.split('-')
    return f'{SPECS[spec]}·{USES[use]}'


def use(scenario):
    return USES[scenario.split('-')[1]]


def color(scenario):
    return COLORS[scenario.split('-')[1]]


def shade(scenario):
    fraction = {'A': 0, 'B': .44, 'C': .22, 'D': 0}[scenario[0]]
    return tuple(np.array(mpl.colors.to_rgb(color(scenario))) * (1 - fraction) + fraction)


def marker(scenario):
    return MARKERS[scenario.split('-')[1]]


def setup():
    for directory in [OUT, SOURCE, TEMP]:
        directory.mkdir(parents=True, exist_ok=True)
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    chinese = next(f for f in ['Noto Sans CJK SC', 'Microsoft YaHei', 'Noto Sans SC'] if f in fonts)
    assert 'Arial' in fonts
    mpl.rcParams.update({
        'font.family': ['Arial', chinese], 'font.size': 6.5,
        'axes.labelsize': 6.5, 'axes.titlesize': 6.5,
        'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
        'legend.frameon': False, 'axes.linewidth': .5,
        'lines.linewidth': .9, 'patch.linewidth': .5,
        'xtick.major.width': .5, 'ytick.major.width': .5,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 600,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.unicode_minus': True,
    })
    return chinese


def panel(fig, letter, x, y, title=None, title_x=None):
    fig.text(x, y, letter, fontsize=8, fontweight='bold', va='top')
    if title:
        fig.text(title_x if title_x is not None else x + .035, y, title, fontsize=6.5, va='top')


def csv(frame, name):
    frame = frame.copy()
    if 'scenario' in frame:
        frame['情境'] = frame.scenario.map(label)
    frame.to_csv(SOURCE / f'{name}.csv', index=False, float_format='%.15g', encoding='utf-8-sig')


def read_panel(name):
    return pd.read_csv(ENGLISH / f'source-data/{name}.csv')


def capacity_main():
    table = pd.read_csv(CAPACITY / 'part2-main-table.csv')
    return table.loc[table.target.eq('economic')].set_index('scenario').loc[SCENARIOS].reset_index()


def guides(ax, axis='y'):
    ax.grid(axis=axis, color='#E6E6E6', lw=.45, zorder=0)
    ax.set_axisbelow(True)


def rows(ax, labels=False):
    y = np.arange(11)[::-1]
    ax.set_ylim(-.65, 10.65)
    ax.set_yticks(y, [label(s) for s in SCENARIOS] if labels else [])
    ax.tick_params(axis='y', length=0)
    for divider in [8.5, 5.5, 2.5]:
        ax.axhline(divider, color='#E6E6E6', lw=.45, zorder=0)
    if labels:
        for t, s in zip(ax.get_yticklabels(), SCENARIOS):
            t.set_color(color(s))
    return y


def export(fig, name, metadata):
    # 固定画布尺寸，保留 180 mm 的实际导出宽度。
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = []
    for t in fig.findobj(mpl.text.Text):
        if t.get_visible() and t.get_text():
            b = t.get_window_extent(renderer).transformed(fig.transFigure.inverted())
            texts.append({'text': t.get_text(), 'size_pt': t.get_fontsize(), 'box': list(b.bounds)})
    metadata.update({'width_mm': fig.get_figwidth() * 25.4, 'height_mm': fig.get_figheight() * 25.4,
                     'png_dpi': 600, 'latin_font': 'Arial', 'chinese_font': setup_font_name(),
                     'minimum_text_pt': min(t['size_pt'] for t in texts),
                     'map_rasterized_only': name == 'fig1'})
    (SOURCE / f'{name}_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (TEMP / f'{name}-text-boxes.json').write_text(json.dumps(texts, ensure_ascii=False, indent=2), encoding='utf-8')
    fig.savefig(OUT / f'{name}.pdf', dpi=600)
    fig.savefig(OUT / f'{name}.png', dpi=600)
    fig.savefig(TEMP / f'{name}-preview.png', dpi=200)
    plt.close(fig)
    print(f'{name}: PDF / PNG / CSV 已导出')


def setup_font_name():
    return mpl.rcParams['font.family'][1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
