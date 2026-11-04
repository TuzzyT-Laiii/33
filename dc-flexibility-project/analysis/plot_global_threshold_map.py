"""全球网格：柔性收益是否超过阈值（默认 3%）的分类地图。

在服务器运行（需要运行目录中的国界与 2° 区块陆地缓存），例如：
  venv/bin/python project/analysis/plot_global_threshold_map.py --output report-full --boundary fixed --threshold 3
输出写入 <output>/figures/threshold-<boundary>-<目标>-<阈值>pct.{png,pdf} 及同名 .json。
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

from global_flex_inputs import digest, settings, utc_now, write_json
from report_global_flex_econ import map_cells
from plot_global_maps_all_scenarios import LAYOUT

SCENARIOS = [('D-TRN', 'GPU 主导型·大模型训练/开发'), ('D-SAI', 'GPU 主导型·科研 AI'),
             ('A-HPC', '纯 CPU 型·科研计算'), ('D-INF', 'GPU 主导型·推理服务')]
BOUNDARIES = {'planning': '规划边界', 'fixed': '已建成边界'}
ABOVE, BELOW = '#1f7a5a', '#e8b04a'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('global-flex-econ-20261103.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--boundary', choices=list(BOUNDARIES), default='fixed')
    parser.add_argument('--alpha', type=float, help='省略时为经济最优')
    parser.add_argument('--threshold', type=float, default=3.)
    parser.add_argument('--all-scenarios', action='store_true', help='画全部 11 个情境（4×3 版式），否则为 4 个代表情境')
    args = parser.parse_args()

    _, paths = settings(args.config)
    run = paths['run_root']
    os.environ.setdefault('MPLCONFIGDIR', str(run / 'matplotlib-cache'))
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import cartopy.crs as ccrs
    import frykit.plot as fplt

    table = args.output / 'grid-summary.parquet'
    manifest = json.loads((args.output / 'summary-manifest.json').read_text('utf-8'))
    if digest(table) != manifest['artifacts']['grid-summary.parquet']['sha256']:
        raise ValueError('地图输入与已验证汇总表的SHA不一致')
    pairs = pd.read_parquet(table)
    cells, countries = map_cells(run, pairs)
    valid = pairs.loc[pairs.boundary.eq(args.boundary) & pairs.baseline_status.eq('optimal') & pairs.flex_status.eq('optimal')]
    valid = valid.loc[valid.alpha.isna() if args.alpha is None else valid.alpha.eq(args.alpha)]

    font_family = 'WenQuanYi Zen Hei'
    mpl.rcParams.update({'font.family': font_family, 'font.size': 9, 'axes.unicode_minus': False,
                         'pdf.fonttype': 42, 'savefig.dpi': 300})
    alpha_label = '经济最优' if args.alpha is None else f'α = {args.alpha * 100:.0f}%'
    borders = countries.geometry.simplify(.08, preserve_topology=True).to_list()
    if args.all_scenarios:
        fig, grid = plt.subplots(4, 3, figsize=(13.2, 10.4), subplot_kw={'projection': ccrs.Robinson()})
        slots = [(grid[r, c], item) for r, row in enumerate(LAYOUT) for c, item in enumerate(row)]
        for ax, item in slots:
            if item is None:
                ax.remove()
        slots = [(ax, item) for ax, item in slots if item is not None]
    else:
        fig, grid = plt.subplots(2, 2, figsize=(12.6, 6.8), subplot_kw={'projection': ccrs.Robinson()})
        slots = list(zip(grid.flat, SCENARIOS))
    counts = []
    for number, (ax, (scenario, label)) in enumerate(slots):
        current = cells.merge(valid.loc[valid.scenario.eq(scenario), ['grid_id', 'savings_pct']], on='grid_id', how='inner', validate='one_to_one')
        above = current.savings_pct > args.threshold
        share = float(above.mean() * 100)
        drawn = current.loc[~current.geometry.is_empty]
        ax.set_global()
        fplt.add_geometries(ax, borders, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='#a8a8a8', linewidth=.22, zorder=1)
        colors = [ABOVE if flag else BELOW for flag in (drawn.savings_pct > args.threshold)]
        fplt.add_geometries(ax, drawn.geometry.to_list(), crs=ccrs.PlateCarree(), facecolor=colors,
                            edgecolor='none', linewidth=0, zorder=2, rasterized=True)
        if args.all_scenarios:
            ax.set_title(f'{scenario}  {label}\n{share:.0f}% 网格超过 {args.threshold:g}%', fontsize=8.5, pad=3)
        else:
            ax.set_title(f'({chr(97 + number)}) {scenario}  {label}：{share:.0f}% 网格超过 {args.threshold:g}%', fontsize=9, pad=4)
        counts.append({'scenario': scenario, 'grids': int(len(current)), 'above': int(above.sum()), 'share_pct': round(share, 2),
                       'drawn_blocks': int(len(drawn))})
    title = f'{BOUNDARIES[args.boundary]}，{alpha_label}：柔性收益是否超过 {args.threshold:g}%'
    handles = [Patch(facecolor=ABOVE, label=f'> {args.threshold:g}%'), Patch(facecolor=BELOW, label=f'≤ {args.threshold:g}%')]
    note = '柔性收益 = 同参数无柔性能源系统成本的节省百分比。每个 2° 区块按其代表 0.5° 网格的结果整体着色（裁到陆地）；空白为无有效网格的区块。'
    if args.all_scenarios:
        fig.suptitle(title + '（全部 11 个情境）', fontsize=12, y=.995)
        fig.subplots_adjust(left=.01, right=.99, top=.925, bottom=.015, hspace=.2, wspace=.02)
        fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.83, .885), fontsize=10)
        fig.text(.83, .835, '列：CPU 主导型（2∶1）／均衡型（2∶2）／GPU 主导型（2∶8）\n行：科研 AI／大模型训练/开发／推理服务\n'
                 '柔性收益 = 同参数无柔性能源系统成本的节省百分比；\n每个 2° 区块按其代表 0.5° 网格结果整体着色（裁到陆地）。',
                 ha='center', va='top', fontsize=7.8, color='#404040', linespacing=1.5)
    else:
        fig.suptitle(title, fontsize=12, y=.99)
        fig.subplots_adjust(left=.015, right=.985, top=.91, bottom=.12, hspace=.12, wspace=.025)
        fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, .055), fontsize=10)
        fig.text(.5, .022, note, ha='center', va='bottom', fontsize=8)
    token = 'economic' if args.alpha is None else f'alpha{args.alpha * 100:.0f}'
    stem = args.output / 'figures' / f'threshold-{args.boundary}-{token}-{args.threshold:g}pct{"-all" if args.all_scenarios else ""}'
    fig.savefig(stem.with_suffix('.png'), dpi=300, facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
    plt.close(fig)
    write_json(stem.with_suffix('.json'), {'utc': utc_now(), 'title': title, 'boundary': args.boundary, 'alpha': args.alpha,
        'threshold_pct': args.threshold, 'rule': 'savings_pct > threshold；分母为各情境有效最优配对网格',
        'renderer_sha256': digest(Path(__file__)), 'data_sha256': digest(table), 'panels': counts})
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == '__main__':
    main()
