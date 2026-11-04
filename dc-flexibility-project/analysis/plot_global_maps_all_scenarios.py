"""全球柔性收益地图：全部 11 个情境（一个边界、一个匹配目标一幅图）。

在服务器运行（需要运行目录中的国界与 2° 区块陆地缓存），例如：
  venv/bin/python project/analysis/plot_global_maps_all_scenarios.py --output report-full --boundary planning
版式：首行为纯 CPU 型的科研计算与云通用计算，其余三行依次为科研 AI、大模型训练/开发、推理服务，
三列为 CPU 主导型（B）、均衡型（C）、GPU 主导型（D）。同一幅图共用色标，上限取全部子图的 98 分位。
输出 <output>/figures/all-scenarios-<boundary>-<目标>.{png,pdf,json}。
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from global_flex_inputs import digest, settings, utc_now, write_json
from report_global_flex_econ import map_cells

LAYOUT = [[('A-HPC', '纯 CPU 型·科研计算'), ('A-CLD', '纯 CPU 型·云通用计算'), None],
          [('B-SAI', 'CPU 主导型·科研 AI'), ('C-SAI', '均衡型·科研 AI'), ('D-SAI', 'GPU 主导型·科研 AI')],
          [('B-TRN', 'CPU 主导型·大模型训练/开发'), ('C-TRN', '均衡型·大模型训练/开发'), ('D-TRN', 'GPU 主导型·大模型训练/开发')],
          [('B-INF', 'CPU 主导型·推理服务'), ('C-INF', '均衡型·推理服务'), ('D-INF', 'GPU 主导型·推理服务')]]
BOUNDARIES = {'planning': '规划边界', 'fixed': '已建成边界'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('global-flex-econ-20261103.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--boundary', choices=list(BOUNDARIES), default='planning')
    parser.add_argument('--alpha', type=float, help='省略时为经济最优')
    args = parser.parse_args()

    _, paths = settings(args.config)
    run = paths['run_root']
    os.environ.setdefault('MPLCONFIGDIR', str(run / 'matplotlib-cache'))
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
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

    mpl.rcParams.update({'font.family': 'WenQuanYi Zen Hei', 'font.size': 9, 'axes.unicode_minus': False,
                         'pdf.fonttype': 42, 'savefig.dpi': 300})
    values = valid.savings_pct.clip(lower=0)
    top = float(values.quantile(.98))
    step = 5. if top > 10 else (1. if top > 1 else 10 ** np.floor(np.log10(max(top, 1e-6))))
    vmax = max(step, float(np.ceil(top / step) * step))
    norm, cmap = mpl.colors.Normalize(0, vmax), mpl.colormaps['viridis']
    borders = countries.geometry.simplify(.08, preserve_topology=True).to_list()
    alpha_label = '经济最优' if args.alpha is None else f'α = {args.alpha * 100:.0f}%'

    fig, axes = plt.subplots(4, 3, figsize=(13.2, 10.4), subplot_kw={'projection': ccrs.Robinson()})
    panels = []
    for r, row in enumerate(LAYOUT):
        for c, item in enumerate(row):
            ax = axes[r, c]
            if item is None:
                ax.remove()
                continue
            scenario, label = item
            current = cells.merge(valid.loc[valid.scenario.eq(scenario), ['grid_id', 'savings_pct']], on='grid_id', how='inner', validate='one_to_one')
            drawn = current.loc[~current.geometry.is_empty]
            ax.set_global()
            fplt.add_geometries(ax, borders, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='#a8a8a8', linewidth=.2, zorder=1)
            fplt.add_geometries(ax, drawn.geometry.to_list(), crs=ccrs.PlateCarree(), facecolor=cmap(norm(drawn.savings_pct.clip(lower=0).to_numpy())),
                                edgecolor='none', linewidth=0, zorder=2, rasterized=True)
            median = float(current.savings_pct.median())
            ax.set_title(f'{scenario}  {label}\n中位 {median:.1f}%', fontsize=8.5, pad=3)
            panels.append({'scenario': scenario, 'grids': int(len(current)), 'drawn_blocks': int(len(drawn)), 'median_pct': round(median, 3),
                           'p25_pct': round(float(current.savings_pct.quantile(.25)), 3), 'p75_pct': round(float(current.savings_pct.quantile(.75)), 3)})
    title = f'{BOUNDARIES[args.boundary]}，{alpha_label}：全部 11 个情境的柔性收益'
    fig.suptitle(title, fontsize=12, y=.995)
    fig.subplots_adjust(left=.01, right=.99, top=.925, bottom=.015, hspace=.2, wspace=.02)
    cax = fig.add_axes([.71, .875, .24, .016])
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal',
                       extend='max' if float(values.max()) > vmax else 'neither')
    bar.set_label('柔性节省 / 无柔性能源系统成本（%）', fontsize=8.5)
    bar.ax.tick_params(labelsize=8)
    fig.text(.83, .825, '列：CPU 主导型（2∶1）／均衡型（2∶2）／GPU 主导型（2∶8）\n行：科研 AI／大模型训练/开发／推理服务\n'
             '色标上限取全部子图 98 分位；每个 2° 区块\n按其代表 0.5° 网格结果整体着色（裁到陆地）。',
             ha='center', va='top', fontsize=7.8, color='#404040', linespacing=1.5)
    token = 'economic' if args.alpha is None else f'alpha{args.alpha * 100:.0f}'
    stem = args.output / 'figures' / f'all-scenarios-{args.boundary}-{token}'
    fig.savefig(stem.with_suffix('.png'), dpi=300, facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
    plt.close(fig)
    write_json(stem.with_suffix('.json'), {'utc': utc_now(), 'title': title, 'boundary': args.boundary, 'alpha': args.alpha,
        'color_min_pct': 0, 'color_max_pct': vmax, 'colormap': 'viridis', 'renderer_sha256': digest(Path(__file__)),
        'data_sha256': digest(table), 'panels': panels})
    print(json.dumps(panels, ensure_ascii=False))


if __name__ == '__main__':
    main()
