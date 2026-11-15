"""论文《算力中心算电协同收益稳定但有限》结果图（英文版）的公共设置与数据读取。

金额统一为“年收益或年节省 / 数据中心年用电量（不卖推理时的基准用电）”，单位 USD/MWh；
窗口内的节省与窗口内的基准用电直接相除，与年化无关。
"""
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'papers/demand-flexibility/figures-stable-limited'
SOURCE = OUT / 'source-data'
INPUT = ROOT / 'results/paper-figs-stable-limited-20261115'
TEMP = ROOT / '.codex-temp/figs-en'
RESIDUAL = ROOT / 'results/residual-flex-global-20261115'
SELL = ROOT / 'results/sell-inference-tiers-20261115'
SECTION1 = ROOT / 'results/global-flex-econ-20261103'

SCENARIOS = ['A-HPC', 'A-CLD', 'B-SAI', 'C-SAI', 'D-SAI', 'B-TRN', 'C-TRN', 'D-TRN', 'B-INF', 'C-INF', 'D-INF']
GPU_SCENARIOS = ['B-SAI', 'C-SAI', 'D-SAI', 'B-TRN', 'C-TRN', 'D-TRN', 'B-INF', 'C-INF', 'D-INF']
REPRESENTATIVES = ['A-HPC', 'A-CLD', 'D-SAI', 'D-TRN', 'D-INF']
USES = {'HPC': 'Scientific computing (CPU)', 'CLD': 'Cloud computing (CPU)',
        'SAI': 'Scientific AI', 'TRN': 'LLM training', 'INF': 'Inference'}
SHORT = {'HPC': 'HPC', 'CLD': 'Cloud', 'SAI': 'Sci. AI', 'TRN': 'Training', 'INF': 'Inference'}
SPECS = {'A': 'CPU-only', 'B': 'CPU-heavy', 'C': 'Balanced', 'D': 'GPU-heavy'}
# 与 PR #63/#70 论文图一致的五类用途颜色
COLORS = {'HPC': '#7B61A8', 'CLD': '#C18B22', 'SAI': '#0072B2', 'TRN': '#009E73', 'INF': '#D55E00'}
LAND = '#E4E4E4'
BORDER = '#A6A6A6'
NOTE = '#555555'
EXTENT = (-180, 180, -58, 84)


def use(scenario):
    return scenario.split('-')[1]


def label(scenario, short=False):
    spec, u = scenario.split('-')
    if short:
        return SHORT[u] if spec == 'A' else f'{SHORT[u]} · {spec}'
    return USES[u] if spec == 'A' else f'{USES[u]} · {SPECS[spec]}'


def color(scenario):
    return COLORS[use(scenario)]


def shade(scenario):
    fraction = {'A': 0, 'B': .47, 'C': .23, 'D': 0}[scenario[0]]
    return tuple(np.array(mpl.colors.to_rgb(color(scenario))) * (1 - fraction) + fraction)


def setup():
    for directory in (OUT, SOURCE, TEMP):
        directory.mkdir(parents=True, exist_ok=True)
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    family = next(name for name in ['Arial', 'Helvetica', 'DejaVu Sans'] if name in fonts)
    mpl.rcParams.update({
        'font.family': family, 'font.size': 6.5, 'axes.labelsize': 6.5,
        'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6, 'legend.frameon': False,
        'axes.linewidth': .5, 'lines.linewidth': 1, 'patch.linewidth': .5,
        'xtick.major.width': .5, 'ytick.major.width': .5, 'xtick.minor.width': .4, 'ytick.minor.width': .4,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 600,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })
    return family


def panel(fig, letter, x, y):
    fig.text(x, y, letter, fontsize=8, fontweight='bold', va='top', ha='left')


GROUP = {'HPC': 'Sci. computing', 'CLD': 'Cloud', 'SAI': 'Scientific AI', 'TRN': 'LLM training', 'INF': 'Inference'}
INNER = {'A': 'CPU-only', 'B': 'CPU-heavy', 'C': 'Balanced', 'D': 'GPU-heavy'}


def scenario_axis(ax, scenarios, pos, axis='y', outer=-.42, fontsize=5.8):
    """两级情境标签：内层为配置（CPU-heavy 等），外层为用途（Scientific AI 等），外层带括线。"""
    import matplotlib.transforms as mtransforms
    inner = [INNER[s[0]] for s in scenarios]
    if axis == 'y':
        ax.set_yticks(pos, inner, fontsize=fontsize)
        ax.tick_params(axis='y', length=0, pad=2)
        tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    else:
        ax.set_xticks(pos, inner, fontsize=fontsize, rotation=90)
        ax.tick_params(axis='x', length=0, pad=2)
        tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    groups = []
    for s, p in zip(scenarios, pos):
        u = use(s)
        if groups and groups[-1][0] == u:
            groups[-1][1].append(p)
        else:
            groups.append((u, [p]))
    for u, ps in groups:
        lo, hi, mid = min(ps), max(ps), (min(ps) + max(ps)) / 2
        if axis == 'y':
            ax.text(outer, mid, GROUP[u], transform=tr, ha='right', va='center', fontsize=fontsize + .3,
                    fontweight='bold', color=COLORS[u], clip_on=False)
            ax.plot([outer + .025] * 2, [lo - .3, hi + .3], transform=tr, color=COLORS[u], lw=.8, clip_on=False,
                    solid_capstyle='butt')
        else:
            ax.text(mid, outer, GROUP[u], transform=tr, ha='center', va='top', fontsize=fontsize + .3,
                    fontweight='bold', color=COLORS[u], clip_on=False)
            ax.plot([lo - .3, hi + .3], [outer + .03] * 2, transform=tr, color=COLORS[u], lw=.8, clip_on=False,
                    solid_capstyle='butt')


def csv(frame, name):
    frame.to_csv(SOURCE / f'{name}.csv', index=False, float_format='%.10g', encoding='utf-8-sig')


def export(fig, name):
    # 不裁切画布，导出宽度固定为 180 mm；地图层在 PDF 中栅格化，其余保持矢量。
    fig.savefig(OUT / f'{name}.pdf', dpi=600)
    fig.savefig(OUT / f'{name}.png', dpi=600)
    fig.savefig(TEMP / f'{name}-preview.png', dpi=200)
    plt.close(fig)


def guides(ax, axis='x'):
    ax.grid(axis=axis, color='#E6E6E6', lw=.4, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- 数据

def baseline():
    """不卖推理（I=0）、顺序调度下逐网格的窗口基准用电与供能成本。"""
    b = pd.read_parquet(INPUT / 'baseline-energy.parquet')
    assert len(b) == 4842 * 11 and not b.duplicated(['grid_id', 'scenario']).any()
    return b


def paired():
    """第三节配对结果（seq/joint × 强度），附加 USD/MWh 指标。"""
    p = pd.read_parquet(RESIDUAL / 'paired.parquet')
    p = p.merge(baseline()[['grid_id', 'scenario', 'facility_mwh0', 'supply_cost0_usd']],
                on=['grid_id', 'scenario'], validate='many_to_one')
    assert len(p) == 213048 and (p.V_usd > 0).all()
    p['saving_usd_mwh'] = p.V_usd / p.facility_mwh0
    p['wind_share'] = p.wind_mw_joint_I0 / (p.wind_mw_joint_I0 + p.pv_mw_joint_I0 + 1e-9)
    return p


def fleet_annual_usd():
    """10 MW 参考机群年化设备费用（PR #60 part1-summary，按 analysis/flex-econ-copt-20261102.json 卡价计算）。"""
    s = pd.read_csv(INPUT / 'part1-summary.csv', encoding='utf-8-sig').set_index('scenario')
    return s.fleet_annual_musd * 1e6


def part1_summary():
    return pd.read_csv(INPUT / 'part1-summary.csv', encoding='utf-8-sig').set_index('scenario')


def sell():
    s = pd.read_parquet(SELL / 'sell-inference-global.parquet')
    b = baseline()[['grid_id', 'scenario', 'facility_mwh0', 'hours0']]
    s = s.merge(b, on=['grid_id', 'scenario'], validate='many_to_one')
    annual_mwh = s.facility_mwh0 * 8760 / s.hours0
    s['net_usd_mwh'] = s.net_usd_year / annual_mwh
    s['net_tou_usd_mwh'] = s.net_tou_usd_year / annual_mwh
    s['coord_usd_mwh'] = s.coord_saving_usd_year / annual_mwh
    assert s.groupby(['scenario', 'tier']).size().eq(4842).all()
    return s


# ---------------------------------------------------------------- 地图

_CELLS = None
_COUNTRIES = None


def map_layers():
    global _CELLS, _COUNTRIES
    if _CELLS is None:
        _CELLS = gpd.read_parquet(TEMP / 'map-assets/map-block-land-cells.parquet')
        _COUNTRIES = gpd.read_file('zip://' + str(TEMP / 'map-assets/countries-10m.zip'))
        _COUNTRIES = _COUNTRIES.loc[_COUNTRIES.geometry.bounds.maxy > EXTENT[2]]
        _COUNTRIES = _COUNTRIES.assign(geometry=_COUNTRIES.geometry.simplify(.05))
    return _CELLS, _COUNTRIES


def draw_map(fig, rect, values, cmap=None, norm=None, projection=None, colors=None, lon_labels=True):
    """2° 代表区块（裁切到陆地）着色；values 为以 grid_id 为索引的 Series。"""
    cells, countries = map_layers()
    ax = fig.add_axes(rect, projection=projection or ccrs.Robinson())
    ax.set_anchor('NW')
    if projection is None or isinstance(projection, ccrs.Robinson):
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    else:
        ax.set_global()
    cells = cells.merge(values.rename('value'), left_on='grid_id', right_index=True, validate='one_to_one')
    cells = cells.loc[~cells.geometry.is_empty & cells.value.notna()]
    pc = ccrs.PlateCarree()
    ax.add_geometries(countries.geometry, crs=pc, facecolor=LAND, edgecolor='none', zorder=0, rasterized=True)
    face = colors.reindex(cells.grid_id).tolist() if colors is not None else cmap(norm(cells.value.to_numpy()))
    ax.add_geometries(cells.geometry, crs=pc, facecolor=face, edgecolor='none',
                      linewidth=0, antialiased=False, zorder=1, rasterized=True)
    ax.add_geometries(countries.geometry, crs=pc, facecolor='none', edgecolor=BORDER, linewidth=.15, zorder=2,
                      rasterized=True)
    gl = ax.gridlines(crs=pc, draw_labels=False, linewidth=.3, color='#CFCFCF', linestyle=(0, (2, 2)),
                      xlocs=range(-180, 181, 60), ylocs=[-30, 0, 30, 60], zorder=0.5)
    ax.spines['geo'].set_linewidth(.4)
    ax.spines['geo'].set_edgecolor('#8C8C8C')
    for lon in ((-120, -60, 0, 60, 120) if lon_labels else ()):
        x, y = ax.projection.transform_point(lon, EXTENT[2], pc)
        ax.text(x, y, f'{abs(lon)}°{"W" if lon < 0 else "E" if lon > 0 else ""}', ha='center', va='top',
                fontsize=5, color=NOTE, transform=ax.transData)
    return ax, len(cells)


def robinson_y(lat):
    return np.array([ccrs.Robinson().transform_point(0, float(v), ccrs.PlateCarree())[1] for v in np.atleast_1d(lat)])


def latitude_strip(fig, ax_map, width, frames, xlabel, xlim=None, gap=.006, band=4, smooth=1):
    """与地图纬度精确对齐的纬度统计条：frames 为 [(label, color, DataFrame[lat,value], shade_iqr)]。"""
    fig.canvas.draw()
    pos = ax_map.get_position()
    ax = fig.add_axes([pos.x1 + gap, pos.y0, width, pos.height])
    y0, y1 = ax_map.get_ylim()
    ax.set_ylim(y0, y1)
    edges = np.arange(-60, 86, band)
    mids = (edges[:-1] + edges[1:]) / 2
    rows = []
    for name, col, frame, iqr in frames:
        cut = pd.cut(frame.lat, edges, labels=mids)
        g = frame.groupby(cut, observed=False).value
        q = pd.DataFrame({'q25': g.quantile(.25), 'median': g.median(), 'q75': g.quantile(.75), 'n': g.size()})
        q = q.loc[q.n >= 3]
        if smooth > 1:
            q[['q25', 'median', 'q75']] = q[['q25', 'median', 'q75']].rolling(smooth, center=True, min_periods=1).mean()
        yy = robinson_y(q.index.astype(float))
        if iqr:
            ax.fill_betweenx(yy, q.q25, q.q75, color=col, alpha=.18, lw=0)
        ax.plot(q['median'], yy, color=col, lw=.9, label=name)
        rows.append(q.reset_index(names='lat_mid').assign(series=name))
    ticks = [-30, 0, 30, 60]
    ax.set_yticks(robinson_y(ticks), [f'{abs(t)}°{"S" if t < 0 else "N" if t > 0 else ""}' for t in ticks])
    ax.yaxis.tick_right()
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.tick_params(axis='y', labelsize=5, pad=1)
    ax.tick_params(axis='x', labelsize=5)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=5.5, labelpad=1.5)
    ax.grid(axis='x', color='#EBEBEB', lw=.4)
    for t in ticks:
        ax.axhline(robinson_y(t)[0], color='#E3E3E3', lw=.3, ls=(0, (2, 2)), zorder=0)
    return ax, pd.concat(rows, ignore_index=True)


def truncated(name, lo=.12, hi=1.0, n=256):
    base = mpl.colormaps[name]
    return mpl.colors.LinearSegmentedColormap.from_list(f'{name}_trunc', base(np.linspace(lo, hi, n)))


def colorbar(fig, rect, cmap, norm, ticks, text, extend='neither'):
    cax = fig.add_axes(rect)
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal',
                       ticks=ticks, extend=extend, extendfrac=.04)
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor('face')
    bar.outline.set_linewidth(.35)
    cax.tick_params(labelsize=5.5, length=1.8, width=.4, pad=1)
    bar.set_label(text, fontsize=6, labelpad=2)
    return bar


def land_axes(fig, rect, projection=None):
    """只画陆地底图与国界的地图轴（供六边形格网等叠加）。"""
    _, countries = map_layers()
    ax = fig.add_axes(rect, projection=projection or ccrs.Robinson())
    ax.set_anchor('NW')
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    pc = ccrs.PlateCarree()
    ax.add_geometries(countries.geometry, crs=pc, facecolor='#F0F0F0', edgecolor='none', zorder=0, rasterized=True)
    ax.add_geometries(countries.geometry, crs=pc, facecolor='none', edgecolor='#9A9A9A', linewidth=.2, zorder=3,
                      rasterized=True)
    ax.spines['geo'].set_linewidth(.4)
    ax.spines['geo'].set_edgecolor('#8C8C8C')
    return ax


def longitude_strip(fig, ax_map, height, frames, ylabel, ylim=None, gap=.004, band=6):
    """地图上方与经度对齐的经度统计条（Robinson 赤道处经度线性）。"""
    fig.canvas.draw()
    pos = ax_map.get_position()
    ax = fig.add_axes([pos.x0, pos.y1 + gap, pos.width, height])
    x0, x1 = ax_map.get_xlim()
    ax.set_xlim(x0, x1)
    edges = np.arange(-180, 181, band)
    mids = (edges[:-1] + edges[1:]) / 2
    xs = np.array([ccrs.Robinson().transform_point(float(m), 0, ccrs.PlateCarree())[0] for m in mids])
    for name, col, frame, iqr in frames:
        cut = pd.cut(frame.lon, edges, labels=mids)
        g = frame.groupby(cut, observed=False).value
        q = pd.DataFrame({'q25': g.quantile(.25), 'median': g.median(), 'q75': g.quantile(.75), 'n': g.size()})
        ok = (q.n >= 3).to_numpy()
        if iqr:
            ax.fill_between(xs[ok], q.q25[ok], q.q75[ok], color=col, alpha=.18, lw=0)
        ax.plot(xs[ok], q['median'][ok], color=col, lw=.9)
    ax.set_xticks([])
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='y', labelsize=5)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=5.5, labelpad=1)
    ax.grid(axis='y', color='#EBEBEB', lw=.4)
    return ax
