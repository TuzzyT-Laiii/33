"""第三节全球结果汇总与作图：配对差值、保留率分布、11 情境保留率地图。

输入：服务器 full/runs.jsonl 压缩成的 runs-compact.parquet 与 max-intensity.csv。
输出（--out）：paired.parquet（逐网格×情境×强度）、summary-retention.csv、summary-saving.csv、
retention-gt1.csv、drivers-spearman.csv、fig-retention-maps-75.png/pdf、fig-retention-curves.png/pdf。
"""
import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
import pandas as pd

ORDER = ['B-TRN', 'C-TRN', 'D-TRN', 'B-SAI', 'C-SAI', 'D-SAI', 'A-HPC', 'A-CLD', 'B-INF', 'C-INF', 'D-INF']
NAMES = {'TRN': 'Training', 'SAI': 'Research AI', 'HPC': 'HPC', 'CLD': 'Cloud', 'INF': 'Inference'}
KEY = ['grid_id', 'scenario', 'intensity_of_max']


def paired(runs):
    r = runs[runs.status == 'optimal']
    s = r[r['mode'] == 'seq'].set_index(KEY)
    j = r[r['mode'] == 'joint'].set_index(KEY)
    t = s[['lat', 'lon', 'iso3', 'country', 'continent', 'carbon_t_mwh', 'hours', 'intensity', 'max_intensity']].copy()
    t['seq_cost_usd'], t['joint_cost_usd'] = s.total_cost_usd, j.total_cost_usd
    t['V_usd'] = t.seq_cost_usd - t.joint_cost_usd
    t['V_pct_of_seq'] = t.V_usd / t.seq_cost_usd * 100
    v0 = t.xs(0.0, level='intensity_of_max').V_usd
    t['retention'] = t.V_usd / v0.reindex(t.index.droplevel('intensity_of_max')).to_numpy()
    j0 = j.xs(0.0, level='intensity_of_max')
    for col in ('cfe_hourly', 'wind_mw', 'pv_mw', 'battery_mwh', 'grid_cap_mw'):
        t[col + '_joint_I0'] = j0[col].reindex(t.index.droplevel('intensity_of_max')).to_numpy()
    return t.reset_index()


def maps(t, out, level=0.75):
    x = t[t.intensity_of_max == level]
    fig = plt.figure(figsize=(16, 12))
    norm = matplotlib.colors.Normalize(0, 1.1)
    cmap = plt.get_cmap('viridis')
    for k, sc in enumerate(ORDER):
        ax = fig.add_subplot(4, 3, k + 1, projection=ccrs.PlateCarree())
        ax.set_extent([-180, 180, -58, 80], crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.3, color='0.4')
        d = x[x.scenario == sc]
        # 每个抽样网格代表一个 2°×2° 块，整块着色
        blat, blon = np.floor(d.lat / 2) * 2, np.floor(d.lon / 2) * 2
        patches = [Rectangle((lo, la), 2, 2) for la, lo in zip(blat, blon)]
        pc = PatchCollection(patches, cmap=cmap, norm=norm, linewidth=0)
        pc.set_array(d.retention.clip(0, 1.1).to_numpy())
        ax.add_collection(pc)
        m = d.retention.median()
        ax.set_title(f"{sc} ({NAMES[sc.split('-')[1]]})  median {m:.2f}", fontsize=10)
    cax = fig.add_axes([0.70, 0.08, 0.22, 0.015])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal',
                 label=f'Retention of joint-dispatch value at {int(level * 100)}% of absorbable inference')
    fig.suptitle('Residual flexibility value under real inference demand (ASI shape), planning boundary, grid fee 120 USD/kW-yr', fontsize=12)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.04, wspace=0.05, hspace=0.18)
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'fig-retention-maps-{int(level * 100)}.{ext}', dpi=200)
    plt.close(fig)


def curves(t, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    g = t.groupby(['scenario', 'intensity_of_max'])
    colors = dict(zip(ORDER, plt.get_cmap('tab20').colors[::2] + plt.get_cmap('tab20').colors[1::2]))
    for sc in ORDER:
        q = g.retention.quantile([0.1, 0.5, 0.9]).xs(sc, level='scenario').unstack()
        axes[0].plot(q.index * 100, q[0.5], marker='o', label=sc, color=colors[sc])
        axes[0].fill_between(q.index * 100, q[0.1], q[0.9], alpha=0.06, color=colors[sc])
        v = g.V_pct_of_seq.median().xs(sc, level='scenario')
        axes[1].plot(v.index * 100, v, marker='o', label=sc, color=colors[sc])
    axes[0].set_xlabel('Inference demand, % of scenario absorbable maximum')
    axes[0].set_ylabel('Retention (median, band = P10–P90)')
    axes[1].set_xlabel('Inference demand, % of scenario absorbable maximum')
    axes[1].set_ylabel('Joint-dispatch saving,
% of sequential energy cost (median)')
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'fig-retention-curves.{ext}', dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--max-table', required=True)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = pd.read_parquet(args.runs)
    status = runs.groupby(['mode', 'status'], dropna=False).size().rename('rows').reset_index()
    status.to_csv(args.out / 'status-counts.csv', index=False)
    t = paired(runs)
    t.to_parquet(args.out / 'paired.parquet', index=False)
    pd.read_csv(args.max_table).to_csv(args.out / 'max-intensity.csv', index=False)
    g = t.groupby(['scenario', 'intensity_of_max'])
    g.retention.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_csv(args.out / 'summary-retention.csv')
    g.V_pct_of_seq.describe(percentiles=[0.1, 0.5, 0.9]).to_csv(args.out / 'summary-saving.csv')
    x = t[t.intensity_of_max == 0.75]
    pd.DataFrame({'share_gt_1': x.groupby('scenario').retention.apply(lambda s: (s > 1).mean()),
                  'share_gt_1_05': x.groupby('scenario').retention.apply(lambda s: (s > 1.05).mean()),
                  'max': x.groupby('scenario').retention.max()}).to_csv(args.out / 'retention-gt1.csv')
    rows = []
    for sc, d in x.groupby('scenario'):
        d = d.assign(V0_pct=t[(t.intensity_of_max == 0) & (t.scenario == sc)].set_index('grid_id').V_pct_of_seq.reindex(d.grid_id).to_numpy(),
                     wind_share=d.wind_mw_joint_I0 / (d.wind_mw_joint_I0 + d.pv_mw_joint_I0 + 1e-9), abs_lat=d.lat.abs())
        c = d[['retention', 'V0_pct', 'cfe_hourly_joint_I0', 'wind_share', 'abs_lat', 'carbon_t_mwh', 'grid_cap_mw_joint_I0']].corr('spearman').retention
        rows.append(c.drop('retention').rename(sc))
    pd.DataFrame(rows).to_csv(args.out / 'drivers-spearman.csv')
    maps(t, args.out)
    curves(t, args.out)
    print(g.retention.median().unstack().round(2).to_string())


if __name__ == '__main__':
    main()
