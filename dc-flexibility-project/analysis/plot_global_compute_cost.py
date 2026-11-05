import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from report_global_flex_econ import map_cells
from plot_global_maps_all_scenarios import LAYOUT


BOUNDARIES = {'planning': '规划阶段', 'fixed': '已建成系统'}
SCENARIOS = [item for row in LAYOUT for item in row if item is not None]
TARGETS = [('economic', None), ('alpha80', .8), ('alpha90', .9), ('alpha95', .95), ('alpha99', .99), ('alpha100', 1.)]


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def select(frame, boundary, alpha):
    current = frame.loc[frame.boundary.eq(boundary)]
    return current.loc[current.alpha.isna() if alpha is None else current.alpha.eq(alpha)]


def calculate(project, result, destination, threshold):
    table = result / 'grid-summary.parquet'
    manifest = json.loads((result / 'summary-manifest.json').read_text('utf-8'))
    assert digest(table) == manifest['artifacts']['grid-summary.parquet']['sha256'], '全球配对表校验值不符'
    config_path = project / 'analysis/flex-econ-copt-20261102.json'
    energy_path = project / 'analysis/eight-city-flex-value-20261029.json'
    template_path = result / 'provenance/template-index.csv'
    config = json.loads(config_path.read_text('utf-8'))
    energy = json.loads(energy_path.read_text('utf-8'))['parameters']
    rate = energy['discount_rate']['value']
    life = energy['compute_life_years']['value']
    fom = energy['compute_fom_fraction']['value']
    factor = rate * (1 + rate) ** life / ((1 + rate) ** life - 1) + fom
    assert (rate, life, fom) == (.08, 4, .03), '年化参数与当前求解器定义不一致'
    templates = pd.read_csv(template_path)
    assert templates.scenario.is_unique and set(templates.scenario) == {x[0] for x in SCENARIOS}
    records = []
    for scenario, label in SCENARIOS:
        fleet = config['cards']['fleet'][scenario]
        price = config['cards']['server_price_usd'][scenario[0]]['value']
        count = fleet['unit_count'] * fleet['u_max']
        records.append({'scenario': scenario, 'label': label, 'reference_server_count': fleet['unit_count'],
                        'peak_utilization_proxy': fleet['u_max'], 'equivalent_servers': count,
                        'server_price_usd': price, 'annual_factor': factor,
                        'fleet_annual_usd': count * price * factor,
                        'hours': int(templates.set_index('scenario').loc[scenario, 'hours'])})
    fleets = pd.DataFrame(records)
    # 与已求解的八城市机群成本字段逐情境核对，避免另设设备计价方法。
    city = pd.read_csv(project / 'results/flex-card-cost-calendar-20261102/runs-cards.csv')
    recorded = city.groupby('scenario').baseline_fleet_usd_year.agg(['min', 'max'])
    expected = fleets.set_index('scenario').fleet_annual_usd
    np.testing.assert_allclose(recorded.loc[expected.index, 'min'], expected, rtol=1e-12)
    np.testing.assert_allclose(recorded.loc[expected.index, 'max'], expected, rtol=1e-12)
    source = pd.read_parquet(table)
    valid = source.loc[source.baseline_status.eq('optimal') & source.flex_status.eq('optimal')].copy()
    assert valid.extra_task_mw.abs().max() <= 1e-6, '当前配对包含新增算力，须先分离新增设备成本'
    assert (valid.baseline_cost_usd > 0).all()
    metric = valid.merge(fleets, on='scenario', how='left', validate='many_to_one')
    assert metric.hours.notna().all() and metric.fleet_annual_usd.gt(0).all()
    metric['model_savings_usd'] = metric.baseline_cost_usd - metric.flex_cost_usd
    metric['energy_savings_pct'] = 100 * metric.model_savings_usd / metric.baseline_cost_usd
    np.testing.assert_allclose(metric.energy_savings_pct, metric.savings_pct, rtol=1e-12, atol=1e-12)
    metric['annualized_savings_usd'] = metric.model_savings_usd * 8760 / metric.hours
    metric['compute_cost_pct'] = 100 * metric.annualized_savings_usd / metric.fleet_annual_usd
    metric['above_threshold'] = metric.compute_cost_pct.gt(threshold)
    # 窗口内分摊设备成本和等效年化节省两种写法必须一致。
    independent = 100 * metric.model_savings_usd / (metric.fleet_annual_usd * metric.hours / 8760)
    np.testing.assert_allclose(metric.compute_cost_pct, independent, rtol=1e-12, atol=1e-12)
    assert not metric.duplicated(['grid_id', 'scenario', 'alpha', 'boundary']).any()
    assert np.isfinite(metric[['energy_savings_pct', 'compute_cost_pct']]).all().all()
    rows = []
    for boundary in BOUNDARIES:
        for target, alpha in TARGETS:
            for scenario, label in SCENARIOS:
                current = select(metric, boundary, alpha)
                current = current.loc[current.scenario.eq(scenario)]
                assert len(current) == 4842, (boundary, target, scenario, len(current))
                rows.append({'boundary': boundary, 'target': target, 'scenario': scenario, 'label': label,
                             'grids': len(current), 'above': int(current.above_threshold.sum()),
                             'share_pct': float(current.above_threshold.mean() * 100),
                             'energy_savings_median_pct': float(current.energy_savings_pct.median()),
                             'compute_cost_median_pct': float(current.compute_cost_pct.median()),
                             'compute_cost_max_pct': float(current.compute_cost_pct.max())})
    summary = pd.DataFrame(rows)
    metric[['grid_id', 'scenario', 'alpha', 'boundary', 'hours', 'fleet_annual_usd', 'annualized_savings_usd',
            'energy_savings_pct', 'compute_cost_pct', 'above_threshold']].to_parquet(
        destination / 'grid-metrics.parquet', index=False, compression='zstd')
    summary.to_csv(destination / 'scenario-summary.csv', index=False, encoding='utf-8-sig')
    fleets.to_csv(destination / 'fleet-costs.csv', index=False, encoding='utf-8-sig')
    economic = metric.loc[metric.alpha.isna()]
    facts = {'threshold_pct': threshold, 'annual_factor': factor, 'discount_rate': rate, 'life_years': life,
             'fom_fraction': fom, 'grids_valid': metric.grid_id.nunique(), 'pairs': len(metric),
             'economic_max_pct': float(economic.compute_cost_pct.max()),
             'economic_above': int(economic.above_threshold.sum()), 'summaries': rows,
             'fleet_rows': [['情境', '等效服务器台数', '整机价 万美元/台', '机群年化成本 万美元/年']]
                           + [[r.scenario, f'{r.equivalent_servers:.2f}', f'{r.server_price_usd / 1e4:g}',
                               f'{r.fleet_annual_usd / 1e4:.1f}'] for r in fleets.itertuples()],
             'economic_rows': [['情境', '能源侧收益率%\n规划／已建成', '机群年化成本比例%\n规划／已建成', '超过3%网格占比%\n规划／已建成']]
                            + [[scenario] + ['／'.join(f'{summary.loc[summary.scenario.eq(scenario) & summary.boundary.eq(b) & summary.target.eq("economic"), field].iloc[0]:.2f}' for b in BOUNDARIES)
                                              for field in ['energy_savings_median_pct', 'compute_cost_median_pct', 'share_pct']]
                               for scenario, _ in SCENARIOS]}
    write_json(destination / 'brief-facts.json', facts)
    write_json(destination / 'verification.json', {
        'utc': datetime.now(timezone.utc).isoformat(), 'passed': True,
        'input_sha256': {str(path.relative_to(project)): digest(path) for path in [table, config_path, energy_path, template_path]},
        'checks': ['原全球配对表SHA一致', '与八城市求解器机群年化成本逐情境一致', '全部新增算力为零',
                   '能源侧收益率与原表逐行一致', '窗口分摊与等效年化写法逐行一致', '132组情境目标边界各4842格',
                   '严格大于3%分类', '指标有限且配对唯一'],
        'excluded_nonoptimal_pairs': len(source) - len(valid), 'valid_pairs': len(metric),
        'denominator_scope': '沿用模型等效机群unit_count×u_max，非真实设备库存；整机资本回收加固定运维',
        'annualization_scope': '原生窗口节省按8760/有效小时数折算，不代表完整全年实际收益',
        'renderer_sha256': digest(Path(__file__))})
    return source, metric, summary, facts


def maps(source, metric, geography, destination, font, threshold):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch
    import cartopy.crs as ccrs
    import frykit.plot as fplt

    mpl.rcParams.update({'font.family': font, 'font.size': 9, 'axes.unicode_minus': False, 'pdf.fonttype': 42})
    cells, countries = map_cells(geography, source)
    borders = countries.geometry.simplify(.08, preserve_topology=True).to_list()
    for boundary, alpha in [('planning', None), ('fixed', None), ('planning', 1.)]:
        binary = alpha is not None
        token = 'alpha100-threshold' if binary else 'economic'
        current = select(metric, boundary, alpha)
        fig, axes = plt.subplots(4, 3, figsize=(13.2, 10.4), subplot_kw={'projection': ccrs.Robinson()})
        cmap, norm = mpl.colormaps['YlOrRd'], Normalize(0, threshold)
        for r, row in enumerate(LAYOUT):
            for c, item in enumerate(row):
                ax = axes[r, c]
                if item is None:
                    ax.remove()
                    continue
                scenario, label = item
                values = current.loc[current.scenario.eq(scenario), ['grid_id', 'compute_cost_pct', 'above_threshold']]
                merged = cells.merge(values, on='grid_id', how='inner', validate='one_to_one')
                assert len(merged) == 4842
                drawn = merged.loc[~merged.geometry.is_empty]
                share = float(merged.above_threshold.mean() * 100)
                colors = ['#1f7a5a' if flag else '#e8b04a' for flag in drawn.above_threshold] if binary else cmap(norm(drawn.compute_cost_pct))
                ax.set_global()
                fplt.add_geometries(ax, borders, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='#a8a8a8', linewidth=.22, zorder=1)
                fplt.add_geometries(ax, drawn.geometry.to_list(), crs=ccrs.PlateCarree(), facecolor=colors,
                                    edgecolor='none', linewidth=0, zorder=2, rasterized=True)
                detail = f'{share:.1f}% 网格超过 {threshold:g}%' if binary else f'中位数 {merged.compute_cost_pct.median():.2f}%'
                ax.set_title(f'{scenario}  {label}\n{detail}', fontsize=8.5, pad=3)
        target = '100% 本地无碳匹配' if binary else '经济最优'
        fig.suptitle(f'{BOUNDARIES[boundary]}，{target}：柔性收益占机群年化成本的比例', fontsize=12, y=.995)
        fig.subplots_adjust(left=.01, right=.99, top=.925, bottom=.015, hspace=.2, wspace=.02)
        if binary:
            fig.legend(handles=[Patch(facecolor='#1f7a5a', label=f'> {threshold:g}%'), Patch(facecolor='#e8b04a', label=f'≤ {threshold:g}%')],
                       loc='upper center', bbox_to_anchor=(.83, .89), ncol=2, frameon=False, fontsize=10)
        else:
            bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([.71, .855, .24, .017]), orientation='horizontal')
            bar.set_label('占机群年化成本的比例（%）', fontsize=8)
            bar.ax.tick_params(labelsize=8)
        note = f'分母为整机资本年化与固定运维成本。\n严格超过 {threshold:g}% 的网格占比按每情境4,842格计算。\n'
        if not binary:
            note += f'当前经济最优下，全部网格均低于 {threshold:g}%。\n'
        note += '2°区块按代表网格结果着色并裁到陆地；\n空白为无有效输入的区块。'
        fig.text(.83, .795, note, ha='center', va='top', fontsize=7.7, linespacing=1.55, color='#404040')
        stem = destination / f'compute-cost-{boundary}-{token}'
        fig.savefig(stem.with_suffix('.png'), dpi=300, facecolor='white')
        fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
        plt.close(fig)


def share_figure(summary, destination, font, threshold):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    mpl.rcParams.update({'font.family': font, 'font.size': 10, 'axes.unicode_minus': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.5), sharey=True)
    norm = Normalize(0, 100)
    for ax, boundary in zip(axes, BOUNDARIES):
        values = np.array([[summary.loc[summary.boundary.eq(boundary) & summary.target.eq(target) & summary.scenario.eq(scenario), 'share_pct'].iloc[0]
                            for target, _ in TARGETS] for scenario, _ in SCENARIOS])
        image = ax.imshow(values, cmap='YlGnBu', norm=norm, aspect='auto')
        ax.set_title(BOUNDARIES[boundary], fontsize=12, pad=10)
        ax.set_xticks(range(6), ['经济最优', '80%', '90%', '95%', '99%', '100%'])
        ax.set_yticks(range(11), [f'{scenario}  {label.split("·")[-1]}' for scenario, label in SCENARIOS])
        ax.set_xlabel('本地无碳匹配目标')
        ax.tick_params(axis='both', length=0, labelsize=9)
        for i in range(11):
            for j in range(6):
                value = values[i, j]
                text = '0' if value == 0 else '<0.1' if value < .1 else f'{value:.1f}'
                ax.text(j, i, text, ha='center', va='center', fontsize=9, color='white' if value > 55 else '#202020')
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(f'柔性收益超过机群年化成本 {threshold:g}% 的网格占比', fontsize=13, y=.99)
    fig.subplots_adjust(left=.16, right=.91, top=.88, bottom=.16, wspace=.08)
    bar = fig.colorbar(image, cax=fig.add_axes([.93, .19, .013, .63]))
    bar.set_label('超过阈值的网格占比（%）', fontsize=9)
    fig.text(.52, .025, f'每格数字以各情境4,842个有效网格为分母；严格使用“成本比例 > {threshold:g}%”。数字0表示零个网格超过。', ha='center', fontsize=9)
    stem = destination / 'compute-cost-threshold-shares'
    fig.savefig(stem.with_suffix('.png'), dpi=300, facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--geography-run', type=Path, required=True)
    parser.add_argument('--font', default='Microsoft YaHei')
    parser.add_argument('--threshold', type=float, default=3.)
    args = parser.parse_args()
    project = args.project.resolve()
    result = project / 'results/global-flex-econ-20261103'
    destination = result / 'compute-cost'
    destination.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(project / '.codex-temp/compute-cost/matplotlib'))
    source_receipt = json.loads((result / 'provenance/source-downloads.json').read_text('utf-8'))['countries']
    assert digest(args.geography_run / 'sources/countries-10m.zip') == source_receipt['sha256'], '地图国界版本与原图不一致'
    source, metric, summary, facts = calculate(project, result, destination, args.threshold)
    maps(source, metric, args.geography_run, destination, args.font, args.threshold)
    share_figure(summary, destination, args.font, args.threshold)
    write_json(destination / 'artifact-manifest.json', {'files': {path.name: digest(path) for path in destination.iterdir() if path.is_file() and path.name != 'artifact-manifest.json'},
        'country_source_sha256': source_receipt['sha256']})
    print(json.dumps({'pairs': facts['pairs'], 'economic_max_pct': facts['economic_max_pct'], 'economic_above': facts['economic_above'],
                      'nonzero_groups': summary.loc[summary.above.gt(0)].to_dict('records')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
