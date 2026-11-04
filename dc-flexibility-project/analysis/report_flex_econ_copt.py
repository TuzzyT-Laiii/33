"""汇总flex_econ_copt批次结果：柔性价值表、容量对比与两张图。"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAMES = {
    'A-HPC': '纯 CPU 型·科研计算', 'A-CLD': '纯 CPU 型·云通用计算',
    'B-SAI': 'CPU 主导型·科研 AI', 'C-SAI': '均衡型·科研 AI', 'D-SAI': 'GPU 主导型·科研 AI',
    'B-TRN': 'CPU 主导型·大模型训练/开发', 'C-TRN': '均衡型·大模型训练/开发', 'D-TRN': 'GPU 主导型·大模型训练/开发',
    'B-INF': 'CPU 主导型·推理服务', 'C-INF': '均衡型·推理服务', 'D-INF': 'GPU 主导型·推理服务'}
CITY = {'lulea': '吕勒奥', 'frankfurt': '法兰克福', 'ashburn': '阿什本', 'ulanqab': '乌兰察布',
        'guiyang': '贵阳', 'phoenix': '菲尼克斯', 'riyadh': '利雅得', 'singapore': '新加坡'}
ORDER = list(NAMES)
ALPHA = ['经济最优', '80%', '90%', '95%', '99%', '100%']


def alpha_label(value):
    return '经济最优' if pd.isna(value) else f'{value:.0%}'


def load(path):
    runs = pd.read_csv(path)
    runs['variant'] = runs.get('variant', pd.Series('main', index=runs.index)).fillna('main')
    runs = runs.drop_duplicates(['city', 'scenario', 'alpha', 'design', 'flex', 'variant'], keep='last')
    runs['alpha_label'] = runs.alpha.map(alpha_label)
    return runs


def paired(runs, design='opt', variant='main'):
    ok = runs[(runs.design == design) & (runs.status == 'optimal') & (runs.variant == variant)]
    base = ok[ok.flex == 'none'].set_index(['city', 'scenario', 'alpha_label'])
    rows = []
    for flex in [f for f in ok.flex.unique() if f != 'none']:
        part = ok[ok.flex == flex].set_index(['city', 'scenario', 'alpha_label'])
        joined = part.join(base, rsuffix='_base', how='inner')
        frame = pd.DataFrame({
            'flex': flex, 'hours': joined.hours,
            'cost_base_usd': joined.total_cost_usd_base, 'cost_flex_usd': joined.total_cost_usd,
            'value_usd': joined.total_cost_usd_base - joined.total_cost_usd,
            'it_mwh': joined.baseline_it_mwh, 'facility_mwh': joined.baseline_facility_mwh,
            'moved_mwh': joined.moved_whole_mwh + joined.moved_split_mwh,
            'flex_mwh': joined.arrival_whole_mwh + joined.arrival_split_mwh,
            'pv_base': joined.pv_mw_base, 'pv_flex': joined.pv_mw, 'wind_base': joined.wind_mw_base, 'wind_flex': joined.wind_mw,
            'bat_mwh_base': joined.battery_mwh_base, 'bat_mwh_flex': joined.battery_mwh,
            'bat_mw_base': joined.battery_mw_base, 'bat_mw_flex': joined.battery_mw,
            'grid_cap_base': joined.grid_cap_mw_base, 'grid_cap_flex': joined.grid_cap_mw,
            'cfe_base': joined.cfe_hourly_base, 'cfe_flex': joined.cfe_hourly})
        rows.append(frame.reset_index())
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out['value_pct'] = 100 * out.value_usd / out.cost_base_usd
    out['value_usd_per_mwh_it'] = out.value_usd / out.it_mwh
    out['lcoe_base_usd_mwh'] = out.cost_base_usd / out.facility_mwh
    out['lcoe_flex_usd_mwh'] = out.cost_flex_usd / out.facility_mwh
    out['value_annual_equiv_kusd'] = out.value_usd * 8760 / out.hours / 1000
    out['value_per_moved_mwh'] = out.value_usd / out.moved_mwh.where(out.moved_mwh > 1e-6)
    return out


def median_range(series, digits=1):
    return f'{series.median():.{digits}f}（{series.min():.{digits}f}–{series.max():.{digits}f}）'


def tables(runs, pairs, out):
    main = pairs[pairs.flex == 'both']
    lines = []
    # 表1：情境×α，柔性价值占系统成本%，8城市中位数（范围）
    t1 = main.pivot_table(index='scenario', columns='alpha_label', values='value_pct', aggfunc=median_range).reindex(ORDER)[ALPHA]
    t1.index = [NAMES[s] for s in t1.index]
    t1.to_csv(out / 'table-value-pct-by-alpha.csv', encoding='utf-8-sig')
    t2 = main.pivot_table(index='scenario', columns='alpha_label', values='value_usd_per_mwh_it', aggfunc=median_range).reindex(ORDER)[ALPHA]
    t2.index = [NAMES[s] for s in t2.index]
    t2.to_csv(out / 'table-value-usd-per-mwh-it-by-alpha.csv', encoding='utf-8-sig')
    # 表3：基准与柔性下的度电成本（α=经济最优与100%）
    rows = []
    for scenario in ORDER:
        for label in ('经济最优', '100%'):
            part = main[(main.scenario == scenario) & (main.alpha_label == label)]
            if part.empty:
                continue
            rows.append({'情境': NAMES[scenario], '目标': label,
                         '无柔性度电成本USD/MWh': median_range(part.lcoe_base_usd_mwh, 1),
                         '有柔性度电成本USD/MWh': median_range(part.lcoe_flex_usd_mwh, 1),
                         '柔性价值%': median_range(part.value_pct, 1),
                         '柔性价值USD/MWh-IT': median_range(part.value_usd_per_mwh_it, 1),
                         '折算年值万美元/年(10MW)': median_range(part.value_annual_equiv_kusd / 10, 0),
                         '每MWh推迟电量价值USD': median_range(part.value_per_moved_mwh, 1)})
    pd.DataFrame(rows).to_csv(out / 'table-lcoe-and-value.csv', index=False, encoding='utf-8-sig')
    # 成本分项：柔性带来的各项节省占总节省的比例（城市中位数）
    ok = runs[(runs.design == 'opt') & (runs.status == 'optimal') & (runs.variant == 'main')]
    keys = ['city', 'scenario', 'alpha_label']
    comp = ['cost_pv', 'cost_wind', 'cost_battery_power', 'cost_battery_energy', 'cost_grid_capacity', 'cost_grid_energy', 'cost_carbon']
    a = ok[ok.flex == 'none'].set_index(keys)[comp]
    b = ok[ok.flex == 'both'].set_index(keys)[comp]
    delta = (a - b.reindex(a.index)).dropna()
    delta = delta.div(delta.sum(axis=1), axis=0) * 100
    delta['光伏+风电'] = delta.cost_pv + delta.cost_wind
    delta['储能'] = delta.cost_battery_power + delta.cost_battery_energy
    delta['市电接入容量'] = delta.cost_grid_capacity
    delta['市电电费+碳价'] = delta.cost_grid_energy + delta.cost_carbon
    t = delta[['光伏+风电', '储能', '市电接入容量', '市电电费+碳价']].groupby(level=['scenario', 'alpha_label']).median()
    t = t[t.index.get_level_values(1).isin(['经济最优', '95%', '100%'])].round(1)
    t.to_csv(out / 'table-saving-components-pct.csv', encoding='utf-8-sig')
    # 表4：100%本地无碳的容量设计，城市中位数
    full = main[main.alpha_label == '100%']
    rows = []
    for scenario in ORDER:
        part = full[full.scenario == scenario]
        if part.empty:
            continue
        rows.append({'情境': NAMES[scenario],
                     '光伏MW 无→有': f'{part.pv_base.median():.0f}→{part.pv_flex.median():.0f}',
                     '风电MW 无→有': f'{part.wind_base.median():.0f}→{part.wind_flex.median():.0f}',
                     '储能MWh 无→有': f'{part.bat_mwh_base.median():.0f}→{part.bat_mwh_flex.median():.0f}',
                     '储能MW 无→有': f'{part.bat_mw_base.median():.1f}→{part.bat_mw_flex.median():.1f}'})
    pd.DataFrame(rows).to_csv(out / 'table-capacity-100pct.csv', index=False, encoding='utf-8-sig')
    # 表5：分解（整体/可拆分/两者），α=经济最优与100%
    rows = []
    for scenario in ORDER:
        for label in ('经济最优', '95%', '100%'):
            row = {'情境': NAMES[scenario], '目标': label}
            for flex, name in (('whole', '仅可推迟·整体12h'), ('split', '仅可拆分48h'), ('both', '两类')):
                part = pairs[(pairs.flex == flex) & (pairs.scenario == scenario) & (pairs.alpha_label == label)]
                row[name + '%'] = f'{part.value_pct.median():.1f}' if len(part) else ''
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / 'table-decomposition.csv', index=False, encoding='utf-8-sig')
    # 表6：推迟时长敏感性（两类同用D），城市中位数柔性价值%
    sens = pairs[pairs.flex.str.startswith('uniform') | (pairs.flex == 'both')]
    sens = sens[sens.alpha_label.isin(['经济最优', '95%', '100%'])]
    t6 = sens.pivot_table(index=['scenario', 'alpha_label'], columns='flex', values='value_pct', aggfunc='median')
    cols = [c for c in ['uniform2', 'uniform4', 'uniform8', 'uniform24', 'both'] if c in t6.columns]
    t6 = t6[cols].round(2)
    t6.to_csv(out / 'table-deadline-sensitivity.csv', encoding='utf-8-sig')
    # 表7：城市×情境，α=100%柔性价值%
    t7 = full.pivot_table(index='scenario', columns='city', values='value_pct').reindex(ORDER)
    t7 = t7[[c for c in CITY if c in t7.columns]]
    t7.round(2).to_csv(out / 'table-city-value-pct-100.csv', encoding='utf-8-sig')
    # 风光按均值设计：可达到的α与成本
    # 市电接入容量费=0敏感性
    g0 = paired(runs, 'opt', 'grid_fee0')
    if len(g0):
        g0 = g0[g0.flex == 'both']
        t = g0.pivot_table(index='scenario', columns='alpha_label', values='value_pct', aggfunc=median_range).reindex(ORDER)
        t = t[[c for c in ALPHA if c in t.columns]]
        t.index = [NAMES[x] for x in t.index]
        t.to_csv(out / 'table-grid-fee0-value-pct.csv', encoding='utf-8-sig')
    mean = runs[(runs.design == 're_mean')]
    feas = mean.pivot_table(index='scenario', columns='alpha_label', values='status',
                            aggfunc=lambda s: f'{(s == "optimal").sum()}/{len(s)}').reindex(ORDER)
    feas.to_csv(out / 'table-re-mean-feasible-cities.csv', encoding='utf-8-sig')
    if len(mean):
        mp = paired(runs, 're_mean')
        mp = mp[mp.flex == 'both']
        t8 = mp.pivot_table(index='scenario', columns='alpha_label', values='value_pct', aggfunc=median_range).reindex(ORDER)
        t8 = t8[[c for c in ALPHA if c in t8.columns]]
        t8.to_csv(out / 'table-re-mean-value-pct.csv', encoding='utf-8-sig')
        opt_cost = runs[(runs.design == 'opt') & (runs.flex == 'none') & (runs.variant == 'main')].set_index(['city', 'scenario', 'alpha_label']).total_cost_usd
        mean_cost = runs[(runs.design == 're_mean') & (runs.flex == 'none') & (runs.status == 'optimal')].set_index(['city', 'scenario', 'alpha_label']).total_cost_usd
        premium = (100 * (mean_cost / opt_cost.reindex(mean_cost.index) - 1)).rename('premium_pct').reset_index()
        premium.pivot_table(index='scenario', columns='alpha_label', values='premium_pct', aggfunc='median').reindex(ORDER).round(2).to_csv(
            out / 'table-re-mean-cost-premium-pct.csv', encoding='utf-8-sig')
    return t1


def figures(pairs, out):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    main = pairs[pairs.flex == 'both']
    groups = [('科研计算/云', ['A-HPC', 'A-CLD']), ('科研 AI', ['B-SAI', 'C-SAI', 'D-SAI']),
              ('大模型训练/开发', ['B-TRN', 'C-TRN', 'D-TRN']), ('推理服务', ['B-INF', 'C-INF', 'D-INF'])]
    colors = ['#2a6f97', '#e07a1f', '#5b8e3e']
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    x = np.arange(len(ALPHA))
    for ax, (title, members) in zip(axes, groups):
        for color, scenario in zip(colors, members):
            part = main[main.scenario == scenario]
            stat = part.groupby('alpha_label').value_pct.agg(['median', 'min', 'max']).reindex(ALPHA)
            ax.plot(x, stat['median'], marker='o', color=color, label=NAMES[scenario], lw=1.8)
            ax.fill_between(x, stat['min'], stat['max'], color=color, alpha=.12, lw=0)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x, ALPHA, fontsize=8)
        ax.grid(axis='y', alpha=.3)
        ax.legend(fontsize=7, frameon=False, loc='upper left')
    axes[0].set_ylabel('柔性价值（占无柔性系统成本 %）')
    fig.supxlabel('本地风光逐小时匹配比例 α', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'value-vs-alpha.png', dpi=200)
    fig.savefig(out / 'value-vs-alpha.pdf')
    plt.close(fig)
    full = main[main.alpha_label == '100%'].pivot_table(index='scenario', columns='city', values='value_pct').reindex(ORDER)
    full = full[[c for c in CITY if c in full.columns]]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    image = ax.imshow(full.to_numpy(), cmap='YlGnBu', aspect='auto')
    ax.set_xticks(range(full.shape[1]), [CITY[c] for c in full.columns], fontsize=9)
    ax.set_yticks(range(len(ORDER)), [NAMES[s] for s in ORDER], fontsize=9)
    for i in range(full.shape[0]):
        for j in range(full.shape[1]):
            v = full.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=8, color='white' if v > np.nanmax(full.to_numpy()) * .6 else 'black')
    fig.colorbar(image, ax=ax, label='柔性价值 %（α=100%）')
    ax.set_title('100% 本地逐小时无碳：柔性降低的系统成本比例', fontsize=11)
    fig.tight_layout()
    fig.savefig(out / 'value-heatmap-100pct.png', dpi=200)
    fig.savefig(out / 'value-heatmap-100pct.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    runs = load(args.runs)
    status = runs.groupby(['variant', 'design', 'flex', 'status']).size().rename('count').reset_index()
    status.to_csv(out / 'status-counts.csv', index=False, encoding='utf-8-sig')
    pairs = paired(runs)
    pairs.to_csv(out / 'paired-values.csv', index=False, encoding='utf-8-sig')
    tables(runs, pairs, out)
    figures(pairs, out)
    print(status.to_string())


if __name__ == '__main__':
    main()
