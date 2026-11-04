"""简表第七章（计入加卡口径）：调度前后周曲线、柔性收益与单位成本对比图及表格数据。"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_flex_dispatch_brief import ALPHA, LINE, REPS, SHORT, alpha_label, dispatch_figure, style


def value_figure(cards, fixed, unit, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw={'width_ratios': [1, 1.3]})
    cards = cards[cards.status == 'optimal'].copy()
    cards['a'] = cards.alpha.map(alpha_label)
    keys = ['city', 'scenario', 'a']
    none = cards[cards.flex == 'none'].set_index(keys).total_cost_usd
    both = cards[(cards.flex == 'both') & (cards.card_price.astype(str) == '1.0')].set_index(keys).total_cost_usd
    pct = (100 * (none - both.reindex(none.index)) / none).rename('pct').reset_index()
    x = np.arange(len(ALPHA))
    for scenario, name in SHORT.items():
        stat = pct[pct.scenario == scenario].groupby('a').pct.agg(['median', 'min', 'max']).reindex(ALPHA)
        ax1.plot(x, stat['median'], marker='o', ms=3.5, lw=1.6, color=LINE[scenario], label=name)
        ax1.fill_between(x, stat['min'], stat['max'], color=LINE[scenario], alpha=.10, lw=0)
    ax1.set_xticks(x, ALPHA)
    ax1.set_ylabel('柔性收益（占无柔性模型成本 %）')
    ax1.set_xlabel('窗口本地风光电量匹配比例')
    ax1.set_title('(a) 规划阶段柔性收益（有效城市中位数与范围）', fontsize=9.5, loc='left')
    ax1.grid(axis='y', alpha=.3)
    ax1.legend(fontsize=7.5, frameon=False, ncol=2, loc='upper left')
    econ = fixed[fixed.a == '经济最优'].groupby('scenario')[['benefit_usd_per_mwh', 'storage_usd_per_mwh']].median()
    bars = [('峰值内柔性：每推迟1 MWh的节省', econ.benefit_usd_per_mwh, '#5b8e3e'),
            ('储能：每放1 MWh的成本', econ.storage_usd_per_mwh, '#8E6BBF'),
            ('加卡柔性：每推迟1 MWh的成本', unit['usd_per_mwh_8h/天'], '#D1495B')]
    width = .26
    pos = np.arange(len(SHORT))
    for k, (name, series, color) in enumerate(bars):
        values = [series[s] for s in SHORT]
        ax2.bar(pos + (k - 1) * width, values, width=width, color=color, label=name)
        for xpos, v in zip(pos + (k - 1) * width, values):
            ax2.text(xpos, v * 1.12, f'{v:,.0f}', ha='center', fontsize=6.5, color='#404040')
    ax2.set_yscale('log')
    ax2.set_ylim(10, 3e5)
    ax2.set_xticks(pos, list(SHORT.values()), fontsize=8)
    ax2.set_ylabel('USD/MWh（对数轴）')
    ax2.set_title('(b) 每 MWh 的柔性收益与成本（经济最优，供给侧固定）', fontsize=9.5, loc='left')
    ax2.grid(axis='y', alpha=.3, which='major')
    ax2.legend(fontsize=7.5, frameon=False, ncol=3, loc='upper left')
    fig.tight_layout()
    fig.savefig(out / 'value-summary.png', dpi=200)
    fig.savefig(out / 'value-summary.pdf')
    plt.close(fig)
    return pct


def table_rows(pct, cards, fixed, unit, shares):
    keys = ['city', 'scenario', 'a']
    targets = ('经济最优', '95%', '100%')
    c = cards[cards.status.eq('optimal')].copy()
    c['a'] = c.alpha.map(alpha_label)
    none = c[c.flex == 'none'].set_index(keys)
    both = c[(c.flex == 'both') & (c.card_price.astype(str) == '1.0')].set_index(keys)
    if not none.index.is_unique or not both.index.is_unique:
        raise ValueError('规划结果存在重复城市、情境和目标配对')
    paired = none.index.intersection(both.index)
    if not np.array_equal(none.loc[paired, 'hours'], both.loc[paired, 'hours']):
        raise ValueError('规划结果的有柔性与无柔性窗口长度不一致')
    saving = ((none.loc[paired, 'total_cost_usd'] - both.loc[paired, 'total_cost_usd'])
              * 8760 / none.loc[paired, 'hours'] / 1e4).rename('s').reset_index()
    med = lambda frame, scenario, a, col: frame[(frame.scenario == scenario) & (frame.a == a)][col].median()
    plan = [['代表情境', '可推迟IT\n电量%', '柔性收益%\n经济最优／95%／100%', '等效年化收益 万美元/年\n经济最优／95%／100%',
             '100%无碳储能MWh\n无柔性→有柔性', '新增算力\nMW任务功率']]
    built = [['代表情境', '柔性收益%\n经济最优／95%／100%', '等效年化收益 万美元/年\n经济最优／95%／100%',
              '每推迟1MWh\n节省 USD', '储能每放\n1MWh USD', '若加卡：每推迟\n1MWh成本 USD', '新增算力\nMW任务功率']]
    extra_plan = both.extra_task_mw.max()
    b_none = none.reset_index()
    b_both = both.reset_index()
    sample_counts = []
    for scenario, name in REPS.items():
        scenario_extra_plan = b_both.loc[b_both.scenario.eq(scenario) & b_both.a.isin(targets), 'extra_task_mw'].max()
        scenario_extra_fixed = fixed.loc[fixed.scenario.eq(scenario) & fixed.a.isin(targets), 'extra_cards_mw'].max()
        if not np.isfinite(scenario_extra_plan) or not np.isfinite(scenario_extra_fixed):
            raise ValueError(f'{scenario} 缺少有效新增算力结果')
        extra_text = lambda value: '0' if abs(value) < 1e-6 else f'{value:.3f}'
        plan.append([name, shares[scenario],
                     '／'.join(f"{med(pct, scenario, a, 'pct'):.1f}" for a in targets),
                     '／'.join(f"{med(saving, scenario, a, 's'):.0f}" for a in targets),
                     f"{med(b_none, scenario, '100%', 'battery_mwh'):.0f}→{med(b_both, scenario, '100%', 'battery_mwh'):.0f}",
                     extra_text(scenario_extra_plan)])
        built.append([name,
                      '／'.join(f"{med(fixed, scenario, a, 'pct'):.1f}" for a in targets),
                      '／'.join(f"{med(fixed, scenario, a, 'save_kusd_year'):.0f}" for a in targets),
                      f"{med(fixed, scenario, '经济最优', 'benefit_usd_per_mwh'):.0f}",
                      f"{med(fixed, scenario, '经济最优', 'storage_usd_per_mwh'):.0f}",
                      f"{unit.loc[scenario, 'usd_per_mwh_8h/天']:,.0f}", extra_text(scenario_extra_fixed)])
        sample_counts.append({'scenario': scenario, 'name': name,
                              'planning': [int(saving[(saving.scenario == scenario) & (saving.a == a)].city.nunique()) for a in targets],
                              'built': [int(fixed[(fixed.scenario == scenario) & (fixed.a == a)].city.nunique()) for a in targets]})
    if any('nan' in cell for rows in (plan, built) for row in rows for cell in row):
        raise ValueError('第七章表格包含缺失值')
    return plan, built, float(max(extra_plan, fixed.extra_cards_mw.max())), sample_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', required=True, help='含规划与固定供给结果CSV的目录')
    parser.add_argument('--hourly-index', required=True, help='逐小时结果索引，含UTC时间版本、相对路径及SHA-256')
    parser.add_argument('--shares', required=True)
    args = parser.parse_args()
    style()
    result = Path(args.result)
    out = result / 'brief'
    out.mkdir(exist_ok=True)
    hourly_index = Path(args.hourly_index)
    indexed_runs = pd.read_csv(hourly_index)
    runs = indexed_runs[indexed_runs.city.eq('frankfurt') & indexed_runs.scenario.eq('D-TRN')
                        & indexed_runs.supply.eq('reoptimized')
                        & (indexed_runs.alpha.isna() | indexed_runs.alpha.eq(1.0))
                        & (indexed_runs.flex.eq('none') | (indexed_runs.flex.eq('both') & indexed_runs.card_price.astype(str).eq('1.0')))].copy()
    if len(runs) != 4 or not runs.status.eq('optimal').all():
        raise ValueError('法兰克福训练示例需要4个最优配对算例')
    if not runs.deadline_clock.eq('actual_utc_next_observed_slot').all():
        raise ValueError('训练示例必须采用实际UTC时间约束')
    runs['alpha_label'] = runs.alpha.map(alpha_label)
    dispatch_figure(hourly_index.parent, runs, 'frankfurt', 'D-TRN', '2025-06-24', 7, out, indexed=True)
    cards = pd.read_csv(result / 'runs-cards.csv')
    fixed = pd.read_csv(result / 'fixed-supply-values.csv')
    unit = pd.read_csv(result / 'card-unit-cost.csv', index_col=0)
    pct = value_figure(cards, fixed, unit, out)
    share = pd.read_csv(args.shares, encoding='utf-8-sig').set_index('scenario')
    shares = {s: f"{share.loc[s, '可推迟·整体'] + share.loc[s, '可推迟·可拆分']:.0f}" for s in REPS}
    plan, built, extra, counts = table_rows(pct, cards, fixed, unit, shares)
    dispatch = runs.set_index(['alpha_label', 'flex'])
    facts = {'plan_rows': plan, 'built_rows': built, 'max_extra_cards_mw_market_price': extra,
             'sample_counts': counts, 'targets': ['经济最优', '95%', '100%'],
             'run_counts': {'planning': len(cards), 'built': len(pd.read_csv(result / 'runs-fixed.csv'))},
             'dispatch_sources': runs[['alpha_label', 'flex', 'hourly_path', 'hourly_sha256', 'deadline_clock']].to_dict('records'),
             'dispatch_battery_mwh_100': [float(dispatch.loc[('100%', 'none'), 'battery_mwh']), float(dispatch.loc[('100%', 'both'), 'battery_mwh'])],
             'dispatch_extra_cards_mw': float(runs.extra_task_mw.max())}
    (out / 'brief-facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=1), 'utf-8')
    print(json.dumps(facts, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
