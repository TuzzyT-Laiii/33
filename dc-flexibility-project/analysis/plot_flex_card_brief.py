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
    ax1.set_ylabel('柔性收益（占无柔性系统成本 %）')
    ax1.set_xlabel('本地风光逐小时匹配比例')
    ax1.set_title('(a) 柔性收益随无碳目标变化（不加卡，八城市中位数与范围）', fontsize=9.5, loc='left')
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
    a_rows = cards[(cards.flex == 'none') & cards.status.eq('optimal')].copy()
    a_rows['a'] = a_rows.alpha.map(alpha_label)
    hours = a_rows.groupby('scenario').hours.first()
    none = a_rows.set_index(['city', 'scenario', 'a']).total_cost_usd
    b = cards[(cards.flex == 'both') & (cards.card_price.astype(str) == '1.0')].copy()
    b['a'] = b.alpha.map(alpha_label)
    saving = (none - b.set_index(['city', 'scenario', 'a']).total_cost_usd.reindex(none.index)).rename('s').reset_index()
    rows = [['代表情境', '可推迟IT\n电量%', '柔性收益%\n经济最优／100%无碳', '万美元/年\n经济最优／100%无碳',
             '峰值内每推迟\n1MWh节省 USD', '储能每放\n1MWh USD', '加卡每推迟\n1MWh USD']]
    for scenario, name in REPS.items():
        p_e = pct[(pct.scenario == scenario) & (pct.a == '经济最优')].pct.median()
        p_f = pct[(pct.scenario == scenario) & (pct.a == '100%')].pct.median()
        s_e = saving[(saving.scenario == scenario) & (saving.a == '经济最优')].s.median() * 8760 / hours[scenario] / 1e4
        s_f = saving[(saving.scenario == scenario) & (saving.a == '100%')].s.median() * 8760 / hours[scenario] / 1e4
        e = fixed[(fixed.scenario == scenario) & (fixed.a == '经济最优')]
        rows.append([name, shares[scenario], f'{p_e:.1f}／{p_f:.1f}', f'{s_e:.0f}／{s_f:.0f}',
                     f'{e.benefit_usd_per_mwh.median():.0f}', f'{e.storage_usd_per_mwh.median():.0f}',
                     f'{unit.loc[scenario, "usd_per_mwh_8h/天"]:,.0f}'])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', required=True, help='results/flex-card-cost-20261102')
    parser.add_argument('--hourly', required=True, help='含dispatch-runs.csv与逐小时文件')
    parser.add_argument('--shares', required=True)
    args = parser.parse_args()
    style()
    result = Path(args.result)
    out = result / 'brief'
    out.mkdir(exist_ok=True)
    runs = pd.read_csv(Path(args.hourly) / 'dispatch-runs.csv')
    runs['alpha_label'] = runs.alpha.map(alpha_label)
    dispatch_figure(args.hourly, runs, 'frankfurt', 'D-TRN', '2025-06-24', 7, out)
    cards = pd.read_csv(result / 'runs-cards.csv')
    fixed = pd.read_csv(result / 'fixed-supply-values.csv')
    unit = pd.read_csv(result / 'card-unit-cost.csv', index_col=0)
    pct = value_figure(cards, fixed, unit, out)
    share = pd.read_csv(args.shares, encoding='utf-8-sig').set_index('scenario')
    shares = {s: f"{share.loc[s, '可推迟·整体'] + share.loc[s, '可推迟·可拆分']:.0f}" for s in REPS}
    rows = table_rows(pct, cards, fixed, unit, shares)
    dispatch = runs.set_index(['alpha_label', 'flex'])
    facts = {'table_rows': rows,
             'dispatch_battery_mwh_100': [float(dispatch.loc[('100%', 'none'), 'battery_mwh']), float(dispatch.loc[('100%', 'both'), 'battery_mwh'])],
             'dispatch_extra_cards_mw': float(runs.extra_task_mw.max())}
    (out / 'brief-facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=1), 'utf-8')
    print(json.dumps(facts, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
