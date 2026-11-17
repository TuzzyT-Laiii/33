"""全球柔性收益归因图：(a) 柔性收益随风电出力系数分档的变化；(b) 标准化回归系数热图。

输入 results/global-flex-econ-20261103/attribution/{attribution-table.parquet, regression.csv, fit.csv}
（由 analysis/global_flex_attribution.py 在服务器生成），输出同目录 attribution.{png,pdf} 与 brief-facts.json。
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from plot_scenario_typical_days import setup

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / 'results/global-flex-econ-20261103/attribution'
SCENARIOS = {'D-TRN': '训练', 'D-SAI': '科研 AI', 'A-HPC': '科研计算', 'A-CLD': '云通用', 'D-INF': '推理'}
COLORS = {'D-TRN': '#69A682', 'D-SAI': '#4C8BA5', 'A-HPC': '#7A6BA8', 'A-CLD': '#D8B36B', 'D-INF': '#CB6B53'}
BOUNDARY = {'planning': '规划', 'fixed': '已建成'}
EDGES = [0, .05, .1, .15, .2, .25, .3, .35, .45, .8]


def main():
    setup()
    table = pd.read_parquet(DIR / 'attribution-table.parquet')
    regression = pd.read_csv(DIR / 'regression.csv', encoding='utf-8-sig')
    fit = pd.read_csv(DIR / 'fit.csv', encoding='utf-8-sig')

    fig = plt.figure(figsize=(13.5, 5.6))
    left = fig.add_axes([.055, .19, .38, .69])
    right = fig.add_axes([.56, .19, .36, .66])

    plan = table.loc[table.boundary.eq('planning')].assign(wind_bin=lambda d: pd.cut(d.cf_wind, EDGES, include_lowest=True))
    centers = list(range(len(EDGES) - 1))
    for scenario, label in SCENARIOS.items():
        frame = plan.loc[plan.scenario.eq(scenario)]
        grouped = frame.groupby('wind_bin', observed=False).savings_pct
        median, low, high = grouped.median(), grouped.quantile(.25), grouped.quantile(.75)
        left.plot(centers, median.to_numpy(), '-o', ms=3.5, lw=1.6, color=COLORS[scenario], label=label)
        if scenario == 'D-TRN':
            left.fill_between(centers, low.to_numpy(), high.to_numpy(), color=COLORS[scenario], alpha=.15, lw=0)
    share = plan.loc[plan.scenario.eq('D-TRN')].groupby('wind_bin', observed=False).grid_share.median() * 100
    twin = left.twinx()
    twin.plot(centers, share.to_numpy(), ls='--', color='#5f6b7a', lw=1.2, label='训练·无柔性方案市电占比（右轴）')
    twin.set_ylim(0, 100)
    twin.set_ylabel('无柔性方案市电占比（%）', fontsize=9, color='#5f6b7a')
    twin.tick_params(labelsize=8, colors='#5f6b7a')
    twin.spines['right'].set_visible(True)
    counts = plan.loc[plan.scenario.eq('D-TRN')].groupby('wind_bin', observed=False).size()
    left.set_xticks(centers, [f'{a:.2f}–\n{b:.2f}\n({n})' for a, b, n in zip(EDGES[:-1], EDGES[1:], counts)], fontsize=7)
    left.set_xlabel('风电年均出力系数分档（括号内为网格数）', fontsize=9)
    left.set_ylabel('柔性收益中位数（%，规划·经济最优）', fontsize=9)
    left.set_ylim(0, 26)
    left.grid(axis='y', alpha=.25, lw=.4)
    left.tick_params(axis='y', labelsize=8)
    handles = left.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    left.legend(handles=handles, loc='upper left', fontsize=8, ncol=2)
    left.set_title('(a) 风电资源越好，柔性收益越高（阴影为训练的四分位）', loc='left', fontsize=10)

    exo = regression.loc[regression.model.eq('exogenous') & regression.scenario.isin(SCENARIOS)]
    columns = [(b, s) for b in BOUNDARY for s in SCENARIOS]
    order = exo.groupby('label').delta_r2.mean().sort_values(ascending=False).index.tolist()
    beta = exo.pivot_table(index='label', columns=['boundary', 'scenario'], values='beta_pp_per_sd').reindex(index=order, columns=columns)
    drop = exo.pivot_table(index='label', columns=['boundary', 'scenario'], values='delta_r2').reindex(index=order, columns=columns)
    r2 = fit.set_index(['boundary', 'scenario']).r2_exogenous.reindex(columns)
    limit = float(np.nanmax(np.abs(beta.to_numpy())))
    image = right.imshow(beta.to_numpy(), cmap='RdBu_r', norm=TwoSlopeNorm(0, -limit, limit), aspect='auto')
    for i in range(beta.shape[0]):
        for j in range(beta.shape[1]):
            value, strong = beta.iat[i, j], drop.iat[i, j] >= .02
            right.text(j, i, f'{value:+.1f}', ha='center', va='center', fontsize=7.5 if strong else 6.8,
                       fontweight='bold' if strong else 'normal', color='white' if abs(value) > .55 * limit else '#222222')
    right.set_yticks(range(len(order)), order, fontsize=8.5)
    right.set_xticks(range(len(columns)), [SCENARIOS[s] + f'\nR²={r2[(b, s)]:.2f}' for b, s in columns], fontsize=7.5)
    right.axvline(len(SCENARIOS) - .5, color='white', lw=3)
    for k, b in enumerate(BOUNDARY):
        right.text((len(SCENARIOS) - 1) / 2 + k * len(SCENARIOS), -.62, BOUNDARY[b] + '·经济最优', ha='center', va='bottom', fontsize=9)
    right.tick_params(length=0)
    for spine in right.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=right, fraction=.04, pad=.02)
    bar.set_label('每 1 个标准差对应的收益变化（百分点）', fontsize=8)
    bar.ax.tick_params(labelsize=7)
    right.set_title('(b) 标准化回归系数（粗体：去掉后 R² 降幅 ≥ 0.02）', loc='left', fontsize=10, pad=26)

    fig.text(.01, .015, '4,842 个有效网格，因变量为柔性收益%（经济最优目标）。自变量取 2025 年全年逐小时风光与气温；OLS 标准化系数，HC1 稳健标准误。'
             '未处理空间自相关，重要性以去掉该变量后的 R² 降幅判断。云通用仅 120 小时窗口，与全年特征匹配差，R² 偏低。', fontsize=7.8, color='#535B63')
    fig.savefig(DIR / 'attribution.png', dpi=300, facecolor='white')
    fig.savefig(DIR / 'attribution.pdf', facecolor='white')

    def top(boundary, scenario, k=2):
        rows = exo.loc[exo.boundary.eq(boundary) & exo.scenario.eq(scenario)].nlargest(k, 'delta_r2')
        return [[row.label, round(row.beta_pp_per_sd, 2), round(row.delta_r2, 3)] for row in rows.itertuples()]
    train = plan.loc[plan.scenario.eq('D-TRN')]
    wind_low, wind_high = train.loc[train.cf_wind < .05], train.loc[train.cf_wind > .45]
    facts = {
        'rows': [['代表情境', '解释度 R²\n规划／已建成', '规划阶段主要因素\n（每1标准差，百分点）', '已建成主要因素\n（每1标准差，百分点）']]
        + [[name, f"{r2[('planning', s)]:.2f}／{r2[('fixed', s)]:.2f}",
            '；'.join(f'{l} {b:+.1f}' for l, b, _ in top('planning', s)), '；'.join(f'{l} {b:+.1f}' for l, b, _ in top('fixed', s))]
           for s, name in [('A-HPC', '纯 CPU 型·科研计算'), ('A-CLD', '纯 CPU 型·云通用计算'), ('D-SAI', 'GPU 主导型·科研 AI'),
                           ('D-TRN', 'GPU 主导型·大模型训练/开发'), ('D-INF', 'GPU 主导型·推理服务')]],
        'train_wind_low': {'n': int(len(wind_low)), 'savings_median': round(float(wind_low.savings_pct.median()), 1),
                           'grid_share_median': round(float(wind_low.grid_share.median() * 100), 0)},
        'train_wind_high': {'n': int(len(wind_high)), 'savings_median': round(float(wind_high.savings_pct.median()), 1),
                            'grid_share_median': round(float(wind_high.grid_share.median() * 100), 0)},
        'train_grid_share_r': round(float(np.corrcoef(train.grid_share, train.savings_pct)[0, 1]), 2),
    }
    (DIR / 'brief-facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(facts, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
