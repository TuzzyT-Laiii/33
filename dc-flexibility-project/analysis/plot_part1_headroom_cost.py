"""第一部分主图：空档充足，但空档本身不值钱。

(a) 峰值余量：各情境平均任务负荷占窗口任务峰值 K 的比例；
(b) 成本量级（10 MW 参考规模，等效年化，全球网格中位数）：机群年化成本、能源系统年化成本、柔性节省（规划／已建成）；
(c) 全球 4,842 个有效网格上，柔性节省占机群年化成本的分布（经济最优，规划／已建成），对照 3% 阈值。

输入：results/flex-class-20261101/native-hourly/<情境>.parquet、results/global-flex-econ-20261103/grid-summary.parquet、
analysis/flex-econ-copt-20261102.json（服务器价、机群台数与年化方法，与求解器 card_spec 一致）。
输出：results/global-flex-econ-20261103/part1/part1-headroom-cost.{png,pdf} 与 part1-summary.csv。
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from plot_scenario_typical_days import setup

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / 'results/flex-class-20261101/native-hourly'
SUMMARY = ROOT / 'results/global-flex-econ-20261103/grid-summary.parquet'
CONFIG = ROOT / 'analysis/flex-econ-copt-20261102.json'
OUT = ROOT / 'results/global-flex-econ-20261103/part1'
SCENARIOS = ['A-HPC', 'A-CLD', 'B-SAI', 'C-SAI', 'D-SAI', 'B-TRN', 'C-TRN', 'D-TRN', 'B-INF', 'C-INF', 'D-INF']
NAMES = {'A-HPC': '纯CPU·科研计算', 'A-CLD': '纯CPU·云通用', 'B-SAI': 'CPU主导·科研AI', 'C-SAI': '均衡·科研AI', 'D-SAI': 'GPU主导·科研AI',
         'B-TRN': 'CPU主导·训练', 'C-TRN': '均衡·训练', 'D-TRN': 'GPU主导·训练', 'B-INF': 'CPU主导·推理', 'C-INF': '均衡·推理', 'D-INF': 'GPU主导·推理'}
TASKS = ['刚性', '未知', '可推迟·整体', '可推迟·可拆分']
C_PLAN, C_FIXED, C_FLEET, C_ENERGY = '#4C8BA5', '#9CC3D5', '#5f6b7a', '#D8B36B'


def fleet_costs():
    cards = json.loads(CONFIG.read_text('utf-8'))['cards']
    rate, years = .08, 4
    factor = rate * (1 + rate) ** years / ((1 + rate) ** years - 1) + .03
    out = {}
    for scenario in SCENARIOS:
        fleet = cards['fleet'][scenario]
        servers = fleet['unit_count'] * fleet['u_max']
        out[scenario] = cards['server_price_usd'][scenario[0]]['value'] * servers * factor
    return out, factor


def main():
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in SCENARIOS:
        frame = pd.read_parquet(NATIVE / f'{scenario}.parquet')
        task = frame[TASKS].sum(axis=1)
        k = float(task.max())
        rows.append({'scenario': scenario, 'hours': len(frame), 'K_mw': k, 'mean_over_K_pct': task.mean() / k * 100,
                     'hours_above_90pct_K': int((task > .9 * k).sum())})
    head = pd.DataFrame(rows).set_index('scenario')

    fleet, factor = fleet_costs()
    grid = pd.read_parquet(SUMMARY)
    valid = grid.loc[grid.baseline_status.eq('optimal') & grid.flex_status.eq('optimal') & grid.alpha.isna()].copy()
    scale = 8760 / valid.scenario.map(head.hours)
    valid['saving_year'] = (valid.baseline_cost_usd - valid.flex_cost_usd) * scale
    valid['energy_year'] = valid.baseline_cost_usd * scale
    valid['fleet_year'] = valid.scenario.map(fleet)
    valid['fleet_pct'] = valid.saving_year / valid.fleet_year * 100
    med = valid.groupby(['scenario', 'boundary'])[['saving_year', 'energy_year', 'fleet_pct', 'savings_pct']].median().unstack('boundary')

    summary = head.copy()
    summary['fleet_annual_musd'] = pd.Series(fleet) / 1e6
    summary['energy_annual_musd'] = med[('energy_year', 'planning')] / 1e6
    for b in ('planning', 'fixed'):
        summary[f'saving_{b}_kusd'] = med[('saving_year', b)] / 1e3
        summary[f'energy_pct_{b}'] = med[('savings_pct', b)]
        summary[f'fleet_pct_{b}'] = med[('fleet_pct', b)]
        summary[f'fleet_pct_{b}_max'] = valid.loc[valid.boundary.eq(b)].groupby('scenario').fleet_pct.max()
    summary.round(4).to_csv(OUT / 'part1-summary.csv', encoding='utf-8-sig')

    fig = plt.figure(figsize=(13.8, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.25, 1.25], wspace=.08, left=.115, right=.985, top=.86, bottom=.27)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))
    y = np.arange(len(SCENARIOS))[::-1]

    used = head.loc[SCENARIOS, 'mean_over_K_pct'].to_numpy()
    ax_a.barh(y, used, color='#69A682', height=.62, label='平均任务负荷')
    ax_a.barh(y, 100 - used, left=used, color='#E3EEE6', height=.62, edgecolor='#69A682', lw=.6, label='峰值余量（空档）')
    for yi, u in zip(y, used):
        ax_a.text(u + (100 - u) / 2, yi, f'{100 - u:.0f}%', ha='center', va='center', fontsize=8, color='#2F5E3E')
    ax_a.set_yticks(y, [NAMES[s] for s in SCENARIOS], fontsize=8.5)
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel('占窗口任务峰值 K（%）', fontsize=9)
    ax_a.set_title('(a) 空档充足：平均有约三到五成峰值余量', loc='left', fontsize=10)
    ax_a.legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2, fontsize=8, frameon=False)

    def dots(ax, values, color, marker, label, size=36):
        ax.scatter(values, y, s=size, color=color, marker=marker, zorder=3, label=label, edgecolor='white', linewidth=.5)
    fleet_m = summary.loc[SCENARIOS, 'fleet_annual_musd'].to_numpy() * 1e6
    energy_m = summary.loc[SCENARIOS, 'energy_annual_musd'].to_numpy() * 1e6
    plan_m = summary.loc[SCENARIOS, 'saving_planning_kusd'].to_numpy() * 1e3
    fixed_m = summary.loc[SCENARIOS, 'saving_fixed_kusd'].to_numpy() * 1e3
    for yi, a, b in zip(y, fixed_m, fleet_m):
        ax_b.plot([a, b], [yi, yi], color='#D0D5DA', lw=1, zorder=1)
    dots(ax_b, fleet_m, C_FLEET, 's', '机群年化成本')
    dots(ax_b, energy_m, C_ENERGY, 'D', '能源系统年化成本（无柔性）')
    dots(ax_b, plan_m, C_PLAN, 'o', '柔性节省·规划')
    dots(ax_b, fixed_m, C_FIXED, 'o', '柔性节省·已建成')
    ax_b.set_xscale('log')
    ax_b.set_xlim(5e4, 5e8)
    ax_b.set_xticks([1e5, 1e6, 1e7, 1e8], ['10万', '100万', '1000万', '1亿'], fontsize=8.5)
    ax_b.set_yticks(y, [])
    ax_b.set_xlabel('美元／年（10MW，等效年化，全球中位数，对数轴）', fontsize=9)
    ax_b.set_title('(b) 柔性节省比机群成本小两到三个数量级', loc='left', fontsize=10)
    ax_b.grid(axis='x', alpha=.25, lw=.4, which='major')
    ax_b.legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2, fontsize=8, frameon=False)

    data_p = [valid.loc[valid.scenario.eq(s) & valid.boundary.eq('planning'), 'fleet_pct'].to_numpy() for s in SCENARIOS]
    data_f = [valid.loc[valid.scenario.eq(s) & valid.boundary.eq('fixed'), 'fleet_pct'].to_numpy() for s in SCENARIOS]
    style = dict(vert=False, widths=.32, showfliers=False, whis=(1, 99), patch_artist=True, medianprops={'color': '#222222', 'lw': 1})
    bp = ax_c.boxplot(data_p, positions=y + .18, **style)
    bf = ax_c.boxplot(data_f, positions=y - .18, **style)
    for box in bp['boxes']:
        box.set(facecolor=C_PLAN, edgecolor=C_PLAN, alpha=.85)
    for box in bf['boxes']:
        box.set(facecolor=C_FIXED, edgecolor='#6E9FB5', alpha=.95)
    ax_c.axvline(3, color='#B2182B', ls='--', lw=1.1)
    ax_c.text(3.05, y.max() + .45, '3% 阈值', color='#B2182B', fontsize=8.5, va='bottom')
    ax_c.set_xlim(0, 3.6)
    ax_c.set_ylim(-.7, y.max() + .9)
    ax_c.set_yticks(y, [])
    ax_c.set_xlabel('柔性节省占机群年化成本（%）', fontsize=9)
    ax_c.set_title('(c) 全球 4,842 格无一超过 3%', loc='left', fontsize=10)
    ax_c.grid(axis='x', alpha=.25, lw=.4)
    ax_c.legend(handles=[Patch(facecolor=C_PLAN, label='规划'), Patch(facecolor=C_FIXED, edgecolor='#6E9FB5', label='已建成')],
                loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2, fontsize=8, frameon=False)
    for ax in (ax_a, ax_b, ax_c):
        ax.tick_params(axis='x', labelsize=8.5)
    ax_a.set_ylim(ax_c.get_ylim())
    ax_b.set_ylim(ax_c.get_ylim())

    fig.suptitle('第一部分：空档充足，但空档本身不值钱', fontsize=12.5, x=.01, ha='left', y=.985)
    fig.text(.01, .015, f'经济最优目标；节省与能源成本按原生窗口小时等效年化；机群年化成本＝整机价×等效台数×{factor:.3f}（4年、8%、运维3%），既有机群在有无柔性两侧相同。'
             '箱体为四分位、须为1–99分位。云通用仅120小时窗口，年化金额不确定性大。', fontsize=7.6, color='#535B63')
    fig.savefig(OUT / 'part1-headroom-cost.png', dpi=300, facecolor='white')
    fig.savefig(OUT / 'part1-headroom-cost.pdf', facecolor='white')
    print(summary.round(2).to_string())


if __name__ == '__main__':
    main()
