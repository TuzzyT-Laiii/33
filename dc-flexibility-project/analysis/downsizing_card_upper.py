"""3.3 补算：算力设备降本下的减配上限。
上限（可行）= 少买设备 × 卡价倍数 m（每 MW 任务功率，相对 2024）+ 供能侧节省（按 2024 口径，占上限 2%–4%）。
与同一路径下的现状柔性节省（paired-lc，碳价 100、不接推理）逐格对比。"""
import sys

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
USE = {'CLD': '云计算', 'HPC': 'HPC', 'INF': '推理', 'SAI': '科研 AI', 'TRN': '训练'}
M = {'2024': 1.0, '2030-conservative': 1.43, '2035-conservative': 1.92, '2030-moderate': 0.76, '2035-moderate': 0.60,
     '2030-advanced': 0.44, '2035-advanced': 0.22}
cap = pd.read_parquet('D:/codex-runs/fig4-conclusion-20261119/worktree/dc-flexibility-project/results/'
                      'pending-results-20261116/global-capacity.parquet')
cap['equip'] = cap.equipment_saving_feasible_usd_year / cap.E0_mwh_year
cap['energy'] = (cap.baseline_energy_usd_year - cap.feasible_energy_usd_year) / cap.E0_mwh_year
cap = cap.set_index(['grid_id', 'scenario'])
lc = pd.read_parquet('D:/codex-runs/meeting-restructure-20261120/out/paired-lc.parquet', columns=['grid_id', 'scenario', 'cost_scenario', 'intensity_of_max', 'V'])
lc = lc[lc.intensity_of_max.eq(0)]
p0 = pd.read_parquet('D:/codex-runs/meeting-restructure-20261120/out/paired-full.parquet', columns=['grid_id', 'scenario', 'intensity_of_max', 'V'])
p0 = p0[p0.intensity_of_max.eq(0)].assign(cost_scenario='2024')
v = pd.concat([lc, p0]).set_index(['grid_id', 'scenario'])


def rng(s, f='{:.0f}'):
    return f'{f.format(s.min())}–{f.format(s.max())}'


for cs, m in M.items():
    sub = v[v.cost_scenario.eq(cs)]
    c = cap.reindex(sub.index)
    upper = c.equip * m + c.energy
    ratio = upper / sub.V
    t = pd.DataFrame({'upper': upper, 'ratio': ratio, 'eq_share': 100 * c.equip * m / upper}).groupby(level=1).median()
    print(f'{cs} (m={m}): 上限 {rng(t.upper)} 美元/MWh，为现状柔性的 {rng(t.ratio)} 倍，设备占 {rng(t.eq_share, "{:.0f}")}%')
    if cs in ('2024', '2035-advanced', '2035-conservative'):
        t['use'] = [USE[s.split('-')[1]] for s in t.index]
        print('   ', '; '.join(f'{u} {rng(g.upper)}（×{rng(g.ratio)}）' for u, g in t.groupby('use')))
