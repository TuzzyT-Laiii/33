"""1.3 上限：规划阶段减配后的购电减排与电网接入容量削减（全球 4,842 网格 × 11 情境）。
A = 原机群、原任务时序（不做柔性）；C = 最少台数 + 已核验的可行时序；L = 最少台数下界 + 自由调度（放松解）。
减排 = (A − X) 购电 × 当地碳因子 ÷ E0；接入容量削减 = (A − X) / A。与现状柔性（paired-full，I=0）同表对比。"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).parent
USE = {'CLD': '云计算', 'HPC': 'HPC', 'INF': '推理', 'SAI': '科研 AI', 'TRN': '训练'}
ORDER = ['TRN', 'HPC', 'SAI', 'CLD', 'INF']
d = pd.read_parquet(sys.argv[1]) if sys.argv[1].endswith('.parquet') else pd.read_json(sys.argv[1], lines=True)
assert len(d) == 53262 and not d.duplicated(['grid_id', 'scenario']).any(), len(d)
for x in ('C', 'L'):
    d[f'dEm_{x}'] = 1000 * (d.A_grid_mwh - d[f'{x}_grid_mwh']) * d.carbon_t_mwh / d.E0_mwh_year
    d[f'dEmPct_{x}'] = 100 * (d.A_grid_mwh - d[f'{x}_grid_mwh']) / d.A_grid_mwh.where(d.A_grid_mwh > 0)
    d[f'dCap_{x}'] = 100 * (d.A_grid_cap_mw - d[f'{x}_grid_cap_mw']) / d.A_grid_cap_mw.where(d.A_grid_cap_mw > 1e-6)
now = pd.read_parquet(HERE / 'out/paired-full.parquet', columns=['grid_id', 'scenario', 'intensity_of_max', 'dEm', 'dEm_pct', 'dCap_pct'])
now = now[now.intensity_of_max.eq(0)].drop(columns='intensity_of_max')
now['dEm'] *= 1000
d = d.merge(now, on=['grid_id', 'scenario'], how='left')
cols = ['dEm', 'dEm_C', 'dEm_L', 'dEm_pct', 'dEmPct_C', 'dEmPct_L', 'dCap_pct', 'dCap_C', 'dCap_L']
m = d.groupby('scenario')[cols].median()


def rng(s, f='{:.0f}'):
    a, b = f.format(s.min()), f.format(s.max())
    return a if a == b else f'{a}–{b}'


print('各情境中位数范围（11 情境）：')
for c in cols:
    print(f'  {c}: {rng(m[c], "{:.1f}")}')
print('按用途：')
for u in ORDER:
    s = m[[k.endswith(u) for k in m.index]]
    print(f'  {USE[u]}: 减排 现状 {rng(s.dEm, "{:.1f}")} / 可行 {rng(s.dEm_C, "{:.1f}")} / 放松 {rng(s.dEm_L, "{:.1f}")} kg；'
          f'接入容量 现状 {rng(s.dCap_pct, "{:.1f}")} / 可行 {rng(s.dCap_C, "{:.1f}")} / 放松 {rng(s.dCap_L, "{:.1f}")}%')
print('合并中位数：', d[cols].median().round(1).to_dict())
print('可行解减排 > 现状 的比例 %.1f%%，放松解 > 现状 %.1f%%；可行解减排为正 %.1f%%' % (
    100 * (d.dEm_C > d.dEm).mean(), 100 * (d.dEm_L > d.dEm).mean(), 100 * (d.dEm_C > 0).mean()))
m.to_csv(HERE / 'out/downsizing-phys-by-scenario.csv', encoding='utf-8-sig')
d.to_parquet(HERE / 'out/downsizing-phys.parquet', index=False)
json.dump({c: [float(m[c].min()), float(m[c].max())] for c in cols}, open(HERE / 'out/down-ranges.json', 'w'), indent=1)
