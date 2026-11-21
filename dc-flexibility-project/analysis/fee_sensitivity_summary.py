"""汇总：① 1.3 上限（全球“允许增减卡 + 联合调度”，不接推理）；② 接入容量费 0/60/240 敏感性（与 120 主结果对照）。
口径同结果稿：每对 = 不做柔性（seq）− 做柔性（joint / free），÷ E0（同网格、同情境 seq 设施用电）。"""
import json
import sys

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
U = '/data3/dataset/ljt/upper-flex-20261121'
F = '/data3/dataset/ljt/residual-flex-global-20261114/full/runs.jsonl'
USE = {'CLD': '云计算', 'HPC': 'HPC', 'INF': '推理', 'SAI': '科研 AI', 'TRN': '训练'}
ORDER = ['TRN', 'HPC', 'SAI', 'CLD', 'INF']
C = ['grid_id', 'region', 'scenario', 'mode', 'status', 'intensity_of_max', 'total_cost_usd', 'emissions_t', 'grid_cap_mw',
     'facility_mwh', 'extra_task_mw', 'K_mw', 'grid_mwh', 'cost_cards', 'cost_grid_capacity']


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get('status') == 'optimal' and float(d.get('intensity_of_max') or 0) == 0:
                rows.append({k: d.get(k) for k in C})
    r = pd.DataFrame(rows)
    if 'region' in r and r.grid_id.isna().all():
        r['grid_id'] = r.region
    return r


def pair(r):
    s = r[r['mode'] == 'seq'].set_index(['grid_id', 'scenario'])
    j = r[r['mode'] == 'joint'].set_index(['grid_id', 'scenario'])
    s, j = s.align(j, join='inner')
    return s, j


def rng(x, f='{:.1f}'):
    a, b = f.format(x.min()), f.format(x.max())
    return a if a == b else f'{a}–{b}'


def by_use(m, c, f='{:.1f}'):
    return '; '.join(f'{USE[u]} {rng(m[[k.endswith(u) for k in m.index]][c], f)}' for u in ORDER)


base = load(F)
s, j = pair(base)
e0 = s.facility_mwh
now = pd.DataFrame({'V': (s.total_cost_usd - j.total_cost_usd) / e0, 'dEm': 1000 * (s.emissions_t - j.emissions_t) / e0,
                    'dCap': 100 * (s.grid_cap_mw - j.grid_cap_mw) / s.grid_cap_mw})

# ② 接入容量费
print('\n== 接入容量费敏感性（不接推理）')
rows = {120: now}
for fee in (0, 60, 240):
    s2, j2 = pair(load(f'{U}/fee-{fee}/runs.jsonl'))
    e = s2.facility_mwh
    rows[fee] = pd.DataFrame({'V': (s2.total_cost_usd - j2.total_cost_usd) / e, 'dEm': 1000 * (s2.emissions_t - j2.emissions_t) / e,
                              'dCap': 100 * (s2.grid_cap_mw - j2.grid_cap_mw) / s2.grid_cap_mw.where(s2.grid_cap_mw > 1e-6),
                              'cap_seq_rel': s2.grid_cap_mw / (s2.facility_mwh / (s2.facility_mwh * 0 + 1))})
out = []
for fee in (0, 60, 120, 240):
    x = rows[fee]
    m = x.groupby(level=1).median()
    print(f'  fee {fee:3d}: 对数 {len(x)}; 节省 {rng(m.V)} 美元/MWh（合并 {x.V.median():.1f}）；减排 {rng(m.dEm)} kg/MWh（合并 {x.dEm.median():.1f}）；'
          f'接入容量削减 {rng(m.dCap)}%（合并 {x.dCap.median():.1f}）')
    print('     接入容量削减按用途：', by_use(m, 'dCap'))
    out.append(m.assign(fee=fee))
pd.concat(out).to_csv(f'{U}/fee-by-scenario.csv')
