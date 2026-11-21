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

# ① 上限
fr = load(f'{U}/full-i0/runs.jsonl')
fr = fr[fr['mode'] == 'free'].set_index(['grid_id', 'scenario']).reindex(s.index)
ok = fr.total_cost_usd.notna()
print('上限对数', int(ok.sum()), '/', len(s))
up = pd.DataFrame({'cut': 100 * (-fr.extra_task_mw / fr.K_mw),
                   'V_total': (s.total_cost_usd - fr.total_cost_usd) / e0,          # 含少买卡（cost_cards 为负）
                   'V_supply': (s.total_cost_usd - (fr.total_cost_usd - fr.cost_cards)) / e0,
                   'dEm': 1000 * (s.emissions_t - fr.emissions_t) / e0,
                   'dCap': 100 * (s.grid_cap_mw - fr.grid_cap_mw) / s.grid_cap_mw})[ok]
m_now, m_up = now[ok].groupby(level=1).median(), up.groupby(level=1).median()
ratio = (up[['dEm', 'dCap']] / now[ok][['dEm', 'dCap']]).where(now[ok][['dEm', 'dCap']].abs() > 1e-6).groupby(level=1).median()
print('减卡比例 %：', by_use(m_up, 'cut', '{:.0f}'))
print('上限 总节省（含少买卡）：', rng(m_up.V_total, '{:.0f}'), '|', by_use(m_up, 'V_total', '{:.0f}'))
print('上限 供能侧节省：', rng(m_up.V_supply), '| 现状', rng(m_now.V))
print('减排 kg/MWh 现状', rng(m_now.dEm), '上限', rng(m_up.dEm), '| 上限按用途', by_use(m_up, 'dEm'))
print('接入容量削减 % 现状', rng(m_now.dCap), '上限', rng(m_up.dCap), '| 上限按用途', by_use(m_up, 'dCap'))
print('上限/现状（逐对中位数） 减排', by_use(ratio, 'dEm', '{:.2f}'), '| 接入容量', by_use(ratio, 'dCap', '{:.2f}'))
print('合并中位数 现状', now[ok].median().round(2).to_dict(), '上限', up.median().round(2).to_dict())
print('上限减排高于现状比例 %.1f%%，接入容量削减高于现状 %.1f%%' % (100 * (up.dEm > now[ok].dEm).mean(), 100 * (up.dCap > now[ok].dCap).mean()))
up.join(now[ok], rsuffix='_now').reset_index().to_parquet(f'{U}/upper-i0-paired.parquet', index=False)

