"""2025 年逐时电网碳强度（生命周期，gCO2/kWh）：德国、瑞典（Energy-Charts public_power，15 分钟 → 小时），
美国 EIA-930 平衡区（US48、PJM、AZPS 等，逐时分燃料净发电）。按发电结构 × IPCC AR5 生命周期中位数因子计算，不计进出口。
输入目录为下载数据（只读），输出到本脚本目录 out/。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
SRC = Path(sys.argv[1])
OUT = Path(__file__).parent / 'out'
OUT.mkdir(exist_ok=True)
# IPCC AR5 WGIII 附录 III 生命周期中位数（gCO2eq/kWh）；油取 650、其他化石 700（AR5 未列，与 Ember 口径相近）
EF = {'coal': 820, 'lignite': 820, 'gas': 490, 'oil': 650, 'other_fossil': 700, 'biomass': 230, 'waste': 700,
      'nuclear': 12, 'hydro': 24, 'wind': 11, 'solar': 48, 'geothermal': 38, 'other_re': 30, 'unknown': 490, 'storage': 0}
EC = {'Fossil brown coal / lignite': 'lignite', 'Fossil hard coal': 'coal', 'Fossil gas': 'gas', 'Fossil oil': 'oil',
      'Fossil coal-derived gas': 'coal', 'Others': 'other_fossil', 'Waste': 'waste', 'Biomass': 'biomass', 'Nuclear': 'nuclear',
      'Hydro Run-of-River': 'hydro', 'Hydro water reservoir': 'hydro', 'Hydro pumped storage': 'storage', 'Hydro': 'hydro',
      'Wind offshore': 'wind', 'Wind onshore': 'wind', 'Wind': 'wind', 'Solar': 'solar', 'Geothermal': 'geothermal',
      'Renewable share of generation': None, 'Renewable share of load': None, 'Load': None, 'Residual load': None,
      'Cross border electricity trading': None, 'Hydro pumped storage consumption': None, 'Battery Storage (discharge)': 'storage',
      'Battery Storage (charge)': None}


def energy_charts(country):
    parts = []
    for f in sorted(SRC.glob(f'{country}-2025-q*.json')):
        j = json.loads(f.read_text('utf-8'))
        t = pd.to_datetime(j['unix_seconds'], unit='s', utc=True)
        df = pd.DataFrame({x['name']: x['data'] for x in j['production_types']}, index=t)
        parts.append(df)
    df = pd.concat(parts)
    df = df[~df.index.duplicated()].apply(pd.to_numeric, errors='coerce').clip(lower=0)
    unknown = sorted(set(df.columns) - set(EC))
    assert not unknown, unknown
    gen = {}
    for c, k in EC.items():
        if k and c in df:
            gen[k] = gen.get(k, 0) + df[c].fillna(0)
    g = pd.DataFrame(gen).resample('1h').mean()
    return g


def eia(bas):
    # 用 EIA 清洗后的 (Adjusted) 列；2025 年起光伏、风电按是否配储拆列，储能放电记 0 排放
    key = [('Coal', 'coal'), ('Natural Gas', 'gas'), ('Nuclear', 'nuclear'), ('Petroleum', 'oil'), ('Hydropower', 'hydro'),
           ('Pumped Storage', 'storage'), ('Solar', 'solar'), ('Wind', 'wind'), ('Battery', 'storage'), ('Other Energy Storage', 'storage'),
           ('Unknown Energy Storage', 'storage'), ('Geothermal', 'geothermal'), ('Other Fuel', 'other_fossil'), ('Unknown Fuel', 'unknown')]
    head = pd.read_csv(next(SRC.glob('EIA930_BALANCE_2025_*.csv')), nrows=1).columns
    cols = {}
    for c in head:
        if c.startswith('Net Generation (MW) from') and c.endswith('(Adjusted)'):
            cols[c] = next(v for k, v in key if k in c)
    assert len(cols) == 16, cols
    parts = [pd.read_csv(f, usecols=['Balancing Authority', 'UTC Time at End of Hour'] + list(cols), thousands=',', low_memory=False)
             for f in sorted(SRC.glob('EIA930_BALANCE_2025_*.csv'))]
    d = pd.concat(parts)
    d['t'] = pd.to_datetime(d['UTC Time at End of Hour'], format='%m/%d/%Y %I:%M:%S %p', utc=True) - pd.Timedelta(hours=1)
    out = {}
    for ba in bas:
        if ba == 'US48':
            x = d.groupby('t')[list(cols)].sum(min_count=1)
        else:
            x = d[d['Balancing Authority'] == ba].set_index('t')[list(cols)]
        x = x.apply(pd.to_numeric, errors='coerce').clip(lower=0).fillna(0)
        out[ba] = x.T.groupby(pd.Series(cols)).sum().T
    return out, sorted(d['Balancing Authority'].unique())


def ci(g):
    w = sum(g[k] * EF[k] for k in g.columns)
    tot = g.drop(columns=[c for c in g.columns if c == 'storage']).sum(axis=1)
    return (w / tot).where(tot > 0)


series = {}
for c in ('de', 'se'):
    series[c.upper()] = ci(energy_charts(c))
us, bas = eia(['US48', 'PJM', 'AZPS', 'SRP', 'ERCO', 'CISO', 'MISO'])
for ba, g in us.items():
    series['US-' + ba] = ci(g)
idx = pd.date_range('2025-01-01', '2025-12-31 23:00', freq='1h', tz='UTC')
S = pd.DataFrame({k: v.reindex(idx) for k, v in series.items()})
print('缺失小时：', S.isna().sum().to_dict())
S = S.interpolate(limit=6).ffill().bfill()
S.index.name = 'time_utc'
S.to_csv(OUT / 'hourly-ci-2025.csv')
print('平衡区数', len(bas))
for k in S:
    s = S[k]
    local = s.groupby(s.index.hour).mean()
    daily_cv = s.groupby(s.index.date).agg(lambda x: x.std() / x.mean()).median()
    print(f'{k:9s} 均值 {s.mean():5.0f}  P10–P90 {s.quantile(.1):4.0f}–{s.quantile(.9):4.0f}  日内变异系数中位 {daily_cv:.2f}  '
          f'UTC 最低时 {local.idxmin():2d} 最高时 {local.idxmax():2d}')
