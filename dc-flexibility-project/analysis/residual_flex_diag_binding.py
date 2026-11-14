import sys
from pathlib import Path
import numpy as np, pandas as pd
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'analysis'))
import residual_flex_pilot as m
INP = 'D:/codex-runs/existing-compute-capacity-20261102/worktree/.codex-temp/calendar-rerun/out-validated/inputs'
ASI = 'D:/codex-runs/brief-ch7-card-cost-20261102/worktree/dc-flexibility-project/results/flex-class-20261101/native-hourly/D-INF.parquet'
p, _ = m.load_config(W)
asi = m.asi_inference(ASI)
for city in ('riyadh', 'frankfurt', 'singapore'):
    f = pd.read_parquet(f'{INP}/{city}-D-TRN.parquet')
    for I in (0, 0.5):
        on, off, _ = m.demand(f, asi, I, 'real')
        s, h = m.solve(f, p, city, on, off, fee=120, mode='cost')
        task0 = (f.rigid + f.unknown + f.whole + f.split).to_numpy()
        headroom = task0.max() - f.rigid.to_numpy() - f.unknown.to_numpy() - on
        x = h['x_whole'] + h['x_split'] + h['x_off']
        bind = x >= headroom - 1e-4
        gen = s['pv_mw'] * f.cf_pv.to_numpy() + s['wind_mw'] * f.cf_wind.to_numpy()
        surplus = gen > h['facility'] + 1e-6
        print(city, I, 'cap-binding hours %.2f' % bind.mean(), '| RE>load hours %.2f' % surplus.mean(),
              '| surplus hours binding %.2f' % ((bind & surplus).sum() / max(surplus.sum(), 1)),
              '| grid-hours binding %.2f' % ((bind & (h['grid'] > 1e-6)).sum() / max((h['grid'] > 1e-6).sum(), 1)))
