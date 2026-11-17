import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'analysis'))
from residual_flex_pilot import solve, demand, asi_inference, load_config
INP = 'D:/codex-runs/existing-compute-capacity-20261102/worktree/.codex-temp/calendar-rerun/out-validated/inputs'
ASI = 'D:/codex-runs/brief-ch7-card-cost-20261102/worktree/dc-flexibility-project/results/flex-class-20261101/native-hourly/D-INF.parquet'

def one(t):
    city, I, sh = t
    p, _ = load_config(W)
    f = pd.read_parquet(f'{INP}/{city}-D-TRN.parquet')
    on, off, _ = demand(f, asi_inference(ASI), I, 'real')
    a, xs = solve(f, p, city, on, off, fee=120, mode='asap', split_h=sh)
    s, _ = solve(f, p, city, on, off, fee=120, mode='cost', fixed_x=xs, split_h=sh)
    j, _ = solve(f, p, city, on, off, fee=120, mode='cost', split_h=sh)
    return dict(city=city, I=I, split_h=sh, V=s['total_cost_usd'] - j['total_cost_usd'])

if __name__ == '__main__':
    tasks = [(c, i, h) for c in ('riyadh', 'frankfurt', 'singapore') for i in (0, 0.5) for h in (6, 12, 24, 48)]
    with ProcessPoolExecutor(14) as ex:
        df = pd.DataFrame(list(ex.map(one, tasks)))
    t = df.pivot_table(index=['city', 'split_h'], columns='I', values='V')
    t['R'] = t[0.5] / t[0]
    t['V0_vs_48h'] = t[0] / t.xs(48, level='split_h')[0].reindex(t.index.get_level_values(0)).values
    print(t.round(3).to_string())
