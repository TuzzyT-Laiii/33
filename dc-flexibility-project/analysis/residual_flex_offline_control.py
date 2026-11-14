"""对照：离线推理不允许推迟（off_h=0），分离离线推理自身带来的柔性。"""
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
    city, scen, I = t
    p, _ = load_config(W)
    f = pd.read_parquet(f'{INP}/{city}-{scen}.parquet')
    on, off, _ = demand(f, asi_inference(ASI), I, 'real')
    a, xs = solve(f, p, city, on, off, fee=120, mode='asap', off_h=0)
    if xs is None:
        return dict(city=city, scenario=scen, intensity=I, status=a['status'])
    s, _ = solve(f, p, city, on, off, fee=120, mode='cost', fixed_x=xs, off_h=0)
    j, _ = solve(f, p, city, on, off, fee=120, mode='cost', off_h=0)
    return dict(city=city, scenario=scen, intensity=I, status=s['status'] + '/' + j['status'],
                seq=s.get('total_cost_usd'), joint=j.get('total_cost_usd'))

if __name__ == '__main__':
    tasks = [(c, s, i) for c in ('riyadh', 'frankfurt', 'singapore') for s in ('B-TRN', 'D-TRN') for i in (0.25, 0.5, 0.75)]
    with ProcessPoolExecutor(14) as ex:
        rows = list(ex.map(one, tasks))
    pd.DataFrame(rows).to_csv(W / 'results/residual-flex-pilot-20261114/offline-not-deferrable.csv', index=False)
    print(pd.DataFrame(rows).to_string())
