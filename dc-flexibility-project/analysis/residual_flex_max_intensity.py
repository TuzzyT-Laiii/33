"""每个情境可承接的最大推理需求强度（算力侧可行性与网格无关，取任一网格帧）。"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
R = Path('/data3/dataset/ljt/global-flex-econ-20261103')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import residual_flex_global as g

def one(scenario):
    g.initialize(R / 'project/analysis/global-flex-econ-20261103.json', R / 'project', '/data3/dataset/ljt/residual-flex-global-20261114/inputs/asi-D-INF-native-hourly.parquet', 120., [0.])
    from residual_flex_pilot import solve
    frame = g.build_frame('g05_r077_c377', scenario)
    frame, on_raw, off_raw, dropped, shift = g.align(frame, scenario)
    task0 = (frame.rigid + frame.unknown + frame.whole + frame.split).to_numpy()
    idle = task0.max() - task0.mean()
    unit = idle / (on_raw.mean() + off_raw.mean())
    p = dict(g.CONTEXT['parameters']); p['emission_factor'] = {'x': 0.3}
    def ok(I):
        a, xs = solve(frame, p, 'x', I * unit * on_raw, I * unit * off_raw, fee=120., mode='asap', solver='copt')
        return xs is not None
    lo, hi = 0., 1.
    for _ in range(14):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    online_only = (task0.max() - (frame.rigid + frame.unknown).to_numpy()) / (unit * on_raw)
    return dict(scenario=scenario, max_intensity=round(lo, 4), online_peak_limit=round(float(online_only.min()), 4),
                hours=len(frame), idle_mean_mw=round(idle, 3), idle_share_of_K=round(idle / task0.max(), 3))

if __name__ == '__main__':
    sc = ["A-HPC", "A-CLD", "B-SAI", "C-SAI", "D-SAI", "B-TRN", "C-TRN", "D-TRN", "B-INF", "C-INF", "D-INF"]
    with ProcessPoolExecutor(11) as ex:
        rows = list(ex.map(one, sc))
    import pandas as pd
    df = pd.DataFrame(rows); print(df.to_string())
    df.to_csv('/data3/dataset/ljt/residual-flex-global-20261114/max-intensity.csv', index=False)
