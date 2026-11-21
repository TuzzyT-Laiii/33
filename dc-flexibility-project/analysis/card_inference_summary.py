"""汇总 card_inference.py 结果：加卡算例数、减卡比例（八城市）与全球角点检验。

用法：python card_inference_summary.py --runs full-city/runs.jsonl --kind city --out summary-city
      python card_inference_summary.py --runs full-global/runs.jsonl --kind global --out summary-global
加卡判定：extra_task_mw > 1e-6 MW（与第四节 card-learning-city-summary 的口径一致）。
"""
import argparse
from pathlib import Path

import pandas as pd

COLS = ['region', 'scenario', 'cost_scenario', 'intensity_of_max', 'mode', 'm', 'idle_scale', 'status', 'K_mw', 'extra_task_mw',
        'total_cost_usd', 'cost_cards', 'max_residual']


def load(path):
    parts = []
    for chunk in pd.read_json(path, lines=True, chunksize=200_000, dtype=False):
        parts.append(chunk.reindex(columns=COLS))
    r = pd.concat(parts, ignore_index=True)
    r['intensity_of_max'] = r.intensity_of_max.astype(float).round(4)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--kind', choices=('city', 'global'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    r = load(a.runs)
    status = r.groupby(['mode', 'status'], dropna=False).size().rename('rows').reset_index()
    status.to_csv(a.out / 'status.csv', index=False)
    ok = r[r.status == 'optimal'].copy()
    ok['adds'] = ok.extra_task_mw > 1e-6
    ok['cut_share'] = -ok.extra_task_mw / ok.K_mw
    add = ok[ok['mode'] == 'add']
    by = ['cost_scenario', 'm', 'idle_scale', 'intensity_of_max']
    t = add.groupby(by).agg(cases=('adds', 'size'), cases_adding_cards=('adds', 'sum'), max_extra_share=('extra_task_mw', 'max')).reset_index()
    t.to_csv(a.out / 'add-cases.csv', index=False)
    ts = add.groupby(['scenario', 'intensity_of_max']).agg(cases=('adds', 'size'), cases_adding_cards=('adds', 'sum')).reset_index()
    ts.to_csv(a.out / 'add-cases-by-scenario.csv', index=False)
    lines = [f'rows {len(r)}, optimal {len(ok)}, max residual {ok.max_residual.max():.2e}',
             f'add cases {len(add)}, adding cards {int(add.adds.sum())}']
    if a.kind == 'city':
        free = ok[ok['mode'] == 'free']
        f = free.groupby(['scenario', 'cost_scenario', 'm', 'idle_scale', 'intensity_of_max']).agg(
            cases=('cut_share', 'size'), cases_reducing=('extra_task_mw', lambda s: int((s < -1e-6).sum())),
            cut_share_median=('cut_share', 'median'), cut_share_min=('cut_share', 'min'), cut_share_max=('cut_share', 'max')).reset_index()
        f.to_csv(a.out / 'free-cut-share.csv', index=False)
        lines.append(f'free cases {len(free)}, reducing {int((free.extra_task_mw < -1e-6).sum())}')
        # 不接推理时的减卡比例 × (1 − 强度占上限比例) 的线性近似误差
        base = free[free.intensity_of_max == 0].set_index(['region', 'scenario', 'cost_scenario', 'm', 'idle_scale']).cut_share
        j = free.join(base.rename('cut0'), on=['region', 'scenario', 'cost_scenario', 'm', 'idle_scale'])
        j = j[j.intensity_of_max > 0]
        dev = (j.cut_share - j.cut0 * (1 - j.intensity_of_max)).abs()
        lines.append(f'linear approx |cut - cut0*(1-level)|: median {dev.median():.4f}, p99 {dev.quantile(.99):.4f}, max {dev.max():.4f}')
    (a.out / 'summary.txt').write_text('\n'.join(lines) + '\n', 'utf-8')
    print('\n'.join(lines))
    print(ts.pivot(index='scenario', columns='intensity_of_max', values='cases_adding_cards').to_string())


if __name__ == '__main__':
    main()
