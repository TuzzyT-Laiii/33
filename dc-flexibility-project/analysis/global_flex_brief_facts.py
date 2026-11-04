"""简表第八章（全球 2° 抽样网格）所需的表格数字。

读取 results/global-flex-econ-20261103/grid-summary.parquet，按有效网格等权取中位数，
写出 results/global-flex-econ-20261103/brief/brief-facts.json，供 docs/job-flexibility-brief/build.js 使用。
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'results/global-flex-econ-20261103'
SCENARIOS = [('A-HPC', '纯 CPU 型·科研计算'), ('A-CLD', '纯 CPU 型·云通用计算'), ('D-SAI', 'GPU 主导型·科研 AI'),
             ('D-TRN', 'GPU 主导型·大模型训练/开发'), ('D-INF', 'GPU 主导型·推理服务')]
TARGETS = [(None, '经济最优'), (0.95, '95%'), (1.0, '100%')]
CONTINENTS = {'Africa': '非洲', 'Asia': '亚洲', 'Europe': '欧洲', 'North America': '北美', 'Oceania': '大洋洲', 'South America': '南美'}
COUNTRIES = [('CHN', '中国'), ('USA', '美国'), ('IND', '印度'), ('BRA', '巴西'), ('AUS', '澳大利亚'), ('SAU', '沙特')]


def fmt(value):
    return f'{value:.1f}'


def main():
    table = RESULT / 'grid-summary.parquet'
    grid = pd.read_parquet(table)
    valid = grid.loc[grid.baseline_status.eq('optimal') & grid.flex_status.eq('optimal')].copy()
    assert (valid.savings_pct > -1e-6).all(), '存在负的柔性节省'
    pick = lambda frame, alpha: frame.loc[frame.alpha.isna()] if alpha is None else frame.loc[frame.alpha.eq(alpha)]
    eight = json.loads((ROOT / 'results/flex-card-cost-calendar-20261102/brief/brief-facts.json').read_text('utf-8'))
    eight_plan = {row[0]: row[2].split('／')[0] for row in eight['plan_rows'][1:]}

    rows = [['代表情境', '规划阶段 柔性收益%\n经济最优／95%／100%', '规划阶段 经济最优\n四分位 P25–P75', '已建成 柔性收益%\n经济最优／95%', '对照：八城市中位\n规划·经济最优']]
    for scenario, name in SCENARIOS:
        current = valid.loc[valid.scenario.eq(scenario)]
        plan, built = current.loc[current.boundary.eq('planning')], current.loc[current.boundary.eq('fixed')]
        econ = pick(plan, None).savings_pct
        rows.append([name,
                     '／'.join(fmt(pick(plan, a).savings_pct.median()) for a, _ in TARGETS),
                     fmt(econ.quantile(.25)) + '–' + fmt(econ.quantile(.75)),
                     '／'.join(fmt(pick(built, a).savings_pct.median()) for a, _ in TARGETS[:2]),
                     eight_plan[name]])

    train = pick(valid.loc[valid.scenario.eq('D-TRN') & valid.boundary.eq('planning')], None)
    region_rows = [['训练·规划·经济最优'] + [label for label in CONTINENTS.values()] + [label for _, label in COUNTRIES],
                   ['中位节省%'] + [fmt(train.loc[train.continent.eq(key)].savings_pct.median()) for key in CONTINENTS]
                   + [fmt(train.loc[train.iso3.eq(code)].savings_pct.median()) for code, _ in COUNTRIES],
                   ['网格数'] + [str(int(train.continent.eq(key).sum())) for key in CONTINENTS]
                   + [str(int(train.iso3.eq(code).sum())) for code, _ in COUNTRIES]]
    top, low = train.nlargest(1, 'savings_pct').iloc[0], train.nsmallest(1, 'savings_pct').iloc[0]

    facts = {
        'source_table': str(table.relative_to(ROOT)).replace('\\', '/'),
        'source_sha256': hashlib.sha256(table.read_bytes()).hexdigest(),
        'grids_sampled': int(grid.grid_id.nunique()),
        'grids_valid': int(valid.grid_id.nunique()),
        'grids_no_land': int(grid.loc[grid.baseline_status.eq('input_no_land_points')].grid_id.nunique()),
        'table_rows': rows,
        'region_rows': region_rows,
        'train_plan_econ': {'median': round(float(train.savings_pct.median()), 2),
                            'p25': round(float(train.savings_pct.quantile(.25)), 2), 'p75': round(float(train.savings_pct.quantile(.75)), 2),
                            'max': [top.country, float(top.lat), float(top.lon), round(float(top.savings_pct), 1)],
                            'min': [low.country, float(low.lat), float(low.lon), round(float(low.savings_pct), 1)]},
        'fixed_alpha100_max_pct': float(pick(valid.loc[valid.boundary.eq('fixed')], 1.0).savings_pct.max()),
        'extra_task_mw_max': float(valid.extra_task_mw.max()),
    }
    out = RESULT / 'brief/brief-facts.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(facts, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: facts[k] for k in ('grids_sampled', 'grids_valid', 'grids_no_land', 'train_plan_econ', 'fixed_alpha100_max_pct', 'extra_task_mw_max')}, ensure_ascii=False))
    for row in rows + region_rows:
        print(row)


if __name__ == '__main__':
    main()
