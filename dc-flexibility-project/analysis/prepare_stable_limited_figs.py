"""为结果图准备小体量输入：逐网格基准用电与供能成本，以及第一节机群费用摘要。

- baseline-energy.parquet：第三节全量 runs-compact.parquet 中 I=0、顺序调度（不协同、不卖推理）的
  窗口设施用电与供能总成本（4,842 网格 × 11 情境）。科研 AI 与 HPC 为与 ASI 推理重叠的窗口。
- part1-summary.csv：PR #60（ab72545）results/global-flex-econ-20261103/part1/part1-summary.csv 原样复制，
  提供峰值余量与 10 MW 参考机群年化设备费用。

用法（仓库根目录）：
  python dc-flexibility-project/analysis/prepare_stable_limited_figs.py \
      --runs <runs-compact.parquet> --part1 <part1-summary.csv>
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

from stable_limited_figs_common import INPUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True, type=Path)
    ap.add_argument('--part1', required=True, type=Path)
    args = ap.parse_args()
    INPUT.mkdir(parents=True, exist_ok=True)
    r = pd.read_parquet(args.runs, columns=['grid_id', 'scenario', 'mode', 'status', 'intensity_of_max', 'hours',
                                            'facility_mwh', 'total_cost_usd', 'grid_mwh', 'moved_whole_mwh',
                                            'moved_split_mwh', 'moved_off_mwh'])
    b = r[(r['mode'] == 'seq') & (r.intensity_of_max == 0)]
    assert (b.status == 'optimal').all() and len(b) == 4842 * 11
    b = b.rename(columns={'hours': 'hours0', 'facility_mwh': 'facility_mwh0', 'total_cost_usd': 'supply_cost0_usd',
                          'grid_mwh': 'grid_mwh0'})
    b = b[['grid_id', 'scenario', 'hours0', 'facility_mwh0', 'supply_cost0_usd', 'grid_mwh0']].sort_values(['scenario', 'grid_id'])
    # 联合调度转移的电量：原任务（整体 + 可拆分）与离线推理
    j = r[r['mode'] == 'joint'].assign(moved_task_mwh=lambda d: d.moved_whole_mwh + d.moved_split_mwh)
    j[['grid_id', 'scenario', 'intensity_of_max', 'moved_task_mwh', 'moved_off_mwh']].sort_values(
        ['scenario', 'grid_id', 'intensity_of_max']).to_parquet(INPUT / 'moved-energy.parquet', index=False)
    b.to_parquet(INPUT / 'baseline-energy.parquet', index=False)
    shutil.copyfile(args.part1, INPUT / 'part1-summary.csv')
    print(b.groupby('scenario').agg(hours=('hours0', 'first'),
                                    supply_usd_mwh=('supply_cost0_usd', lambda x: (x / b.loc[x.index, 'facility_mwh0']).median())))


if __name__ == '__main__':
    main()
