#!/usr/bin/env bash
# 接入容量费敏感性：每 10 个网格取 1 个 × 11 情境 × 不接推理，fee 取参数。用法 launch-fee.sh <fee> <workers>
set -euo pipefail
N=/data3/dataset/ljt/upper-flex-20261121
R=/data3/dataset/ljt/global-flex-econ-20261103
cd $N
export COPT_LICENSE_DIR=$R/license OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_CACHE_DIR=$N/numba-cache
exec nice -n 10 $R/venv/bin/python -u code/residual_flex_global.py --config $R/project/analysis/global-flex-econ-20261103.json --project $R/project --asi inputs/asi-D-INF-native-hourly.parquet --max-table code/max-intensity.csv --intensities 0 --fee $1 --grid-ids inputs/fee-grids.json --out fee-$1 --workers $2
