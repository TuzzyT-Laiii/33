#!/usr/bin/env bash
# 1.3 上限重算：全球“允许增减卡（卡价 1 倍、待机 1 倍）+ 联合调度”，2024 成本、碳价 100。用法 launch-upper.sh <out> <intensities> <workers> [extra args]
set -euo pipefail
N=/data3/dataset/ljt/upper-flex-20261121
R=/data3/dataset/ljt/global-flex-econ-20261103
cd $N
export COPT_LICENSE_DIR=$R/license OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_CACHE_DIR=$N/numba-cache
out=$1; lv=$2; w=$3; shift 3
exec nice -n 10 $R/venv/bin/python -u code/card_inference.py --kind global --config $R/project/analysis/global-flex-econ-20261103.json --project $R/project --asi inputs/asi-D-INF-native-hourly.parquet --max-table code/max-intensity.csv --cost-scenarios code/learning-curves-atb2025.json --modes free --m 1 --idle 1 --costs base-2024 --intensities $lv --out $out --workers $w "$@"
