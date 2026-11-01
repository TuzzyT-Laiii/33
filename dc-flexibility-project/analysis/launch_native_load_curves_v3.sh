#!/usr/bin/env bash
set -euo pipefail
run_root="$1"
source_project="$2"
cd "$run_root"
mkdir -p logs scratch project/.codex-temp
export TMPDIR="$run_root/scratch"
export MPLCONFIGDIR="$run_root/scratch/matplotlib"
export NUMBA_CACHE_DIR="$run_root/scratch/numba"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec 9>"$run_root/native-calculation.lock"
flock -n 9
exec "$run_root/.venv/bin/python" project/analysis/build_native_load_curves_v3.py \
  --project "$run_root/project" --source-project "$source_project" \
  --baseline "$run_root/inputs/v2" --weather "$run_root/inputs/weather" \
  --mit-hourly "$run_root/inputs/native/research-gpu-observed-hourly.csv" \
  --mit-reference "$run_root/inputs/native" --input-commit ab07a41157442e837659535e1a33579ced5af8fa
