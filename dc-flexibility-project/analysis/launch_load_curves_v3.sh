#!/usr/bin/env bash
set -euo pipefail
run_root="$1"
python_exe="$2"
base_commit="$3"
stage="${4:-full}"
cd "$run_root"
mkdir -p logs scratch project/.codex-temp
export TMPDIR="$run_root/scratch"
export MPLCONFIGDIR="$run_root/scratch/matplotlib"
export NUMBA_CACHE_DIR="$run_root/scratch/numba"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec "$python_exe" project/analysis/build_typical_load_curves_v3.py \
  --project "$run_root/project" --baseline "$run_root/inputs/v2" \
  --weather "$run_root/inputs/weather" --base-commit "$base_commit" --stage "$stage"
