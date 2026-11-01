#!/usr/bin/env bash
set -euo pipefail
run_root="$1"
source_project="$2"
cd "$run_root"
export TMPDIR="$run_root/scratch"
export MPLCONFIGDIR="$run_root/scratch/matplotlib"
export NUMBA_CACHE_DIR="$run_root/scratch/numba"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python_exe="$run_root/.venv/bin/python"
exec 9>"$run_root/native-calculation.lock"
flock -s 9
"$python_exe" project/analysis/verify_native_load_curves_v3.py \
  --project "$run_root/project" --source-project "$source_project" \
  --baseline "$run_root/inputs/v2" --weather "$run_root/inputs/weather" \
  --mit-hourly "$run_root/inputs/native/research-gpu-observed-hourly.csv"
"$python_exe" project/analysis/plot_native_load_curves_v3.py --project "$run_root/project"
"$python_exe" project/analysis/report_native_load_curves_v3.py --project "$run_root/project"
"$python_exe" project/analysis/publish_native_load_curves_v3.py --project "$run_root/project" --destination "$source_project"
