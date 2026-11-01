#!/usr/bin/env bash
set -euo pipefail
run_root="$1"
python_exe="$2"
base_commit="$3"
cd "$run_root"
mkdir -p logs scratch
export TMPDIR="$run_root/scratch" MPLCONFIGDIR="$run_root/scratch/matplotlib" NUMBA_CACHE_DIR="$run_root/scratch/numba"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
bash project/analysis/launch_load_curves_v3.sh "$run_root" "$python_exe" "$base_commit" full
"$python_exe" project/analysis/verify_typical_load_curves_v3.py --project project --baseline inputs/v2
"$python_exe" project/analysis/summarize_flexibility_v3.py --project project
"$python_exe" project/analysis/plot_typical_load_curves_v3.py --project project
"$python_exe" project/analysis/report_load_curves_v3.py --project project
printf 'finalization_complete\n'
