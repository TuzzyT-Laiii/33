set -euo pipefail
run_root="$1"
inputs="$2"
mkdir -p "$run_root/scratch" "$run_root/logs"
export TMPDIR="$run_root/scratch"
export MPLCONFIGDIR="$run_root/scratch/matplotlib"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$run_root"
"$inputs/.codex-temp/load-curves-v3-20261101/.venv/bin/python" project/analysis/build_native_combined.py --project "$run_root/project" --inputs "$inputs"
