set -euo pipefail
run_root="$1"
inputs="$2"
cd "$run_root"
export TMPDIR="$run_root/scratch"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for attempt in $(seq 1 120); do
  if [ -s project/results/typical-load-curves-20261101-v3-native-combined/manifest.json ]; then
    exec "$inputs/.codex-temp/load-curves-v3-20261101/.venv/bin/python" project/analysis/verify_native_combined.py --project "$run_root/project" --inputs "$inputs"
  fi
  sleep 15
done
exit 1
