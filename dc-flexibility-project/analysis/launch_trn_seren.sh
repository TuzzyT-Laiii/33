set -euo pipefail
shared=/data3/dataset/ljt/global-grid-flexibility-20261023
run="$shared/.codex-temp/trn-seren-20261101"
project="$run/project"
python="$shared/.codex-temp/load-curves-v3-20261101/.venv/bin/python"
mkdir -p "$run/scratch" "$run/cache" "$run/deps"
export TMPDIR="$run/scratch" XDG_CACHE_HOME="$run/cache" PIP_CACHE_DIR="$run/cache/pip"
export PYTHONPATH="$run/deps" PYTHONUNBUFFERED=1 MPLCONFIGDIR="$run/cache/matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec 9>"$run/calculation.lock"
flock -n 9
"$python" -m pip install --target "$run/deps" --no-deps 'statsmodels==0.14.6' 'patsy==1.0.2'
"$python" "$project/analysis/prepare_trn_seren.py" --project "$project" --jobs "$shared/data/raw/trn-power-model-20261101/github/data/job_trace/trace_seren.csv"
"$python" "$project/analysis/build_trn_seren.py" --project "$project" --baseline "$shared/.codex-temp/load-curves-v3-20261101/inputs/v2" --weather "$shared/.codex-temp/load-curves-v3-20261101/inputs/weather"
"$python" "$project/analysis/verify_trn_seren.py" --project "$project" --baseline "$shared/.codex-temp/load-curves-v3-20261101/inputs/v2"
