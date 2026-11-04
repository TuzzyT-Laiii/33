#!/usr/bin/env bash
set -euo pipefail
RUN=/data3/dataset/ljt/global-flex-econ-20261103
scope=${1:-pilot}
workers=${2:-8}
output=${3:-pilot}
limit=${4:-0}
cd "$RUN"
export COPT_LICENSE_DIR="$RUN/license" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_CACHE_DIR="$RUN/numba-cache" PIP_CACHE_DIR="$RUN/pip-cache"
test -f "$RUN/validation/city-validation-accepted.json"
if [[ "$scope" == all ]]; then
    test -f "$RUN/validation/pilot-validation-accepted.json"
    test -f "$RUN/validation/global-inputs-audit.json"
fi
mkdir -p "$RUN/$output"
pidfile="$RUN/$output/controller.pid"
if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "已有求解进程运行：$(cat "$pidfile")"
    exit 1
fi
uptime
free -g
args=(--scope "$scope" --workers "$workers" --output "$output" --audit-all)
if [[ "$limit" != 0 ]]; then
    args+=(--limit-groups "$limit")
fi
setsid nohup nice -n 10 "$RUN/venv/bin/python" -u "$RUN/project/analysis/global_flex_econ_run.py" "${args[@]}" >> "$RUN/logs/${output}.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$pidfile"
echo "started scope=$scope output=$output pid=$pid workers=$workers"
