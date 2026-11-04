#!/usr/bin/env bash
set -euo pipefail
RUN=/data3/dataset/ljt/global-flex-econ-20261103
stage=${1:?需要指定extract或wind10}
scope=${2:-all}
workers=${3:-4}
cd "$RUN"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_CACHE_DIR="$RUN/numba-cache" PIP_CACHE_DIR="$RUN/pip-cache"
pidfile="$RUN/inputs/${stage}-${scope}.pid"
if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "已有进程运行：$(cat "$pidfile")"
    exit 1
fi
setsid nohup nice -n 10 "$RUN/venv/bin/python" -u "$RUN/project/analysis/global_flex_inputs.py" "$stage" --scope "$scope" --workers "$workers" > "$RUN/logs/${stage}-${scope}.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$pidfile"
echo "started $stage $scope pid=$pid workers=$workers"
