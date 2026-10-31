#!/usr/bin/env bash
set -euo pipefail

ASI_PROJECT="${1:-/data3/dataset/ljt/global-grid-flexibility-20261023}"
ASI_CODE="$ASI_PROJECT/analysis/asi-185day-workload-types-20261031"
ASI_OUTPUT="$ASI_PROJECT/results/asi-185day-workload-types-20261031"
mkdir -p "$ASI_OUTPUT"
if [ -f "$ASI_OUTPUT/launch.pid" ]; then
    read -r ASI_PREVIOUS_PID < "$ASI_OUTPUT/launch.pid"
    if kill -0 "$ASI_PREVIOUS_PID" 2>/dev/null; then
        printf '进程 %s 仍在运行，请读取 status.json。\n' "$ASI_PREVIOUS_PID" >&2
        exit 1
    fi
fi
setsid nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$ASI_PROJECT/.venv/bin/python" -u "$ASI_CODE/stream_asi_185days.py" \
    --output "$ASI_OUTPUT" --workers "${2:-16}" \
    >> "$ASI_OUTPUT/run.log" 2>&1 < /dev/null &
ASI_PID=$!
printf '%s\n' "$ASI_PID" > "$ASI_OUTPUT/launch.pid"
printf '已启动 ASI 后台进程 %s，进度目录：%s\n' "$ASI_PID" "$ASI_OUTPUT"
ASI_FINALIZER_RUNNING=0
if [ -f "$ASI_OUTPUT/finalize.pid" ]; then
    read -r ASI_FINALIZER_PID < "$ASI_OUTPUT/finalize.pid"
    if kill -0 "$ASI_FINALIZER_PID" 2>/dev/null; then
        ASI_FINALIZER_RUNNING=1
    fi
fi
if [ "$ASI_FINALIZER_RUNNING" -eq 0 ]; then
    setsid nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        PYTHONPATH="$ASI_CODE:$ASI_PROJECT/analysis" \
        "$ASI_PROJECT/.venv/bin/python" -u "$ASI_CODE/finalize_asi_185days.py" \
        --project "$ASI_PROJECT" >> "$ASI_OUTPUT/postprocess.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$ASI_OUTPUT/finalize.pid"
fi
