#!/bin/bash
set -euo pipefail
run=/data3/dataset/ljt/global-grid-flexibility-20261023/.codex-temp/eight-city-native-flex-20261102
mkdir -p "$run/scratch" "$run/result"
setsid nohup env TMPDIR="$run/scratch" PYTHONPATH="$run/analysis" PYTHONUNBUFFERED=1 \
  /data3/dataset/ljt/global-grid-flexibility-20261023/.codex-temp/load-curves-v3-20261101/.venv/bin/python \
  "$run/analysis/collect_eight_city_native.py" --inputs "$run/inputs" \
  --source-root /data3/dataset/ljt/global-grid-flexibility-20261023 \
  --weather-root /data3/dataset/ljt/eight-city-weather-20261029 --output "$run/result" \
  > "$run/run.log" 2>&1 < /dev/null &
echo $! > "$run/process.pid"
cat "$run/process.pid"
