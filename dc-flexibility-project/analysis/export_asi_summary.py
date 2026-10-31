import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


FILES = [
    "hourly-type-priority-model.parquet", "hourly-total.csv", "inventory-attribution-anomalies.csv",
    "hourly-by-type.csv", "type-priority-model-total.csv", "daily-by-type.csv", "diurnal-by-type.csv",
    "relative-week-by-type.csv", "mod7-by-type.csv", "pod-scale-histogram.csv", "workload-hour-scale-histogram.csv",
    "gpu-model-total.csv", "state-total.csv", "execution-span-metrics.csv", "execution-duration-histogram.csv",
    "execution-scale-histogram.csv", "execution-by-type-summary.csv", "task-type-summary.csv", "inf-hourly.csv",
    "member-audit.parquet", "checkpoint-manifest.csv", "summary.json", "processing-audit.json", "inf-source-manifest.json",
    "independent-sample-validation.json", "inf-annual-replay-2025-phase000.parquet", "inf-annual-replay-2025-phase000.json",
    "validation.json", "postprocess-status.json",
    "inf-hourly-quality.csv", "quality-summary.json", "quality-hour-audit.csv",
]
OPTIONAL_FILES = ["restart-events.json"]


def checksum(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            hasher.update(block)
    return hasher.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只读导出已完成 ASI 汇总的字节数与 SHA，供跨环境下载读回核验")
    parser.add_argument("--directory", type=Path, default=Path("/data3/dataset/ljt/global-grid-flexibility-20261023/results/asi-185day-workload-types-20261031"))
    args = parser.parse_args()
    status = json.loads((args.directory / "postprocess-status.json").read_text("utf-8"))
    if status["state"] != "complete":
        raise ValueError("汇总和验证尚未全部完成")
    files = FILES + [name for name in OPTIONAL_FILES if (args.directory / name).exists()]
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "directory": str(args.directory), "files": [
        {"file": name, "bytes": (args.directory / name).stat().st_size, "sha256": checksum(args.directory / name)} for name in files]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
