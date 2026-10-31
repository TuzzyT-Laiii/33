import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只读核验 ASI 运行日志、保留文件体积与共享卷空间")
    parser.add_argument("--directory", type=Path, default=Path("/data3/dataset/ljt/global-grid-flexibility-20261023/results/asi-185day-workload-types-20261031"))
    args = parser.parse_args()
    logs = [json.loads(line) for line in (args.directory / "run.log").read_text("utf-8").splitlines() if line]
    retries = [record for record in logs if "attempt" in record and "error" in record]
    files = [path for path in args.directory.rglob("*") if path.is_file()]
    raw = [str(path) for path in files if path.name.startswith("asi_opensource_")]
    if raw:
        raise ValueError(f"汇总目录出现原始数据文件: {raw}")
    volume = shutil.disk_usage("/data3/dataset")
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "directory": str(args.directory),
              "volume": "/data3/dataset", "volume_free_bytes": volume.free,
              "logical_retained_file_bytes": sum(path.stat().st_size for path in files),
              "allocated_file_bytes": sum(path.stat().st_blocks * 512 for path in files),
              "retained_file_count": len(files), "raw_file_paths": raw,
              "retry_count": len(retries), "retry_events": retries,
              "partial_failed_transfer_bytes": "unknown; not included in committed successful Range byte count"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
