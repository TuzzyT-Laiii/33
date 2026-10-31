import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil


FILES = ["BurstGPT_1.csv", "BurstGPT_2.csv", "chutes-hour-of-week.parquet",
         "chutes-daily-requests.parquet", "chutes-daily-tokens.parquet"]


def audit(raw_dir, volume):
    storage = shutil.disk_usage(volume)
    if storage.free < 32 * 1024**3:
        raise ValueError("共享卷不足 30 GiB 保留量和 2 GiB 并发余量")
    records = []
    for name in FILES:
        path = raw_dir / name
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                hasher.update(block)
        records.append({"name": name, "path": str(path), "bytes": path.stat().st_size,
                        "sha256": hasher.hexdigest()})
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
            "volume": str(volume), "total_bytes": storage.total, "used_bytes": storage.used,
            "free_bytes": storage.free, "raw_directory": str(raw_dir), "files": records,
            "persisted_bytes": sum(item["bytes"] for item in records)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只读核验服务器五个已下载文件的字节、SHA-256 与共享卷空间")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--volume", type=Path, default=Path("/data3/dataset"))
    args = parser.parse_args()
    print(json.dumps(audit(args.raw_dir, args.volume), ensure_ascii=False, indent=2))
