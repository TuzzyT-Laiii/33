import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import requests


BASE = "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces"
EXPECTED = {
    "machine_meta": "b5b1b786b22cd413a3674b8f2ebfb2f02fac991c95df537f363ef2797c8f6d55",
    "machine_usage": "3e6ee87fd204bb85b9e234c5c75a5096580fdabc8f085b224033080090753a7a",
    "batch_task": "7c4b32361bd1ec2083647a8f52a6854a03bc125ca5c202652316c499fbf978c6",
    "container_meta": "febd75e693d1f208a8941395e7faa7e466e50d21c256eff12a815b7e2fa2053f",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def acquire(directory, reserve_gib=5):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    audit_path = directory / "download-audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {
        "started_at_utc": now(), "raw_directory": str(directory.resolve()),
        "official_schema": "https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/schema.txt",
        "official_checksums": "https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md",
        "license": "官方允许 study/research；未发现标准数据许可证，不能推定商业使用或再分发授权",
        "reserve_gib": reserve_gib, "files": {}, "space_checks": [],
    }
    session = requests.Session()
    headers = {}
    for name in EXPECTED:
        response = session.head(f"{BASE}/{name}.tar.gz", timeout=60)
        response.raise_for_status()
        headers[name] = {"url": response.url, "bytes": int(response.headers["Content-Length"]),
                         "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified")}
    remaining = sum(item["bytes"] for name, item in headers.items() if not (directory / f"{name}.tar.gz").exists())
    reserve = int(reserve_gib * 1024**3)
    free = shutil.disk_usage(directory).free
    if free < remaining + reserve:
        raise OSError(f"下载前空间不足：free={free}, remaining={remaining}, reserve={reserve}")
    audit["space_checks"].append({"at_utc": now(), "stage": "before", "free_bytes": free, "remaining_bytes": remaining})
    for name, info in headers.items():
        path = directory / f"{name}.tar.gz"
        started = now()
        if path.exists():
            if path.stat().st_size != info["bytes"] or sha256(path) != EXPECTED[name]:
                raise ValueError(f"已有文件校验失败，保留现场：{path}")
            print(f"已校验并复用 {name}", flush=True)
            continue
        partial = path.with_suffix(path.suffix + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        request_headers = {"Range": f"bytes={offset}-"} if offset else {}
        minimum = shutil.disk_usage(directory).free
        with session.get(info["url"], headers=request_headers, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            if offset and response.status_code != 206:
                raise ValueError("服务器不接受断点续传，保留部分文件")
            if response.headers.get("ETag") != info["etag"]:
                raise ValueError("下载前后 ETag 发生变化")
            with partial.open("ab" if offset else "wb") as stream:
                for block in response.iter_content(8 * 1024**2):
                    free = shutil.disk_usage(directory).free
                    if free < reserve + len(block):
                        raise OSError("下载期间可用空间接近保留阈值，已停止写入")
                    stream.write(block)
                    minimum = min(minimum, shutil.disk_usage(directory).free)
        actual = sha256(partial)
        if partial.stat().st_size != info["bytes"] or actual != EXPECTED[name]:
            raise ValueError(f"官方 SHA-256 或字节数不符：{name}: {actual}")
        partial.rename(path)
        audit["files"][name] = {**info, "sha256": actual, "official_sha256_matches": True,
                                "download_started_at_utc": started, "download_completed_at_utc": now(),
                                "minimum_observed_free_bytes": minimum}
        audit["space_checks"].append({"at_utc": now(), "stage": f"after_{name}", "free_bytes": shutil.disk_usage(directory).free})
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(audit["files"][name], ensure_ascii=False), flush=True)
    audit["completed_at_utc"] = now()
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="仅获取 Alibaba CPU 2018 四个指定小表并核对官方 SHA-256")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--reserve-gib", type=float, default=5)
    args = parser.parse_args()
    acquire(args.directory, args.reserve_gib)
