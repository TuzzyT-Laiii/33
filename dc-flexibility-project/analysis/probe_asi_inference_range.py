import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import platform
from pathlib import Path
import re
import shutil
import sys
import time
import zipfile

import pandas as pd
import pyarrow.parquet as pq
import requests


BASE = "https://tre-clusterdata.oss-cn-hangzhou.aliyuncs.com/cluster-trace-gpu-v2026/data/"


class HTTPRangeFile(io.BufferedIOBase):
    # 这里只实现可定位的 HTTP 传输，ZIP 及 ZIP64 格式统一交给标准库。
    def __init__(self, url, budget_bytes):
        self.url = url
        self.session = requests.Session()
        response = self.session.head(url, timeout=60)
        response.raise_for_status()
        self.size = int(response.headers["Content-Length"])
        self.etag = response.headers.get("ETag")
        if not self.etag:
            raise ValueError("对象缺少 ETag，无法固定抽样版本")
        self.position = 0
        self.ranges = []
        self.transferred = 0
        self.budget_bytes = budget_bytes

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if whence not in (0, 1, 2) or position < 0:
            raise ValueError((offset, whence))
        self.position = position
        return position

    def read(self, size=-1):
        size = min(self.size - self.position, size if size >= 0 else self.size)
        if size <= 0:
            return b""
        if size > 512 * 1024**2 or self.transferred + size > self.budget_bytes:
            raise ValueError("Range 大小或总传输量超过本次抽样上限")
        start = self.position
        began = time.perf_counter()
        with self.session.get(self.url, headers={"Range": f"bytes={start}-{start + size - 1}",
                "If-Match": self.etag, "Accept-Encoding": "identity"}, timeout=120, stream=True) as response:
            response.raise_for_status()
            expected = f"bytes {start}-{start + size - 1}/{self.size}"
            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError(f"服务器未严格响应 Range: {response.status_code}, {response.headers}")
            if int(response.headers["Content-Length"]) != size:
                raise ValueError("Range 长度不符合请求")
            payload = response.raw.read(size + 1)
        if len(payload) != size:
            raise ValueError("Range 响应字节数不符")
        self.ranges.append({"start": start, "bytes": size, "seconds": time.perf_counter() - began})
        self.transferred += size
        self.position += size
        return payload


def probe(args):
    if not 1 <= len(set(args.days)) <= 2 or len(args.days) != len(set(args.days)):
        raise ValueError("可行性验证只允许 1 至 2 个不同相对日")
    if any(day not in range(185) for day in args.days):
        raise ValueError("相对日超出 0 至 184")
    if not args.hours or len(set(args.hours)) != len(args.hours) or any(h not in range(24) for h in args.hours):
        raise ValueError("小时必须为 0 至 23 内无重复的整数")
    free_before = shutil.disk_usage(args.disk_path).free
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
        "disk_path": args.disk_path, "free_before_bytes": free_before, "raw_disk_bytes_written": 0,
        "mode": "range_to_memory", "days": args.days, "hours": args.hours,
        "python": platform.python_version(), "pandas": pd.__version__, "archives": [], "samples": [], "hourly": []}
    began = time.perf_counter()
    for kind in args.kinds:
        name = f"asi_opensource_{kind}_hourly"
        url = BASE + name + ".zip"
        remote = HTTPRangeFile(url, args.max_transfer_gib * 1024**3)
        with zipfile.ZipFile(remote) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            pattern = re.compile(rf"{name}/day=(\d+)/hour=(\d+)/part-000.parquet")
            partitions = [tuple(map(int, pattern.fullmatch(item.filename).groups())) for item in members
                          if pattern.fullmatch(item.filename)]
            if sorted(partitions) != [(day, hour) for day in range(185) for hour in range(24)]:
                raise ValueError("ZIP 分区不等于完整的 185×24 小时")
            directory_bytes = remote.transferred
            meta = {"url": url, "archive_bytes": remote.size, "etag": remote.etag,
                "central_directory_transfer_bytes": directory_bytes, "partitions": len(partitions),
                "compressed_members_bytes": sum(m.compress_size for m in members),
                "uncompressed_members_bytes": sum(m.file_size for m in members),
                "largest_member_bytes": max(m.file_size for m in members),
                "largest_compressed_member_bytes": max(m.compress_size for m in members),
                "zip_compression_methods": sorted(set(m.compress_type for m in members))}
            for day in sorted(args.days):
                for hour in sorted(args.hours):
                    member = f"{name}/day={day}/hour={hour:02d}/part-000.parquet"
                    info = archive.getinfo(member)
                    sample_start, transfer_start = time.perf_counter(), remote.transferred
                    payload = archive.read(info)
                    parquet = pq.ParquetFile(io.BytesIO(payload))
                    sample = {"kind": kind, "day": day, "hour": hour, "member": member,
                        "compressed_bytes": info.compress_size, "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(), "crc32": info.CRC,
                        "rows": parquet.metadata.num_rows, "schema": str(parquet.schema_arrow)}
                    if kind == "pod":
                        columns = ["job_type_public", "priority_class", "gpu_spec_public", "state_public",
                                   "gpu_request", "used_gpu_hours", "cpu_request_cores"]
                        groups, excluded_used = [], 0.0
                        for batch in parquet.iter_batches(batch_size=100000, columns=columns):
                            frame = batch.to_pandas()
                            if (frame[["gpu_request", "used_gpu_hours"]] < 0).any().any():
                                raise ValueError("发现负 GPU 资源记录")
                            excluded_used += float(frame.loc[~(frame.gpu_request > 0), "used_gpu_hours"].sum())
                            frame = frame.loc[frame.gpu_request > 0].copy()
                            frame["rows"] = 1
                            frame["missing_used_gpu_hours"] = frame.used_gpu_hours.isna().astype(int)
                            keys = ["job_type_public", "priority_class", "gpu_spec_public", "state_public"]
                            groups.append(frame.groupby(keys, dropna=False, observed=True)[
                                ["gpu_request", "used_gpu_hours", "cpu_request_cores", "rows", "missing_used_gpu_hours"]
                            ].sum(min_count=1).reset_index())
                        grouped = pd.concat(groups).groupby(keys, dropna=False, observed=True).sum(min_count=1).reset_index()
                        grouped = grouped.astype(object).where(grouped.notna(), None)
                        result["hourly"].extend({"kind": kind, "day": day, "hour": hour, **row}
                                                for row in grouped.to_dict("records"))
                        sample["used_gpu_hours_outside_positive_requests"] = excluded_used
                    else:
                        total = 0.0
                        for batch in parquet.iter_batches(columns=["gpu_count"]):
                            total += float(batch.column(0).to_pandas().sum())
                        result["hourly"].append({"kind": kind, "day": day, "hour": hour, "inventory_gpu": total})
                    sample.update(seconds=time.perf_counter() - sample_start,
                                  range_bytes=remote.transferred - transfer_start)
                    result["samples"].append(sample)
                    print(json.dumps({k: sample[k] for k in ("kind", "day", "hour", "rows", "seconds")}),
                          file=sys.stderr, flush=True)
                    del parquet, payload
            meta.update(range_requests=len(remote.ranges), range_bytes=remote.transferred,
                        range_seconds=sum(r["seconds"] for r in remote.ranges), ranges=remote.ranges)
            result["archives"].append(meta)
        remote.session.close()
    result.update(elapsed_seconds=time.perf_counter() - began,
                  free_after_bytes=shutil.disk_usage(args.disk_path).free)
    if sys.platform.startswith("linux"):
        # 内核的最高常驻内存包含 Arrow 原生分配。
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                result["peak_rss_kib"] = int(line.split()[1])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只抽取最多两天 ASI；原始 Parquet 仅进入内存，汇总输出到 stdout")
    parser.add_argument("--days", nargs="+", type=int, default=[0, 184])
    parser.add_argument("--hours", nargs="+", type=int, default=list(range(24)))
    parser.add_argument("--kinds", nargs="+", choices=["pod", "server"], default=["pod", "server"])
    parser.add_argument("--disk-path", default="/data3/dataset")
    parser.add_argument("--max-transfer-gib", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(probe(args), ensure_ascii=False, indent=2, allow_nan=False))
