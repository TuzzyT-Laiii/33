import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import requests

from summarize_inference_requests import ASSETS


def get_range(url, start, end, size):
    response = requests.get(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}, timeout=60)
    response.raise_for_status()
    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
        raise ValueError("请求文件未返回精确 Range")
    if len(response.content) != end - start + 1:
        raise ValueError("Range 字节数不符")
    return response.content


def main(output):
    records = []
    for name, url, size, _ in ASSETS:
        if not name.startswith("azure-"):
            continue
        prefix = get_range(url, 0, 65535, size)
        first = pd.read_csv(io.BytesIO(prefix), nrows=3)
        tail = get_range(url, size - 65536, size - 1, size)
        # 使用 CSV 库读取尾部完整记录，首行可能从记录中间开始，单独排除。
        last = pd.read_csv(io.BytesIO(tail), names=first.columns).iloc[1:]
        begin, end = pd.to_datetime(first.TIMESTAMP.iloc[0]), pd.to_datetime(last.TIMESTAMP.iloc[-1])
        records.append({"dataset": name, "url": url, "file_bytes": size,
            "range_bytes": len(prefix) + len(tail), "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
            "tail_sha256": hashlib.sha256(tail).hexdigest(), "schema": list(first.columns),
            "first_timestamp": str(begin), "last_timestamp": str(end),
            "span_days": (end - begin).total_seconds() / 86400,
            "scope": "first and last rows only; no full-file completeness or SHA-256 verification"})
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "ranges": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只读取 Azure 2024 首尾各 64 KiB，核对实际发布文件的时间边界")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
