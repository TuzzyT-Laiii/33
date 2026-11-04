import argparse
from collections import Counter
import fcntl
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

import numpy as np
import pandas as pd
import requests

from acquire_alibaba_cpu import BASE, now, sha256
from audit_alibaba_cpu import END, HOURS


CHECKSUMS = {
    "container_usage": "b4bd3b1b82f5407c1b0dd47fe736ff5152c4b016a9c30cb26f988e2d1c5e5031",
    "batch_instance": "e73e5a9326669aa079ba20048ddd759383cabe1fe3e58620aa75bd034e2450c6",
}
COMPATIBLE_CHECKPOINT_CODES = {"9676f6db9b68edfd0c12cccdb6d910d3b49adb6fa10a3bab9e95714fb5a45a5f"}


class Forward(io.RawIOBase):
    def __init__(self, stream):
        self.stream = stream

    def readable(self):
        return True

    def readinto(self, buffer):
        block = self.stream.read(len(buffer))
        buffer[:len(block)] = block
        return len(block)


class HashStream:
    def __init__(self, stream):
        self.stream, self.bytes, self.digest = stream, 0, hashlib.sha256()

    def read(self, size=-1):
        block = self.stream.read(size)
        self.bytes += len(block)
        self.digest.update(block)
        return block

    def readinto(self, buffer):
        block = self.read(len(buffer))
        buffer[:len(block)] = block
        return len(block)


def add_intervals(target, machine, start, end, weight):
    start, end = np.maximum(np.asarray(start, float), 0), np.minimum(np.asarray(end, float), END)
    machine, weight = np.asarray(machine, int), np.asarray(weight, float)
    good = (end > start) & np.isfinite(weight) & (weight >= 0) & (machine >= 0)
    start, end, machine, weight = start[good], end[good], machine[good], weight[good]
    first, last = (start // 3600).astype(int), (np.nextafter(end, -np.inf) // 3600).astype(int)
    same = first == last
    np.add.at(target, (machine[same], first[same]), (end[same] - start[same]) * weight[same] / 3600)
    a, b, m, w, f, l = start[~same], end[~same], machine[~same], weight[~same], first[~same], last[~same]
    np.add.at(target, (m, f), ((f + 1)*3600 - a)*w/3600)
    np.add.at(target, (m, l), (b - l*3600)*w/3600)
    interior = l > f+1
    if interior.any():
        difference = np.zeros((target.shape[0], HOURS+1))
        np.add.at(difference, (m[interior], f[interior]+1), w[interior])
        np.add.at(difference, (m[interior], l[interior]), -w[interior])
        target += np.cumsum(difference, axis=1)[:, :HOURS]


def run(raw, output, name, chunk_rows=1000000, stop_after_chunks=None, archive_path=None):
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / f"{name}.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checkpoint = output / f"{name}-checkpoint.npz"
    state_path = output / f"{name}-status.json"
    inventory = pd.read_csv(output.parent / "machine-inventory.csv")
    machine_index = dict(zip(inventory.machine_id, range(len(inventory))))
    shape = (len(inventory), HOURS)
    arrays = {key: np.zeros(shape) for key in ("cpu_core_hours", "cpu_alternative_core_hours", "covered_hours")}
    stats, durations, finished_groups, processed_rows = Counter(), Counter(), set(), 0
    state = {"source": name, "url": f"{BASE}/{name}.tar.gz", "started_at_utc": now(),
             "method": "HTTP→tarfile r|gz→pandas chunks；压缩包和 CSV 均不落盘",
             "restart": "从 HTTP 起点重新解压；跳过 checkpoint 已提交行；后续行继续累计，SHA 对完整压缩流重算",
             "code_sha256": sha256(__file__)}
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            if metadata["state"]["code_sha256"] not in {state["code_sha256"], *COMPATIBLE_CHECKPOINT_CODES}:
                raise ValueError("恢复点代码版本不同，请使用独立输出目录")
            current_code = state["code_sha256"]
            arrays = {key: saved[key].copy() for key in arrays}
            processed_rows = metadata["processed_rows"]
            stats, durations = Counter(metadata["stats"]), Counter({int(k): v for k, v in metadata["durations"].items()})
            finished_groups = set(metadata["finished_groups"])
            state = metadata["state"]
            if state.get("status") == "complete":
                print(f"已完成并复用 {name}", flush=True)
                return
            state["resumed_at_utc"] = now()
            state["resume_from_rows"] = processed_rows
            state["code_sha256"] = current_code
    session = requests.Session()
    session.trust_env = False
    response = session.head(state["url"],timeout=60) if archive_path else session.get(state["url"], stream=True, timeout=(30, 120), headers={"Accept-Encoding": "identity"})
    response.raise_for_status()
    size, etag = int(response.headers["Content-Length"]), response.headers.get("ETag")
    if "etag" in state and state["etag"] != etag:
        raise ValueError("恢复时上游 ETag 发生变化")
    state.update(bytes_expected=size, etag=etag, last_modified=response.headers.get("Last-Modified"), status="processing")
    if archive_path:
        if archive_path.stat().st_size != size:
            raise ValueError("本地缓存字节数尚未完整")
        input_stream=archive_path.open("rb")
        state["input_archive"]=str(archive_path.resolve())
    else:
        input_stream=response.raw
    compressed = HashStream(input_stream)
    started = time.monotonic()

    def save(status):
        free = shutil.disk_usage(output).free
        if free < 5*1024**3:
            raise OSError("汇总写入前可用空间低于 5 GiB")
        state.update(status=status, updated_at_utc=now(), processed_rows=processed_rows,
                     compressed_bytes_read=compressed.bytes, free_bytes=free,
                     elapsed_seconds_this_run=time.monotonic()-started)
        metadata = {"state": state, "processed_rows": processed_rows, "stats": dict(stats),
                    "durations": dict(durations), "finished_groups": sorted(finished_groups)}
        temporary = checkpoint.with_suffix(".partial.npz")
        np.savez_compressed(temporary, **arrays, metadata=json.dumps(metadata))
        temporary.replace(checkpoint)
        temporary_state = state_path.with_suffix(".partial.json")
        temporary_state.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_state.replace(state_path)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    def batch_chunk(frame):
        stats["raw_rows"] += len(frame)
        for key, value in frame.status.value_counts().items():
            stats[f"status_{key}"] += int(value)
        t, e = frame.start.to_numpy(float), frame.end.to_numpy(float)
        avg, maximum = frame.cpu_avg.to_numpy(float), frame.cpu_max.to_numpy(float)
        m = frame.machine_id.map(machine_index).fillna(-1).to_numpy(int)
        valid = np.isfinite(t) & np.isfinite(e) & (t >= 0) & (e > t) & np.isfinite(avg) & (avg >= 0) & (m >= 0)
        stats["valid_positive_duration_rows"] += int(valid.sum())
        stats["invalid_or_zero_duration_rows"] += int((~valid).sum())
        stats["average_above_maximum_rows"] += int((valid & (avg > maximum)).sum())
        stats["invalid_maximum_cpu_rows"] += int((valid & (~np.isfinite(maximum) | (maximum < 0))).sum())
        duration, count = np.unique((e[valid]-t[valid]).astype(int), return_counts=True)
        durations.update(dict(zip(map(int, duration), map(int, count))))
        add_intervals(arrays["cpu_core_hours"], m[valid], t[valid], e[valid], avg[valid]/100)
        maxvalid = valid & np.isfinite(maximum) & (maximum >= 0)
        add_intervals(arrays["cpu_alternative_core_hours"], m[maxvalid], t[maxvalid], e[maxvalid], maximum[maxvalid]/100)
        add_intervals(arrays["covered_hours"], m[valid], t[valid], e[valid], np.ones(valid.sum()))

    save("processing")
    seen, chunks = 0, 0
    with tarfile.open(fileobj=compressed, mode="r|gz") as archive:
        member = archive.next()
        if member.name != f"{name}.csv":
            raise ValueError(f"非预期归档成员：{member.name}")
        state["uncompressed_member_bytes"] = member.size
        with archive.extractfile(member) as stream:
            buffered = io.BufferedReader(Forward(stream), buffer_size=4*1024**2)
            columns = ["status", "start", "end", "machine_id", "cpu_avg", "cpu_max"]
            usecols = [4,5,6,7,10,11]
            reader = pd.read_csv(buffered, header=None, usecols=usecols, chunksize=chunk_rows)
            for frame in reader:
                frame.columns = columns
                before, seen = seen, seen+len(frame)
                if seen <= processed_rows:
                    continue
                frame = frame.iloc[max(0, processed_rows-before):].copy()
                batch_chunk(frame)
                processed_rows = seen
                chunks += 1
                if chunks % 5 == 0:
                    save("processing")
                if stop_after_chunks and chunks >= stop_after_chunks:
                    save("paused_for_resume_check")
                    response.close()
                    return
        if archive.next() is not None:
            raise ValueError("归档存在额外成员")
    while compressed.read(8*1024**2):
        pass
    response.close()
    input_stream.close()
    actual = compressed.digest.hexdigest()
    if compressed.bytes != size or actual != CHECKSUMS[name]:
        raise ValueError(f"压缩流 SHA 或字节数不符：{compressed.bytes}, {actual}")
    state.update(sha256=actual, official_sha256_matches=True, completed_at_utc=now(), stats=dict(stats))
    table = pd.DataFrame({"machine_id": np.repeat(inventory.machine_id.to_numpy(), HOURS),
                          "relative_hour": np.tile(np.arange(HOURS), len(inventory))})
    for key, values in arrays.items():
        table[key] = values.ravel()
    table.to_parquet(output / f"{name}-machine-hourly.parquet", index=False)
    table.groupby("relative_hour")[list(arrays)].sum().to_csv(output / f"{name}-hourly.csv")
    if durations:
        pd.DataFrame(sorted(durations.items()), columns=["duration_seconds", "instances"]).to_csv(output / "batch-instance-duration-histogram.csv", index=False)
    save("complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="流式核对 Alibaba 大表；仅保留汇总和可恢复检查点")
    parser.add_argument("source", choices=["batch_instance"])
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=1000000)
    parser.add_argument("--stop-after-chunks", type=int)
    parser.add_argument("--archive",type=Path,help="复用已有压缩包，避免重新下载")
    args = parser.parse_args()
    def failure_status(error_type, error, traceback):
        path = args.output / f"{args.source}-failure.json"
        path.write_text(json.dumps({"status": "failed", "at_utc": now(), "error_type": error_type.__name__, "error": str(error)}, ensure_ascii=False, indent=2), encoding="utf-8")
        sys.__excepthook__(error_type, error, traceback)
    sys.excepthook = failure_status
    run(args.raw, args.output, args.source, args.chunk_rows, args.stop_after_chunks,args.archive)
