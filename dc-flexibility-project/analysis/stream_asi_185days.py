import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests
from urllib3.exceptions import HTTPError as UrllibHTTPError

from probe_asi_inference_range import BASE, HTTPRangeFile


KEYS = ["job_type_public", "priority_class", "model_type_public"]
KINDS = ["pod_hourly", "server_hourly", "job_execution_summary"]
EXPECTED_BYTES = [351803513445, 3080121387, 1188295031]
SIZE_EDGES = [0, .125, .25, .5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, float("inf")]
DURATION_EDGES = [0, 1/60, 5/60, 1/6, .5, 1, 3, 6, 12, 24, 72, 168, 720, 2160, 4440, float("inf")]
WORKER_ARCHIVES = {}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def records(frame):
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


class RetryingRangeFile(HTTPRangeFile):
    def __init__(self, kind):
        super().__init__(BASE + "asi_opensource_" + kind + ".zip", 500 * 1024**3, 2 * 1024**3)
        expected = EXPECTED_BYTES[KINDS.index(kind)]
        if self.size != expected:
            raise ValueError(f"对象大小与核验版本不同: {kind}, {self.size}")
        self.retry_events = []

    def read(self, size=-1):
        for attempt in range(4):
            try:
                return super().read(size)
            except (requests.RequestException, UrllibHTTPError) as error:
                event = {"at": stamp(), "offset": self.position, "attempt": attempt + 1,
                         "error": repr(error), "partial_transfer_bytes": "unknown"}
                self.retry_events.append(event)
                print(json.dumps(event), file=sys.stderr, flush=True)
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)


def archive(kind):
    if kind not in WORKER_ARCHIVES:
        remote = RetryingRangeFile(kind)
        WORKER_ARCHIVES[kind] = (remote, zipfile.ZipFile(remote))
    return WORKER_ARCHIVES[kind]


def transfer_snapshot():
    return {k: r.transferred for k, (r, _) in WORKER_ARCHIVES.items()}


def member(kind, name, pinned):
    remote, handle = archive(kind)
    if remote.etag != pinned[kind]["etag"]:
        raise ValueError(f"ETag 已改变: {kind}")
    info = handle.getinfo(name)
    payload = handle.read(info)
    metadata = {"kind": kind, "member": name, "bytes": len(payload), "crc32": info.CRC,
                "sha256": hashlib.sha256(payload).hexdigest(), "etag": remote.etag}
    return payload, metadata


def histogram(frame, column, edges, weight=None):
    labels = [f"[{a:g},{b:g})" for a, b in zip(edges[:-1], edges[1:])]
    selected = frame.copy()
    selected["bin"] = pd.cut(selected[column], bins=edges, labels=labels, right=False)
    selected["observations"] = 1
    values = ["observations"] + ([weight] if weight else [])
    return records(selected.groupby(KEYS + ["bin"], observed=True, dropna=False)[values].sum().reset_index())


def aggregate(table):
    for name in ["gpu_request", "used_gpu_hours", "cpu_request_cores"]:
        values = table[name]
        invalid = pc.or_(pc.less(values, 0), pc.invert(pc.is_finite(values)))
        if pc.any(pc.fill_null(invalid, False)).as_py():
            raise ValueError(f"负值或非有限资源: {name}")
    table = table.append_column("observations", pa.array(np.ones(len(table), dtype=np.int64)))
    table = table.append_column("missing_gpu_hours", pc.cast(pc.is_null(table["used_gpu_hours"]), pa.int64()))
    table = table.append_column("missing_gpu_request", pc.cast(pc.is_null(table["gpu_request"]), pa.int64()))
    table = table.append_column("missing_workload", pc.cast(pc.is_null(table["workload_id"]), pa.int64()))
    operations = [("gpu_request", "sum"), ("used_gpu_hours", "sum"), ("cpu_request_cores", "sum"),
                  ("observations", "sum"), ("missing_gpu_hours", "sum"), ("missing_gpu_request", "sum"), ("missing_workload", "sum"),
                  ("pod_id", "count_distinct"), ("workload_id", "count_distinct")]
    grouped = table.group_by(KEYS, use_threads=False).aggregate(operations).to_pandas()
    names = {"gpu_request_sum": "requested_gpu", "used_gpu_hours_sum": "used_gpu_hours",
             "cpu_request_cores_sum": "requested_cpu_cores", "observations_sum": "pod_rows",
             "missing_gpu_hours_sum": "missing_gpu_hours", "missing_workload_sum": "missing_workload_rows",
             "missing_gpu_request_sum": "missing_gpu_request_rows",
             "pod_id_count_distinct": "pod_count", "workload_id_count_distinct": "workload_count"}
    grouped.rename(columns=names, inplace=True)
    positive = table.filter(pc.fill_null(pc.greater(table["gpu_request"], 0), False))
    gpu_counts = positive.group_by(KEYS, use_threads=False).aggregate([
        ("pod_id", "count_distinct"), ("workload_id", "count_distinct"), ("observations", "sum")]).to_pandas()
    gpu_counts.rename(columns={"pod_id_count_distinct": "gpu_requesting_pods",
        "workload_id_count_distinct": "gpu_requesting_workloads", "observations_sum": "gpu_requesting_rows"}, inplace=True)
    grouped = grouped.merge(gpu_counts, on=KEYS, how="left")
    for column in ["gpu_requesting_pods", "gpu_requesting_workloads", "gpu_requesting_rows"]:
        grouped[column] = grouped[column].fillna(0).astype("int64")
    size_frame = table.select(KEYS + ["gpu_request", "used_gpu_hours"]).to_pandas()
    size_hist = histogram(size_frame, "gpu_request", SIZE_EDGES, "used_gpu_hours")
    known = positive.filter(pc.is_valid(positive["workload_id"]))
    workload = known.group_by(KEYS + ["workload_id"], use_threads=False).aggregate([("gpu_request", "sum")]).to_pandas()
    workload_hist = histogram(workload, "gpu_request_sum", SIZE_EDGES)
    gpu_model = positive.group_by(KEYS + ["gpu_spec_public"], use_threads=False).aggregate([
        ("gpu_request", "sum"), ("used_gpu_hours", "sum"), ("observations", "sum")]).to_pandas()
    state = positive.group_by(KEYS + ["state_public"], use_threads=False).aggregate([
        ("gpu_request", "sum"), ("used_gpu_hours", "sum"), ("observations", "sum")]).to_pandas()
    return records(grouped), size_hist, workload_hist, records(gpu_model), records(state)


def process_hour(index, output, pinned):
    pa.set_cpu_count(2)
    started, before = time.perf_counter(), transfer_snapshot()
    day, hour = divmod(index, 24)
    content, pod_meta = member("pod_hourly", f"asi_opensource_pod_hourly/day={day}/hour={hour:02d}/part-000.parquet", pinned)
    parquet = pq.ParquetFile(io.BytesIO(content))
    columns = KEYS + ["pod_id", "workload_id", "gpu_request", "used_gpu_hours", "cpu_request_cores", "gpu_spec_public", "state_public"]
    table = parquet.read(columns=columns, use_threads=False)
    if table["pod_id"].null_count:
        raise ValueError("pod_id 缺失")
    groups, scale, workload_scale, models, states = aggregate(table)
    pod_meta.update(rows=len(table), schema=str(parquet.schema_arrow), distinct_pods=pc.count_distinct(table["pod_id"]).as_py(),
                    distinct_workloads=pc.count_distinct(table["workload_id"]).as_py())
    del content, parquet, table
    content, inventory_meta = member("server_hourly", f"asi_opensource_server_hourly/day={day}/hour={hour:02d}/part-000.parquet", pinned)
    inventory_table = pq.read_table(io.BytesIO(content), columns=["gpu_count", "gpu_spec_public"], use_threads=False)
    inventory = inventory_table.group_by("gpu_spec_public", use_threads=False).aggregate([("gpu_count", "sum")]).to_pandas()
    inventory_meta["rows"] = len(inventory_table)
    value = {"index": index, "day": day, "hour": hour, "finished_at": stamp(), "pod": pod_meta,
             "server": inventory_meta, "groups": groups, "pod_size_histogram": scale,
             "workload_size_histogram": workload_scale, "gpu_models": models, "states": states,
             "inventory": records(inventory), "seconds": time.perf_counter() - started,
             "range_bytes": {k: r.transferred - before.get(k, 0) for k, (r, _) in WORKER_ARCHIVES.items()},
             "retry_events_total": sum(len(r.retry_events) for r, _ in WORKER_ARCHIVES.values()),
             "worker_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    path = Path(output) / "checkpoints" / f"hour-{index:05d}.json"
    write_json(path, value)
    return {k: value[k] for k in ("index", "day", "hour", "finished_at", "seconds", "range_bytes", "worker_peak_rss_kib")}


def process_jobs(output, pinned):
    pa.set_cpu_count(2)
    started, before = time.perf_counter(), transfer_snapshot()
    content, metadata = member("job_execution_summary", "asi_opensource_job_execution_summary/part-000.parquet", pinned)
    parquet = pq.ParquetFile(io.BytesIO(content))
    table = parquet.read(columns=KEYS + ["pod_id", "workload_id", "gpu_request", "duration_hours"], use_threads=False)
    metadata.update(rows=len(table), schema=str(parquet.schema_arrow))
    for name in ["gpu_request", "duration_hours"]:
        if table[name].null_count or pc.any(pc.or_(pc.less(table[name], 0), pc.invert(pc.is_finite(table[name])))).as_py():
            raise ValueError(f"作业跨度字段无效: {name}")
    unique = table.group_by(KEYS, use_threads=False).aggregate([("pod_id", "count_distinct"),
                 ("workload_id", "count_distinct")]).to_pandas()
    frame = table.drop(["pod_id", "workload_id"]).to_pandas()
    del table, content, parquet
    frame["requested_gpu_hour_proxy"] = frame.gpu_request * frame.duration_hours
    metrics = []
    for keys, group in frame.groupby(KEYS, dropna=False, observed=True):
        values = group.duration_hours
        metrics.append({**dict(zip(KEYS, keys)), "execution_spans": len(group),
            "zero_duration_spans": int((values == 0).sum()), "duration_min_hours": float(values.min()),
            **{f"duration_p{int(q*100):02d}_hours": float(values.quantile(q)) for q in [.1, .25, .5, .75, .9, .95, .99]},
            "duration_max_hours": float(values.max()), "duration_sum_hours": float(values.sum()),
            "gpu_request_mean": float(group.gpu_request.mean()), "gpu_request_max": float(group.gpu_request.max()),
            "requested_gpu_hour_proxy": float(group.requested_gpu_hour_proxy.sum())})
    metrics = pd.DataFrame(metrics).merge(unique, on=KEYS, how="left")
    value = {"finished_at": stamp(), "member": metadata, "metrics": records(metrics),
             "duration_histogram": histogram(frame, "duration_hours", DURATION_EDGES, "requested_gpu_hour_proxy"),
             "size_histogram": histogram(frame, "gpu_request", SIZE_EDGES, "requested_gpu_hour_proxy"),
             "seconds": time.perf_counter() - started,
             "range_bytes": {k: r.transferred - before.get(k, 0) for k, (r, _) in WORKER_ARCHIVES.items()},
             "worker_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(Path(output) / "checkpoints/jobs.json", value)
    return {"task": "jobs", "rows": metadata["rows"], "seconds": value["seconds"], "finished_at": stamp(),
            "range_bytes": value["range_bytes"], "worker_peak_rss_kib": value["worker_peak_rss_kib"]}


def plan(output):
    result = {}
    for kind in KINDS:
        remote, handle = archive(kind)
        entries = [item for item in handle.infolist() if not item.is_dir()]
        if kind.endswith("hourly"):
            expected = {f"asi_opensource_{kind}/day={d}/hour={h:02d}/part-000.parquet" for d in range(185) for h in range(24)}
        else:
            expected = {"asi_opensource_job_execution_summary/part-000.parquet"}
        if {item.filename for item in entries} != expected or any(item.compress_type != zipfile.ZIP_STORED for item in entries):
            raise ValueError(f"分区或压缩方式发生变化: {kind}")
        result[kind] = {"url": remote.url, "etag": remote.etag, "bytes": remote.size,
                        "members": len(entries), "central_directory_range_bytes": remote.transferred}
    existing = output / "objects.json"
    if existing.exists():
        saved = json.loads(existing.read_text("utf-8"))
        for kind in KINDS:
            if (saved[kind]["etag"], saved[kind]["bytes"]) != (result[kind]["etag"], result[kind]["bytes"]):
                raise ValueError("续跑对象版本与检查点不同")
    else:
        write_json(existing, result)
    for remote, handle in WORKER_ARCHIVES.values():
        handle.close()
        remote.session.close()
    WORKER_ARCHIVES.clear()
    return result


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    with (output / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if shutil.disk_usage(args.volume).free < 32 * 1024**3:
            raise ValueError("共享卷不足 30 GiB 保留量与 2 GiB 并发余量")
        pinned = plan(output)
        completed = set()
        for path in (output / "checkpoints").glob("hour-*.json"):
            item = json.loads(path.read_text("utf-8"))
            if any(item[key]["etag"] != pinned[kind]["etag"] for key, kind in [("pod", "pod_hourly"), ("server", "server_hourly")]):
                raise ValueError(f"检查点对象版本错误: {path}")
            completed.add(item["index"])
        wanted = list(range(args.start_hour, args.end_hour))
        pending = [h for h in wanted if h not in completed]
        state = {"run_started_at": stamp(), "pid": os.getpid(), "workers": args.workers,
                 "state": "running", "completed_hours": len(completed), "target_hours": 4440,
                 "reused_hours": len(completed), "pending_this_run": len(pending), "raw_disk_bytes_written": 0,
                 "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "volume_free_at_start_bytes": shutil.disk_usage(args.volume).free}
        write_json(output / "status.json", state)
        with (output / "runs.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(state) + "\n")
        began = time.perf_counter()
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {}
                if not args.skip_jobs and not (output / "checkpoints/jobs.json").exists():
                    futures[pool.submit(process_jobs, str(output), pinned)] = "jobs"
                for index in pending:
                    futures[pool.submit(process_hour, index, str(output), pinned)] = index
                try:
                    for future in as_completed(futures):
                        info = future.result()
                        if "index" in info:
                            completed.add(info["index"])
                        if shutil.disk_usage(args.volume).free < 32 * 1024**3:
                            raise ValueError("共享卷低于保留量，停止后续处理")
                        state.update(completed_hours=len(completed), last_completed=info, updated_at=stamp(),
                                     elapsed_this_run_seconds=time.perf_counter() - began,
                                     jobs_complete=(output / "checkpoints/jobs.json").exists())
                        write_json(output / "status.json", state)
                        with (output / "progress.jsonl").open("a", encoding="utf-8") as log:
                            log.write(json.dumps(info) + "\n")
                        print(json.dumps({"completed_hours": len(completed), **info}), flush=True)
                except BaseException:
                    for future in futures:
                        future.cancel()
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise
            state.update(state="complete" if len(completed) == 4440 and (output / "checkpoints/jobs.json").exists() else "partial",
                         finished_at=stamp(), elapsed_this_run_seconds=time.perf_counter() - began,
                         volume_free_at_end_bytes=shutil.disk_usage(args.volume).free)
            write_json(output / "status.json", state)
        except BaseException as error:
            state.update(state="failed", failed_at=stamp(), error=repr(error), traceback=traceback.format_exc())
            write_json(output / "status.json", state)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="后台流式统计 ASI 全任务类型；原始成员仅在内存，按小时原子检查点续跑")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--volume", type=Path, default=Path("/data3/dataset"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--start-hour", type=int, default=0)
    parser.add_argument("--end-hour", type=int, default=4440)
    parser.add_argument("--skip-jobs", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.start_hour < args.end_hour <= 4440 or not 1 <= args.workers <= 16:
        raise ValueError("小时范围须位于 0..4439，工作进程数须为 1..16")
    main(args)
