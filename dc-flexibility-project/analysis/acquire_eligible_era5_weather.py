import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xarray as xr

from acquire_buildability_protection import sha, utc_now, write_json
from trial_era5_landuse_download import CachedHTTPStore, SOURCE_URL, VARIABLES


class LockedHTTPStore(CachedHTTPStore):
    def __getitem__(self, key):
        directory = Path(self.path) / ".object-locks"
        directory.mkdir(parents=True, exist_ok=True)
        if not all(part and part != ".." for part in key.split("/")):
            raise ValueError("ERA5对象键不符合缓存范围")
        with (directory / (key.replace("/", "_") + ".lock")).open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return super().__getitem__(key)


def credential_path(root):
    return root / ".codex-temp/buildability-20261025/cds-credential.json"


def source_session(root):
    key = os.environ.get("CDSAPI_KEY")
    if not key:
        key = json.loads(credential_path(root).read_text(encoding="utf-8"))["key"]
    session = requests.Session()
    session.headers.update({"Authorization": "Bearer " + key, "User-Agent": "dc-flexibility-research/1.0"})
    retry = Retry(total=4, connect=4, read=4, status=4, backoff_factor=1,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def open_source(root):
    session = source_session(root)
    store = LockedHTTPStore(root / "data/raw/era5-eligible-ecmwf-geo", session)
    source = xr.open_zarr(store, consolidated=True, chunks=None)
    times = pd.date_range("2023-01-01", "2026-01-01T01:00:00", inclusive="left", freq="h").to_numpy(dtype="datetime64[ns]")
    if source.indexes["time"].get_indexer(times).min() < 0 or len(times) != 26305:
        raise ValueError("ECMWF空间源未覆盖2023至2025及右边界的全部小时")
    metadata = json.loads(store[".zmetadata"])["metadata"]
    np.testing.assert_array_equal(source.latitude.values, np.arange(-90, 90.25, 0.25))
    np.testing.assert_array_equal(source.longitude.values, np.arange(-180, 180, 0.25))
    for target, (name, units, divisor) in VARIABLES.items():
        spec = metadata[name + "/.zarray"]
        if source[name].dims != ("time", "latitude", "longitude") or source[name].attrs.get("units") != units:
            raise ValueError(f"ECMWF源变量维度或单位异常：{name}")
        if spec["chunks"] != [67584, 4, 4] or spec["dtype"] != "<f4":
            raise ValueError(f"ECMWF源变量分块或数据类型异常：{name}")
    return source, session, times, hashlib.sha256(store[".zmetadata"]).hexdigest()


def process_block(root, label, points, selection_sha, metadata_sha, times):
    output = root / "data/processed/era5-eligible-2023-2025" / (label + ".parquet")
    checkpoint = root / "checkpoints/era5-eligible-blocks" / (label + ".json")
    native = points.native_flat_index.to_numpy(dtype=np.int32)
    if checkpoint.exists():
        record = json.loads(checkpoint.read_text(encoding="utf-8"))
        if record["status"] != "done" or record["selection_manifest_sha256"] != selection_sha or record["source_metadata_sha256"] != metadata_sha:
            raise ValueError(f"已有ERA5空间块来源或筛选版本不一致：{label}")
        if record["output_sha256"] != sha(output) or record["native_flat_indices"] != native.tolist():
            raise ValueError(f"已有ERA5空间块内容或节点不一致：{label}")
        return {"label": label, "points": len(points), "status": "reused_verified", "source_bytes": 0, "output_bytes": output.stat().st_size}
    with source_session(root) as session:
        store = LockedHTTPStore(root / "data/raw/era5-eligible-ecmwf-geo", session)
        source = xr.open_zarr(store, consolidated=True, chunks=None)
        with source:
            if hashlib.sha256(store[".zmetadata"]).hexdigest() != metadata_sha:
                raise ValueError("ERA5空间块使用的冻结元数据与任务版本不一致")
            lat_indices = source.indexes["latitude"].get_indexer(points.latitude.to_numpy())
            longitude = (points.longitude_0_360.to_numpy() + 180) % 360 - 180
            lon_indices = source.indexes["longitude"].get_indexer(longitude)
            if np.any(lat_indices < 0) or np.any(lon_indices < 0):
                raise ValueError("合格用地原生天气节点没有精确坐标匹配")
            pairs = set(zip(lat_indices // 4, lon_indices // 4))
            if len(pairs) != 1:
                raise ValueError("ERA5空间块计划跨越多个源分块")
            table = pd.DataFrame({"time": np.repeat(times, len(points)), "native_flat_index": np.tile(native, len(times))})
            source_digests, converted_digests, source_supplements = {}, {}, {}
            for target, (name, units, divisor) in VARIABLES.items():
                values = source[name].sel(time=times).isel(latitude=xr.DataArray(lat_indices, dims="weather_point"),
                                                         longitude=xr.DataArray(lon_indices, dims="weather_point")).values
                repair_path = root / "data/metadata/buildability-20261025/weather-cds-repairs" / (label + ".json")
                if name in ("u100", "v100", "d2m") and np.isnan(values).any() and repair_path.exists():
                    from era5_source_validation import apply_verified_source_repair

                    values, source_supplements[name] = apply_verified_source_repair(
                        root, label, name, points, times, values, selection_sha, metadata_sha)
                if values.shape != (26305, len(points)) or values.dtype != np.dtype("float32") or not np.isfinite(values).all():
                    raise ValueError(f"ERA5空间块存在小时缺失、非有限值或形状错误：{label}/{name}")
                converted = values / np.float32(divisor)
                source_digests[name] = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
                converted_digests[target] = hashlib.sha256(np.ascontiguousarray(converted).tobytes()).hexdigest()
                table[target] = converted.reshape(-1)
            data_events = [event for event in store.events if event["key"].split("/")[0] in {value[0] for value in VARIABLES.values()}]
            for event in data_events:
                fields = event["key"].split("/")[1].split(".")
                if len(fields) != 3 or (int(fields[1]), int(fields[2])) not in pairs:
                    raise ValueError("ERA5获取对象超出合格网格需要的空间块")
            if shutil.disk_usage(root).free < 100 * 1024 ** 3 + table.memory_usage(index=True, deep=True).sum() * 2:
                raise OSError("ERA5获取可用磁盘不足，须保留100 GiB")
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".parquet.part")
            table.to_parquet(temporary, index=False, compression="zstd")
            returned = pd.read_parquet(temporary)
            pd.testing.assert_frame_equal(table, returned)
            if returned.duplicated(["time", "native_flat_index"]).any():
                raise ValueError("ERA5输出包含重复时间与节点")
            for name in VARIABLES:
                received = returned[name].to_numpy().reshape(26305, len(points))
                if hashlib.sha256(np.ascontiguousarray(received).tobytes()).hexdigest() != converted_digests[name]:
                    raise ValueError("ERA5输出回读值SHA与输入不一致")
            temporary.replace(output)
            record = {"status": "done", "completed_utc": utc_now(), "label": label,
                      "selection_manifest_sha256": selection_sha, "source_metadata_sha256": metadata_sha,
                      "native_flat_indices": native.tolist(), "hours": 26305, "source_values_sha256": source_digests,
                      "output_values_sha256": converted_digests, "output_sha256": sha(output), "output_path": str(output),
                      "output_bytes": output.stat().st_size, "http_source_objects": data_events,
                      "source_bytes": sum(item["bytes"] for item in data_events),
                      "source_supplements": source_supplements,
                      "radiation_conversion": "ERA5逐小时累计辐射除以3600秒，原始空间块保留，时刻表示此前一小时平均通量"}
            write_json(checkpoint, record)
            return {"label": label, "points": len(points), "status": "downloaded_verified", "source_bytes": record["source_bytes"], "output_bytes": output.stat().st_size}


def run(args):
    root = args.root.resolve()
    status_path = root / "status/era5-eligible-acquisition.json"
    state = {"status": "waiting_for_buildability_selection", "started_utc": utc_now(), "updated_utc": utc_now(),
             "completed_blocks": 0, "completed_points": 0, "workers": args.workers}
    write_json(status_path, state)
    selection_path = root / "data/metadata/buildability-20261025/selection-manifest.json"
    deadline = time.monotonic() + 72 * 3600
    while not selection_path.exists():
        producer = root / "status/buildability-20261025.json"
        if producer.exists() and json.loads(producer.read_text(encoding="utf-8"))["status"] == "failed":
            raise RuntimeError("用地筛选任务已明确失败；ERA5任务停止，不提交未经筛选的区域")
        if time.monotonic() > deadline:
            raise TimeoutError("等待完整合格用地清单超过72小时")
        state.update(updated_utc=utc_now())
        write_json(status_path, state)
        time.sleep(30)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["status"] != "complete" or not selection["protected_areas_applied"] or not selection["slope_applied"]:
        raise ValueError("天气获取要求已通过保护地与坡度联合筛选的完整清单")
    for output in selection["outputs"].values():
        if sha(Path(output["path"])) != output["sha256"]:
            raise ValueError("合格网格输出SHA与清单不一致")
    points = pd.read_parquet(selection["outputs"]["points"]["path"]).sort_values("native_flat_index")
    if points.native_flat_index.duplicated().any() or len(points) != selection["weather_points"]:
        raise ValueError("合格天气节点数量或唯一性不一致")
    points["source_lat_block"] = np.rint((points.latitude + 90) * 4).astype(int) // 4
    points["source_lon_block"] = np.rint(((points.longitude_0_360 + 180) % 360) * 4).astype(int) // 4
    blocks = [(f"lat-{lat:03d}-lon-{lon:03d}", part.copy()) for (lat, lon), part in points.groupby(["source_lat_block", "source_lon_block"], sort=True)]
    selection_sha = sha(selection_path)
    source, session, times, metadata_sha = open_source(root)
    source.close()
    session.close()
    plan = {"status": "ready", "created_utc": utc_now(), "selection_manifest_sha256": selection_sha,
            "source_url": SOURCE_URL, "source_metadata_sha256": metadata_sha, "hours": 26305,
            "analysis_hours": 26304, "right_boundary_samples": 1, "selected_grids": selection["selected_grids"],
            "weather_points": len(points), "blocks": len(blocks), "units": {name: "W m**-2" if value[2] == 3600 else value[1] for name, value in VARIABLES.items()},
            "network_scope": "仅请求与合格网格必要原生节点相交的4乘4空间块；块内额外空间与时间覆盖按源压缩布局保留",
            "source_policy": "新任务采用冻结ECMWF空间分块源，复用同源原始缓存；原Google天气库单独保留"}
    write_json(root / "data/metadata/buildability-20261025/weather-acquisition-manifest.json", plan)
    state.update(status="downloading", total_blocks=len(blocks), total_points=len(points), source_bytes_this_attempt=0,
                 selection_manifest_sha256=selection_sha, source_metadata_sha256=metadata_sha, updated_utc=utc_now())
    write_json(status_path, state)
    selected_blocks = blocks if args.limit_blocks is None else blocks[:args.limit_blocks]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_block, root, label, part, selection_sha, metadata_sha, times): label for label, part in selected_blocks}
        for future in as_completed(futures):
            error = future.exception()
            if error is not None:
                for pending in futures:
                    pending.cancel()
                state.update(status="failed", failed_block=futures[future], error_type=type(error).__name__, error=str(error), updated_utc=utc_now())
                write_json(status_path, state)
                raise error
            result = future.result()
            state.update(completed_blocks=state["completed_blocks"] + 1, completed_points=state["completed_points"] + result["points"],
                         source_bytes_this_attempt=state["source_bytes_this_attempt"] + result["source_bytes"],
                         last_completed_block=result["label"], updated_utc=utc_now())
            write_json(status_path, state)
            print(json.dumps(state, ensure_ascii=False), flush=True)
    if args.limit_blocks is None and (state["completed_blocks"] != len(blocks) or state["completed_points"] != len(points)):
        raise ValueError("ERA5完成空间块或节点数量与合格用地计划不一致")
    state.update(status="complete" if args.limit_blocks is None else "limit_reached", updated_utc=utc_now(), completed_utc=utc_now())
    write_json(status_path, state)
    write_json(root / "results/buildability-20261025/weather-acquisition-summary.json", state)
    print(json.dumps(state, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="只获取通过保护地和坡度筛选的网格所需ERA5，完整保存三年逐小时七变量。")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--limit-blocks", type=int)
    parser.add_argument("--receive-cds-key", action="store_true")
    parser.add_argument("--probe-source", action="store_true")
    args = parser.parse_args()
    if args.receive_cds_key:
        import sys

        value = json.load(sys.stdin)
        if not isinstance(value.get("key"), str) or not value["key"]:
            raise ValueError("CDS凭据内容无效")
        path = credential_path(args.root.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream)
        os.chmod(path, 0o600)
        print(json.dumps({"status": "credential_configured"}))
        return
    if args.probe_source:
        source, session, times, metadata_sha = open_source(args.root.resolve())
        source.close()
        session.close()
        result = {"status": "passed", "verified_utc": utc_now(), "source_metadata_sha256": metadata_sha,
                  "all_26305_hours_available": True, "seven_variable_metadata_validated": True}
        write_json(args.root / "results/buildability-20261025/weather-source-probe.json", result)
        print(json.dumps(result), flush=True)
        return
    with (args.root / ".era5-eligible-acquisition.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
