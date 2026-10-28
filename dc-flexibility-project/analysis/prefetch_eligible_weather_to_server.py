import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import time
import zipfile

import requests
import yaml

from acquire_buildability_protection import sha, utc_now, write_json
from prefetch_copdem_to_server import remote_command, relay_files
from trial_era5_landuse_download import SOURCE_URL, VARIABLES


def command(args, mode, extra=()):
    main_root = "/data3/dataset/ljt/global-grid-flexibility-20261023"
    return shlex.join([main_root + "/.venv/bin/python", main_root + "/analysis/prefetch_eligible_weather_to_server.py",
                       "--root", args.server_root, mode, *extra])


def plan_for_server(root):
    import numpy as np
    import pandas as pd
    import acquire_eligible_era5_weather as weather

    manifest_path = root / "data/metadata/buildability-20261025/weather-acquisition-manifest.json"
    if not manifest_path.exists():
        print(json.dumps({"status": "waiting_for_qualified_weather_plan"}))
        return
    selection_path = root / "data/metadata/buildability-20261025/selection-manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if selection["status"] != "complete" or not selection["protected_areas_applied"] or not selection["slope_applied"] or sha(selection_path) != manifest["selection_manifest_sha256"]:
        raise ValueError("ERA5中转计划未通过保护地、坡度和输入版本检查")
    points = pd.read_parquet(selection["outputs"]["points"]["path"])
    lat = np.rint((points.latitude + 90) * 4).astype(int) // 4
    lon = np.rint(((points.longitude_0_360 + 180) % 360) * 4).astype(int) // 4
    blocks = sorted(set(zip(lat, lon)))
    source, session, times, metadata_sha = weather.open_source(root)
    time_chunks = np.unique(source.indexes["time"].get_indexer(times) // 67584).tolist()
    source.close()
    session.close()
    if metadata_sha != manifest["source_metadata_sha256"] or len(blocks) != manifest["blocks"]:
        raise ValueError("天气中转计划与正式源元数据或空间块数量不同")
    plan = {"status": "ready", "source_url": SOURCE_URL, "source_metadata_sha256": metadata_sha,
            "selection_manifest_sha256": sha(selection_path), "time_chunks": time_chunks,
            "variables": [value[0] for value in VARIABLES.values()], "blocks": blocks}
    write_json(root / "data/metadata/buildability-20261025/weather-relay-plan.json", plan)
    print(json.dumps(plan), flush=True)


def plan_for_stream(root):
    import numpy as np
    import pandas as pd
    import acquire_eligible_era5_weather as weather

    directory = root / "data/metadata/buildability-20261025"
    source_manifest = directory / "stream-source-manifest.json"
    requests_paths = sorted((directory / "stream-requests").glob("*.json"))
    if not requests_paths:
        print(json.dumps({"status": "waiting_for_qualified_weather_plan"}), flush=True)
        return
    if not source_manifest.exists():
        from stream_buildable_weather import publish_stream_source

        first = json.loads(requests_paths[0].read_text())
        producer = json.loads((root / "status/buildability-stream.json").read_text())
        if first["source_fingerprint"] != producer["source_fingerprint"]:
            raise ValueError("已完成区域与分批筛选来源不同")
        publish_stream_source(root, first["source_fingerprint"], first["config"])
    source_identity = json.loads(source_manifest.read_text())
    blocks, grids = set(), 0
    for path in requests_paths:
        request = json.loads(path.read_text())
        if request["status"] != "complete" or not request["protected_areas_applied"] or not request["slope_applied"] or request["source_fingerprint"] != source_identity["source_fingerprint"]:
            raise ValueError("分批天气请求的保护地、坡度或来源版本不一致")
        if sha(Path(request["screen_output"])) != request["screen_output_sha256"]:
            raise ValueError("分批天气请求的筛选文件SHA不一致")
        selected = pd.read_parquet(request["screen_output"])
        selected = selected.loc[selected.selected_for_weather]
        if selected.grid_id.tolist() != request["selected_grid_ids"] or not selected.verified_eligible_area_lower_bound_km2.ge(source_identity["config"]["minimum_area_km2"]).all():
            raise ValueError("分批天气请求包含未经核实的合格网格")
        approved = {(int(row.grid_row) * 2 + dy) * 1440 + (int(row.grid_col) * 2 + 720 + dx) % 1440
                    for row in selected.itertuples() for dy in range(3) for dx in range(3)}
        if {point["native_flat_index"] for point in request["points"]} != approved or len(request["points"]) != len(approved):
            raise ValueError("分批天气节点集合与合格网格的九节点映射不同")
        for point in request["points"]:
            native = point["native_flat_index"]
            if point["latitude"] != 90 - native // 1440 * 0.25 or point["longitude_0_360"] != native % 1440 * 0.25:
                raise ValueError("分批天气节点坐标与原生索引不同")
            blocks.add((round((point["latitude"] + 90) * 4) // 4,
                        round(((point["longitude_0_360"] + 180) % 360) * 4) // 4))
        grids += request["selected_grids"]
    source, session, times, metadata_sha = weather.open_source(root)
    time_chunks = np.unique(source.indexes["time"].get_indexer(times) // 67584).tolist()
    source.close()
    session.close()
    identity_sha = sha(source_manifest)
    complete, source_bytes = [], 0
    for lat, lon in sorted(blocks):
        receipt_path = directory / "weather-stream-relay-receipts" / f"lat-{lat:03d}-lon-{lon:03d}.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["status"] != "verified" or receipt["source_metadata_sha256"] != metadata_sha or receipt["selection_manifest_sha256"] != identity_sha:
                raise ValueError("分批天气下载完成记录与当前来源不同")
            complete.append([lat, lon])
            source_bytes += sum(item["bytes"] for item in receipt["objects"])
    producer = json.loads((root / "status/buildability-stream.json").read_text())
    plan = {"status": "ready", "mode": "stream", "source_url": SOURCE_URL,
            "source_metadata_sha256": metadata_sha, "selection_manifest_sha256": identity_sha,
            "source_fingerprint": source_identity["source_fingerprint"], "time_chunks": time_chunks,
            "variables": [value[0] for value in VARIABLES.values()], "blocks": sorted(blocks),
            "completed_blocks": complete, "selected_grids": grids, "verified_tiles": len(requests_paths),
            "stream_complete": producer["status"] == "complete" and len(requests_paths) == producer["total_cover_tiles"]}
    write_json(directory / "weather-stream-relay-plan.json", plan)
    write_json(root / "status/era5-stream-prefetch.json",
               {"status": "complete" if plan["stream_complete"] and len(complete) == len(blocks) else "prefetching_verified_regions",
                "updated_utc": utc_now(), "verified_tiles": len(requests_paths), "eligible_grids": grids,
                "authorized_blocks": len(blocks), "completed_blocks": len(complete), "source_bytes": source_bytes,
                "selection_manifest_sha256": identity_sha, "source_metadata_sha256": metadata_sha})
    print(json.dumps(plan), flush=True)


def publish(root, name, stream=False):
    import fcntl

    matched = re.fullmatch(r"lat-(\d{3})-lon-(\d{3})\.zip", name)
    if not matched:
        raise ValueError("天气中转压缩包名不正确")
    pair = (int(matched[1]), int(matched[2]))
    plan_name = "weather-stream-relay-plan.json" if stream else "weather-relay-plan.json"
    plan = json.loads((root / "data/metadata/buildability-20261025" / plan_name).read_text(encoding="utf-8"))
    if pair not in {tuple(item) for item in plan["blocks"]}:
        raise ValueError("天气中转压缩包没有对应合格网格")
    cache = root / "data/raw/era5-eligible-ecmwf-geo"
    if sha(cache / ".zmetadata") != plan["source_metadata_sha256"]:
        raise ValueError("天气中转冻结元数据与已批准计划不一致")
    archive_path = cache / ".incoming-client" / name
    expected = {f"{variable}/{hour}.{pair[0]}.{pair[1]}" for variable in plan["variables"] for hour in plan["time_chunks"]}
    actual = []
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["selection_manifest_sha256"] != plan["selection_manifest_sha256"] or manifest["source_metadata_sha256"] != plan["source_metadata_sha256"]:
            raise ValueError("天气中转输入版本与当前合格清单不一致")
        if set(archive.namelist()) != expected | {"manifest.json"} or set(manifest["objects"]) != expected:
            raise ValueError("天气中转压缩包对象超出合格空间块或变量范围")
        locks = cache / ".object-locks"
        locks.mkdir(parents=True, exist_ok=True)
        for key in sorted(expected):
            if archive.getinfo(key).file_size > 8 * 1024 * 1024:
                raise ValueError("天气原始分块大小超出已核验结构")
            payload = archive.read(key)
            receipt = manifest["objects"][key]
            if len(payload) != receipt["bytes"] or hashlib.sha256(payload).hexdigest() != receipt["sha256"] or receipt["url"] != SOURCE_URL + "/" + key:
                raise ValueError("天气中转原文校验值、字节数或来源不一致")
            target = cache / key
            if not target.resolve().is_relative_to(cache.resolve()):
                raise ValueError("天气中转对象路径超出缓存目录")
            with (locks / (key.replace("/", "_") + ".lock")).open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if target.exists():
                    if sha(target) != receipt["sha256"]:
                        from era5_source_validation import retain_equivalent_source

                        receipt = retain_equivalent_source(cache, key, payload, receipt)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + f".relay-{os.getpid()}")
                    temporary.write_bytes(payload)
                    temporary.replace(target)
            actual.append(receipt)
    receipts_directory = "weather-stream-relay-receipts" if stream else "weather-relay-receipts"
    write_json(root / "data/metadata/buildability-20261025" / receipts_directory / name.replace(".zip", ".json"),
               {"status": "verified", "verified_utc": utc_now(), "source_metadata_sha256": plan["source_metadata_sha256"],
                "selection_manifest_sha256": plan["selection_manifest_sha256"], "objects": actual})
    archive_path.unlink()
    print(json.dumps({"status": "verified", "block": name, "objects": len(actual)}), flush=True)


def transfer(args, pair, plan, key):
    label = f"lat-{pair[0]:03d}-lon-{pair[1]:03d}.zip"
    directory = args.root / ".codex-temp/buildability-20261025/weather-relay-cache"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / label
    total = 0
    objects = {}
    with requests.Session() as session, zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        session.headers["Authorization"] = "Bearer " + key
        for variable in plan["variables"]:
            for hour in plan["time_chunks"]:
                object_key = f"{variable}/{hour}.{pair[0]}.{pair[1]}"
                url = SOURCE_URL + "/" + object_key
                for attempt in range(1, 5):
                    try:
                        response = session.get(url, timeout=(15, 90))
                        response.raise_for_status()
                    except requests.RequestException as error:
                        if attempt == 4:
                            raise RuntimeError(f"合格天气中转获取失败：{object_key}/{type(error).__name__}") from None
                        time.sleep(2 ** attempt)
                        continue
                    break
                payload = response.content
                if len(payload) > 8 * 1024 * 1024:
                    raise ValueError("天气原文大小不符合已核验的空间分块结构")
                objects[object_key] = {"url": url, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                                       "downloaded_utc": utc_now(), "etag": response.headers.get("ETag"),
                                       "last_modified": response.headers.get("Last-Modified")}
                archive.writestr(object_key, payload)
                total += len(payload)
        archive.writestr("manifest.json", json.dumps({"selection_manifest_sha256": plan["selection_manifest_sha256"],
                         "source_metadata_sha256": plan["source_metadata_sha256"], "objects": objects}))
    remote = args.server_root + "/data/raw/era5-eligible-ecmwf-geo/.incoming-client/"
    publish_mode = "--publish-stream" if args.stream else "--publish"
    result = relay_files(args.server, [path], remote, command(args, publish_mode, (label,)))
    if json.loads(result.stdout)["status"] != "verified":
        raise ValueError("服务器没有确认合格天气原文校验通过")
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("天气中转缓存清理超出本任务目录")
    path.unlink()
    return total


def run(args):
    if args.stream:
        return run_stream(args)
    status_path = args.root / ".codex-temp/buildability-20261025" / args.status_name
    state = {"status": "waiting_for_qualified_weather_plan", "started_utc": utc_now(), "updated_utc": utc_now()}
    write_json(status_path, state)
    deadline = time.monotonic() + 72 * 3600
    while True:
        result = remote_command(args.server, command(args, "--plan"))
        plan = json.loads(result.stdout)
        if plan["status"] == "ready":
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("合格天气中转等待正式清单超过72小时")
        state.update(updated_utc=utc_now())
        write_json(status_path, state)
        time.sleep(30)
    key = os.environ.get("CDSAPI_KEY") or yaml.safe_load((Path.home() / ".cdsapirc").read_text(encoding="utf-8"))["key"]
    directory = args.server_root + "/data/raw/era5-eligible-ecmwf-geo/.incoming-client"
    remote_command(args.server, shlex.join(["mkdir", "-p", directory]))
    blocks = list(reversed(plan["blocks"]))
    if args.limit_blocks is not None:
        blocks = plan["blocks"][:args.limit_blocks]
    state.update(status="running", total_blocks=len(blocks), transferred_blocks=0, source_bytes=0, updated_utc=utc_now())
    write_json(status_path, state)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(transfer, args, pair, plan, key): pair for pair in blocks}
        for future in as_completed(futures):
            error = future.exception()
            if error is not None:
                for pending in futures:
                    pending.cancel()
                state.update(status="failed", error_type=type(error).__name__, failed_block=futures[future], updated_utc=utc_now())
                write_json(status_path, state)
                raise error
            size = future.result()
            state.update(transferred_blocks=state["transferred_blocks"] + 1, source_bytes=state["source_bytes"] + size, updated_utc=utc_now())
            write_json(status_path, state)
            print(json.dumps(state), flush=True)
    state.update(status="complete", completed_utc=utc_now(), updated_utc=utc_now())
    write_json(status_path, state)
    print(json.dumps(state), flush=True)


def run_stream(args):
    status_path = args.root / ".codex-temp/buildability-20261025" / args.status_name
    state = {"status": "waiting_for_qualified_weather_plan", "started_utc": utc_now(), "updated_utc": utc_now(),
             "transferred_blocks": 0, "source_bytes": 0, "mode": "stream"}
    write_json(status_path, state)
    key = os.environ.get("CDSAPI_KEY") or yaml.safe_load((Path.home() / ".cdsapirc").read_text(encoding="utf-8"))["key"]
    directory = args.server_root + "/data/raw/era5-eligible-ecmwf-geo/.incoming-client"
    remote_command(args.server, shlex.join(["mkdir", "-p", directory]))
    while True:
        result = remote_command(args.server, command(args, "--stream-plan"))
        plan = json.loads(result.stdout)
        if plan["status"] != "ready":
            state.update(status="waiting_for_qualified_weather_plan", updated_utc=utc_now())
            write_json(status_path, state)
            time.sleep(30)
            continue
        completed = {tuple(pair) for pair in plan["completed_blocks"]}
        pending = [pair for pair in reversed(plan["blocks"]) if tuple(pair) not in completed]
        if not pending and plan["stream_complete"]:
            state.update(status="complete", updated_utc=utc_now(), completed_utc=utc_now())
            write_json(status_path, state)
            return
        state.update(status="running" if pending else "waiting_for_more_qualified_tiles", updated_utc=utc_now(),
                     authorized_blocks=len(plan["blocks"]), verified_server_blocks=len(completed),
                     eligible_grids=plan["selected_grids"], verified_tiles=plan["verified_tiles"])
        write_json(status_path, state)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(transfer, args, pair, plan, key): pair for pair in pending[:16]}
            for future in as_completed(futures):
                error = future.exception()
                if error is not None:
                    for waiting in futures:
                        waiting.cancel()
                    state.update(status="failed", failed_block=futures[future], error_type=type(error).__name__, updated_utc=utc_now())
                    write_json(status_path, state)
                    raise error
                state.update(transferred_blocks=state["transferred_blocks"] + 1, source_bytes=state["source_bytes"] + future.result(), updated_utc=utc_now())
                write_json(status_path, state)
                print(json.dumps(state), flush=True)
        if not pending:
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description="按已完成的保护地与坡度清单中转同源ERA5空间块，服务器校验后供正式读取。")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--server", default="pku-test")
    parser.add_argument("--server-root", default="/data3/dataset/ljt/global-grid-flexibility-20261023")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--publish")
    parser.add_argument("--publish-stream")
    parser.add_argument("--stream-plan", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--limit-blocks", type=int)
    parser.add_argument("--status-name", default="weather-relay-status.json")
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.plan:
        plan_for_server(args.root)
    elif args.stream_plan:
        plan_for_stream(args.root)
    elif args.publish_stream:
        publish(args.root, args.publish_stream, stream=True)
    elif args.publish:
        publish(args.root, args.publish)
    else:
        run(args)


if __name__ == "__main__":
    main()
