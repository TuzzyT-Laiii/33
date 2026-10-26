import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

import requests

from acquire_buildability_protection import sha, utc_now, write_json


BASE = "https://copernicus-dem-90m.s3.amazonaws.com/"
NAME = re.compile(r"Copernicus_DSM_COG_30_[NS]\d{2}_00_[EW]\d{3}_00_DEM")


def run_remote_commands(commands):
    network_errors = ("connection reset", "connection closed", "connection timed out", "connection refused",
                      "network is unreachable", "no route to host", "connection aborted", "temporary failure",
                      "kex_exchange_identification: read: unknown error", "banner exchange: connection to")
    for attempt in range(1, 6):
        for command in commands:
            transport = [command[0], "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                         "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3", *command[1:]]
            result = subprocess.run(transport, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if result.returncode:
                if result.returncode == 255 and any(message in result.stderr.lower() for message in network_errors) and attempt < 5:
                    print(json.dumps({"event": "ssh_network_retry", "attempt": attempt, "utc": utc_now()}), flush=True)
                    time.sleep(2 ** attempt)
                    break
                raise RuntimeError(f"SSH传输操作失败，退出码{result.returncode}：{result.stderr[-1500:]}")
        else:
            return result


def server_command(args, mode, extra=()):
    return shlex.join([args.server_root + "/.venv/bin/python", args.server_root + "/analysis/prefetch_copdem_to_server.py",
                       "--root", args.server_root, mode, *extra])


def remote_command(server, command):
    from ssh_relay_transport import remote_command as execute

    return execute(server, command)


def relay_files(server, files, remote_directory, command):
    from ssh_relay_transport import relay_files as publish_files

    return publish_files(server, files, remote_directory, command)


def make_plan(root):
    import math
    import prepare_buildable_grid as screen

    index, receipt = screen.dem_index(root)
    config = json.loads((root / "data/metadata/buildability-20261025/screen-config.json").read_text(encoding="utf-8"))
    frame, _ = screen.initial_grids(root, config["minimum_area_km2"])
    selected = frame.loc[frame.requires_pixel_screen]
    positions = set()
    for row in selected.itertuples():
        lat, lon = math.floor(row.south), math.floor(row.west)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                other = (lat + dy, (lon + dx + 180) % 360 - 180)
                if other in index:
                    positions.add(other)
    plan = {"source_index_sha256": receipt["sha256"], "tiles": [index[position] for position in sorted(positions, reverse=True)]}
    write_json(root / "data/metadata/buildability-20261025/dem-download-plan.json", plan)
    print(json.dumps(plan), flush=True)


def publish(root, name, expected_sha):
    import fcntl

    if not NAME.fullmatch(name) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError("DEM中转切片名或SHA不符合登记规则")
    directory = root / "data/raw/copdem-glo90-2021/tiles"
    incoming = directory / ".incoming-client"
    source = incoming / (name + ".tif")
    metadata = incoming / (name + ".tif.manifest.json")
    target = directory / source.name
    receipt_path = directory / metadata.name
    if not source.resolve().is_relative_to(directory.resolve()) or not target.resolve().is_relative_to(directory.resolve()):
        raise ValueError("DEM中转路径超出已登记的数据目录")
    receipt = json.loads(metadata.read_text(encoding="utf-8"))
    if receipt["sha256"] != expected_sha or receipt["url"] != BASE + name + "/" + name + ".tif" or source.stat().st_size != receipt["bytes"] or sha(source) != expected_sha:
        raise ValueError("DEM中转字节数、来源或SHA不一致")
    with target.with_suffix(".tif.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            existing = json.loads(receipt_path.read_text(encoding="utf-8"))
            if existing["sha256"] != expected_sha or sha(target) != expected_sha:
                raise ValueError("已存服务器DEM与同源中转原文不同")
            source.unlink()
            metadata.unlink()
        else:
            metadata.replace(receipt_path)
            source.replace(target)
    print(json.dumps({"status": "verified", "tile": name, "sha256": expected_sha}), flush=True)


def transfer(args, name):
    cache = args.root / ".codex-temp/buildability-20261025/dem-relay-cache"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (name + ".tif")
    metadata = path.with_suffix(".tif.manifest.json")
    url = BASE + name + "/" + name + ".tif"
    if path.exists():
        receipt = json.loads(metadata.read_text(encoding="utf-8"))
        if receipt["sha256"] != sha(path) or receipt["url"] != url:
            raise ValueError("本机DEM中转缓存与来源清单不同")
    else:
        for attempt in range(1, 5):
            try:
                response = requests.get(url, timeout=(15, 90))
                response.raise_for_status()
            except requests.RequestException as error:
                if attempt == 4:
                    raise RuntimeError(f"DEM中转获取失败：{name}/{type(error).__name__}") from None
                time.sleep(2 ** attempt)
                continue
            if response.headers.get("Content-Length") is not None and len(response.content) != int(response.headers["Content-Length"]):
                raise ValueError("本机中转DEM字节数不完整")
            partial = path.with_suffix(".tif.part")
            partial.write_bytes(response.content)
            partial.replace(path)
            receipt = {"url": url, "bytes": len(response.content), "sha256": hashlib.sha256(response.content).hexdigest(),
                       "downloaded_utc": utc_now(), "etag": response.headers.get("ETag"), "attempts": attempt,
                       "transport": "local_download_ssh_verified_relay"}
            write_json(metadata, receipt)
            break
    remote = args.server_root + "/data/raw/copdem-glo90-2021/tiles/.incoming-client/"
    # 连接中断后重传整个事务，服务器可能已经发布并清理过在途文件。
    response = relay_files(args.server, [path, metadata], remote,
                           server_command(args, "--publish", (name, "--expected-sha", receipt["sha256"])))
    confirmed = json.loads(response.stdout)
    if confirmed["status"] != "verified" or confirmed["sha256"] != receipt["sha256"]:
        raise ValueError("服务器未确认DEM原始字节校验通过")
    for own_file in (path, metadata):
        if not own_file.resolve().is_relative_to(cache.resolve()):
            raise ValueError("中转缓存清理路径超出本任务缓存目录")
        own_file.unlink()
    return receipt["bytes"]


def run(args):
    response = remote_command(args.server, server_command(args, "--make-plan"))
    plan = json.loads(response.stdout)
    response = remote_command(args.server, server_command(args, "--list-present"))
    present = set(json.loads(response.stdout))
    names = [name for name in plan["tiles"] if name not in present]
    if any(not NAME.fullmatch(name) for name in names):
        raise ValueError("DEM中转计划出现未登记的切片名")
    command = shlex.join(["mkdir", "-p", args.server_root + "/data/raw/copdem-glo90-2021/tiles/.incoming-client"])
    remote_command(args.server, command)
    status_path = args.root / ".codex-temp/buildability-20261025/dem-relay-status.json"
    state = {"status": "running", "started_utc": utc_now(), "updated_utc": utc_now(), "selected_tiles": len(plan["tiles"]),
             "already_on_server": len(present.intersection(plan["tiles"])), "remaining_at_start": len(names), "transferred_tiles": 0, "transferred_bytes": 0}
    write_json(status_path, state)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(transfer, args, name): name for name in names}
        for future in as_completed(futures):
            error = future.exception()
            if error is not None:
                for pending in futures:
                    pending.cancel()
                state.update(status="failed", error=str(error), failed_tile=futures[future], updated_utc=utc_now())
                write_json(status_path, state)
                raise error
            size = future.result()
            state.update(transferred_tiles=state["transferred_tiles"] + 1, transferred_bytes=state["transferred_bytes"] + size, updated_utc=utc_now())
            write_json(status_path, state)
            if state["transferred_tiles"] % 50 == 0:
                print(json.dumps(state), flush=True)
    state.update(status="complete", updated_utc=utc_now(), completed_utc=utc_now())
    write_json(status_path, state)
    print(json.dumps(state), flush=True)


def main():
    parser = argparse.ArgumentParser(description="中转同源Copernicus DEM，服务器校验后发布，本机仅保留临时在途缓存。")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--server", default="pku-test")
    parser.add_argument("--server-root", default="/data3/dataset/ljt/global-grid-flexibility-20261023")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--make-plan", action="store_true")
    parser.add_argument("--list-present", action="store_true")
    parser.add_argument("--publish")
    parser.add_argument("--expected-sha")
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.make_plan:
        make_plan(args.root)
    elif args.list_present:
        directory = args.root / "data/raw/copdem-glo90-2021/tiles"
        print(json.dumps([path.name[:-len(".tif.manifest.json")] for path in directory.glob("*.tif.manifest.json") if path.with_name(path.name[:-len(".manifest.json")]).exists()]))
    elif args.publish:
        publish(args.root, args.publish, args.expected_sha)
    else:
        run(args)


if __name__ == "__main__":
    main()
