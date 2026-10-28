import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

import requests


BASE = "https://data-gis.unep-wcmc.org/server/rest/services/ProtectedPlanet/WDPCA/MapServer"
SNAPSHOT = "October 2026"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def download_json(url, parameters, target):
    target = Path(target)
    receipt_path = target.with_suffix(target.suffix + ".manifest.json")
    if target.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["url"] != url or receipt["parameters"] != parameters or sha(target) != receipt["sha256"]:
            raise ValueError(f"保护地原文与来源清单不一致：{target}")
        return json.loads(target.read_text(encoding="utf-8")), receipt
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 5):
        try:
            response = requests.get(url, params=parameters, timeout=(15, 90))
            response.raise_for_status()
        except requests.RequestException as error:
            if attempt == 4:
                raise RuntimeError(f"保护地请求四次失败：{url}；错误类型={type(error).__name__}") from None
            time.sleep(2 ** attempt)
            continue
        value = response.json()
        if "error" in value:
            raise ValueError(f"保护地服务返回错误：{value['error']}")
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(response.content)
        temporary.replace(target)
        receipt = {"url": url, "parameters": parameters, "sha256": sha(target), "bytes": target.stat().st_size,
                   "downloaded_utc": utc_now(), "http_status": response.status_code, "attempts": attempt}
        write_json(receipt_path, receipt)
        return value, receipt
    raise RuntimeError("保护地请求没有返回结果")


def page(root, layer, number, identifiers):
    target = root / "data/raw/wdpca-2026-10" / f"layer-{layer}" / f"page-{number:05d}.geojson"
    split_path = target.parent / "split-pages.json"
    if split_path.exists():
        splits = json.loads(split_path.read_text(encoding="utf-8"))
        if str(number) in splits:
            size = int(splits[str(number)])
            if not 0 < size < len(identifiers):
                raise ValueError("保护地显式子请求大小不正确")
            children = [fetch_page(root, layer, number, identifiers[offset:offset + size],
                                   target.with_name(f"page-{number:05d}-part-{part:03d}.geojson"))
                        for part, offset in enumerate(range(0, len(identifiers), size))]
            return {"subpages": children, "features": sum(item["features"] for item in children), "bytes": sum(item["bytes"] for item in children)}
    return fetch_page(root, layer, number, identifiers, target)


def fetch_page(root, layer, number, identifiers, target):
    parameters = {"f": "geojson", "objectIds": ",".join(str(item) for item in identifiers),
                  "outFields": "objectid,site_id,site_pid,site_type,realm,rep_area,status,iucn_cat", "outSR": "4326", "returnGeometry": "true"}
    value, receipt = download_json(BASE + f"/{layer}/query", parameters, target)
    received = [int(feature["properties"]["objectid"]) for feature in value["features"]]
    if value.get("exceededTransferLimit") or len(received) != len(identifiers) or sorted(received) != sorted(identifiers):
        raise ValueError(f"保护地页未返回全部已冻结ID：{layer}/{number}")
    if any(feature["geometry"] is None for feature in value["features"]):
        raise ValueError(f"保护地页存在没有空间几何的记录：{layer}/{number}")
    return {"relative_path": str(target.relative_to(root)).replace("\\", "/"), "layer": layer,
            "page": number, "features": len(received), "sha256": receipt["sha256"], "bytes": receipt["bytes"]}


def run(args):
    root = args.root.resolve()
    raw = root / "data/raw/wdpca-2026-10"
    metadata_dir = root / "data/metadata/buildability-20261025"
    status_path = root / ".codex-temp/buildability-20261025/protection-acquisition.json"
    service, service_receipt = download_json(BASE, {"f": "json"}, raw / "service.json")
    if SNAPSHOT not in service["copyrightText"] or SNAPSHOT not in service["documentInfo"]["Subject"]:
        raise ValueError("保护地服务版本不属于已登记的2026年10月快照")
    state = {"status": "downloading", "started_utc": utc_now(), "updated_utc": utc_now(), "completed_pages": 0}
    write_json(status_path, state)
    jobs, layers = [], []
    for layer in (0, 1):
        description, receipt = download_json(BASE + f"/{layer}", {"f": "json"}, raw / f"layer-{layer}.json")
        if SNAPSHOT not in description["copyrightText"]:
            raise ValueError("保护地点与面图层月份不一致")
        ids, id_receipt = download_json(BASE + f"/{layer}/query", {"f": "json", "where": "1=1", "returnIdsOnly": "true"}, raw / f"layer-{layer}-ids.json")
        identifiers = sorted(int(item) for item in ids["objectIds"])
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("保护地服务ID列表不唯一")
        layers.append({"id": layer, "name": description["name"], "features": len(identifiers),
                       "metadata_sha256": receipt["sha256"], "ids_sha256": id_receipt["sha256"]})
        for number, offset in enumerate(range(0, len(identifiers), args.batch_size)):
            jobs.append((layer, number, identifiers[offset:offset + args.batch_size]))
    state.update(total_pages=len(jobs), total_features=sum(layer["features"] for layer in layers))
    write_json(status_path, state)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(page, root, *job): job[:2] for job in jobs}
        for future in as_completed(futures):
            error = future.exception()
            if error is not None:
                for pending in futures:
                    pending.cancel()
                state.update(status="failed", failed_source_batch=futures[future], error_type=type(error).__name__, error=str(error), updated_utc=utc_now())
                write_json(status_path, state)
                raise error
            result = future.result()
            records.extend(result["subpages"] if "subpages" in result else [result])
            state.update(completed_pages=state["completed_pages"] + 1, verified_source_files=len(records), updated_utc=utc_now(), downloaded_bytes=sum(item["bytes"] for item in records))
            write_json(status_path, state)
            if state["completed_pages"] % 10 == 0 or state["completed_pages"] == len(jobs):
                print(json.dumps(state, ensure_ascii=False), flush=True)
    current = requests.get(BASE, params={"f": "json"}, timeout=(15, 45))
    current.raise_for_status()
    if current.json()["copyrightText"] != service["copyrightText"]:
        raise ValueError("保护地获取过程中服务月份发生变化，不能发布混合月份快照")
    manifest = {"status": "complete", "completed_utc": utc_now(), "version": "WDPCA October 2026",
                "service_url": BASE, "service_metadata_sha256": service_receipt["sha256"], "layers": layers,
                "feature_count": sum(item["features"] for item in records), "pages": sorted(records, key=lambda item: (item["layer"], item["page"])),
                "source_citation": service["copyrightText"], "license_url": "https://www.unep-wcmc.org/en/wdpa-data-license",
                "use": "非商业科研，原始几何与属性仅存本研究工作区，不提交Git或公开再分发"}
    if manifest["feature_count"] != state["total_features"]:
        raise ValueError("保护地原文总数与冻结ID总数不一致")
    write_json(metadata_dir / "protection-source-manifest.json", manifest)
    if args.server:
        remote_root = args.server_root
        command = shlex.join(["mkdir", "-p", remote_root + "/data/raw", remote_root + "/data/metadata/buildability-20261025"])
        subprocess.run(["ssh", "-o", "BatchMode=yes", args.server, command], check=True)
        state.update(status="uploading_verified_snapshot", updated_utc=utc_now())
        write_json(status_path, state)
        subprocess.run(["scp", "-q", "-r", str(raw), args.server + ":" + remote_root + "/data/raw/"], check=True)
        subprocess.run(["scp", "-q", str(metadata_dir / "protection-source-manifest.json"),
                        args.server + ":" + remote_root + "/data/metadata/buildability-20261025/"], check=True)
    state.update(status="complete", completed_utc=utc_now(), updated_utc=utc_now())
    write_json(status_path, state)
    print(json.dumps(state, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="冻结2026年10月官方保护地面与点，按ID完整获取并校验。")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--batch-size", type=int, default=500, choices=range(1, 1001))
    parser.add_argument("--server")
    parser.add_argument("--server-root", default="/data3/dataset/ljt/global-grid-flexibility-20261023")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
