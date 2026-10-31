import argparse
import json
from pathlib import Path
import shutil

import requests

from acquire_alibaba_cpu import BASE, now, sha256
from stream_alibaba_components import CHECKSUMS


def audit(directory,output):
    before=shutil.disk_usage(directory).free
    if before < 35*1024**3:
        raise OSError("系统盘可用空间低于 35 GiB")
    result={"checked_at_utc":now(),"directory":str(directory.resolve()),"retention":"用户于 2026-10-31 16:37 北京时间批准保存；不解压到磁盘",
            "free_before_bytes":before,"floor_bytes":35*1024**3,"files":{}}
    session=requests.Session()
    session.trust_env=False
    for name,expected in CHECKSUMS.items():
        path=directory / f"{name}.tar.gz"
        response=session.head(f"{BASE}/{path.name}",timeout=60)
        response.raise_for_status()
        size=int(response.headers["Content-Length"])
        actual=sha256(path)
        if path.stat().st_size != size or actual != expected:
            raise ValueError(f"本地压缩包与官方 SHA/字节数不符：{path}")
        result["files"][name]={"path":str(path.resolve()),"url":response.url,"bytes":size,"sha256":actual,
                               "official_sha256_matches":True,"etag":response.headers.get("ETag"),
                               "last_modified":response.headers.get("Last-Modified")}
        print(json.dumps(result["files"][name]),flush=True)
    result["external_download_log"]=(directory / "fetch.log").read_text(encoding="utf-8")
    monitor=directory / "space-monitor.json"
    if monitor.exists():
        result["space_monitor"]=json.loads(monitor.read_text())
        if result["space_monitor"]["minimum_observed_free_bytes"] < result["floor_bytes"]:
            raise ValueError("系统盘空间监测曾低于约定阈值")
    result["free_after_bytes"]=shutil.disk_usage(directory).free
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="审计已有系统盘 Alibaba 大表缓存；不下载、不解压")
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    audit(args.directory,args.output)
