import argparse
from collections import Counter
import fcntl
import io
import json
from pathlib import Path
import shutil
import tarfile
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as arrow_csv
import requests
from isal import igzip_threaded

from acquire_alibaba_cpu import BASE, now, sha256
from audit_alibaba_cpu import END, HOURS, read_table
from alibaba_online_accumulator import accumulate
from stream_alibaba_components import CHECKSUMS, Forward, HashStream


def csv_tables(stream, chunk_rows=2000000):
    # 限制解析库的线程与预读资源，避免按服务器总核数扩大缓存。
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    buffered=io.BufferedReader(Forward(stream),buffer_size=4*1024**2)
    reader=arrow_csv.open_csv(buffered,
        read_options=arrow_csv.ReadOptions(use_threads=False,autogenerate_column_names=True,block_size=8*1024**2),
        convert_options=arrow_csv.ConvertOptions(include_columns=["f0","f1","f2","f3"],
            column_types={"f0":pa.dictionary(pa.int32(),pa.string()),
                          "f1":pa.dictionary(pa.int32(),pa.string()),
                          "f2":pa.int64(),"f3":pa.float64()}))
    pending=[]
    for batch in reader:
        pending.append(batch)
        table=pa.Table.from_batches(pending)
        while table.num_rows >= chunk_rows:
            yield table.slice(0,chunk_rows)
            table=table.slice(chunk_rows)
        pending=table.to_batches()
    if pending:
        yield pa.Table.from_batches(pending)


def dictionary_codes(column, lookup):
    values=column.combine_chunks()
    if values.null_count:
        raise ValueError("标识符字典包含未定义的空值")
    mapping=np.array([lookup.get(value,-1) for value in values.dictionary.to_pylist()],dtype=np.int64)
    return mapping[values.indices.to_numpy()]


def run(raw, output, stop_after_chunks=None, archive_path=None):
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "online-native.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inventory = pd.read_csv(output.parent / "machine-inventory.csv")
    hosts = dict(zip(inventory.machine_id, range(len(inventory))))
    meta = read_table(raw, "container_meta")
    containers = {key: i for i, key in enumerate(sorted(meta.container_id.unique()))}
    meta["cid"] = meta.container_id.map(containers)
    meta = meta.sort_values(["cid", "timestamp"])
    if meta.timestamp.max() >= 10**9 or meta.timestamp.min() < 0:
        raise ValueError("元数据时间超出连接范围")
    keys = meta.cid.to_numpy(np.int64)*10**9+meta.timestamp.to_numpy(np.int64)
    variation=meta.groupby("cid")[["machine_id","cpu_request","cpu_limit"]].nunique()
    if variation.machine_id.gt(1).any() or variation.cpu_request.gt(1).any():
        raise ValueError("主机或 CPU request 随时间变化，需按事件连接后重新定义累计")
    first_meta=meta.groupby("cid",sort=True).first()
    request=first_meta.cpu_request.to_numpy(float)
    limit=first_meta.cpu_limit.to_numpy(float)
    first_time=first_meta.timestamp.to_numpy(float)
    host_name_index={name:i for i,name in enumerate(sorted(meta.machine_id.dropna().unique()))}
    host_name_codes=first_meta.machine_id.map(host_name_index).fillna(-2).to_numpy(np.int64)
    host_codes=first_meta.machine_id.map(hosts).fillna(-1).to_numpy(int)
    changing_limit=variation.index[variation.cpu_limit.gt(1)].to_numpy()
    all_limits=meta.cpu_limit.to_numpy(float)
    names = ("request_based_core_hours", "limit_based_core_hours", "one_core_based_core_hours", "sample_container_hours")
    totals = {key: np.zeros((len(inventory), HOURS)) for key in names}
    counts = np.zeros((len(containers), HOURS), dtype=np.uint16)
    slots_per_container=END//10
    seen_slots=np.zeros((len(containers)*slots_per_container+7)//8,dtype=np.uint8)
    stats, daily_counts = Counter(), Counter()
    processed, source_min, source_max = 0, np.inf, -np.inf
    checkpoint = output / "online-exact-checkpoint.npz"
    state = {"source": "container_usage", "url": f"{BASE}/container_usage.tar.gz", "started_at_utc": now(),
        "method": "HTTP 流解压，按原始 10 秒测量格累计；不依赖原始记录顺序；仅保留小时统计及恢复点",
        "sample_support_seconds": 10, "time_shift_seconds": 0,
        "deduplication":"容器×原始时间戳仅保留文件首条有效记录；位图仅为恢复状态，不保存逐条 CPU 值",
        "restart": "重新读取压缩流，跳过检查点已提交行；压缩流 SHA-256 每次完整重算",
        "code_sha256": sha256(__file__)}
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            previous = json.loads(str(saved["metadata"]))
            if previous["state"]["code_sha256"] not in (state["code_sha256"], "cb7562e45ac6d1fad045a54742076ccbac4b8490e9334b739b58f5679cda37b6", "56d0fb939c4499cf1a20c02c929bfd074162653840e325af3199929b04842ac2", "d7d47770a00b5baa470817e48536ce16734a4fea447dbeb0c55340d082bd564f", "a5889d40af57fddb7e26805f67d3304c9035d98663432e2dc2452553a2f76a80"):
                raise ValueError("检查点代码版本不符")
            accumulator_digest=sha256(Path(__file__).with_name("alibaba_online_accumulator.py"))
            if previous["state"].get("accumulator_sha256",accumulator_digest) != accumulator_digest:
                raise ValueError("检查点累计器版本不符")
            current_code = state["code_sha256"]
            totals = {key: saved[key].copy() for key in names}
            counts = saved["counts"].copy()
            seen_slots=saved["seen_slots"].copy()
            processed, stats = previous["processed"], Counter(previous["stats"])
            daily_counts = Counter(previous["daily_counts"])
            source_min=previous["source_min"] if previous["source_min"] is not None else np.inf
            source_max=previous["source_max"] if previous["source_max"] is not None else -np.inf
            state = previous["state"]
            if state.get("status") == "complete":
                print("已校验的容器汇总可复用", flush=True)
                return
            state.update(resumed_at_utc=now(), resume_from_rows=processed)
            state["resumed_code_sha256"]=state["code_sha256"]
            state["code_sha256"] = current_code
    session = requests.Session()
    session.trust_env = False
    response = session.head(state["url"],timeout=60) if archive_path else session.get(state["url"],stream=True,timeout=(30,120),headers={"Accept-Encoding":"identity"})
    response.raise_for_status()
    etag = response.headers.get("ETag")
    if state.get("etag",etag) != etag:
        raise ValueError("恢复时上游 ETag 改变")
    state.update(etag=etag,bytes_expected=int(response.headers["Content-Length"]),last_modified=response.headers.get("Last-Modified"))
    if archive_path:
        if archive_path.stat().st_size != state["bytes_expected"]:
            raise ValueError("本地缓存字节数尚未完整")
        input_stream=archive_path.open("rb")
        state["input_archive"]=str(archive_path.resolve())
    else:
        input_stream=response.raw
    compressed, started = HashStream(input_stream), time.monotonic()
    state["gzip_decoder"]="python-isal igzip_threaded"
    state["csv_parser"]="pyarrow.csv.open_csv；字典化标识符与原生数值数组，每批至多 2000000 行"
    state["accumulator_sha256"]=sha256(Path(__file__).with_name("alibaba_online_accumulator.py"))

    def save(status,write_checkpoint=True):
        free = shutil.disk_usage(output).free
        if free < 5*1024**3:
            raise OSError("写入汇总前空间低于 5 GiB")
        state.update(status=status,updated_at_utc=now(),processed_rows=processed,compressed_bytes_read=compressed.bytes,
                     free_bytes=free,elapsed_seconds_this_run=time.monotonic()-started,stats=dict(stats))
        metadata = {"state":state,"processed":processed,"stats":dict(stats),"daily_counts":dict(daily_counts),
                    "source_min":source_min if np.isfinite(source_min) else None,"source_max":source_max if np.isfinite(source_max) else None}
        if write_checkpoint:
            state["checkpoint_rows"]=processed
            temporary = checkpoint.with_suffix(".partial.npz")
            np.savez(temporary,**totals,counts=counts,seen_slots=seen_slots,metadata=json.dumps(metadata))
            temporary.replace(checkpoint)
        status_path=output / "online-native-status.json"
        status_temp=status_path.with_suffix(".partial.json")
        status_temp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
        status_temp.replace(status_path)
        print(json.dumps(state,ensure_ascii=False),flush=True)

    save("processing",not checkpoint.exists())
    seen, chunks = 0, 0
    with igzip_threaded.open(compressed,"rb",threads=1,block_size=8*1024**2) as decoded, tarfile.open(fileobj=decoded,mode="r|",bufsize=4*1024**2) as archive:
        member=archive.next()
        if member.name != "container_usage.csv":
            raise ValueError("非预期归档成员")
        state["uncompressed_member_bytes"] = member.size
        with archive.extractfile(member) as stream:
            for table in csv_tables(stream):
                before, seen=seen,seen+table.num_rows
                if seen <= processed:
                    continue
                table=table.slice(max(0,processed-before))
                stats["raw_rows"] += table.num_rows
                t=np.asarray(table["f2"].combine_chunks().to_numpy(zero_copy_only=False),dtype=float)
                u=np.asarray(table["f3"].combine_chunks().to_numpy(zero_copy_only=False),dtype=float)
                if not np.isfinite(t).all() or (t < 0).any() or (t >= 10**9).any() or (t % 10 != 0).any():
                    raise ValueError("容器时间戳不在有效 10 秒时间格")
                source_min,source_max=min(source_min,float(t.min())),max(source_max,float(t.max()))
                unique,n=np.unique((t//86400).astype(int),return_counts=True)
                daily_counts.update({str(int(day)):int(count) for day,count in zip(unique,n)})
                cid=dictionary_codes(table["f0"],containers)
                raw_host=dictionary_codes(table["f1"],host_name_index)
                safe_cid=np.maximum(cid,0)
                machine=host_codes[safe_cid]
                in_window=(t >= 0)&(t < END)
                invalid=~np.isfinite(u)|(u < 0)|(u > 100)
                stats["invalid_cpu_rows"] += int(invalid.sum())
                stats["outside_machine_window_rows"] += int((~in_window).sum())
                stats["unknown_container_rows"] += int((cid < 0).sum())
                request_rows=request[safe_cid]
                limit_rows=limit[safe_cid].copy()
                dynamic=(cid >= 0)&np.isin(cid,changing_limit)
                if dynamic.any():
                    lookup=cid[dynamic]*10**9+t[dynamic].astype(np.int64)
                    positions=np.maximum(np.searchsorted(keys,lookup,side="right")-1,0)
                    limit_rows[dynamic]=all_limits[positions]
                valid_meta=(cid >= 0)&(first_time[safe_cid] <= t)&(raw_host == host_name_codes[safe_cid])
                stats["unmatched_meta_rows_in_window"] += int((in_window&~valid_meta).sum())
                good=in_window&~invalid&valid_meta&(machine >= 0)&(request_rows >= 0)&(limit_rows >= 0)
                stats["valid_joined_rows"] += int(good.sum())
                stats["request_limit_different_valid_rows"] += int((good&(request_rows != limit_rows)).sum())
                kept,duplicates=accumulate(t,u,cid,machine,request_rows,limit_rows,good,slots_per_container,
                    seen_slots,counts,totals["request_based_core_hours"],totals["limit_based_core_hours"],
                    totals["one_core_based_core_hours"],totals["sample_container_hours"])
                stats["duplicate_valid_rows"] += duplicates
                stats["unique_valid_joined_rows"] += kept
                if counts.max() > 360:
                    raise ValueError("同容器小时超过 360 条测量，须核对重复时间或主机迁移")
                processed=seen
                chunks+=1
                if chunks % 100 == 0:
                    save("processing")
                elif chunks % 10 == 0:
                    save("processing",False)
                if stop_after_chunks and chunks >= stop_after_chunks:
                    save("paused_for_resume_check")
                    response.close()
                    return
        if archive.next() is not None:
            raise ValueError("归档存在额外成员")
        while decoded.read(8*1024**2):
            pass
    while compressed.read(8*1024**2):
        pass
    response.close()
    input_stream.close()
    digest=compressed.digest.hexdigest()
    if digest != CHECKSUMS["container_usage"] or compressed.bytes != state["bytes_expected"]:
        raise ValueError(f"容器压缩流 SHA 或字节数不符：{digest}, {compressed.bytes}")
    state.update(sha256=digest,official_sha256_matches=True,completed_at_utc=now(),
                 timestamp_min=source_min,timestamp_max=source_max,daily_raw_sample_counts=dict(daily_counts),
                 maximum_samples_per_container_hour=int(counts.max()))
    popcount=np.array([value.bit_count() for value in range(256)],dtype=np.uint8)
    unique_slots=int(popcount[seen_slots].sum())
    if unique_slots != int(counts.sum()) or unique_slots != stats["unique_valid_joined_rows"]:
        raise ValueError("去重位图、小时计数与累计行数不一致")
    state["unique_valid_slots"]=unique_slots
    table=pd.DataFrame({"machine_id":np.repeat(inventory.machine_id.to_numpy(),HOURS),"relative_hour":np.tile(np.arange(HOURS),len(inventory))})
    for key,values in totals.items():
        table[key]=values.ravel()
    table.to_parquet(output / "container_usage-machine-hourly.parquet",index=False)
    table.groupby("relative_hour")[list(names)].sum().to_csv(output / "container_usage-hourly.csv")
    count_hist=np.bincount(counts.ravel(),minlength=361)
    pd.DataFrame({"samples_per_container_hour":np.arange(len(count_hist)),"container_hours":count_hist}).to_csv(output / "online-sample-coverage-histogram.csv",index=False)
    save("complete")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="按无序容器样本累计原始 10 秒支持时段的 CPU 核小时")
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stop-after-chunks",type=int)
    parser.add_argument("--archive",type=Path,help="复用用户已下载的系统盘原始压缩包；不解压落盘")
    args=parser.parse_args()
    run(args.raw,args.output,args.stop_after_chunks,args.archive)
