import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
ASSETS = [
    ("burst-1", "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_1.csv", 52283111,
     "4bb3783693d0a435686fbfc885615d2349bd067239079fa4b749f2e679e12122"),
    ("burst-2", "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_2.csv", 144819209,
     "44bf5942b03fca42c01a226a545ad3a750ec62688faecd67b6831c56a8928bf7"),
    ("burst-3", "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_3.csv", 231682327,
     "2299986a07388aa303ec2c41d1131e756db650a39ed6ef9dfe7cc3d7f9a43b8f"),
    ("azure-code", "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv",
     691989454, "71de5c55cbc35f8f1ed0b6b7806b4cd1e9764b0058469725a6aac98023a1448f"),
    ("azure-conv", "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv",
     1135195393, "a0cc9b969a9bbf0fd811802cbf4323edd3a209ace791e3799ad4f9207f213941"),
]


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def download(url, path, size, expected, reserve_gib, parallel_gib):
    if size > 5 * 1024**3:
        raise ValueError("单个文件超过小数据下载上限")
    if path.exists():
        if path.stat().st_size != size or digest(path) != expected:
            raise ValueError(f"已有文件校验失败: {path}")
        return {"reused": True, "seconds": 0}
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    # 保留量涵盖另一个任务的并发下载；每次写入前再次检查当前文件系统。
    reserve = (reserve_gib + parallel_gib) * 1024**3
    if shutil.disk_usage(path.parent).free - size < reserve:
        raise ValueError("可用磁盘不足以满足下载后保留量")
    began = time.perf_counter()
    curl = "curl.exe" if sys.platform == "win32" else "curl"
    range_args = ["--continue-at", "-"] if partial.exists() and partial.stat().st_size else ["--range", f"0-{size - 1}"]
    process = subprocess.Popen([curl, "--fail", "--location", "--silent", "--show-error",
        "--connect-timeout", "30", "--max-time", "1200", "--speed-time", "30", "--speed-limit", "1024",
        "--max-filesize", str(size), *range_args, "--output", str(partial), url])
    while process.poll() is None:
        if shutil.disk_usage(path.parent).free < reserve:
            process.terminate()
            process.wait()
            raise ValueError("下载期间磁盘低于保留量，停止写入")
        time.sleep(.25)
    if process.returncode != 0:
        raise RuntimeError(f"curl 下载失败: {path.name}, exit={process.returncode}")
    if partial.stat().st_size != size or digest(partial) != expected:
        raise ValueError(f"下载校验失败: {path.name}")
    partial.replace(path)
    return {"reused": False, "seconds": time.perf_counter() - began}


def summarize_file(name, path):
    chunks, min_s, max_s, rows, failures = [], float("inf"), -float("inf"), 0, 0
    schema = list(pd.read_csv(path, nrows=0).columns)
    for frame in pd.read_csv(path, chunksize=250000):
        if name.startswith("burst"):
            seconds = pd.to_numeric(frame["Timestamp"], errors="raise")
            frame = frame.assign(input_tokens=frame["Request tokens"], output_tokens=frame["Response tokens"],
                service=frame["Model"].astype(str) + " / " + frame["Log Type"].astype(str))
        else:
            timestamps = pd.to_datetime(frame["TIMESTAMP"], format="mixed", errors="raise", utc=True)
            # 2024 发布文件的时间戳带有 +00:00，保留其 UTC 坐标。
            seconds = (timestamps - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
            frame = frame.assign(input_tokens=frame.ContextTokens, output_tokens=frame.GeneratedTokens,
                                 service=name.removeprefix("azure-"))
        if not np.isfinite(seconds).all() or (frame[["input_tokens", "output_tokens"]] < 0).any().any():
            raise ValueError(f"时间或 token 字段无效: {name}")
        if frame[["input_tokens", "output_tokens"]].isna().any().any():
            raise ValueError(f"token 缺失: {name}")
        min_s, max_s = min(min_s, float(seconds.min())), max(max_s, float(seconds.max()))
        rows += len(frame)
        frame["hour_index"] = np.floor(seconds / 3600).astype("int64")
        frame["requests"] = 1
        frame["zero_output_requests"] = (frame.output_tokens == 0).astype(int)
        failures += int(frame.zero_output_requests.sum())
        frame["successful_input_tokens"] = frame.input_tokens.where(frame.output_tokens > 0, 0)
        chunks.append(frame.groupby(["hour_index", "service"], observed=True)[
            ["requests", "input_tokens", "output_tokens", "successful_input_tokens", "zero_output_requests"]].sum())
    hourly = pd.concat(chunks).groupby(level=[0, 1]).sum().reset_index()
    return hourly, {"rows": rows, "min_seconds": min_s, "max_seconds": max_s,
                    "zero_output_requests": failures, "schema": schema}


def ratio(values):
    minimum = float(values.min())
    return float(values.max() / minimum) if minimum > 0 else None


def profiles(hourly, series, minimum, maximum):
    first_day, end_day = int(np.ceil(minimum / 86400)), int(np.floor(maximum / 86400))
    if end_day <= first_day:
        raise ValueError(f"没有完整日: {series}")
    index = pd.Index(range(first_day * 24, end_day * 24), name="hour_index")
    total = hourly.groupby("hour_index").sum(numeric_only=True).reindex(index, fill_value=0)
    columns = ["requests", "input_tokens", "output_tokens", "successful_input_tokens", "zero_output_requests"]
    total = total[columns]
    if total.requests.sum() > hourly.requests.sum():
        raise ValueError("完整日汇总违反请求数量守恒")
    total["relative_day"] = total.index // 24
    total["hour_of_day"] = total.index % 24
    total["relative_day_mod7"] = total.relative_day % 7
    complete = total.copy()
    daily = total.groupby("relative_day")[columns].sum()
    diurnal = total.groupby("hour_of_day")[columns].mean()
    weekly = total.groupby("relative_day_mod7")[columns].mean()
    metrics = {"series": series, "first_complete_day_index": first_day, "end_day_index_exclusive": end_day,
        "complete_days": end_day - first_day, "complete_hours": len(total),
        "zero_observed_hours": int((total.requests == 0).sum()),
        "weekday_basis": "UTC epoch day modulo 7; 0 is Thursday" if series.startswith("azure-")
            else "relative day modulo 7; original first weekday is not provided",
        "requests_hourly_peak_valley": ratio(total.requests),
        "requests_hourly_p95_p05": float(total.requests.quantile(.95) / total.requests.quantile(.05))
            if total.requests.quantile(.05) > 0 else None,
        "requests_diurnal_peak_hour": int(diurnal.requests.idxmax()),
        "requests_diurnal_valley_hour": int(diurnal.requests.idxmin()),
        "requests_diurnal_peak_valley": ratio(diurnal.requests),
        "requests_mod7_peak_valley": ratio(weekly.requests),
        "output_tokens_hourly_peak_valley": ratio(total.output_tokens),
        "output_tokens_diurnal_peak_valley": ratio(diurnal.output_tokens),
        "output_tokens_mod7_peak_valley": ratio(weekly.output_tokens),
        "requests_lag24_correlation": float(total.requests.autocorr(24)),
        "requests_lag168_correlation": float(total.requests.autocorr(168)) if len(total) > 168 else None,
        "input_tokens_total": int(total.input_tokens.sum()), "output_tokens_total": int(total.output_tokens.sum()),
        "requests_total": int(total.requests.sum()),
        "token_to_gpu_hours": "not calibrated; no observed GPU-hour or power is present"}
    for frame in (complete, daily, diurnal, weekly):
        frame.insert(0, "series", series)
    return (complete, daily, diurnal, weekly), metrics


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    metadata, frames = [], {}
    for name, url, size, expected in ASSETS:
        if name not in args.datasets:
            continue
        path = args.raw_dir / url.rsplit("/", 1)[1]
        transfer = download(url, path, size, expected, args.reserve_gib, args.parallel_gib)
        frame, detail = summarize_file(name, path)
        if int(frame.requests.sum()) != detail["rows"]:
            raise ValueError("请求数量不守恒")
        frames[name] = frame
        record = dict(dataset=name, url=url, bytes=size, sha256=expected,
                      local_path=str(path.resolve()), **transfer, **detail)
        metadata.append(record)
        print(json.dumps(record, ensure_ascii=False), file=sys.stderr, flush=True)
        # 每个数据集完成后保存清单，重跑时复用经过校验的文件。
        (args.output / "download-manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    selections = []
    lookup = {item["dataset"]: item for item in metadata}
    if {"burst-1", "burst-2"}.issubset(frames):
        if lookup["burst-2"]["min_seconds"] <= lookup["burst-1"]["max_seconds"]:
            raise ValueError("BurstGPT 前两文件时间覆盖重叠，禁止直接连接")
        selections.append(("burst-1+2", pd.concat([frames["burst-1"], frames["burst-2"]]),
                           lookup["burst-1"]["min_seconds"], lookup["burst-2"]["max_seconds"]))
    for name in ("burst-3", "azure-code", "azure-conv"):
        if name in frames:
            item = lookup[name]
            selections.append((name, frames[name], item["min_seconds"], item["max_seconds"]))
    results, shapes = [], [[], [], [], []]
    for name, frame, minimum, maximum in selections:
        profiles_out, metrics = profiles(frame, name, minimum, maximum)
        results.append(metrics)
        for index, part in enumerate(profiles_out):
            shapes[index].append(part)
        for service, part in frame.groupby("service"):
            _, service_metrics = profiles(part, f"{name} / {service}", minimum, maximum)
            results.append(service_metrics)
    for name, parts in zip(("request-hourly.csv", "request-daily.csv", "request-diurnal.csv", "request-mod7.csv"), shapes):
        pd.concat(parts).to_csv(args.output / name, index=True)
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "datasets": metadata, "profiles": results,
              "boundary_policy": "exclude partial first/last calendar days; zero fill unobserved bins only inside coverage"}
    (args.output / "request-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["profiles"], ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载官方小型推理请求表并核验 SHA-256，输出跨日、日内和相对七日曲线")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=[a[0] for a in ASSETS], default=["burst-1", "burst-2"])
    parser.add_argument("--reserve-gib", type=float, default=30)
    parser.add_argument("--parallel-gib", type=float, default=2)
    main(parser.parse_args())
