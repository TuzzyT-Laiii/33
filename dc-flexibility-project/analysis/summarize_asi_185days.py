import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import requests


KEYS = ["job_type_public", "priority_class", "model_type_public"]
VALUES = ["used_gpu_hours", "requested_gpu", "requested_cpu_cores", "pod_rows", "pod_count", "workload_count",
          "gpu_requesting_rows", "gpu_requesting_pods", "gpu_requesting_workloads", "missing_gpu_hours",
          "missing_gpu_request_rows", "missing_workload_rows"]
COUNT_SUMS = {"pod_count": "sum_group_hourly_distinct_pods", "workload_count": "sum_group_hourly_distinct_workloads",
              "gpu_requesting_pods": "sum_group_hourly_gpu_pods", "gpu_requesting_workloads": "sum_group_hourly_gpu_workloads"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def ratio(values):
    return float(values.max() / values.min()) if values.min() > 0 else None


def divide(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def correlation(values, lag):
    value = values.autocorr(lag) if len(values) > lag + 1 and values.std() > 0 else float("nan")
    return float(value) if np.isfinite(value) else None


def execution_type_summary(jobs):
    metrics = pd.DataFrame(jobs["metrics"])
    durations = pd.DataFrame(jobs["duration_histogram"])
    scales = pd.DataFrame(jobs["size_histogram"])
    duration_edges = [0, 1/60, 5/60, 1/6, .5, 1, 3, 6, 12, 24, 72, 168, 720, 2160, 4440, float("inf")]
    size_edges = [0, .125, .25, .5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, float("inf")]

    def bins(edges):
        return [f"[{a:g},{b:g})" for a, b in zip(edges[:-1], edges[1:])]

    rows = []
    for kind, group in metrics.groupby("job_type_public"):
        count = int(group.execution_spans.sum())
        duration_counts = durations[durations.job_type_public == kind].groupby("bin").observations.sum().reindex(bins(duration_edges), fill_value=0)
        size_counts = scales[scales.job_type_public == kind].groupby("bin").observations.sum().reindex(bins(size_edges), fill_value=0)
        row = {"job_type_public": kind, "execution_spans": count, "execution_span_share": count / jobs["member"]["rows"],
               "duration_mean_hours": float(group.duration_sum_hours.sum() / count),
               "duration_max_hours": float(group.duration_max_hours.max()),
               "gpu_request_mean": float((group.gpu_request_mean * group.execution_spans).sum() / count),
               "gpu_request_max": float(group.gpu_request_max.max()),
               "duration_ge_24h_span_share": float(duration_counts.iloc[duration_edges.index(24):].sum() / count),
               "duration_ge_168h_span_share": float(duration_counts.iloc[duration_edges.index(168):].sum() / count)}
        for q in [.5, .9, .99]:
            row[f"duration_p{int(q*100)}_empirical_bin_hours"] = duration_counts.index[(duration_counts.cumsum() >= count * q).argmax()]
            row[f"gpu_request_p{int(q*100)}_empirical_bin"] = size_counts.index[(size_counts.cumsum() >= count * q).argmax()]
        rows.append(row)
    return pd.DataFrame(rows)


def shape(values, inventory):
    hours = np.arange(len(values))
    frame = pd.DataFrame({"value": np.asarray(values), "inventory": np.asarray(inventory),
                          "day": hours // 24, "hour": hours % 24})
    daily = frame.groupby("day").value.sum()
    diurnal = frame.groupby("hour").value.mean()
    mod7 = daily.groupby(daily.index % 7).mean()
    daily_normalized = (frame.value / frame.inventory).groupby(frame.day).mean()
    return {"gpu_hours": float(frame.value.sum()), "hourly_mean_gpu_hours": float(frame.value.mean()),
            "hourly_min_gpu_hours": float(frame.value.min()), "hourly_max_gpu_hours": float(frame.value.max()),
            "hourly_peak_valley": ratio(frame.value), "hourly_zero_hours": int((frame.value == 0).sum()),
            "hourly_cv": divide(frame.value.std(ddof=0), frame.value.mean()),
            "daily_cv": divide(daily.std(ddof=0), daily.mean()), "daily_peak_valley": ratio(daily),
            "diurnal_peak_valley": ratio(diurnal), "diurnal_peak_hour": int(diurnal.idxmax()),
            "diurnal_valley_hour": int(diurnal.idxmin()), "relative_mod7_peak_valley": ratio(mod7),
            "first_14_days_daily_mean": float(daily.iloc[:14].mean()),
            "last_14_days_daily_mean": float(daily.iloc[-14:].mean()),
            "last14_first14_ratio": float(daily.iloc[-14:].mean() / daily.iloc[:14].mean()) if daily.iloc[:14].mean() > 0 else None,
            "inventory_normalized_daily_cv": divide(daily_normalized.std(ddof=0), daily_normalized.mean()),
            "lag24_correlation": correlation(frame.value, 24),
            "lag168_correlation": correlation(frame.value, 168),
            "daily_change_abs_p95_gpu_hours": float(daily.diff().abs().dropna().quantile(.95))}


def build(run_dir, output, baseline):
    output.mkdir(parents=True, exist_ok=True)
    state = json.loads((run_dir / "status.json").read_text("utf-8"))
    if state["state"] != "complete" or state["completed_hours"] != 4440:
        raise ValueError("全量处理尚未完成，不能生成 185 天报告")
    groups, totals, sizes, workload_sizes, models, states, member_audits, checkpoint_hashes = [], [], [], [], [], [], [], []
    successful_range_bytes, peak_rss, raw_rows = 0, 0, 0
    maximum_retry_counter = 0
    for index in range(4440):
        path = run_dir / "checkpoints" / f"hour-{index:05d}.json"
        item = json.loads(path.read_text("utf-8"))
        if item["index"] != index or (item["day"], item["hour"]) != divmod(index, 24):
            raise ValueError(f"检查点时钟错误: {path}")
        checkpoint_hashes.append({"checkpoint": path.name, "sha256": digest(path), "bytes": path.stat().st_size})
        frame = pd.DataFrame(item["groups"])
        if int(frame.pod_rows.sum()) != item["pod"]["rows"]:
            raise ValueError(f"pod 行数不守恒: {index}")
        clock = {"hour_index": index, "day": item["day"], "hour": item["hour"]}
        groups.extend({**clock, **row} for row in item["groups"])
        sizes.extend({**row} for row in item["pod_size_histogram"])
        workload_sizes.extend({**row} for row in item["workload_size_histogram"])
        models.extend(item["gpu_models"])
        states.extend(item["states"])
        inventory = sum(row["gpu_count_sum"] for row in item["inventory"])
        totals.append({**clock, **frame[VALUES].sum().to_dict(), "inventory_gpu": inventory,
                       "positive_request_used_gpu_hours": sum(row["used_gpu_hours_sum"] for row in item["gpu_models"]),
                       "distinct_pods": item["pod"]["distinct_pods"], "distinct_workloads": item["pod"]["distinct_workloads"]})
        member_audits.extend({**clock, **item[kind]} for kind in ("pod", "server"))
        successful_range_bytes += sum(item["range_bytes"].values())
        peak_rss = max(peak_rss, item["worker_peak_rss_kib"])
        maximum_retry_counter = max(maximum_retry_counter, item["retry_events_total"])
        raw_rows += item["pod"]["rows"]
    detailed = pd.DataFrame(groups)
    hourly = pd.DataFrame(totals)
    if hourly.inventory_gpu.min() <= 0 or detailed.used_gpu_hours.isna().any():
        raise ValueError("库存或归属 GPU 小时无效")
    hourly["attributed_share_of_inventory"] = hourly.used_gpu_hours / hourly.inventory_gpu
    hourly["requested_share_of_inventory"] = hourly.requested_gpu / hourly.inventory_gpu
    hourly["attributed_outside_positive_request_gpu_hours"] = hourly.used_gpu_hours - hourly.positive_request_used_gpu_hours
    detailed.to_parquet(output / "hourly-type-priority-model.parquet", index=False)
    hourly.rename(columns=COUNT_SUMS).to_csv(output / "hourly-total.csv", index=False)
    hourly[hourly.attributed_share_of_inventory > 1].rename(columns=COUNT_SUMS).to_csv(output / "inventory-attribution-anomalies.csv", index=False)
    type_hours = detailed.groupby(["hour_index", "day", "hour", "job_type_public"], dropna=False)[VALUES].sum().reset_index()
    type_hours.rename(columns=COUNT_SUMS).to_csv(output / "hourly-by-type.csv", index=False)
    by_group = detailed.groupby(KEYS, dropna=False)[VALUES].sum().reset_index()
    by_group["gpu_hour_share"] = by_group.used_gpu_hours / hourly.used_gpu_hours.sum()
    by_group.rename(columns=COUNT_SUMS).to_csv(output / "type-priority-model-total.csv", index=False)
    type_names = sorted(type_hours.job_type_public.unique())
    index = pd.MultiIndex.from_product([range(4440), type_names], names=["hour_index", "job_type_public"])
    supported = type_hours.set_index(["hour_index", "job_type_public"])[VALUES].reindex(index, fill_value=0).reset_index()
    supported["day"] = supported.hour_index // 24
    supported["hour"] = supported.hour_index % 24
    supported["relative_weekday"] = supported.day % 7
    supported["relative_week_hour"] = supported.relative_weekday * 24 + supported.hour
    daily = supported.groupby(["day", "job_type_public"], dropna=False)[VALUES].sum().reset_index()
    daily.rename(columns=COUNT_SUMS).to_csv(output / "daily-by-type.csv", index=False)
    for name, clock in [("diurnal-by-type.csv", "hour"), ("relative-week-by-type.csv", "relative_week_hour")]:
        supported.groupby(["job_type_public", clock], dropna=False)[VALUES].mean().reset_index().rename(columns=COUNT_SUMS).to_csv(output / name, index=False)
    daily["relative_weekday"] = daily.day % 7
    daily.groupby(["job_type_public", "relative_weekday"], dropna=False)[VALUES].mean().reset_index().rename(columns=COUNT_SUMS).to_csv(output / "mod7-by-type.csv", index=False)
    for filename, data, keys in [("pod-scale-histogram.csv", sizes, KEYS + ["bin"]),
                                  ("workload-hour-scale-histogram.csv", workload_sizes, KEYS + ["bin"]),
                                  ("gpu-model-total.csv", models, KEYS + ["gpu_spec_public"]),
                                  ("state-total.csv", states, KEYS + ["state_public"])]:
        pd.DataFrame(data).groupby(keys, observed=True, dropna=False).sum(numeric_only=True).reset_index().to_csv(output / filename, index=False)
    jobs_path = run_dir / "checkpoints/jobs.json"
    jobs = json.loads(jobs_path.read_text("utf-8"))
    checkpoint_hashes.append({"checkpoint": jobs_path.name, "sha256": digest(jobs_path), "bytes": jobs_path.stat().st_size})
    successful_range_bytes += sum(jobs["range_bytes"].values())
    peak_rss = max(peak_rss, jobs["worker_peak_rss_kib"])
    member_audits.append({"hour_index": None, "day": None, "hour": None, **jobs["member"]})
    for key, name in [("metrics", "execution-span-metrics.csv"), ("duration_histogram", "execution-duration-histogram.csv"),
                      ("size_histogram", "execution-scale-histogram.csv")]:
        pd.DataFrame(jobs[key]).to_csv(output / name, index=False)
    execution_type_summary(jobs).to_csv(output / "execution-by-type-summary.csv", index=False)
    job_counts = pd.DataFrame(jobs["metrics"]).groupby("job_type_public").execution_spans.sum()
    if int(pd.DataFrame(jobs["duration_histogram"]).observations.sum()) != jobs["member"]["rows"]:
        raise ValueError("执行跨度时长直方图不守恒")
    if int(pd.DataFrame(jobs["size_histogram"]).observations.sum()) != jobs["member"]["rows"]:
        raise ValueError("执行跨度规模直方图不守恒")
    if not np.isclose(detailed.used_gpu_hours.sum(), hourly.used_gpu_hours.sum()):
        raise ValueError("全量归属 GPU 小时分组不守恒")
    profiles = []
    for name, group in supported.groupby("job_type_public"):
        metrics = shape(group.sort_values("hour_index").used_gpu_hours, hourly.inventory_gpu)
        selected = detailed[detailed.job_type_public == name]
        metrics.update(job_type_public=name, gpu_hour_share=metrics["gpu_hours"] / hourly.used_gpu_hours.sum(),
                       execution_spans=int(job_counts.get(name, 0)),
                       priority_gpu_hours=selected.groupby("priority_class").used_gpu_hours.sum().to_dict(),
                       model_gpu_hours=selected.groupby("model_type_public").used_gpu_hours.sum().to_dict())
        profiles.append(metrics)
    profile_table = pd.DataFrame([{k: v for k, v in item.items() if not isinstance(v, dict)} for item in profiles])
    profile_table.to_csv(output / "task-type-summary.csv", index=False)
    inf = hourly[["hour_index", "day", "hour", "inventory_gpu"]].copy()
    for kind, prefix in [("online_inference", "online"), ("offline_inference", "offline")]:
        selected = supported[supported.job_type_public == kind].sort_values("hour_index")
        if len(selected) != 4440:
            raise ValueError("推理类型小时覆盖不完整")
        for column in ["used_gpu_hours", "requested_gpu", "requested_cpu_cores"]:
            inf[f"{prefix}_{column}"] = selected[column].to_numpy()
    inf["combined_used_gpu_hours"] = inf.online_used_gpu_hours + inf.offline_used_gpu_hours
    peak_reference = float(inf.combined_used_gpu_hours.max())
    for prefix in ["online", "offline", "combined"]:
        inf[f"{prefix}_shape_fraction"] = inf[f"{prefix}_used_gpu_hours"] / peak_reference
    inf.to_csv(output / "inf-hourly.csv", index=False)
    endpoint = pd.read_csv(baseline / "asi-hourly.csv")
    check = endpoint.merge(hourly, on=["day", "hour"], suffixes=("_baseline", "_new"))
    assert len(check) == 48
    np.testing.assert_allclose(check.used_gpu_hours_baseline, check.used_gpu_hours_new, rtol=1e-10)
    np.testing.assert_allclose(check.gpu_request, check.requested_gpu, rtol=1e-10)
    np.testing.assert_allclose(check.inventory_gpu_baseline, check.inventory_gpu_new)
    pd.DataFrame(member_audits).to_parquet(output / "member-audit.parquet", index=False)
    pd.DataFrame(checkpoint_hashes).to_csv(output / "checkpoint-manifest.csv", index=False)
    objects = json.loads((run_dir / "objects.json").read_text("utf-8"))
    run_records = [json.loads(line) for line in (run_dir / "runs.jsonl").read_text("utf-8").splitlines()]
    successful_range_bytes += len(run_records) * sum(item["central_directory_range_bytes"] for item in objects.values())
    full_start = min(item["run_started_at"] for item in run_records if item["pending_this_run"] > 1)
    events_path = run_dir / "restart-events.json"
    events = json.loads(events_path.read_text("utf-8")) if events_path.exists() else []
    audit = {"checked_at": datetime.now(timezone.utc).isoformat(), "state": state, "objects": objects,
             "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                             "pandas": pd.__version__, "pyarrow": pa.__version__, "requests": requests.__version__},
             "code_sha256": {name: digest(Path(__file__).with_name(name)) for name in ["stream_asi_185days.py", "probe_asi_inference_range.py",
                              "summarize_asi_185days.py", "annual_task_profiles.py", "asi_inference_profiles.py", "build_asi_inference_annual.py"]},
             "days": 185, "hours": 4440, "hourly_members": 8880, "job_members": 1,
             "pod_rows": raw_rows, "execution_span_rows": jobs["member"]["rows"],
             "successful_http_range_bytes": successful_range_bytes,
             "range_byte_scope": "committed checkpoints and planning; excludes independent validation and interrupted in-flight reads",
             "full_background_started_at": full_start,
             "full_background_elapsed_seconds": (datetime.fromisoformat(state["finished_at"]) - datetime.fromisoformat(full_start)).total_seconds(),
             "run_starts": run_records, "control_restarts": events,
             "member_payload_bytes": int(sum(item["bytes"] for item in member_audits)),
             "maximum_hourly_retry_counter": maximum_retry_counter,
             "failed_attempt_partial_bytes": "not measured; inspect run.log if a retry occurred",
             "max_worker_peak_rss_bytes": peak_rss * 1024, "raw_disk_bytes_written": 0,
             "resumed_hours": state["reused_hours"], "baseline_endpoint_hours_verified": 48,
             "group_gpu_hour_conservation": True, "duration_histogram_conservation": True,
             "attribution_above_inventory_hours": int((hourly.attributed_share_of_inventory > 1).sum()),
             "max_attributed_share_of_inventory": float(hourly.attributed_share_of_inventory.max()),
             "request_above_inventory_hours": int((hourly.requested_share_of_inventory > 1).sum()),
             "max_requested_share_of_inventory": float(hourly.requested_share_of_inventory.max()),
             "missing_gpu_hours_rows": int(detailed.missing_gpu_hours.sum()),
             "missing_gpu_request_rows": int(detailed.missing_gpu_request_rows.sum()),
             "missing_workload_rows": int(detailed.missing_workload_rows.sum()),
             "hour_pod_rows_minus_distinct_sum": int((hourly.pod_rows - hourly.distinct_pods).sum()),
             "attributed_outside_positive_request_gpu_hours": float(hourly.attributed_outside_positive_request_gpu_hours.sum()),
             "hours_with_material_attribution_outside_positive_requests": int(hourly.attributed_outside_positive_request_gpu_hours.abs().gt(1e-6).sum()),
             "checkpoint_total_bytes": sum(row["bytes"] for row in checkpoint_hashes)}
    result = {"profiles": profiles, "inf_combined": shape(inf.combined_used_gpu_hours, hourly.inventory_gpu),
              "inf_shape_reference_gpu_hours": peak_reference,
              "interpretation": "归属资源代理；异构 GPU 未换算吞吐；相对七日时钟；无功率或 SLA 权限"}
    save_json(output / "summary.json", result)
    save_json(output / "processing-audit.json", audit)
    manifest = {"input_file": "inf-hourly.csv", "input_sha256": digest(output / "inf-hourly.csv"),
                "source_hours": 4440, "source_days": 185, "peak_reference_gpu_hours": peak_reference,
                "normalization": "online/offline/combined 共用组合推理的观测最高小时归属 GPU 资源量，形状归一化，非物理利用率",
                "processing_audit_sha256": digest(output / "processing-audit.json")}
    save_json(output / "inf-source-manifest.json", manifest)
    print(json.dumps({"audit": audit, "profiles": profiles, "inf_combined": result["inf_combined"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="核验 ASI 全量检查点，生成任务类型、跨度、规模、时间形态与 INF 185 天曲线")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    build(args.run_dir, args.output, args.baseline)
