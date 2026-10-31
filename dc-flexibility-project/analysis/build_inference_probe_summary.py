import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build(output):
    audit = json.loads((output / "asi-range-probe.json").read_text("utf-8"))
    frame = pd.DataFrame(audit["hourly"])
    pod, server = frame[frame.kind == "pod"].copy(), frame[frame.kind == "server"].copy()
    keys = ["day", "hour"]
    totals = pod.groupby(keys)[["used_gpu_hours", "gpu_request", "cpu_request_cores", "rows"]].sum()
    inventory = server.set_index(keys).inventory_gpu
    totals = totals.join(inventory)
    totals["attributed_share_of_inventory"] = totals.used_gpu_hours / totals.inventory_gpu
    totals["requested_share_of_inventory"] = totals.gpu_request / totals.inventory_gpu
    if len(totals) != len(audit["days"]) * len(audit["hours"]) or totals.isna().any().any():
        raise ValueError("小时数量或库存关联不完整")
    samples = pd.DataFrame(audit["samples"])
    if (totals.rows > samples[samples.kind == "pod"].set_index(keys).rows).any():
        raise ValueError("筛选后记录数超过原始行数")
    type_priority = pod.groupby(keys + ["job_type_public", "priority_class"], dropna=False)[
        ["used_gpu_hours", "gpu_request", "cpu_request_cores", "rows"]].sum().reset_index()
    if not np.isclose(type_priority.used_gpu_hours.sum(), totals.used_gpu_hours.sum()):
        raise ValueError("类型×优先级汇总不守恒")
    pod.drop(columns=["inventory_gpu"]).to_csv(output / "asi-hourly-by-type-priority-gpu-state.csv", index=False)
    type_priority.to_csv(output / "asi-hourly-by-type-priority.csv", index=False)
    totals.to_csv(output / "asi-hourly.csv")
    inference = type_priority[type_priority.job_type_public.isin(["online_inference", "offline_inference"])]
    inference_hourly = inference.pivot_table(index=keys, columns="job_type_public", values="used_gpu_hours",
                                             aggfunc="sum", fill_value=0)
    inference_hourly["inference_gpu_hours"] = inference_hourly.online_inference + inference_hourly.offline_inference
    if len(inference_hourly) != len(totals):
        raise ValueError("推理类别小时覆盖不完整")
    inference_hourly.to_csv(output / "asi-inference-hourly.csv")
    detail = []
    for day, group in totals.groupby(level="day"):
        by_type = pod[pod.day == day].groupby("job_type_public").used_gpu_hours.sum()
        detail.append({"day": day, "hours": len(group), "gpu_hours": float(group.used_gpu_hours.sum()),
            "inventory_min": float(group.inventory_gpu.min()), "inventory_max": float(group.inventory_gpu.max()),
            "hourly_peak_valley": float(group.used_gpu_hours.max() / group.used_gpu_hours.min()),
            "inference_gpu_hours": float(inference_hourly.loc[day].inference_gpu_hours.sum()),
            "inference_hourly_peak_valley": float(inference_hourly.loc[day].inference_gpu_hours.max()
                                                  / inference_hourly.loc[day].inference_gpu_hours.min()),
            "peak_hour": int(group.used_gpu_hours.idxmax()[1]), "valley_hour": int(group.used_gpu_hours.idxmin()[1]),
            "type_gpu_hours": by_type.to_dict(), "type_shares": (by_type / by_type.sum()).to_dict(),
            "attributed_share_min": float(group.attributed_share_of_inventory.min()),
            "attributed_share_max": float(group.attributed_share_of_inventory.max()),
            "requested_share_min": float(group.requested_share_of_inventory.min()),
            "requested_share_max": float(group.requested_share_of_inventory.max())})
    transfer_bytes = sum(a["range_bytes"] for a in audit["archives"])
    all_bytes = sum(a["archive_bytes"] for a in audit["archives"])
    sample_seconds = sum(s["seconds"] for s in audit["samples"])
    rows = int(samples[samples.kind == "pod"].rows.sum())
    performance = {"range_bytes": transfer_bytes, "sample_pod_rows": rows,
        "full_pod_and_server_bytes": all_bytes,
        "network_hours_at_48_decimal_MB_per_s": all_bytes / 48e6 / 3600,
        "serial_hours_at_sample_member_rate": sample_seconds * (4440 / 48) / 3600,
        "serial_hours_at_sample_byte_rate": audit["elapsed_seconds"] * all_bytes / transfer_bytes / 3600,
        "sample_elapsed_seconds": audit["elapsed_seconds"], "peak_rss_bytes": audit["peak_rss_kib"] * 1024,
        "largest_parquet_bytes": max(a["largest_member_bytes"] for a in audit["archives"]),
        "full_type_priority_csv_estimated_bytes": (output / "asi-hourly-by-type-priority.csv").stat().st_size * 4440 / 48,
        "full_detailed_csv_estimated_bytes": (output / "asi-hourly-by-type-priority-gpu-state.csv").stat().st_size * 4440 / 48,
        "raw_disk_bytes_written": audit["raw_disk_bytes_written"],
        "estimate_basis": "two endpoint days; planning estimate, not measured full-run completion time"}
    result = {"days": detail, "performance": performance}
    (output / "asi-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="核验 ASI 抽样汇总守恒并估算全量成本")
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
