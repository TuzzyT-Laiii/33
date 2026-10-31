import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from annual_task_profiles import replay_hourly
from asi_inference_profiles import load_asi_sources
from compute_load.model import sha256


def verify(project, run_dir):
    directory = Path(project) / "results/asi-185day-workload-types-20261031"
    audit = json.loads((directory / "processing-audit.json").read_text("utf-8"))
    assert audit["hours"] == 4440 and audit["days"] == 185 and audit["state"]["state"] == "complete"
    hourly = pd.read_csv(directory / "hourly-total.csv", float_precision="round_trip")
    detailed = pd.read_parquet(directory / "hourly-type-priority-model.parquet")
    inf = pd.read_csv(directory / "inf-hourly.csv", float_precision="round_trip")
    members = pd.read_parquet(directory / "member-audit.parquet")
    np.testing.assert_array_equal(hourly.hour_index, np.arange(4440))
    np.testing.assert_array_equal(inf.hour_index, np.arange(4440))
    assert len(members) == 8881 and not members.member.duplicated().any()
    assert not detailed.duplicated(["hour_index", "job_type_public", "priority_class", "model_type_public"]).any()
    assert detailed.pod_count.le(detailed.pod_rows).all()
    assert detailed.workload_count.le(detailed.pod_rows).all()
    assert detailed.gpu_requesting_pods.le(detailed.pod_count).all()
    assert detailed.gpu_requesting_workloads.le(detailed.workload_count).all()
    groups = detailed.groupby("hour_index")[["used_gpu_hours", "requested_gpu", "pod_rows"]].sum()
    np.testing.assert_allclose(groups.used_gpu_hours, hourly.used_gpu_hours, rtol=1e-12)
    np.testing.assert_allclose(groups.requested_gpu, hourly.requested_gpu, rtol=1e-12)
    np.testing.assert_array_equal(groups.pod_rows, hourly.pod_rows)
    assert int(groups.pod_rows.sum()) == audit["pod_rows"]
    assert int(members.bytes.sum()) == audit["member_payload_bytes"]
    np.testing.assert_allclose(hourly.attributed_share_of_inventory, hourly.used_gpu_hours / hourly.inventory_gpu)
    assert int(hourly.attributed_share_of_inventory.gt(1).sum()) == audit["attribution_above_inventory_hours"]
    for name, prefix in [("online_inference", "online"), ("offline_inference", "offline")]:
        selected = detailed[detailed.job_type_public == name].groupby("hour_index").used_gpu_hours.sum().reindex(range(4440), fill_value=0)
        np.testing.assert_allclose(selected, inf[f"{prefix}_used_gpu_hours"], rtol=1e-12)
    np.testing.assert_allclose(inf.combined_used_gpu_hours, inf.online_used_gpu_hours + inf.offline_used_gpu_hours)
    sample = json.loads((Path(run_dir) / "independent-sample-validation.json").read_text("utf-8"))
    assert {item["hour_index"] for item in sample["samples"]} == {0, 4439}
    for item in sample["samples"]:
        expected = members[members.kind.eq("pod_hourly") & members.hour_index.eq(item["hour_index"])].iloc[0]
        assert expected.sha256 == item["sha256"]
    sources = load_asi_sources(project)
    replay_file = directory / "inf-annual-replay-2025-phase000.parquet"
    replay = pd.read_parquet(replay_file)
    np.testing.assert_array_equal(replay.source_hour_index, np.arange(8760) % 4440)
    np.testing.assert_array_equal(replay.replay_wrap, np.arange(8760) == 4440)
    for prefix in ["online", "offline", "combined"]:
        expected = inf[f"{prefix}_used_gpu_hours"].sum() + inf[f"{prefix}_used_gpu_hours"].iloc[:4320].sum()
        np.testing.assert_allclose(replay[f"{prefix}_used_gpu_hours"].sum(), expected, rtol=1e-12)
    times = pd.date_range("2024-01-01", "2025-01-01", freq="h", inclusive="left", tz="UTC")
    leap_cases = []
    for phase in [0, 184]:
        for key, source in sources.items():
            profile, mapping = replay_hourly(source, times, phase * 24)
            positions = (np.arange(8784) + phase * 24) % 4440
            np.testing.assert_array_equal(mapping.source_hour_index, positions)
            np.testing.assert_array_equal(profile.fraction, source["profile"].fraction[positions])
            leap_cases.append({"source": key, "phase_days": phase, "hours": len(profile.fraction)})
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "status": "passed",
              "coverage_days": 185, "coverage_hours": 4440, "members": len(members),
              "independent_sample_hours": [0, 4439], "baseline_endpoint_hours": audit["baseline_endpoint_hours_verified"],
              "resource_and_count_conservation": True, "type_priority_model_unique_keys": True,
              "anomalous_inventory_ratios_preserved": True, "inf_exact_category_filter": True,
              "annual_8760_integral": True, "actual_source_leap_replays": leap_cases,
              "annual_output_sha256": sha256(replay_file), "raw_disk_bytes_written": 0,
              "scope": "真实原始成员、真实完整观测聚合与资源年度重放；未运行气象、设备功率或优化"}
    (directory / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="核验真实 ASI 185 天汇总、独立样本、INF 分类与既有年度重放接口")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    verify(args.project, args.run_dir)
