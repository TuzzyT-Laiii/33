import argparse
import calendar
import json
from pathlib import Path

import numpy as np
import pandas as pd

from annual_task_profiles import replay_hourly
from asi_inference_profiles import load_asi_sources
from compute_load.model import sha256


def build(project, year, phase_days):
    project = Path(project)
    directory = project / "results/asi-185day-workload-types-20261031"
    sources = load_asi_sources(project)
    times = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", freq="h", inclusive="left", tz="UTC")
    combined, mapping = replay_hourly(sources["asi_gpu"], times, phase_days * 24)
    output = mapping.rename(columns={"resource_fraction": "combined_shape_fraction"})
    raw = pd.read_csv(directory / "inf-hourly.csv", float_precision="round_trip")
    position = output.source_hour_index.to_numpy()
    for key, prefix in [("asi_online_gpu", "online"), ("asi_offline_gpu", "offline")]:
        profile, own_mapping = replay_hourly(sources[key], times, phase_days * 24)
        np.testing.assert_array_equal(own_mapping.source_hour_index, position)
        output[f"{prefix}_shape_fraction"] = profile.fraction
    for prefix in ["online", "offline", "combined"]:
        output[f"{prefix}_used_gpu_hours"] = raw[f"{prefix}_used_gpu_hours"].to_numpy()[position]
    output["source_day"] = position // 24
    output["source_hour"] = position % 24
    np.testing.assert_allclose(output.combined_shape_fraction, output.online_shape_fraction + output.offline_shape_fraction, atol=1e-14)
    expected_hours = 8784 if calendar.isleap(year) else 8760
    assert len(output) == expected_hours and len(raw) == 4440
    assert np.allclose(combined.fraction, raw.combined_shape_fraction.to_numpy()[position])
    path = directory / f"inf-annual-replay-{year}-phase{phase_days:03d}.parquet"
    output.to_parquet(path, index=False)
    record = {"year": year, "phase_days": phase_days, "hours": len(output), "source_hours": 4440,
              "complete_cycles": len(output) // 4440, "remaining_hours": len(output) % 4440,
              "wrap_count": int(output.replay_wrap.sum()), "source_file_sha256": sha256(directory / "inf-hourly.csv"),
              "output_file": path.name, "output_sha256": sha256(path),
              "resource_integrals": {prefix: float(output[f"{prefix}_used_gpu_hours"].sum()) for prefix in ["online", "offline", "combined"]},
              "quality_input_sha256": sources["asi_gpu"]["metadata"]["input_sha256"],
              "unresolved_target_hours": int(output.source_quality_flag.ne("observed").sum()),
              "provenance": combined.provenance,
              "scope": "资源需求年度重放；不运行气象、设备功率或柔性优化"}
    (directory / f"inf-annual-replay-{year}-phase{phase_days:03d}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 INF 在线/离线 185 天需求接入既有年度重放函数，输出资源需求及来源映射")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--phase-days", type=int, default=0)
    args = parser.parse_args()
    if args.phase_days not in range(185):
        raise ValueError("相位日须为 0..184")
    build(args.project, args.year, args.phase_days)
