import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from annual_task_profiles import load_sources, replay_hourly
from compute_load.model import ComputeLoadModel, SIZES_MW, annual_summary, sha256
from global_grid_compute_load import grid_weather


def build(project, year=2025, grid_id="g05_r100_c592", weather_path=None, phase_sensitivity=True):
    project = Path(project).resolve()
    output = project / "data/processed/task-annual-loads-20261026-v2" / f"{grid_id}-{year}"
    summary_dir = project / "results/scenario-load-methods-20261026-v2"
    if (grid_id, year) != ("g05_r100_c592", 2025):
        summary_dir = summary_dir / "annual-runs" / f"{grid_id}-{year}"
    output.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = project / "data/processed/era5-landuse-selection/grid-weather-mapping.parquet"
    weather_path = Path(weather_path) if weather_path else project / f"data/processed/era5-landuse-trial/{grid_id}-hourly.parquet"
    mapping = pd.read_parquet(mapping_path)
    center = mapping.loc[mapping.grid_id.eq(grid_id) & mapping.row_offset.eq(1) & mapping.col_offset.eq(1)]
    if len(center) != 1:
        raise ValueError("所选网格必须登记唯一中心气象节点")
    weather = grid_weather(ds.dataset(weather_path, format="parquet"), center.iloc[0], [year])
    model, sources = ComputeLoadModel(), load_sources(project)
    annual, sensitivity, source_manifest = [], [], {}
    files = {}
    for key, source in sources.items():
        profile, replay = replay_hourly(source, weather.time_utc)
        replay_path = output / f"{key}-replay-map.parquet"
        replay.to_parquet(replay_path, index=False)
        files[replay_path.name] = sha256(replay_path)
        source_manifest[key] = profile.provenance
        for size in SIZES_MW:
            curve = model.calculate(weather, profile, size)
            curve["demand_source"] = key
            curve["annualization_status"] = source["metadata"]["status"]
            path = output / f"{key}-{size}MW.parquet"
            curve.to_parquet(path, index=False)
            files[path.name] = sha256(path)
            annual.append({"source": key, "status": source["metadata"]["status"], **annual_summary(curve)})
        if phase_sensitivity:
            for offset in range(0, len(source["profile"].fraction), 24):
                alternative, _ = replay_hourly(source, weather.time_utc, offset)
                curve = model.calculate(weather, alternative, 100)
                sensitivity.append({"source": key, "phase_hours": offset,
                    "reference_it_capacity_mw": 100, "annual_it_mwh": float(curve.it_mw.sum()),
                    "annual_cooling_mwh": float(curve.cooling_mw.sum()),
                    "annual_facility_mwh": float(curve.facility_mw.sum()),
                    "peak_facility_mw": float(curve.facility_mw.max())})
        print(json.dumps({"source": key, "source_days": source["metadata"]["days"], "annual_cases": 4}, ensure_ascii=False), flush=True)
    pd.DataFrame(annual).to_csv(summary_dir / "annual-load-summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sensitivity).to_csv(summary_dir / "annual-phase-sensitivity.csv", index=False, encoding="utf-8-sig")
    manifest = {"grid_id": grid_id, "year": year, "hours_per_curve": len(weather), "curves": len(annual),
        "source_profiles": source_manifest, "weather_file_sha256": sha256(weather_path),
        "weather_mapping_sha256": sha256(mapping_path), "native_flat_index": int(center.iloc[0].native_flat_index),
        "weather_location": "研究网格的中心 0.25 度 ERA5 节点；逐小时 UTC 干球与露点",
        "curves_directory": output.relative_to(project).as_posix(), "curves_sha256": files,
        "code_sha256": {name: sha256(Path(__file__).with_name(name)) for name in ("annual_task_profiles.py", "build_task_annual_loads.py")},
        "power_model": model.metadata,
        "interpretation": "跨数据集资源到整机功率代理与参考冷却的全年重放；不表示原设施全年实测、不证明任务可柔性。ASI 单日仅演示。"}
    (summary_dir / "annualization-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="将真实任务需求连续窗口重放到完整 UTC 年，计算四档 IT 规模及气象耦合负荷")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--grid-id", default="g05_r100_c592")
    parser.add_argument("--weather-path", type=Path)
    parser.add_argument("--skip-phase-sensitivity", action="store_true")
    args = parser.parse_args()
    result = build(args.project, args.year, args.grid_id, args.weather_path, not args.skip_phase_sensitivity)
    print(json.dumps({"curves": result["curves"], "hours": result["hours_per_curve"]}))


if __name__ == "__main__":
    main()
