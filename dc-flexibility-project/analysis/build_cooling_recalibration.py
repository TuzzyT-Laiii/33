import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from compute_load import CoolingModel
from compute_load.cooling import COOLING_COMPONENTS, LEGACY_VERSION, PARAMETER_SETS, VERSION
from compute_load.model import normalize_weather, sha256
from global_grid_compute_load import grid_weather


RUN = "cooling-recalibration-20261028-v2"
GRID = "g05_r100_c592"


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def summary(frame, protocol):
    return {"protocol": protocol, "cooling_version": frame.cooling_version.iloc[0],
            "cooling_type": frame.cooling_type.iloc[0], "parameter_set": frame.parameter_set.iloc[0],
            "hours": len(frame), "it_mwh": float(frame.it_mw.sum()),
            "cooling_mwh": float(frame.cooling_mw.sum()), "other_mwh": float(frame.other_mw.sum()),
            "facility_mwh": float(frame.facility_mw.sum()),
            "annual_pue": float(frame.facility_mw.sum() / frame.it_mw.sum()),
            "annual_cooling_to_it_ratio": float(frame.cooling_mw.sum() / frame.it_mw.sum()),
            "annual_cooling_to_facility_ratio": float(frame.cooling_mw.sum() / frame.facility_mw.sum()),
            "air_full_free_hours": int(frame.air_free_cooling_fraction.ge(1 - 1e-12).sum()) if "air_free_cooling_fraction" in frame else None,
            "liquid_full_dry_hours": int(frame.liquid_dry_cooling_fraction.ge(1 - 1e-12).sum()) if "liquid_dry_cooling_fraction" in frame else None,
            "air_compressor_hours": int(frame.air_compressor_mw.gt(1e-10).sum()) if "air_compressor_mw" in frame else None,
            "liquid_compressor_hours": int(frame.liquid_compressor_mw.gt(1e-10).sum()) if "liquid_compressor_mw" in frame else None}


def response_curves(output):
    frames = []
    temperatures = np.linspace(-40, 50, 361)
    for humidity in (10, 50, 90):
        weather = pd.DataFrame({"time_utc": pd.date_range("2025-01-01", periods=len(temperatures), freq="h", tz="UTC"),
                                "temperature_c": temperatures, "relative_humidity_pct": humidity})
        for version, kind, case in [(LEGACY_VERSION, "air", "central")] + [
                (VERSION, kind, case) for kind in ("air", "direct_liquid") for case in PARAMETER_SETS]:
            result = CoolingModel(version=version, cooling_type=kind, parameter_set=case).calculate(weather, 7., 10., full_year=False)
            frames.append(result[["temperature_c", "relative_humidity_pct", "cooling_version", "cooling_type", "parameter_set", "pue", "cooling_to_it_ratio"]])
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(output / "response-curves.csv", index=False)
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 10, "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.1), sharey=True)
    colors = {10: "#0072B2", 50: "#009E73", 90: "#D55E00"}
    selections = [(LEGACY_VERSION, "air", "旧版 ESIF 迁移模型"), (VERSION, "air", "v2 风冷"),
                  (VERSION, "direct_liquid", "v2 GB200 直接液冷")]
    for index, (axis, (version, kind, title)) in enumerate(zip(axes, selections)):
        selected = data.loc[data.cooling_version.eq(version) & data.cooling_type.eq(kind)]
        for humidity, color in colors.items():
            part = selected.loc[selected.relative_humidity_pct.eq(humidity)]
            middle = part.loc[part.parameter_set.eq("central")]
            axis.plot(middle.temperature_c, middle.pue, color=color, label=f"RH {humidity}%", linewidth=1.8)
            if version == VERSION:
                low = part.loc[part.parameter_set.eq("low")]
                high = part.loc[part.parameter_set.eq("high")]
                axis.fill_between(low.temperature_c, low.pue, high.pue, color=color, alpha=0.10, linewidth=0)
        axis.set(xlabel="室外干球温度（°C）", title=title, xlim=(-40, 50))
        axis.text(0.02, 1.08, chr(97 + index), transform=axis.transAxes, weight="bold")
        axis.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("PUE（设施功率 / IT 功率）")
    axes[0].set_ylim(1, max(2., float(data.pue.max()) + .04))
    axes[0].legend(frameon=False, loc="upper left")
    fig.text(.5, .015, "IT = 7 MW，安装参考容量 = 10 MW；实线为中央情境，色带为高低参数范围（非置信区间）", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"pue-temperature-response.{suffix}", dpi=300)
    plt.close(fig)


def build(args):
    project = args.project.resolve()
    output = project / "results" / RUN
    output.mkdir(parents=True, exist_ok=True)
    weather_path = args.input_project / "data/processed/era5-landuse-trial/g05_r100_c592-hourly.parquet"
    mapping_path = args.input_project / "data/processed/era5-landuse-selection/grid-weather-mapping.parquet"
    prior = project / "results/typical-load-curves-20261028/manifest.json"
    manifest13 = json.loads(prior.read_text(encoding="utf-8"))
    if sha256(weather_path) != manifest13["weather_file_sha256"] or sha256(mapping_path) != manifest13["weather_mapping_sha256"]:
        raise ValueError("天气或映射与PR13记录不一致")
    mapping = pd.read_parquet(mapping_path)
    center = mapping.loc[mapping.grid_id.eq(GRID) & mapping.row_offset.eq(1) & mapping.col_offset.eq(1)]
    if len(center) != 1:
        raise ValueError("需要唯一中心天气节点")
    weather = grid_weather(ds.dataset(weather_path, format="parquet"), center.iloc[0], [2025])
    weather = normalize_weather(weather)
    legacy = CoolingModel(version=LEGACY_VERSION)
    bounds = legacy.legacy.cooling["calibration_range"]
    flags = pd.DataFrame({"dry_below": weather.temperature_c.lt(bounds["temperature_c"][0]),
                          "dry_above": weather.temperature_c.gt(bounds["temperature_c"][1]),
                          "dew_below": weather.dew_point_c.lt(bounds["dew_point_c"][0]),
                          "dew_above": weather.dew_point_c.gt(bounds["dew_point_c"][1])})
    flagged = flags.any(axis=1)
    diagnostics = {"grid_id": GRID, "year": 2025, "hours": len(weather), "native_flat_index": int(center.iloc[0].native_flat_index),
                   "weather_min_max": {column: [float(weather[column].min()), float(weather[column].max())]
                                       for column in ("temperature_c", "dew_point_c", "relative_humidity_pct")},
                   "legacy_bounds": bounds, "outside_hours": int(flagged.sum()),
                   "outside_by_condition": {column: int(flags[column].sum()) for column in flags},
                   "overlap_hours": int(flags.sum(axis=1).gt(1).sum()),
                   "outside_combinations": flags.value_counts().rename("hours").reset_index().to_dict("records")}
    save_json(output / "legacy-diagnosis.json", diagnostics)
    bins = pd.DataFrame({"dry_bin": pd.cut(weather.temperature_c, [-40, -20, 0, 10, 20, 30, 40, 50]),
                         "rh_bin": pd.cut(weather.relative_humidity_pct, [0, 20, 40, 60, 80, 100], include_lowest=True),
                         "outside": flagged})
    bins.groupby(["dry_bin", "rh_bin"], observed=False).agg(hours=("outside", "size"), outside_hours=("outside", "sum")).to_csv(output / "legacy-weather-bins.csv")
    annual, hourly, components = [], weather[["grid_id", "time_utc", "temperature_c", "relative_humidity_pct"]].copy(), []
    # 两种技术使用同一恒定热负荷，单独识别冷却结构对比。
    for version, kind, case in [(LEGACY_VERSION, "air", "central")] + [
            (VERSION, kind, case) for kind in ("air", "direct_liquid") for case in PARAMETER_SETS]:
        model = CoolingModel(version=version, cooling_type=kind, parameter_set=case)
        result = model.calculate(weather, 7., 10.)
        annual.append(summary(result, "matched_7MW_IT_10MW_capacity"))
        if case == "central":
            label = "legacy_air" if version == LEGACY_VERSION else kind
            for column in ("cooling_mw", "facility_mw", "pue", "cooling_to_it_ratio", "cooling_to_facility_ratio"):
                hourly[f"{label}_{column}"] = result[column].to_numpy()
            if version == VERSION:
                components.append({"protocol": "matched_7MW_IT_10MW_capacity", "cooling_type": kind,
                    **{column: float(result[column].mean()) for column in (*COOLING_COMPONENTS, "ups_distribution_loss_mw", "lighting_controls_mw")}})
    hourly.to_csv(output / "g05_r100_c592-2025-hourly.csv", index=False, float_format="%.9g")
    inputs = []
    # 仅读取两个已发布中央IT输入，典型日、周及全年曲线库保持原版。
    for name, kind in [("C20-Lisa-central.parquet", "air"), ("R-GB200-Acme-central.parquet", "direct_liquid")]:
        path = args.typical_project / "data/processed/typical-load-curves-20261028" / name
        if sha256(path) != manifest13["processed_sha256"][name]:
            raise ValueError(f"既有IT输入与PR13校验值不一致：{name}")
        prior_curve = pd.read_parquet(path)
        np.testing.assert_array_equal(pd.to_datetime(prior_curve.time_utc, utc=True), weather.time_utc)
        cap = float(prior_curve.installed_reference_it_mw.iloc[0])
        it = prior_curve.it_mw.to_numpy()
        for case in PARAMETER_SETS:
            result = CoolingModel(cooling_type=kind, parameter_set=case).calculate(weather, it, cap)
            annual.append(summary(result, name.replace("-central.parquet", "_frozen_PR13_IT")))
            if case == "central":
                selected = result[["time_utc", "it_mw", "cooling_mw", "other_mw", "facility_mw", "pue", "cooling_to_it_ratio", "cooling_to_facility_ratio"]]
                selected.to_csv(output / name.replace("-central.parquet", "-2025-hourly.csv"), index=False, float_format="%.9g")
        if kind == "air":
            result = legacy.calculate(weather, it, cap)
            for column in ("cooling_mw", "other_mw", "facility_mw", "pue"):
                np.testing.assert_allclose(result[column], prior_curve[column], atol=1e-12, rtol=1e-12)
            annual.append(summary(result, "C20-Lisa_frozen_PR13_IT"))
        inputs.append({"file": name, "sha256": sha256(path), "capacity_mw": cap})
    pd.DataFrame(annual).to_csv(output / "annual-comparison.csv", index=False)
    pd.DataFrame(components).to_csv(output / "central-power-components.csv", index=False)
    climate_path = args.input_project / "results/compute-center-data-audit-20261022/coupled-power-weather-20261025-v4/five-location-hourly-weather-2023-2025.parquet"
    climates = pd.read_parquet(climate_path)
    climate_rows = []
    for site, group in climates.loc[climates.calendar_year.eq(2025)].groupby("logical_site_id"):
        group = group.sort_values("time_utc").reset_index(drop=True)
        for kind in ("air", "direct_liquid"):
            for case in PARAMETER_SETS:
                result = CoolingModel(cooling_type=kind, parameter_set=case).calculate(group, 7., 10.)
                climate_rows.append({"logical_site_id": site, "example_label": str(group.example_label.iloc[0]),
                    "mean_dry_c": float(group.temperature_c.mean()), "mean_rh_pct": float(group.relative_humidity_pct.mean()),
                    **summary(result, "real_ERA5_climate_7MW_IT_10MW_capacity")})
    pd.DataFrame(climate_rows).to_csv(output / "climate-validation.csv", index=False)
    asset = CoolingModel().asset
    records = [{"parameter": key, **row, "source_url": asset["sources"][row["source"]]["url"]} for key, row in asset["parameters"].items()]
    pd.DataFrame(records).to_csv(output / "parameter-register.csv", index=False)
    response_curves(output)
    save_json(output / "manifest.json", {"base_commit": "8b7c6eb", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "weather_sha256": sha256(weather_path), "mapping_sha256": sha256(mapping_path), "legacy_PR13_manifest_sha256": sha256(prior),
        "frozen_IT_inputs": inputs, "climate_file_sha256": sha256(climate_path), "model": CoolingModel().metadata,
        "fixed_protocol": "8760 UTC hours of actual 2025 weather; constant IT=7MW, installed reference=10MW; controlled thermal experiment, not observed workload",
        "replay_scope": "only two frozen PR13 IT inputs as spot checks; no complete typical-load-curve rebuild",
        "software": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "code_sha256": {name: sha256(project / "analysis" / name) for name in
            ("build_cooling_recalibration.py", "replay_facility_cooling.py", "compute_load/cooling.py", "compute_load/model.py")},
        "files_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name not in ("manifest.json", "validation.json", "README.md")}})
    print(pd.DataFrame(annual)[["protocol", "cooling_type", "cooling_version", "parameter_set", "annual_pue"]].to_string(index=False))
    print(json.dumps(diagnostics, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-project", type=Path, required=True)
    parser.add_argument("--typical-project", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
