import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from annual_task_profiles import complete_days, load_sources, replay_hourly
from compute_load.model import COMPONENTS, ComputeLoadModel, ResourceProfile, normalize_weather, sha256
from global_grid_compute_load import grid_weather
from typical_power_models import CASES, PARAMETERS, component_power, evidence_table, parameter_table, reference_power
from plot_typical_load_curves import plot_all


RUN = "typical-load-curves-20261028"
GRID = "g05_r100_c592"
YEAR = 2025


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def air_cooling(model, weather, it_mw, capacity_mw):
    q = np.asarray(it_mw) / capacity_mw
    knots = model.cooling["knots"]
    basis = [np.ones(len(weather)), q, q ** 2]
    for threshold in knots["dry_warm"]:
        basis.append(q * np.maximum(weather.temperature_c.to_numpy() - threshold, 0))
    for threshold in knots["dew_warm"]:
        basis.append(q * np.maximum(weather.dew_point_c.to_numpy() - threshold, 0))
    basis.append(np.maximum(knots["dry_cold"] - weather.temperature_c.to_numpy(), 0))
    values = np.column_stack(basis) @ model.coefficients * capacity_mw
    result = pd.DataFrame(values, columns=COMPONENTS)
    result["cooling_mw"] = result[list(COMPONENTS[:3])].sum(axis=1)
    result["facility_mw"] = np.asarray(it_mw) + result.cooling_mw + result.other_mw
    bounds = model.cooling["calibration_range"]
    result["outside_calibration_weather"] = ~(weather.temperature_c.between(*bounds["temperature_c"])
        & weather.dew_point_c.between(*bounds["dew_point_c"]))
    result["outside_calibration_it_range"] = ~pd.Series(q).between(*bounds["it_fraction"])
    return result


def add_alibaba(sources, path):
    frame = pd.read_csv(path)
    frame["source_time"] = pd.to_datetime(frame.source_time, utc=True)
    selected = complete_days(frame, "source_time", "cpu_fraction")
    if not selected.cpu_fraction.between(0, 1).all():
        raise ValueError("Alibaba CPU 利用率须在 0 至 1 之间")
    sources["alibaba_cpu"] = {"profile": ResourceProfile("cpu", selected.cpu_fraction.to_numpy(),
        {"source": "alibaba_cpu", "capacity_basis": "有效机器小时 CPU 使用率按机器 CPU 容量加权；不按样本峰值缩放"}),
        "clock": selected.source_time.astype(str).to_numpy(),
        "metadata": {"source": "alibaba_cpu", "hardware": "cpu", "days": len(selected) // 24,
            "source_start": str(selected.source_time.iloc[0]), "source_end_inclusive": str(selected.source_time.iloc[-1]),
            "normalization_basis": "CPU 使用率百分数 / 100；按机器容量加权",
            "status": "short_window_utilization_replay_proxy", "input_file": path.name,
            "input_sha256": sha256(path), "source_clock": "从相对时间零点映射到2000-01-01 UTC；原始日期未公开"}}


def resource_pair(config, demand, replay, case):
    if demand in ("Lisa", "Alibaba"):
        return replay[{"Lisa": "lisa_cpu", "Alibaba": "alibaba_cpu"}[demand]].copy(), np.zeros(8760)
    gpu = replay[{"Acme": "acme_gpu", "ASI": "asi_gpu"}[demand]].copy()
    host = {"low": .10, "central": .25, "high": .40}[case] * gpu
    # Acme 的 CPU 作业与 GPU 作业宿主活动分别保留；宿主活动比例属于假设。
    cpu = np.maximum(replay["acme_cpu"], host) if demand == "Acme" else host
    return cpu, gpu


def summarize(frame, config, demand, case, source):
    air = config["cooling_type"] == "air"
    reported = frame.facility_mw if air else frame.it_mw
    it_energy = float(frame.it_mw.sum())
    facility_energy = float(reported.sum()) if air else None
    return {"curve_id": f'{config["configuration_id"]}-{demand}', "configuration_id": config["configuration_id"],
        "demand": demand, "case": case, "hours": len(frame), "source_days": source["metadata"]["days"],
        "unit": config["research_unit"], "units": int(frame.unit_count.iloc[0]),
        "sizing_reference_w": reference_power(config), "installed_reference_it_mw": float(frame.installed_reference_it_mw.iloc[0]),
        "annual_it_mwh": it_energy, "annual_cooling_mwh": float(frame.cooling_mw.sum()) if air else None,
        "annual_other_mwh": float(frame.other_mw.sum()) if air else None, "annual_facility_mwh": facility_energy,
        "reported_scope": "facility" if air else "IT_only_liquid_cooling_pending",
        "reported_annual_mwh": float(reported.sum()), "peak_mw": float(reported.max()),
        "minimum_mw": float(reported.min()), "average_mw": float(reported.mean()),
        "average_load_factor": float(reported.mean() / reported.max()),
        "mean_it_over_reference": float(frame.it_mw.mean() / frame.installed_reference_it_mw.iloc[0]),
        "peak_valley_ratio": float(reported.max() / reported.min()),
        "annual_pue": facility_energy / it_energy if air else None,
        "cooling_share": float(frame.cooling_mw.sum() / facility_energy) if air else None,
        "outside_calibration_weather_hours": int(frame.outside_calibration_weather.sum()) if air else None,
        "outside_calibration_it_hours": int(frame.outside_calibration_it_range.sum()) if air else None,
        "above_10mw_it_hours": int(frame.it_mw.gt(10).sum()),
        "data_trust_flag": ("单日重复演示" if demand == "ASI" else "实测迹的资源形状重放") + ";假设功率;"
            + ("风冷跨设施外推" if air else "液冷待补;仅IT"),
        "resource_basis": "CPU使用率" if demand == "Alibaba" else "分配/归属资源占用代理，非芯片实测利用率"}


def select_week(weather, replay):
    columns = [replay[key] for key in ("lisa_cpu", "acme_gpu", "asi_gpu")]
    columns.append(weather.temperature_c.to_numpy())
    features = np.column_stack(columns)
    starts = [h for h, t in enumerate(weather.time_utc) if t.dayofweek == 0 and t.hour == 0 and h + 168 <= 8760]
    weeks = np.stack([features[start:start + 168] for start in starts])
    scale = np.maximum(features.std(axis=0), 1e-8)
    score = (((weeks - weeks.mean(axis=0)) / scale) ** 2).mean(axis=(1, 2))
    selected = int(np.argmin(score))
    return starts[selected], pd.DataFrame({"start_utc": [str(weather.time_utc.iloc[h]) for h in starts],
                                         "hour_index": starts, "standardized_mse": score})


def build(args):
    project = args.project.resolve()
    inputs = args.input_project.resolve() if args.input_project else project
    output = project / "results" / RUN
    processed = project / "data/processed" / RUN
    output.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    config_path = project / "results/server-count-scenarios-20261028/server-configurations.json"
    configs = json.loads(config_path.read_text(encoding="utf-8"))
    if {c["configuration_id"] for c in configs} != set(PARAMETERS):
        raise ValueError("设备配置须与五种功率模型逐一匹配")
    mapping_path = inputs / "data/processed/era5-landuse-selection/grid-weather-mapping.parquet"
    weather_path = inputs / f"data/processed/era5-landuse-trial/{GRID}-hourly.parquet"
    mapping = pd.read_parquet(mapping_path)
    center = mapping.loc[mapping.grid_id.eq(GRID) & mapping.row_offset.eq(1) & mapping.col_offset.eq(1)]
    if len(center) != 1:
        raise ValueError("天气映射必须有唯一中心节点")
    old_manifest = json.loads((inputs / "results/scenario-load-methods-20261026-v2/annualization-manifest.json").read_text(encoding="utf-8"))
    for path, key in ((mapping_path, "weather_mapping_sha256"), (weather_path, "weather_file_sha256")):
        if sha256(path) != old_manifest[key]:
            raise ValueError(f"天气输入与已核验全年流程不一致：{path.name}")
    weather = normalize_weather(grid_weather(ds.dataset(weather_path, format="parquet"), center.iloc[0], [YEAR]))
    sources = load_sources(inputs)
    if args.alibaba_hourly:
        add_alibaba(sources, args.alibaba_hourly)
    replay, replay_metadata, data_hashes = {}, {}, {}
    for key, source in sources.items():
        profile, provenance = replay_hourly(source, weather.time_utc)
        replay[key] = profile.fraction
        replay_metadata[key] = profile.provenance
        path = processed / f"{key}-replay-map.parquet"
        provenance.to_parquet(path, index=False)
        data_hashes[path.name] = sha256(path)
    model = ComputeLoadModel()
    frames, summaries = {}, []
    checks = {}
    for config in configs:
        key = config["configuration_id"]
        demands = (["Lisa"] + (["Alibaba"] if "alibaba_cpu" in sources else [])) if key == "C20" else (
            ["Acme", "ASI"] if key == "R-GB200" else ["Lisa", "Acme", "ASI"])
        ref = reference_power(config)
        units = int(10e6 // ref)
        capacity = units * ref / 1e6
        for demand in demands:
            case_frames = {}
            source_key = {"Lisa": "lisa_cpu", "Acme": "acme_gpu", "ASI": "asi_gpu", "Alibaba": "alibaba_cpu"}[demand]
            for case in CASES:
                cpu, gpu = resource_pair(config, demand, replay, case)
                power = component_power(config, cpu, gpu, case)
                frame = weather[["time_utc", "temperature_c", "dew_point_c"]].copy()
                frame["u_cpu_proxy"], frame["u_gpu_proxy"] = cpu, gpu
                for component in ("base", "cpu", "gpu"):
                    frame[f"{component}_it_mw"] = power[f"{component}_w"] * units / 1e6
                frame["it_mw"] = power["unit_w"] * units / 1e6
                frame["unit_count"], frame["installed_reference_it_mw"] = units, capacity
                frame["configuration_id"], frame["demand"], frame["case"] = key, demand, case
                if config["cooling_type"] == "air":
                    frame = pd.concat([frame, air_cooling(model, weather, frame.it_mw, capacity)], axis=1)
                    frame["pue"] = frame.facility_mw / frame.it_mw
                else:
                    for column in (*COMPONENTS, "cooling_mw", "facility_mw", "pue"):
                        frame[column] = np.nan
                    frame["outside_calibration_weather"] = pd.array([pd.NA] * len(frame), dtype="boolean")
                    frame["outside_calibration_it_range"] = pd.array([pd.NA] * len(frame), dtype="boolean")
                path = processed / f"{key}-{demand}-{case}.parquet"
                frame.to_parquet(path, index=False)
                data_hashes[path.name] = sha256(path)
                summaries.append(summarize(frame, config, demand, case, sources[source_key]))
                case_frames[case] = frame
                if case == "central":
                    frames[(key, demand)] = frame
            low, central, high = [case_frames[c].it_mw.to_numpy() for c in CASES]
            checks[f"{key}-{demand}-ordered_sensitivity"] = bool(np.all(low <= central + 1e-12) and np.all(central <= high + 1e-12))
    summary = pd.DataFrame(summaries)
    central = summary.loc[summary.case.eq("central")].reset_index(drop=True)
    central.to_csv(output / "annual-summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "sensitivity-summary.csv", index=False, encoding="utf-8-sig")
    parameter_table(configs).to_csv(output / "power-parameters.csv", index=False, encoding="utf-8-sig")
    evidence_table().to_csv(output / "specpower-evidence.csv", index=False, encoding="utf-8-sig")
    week_start, week_scores = select_week(weather, replay)
    week_scores.to_csv(output / "representative-week-selection.csv", index=False, encoding="utf-8-sig")
    plot_result = plot_all(frames, output, week_start)
    for key, frame in frames.items():
        name = "-".join(key)
        checks[f"{name}-hours"] = bool(len(frame) == 8760 and frame.time_utc.is_unique)
        checks[f"{name}-finite_it"] = bool(np.isfinite(frame.it_mw).all() and frame.it_mw.gt(0).all())
        checks[f"{name}-components"] = bool(np.allclose(frame.it_mw, frame[["base_it_mw", "cpu_it_mw", "gpu_it_mw"]].sum(axis=1), atol=1e-12))
        if key[0] != "R-GB200":
            checks[f"{name}-facility_balance"] = bool(np.allclose(frame.facility_mw, frame.it_mw + frame.cooling_mw + frame.other_mw, atol=1e-12))
        else:
            checks[f"{name}-liquid_not_inferred"] = bool(frame[["facility_mw", "cooling_mw", "pue"]].isna().all().all())
        checks[f"{name}-10mw_sizing"] = bool(frame.installed_reference_it_mw.iloc[0] <= 10 and frame.it_mw.max() <= 10 + 1e-12)
    old_curve = model.calculate(weather, replay_hourly(sources["lisa_cpu"], weather.time_utc)[0], 10)
    parity = air_cooling(model, weather, old_curve.it_mw, float(old_curve.installed_reference_it_mw.iloc[0]))
    checks["existing_air_cooling_parity"] = bool(np.allclose(parity[list(COMPONENTS)], old_curve[list(COMPONENTS)], atol=1e-12))
    checks["summary_energy_reconciles"] = all(np.isclose(row.annual_it_mwh, frames[(row.configuration_id, row.demand)].it_mw.sum())
                                                  for row in central.itertuples())
    checks["ASI_24h_exact_repetition"] = bool(np.array_equal(replay["asi_gpu"], np.tile(replay["asi_gpu"][:24], 365)))
    checks.update(plot_result["checks"])
    if not all(checks.values()):
        raise AssertionError([key for key, passed in checks.items() if not passed])
    stamp = datetime.now(timezone.utc).isoformat()
    validation = {"checked_at_utc": stamp, "status": "numerical_checks_passed; visual_review_required",
        "checks": checks, "baseline_curves": len(central), "sensitivity_curves_including_baseline": len(summary),
        "hourly_rows": int(len(summary) * 8760), "weather": {"grid_id": GRID, "year": YEAR,
            "native_flat_index": int(center.iloc[0].native_flat_index), "hours": len(weather),
            "temperature_range_c": [float(weather.temperature_c.min()), float(weather.temperature_c.max())]},
        "figure_checks": plot_result, "scientific_scope": "可重现演示曲线；功率参数与资源到利用率转换含假设，未验证目标设备实际年电耗"}
    save_json(output / "validation.json", validation)
    manifest = {"created_at_utc": stamp, "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip(),
        "dependency_pr": "https://github.com/TuzzyT-Laiii/33/pull/12", "configuration_file": str(config_path.relative_to(project)),
        "configuration_sha256": sha256(config_path), "input_project": str(inputs),
        "weather_file_sha256": sha256(weather_path), "weather_mapping_sha256": sha256(mapping_path),
        "weather_grid": GRID, "year": YEAR, "target_time_basis": "UTC",
        "sources": replay_metadata, "processed_directory": str(processed.relative_to(project)), "processed_sha256": data_hashes,
        "representative_week_start": str(weather.time_utc.iloc[week_start]),
        "typical_day": "目标2025年周一至周五，按UTC小时求平均；不代表原始设施真实工作日",
        "typical_week": "完整周一至周日候选周；Lisa/AcmeGPU/ASI/干球温度标准化后距168小时平均模板最近的共同周",
        "power_model": "P_unit=P_base+n_cpu*P_cpu(u_cpu)+n_gpu*P_gpu(u_gpu); public_sources_plus_assumptions",
        "sensitivity": "固定中央参考台数，联合改变端点、CPU/GPU响应形状及宿主CPU映射；区间不表示置信区间",
        "air_cooling_model": model.metadata, "liquid_cooling_model": None,
        "alibaba_status": "included" if "alibaba_cpu" in sources else "pending_server_only_download_and_validation",
        "code_sha256": {name: sha256(project / "analysis" / name) for name in (
            "typical_power_models.py", "build_typical_load_curves.py", "plot_typical_load_curves.py", "annual_task_profiles.py", "compute_load/model.py")}}
    save_json(output / "manifest.json", manifest)
    print(json.dumps({"baseline_curves": len(central), "hourly_rows": len(summary) * 8760,
                      "week_start": manifest["representative_week_start"], "checks": len(checks)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="五种设备配置的10MW典型负荷曲线与功率敏感性")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-project", type=Path)
    parser.add_argument("--alibaba-hourly", type=Path)
    build(parser.parse_args())
