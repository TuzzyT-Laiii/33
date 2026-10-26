import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from annual_task_profiles import load_sources, replay_hourly
from compute_load.model import ComputeLoadModel, sha256
from global_grid_compute_load import grid_weather


def verify(project):
    project = Path(project)
    result = project / "results/scenario-load-methods-20261026-v2"
    manifest = json.loads((result / "annualization-manifest.json").read_text(encoding="utf-8"))
    output = project / manifest["curves_directory"]
    summary = pd.read_csv(result / "annual-load-summary.csv")
    sensitivity = pd.read_csv(result / "annual-phase-sensitivity.csv")
    sources = load_sources(project)
    for name, expected in manifest["code_sha256"].items():
        assert sha256(Path(__file__).with_name(name)) == expected, name
    assert len(summary) == 16 and summary.hours.eq(8760).all()
    assert len(sensitivity) == 490
    for name, expected in manifest["curves_sha256"].items():
        assert sha256(output / name) == expected, name
    checks = []
    for key, source in sources.items():
        replay = pd.read_parquet(output / f"{key}-replay-map.parquet")
        n = len(source["profile"].fraction)
        expected_positions = np.arange(8760) % n
        np.testing.assert_array_equal(replay.source_hour_index, expected_positions)
        np.testing.assert_array_equal(replay.resource_fraction, source["profile"].fraction[expected_positions])
        np.testing.assert_array_equal(replay.replay_wrap, np.r_[False, np.diff(expected_positions)<0])
        expected_sum = (8760//n)*source["profile"].fraction.sum()+source["profile"].fraction[:8760%n].sum()
        np.testing.assert_allclose(replay.resource_fraction.sum(), expected_sum, rtol=1e-12)
        curves = {}
        for size in (1,10,100,500):
            path = output / f"{key}-{size}MW.parquet"
            frame = pd.read_parquet(path)
            assert len(frame) == 8760 and frame.time_utc.is_unique
            assert frame.time_utc.iloc[0] == pd.Timestamp("2025-01-01", tz="UTC")
            assert frame.time_utc.iloc[-1] == pd.Timestamp("2025-12-31 23:00", tz="UTC")
            assert frame.time_utc.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
            np.testing.assert_array_equal(frame.resource_fraction, replay.resource_fraction)
            np.testing.assert_allclose(frame.facility_mw, frame.it_mw+frame.cooling_mw+frame.other_mw, atol=1e-12)
            np.testing.assert_array_equal(frame.it_mw, frame.heat_demand_mw_th)
            row = summary.loc[summary.source.eq(key) & summary.reference_it_capacity_mw.eq(size)].iloc[0]
            for component in ("it", "cooling", "other", "facility"):
                np.testing.assert_allclose(frame[f"{component}_mw"].sum(), row[f"annual_{component}_mwh"], rtol=1e-12)
                np.testing.assert_allclose(frame[f"{component}_mw"].sum(), frame[f"cumulative_{component}_mwh"].iloc[-1], rtol=1e-11)
            assert frame[["it_mw","cooling_mw","other_mw","facility_mw"]].ge(0).all().all()
            if len(frame) > n:
                np.testing.assert_allclose(frame.it_mw.iloc[n:], frame.it_mw.iloc[:8760-n], rtol=1e-12)
            curves[size] = frame
            checks.append({"source":key, "size_mw":size, "hours":len(frame), "status":"passed"})
        ref = curves[100]
        for size, frame in curves.items():
            ratio = frame.installed_reference_it_mw.iloc[0]/ref.installed_reference_it_mw.iloc[0]
            for component in ("it_mw","cooling_mw","other_mw","facility_mw"):
                np.testing.assert_allclose(frame[component], ref[component]*ratio, rtol=1e-12, atol=1e-12)
        phases = sensitivity.loc[sensitivity.source.eq(key)]
        assert set(phases.phase_hours) == set(range(0,n,24))
        baseline = phases.loc[phases.phase_hours.eq(0)].iloc[0]
        np.testing.assert_allclose(baseline.annual_facility_mwh, ref.facility_mw.sum(), rtol=1e-12)
        if key == "acme_cpu":
            idle_hours = ref.resource_fraction.eq(0)
            assert idle_hours.any()
            np.testing.assert_allclose(ref.it_mw.loc[idle_hours], ref.node_count.iloc[0]*120/1e6)
    # 使用实际闰年天气检查 8784 小时日历与重放接口。
    mapping = pd.read_parquet(project / "data/processed/era5-landuse-selection/grid-weather-mapping.parquet")
    center = mapping.loc[mapping.grid_id.eq(manifest["grid_id"]) & mapping.row_offset.eq(1) & mapping.col_offset.eq(1)].iloc[0]
    weather = grid_weather(ds.dataset(project/f"data/processed/era5-landuse-trial/{manifest['grid_id']}-hourly.parquet", format="parquet"), center, [2024])
    model = ComputeLoadModel()
    for key in ("lisa_cpu", "acme_gpu"):
        profile, replay = replay_hourly(sources[key], weather.time_utc)
        curve = model.calculate(weather, profile, 100)
        assert len(curve) == 8784 and curve.time_utc.dt.strftime("%m-%d").eq("02-29").sum() == 24
        np.testing.assert_allclose(curve.facility_mw, curve.it_mw+curve.cooling_mw+curve.other_mw)
    # 相位取整与源时钟范围必须显式有效。
    rejected = []
    for offset in (-24, 1, 7416):
        try:
            replay_hourly(sources["lisa_cpu"], replay.time_utc, offset)
        except ValueError:
            rejected.append(offset)
    assert rejected == [-24,1,7416]
    record = {"status":"passed", "annual_cases":checks, "hourly_records":140160,
        "replay_maps":4, "phase_cases":490, "leap_year_actual_weather_cases":2,
        "invalid_phase_rejections":rejected, "checks":["source_hash", "exact_source_mapping", "resource_integral", "calendar", "power_balance", "heat_balance", "annual_integral", "capacity_rounding_scaling", "idle_power", "cycle_it_repetition", "phase_baseline"],
        "code_sha256":sha256(Path(__file__))}
    (result / "validation.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status":"passed", "curves":16, "hours":140160, "phase_cases":490, "leap_cases":2}))


if __name__ == "__main__":
    verify(Path(__file__).resolve().parents[1])
