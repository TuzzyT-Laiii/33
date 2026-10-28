import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from compute_load import ComputeLoadModel, CoolingModel, ProfileLibrary
from compute_load.cooling import COOLING_COMPONENTS, LEGACY_VERSION, PARAMETER_SETS, VERSION
from compute_load.model import sha256
from global_grid_compute_load import grid_weather
from replay_facility_cooling import filesystem_path, replay


def verify(args):
    project = Path(__file__).resolve().parents[1]
    output = project / "results/cooling-recalibration-20261028-v2"
    temporary = project / ".codex-temp/cooling-validation"
    temporary.mkdir(parents=True, exist_ok=True)
    test = unittest.TestCase()
    checked, maximum_balance = 0, 0.
    # 温湿度网格是明确的边界条件实验，不表示气象观测或站点实测。
    temperatures = np.linspace(-40, 50, 361)
    humidity = np.arange(0., 101., 5)
    t, rh = np.repeat(temperatures, len(humidity)), np.tile(humidity, len(temperatures))
    sweep = pd.DataFrame({"time_utc": pd.date_range("2025-01-01", periods=len(t), freq="h", tz="UTC"),
                          "temperature_c": t, "relative_humidity_pct": rh})
    extremes, matrices = [], {}
    for fraction in (.05, .25, .7, 1., 1.1):
        for kind in ("air", "direct_liquid"):
            by_case = []
            for case in PARAMETER_SETS:
                result = CoolingModel(cooling_type=kind, parameter_set=case).calculate(sweep, fraction * 10, 10, full_year=False)
                checked += len(result)
                powers = result[[*COOLING_COMPONENTS, "other_mw", "facility_mw"]].to_numpy()
                test.assertTrue(np.isfinite(powers).all())
                test.assertTrue((powers >= 0).all())
                test.assertTrue(result.pue.ge(1).all())
                maximum_balance = max(maximum_balance, float(result.heat_balance_residual_mw.abs().max()))
                np.testing.assert_allclose(result.heat_rejected_mw_th, result.facility_mw, atol=1e-10, rtol=1e-12)
                matrix = result.pue.to_numpy().reshape(len(temperatures), len(humidity))
                # -15°C以下允许伴热导致的轻微冷端上升；以上温度须单调不减。
                test.assertGreaterEqual(float(np.diff(matrix[temperatures >= -15], axis=0).min()), -1e-10)
                test.assertGreaterEqual(float(np.diff(matrix, axis=1).min()), -1e-10)
                by_case.append(result.pue.to_numpy())
                matrices[(kind, case, fraction)] = matrix
                if fraction == .7:
                    extremes.append({"cooling_type": kind, "parameter_set": case,
                                     "minus40_dry_rh0_pue": float(matrix[0, 0]), "plus50_saturated_pue": float(matrix[-1, -1])})
            test.assertTrue((by_case[0] <= by_case[1] + 1e-12).all())
            test.assertTrue((by_case[1] <= by_case[2] + 1e-12).all())
        test.assertTrue((matrices[("direct_liquid", "central", fraction)] <= matrices[("air", "central", fraction)] + 1e-12).all())
    pd.DataFrame(extremes).to_csv(output / "extreme-conditions.csv", index=False)
    baseline = sweep.iloc[:48].copy()
    checks = 0
    for column, value in [("temperature_c", 50.01), ("temperature_c", -40.01),
                          ("relative_humidity_pct", 100.01), ("relative_humidity_pct", -.01),
                          ("pressure_pa", 20000), ("temperature_c", np.nan)]:
        broken = baseline.copy()
        broken.loc[0, column] = value
        if column == "pressure_pa":
            broken.loc[1:, column] = 101325
        with test.assertRaises(ValueError):
            CoolingModel().calculate(broken, 7, 10, full_year=False)
        checks += 1
    for it, cap in [(-1, 10), (np.nan, 10), (7, 0), (7, -1), (np.ones(2), 10), (np.ones((48, 1)), 10)]:
        with test.assertRaises(ValueError):
            CoolingModel().calculate(baseline, it, cap, full_year=False)
        checks += 1
    with test.assertRaises(ValueError):
        CoolingModel().calculate(baseline.drop(index=4), 7, 10, full_year=False)
    with test.assertRaises(ValueError):
        CoolingModel().calculate(baseline, 7, 10)
    with test.assertRaises(ValueError):
        CoolingModel(version=LEGACY_VERSION, cooling_type="direct_liquid")
    checks += 3
    zero = CoolingModel().calculate(baseline, 0, 10, full_year=False)
    test.assertTrue(zero.pue.isna().all())
    test.assertTrue(zero.facility_mw.gt(0).all())
    weather_path = args.input_project / "data/processed/era5-landuse-trial/g05_r100_c592-hourly.parquet"
    mapping_path = args.input_project / "data/processed/era5-landuse-selection/grid-weather-mapping.parquet"
    mapping = pd.read_parquet(mapping_path)
    center = mapping.loc[mapping.grid_id.eq("g05_r100_c592") & mapping.row_offset.eq(1) & mapping.col_offset.eq(1)].iloc[0]
    weather = grid_weather(ds.dataset(weather_path, format="parquet"), center, [2024, 2025])
    library = ProfileLibrary(args.input_project / "data/processed/compute-load-profiles-20261025")
    regressions = 0
    for year, group in weather.groupby(weather.time_utc.dt.year):
        for hardware in ("cpu", "gpu"):
            old = ComputeLoadModel().calculate(group, library.for_year(hardware, year), 10)
            old2 = CoolingModel(version=LEGACY_VERSION).calculate(group, old.it_mw.to_numpy(), float(old.installed_reference_it_mw.iloc[0]))
            for column in ("outdoor_cooling_mw", "hvac_mw", "pump_mw", "cooling_mw", "other_mw", "facility_mw", "pue"):
                np.testing.assert_array_equal(old[column], old2[column])
            regressions += len(group)
        for kind in ("air", "direct_liquid"):
            result = CoolingModel(cooling_type=kind).calculate(group, 7, 10)
            test.assertEqual(len(result), 8784 if year == 2024 else 8760)
            dew_only = CoolingModel(cooling_type=kind).calculate(group.drop(columns="relative_humidity_pct", errors="ignore"), 7, 10)
            np.testing.assert_allclose(result.facility_mw, dew_only.facility_mw, atol=1e-10)
    prior = args.typical_project / "data/processed/typical-load-curves-20261028/R-GB200-Acme-central.parquet"
    own_it = temporary / "gb200-frozen-it.parquet"
    raw_it = pd.read_parquet(prior)
    raw_it.to_parquet(own_it, index=False)
    external = weather.loc[weather.time_utc.dt.year.eq(2025)].reset_index(drop=True)
    source = {"sha256": sha256(weather_path), "mapping_sha256": sha256(mapping_path)}
    model = CoolingModel(cooling_type="direct_liquid")
    replay_output = filesystem_path(temporary / model.metadata["cooling_code_sha256"][:8] / model.metadata["parameters_sha256"][:8])
    first = replay(own_it, model, replay_output, weather=external, weather_source=source)
    second = replay(own_it, model, replay_output, weather=external, weather_source=source)
    test.assertEqual(first, second)
    saved = next(replay_output.rglob("hourly.parquet"))
    restored = pd.read_parquet(saved)
    test.assertNotIn("hvac_mw", restored)
    test.assertNotIn("outside_calibration_weather", restored)
    np.testing.assert_array_equal(restored.it_mw, raw_it.it_mw)
    altered = external.copy()
    altered.loc[0, "temperature_c"] += .5
    with test.assertRaisesRegex(ValueError, "输入或版本不同"):
        replay(own_it, model, replay_output, weather=altered, weather_source=source)
    with test.assertRaisesRegex(ValueError, "相同的UTC小时"):
        replay(own_it, model, temporary / "wrong-time", weather=external.iloc[:-1])
    checks += 2
    # 真实气候结果与公开调查作量级校验；不声明同站点误差或置信覆盖。
    climate = pd.read_csv(output / "climate-validation.csv")
    central = climate.loc[climate.parameter_set.eq("central")]
    air = central.loc[central.cooling_type.eq("air")]
    liquid = central.loc[central.cooling_type.eq("direct_liquid")]
    test.assertTrue(air.annual_pue.between(1.2, 1.5).all())
    test.assertTrue(liquid.annual_pue.between(1.05, 1.3).all())
    test.assertLess(air.loc[air.example_label.str.contains("芬兰"), "annual_pue"].iloc[0],
                    air.loc[air.example_label.str.contains("盖恩斯维尔"), "annual_pue"].iloc[0])
    response = pd.read_csv(output / "response-curves.csv")
    test.assertEqual(len(response), 361 * 3 * 7)
    hourly = pd.read_csv(output / "g05_r100_c592-2025-hourly.csv")
    test.assertEqual(len(hourly), 8760)
    annual = pd.read_csv(output / "annual-comparison.csv")
    for prefix, version, kind in [("legacy_air", LEGACY_VERSION, "air"), ("air", VERSION, "air"), ("direct_liquid", VERSION, "direct_liquid")]:
        expected = annual.loc[annual.protocol.eq("matched_7MW_IT_10MW_capacity") & annual.cooling_version.eq(version)
                              & annual.cooling_type.eq(kind) & annual.parameter_set.eq("central"), "annual_pue"].item()
        test.assertAlmostEqual(float(hourly[f"{prefix}_facility_mw"].sum() / (7 * 8760)), expected, places=8)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        test.assertEqual(sha256(output / name), expected)
    result = {"status": "passed", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "analytic_boundary_rows": checked, "boundary_experiment": "-40..50°C × RH0..100% × IT/C0.05,0.25,0.7,1,1.1; not weather observations",
              "maximum_heat_balance_residual_mw": maximum_balance, "nonnegative_components": True,
              "temperature_monotonic_above_minus15": True, "humidity_monotonic": True, "sensitivity_order": True,
              "legacy_exact_regression_rows": regressions, "leap_year_hours": 8784, "regular_year_hours": 8760,
              "invalid_input_checks": checks, "native_ERA5_adapter_and_replay": True, "idempotent_resume": True,
              "PR13_IT_unchanged": True, "stale_legacy_columns_excluded": True,
              "external_validation_scope": "public annual PUE consistency only; no matched facility meter/weather data, no empirical error or coverage claim",
              "model": model.metadata, "verification_code_sha256": sha256(Path(__file__))}
    (output / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-project", type=Path, required=True)
    parser.add_argument("--typical-project", type=Path, required=True)
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
