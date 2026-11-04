import calendar
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import numba
import pandas as pd
import psychrolib


psychrolib.SetUnitSystem(psychrolib.SI)
VERSION = "compute-load-20261025-v1"
SIZES_MW = (1, 10, 100, 500)
DEW_POINT_ROUNDOFF_C = 0.01
COMPONENTS = ("outdoor_cooling_mw", "hvac_mw", "pump_mw", "other_mw")
HARDWARE = {
    "cpu": {"full_w": 546.0, "cores": 128, "cards": 0},
    "gpu": {"full_w": 8430.0, "cores": None, "cards": 8},
}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def hourly_index(values, full_year=True, *, allow_gaps=False):
    times = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    if len(times) < 2 or times.isna().any() or not times.is_unique:
        raise ValueError("必须提供至少两个无重复的 UTC 小时")
    differences = times.to_series().diff().iloc[1:]
    if allow_gaps and full_year:
        raise ValueError("全年输入不能允许时间缺口")
    valid = differences.ge(pd.Timedelta(hours=1)) if allow_gaps else differences.eq(pd.Timedelta(hours=1))
    if not valid.all():
        raise ValueError("UTC 时间必须按小时严格递增，并满足指定的连续性要求")
    if (times != times.floor("h")).any():
        raise ValueError("UTC 时间必须位于整点")
    if full_year:
        year = times[0].year
        expected = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", freq="h", inclusive="left", tz="UTC")
        if not times.equals(expected):
            raise ValueError("全年输入必须完整覆盖 UTC 日历年；平年 8760 小时，闰年 8784 小时")
    return times


def normalize_weather(weather, full_year=True):
    if psychrolib.isIP():
        raise RuntimeError("PsychroLib 当前单位制已被其他调用更改；本模块要求 SI")
    frame = weather.copy().reset_index(drop=True)
    frame["time_utc"] = hourly_index(frame["time_utc"], full_year)
    temperature = frame["temperature_c"].to_numpy(dtype=float)
    if not np.isfinite(temperature).all() or ((temperature < -100) | (temperature > 100)).any():
        raise ValueError("temperature_c 必须为有限摄氏温度，范围为 -100 至 100；开尔文须显式转换")
    has_rh = "relative_humidity_pct" in frame
    has_dew = "dew_point_c" in frame
    if not has_rh and not has_dew:
        raise ValueError("必须提供 relative_humidity_pct 或 dew_point_c")
    if has_rh:
        rh = frame.relative_humidity_pct.to_numpy(dtype=float)
        if not np.isfinite(rh).all() or ((rh <= 0) | (rh > 100)).any():
            raise ValueError("相对湿度必须满足 0 < RH <= 100，单位为百分数")
        calculated = psychrolib.GetTDewPointFromRelHum(temperature, rh / 100)
    if has_dew:
        original = frame["input_dew_point_c"] if "input_dew_point_c" in frame else frame.dew_point_c
        dew = original.to_numpy(dtype=float)
        if not np.isfinite(dew).all() or ((dew < -100) | (dew > temperature + DEW_POINT_ROUNDOFF_C)).any():
            raise ValueError("露点必须有限且不低于 -100℃，高于干球的超差不能超过 0.01℃")
        if has_rh and not np.allclose(dew, calculated, atol=0.05, rtol=0):
            raise ValueError("相对湿度与露点不一致，最大允许偏差为 0.05 摄氏度")
        frame["input_dew_point_c"] = dew
        frame["dew_point_roundoff_adjusted"] = dew > temperature
        dew = np.minimum(dew, temperature)
        frame["dew_point_c"] = dew
        if not has_rh:
            frame["relative_humidity_pct"] = psychrolib.GetRelHumFromTDewPoint(temperature, dew) * 100
    else:
        frame["dew_point_c"] = calculated
        frame["input_dew_point_c"] = calculated
        frame["dew_point_roundoff_adjusted"] = False
    if "grid_id" in frame and (frame.grid_id.isna().any() or frame.grid_id.nunique() != 1):
        raise ValueError("一次计算只能包含一个非空 grid_id")
    return frame


@dataclass(frozen=True)
class ResourceProfile:
    hardware: str
    fraction: np.ndarray
    provenance: dict

    def __post_init__(self):
        if self.hardware not in HARDWARE:
            raise ValueError("hardware 只支持 cpu 和 gpu")
        values = np.asarray(self.fraction, dtype=float).copy()
        if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("资源份额必须是一维有限数列，取值为 0 至 1")
        if not self.provenance.get("source"):
            raise ValueError("资源曲线必须说明 source")
        values.setflags(write=False)
        object.__setattr__(self, "fraction", values)


class ProfileLibrary:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        self.cache = {}

    def for_year(self, hardware, year):
        hours = 8784 if calendar.isleap(year) else 8760
        key = f"{hardware}-{hours}"
        if key not in self.cache:
            record = self.manifest["profiles"][key]
            path = self.directory / record["file"]
            if sha256(path) != record["sha256"]:
                raise ValueError(f"资源曲线校验值不一致：{path}")
            table = pd.read_parquet(path)
            if len(table) != hours or not np.array_equal(table.hour_of_year, np.arange(hours)):
                raise ValueError(f"资源曲线小时数量或顺序错误：{path}")
            self.cache[key] = ResourceProfile(hardware, table.resource_fraction.to_numpy(), {
                "source": record["source"], "profile_file_sha256": record["sha256"],
                "source_trace": record["source_trace"], "source_year": record["source_year"],
                "replay_calendar": "按相对小时顺序映射到目标 UTC 日历；保留原始份额，不调整均值",
            })
        return self.cache[key]


class ComputeLoadModel:
    def __init__(self, assets=None):
        self.assets = Path(assets) if assets is not None else Path(__file__).with_name("assets")
        self.cooling = json.loads((self.assets / "cooling-calibration.json").read_text(encoding="utf-8"))
        self.curves = {}
        for hardware in HARDWARE:
            frame = pd.read_csv(self.assets / f"{hardware}-measured-power-anchors.csv")
            if hardware == "gpu":
                frame = frame.loc[~frame.measurement.str.contains("no CPU", regex=False)]
            frame = frame.sort_values("load_fraction")
            x, y = frame.load_fraction.to_numpy(), frame.node_power_w.to_numpy()
            if (np.diff(x) <= 0).any() or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("功率标定点必须有限，资源份额必须严格递增")
            if x[0] != 0 or x[-1] > 1 or y[0] <= 0 or y[-1] != HARDWARE[hardware]["full_w"]:
                raise ValueError("功率标定点与参考硬件不一致")
            self.curves[hardware] = (x, y)
        coefficients = np.asarray(self.cooling["coefficients"], dtype=float)
        expected = ["capacity_fixed", "heat_linear", "heat_squared_over_capacity",
                    "heat_times_dry_hinge_1", "heat_times_dry_hinge_2",
                    "heat_times_dew_hinge_1", "heat_times_dew_hinge_2", "cold_weather_fixed_heater"]
        if self.cooling["feature_names"] != expected:
            raise ValueError("冷却模型特征顺序与实现不一致")
        if coefficients.shape != (8, 4) or not np.isfinite(coefficients).all() or (coefficients < 0).any():
            raise ValueError("冷却模型必须包含 8×4 个有限非负系数")
        self.coefficients = coefficients
        self.metadata = {
            "version": VERSION, "calibration_version": "20261025-v4",
            "model_source_sha256": sha256(Path(__file__)),
            "dew_point_roundoff_tolerance_c": DEW_POINT_ROUNDOFF_C,
            "software": {"numpy": np.__version__, "pandas": pd.__version__, "numba": numba.__version__},
            "assets_sha256": {p.name: sha256(p) for p in sorted(self.assets.iterdir()) if p.is_file()},
            "power_scope": "实测整机功率；CPU/GPU 资源到利用率为跨数据集代理",
            "cooling_scope": self.cooling["interpretation"],
        }

    def calculate(self, weather, profile, it_capacity_mw, *, extrapolation="flag", full_year=True):
        if it_capacity_mw not in SIZES_MW:
            raise ValueError(f"IT 参考规模必须为 {SIZES_MW} MW")
        if extrapolation not in ("flag", "error"):
            raise ValueError("extrapolation 必须为 flag 或 error")
        frame = normalize_weather(weather, full_year)
        if len(frame) != len(profile.fraction):
            raise ValueError("天气与资源曲线小时数量不一致")
        hardware = HARDWARE[profile.hardware]
        nodes = int(np.floor(it_capacity_mw * 1e6 / hardware["full_w"]))
        capacity = nodes * hardware["full_w"] / 1e6
        x, y = self.curves[profile.hardware]
        frame["hardware"] = profile.hardware
        frame["reference_it_capacity_mw"] = it_capacity_mw
        frame["installed_reference_it_mw"] = capacity
        frame["node_count"] = nodes
        frame["gpu_count"] = nodes * hardware["cards"]
        frame["resource_fraction"] = profile.fraction
        frame["it_mw"] = np.interp(profile.fraction, x, y) * nodes / 1e6
        frame["heat_demand_mw_th"] = frame.it_mw
        q = frame.it_mw.to_numpy() / capacity
        temperature = frame.temperature_c.to_numpy(dtype=float)
        dew = frame.dew_point_c.to_numpy(dtype=float)
        knots = self.cooling["knots"]
        basis = [np.ones(len(frame)), q, q ** 2]
        for threshold in knots["dry_warm"]:
            basis.append(q * np.maximum(temperature - threshold, 0))
        for threshold in knots["dew_warm"]:
            basis.append(q * np.maximum(dew - threshold, 0))
        basis.append(np.maximum(knots["dry_cold"] - temperature, 0))
        prediction = np.column_stack(basis) @ self.coefficients * capacity
        for index, component in enumerate(COMPONENTS):
            frame[component] = prediction[:, index]
        frame["cooling_mw"] = frame[list(COMPONENTS[:3])].sum(axis=1)
        frame["facility_mw"] = frame.it_mw + frame.cooling_mw + frame.other_mw
        frame["pue"] = frame.facility_mw / frame.it_mw
        bounds = self.cooling["calibration_range"]
        supported = np.ones(len(frame), dtype=bool)
        for column in ("temperature_c", "dew_point_c"):
            supported &= frame[column].between(*bounds[column]).to_numpy()
        frame["outside_calibration_weather"] = ~supported
        frame["outside_calibration_it_range"] = ~pd.Series(q).between(*bounds["it_fraction"]).to_numpy()
        frame["outside_power_anchor_range"] = (profile.fraction < x[0]) | (profile.fraction > x[-1])
        if extrapolation == "error" and (frame.outside_calibration_weather | frame.outside_calibration_it_range | frame.outside_power_anchor_range | frame.dew_point_roundoff_adjusted).any():
            raise ValueError("输入存在超出标定范围的小时；使用 flag 策略可以保留外推标记并计算")
        lo, hi = self.cooling["cooling_holdout_residual_quantiles_per_reference_mw"]
        frame["cooling_residual_band_low_mw"] = np.maximum(0, frame.cooling_mw + lo * capacity)
        frame["cooling_residual_band_high_mw"] = np.maximum(0, frame.cooling_mw + hi * capacity)
        if profile.hardware == "cpu":
            frame["cpu_core_usage_proxy"] = nodes * hardware["cores"] * profile.fraction
            frame["cumulative_cpu_core_proxy_hours"] = frame.cpu_core_usage_proxy.cumsum()
        else:
            frame["allocated_gpu_equivalents"] = nodes * hardware["cards"] * profile.fraction
            frame["cumulative_allocated_gpu_hours"] = frame.allocated_gpu_equivalents.cumsum()
        for component in ("it", "cooling", "other", "facility"):
            frame[f"cumulative_{component}_mwh"] = frame[f"{component}_mw"].cumsum()
        frame.attrs = {**self.metadata, "resource_profile": profile.provenance}
        return frame


def annual_summary(frame):
    hardware = frame.hardware.iloc[0]
    resource = "cpu_core_usage_proxy" if hardware == "cpu" else "allocated_gpu_equivalents"
    return {
        "grid_id": str(frame.grid_id.iloc[0]) if "grid_id" in frame else None,
        "year": int(frame.time_utc.iloc[0].year), "hardware": hardware,
        "reference_it_capacity_mw": float(frame.reference_it_capacity_mw.iloc[0]),
        "installed_reference_it_mw": float(frame.installed_reference_it_mw.iloc[0]),
        "node_count": int(frame.node_count.iloc[0]), "gpu_count": int(frame.gpu_count.iloc[0]),
        "hours": len(frame), "mean_resource_fraction": float(frame.resource_fraction.mean()),
        "resource_proxy_hours": float(frame[resource].sum()),
        "annual_it_mwh": float(frame.it_mw.sum()), "annual_cooling_mwh": float(frame.cooling_mw.sum()),
        "annual_other_mwh": float(frame.other_mw.sum()), "annual_facility_mwh": float(frame.facility_mw.sum()),
        "heat_energy_mwh_th": float(frame.heat_demand_mw_th.sum()),
        "mean_facility_mw": float(frame.facility_mw.mean()), "peak_facility_mw": float(frame.facility_mw.max()),
        "energy_weighted_pue": float(frame.facility_mw.sum() / frame.it_mw.sum()),
        "outside_calibration_weather_hours": int(frame.outside_calibration_weather.sum()),
        "outside_calibration_it_hours": int(frame.outside_calibration_it_range.sum()),
        "outside_power_anchor_hours": int(frame.outside_power_anchor_range.sum()),
        "dew_point_roundoff_adjusted_hours": int(frame.dew_point_roundoff_adjusted.sum()),
    }
