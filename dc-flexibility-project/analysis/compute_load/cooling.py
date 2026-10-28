import json
from pathlib import Path

import numpy as np
import pandas as pd
import psychrolib

from .model import COMPONENTS, ComputeLoadModel, hourly_index, normalize_weather, sha256


VERSION = "cooling-20261028-v2"
LEGACY_VERSION = "cooling-20261025-v1"
PARAMETER_SETS = ("low", "central", "high")
COOLING_TYPES = ("air", "direct_liquid")
WEATHER_COLUMNS = ("grid_id", "time_utc", "temperature_c", "relative_humidity_pct",
                   "dew_point_c", "input_dew_point_c", "pressure_pa")
COOLING_COMPONENTS = (
    "room_fan_mw", "chilled_pump_mw", "cdu_mw", "primary_pump_mw",
    "condenser_pump_mw", "tower_fan_mw", "dry_cooler_fan_mw",
    "air_compressor_mw", "liquid_compressor_mw", "freeze_heater_mw",
)


def weather_v2(weather, full_year):
    if psychrolib.isIP():
        raise RuntimeError("冷却模型要求PsychroLib使用SI单位")
    frame = weather[[column for column in WEATHER_COLUMNS if column in weather]].copy().reset_index(drop=True)
    frame["time_utc"] = hourly_index(frame.time_utc, full_year=full_year)
    if "grid_id" in frame and (frame.grid_id.isna().any() or frame.grid_id.nunique() != 1):
        raise ValueError("一次计算只接受一个非空网格标识")
    t = frame.temperature_c.to_numpy(dtype=float)
    if not np.isfinite(t).all() or ((t < -40) | (t > 50)).any():
        raise ValueError("v2室外干球温度支持-40至50°C，范围外拒绝外推")
    has_rh, has_dew = "relative_humidity_pct" in frame, "dew_point_c" in frame
    if not (has_rh or has_dew):
        raise ValueError("需要相对湿度或露点")
    if has_rh:
        rh = frame.relative_humidity_pct.to_numpy(dtype=float) / 100
        if not np.isfinite(rh).all() or ((rh < 0) | (rh > 1)).any():
            raise ValueError("v2室外相对湿度必须为0至100%")
    if has_dew:
        dew = frame.get("input_dew_point_c", frame.dew_point_c).to_numpy(dtype=float)
        if not np.isfinite(dew).all() or ((dew < -100) | (dew > t + 0.01)).any():
            raise ValueError("露点必须有限、至少-100°C，且高于干球的超差不超过0.01°C")
        frame["input_dew_point_c"] = dew
        frame["dew_point_roundoff_adjusted"] = dew > t
        dew = np.minimum(dew, t)
        calculated = psychrolib.GetRelHumFromTDewPoint(t, dew)
        if has_rh and not np.allclose(rh, calculated, atol=0.003, rtol=0):
            raise ValueError("相对湿度和露点不一致，允许绝对RH差0.3个百分点")
        if not has_rh:
            rh = calculated
        frame["dew_point_c"] = dew
    else:
        # RH=0时露点无有限定义，湿球计算仍使用准确的零湿度。
        dew = psychrolib.GetTDewPointFromRelHum(t, np.maximum(rh, 1e-12))
        frame["dew_point_c"] = np.where(rh == 0, -np.inf, dew)
        frame["input_dew_point_c"] = frame.dew_point_c
        frame["dew_point_roundoff_adjusted"] = False
    frame["relative_humidity_pct"] = rh * 100
    pressure = frame.get("pressure_pa", pd.Series(101325., index=frame.index)).to_numpy(dtype=float)
    if not np.isfinite(pressure).all() or ((pressure < 30000) | (pressure > 110000)).any():
        raise ValueError("pressure_pa支持30000至110000Pa")
    frame["pressure_pa"] = pressure
    frame["pressure_assumed"] = "pressure_pa" not in weather
    vapor = rh * psychrolib.GetSatVapPres(t)
    if (vapor >= pressure).any():
        raise ValueError("水蒸气分压力必须小于总气压")
    frame["wet_bulb_c"] = psychrolib.GetTWetBulbFromRelHum(t, rh, pressure)
    return frame


def validate_it(it_mw, capacity_mw, hours):
    if not np.isfinite(capacity_mw) or capacity_mw <= 0:
        raise ValueError("安装参考IT容量必须为正有限数")
    it = np.asarray(it_mw, dtype=float)
    if it.ndim == 0:
        it = np.full(hours, it.item())
    if it.shape != (hours,) or not np.isfinite(it).all() or (it < 0).any():
        raise ValueError("IT功率必须为非负有限标量或与天气等长的一维数组")
    return it


class CoolingModel:
    def __init__(self, *, cooling_type="air", parameter_set="central", version=VERSION):
        if version not in (VERSION, LEGACY_VERSION):
            raise ValueError(f"未知冷却版本：{version}")
        if cooling_type not in COOLING_TYPES or parameter_set not in PARAMETER_SETS:
            raise ValueError("cooling_type需为air/direct_liquid，parameter_set需为low/central/high")
        if version == LEGACY_VERSION and (cooling_type != "air" or parameter_set != "central"):
            raise ValueError("旧版本只提供原中央风冷模型")
        self.version, self.cooling_type, self.parameter_set = version, cooling_type, parameter_set
        path = Path(__file__).with_name("assets") / "cooling-20261028-v2.json"
        self.asset = json.loads(path.read_text(encoding="utf-8"))
        self.p = {key: row[parameter_set] for key, row in self.asset["parameters"].items()}
        self.legacy = ComputeLoadModel() if version == LEGACY_VERSION else None
        self.metadata = {"cooling_version": version, "cooling_type": cooling_type, "parameter_set": parameter_set,
                         "cooling_code_sha256": sha256(Path(__file__)), "parameters_sha256": sha256(path),
                         "scope": self.asset["scope"], "parameters_are_scenario_assumptions": True}
        if self.legacy is not None:
            self.metadata["legacy_model"] = self.legacy.metadata

    def calculate(self, weather, it_mw, installed_reference_it_mw, *, full_year=True):
        if self.legacy is not None:
            return self._legacy(weather, it_mw, installed_reference_it_mw, full_year)
        frame = weather_v2(weather, full_year)
        it = validate_it(it_mw, installed_reference_it_mw, len(frame))
        cap, p = float(installed_reference_it_mw), self.p
        t, wb = frame.temperature_c.to_numpy(), frame.wet_bulb_c.to_numpy()
        q = it / cap
        liquid_share = p["liquid_fraction"] if self.cooling_type == "direct_liquid" else 0.
        loss = it * (1 / (p["ups_efficiency"] * p["distribution_efficiency"]) - 1) + cap * p["electrical_standby_ratio"]
        fixed = np.full(len(it), cap * p["lighting_controls_ratio"])
        indoor_humidity = psychrolib.GetHumRatioFromRelHum(p["air_supply_c"], 0.5, frame.pressure_pa.to_numpy())
        density = psychrolib.GetMoistAirDensity(p["air_supply_c"], indoor_humidity, frame.pressure_pa.to_numpy())
        fan_ratio = p["air_pressure_drop_pa"] / (density * 1006 * p["air_delta_k"] * p["air_fan_efficiency"])
        if (fan_ratio >= 1).any():
            raise ValueError("所选压降、空气温差和气压导致风机热反馈无稳态解")
        air_base = (1 - liquid_share) * it + loss + fixed
        room_fan = air_base * fan_ratio / (1 - fan_ratio)
        chilled_ratio = p["chilled_pump_pressure_pa"] / (1000 * 4180 * p["chilled_water_delta_k"] * p["pump_efficiency"])
        chilled_pump = (air_base + room_fan) * chilled_ratio / (1 - chilled_ratio)
        air_heat = air_base + room_fan + chilled_pump
        supply = p["air_supply_c"] - p["coil_approach_k"]
        available_tower_water = np.maximum(p["minimum_tower_water_c"], wb + p["tower_approach_k"])
        available_free_water = available_tower_water + p["economizer_hx_approach_k"]
        air_free = np.clip((supply + p["chilled_water_delta_k"] - available_free_water) / p["chilled_water_delta_k"], 0, 1)
        condenser_c = available_tower_water + p["condenser_approach_k"]

        def cop(supply_c):
            evap = supply_c - p["evaporator_approach_k"]
            reference_evap = p["reference_chilled_supply_c"] - p["evaporator_approach_k"]
            reference_lift = p["reference_condenser_water_c"] + p["condenser_approach_k"] - reference_evap
            value = p["cop_reference"] * (evap + 273.15) / (reference_evap + 273.15)
            value = value * reference_lift / np.maximum(condenser_c - evap, p["minimum_lift_k"])
            return np.minimum(value, p["cop_maximum"]) * (1 - p["part_load_penalty"] * (1 - np.minimum(q, 1)))

        air_cop = cop(supply)
        air_compressor = air_heat * (1 - air_free) / air_cop
        liquid_it = it * liquid_share
        cdu_pump = liquid_it * p["cdu_power_ratio"] / (1 - p["cdu_power_ratio"])
        primary_ratio = p["primary_pump_pressure_pa"] / (p["primary_fluid_density_kg_m3"] * p["primary_fluid_cp_j_kgk"] * p["liquid_delta_k"] * p["pump_efficiency"])
        primary_pump = (liquid_it + cdu_pump) * primary_ratio / (1 - primary_ratio)
        liquid_heat = liquid_it + cdu_pump + primary_pump
        primary_supply = p["secondary_supply_c"] - p["cdu_approach_k"]
        dry_free = np.clip((primary_supply + p["liquid_delta_k"] - (t + p["dry_cooler_approach_k"])) / p["liquid_delta_k"], 0, 1)
        liquid_tower_free = np.clip((primary_supply + p["liquid_delta_k"] - available_free_water) / p["liquid_delta_k"], 0, 1)
        liquid_cop = cop(primary_supply)
        liquid_compressor = liquid_heat * (1 - dry_free) * (1 - liquid_tower_free) / liquid_cop
        tower_heat = air_heat + air_compressor + (1 - dry_free) * liquid_heat + liquid_compressor
        condenser_pump = tower_heat * p["condenser_pump_ratio"] / (1 - p["condenser_pump_ratio"])
        # 冷塔低于5°C湿球采用最低运行和旁通，高温段再提高风机速度。
        tower_fan_ratio = p["tower_fan_minimum_ratio"] + p["tower_fan_increment_ratio"] * np.clip((wb - 5) / 20, 0, 1) ** 3
        tower_fan = (tower_heat + condenser_pump) * tower_fan_ratio
        dry_fan_ratio = p["dry_fan_minimum_ratio"] + p["dry_fan_increment_ratio"] * np.clip((t + 10) / 35, 0, 1) ** 3
        dry_fan = dry_free * liquid_heat * dry_fan_ratio
        freeze = cap * p["freeze_heater_ratio_per_k"] * np.maximum(-15 - t, 0)
        values = (room_fan, chilled_pump, cdu_pump, primary_pump, condenser_pump, tower_fan,
                  dry_fan, air_compressor, liquid_compressor, freeze)
        for column, value in zip(COOLING_COMPONENTS, values):
            frame[column] = value
        frame["it_mw"], frame["installed_reference_it_mw"] = it, cap
        frame["cooling_version"], frame["cooling_type"], frame["parameter_set"] = self.version, self.cooling_type, self.parameter_set
        frame["ups_distribution_loss_mw"], frame["lighting_controls_mw"] = loss, fixed
        frame["other_mw"] = loss + fixed
        frame["cooling_mw"] = frame[list(COOLING_COMPONENTS)].sum(axis=1)
        frame["facility_mw"] = it + frame.cooling_mw + frame.other_mw
        frame["pue"] = np.divide(frame.facility_mw, it, out=np.full(len(it), np.nan), where=it > 0)
        frame["cooling_to_it_ratio"] = np.divide(frame.cooling_mw, it, out=np.full(len(it), np.nan), where=it > 0)
        frame["cooling_to_facility_ratio"] = frame.cooling_mw / frame.facility_mw
        frame["heat_demand_mw_th"] = it
        frame["air_coil_heat_mw_th"], frame["liquid_loop_heat_mw_th"] = air_heat, liquid_heat
        frame["air_free_cooling_fraction"] = air_free
        frame["liquid_dry_cooling_fraction"] = dry_free if liquid_share else 0.
        frame["liquid_tower_free_cooling_fraction"] = liquid_tower_free if liquid_share else 0.
        frame["air_cop"], frame["liquid_cop"] = air_cop, liquid_cop
        frame["air_supply_c"], frame["chilled_supply_c"] = p["air_supply_c"], supply
        frame["secondary_supply_c"] = p["secondary_supply_c"] if liquid_share else np.nan
        frame["primary_supply_c"] = primary_supply if liquid_share else np.nan
        frame["outside_reference_capacity"] = it > cap
        frame["outside_empirical_validation"] = True
        # 塔风机和干冷器风机直接向室外排热，室内及水泵热量已经逐支路计入。
        frame["heat_rejected_mw_th"] = (tower_heat + condenser_pump + tower_fan
                                          + dry_free * liquid_heat + dry_fan + freeze)
        frame["heat_balance_residual_mw"] = frame.heat_rejected_mw_th - frame.facility_mw
        frame.attrs = self.metadata.copy()
        return frame

    def _legacy(self, weather, it_mw, capacity, full_year):
        frame = normalize_weather(weather[[column for column in WEATHER_COLUMNS if column in weather]], full_year=full_year)
        it = validate_it(it_mw, capacity, len(frame))
        q, knots = it / capacity, self.legacy.cooling["knots"]
        basis = [np.ones(len(it)), q, q ** 2]
        for threshold in knots["dry_warm"]:
            basis.append(q * np.maximum(frame.temperature_c.to_numpy() - threshold, 0))
        for threshold in knots["dew_warm"]:
            basis.append(q * np.maximum(frame.dew_point_c.to_numpy() - threshold, 0))
        basis.append(np.maximum(knots["dry_cold"] - frame.temperature_c.to_numpy(), 0))
        prediction = np.column_stack(basis) @ self.legacy.coefficients * capacity
        for index, column in enumerate(COMPONENTS):
            frame[column] = prediction[:, index]
        frame["it_mw"], frame["installed_reference_it_mw"] = it, capacity
        frame["cooling_mw"] = frame[list(COMPONENTS[:3])].sum(axis=1)
        frame["facility_mw"] = it + frame.cooling_mw + frame.other_mw
        frame["pue"] = frame.facility_mw / it
        frame["cooling_to_it_ratio"] = frame.cooling_mw / it
        frame["cooling_to_facility_ratio"] = frame.cooling_mw / frame.facility_mw
        bounds = self.legacy.cooling["calibration_range"]
        frame["outside_calibration_weather"] = ~(frame.temperature_c.between(*bounds["temperature_c"])
                                                  & frame.dew_point_c.between(*bounds["dew_point_c"]))
        frame["outside_calibration_it_range"] = ~pd.Series(q).between(*bounds["it_fraction"])
        frame["cooling_version"], frame["cooling_type"], frame["parameter_set"] = self.version, "air", "central"
        frame.attrs = self.metadata.copy()
        return frame
