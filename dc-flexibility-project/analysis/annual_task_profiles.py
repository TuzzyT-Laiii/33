import json
from pathlib import Path

import numpy as np
import pandas as pd

from compute_load.model import ResourceProfile, hourly_index, sha256


def complete_days(frame, clock, value):
    selected = frame[[clock, value]].sort_values(clock).copy()
    counts = selected.groupby(selected[clock].dt.floor("D"))[value].count()
    selected = selected.loc[selected[clock].dt.floor("D").isin(counts.loc[counts.eq(24)].index)]
    if selected.empty or selected[clock].duplicated().any():
        raise ValueError("源需求必须包含无重复的完整日期")
    if not selected[clock].diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("源需求完整日期之间存在缺口")
    values = selected[value].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("源需求必须为非负有限数值")
    return selected.reset_index(drop=True)


def load_sources(project):
    project = Path(project)
    data = project / "data/processed/task-trace-audit-20261026"
    audit = json.loads((project / "data/metadata/task-trace-audit-20261026/asi-acme-validation.json").read_text(encoding="utf-8"))
    inputs = ("lisa/lisa-hourly-core-demand.parquet", "acme-hourly-resource-demand.parquet", "asi-hourly-resource-demand.csv")
    for name in inputs:
        if sha256(data / name) != audit["outputs_sha256"][name]:
            raise ValueError(f"需求文件与核验记录不一致：{name}")
    lisa = pd.read_parquet(data / inputs[0])
    acme = pd.read_parquet(data / inputs[1])
    asi = pd.read_csv(data / inputs[2])
    sources = {}

    def add(key, hardware, frame, clock, column, denominator, basis, status, source_file):
        selected = complete_days(frame, clock, column)
        fraction = selected[column].to_numpy(dtype=float) / np.asarray(denominator, dtype=float)
        profile = ResourceProfile(hardware, fraction, {"source": key, "capacity_basis": basis})
        sources[key] = {"profile": profile, "clock": selected[clock].astype(str).to_numpy(),
            "metadata": {"source": key, "hardware": hardware, "days": len(selected)//24,
                "source_start": str(selected[clock].iloc[0]), "source_end_inclusive": str(selected[clock].iloc[-1]),
                "normalization_basis": basis, "reference_resource_count": float(denominator) if np.ndim(denominator) == 0 else None,
                "status": status, "input_file": source_file, "input_sha256": sha256(data / source_file),
                "source_mean_fraction": float(fraction.mean()), "source_peak_fraction": float(fraction.max()),
                "boundary_jump_fraction": float(abs(fraction[-1]-fraction[0])),
                "within_window_hourly_change_p95": float(np.quantile(abs(np.diff(fraction)), .95))}}

    selected = lisa.loc[lisa.hardware_group.eq("cpu_partition")]
    denominator = float(complete_days(selected, "source_time", "allocated_core_hours").allocated_core_hours.max())
    add("lisa_cpu", "cpu", selected, "source_time", "allocated_core_hours", denominator,
        "完整观测窗口最高小时平均分配核心数；形状归一化参考，非库存或实测利用率", "long_window_replay_proxy", inputs[0])
    for hardware in ("cpu", "gpu"):
        denominator = audit["acme"]["resource_integrals"][hardware]["maximum_concurrent_allocated_units"]
        add(f"acme_{hardware}", hardware, acme.loc[acme.hardware.eq(hardware)], "time_utc", "mean_allocated_units", denominator,
            "全部有效执行区间的观测瞬时并发峰值；非完整设施库存",
            "cpu_development_diagnostic" if hardware == "cpu" else "development_replay_proxy", inputs[1])
    if len(asi) != 24 or asi.relative_day.nunique() != 1 or asi.hour.tolist() != list(range(24)):
        raise ValueError("ASI 演示输入须为已核验的一个连续完整日")
    asi["relative_clock"] = pd.date_range("2000-01-01", periods=24, freq="h")
    add("asi_gpu", "gpu", asi, "relative_clock", "used_gpu_hour_fraction", 1,
        "源每小时归属 GPU 小时/同小时异构 GPU 库存；未换算吞吐等价", "one_day_repetition_demonstration", inputs[2])
    sources["asi_gpu"]["clock"] = np.array([f"relative_day={int(asi.relative_day.iloc[0])};hour={h:02d}" for h in asi.hour])
    sources["asi_gpu"]["metadata"].update(source_start=sources["asi_gpu"]["clock"][0], source_end_inclusive=sources["asi_gpu"]["clock"][-1])
    return sources


def replay_hourly(source, times, phase_hours=0):
    times = hourly_index(times)
    profile = source["profile"]
    length = len(profile.fraction)
    if length % 24 or phase_hours < 0 or phase_hours >= length or phase_hours % 24:
        raise ValueError("源序列和相位必须以完整日期为单位，且相位位于源窗口内")
    position = (np.arange(len(times)) + phase_hours) % length
    provenance = {**profile.provenance, **source["metadata"],
        "annualization": "完整连续观测窗口按原顺序循环；不平滑、不调整均值、不添加季节需求",
        "target_clock": "UTC；源午夜对应目标 UTC 午夜为相位设定，不表示已识别原时区",
        "source_hours": length, "target_hours": len(times), "phase_hours": phase_hours,
        "complete_cycles": len(times)//length, "remaining_hours": len(times)%length,
        "seasonality": "样本外季节性、增长与节假日需求未识别"}
    result = ResourceProfile(profile.hardware, profile.fraction[position], provenance)
    mapping = pd.DataFrame({"time_utc": times, "source_hour_index": position,
        "source_clock": source["clock"][position], "replay_cycle_index": (np.arange(len(times))+phase_hours)//length,
        "replay_wrap": np.r_[False, np.diff(position) < 0], "resource_fraction": result.fraction})
    return result, mapping
