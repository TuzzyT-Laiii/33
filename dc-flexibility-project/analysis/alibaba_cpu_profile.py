import json
from pathlib import Path

import numpy as np
import pandas as pd

from compute_load.model import ResourceProfile, sha256


def load_alibaba_source(project):
    directory = Path(project) / "results/alibaba-cpu-cld-20261031"
    audit = json.loads((directory / "source-audit.json").read_text(encoding="utf-8"))
    path = directory / "complete-window.csv"
    if sha256(path) != audit["outputs_sha256"][path.name]:
        raise ValueError("Alibaba 完整窗口与来源审计不符")
    frame = pd.read_csv(path)
    if len(frame) != audit["complete_hours"] or len(frame) % 24 or not frame.relative_hour.diff().iloc[1:].eq(1).all():
        raise ValueError("Alibaba 窗口必须由连续完整日期组成")
    if not frame.relative_day.isin(audit["complete_days"]).all() or frame.coverage_fraction.lt(audit["minimum_hour_core_coverage"]).any():
        raise ValueError("Alibaba 完整窗口覆盖率未通过审计阈值")
    fraction = frame.resource_fraction.to_numpy(float)
    metadata = {
        "source": "alibaba_cpu_2018", "hardware": "cpu", "scenario": "A-CLD", "days": len(frame)//24,
        "status": "short_window_replay_proxy", "input_sha256": sha256(path),
        "source_start": int(frame.relative_hour.iloc[0]), "source_end_inclusive": int(frame.relative_hour.iloc[-1]),
        "normalization_basis": "有效核秒加权 CPU 利用率；缺测不视为空闲，覆盖率另列",
        "reference_resource_count": audit["inventory_cores"],
        "source_mean_fraction": float(fraction.mean()), "source_peak_fraction": float(fraction.max()),
        "boundary_jump_fraction": float(abs(fraction[-1]-fraction[0])),
        "within_window_hourly_change_p95": float(np.quantile(abs(np.diff(fraction)),.95)),
        "seasonality": "无样本外季节变化；相对时间无真实星期或时区标识",
    }
    return {"profile": ResourceProfile("cpu",fraction,metadata), "metadata": metadata,
            "clock": np.array([f"relative_day={int(d)};hour={int(h):02d}" for d,h in zip(frame.relative_day,frame.hour_of_day)])}
