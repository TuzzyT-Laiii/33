import json
from pathlib import Path

import numpy as np
import pandas as pd

from compute_load.model import ResourceProfile, sha256


def load_asi_sources(project):
    directory = Path(project) / "results/asi-185day-workload-types-20261031"
    manifest = json.loads((directory / "inf-source-manifest.json").read_text("utf-8"))
    path = directory / manifest["input_file"]
    if sha256(path) != manifest["input_sha256"]:
        raise ValueError("INF 185 天小时输入 SHA-256 不匹配")
    frame = pd.read_csv(path, float_precision="round_trip")
    if len(frame) != 4440 or not np.array_equal(frame.hour_index, np.arange(4440)):
        raise ValueError("INF 输入须完整覆盖 185×24 个连续小时")
    if not np.array_equal(frame.day, frame.hour_index // 24) or not np.array_equal(frame.hour, frame.hour_index % 24):
        raise ValueError("INF 相对日期或小时错误")
    np.testing.assert_allclose(frame.combined_used_gpu_hours, frame.online_used_gpu_hours + frame.offline_used_gpu_hours, rtol=1e-12)
    reference = manifest["peak_reference_gpu_hours"]
    if reference <= 0 or frame.combined_used_gpu_hours.max() != reference:
        raise ValueError("INF 观测峰值归一化分母不匹配")
    clock = np.array([f"relative_day={d};hour={h:02d}" for d, h in zip(frame.day, frame.hour)])
    sources = {}
    for key, prefix in [("asi_gpu", "combined"), ("asi_online_gpu", "online"), ("asi_offline_gpu", "offline")]:
        values = frame[f"{prefix}_used_gpu_hours"].to_numpy()
        metadata = {"source": key, "hardware": "gpu", "days": 185, "source_start": clock[0], "source_end_inclusive": clock[-1],
                    "normalization_basis": manifest["normalization"], "reference_resource_count": reference,
                    "status": "185_day_inference_replay_proxy", "input_file": str(path), "input_sha256": manifest["input_sha256"],
                    "source_mean_fraction": float(values.mean() / reference), "source_peak_fraction": float(values.max() / reference),
                    "boundary_jump_fraction": float(abs(values[-1] - values[0]) / reference),
                    "within_window_hourly_change_p95": float(np.quantile(abs(np.diff(values)), .95) / reference),
                    "business_scope": prefix + "_inference_only", "physical_utilization": False,
                    "gpu_equivalence": "异构 GPU 归属资源代理，未换算指定硬件吞吐；B-INF/C-INF/D-INF 须分别标定"}
        sources[key] = {"profile": ResourceProfile("gpu", values / reference, metadata), "clock": clock, "metadata": metadata}
    return sources
