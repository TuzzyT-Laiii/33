import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def load_module(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ValueError(f"无法读取模型源码：{path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def contiguous_reference(arrival, deadline):
    # 连续小时的既有约束，用于核对未受缺口影响的数据。
    size = len(arrival)
    cumulative = np.r_[0.0, np.cumsum(arrival)]
    index = np.arange(size)
    bound = cumulative[1:] - cumulative[np.maximum(index - deadline + 1, 0)] if deadline > 0 else np.zeros(size)
    bound = np.maximum(bound, 0.0)
    bound[-1] = 0.0
    return bound


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--utc-model", type=Path, required=True)
    parser.add_argument("--utc-model-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_hash = digest(args.utc_model)
    if old_hash != args.utc_model_sha256:
        raise ValueError("UTC模型源码指纹与已求解版本不同")
    source_model = load_module(args.source_model, "source_hour_model_for_audit")
    utc_model = load_module(args.utc_model, "utc_model_for_audit")
    paths = sorted(args.input_root.glob("*.parquet"))
    if len(paths) != 88:
        raise ValueError(f"预期88份输入，实际{len(paths)}")
    records = []
    for path in paths:
        frame = pd.read_parquet(path, columns=["time_utc", "source_hour_index", "whole", "split"])
        source_hours = frame.source_hour_index.to_numpy(float)
        if not np.isfinite(source_hours).all() or np.any(np.diff(source_hours) <= 0):
            raise ValueError(f"源小时轴无效：{path.name}")
        training = path.name.endswith("-TRN.parquet")
        if training:
            clock = pd.to_datetime(frame.time_utc, utc=True)
            elapsed = (clock - clock.iloc[0]).dt.total_seconds().to_numpy() / 3600.0
            if not np.array_equal(np.diff(source_hours), np.diff(elapsed)):
                raise ValueError(f"训练UTC与源小时间距不等：{path.name}")
        elif not np.all(np.diff(source_hours) == 1.0):
            raise ValueError(f"连续来源出现缺口，不能复用旧约束：{path.name}")
        checks = []
        for category in ("whole", "split"):
            arrival = frame[category].to_numpy(float)
            for deadline in (0, 12, 48):
                reference = utc_model.backlog_limit(arrival, deadline, frame.time_utc) if training else contiguous_reference(arrival, deadline)
                actual = source_model.backlog_limit(arrival, deadline, source_hours)
                checks.append({"category": category, "deadline_hours": deadline,
                    "exact_array_equality": bool(np.array_equal(reference, actual)),
                    "maximum_absolute_difference_mwh": float(np.max(np.abs(reference - actual)))})
        records.append({"input": path.name, "input_sha256": digest(path), "hours": len(frame),
            "reference": "verified_real_utc_deadline_model" if training else "original_contiguous_hour_model",
            "checks": checks, "passed": all(check["exact_array_equality"] for check in checks)})
    result = {"passed": all(record["passed"] for record in records), "input_count": len(records),
        "array_comparisons": sum(len(record["checks"]) for record in records),
        "utc_model_sha256": old_hash, "source_hour_model_sha256": digest(args.source_model),
        "scope": "仅证明任务时限约束数组一致；源小时轴区分天气年映射与真实经过时间", "inputs": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("passed", "input_count", "array_comparisons")}, ensure_ascii=False))
    if not result["passed"]:
        raise AssertionError("源小时轴与对应有效参考约束不等，详见报告")


if __name__ == "__main__":
    main()
