import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-count", type=int, default=88)
    args = parser.parse_args()
    paths = sorted(args.input_root.glob("*.parquet"))
    if len(paths) != args.expect_count:
        raise ValueError(f"输入文件数量应为{args.expect_count}，实际为{len(paths)}")
    rows = []
    for path in paths:
        frame = pd.read_parquet(path, columns=["time_utc", "source_hour_index"])
        clock = pd.to_datetime(frame.time_utc, utc=True)
        index = frame.source_hour_index.to_numpy(float)
        if len(frame) < 2 or clock.isna().any() or not np.isfinite(index).all():
            raise ValueError(f"时钟含缺失值或数据不足：{path.name}")
        elapsed = (clock - clock.iloc[0]).dt.total_seconds().to_numpy() / 3600.0
        delta = np.diff(elapsed)
        index_delta = np.diff(index)
        failures = []
        if (index_delta <= 0).any():
            failures.append("源小时索引未严格递增")
        if not np.allclose(index_delta, np.round(index_delta), atol=1e-10, rtol=0):
            failures.append("源记录间距包含不足整小时的间隔")
        clock_difference = delta - index_delta
        # 当前HPC输入按月日映射到同一气象年，源小时轴仍连续。
        wraps = np.isclose(clock_difference, -8760.0, atol=1e-10, rtol=0)
        identical_steps = np.isclose(clock_difference, 0.0, atol=1e-10, rtol=0)
        if not np.all(identical_steps | (wraps & path.name.endswith("-A-HPC.parquet"))):
            failures.append("气象时间与源小时轴存在未解释的间距差异")
        if path.name.endswith("-A-HPC.parquet") and not np.all(index_delta == 1.0):
            failures.append("HPC源小时轴存在内部缺口")
        with path.open("rb") as handle:
            fingerprint = hashlib.file_digest(handle, "sha256").hexdigest()
        rows.append({
            "file": path.name, "sha256": fingerprint, "hours": len(frame),
            "clock_dtype": str(clock.dtype), "start_utc": clock.iloc[0].isoformat(), "end_utc": clock.iloc[-1].isoformat(),
            "gap_count": int((index_delta > 1).sum()), "missing_internal_hours": float(np.maximum(index_delta - 1, 0).sum()),
            "maximum_step_hours": float(index_delta.max()), "weather_axis_annual_wrap_count": int(wraps.sum()),
            "execution_clock": "source_hour_index preserves actual elapsed hours",
            "max_clock_vs_source_index_step_error_hours": float(np.max(np.abs(delta - index_delta))),
            "passed": not failures, "failures": failures,
        })
    report = {"passed": all(row["passed"] for row in rows), "input_count": len(rows),
        "scope": "源小时轴严格递增；80份输入与UTC间距完全一致，8份HPC保留连续源小时并单列同一气象年的回退映射",
        "files": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "input_count": len(rows),
        "files_with_gaps": sum(row["gap_count"] > 0 for row in rows), "output": str(args.output)}, ensure_ascii=False))
    if not report["passed"]:
        raise AssertionError("部分输入时钟检查失败，详见输出报告")


if __name__ == "__main__":
    main()
