import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fifo_check(arrivals, executions, clock_hours, deadline_hours, tolerance):
    # 按实际到达顺序分配执行量，独立于求解器的积压约束公式。
    pending = deque()
    early = 0.0
    late = 0.0
    maximum_delay = 0.0
    delayed_work = 0.0
    delay_work_hours = 0.0
    for timestamp, arrival, execution in zip(clock_hours, arrivals, executions, strict=True):
        if arrival > 0.0:
            pending.append([float(timestamp), float(arrival)])
        remaining = max(0.0, float(execution))
        while remaining > 0.0 and pending:
            start, available = pending[0]
            amount = min(remaining, available)
            delay = float(timestamp - start)
            if amount > tolerance:
                maximum_delay = max(maximum_delay, delay)
            if delay > deadline_hours + 1e-9:
                late += amount
            if delay > 1e-9:
                delayed_work += amount
                delay_work_hours += delay * amount
            remaining -= amount
            pending[0][1] -= amount
            if pending[0][1] <= 0.0:
                pending.popleft()
        early += remaining
    return {
        "early_execution_mwh": early,
        "late_execution_mwh": late,
        "unserved_mwh": float(sum(item[1] for item in pending)),
        "maximum_delay_hours_above_tolerance": maximum_delay,
        "fifo_delayed_mwh": delayed_work,
        "fifo_delay_mwh_hours": delay_work_hours,
        "conservation_error_mwh": float(abs(np.sum(arrivals) - np.sum(executions))),
        "minimum_cumulative_backlog_mwh": float(np.min(np.cumsum(arrivals - executions))),
    }


def load_input(path):
    frame = pd.read_parquet(path)
    clock = pd.to_datetime(frame["time_utc"], utc=True)
    if clock.isna().any():
        raise ValueError(f"输入气象时间包含缺失值：{path}")
    source_hours = frame["source_hour_index"].to_numpy(float)
    if not np.isfinite(source_hours).all() or (np.diff(source_hours) <= 0).any():
        raise ValueError(f"源小时索引必须有效且严格递增：{path}")
    # 统一使用源小时轴，天气年份映射的跨年回退不改变任务经过时间。
    hours = source_hours - source_hours[0]
    numeric = frame[["base", "rigid", "unknown", "whole", "split", "pue"]].to_numpy(float)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError(f"输入含非有限值或负负荷：{path}")
    return frame, clock, hours, sha256(path)


def check_case(row, root, input_cache, tolerance):
    input_path = root / row.input_path
    hourly_path = root / row.hourly_path
    if input_path not in input_cache:
        input_cache[input_path] = load_input(input_path)
    frame, clock, clock_hours, input_hash = input_cache[input_path]
    hourly = pd.read_parquet(hourly_path)
    actual_clock = pd.to_datetime(hourly["time_utc"], utc=True)
    if len(hourly) != len(frame) or not np.array_equal(actual_clock.to_numpy(), clock.to_numpy()):
        raise ValueError(f"调度输出与输入的真实时间不一致：{hourly_path}")
    columns = [
        "fixed_it_mw", "whole_arrival_mw", "split_arrival_mw", "whole_exec_mw", "split_exec_mw",
        "facility_mw", "pv_mw", "wind_mw", "grid_mw", "charge_mw", "discharge_mw", "curtail_mw",
    ]
    values = hourly[columns].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"调度输出含非有限值：{hourly_path}")
    failures = []
    minimum_value = float(values.min())
    if minimum_value < -tolerance:
        failures.append(f"负功率 {minimum_value}")
    if row.status != "optimal":
        failures.append(f"求解状态 {row.status}")
    fixed = (frame.base + frame.rigid + frame.unknown).to_numpy(float)
    source_error = max(
        float(np.max(np.abs(hourly.fixed_it_mw.to_numpy() - fixed))),
        float(np.max(np.abs(hourly.whole_arrival_mw.to_numpy() - frame.whole.to_numpy()))),
        float(np.max(np.abs(hourly.split_arrival_mw.to_numpy() - frame.split.to_numpy()))),
    )
    if source_error > tolerance:
        failures.append(f"输入层差异 {source_error}")
    records = {}
    for category, deadline in (("whole", float(row.whole_h)), ("split", float(row.split_h))):
        arrival = frame[category].to_numpy(float)
        execution = hourly[f"{category}_exec_mw"].to_numpy(float)
        record = fifo_check(arrival, execution, clock_hours, deadline, tolerance)
        records[category] = record
        for metric in ("early_execution_mwh", "late_execution_mwh", "unserved_mwh", "conservation_error_mwh"):
            if record[metric] > tolerance:
                failures.append(f"{category}.{metric}={record[metric]}")
        if record["minimum_cumulative_backlog_mwh"] < -tolerance:
            failures.append(f"{category} 累计执行超过到达量")
    task_peak = float((frame.rigid + frame.unknown + frame.whole + frame.split).max())
    executed_task = frame.rigid.to_numpy() + frame.unknown.to_numpy() + hourly.whole_exec_mw.to_numpy() + hourly.split_exec_mw.to_numpy()
    cap_error = max(0.0, float(np.max(executed_task - task_peak - float(row.extra_task_mw))))
    idle = float(row.extra_idle_mwh) / len(frame)
    expected_facility = frame.pue.to_numpy() * (fixed + hourly.whole_exec_mw.to_numpy() + hourly.split_exec_mw.to_numpy() + idle)
    facility_error = float(np.max(np.abs(hourly.facility_mw.to_numpy() - expected_facility)))
    supply = hourly.pv_mw + hourly.wind_mw - hourly.charge_mw - hourly.curtail_mw + hourly.grid_mw + hourly.discharge_mw
    balance_error = float(np.max(np.abs(supply.to_numpy() - hourly.facility_mw.to_numpy())))
    grid_total = float(hourly.grid_mw.sum())
    facility_total = float(hourly.facility_mw.sum())
    alpha_error = 0.0 if pd.isna(row.alpha) else max(0.0, grid_total - (1.0 - float(row.alpha)) * facility_total)
    for label, error in (("任务功率容量", cap_error), ("设施功率", facility_error), ("供需平衡", balance_error), ("窗口CFE电量", alpha_error)):
        if error > tolerance:
            failures.append(f"{label}残差={error}")
    return {
        "city": row.city, "scenario": row.scenario, "alpha": None if pd.isna(row.alpha) else float(row.alpha),
        "flex": row.flex, "card_price": str(row.card_price), "supply": row.supply,
        "input_path": str(row.input_path), "hourly_path": str(row.hourly_path),
        "input_sha256": input_hash, "hourly_sha256": sha256(hourly_path),
        "hours": len(frame), "internal_gap_hours": float(clock_hours[-1] - clock_hours[0] + 1 - len(frame)),
        "whole_deadline_hours": float(row.whole_h), "split_deadline_hours": float(row.split_h),
        "extra_task_mw": float(row.extra_task_mw), "dynamic_task_capacity_mw": task_peak + float(row.extra_task_mw),
        "source_layer_error_mw": source_error, "capacity_violation_mw": cap_error,
        "facility_power_error_mw": facility_error, "energy_balance_error_mw": balance_error,
        "window_cfe_violation_mwh": alpha_error, "minimum_power_mw": minimum_value,
        "fifo": records, "passed": not failures, "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    if args.tolerance <= 0:
        raise ValueError("数值容差必须为正")
    index = pd.read_csv(args.index)
    case_keys = ["city", "scenario", "alpha", "flex", "card_price", "supply"]
    if index.empty or index.duplicated(case_keys).any() or index[["input_path", "hourly_path"]].isna().any().any():
        raise ValueError("调度索引为空、案例键重复或输入输出路径缺失")
    cache = {}
    report = {
        "method": "按源小时索引保留的真实时间间距FIFO重分配执行量，独立检查聚合任务时限、功率容量、服务量及能源平衡",
        "boundary": "容量为任务动态功率代理；不验证逐卡拓扑、不可拆分作业连续性或真实SLA授权",
        "absolute_tolerance_mw_or_mwh": args.tolerance,
        "index_sha256": sha256(args.index), "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in index.itertuples(index=False):
        result = check_case(row, args.index.parent, cache, args.tolerance)
        report["cases"].append(result)
    # 完整列出不合格算例，便于仅重算这些真实输入；读入和结构错误仍立即报错。
    failed_cases = [case for case in report["cases"] if not case["passed"]]
    report["passed"] = not failed_cases
    report["failed_case_count"] = len(failed_cases)
    report["case_count"] = len(report["cases"])
    report["input_count"] = len(cache)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "case_count": report["case_count"],
        "failed_case_count": len(failed_cases), "input_count": len(cache), "output": str(args.output)}, ensure_ascii=False))
    if failed_cases:
        raise AssertionError(f"{len(failed_cases)}个调度案例未通过验证，详见{args.output}")


if __name__ == "__main__":
    main()
