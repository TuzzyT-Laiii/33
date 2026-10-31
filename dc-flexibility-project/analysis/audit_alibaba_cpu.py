import argparse
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd

from acquire_alibaba_cpu import EXPECTED, now, sha256
from annual_task_profiles import replay_hourly
from compute_load.model import ResourceProfile


SCHEMAS = {
    "machine_meta": "machine_id timestamp failure_domain_1 failure_domain_2 cpu_num mem_size status".split(),
    "machine_usage": "machine_id timestamp cpu mem mem_gps mkpi net_in net_out disk".split(),
    "container_meta": "container_id machine_id timestamp app_du status cpu_request cpu_limit mem_size".split(),
    "batch_task": "task_name instance_num job_name task_type status start end plan_cpu plan_mem".split(),
}
END = 8 * 86400
HOURS = END // 3600
ABSENT = -32768
MISSING = -32767


@contextmanager
def csv_stream(raw, name):
    with tarfile.open(raw / f"{name}.tar.gz", "r:gz") as archive:
        members = 0
        for member in archive:
            if member.isfile():
                if Path(member.name).name != f"{name}.csv" or members:
                    raise ValueError(f"非预期归档成员：{member.name}")
                members += 1
                with archive.extractfile(member) as stream:
                    yield stream
        if members != 1:
            raise ValueError(f"归档缺少唯一 CSV：{name}")


def read_table(raw, name):
    with csv_stream(raw, name) as stream:
        return pd.read_csv(stream, names=SCHEMAS[name])


def intervals_by_hour(start, end, weight, hours=HOURS):
    start, end, weight = np.asarray(start), np.asarray(end), np.asarray(weight)
    good = np.isfinite(start) & np.isfinite(end) & np.isfinite(weight) & (end > start) & (weight >= 0)
    start, end, weight = np.maximum(start[good], 0), np.minimum(end[good], hours * 3600), weight[good]
    good = end > start
    start, end, weight = start[good], end[good], weight[good]
    # 对累计资源秒做差分，精确保留跨小时的区间重叠。
    first = (start // 3600).astype(int)
    last = (np.nextafter(end, -np.inf) // 3600).astype(int)
    result = np.zeros(hours)
    same = first == last
    np.add.at(result, first[same], (end[same] - start[same]) * weight[same] / 3600)
    a, b, w, f, l = start[~same], end[~same], weight[~same], first[~same], last[~same]
    np.add.at(result, f, ((f + 1) * 3600 - a) * w / 3600)
    np.add.at(result, l, (b - l * 3600) * w / 3600)
    difference = np.zeros(hours + 1)
    np.add.at(difference, f + 1, w)
    np.add.at(difference, l, -w)
    return result + np.cumsum(difference)[:hours]


def machine_hourly(cpu, mem, cores, hold):
    occupied, exposure, memory = (np.zeros((len(cores), HOURS)) for _ in range(3))
    for i in range(len(cores)):
        ticks = np.flatnonzero(cpu[i] != ABSENT)
        times = ticks.astype(float) * 10
        ends = np.minimum(np.r_[times[1:], END], times + hold)
        cv, mv = cpu[i, ticks], mem[i, ticks]
        good = (cv >= 0) & (cv <= 100)
        occupied[i] = intervals_by_hour(times[good], ends[good], cv[good] / 100 * cores[i])
        exposure[i] = intervals_by_hour(times[good], ends[good], np.full(good.sum(), cores[i]))
        mgood = (mv >= 0) & (mv <= 100)
        memory[i] = intervals_by_hour(times[mgood], ends[mgood], mv[mgood])
    return occupied, exposure, memory


def summarize(frame, value):
    values = frame[value].to_numpy(float)
    return {"hours": len(values), "mean": float(values.mean()), "minimum": float(values.min()),
            "maximum": float(values.max()), "peak_to_trough": float(values.max() / values.min()) if values.min() > 0 else None,
            "cv": float(values.std() / values.mean()),
            "p05": float(np.quantile(values, .05)), "p50": float(np.quantile(values, .5)),
            "p95": float(np.quantile(values, .95))}


def build(raw, output, lisa_path, hold=300, minimum_coverage=.95):
    raw, output = Path(raw), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, expected in EXPECTED.items():
        if sha256(raw / f"{name}.tar.gz") != expected:
            raise ValueError(f"输入 SHA-256 不符：{name}")
    meta = read_table(raw, "machine_meta")
    if meta.groupby("machine_id").cpu_num.nunique().ne(1).any():
        raise ValueError("机器核数随时间变化，须按事件连接")
    inventory = meta.groupby("machine_id", sort=True).cpu_num.first()
    if inventory.le(0).any():
        raise ValueError("非正机器核数")
    ids = inventory.index
    cores = inventory.to_numpy(float)
    index = dict(zip(ids, range(len(ids))))
    cpu = np.full((len(ids), END // 10 + 1), ABSENT, dtype=np.int16)
    mem = np.full_like(cpu, ABSENT)
    hist = np.zeros((HOURS + 1, 101), dtype=np.int64)
    counts, invalid_cpu, invalid_mem, invalid_disk = Counter(), Counter(), Counter(), Counter()
    minimum_time, maximum_time = END, 0
    with csv_stream(raw, "machine_usage") as stream:
        for chunk in pd.read_csv(stream, names=SCHEMAS["machine_usage"],
                                 usecols=["machine_id", "timestamp", "cpu", "mem", "disk"], chunksize=1000000):
            counts["raw_rows"] += len(chunk)
            if chunk.machine_id.map(index).isna().any():
                raise ValueError("监测出现无元数据的机器")
            times = chunk.timestamp.to_numpy(float)
            if not np.isfinite(times).all() or (times < 0).any() or (times > END).any() or (times % 10 != 0).any():
                raise ValueError("时间戳超出八天或不在实测 10 秒时间格上")
            minimum_time, maximum_time = min(minimum_time, times.min()), max(maximum_time, times.max())
            machine = chunk.machine_id.map(index).to_numpy(int)
            slot = (times / 10).astype(int)
            values = []
            for column, invalid in (("cpu", invalid_cpu), ("mem", invalid_mem)):
                v = chunk[column].to_numpy(float)
                good = np.isfinite(v) & (v >= 0) & (v <= 100)
                if (v[np.isfinite(v)] % 1 != 0).any():
                    raise ValueError("CPU/内存百分数出现非整数，需修改存储类型")
                invalid.update(chunk.loc[~good, column].fillna("missing").astype(str))
                values.append(np.where(np.isfinite(v), v, MISSING).astype(np.int16))
            disk = chunk.disk
            invalid_disk.update(disk.loc[~disk.between(0, 100)].fillna("missing").astype(str))
            positions = machine * cpu.shape[1] + slot
            unique, take = np.unique(positions, return_index=True)
            if len(unique) != len(positions):
                # 同时间重复值只保留一次，冲突即停止。
                for v in values:
                    lookup = pd.Series(v, index=positions).groupby(level=0).nunique()
                    if lookup.gt(1).any():
                        raise ValueError("同机器同时间存在冲突记录")
            previous_cpu, previous_mem = cpu.ravel()[unique], mem.ravel()[unique]
            fresh = previous_cpu == ABSENT
            if ((~fresh) & ((previous_cpu != values[0][take]) | (previous_mem != values[1][take]))).any():
                raise ValueError("跨批次重复记录冲突")
            chosen = take[fresh]
            counts["duplicate_rows"] += len(chunk) - len(chosen)
            cpu.ravel()[positions[chosen]] = values[0][chosen]
            mem.ravel()[positions[chosen]] = values[1][chosen]
            valid = (values[0][chosen] >= 0) & (values[0][chosen] <= 100)
            np.add.at(hist, ((times[chosen][valid] // 3600).astype(int), values[0][chosen][valid]), 1)
            if counts["raw_rows"] % 10000000 == 0:
                print(json.dumps(dict(counts)), flush=True)
    print("machine_usage 读取完成", flush=True)
    occupied, exposure, memory = machine_hourly(cpu, mem, cores, hold)
    machine_hours = pd.DataFrame({"machine_id": np.repeat(ids.to_numpy(), HOURS),
        "relative_hour": np.tile(np.arange(HOURS), len(ids)), "cpu_num": np.repeat(cores, HOURS),
        "occupied_core_hours": occupied.ravel(), "observed_core_hours": exposure.ravel()})
    machine_hours.to_parquet(output / "machine-hourly.parquet", index=False)
    hourly = pd.DataFrame({"relative_hour": np.arange(HOURS), "relative_day": np.arange(HOURS)//24,
                           "hour_of_day": np.arange(HOURS)%24})
    hourly["inventory_machines"] = len(ids)
    hourly["inventory_cores"] = cores.sum()
    hourly["observed_machines"] = (exposure > 0).sum(axis=0)
    hourly["observed_core_hours"] = exposure.sum(axis=0)
    hourly["coverage_fraction"] = hourly.observed_core_hours / cores.sum()
    hourly["occupied_core_hours"] = occupied.sum(axis=0)
    hourly["cpu_util_time_weighted_pct"] = hourly.occupied_core_hours / hourly.observed_core_hours * 100
    hourly["resource_fraction_inventory"] = hourly.occupied_core_hours / cores.sum()
    hourly["resource_fraction"] = hourly.cpu_util_time_weighted_pct / 100
    hourly["sample_count"] = hist[:HOURS].sum(axis=1)
    hourly["cpu_util_sample_mean_pct"] = hist[:HOURS] @ np.arange(101) / hourly.sample_count
    per_machine_mean = np.divide(occupied, exposure, out=np.full_like(occupied, np.nan), where=exposure > 0) * 100
    hourly["cpu_util_machine_mean_pct"] = np.nanmean(per_machine_mean, axis=0)
    for q in (.05, .25, .5, .75, .95):
        hourly[f"machine_hour_cpu_p{int(q*100):02d}_pct"] = np.nanquantile(per_machine_mean, q, axis=0)
    hourly["timeline_full_hour"] = (hourly.relative_hour * 3600 >= minimum_time) & ((hourly.relative_hour + 1)*3600 <= maximum_time)
    hourly["coverage_pass"] = hourly.coverage_fraction >= minimum_coverage
    daily = hourly.groupby("relative_day").agg(hours=("relative_hour", "size"),
        timeline_full_hours=("timeline_full_hour", "sum"), coverage_pass_hours=("coverage_pass", "sum"),
        minimum_coverage=("coverage_fraction", "min"), mean_coverage=("coverage_fraction", "mean"),
        occupied_core_hours=("occupied_core_hours", "sum"), mean_resource_fraction=("resource_fraction", "mean"))
    daily["complete_day"] = daily.hours.eq(24) & daily.timeline_full_hours.eq(24) & daily.coverage_pass_hours.eq(24)
    selected = hourly.loc[hourly.relative_day.isin(daily.index[daily.complete_day])].copy()
    daily.to_csv(output / "daily-coverage.csv")
    if selected.empty or not selected.relative_hour.diff().iloc[1:].eq(1).all():
        raise ValueError("没有满足覆盖阈值的连续完整日窗口；请检查 daily-coverage.csv")
    hourly["complete_day"] = hourly.relative_day.isin(daily.index[daily.complete_day])
    # 同时保存样本均值、时间加权均值与覆盖率，避免把缺测直接视为空闲。
    hourly.to_csv(output / "alibaba-hourly-cpu-demand.csv", index=False)
    daily.to_csv(output / "daily-coverage.csv")
    pd.DataFrame(hist.T, index=pd.Index(range(101), name="cpu_util_pct"), columns=[f"hour_{i}" for i in range(HOURS+1)]).to_csv(output / "cpu-sample-histogram.csv")
    pd.DataFrame({"machine_id": ids, "cpu_num": cores}).to_csv(output / "machine-inventory.csv", index=False)
    selected.to_csv(output / "complete-window.csv", index=False)
    sensitivity = []
    for cap in (60, 300, 900):
        oc, ex, _ = (occupied, exposure, memory) if cap == hold else machine_hourly(cpu, mem, cores, cap)
        sensitivity.append({"maximum_hold_seconds": cap, "occupied_core_hours": oc.sum(),
                            "coverage_fraction": ex.sum()/(cores.sum()*HOURS),
                            "mean_util_observed_pct": oc.sum()/ex.sum()*100,
                            "minimum_hour_coverage": (ex.sum(axis=0)/cores.sum()).min()})
    pd.DataFrame(sensitivity).to_csv(output / "gap-sensitivity.csv", index=False)
    containers = read_table(raw, "container_meta")
    container_counts = {"rows": len(containers), "containers": containers.container_id.nunique(),
                        "status_counts": containers.status.value_counts().to_dict()}
    if containers.duplicated(["container_id", "timestamp"]).any():
        raise ValueError("容器事件有同时间重复，须核对顺序")
    containers = containers.sort_values(["container_id", "timestamp"])
    containers["end"] = containers.groupby("container_id").timestamp.shift(-1).fillna(END)
    started = containers.status.eq("started") & containers.cpu_request.ge(0)
    online = intervals_by_hour(containers.loc[started, "timestamp"], containers.loc[started, "end"], containers.loc[started, "cpu_request"]/100)
    batch = np.zeros(HOURS)
    batch_counts = Counter()
    with csv_stream(raw, "batch_task") as stream:
        for chunk in pd.read_csv(stream, names=SCHEMAS["batch_task"], chunksize=500000):
            batch_counts["rows"] += len(chunk)
            batch_counts.update({f"status_{key}": int(value) for key, value in chunk.status.value_counts().items()})
            valid = chunk.start.ge(0) & chunk.end.gt(chunk.start) & chunk.plan_cpu.ge(0) & chunk.instance_num.gt(0)
            batch_counts["positive_envelope_rows"] += int(valid.sum())
            batch += intervals_by_hour(chunk.loc[valid, "start"], chunk.loc[valid, "end"],
                                       chunk.loc[valid, "instance_num"] * chunk.loc[valid, "plan_cpu"]/100)
    shares = hourly[["relative_hour", "relative_day", "complete_day"]].copy()
    shares["online_requested_core_hours"] = online
    shares["batch_planned_envelope_core_hours"] = batch
    shares["online_share_of_request_envelope"] = online / (online + batch)
    shares.to_csv(output / "online-batch-request-envelope.csv", index=False)
    source = {"profile": ResourceProfile("cpu", selected.resource_fraction.to_numpy(), {"source": "alibaba_cpu_2018", "scenario": "A-CLD"}),
              "clock": np.array([f"relative_day={d};hour={h:02d}" for d, h in zip(selected.relative_day, selected.hour_of_day)]),
              "metadata": {"days": len(selected)//24, "status": "short_window_replay_proxy",
                           "normalization_basis": "按有效核秒加权的 CPU 利用率；缺测时段不作为空闲，小时覆盖率另列"}}
    annual = {}
    for year in (2025, 2024):
        times = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", freq="h", inclusive="left", tz="UTC")
        profile, mapping = replay_hourly(source, times)
        mapping.to_csv(output / f"A-CLD-{year}-replay-map.csv", index=False)
        annual[str(year)] = profile.provenance
    lisa = pd.read_parquet(lisa_path)
    lisa = lisa.loc[lisa.hardware_group.eq("cpu_partition")].copy()
    lisa["day"] = lisa.source_time.dt.floor("D")
    complete = lisa.groupby("day").allocated_core_hours.count().eq(24)
    lisa = lisa.loc[lisa.day.isin(complete.index[complete])]
    lisa["resource_fraction"] = lisa.allocated_core_hours/lisa.allocated_core_hours.max()
    lisa["hour_of_day"] = lisa.source_time.dt.hour
    lisa[["source_time", "allocated_core_hours", "resource_fraction", "hour_of_day"]].to_csv(output / "lisa-cpu-complete-hourly.csv", index=False)
    comparisons = []
    for label, frame, column in (("A-CLD", selected, "resource_fraction"), ("A-HPC", lisa, "resource_fraction")):
        comparisons.append({"scenario": label, **summarize(frame, column)})
        frame.groupby("hour_of_day")[column].agg(["mean", "min", "max"]).to_csv(output / f"{label}-intraday.csv")
    pd.DataFrame(comparisons).to_csv(output / "shape-comparison.csv", index=False)
    selected.groupby("relative_day").resource_fraction.mean().to_csv(output / "relative-day-shape.csv")
    total_hist = hist.sum(axis=0)
    quantiles = {f"p{int(q*100):02d}": int(np.searchsorted(np.cumsum(total_hist), q*total_hist.sum(), side="left")) for q in (.05,.25,.5,.75,.95)}
    summary = {"built_at_utc": now(), "counts": dict(counts), "machine_meta_rows": len(meta),
        "machines": len(ids), "inventory_cores": float(cores.sum()), "cpu_num_distribution": inventory.value_counts().to_dict(),
        "observed_timestamp_min": float(minimum_time), "observed_timestamp_max": float(maximum_time),
        "invalid_cpu_raw_rows": dict(invalid_cpu), "invalid_memory_raw_rows": dict(invalid_mem),
        "invalid_disk_raw_rows": dict(invalid_disk), "valid_cpu_unique_samples": int(total_hist.sum()),
        "sample_cpu_quantiles_pct": quantiles, "complete_days": daily.index[daily.complete_day].tolist(),
        "complete_hours": len(selected), "maximum_hold_seconds": hold, "minimum_hour_core_coverage": minimum_coverage,
        "containers": container_counts, "batch_tasks": dict(batch_counts), "annual_replay": annual,
        "lisa_input_sha256": sha256(lisa_path), "lisa_input": str(lisa_path),
        "observed_occupied_core_hours_complete_window": float(selected.occupied_core_hours.sum()),
        "online_actual_cpu_share": None, "batch_actual_cpu_share": None,
        "share_limit": "容器请求与 batch 任务计划包络仅为资源配置诊断；缺少容器监测和逐实例执行记录，不能拆分实际 CPU 占用或推断柔性份额",
        "outputs_sha256": {p.name: sha256(p) for p in output.glob("*.csv")},
        "code_sha256": {name: sha256(Path(__file__).with_name(name)) for name in ("audit_alibaba_cpu.py", "acquire_alibaba_cpu.py", "annual_task_profiles.py")}}
    (output / "source-audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="流式读取 Alibaba 2018 四表，生成 A-CLD 小时需求及完整窗口重放")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lisa-hourly", type=Path, required=True)
    parser.add_argument("--maximum-hold-seconds", type=int, default=300)
    parser.add_argument("--minimum-hour-core-coverage", type=float, default=.95)
    args = parser.parse_args()
    build(args.raw, args.output, args.lisa_hourly, args.maximum_hold_seconds, args.minimum_hour_core_coverage)
