import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stream_asi_185days import KEYS, member


def verify(run_dir, hours):
    pinned = json.loads((run_dir / "objects.json").read_text("utf-8"))
    results = []
    for index in hours:
        checkpoint = json.loads((run_dir / "checkpoints" / f"hour-{index:05d}.json").read_text("utf-8"))
        day, hour = divmod(index, 24)
        payload, metadata = member("pod_hourly", f"asi_opensource_pod_hourly/day={day}/hour={hour:02d}/part-000.parquet", pinned)
        if metadata["sha256"] != checkpoint["pod"]["sha256"]:
            raise ValueError("独立读取的原始成员 SHA 与检查点不同")
        frame = pq.read_table(io.BytesIO(payload), columns=KEYS + ["pod_id", "workload_id", "gpu_request", "used_gpu_hours", "cpu_request_cores"]).to_pandas()
        expected = frame.groupby(KEYS, dropna=False).agg(
            requested_gpu=("gpu_request", "sum"), used_gpu_hours=("used_gpu_hours", "sum"),
            requested_cpu_cores=("cpu_request_cores", "sum"), pod_rows=("pod_id", "size"),
            pod_count=("pod_id", "nunique"), workload_count=("workload_id", "nunique")).reset_index()
        actual = pd.DataFrame(checkpoint["groups"])
        columns = KEYS + ["requested_gpu", "used_gpu_hours", "requested_cpu_cores", "pod_rows", "pod_count", "workload_count"]
        pd.testing.assert_frame_equal(actual[columns].sort_values(KEYS).reset_index(drop=True),
            expected[columns].sort_values(KEYS).reset_index(drop=True), check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-8)
        positive = frame[frame.gpu_request > 0]
        gpu = positive.groupby(KEYS, dropna=False).agg(gpu_requesting_pods=("pod_id", "nunique"),
            gpu_requesting_workloads=("workload_id", "nunique"), gpu_requesting_rows=("pod_id", "size")).reset_index()
        joined = actual.merge(gpu, on=KEYS, how="left", suffixes=("_arrow", "_pandas"))
        for name in ["gpu_requesting_pods", "gpu_requesting_workloads", "gpu_requesting_rows"]:
            np.testing.assert_array_equal(joined[name + "_arrow"], joined[name + "_pandas"].fillna(0))
        assert frame.pod_id.nunique() == checkpoint["pod"]["distinct_pods"]
        assert frame.workload_id.nunique() == checkpoint["pod"]["distinct_workloads"]
        assert sum(row["observations"] for row in checkpoint["pod_size_histogram"]) == len(frame)
        workload_count = len(positive[positive.workload_id.notna()].groupby(KEYS + ["workload_id"], dropna=False))
        assert sum(row["observations"] for row in checkpoint["workload_size_histogram"]) == workload_count
        results.append({"hour_index": index, "day": day, "hour": hour, "rows": len(frame), "sha256": metadata["sha256"],
                        "resource_sums": "passed", "distinct_counts": "passed", "scale_histograms": "passed",
                        "raw_bytes": metadata["bytes"], "raw_disk_bytes_written": 0})
    record = {"checked_at": datetime.now(timezone.utc).isoformat(), "method": "real member reread; independent pandas versus Arrow aggregation",
              "samples": results}
    (run_dir / "independent-sample-validation.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="独立重读真实成员，以 pandas 核对 Arrow 资源汇总、精确去重数与规模直方图")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--hours", type=int, nargs="+", default=[0, 4439])
    args = parser.parse_args()
    verify(args.run_dir, args.hours)
