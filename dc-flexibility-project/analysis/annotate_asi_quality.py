import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def annotate(directory):
    manifest_path = directory / "inf-source-manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    raw_path = directory / "inf-hourly.csv"
    expected = manifest["raw_input_sha256"] if "raw_input_sha256" in manifest else manifest["input_sha256"]
    if checksum(raw_path) != expected:
        raise ValueError("原始小时汇总 SHA 不匹配")
    raw = pd.read_csv(raw_path, float_precision="round_trip")
    hourly = pd.read_csv(directory / "hourly-total.csv", float_precision="round_trip")
    mask = (hourly.used_gpu_hours.eq(0) & hourly.requested_gpu.eq(0) & hourly.requested_cpu_cores.eq(0)
            & hourly.pod_rows.gt(0) & hourly.inventory_gpu.gt(0))
    indices = hourly.loc[mask, "hour_index"].astype(int).tolist()
    annotated = raw.copy()
    annotated["source_quality_flag"] = np.where(annotated.hour_index.isin(indices),
        "all_resources_zero_with_pod_inventory", "observed")
    annotated_path = directory / "inf-hourly-quality.csv"
    temporary = annotated_path.with_suffix(".csv.part")
    annotated.to_csv(temporary, index=False)
    temporary.replace(annotated_path)
    reread = pd.read_csv(annotated_path, float_precision="round_trip")
    pd.testing.assert_frame_equal(raw, reread[raw.columns], check_exact=True)
    hourly.loc[mask].to_csv(directory / "quality-hour-audit.csv", index=False)
    ratios = {}
    for prefix in ["online", "offline", "combined"]:
        values = raw[f"{prefix}_used_gpu_hours"]
        positive = values[values > 0]
        ratios[prefix] = {"positive_observed_hour_min": float(positive.min()),
                         "positive_observed_hour_max": float(positive.max()),
                         "positive_observed_hour_peak_valley": float(positive.max() / positive.min())}
    quality = {"checked_at": datetime.now(timezone.utc).isoformat(), "source_hours": len(raw),
               "unresolved_source_hour_indices": indices, "unresolved_hours": len(indices),
               "raw_input_file": raw_path.name, "raw_input_sha256": checksum(raw_path),
               "annotated_file": annotated_path.name, "annotated_sha256": checksum(annotated_path),
               "raw_values_preserved_exactly": True, "imputed_hours": 0,
               "interpretation": "全源申请与归属资源同时为零而 pod 与库存仍非零；保留原值，未确认真实停机",
               "positive_observations": ratios, "code_sha256": checksum(Path(__file__))}
    quality_path = directory / "quality-summary.json"
    atomic_json(quality_path, quality)
    manifest.update(input_file=annotated_path.name, input_sha256=checksum(annotated_path),
                    raw_input_file=raw_path.name, raw_input_sha256=checksum(raw_path),
                    quality_report_sha256=checksum(quality_path), unresolved_source_hour_indices=indices)
    atomic_json(manifest_path, manifest)
    print(json.dumps(quality, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="以完整汇总标记全源共同零观测；保留原值并提供质量增强的 INF 输入")
    parser.add_argument("--directory", type=Path, required=True)
    annotate(parser.parse_args().directory)
