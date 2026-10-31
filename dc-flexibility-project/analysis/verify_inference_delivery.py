from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

project = Path(__file__).resolve().parents[1]
output = project / "results/inference-data-alternatives-20261031"
metadata = project / "data/metadata/inference-alternatives-20261031"
server_root = "/data3/dataset/ljt/global-grid-flexibility-20261023"
command = f"{server_root}/.venv/bin/python - --raw-dir {server_root}/data/raw/inference-alternatives-20261031"
storage = json.loads(subprocess.check_output(["ssh", "pku-test", command],
                    input=(project / "analysis/audit_inference_server_storage.py").read_bytes()))
(metadata / "server-storage-verification.json").write_text(json.dumps(storage, ensure_ascii=False, indent=2), encoding="utf-8")
manifest = json.loads((metadata / "server-download-manifest.json").read_text("utf-8"))[:2]
extra = json.loads((output / "additional-source-audit.json").read_text("utf-8"))
for item in extra["files"]:
    if item["name"].startswith("chutes-"):
        manifest.append({"dataset": item["name"], "url": item["url"], "bytes": item["bytes"],
                         "sha256": item["sha256"], "local_path": f"{storage['raw_directory']}/{item['name']}.parquet",
                         "rows": item["rows"], "schema": item["columns"]})
lookup = {Path(item["local_path"]).name: item for item in manifest}
for item in storage["files"]:
    expected = lookup[item["name"]]
    assert (item["bytes"], item["sha256"]) == (expected["bytes"], expected["sha256"]), item["name"]
(metadata / "server-download-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
server_copy = project / ".codex-temp/server-readback"
server_copy.mkdir(parents=True, exist_ok=True)
names = ["request-hourly.csv", "request-daily.csv", "request-diurnal.csv", "request-mod7.csv", "request-summary.json"]
for name in names:
    subprocess.run(["scp", f"pku-test:{server_root}/results/inference-alternatives-20261031-burst121/{name}",
                    str(server_copy / name)], check=True)
    if name.endswith("csv"):
        pd.testing.assert_frame_equal(pd.read_csv(output / name), pd.read_csv(server_copy / name),
                                      check_exact=False, rtol=1e-12, atol=1e-8)
local_summary = json.loads((output / "request-summary.json").read_text("utf-8"))
server_summary = json.loads((server_copy / "request-summary.json").read_text("utf-8"))
pd.testing.assert_frame_equal(pd.DataFrame(local_summary["profiles"]), pd.DataFrame(server_summary["profiles"]),
                              check_exact=False, rtol=1e-12, atol=1e-8)
assert local_summary["profiles"][0]["complete_hours"] == 2856
assert len(pd.read_csv(output / "asi-hourly.csv")) == 48
summary = json.loads((output / "asi-summary.json").read_text("utf-8"))
asi = pd.read_csv(output / "asi-hourly.csv")
assert np.isclose(asi.used_gpu_hours.sum(), sum(d["gpu_hours"] for d in summary["days"]))
assert len(asi[asi.attributed_share_of_inventory > 1]) == 4
inference = pd.read_csv(output / "asi-inference-hourly.csv")
assert len(inference) == 48
assert np.allclose(inference.inference_gpu_hours, inference.online_inference + inference.offline_inference)
assert np.isclose(inference.inference_gpu_hours.sum(), sum(d["inference_gpu_hours"] for d in summary["days"]))
for path in [*output.glob("*.json"), *metadata.glob("*.json")]:
    json.loads(path.read_text("utf-8"))
week = pd.read_csv(output / "chutes-hour-of-week.csv")
day = pd.read_csv(output / "chutes-daily-requests.csv")
tokens = pd.read_csv(output / "chutes-daily-tokens.csv")
week_hour = week.groupby("hour").cnt.sum()
week_day = week.groupby("dow").cnt.sum()
chutes = {"hour_bins": len(week), "daily_bins": len(day),
          "diurnal_request_peak_valley": float(week_hour.max() / week_hour.min()),
          "diurnal_peak_hour": int(week_hour.idxmax()), "diurnal_valley_hour": int(week_hour.idxmin()),
          "relative_weekday_request_peak_valley": float(week_day.max() / week_day.min()),
          "daily_request_peak_valley": float(day.cnt.max() / day.cnt.min()),
          "daily_token_peak_valley": float(tokens.tokens.max() / tokens.tokens.min()),
          "daily_request_sum": int(day.cnt.sum()),
          "basis": "author aggregates; no chronological hourly reconstruction; normalized timestamps"}
(output / "chutes-summary.json").write_text(json.dumps(chutes, ensure_ascii=False, indent=2), encoding="utf-8")
validation = {"checked_at": datetime.now(timezone.utc).isoformat(), "server_file_count": len(manifest),
              "persisted_bytes": storage["persisted_bytes"], "free_bytes": storage["free_bytes"],
              "local_server_csv_readback": names[:-1], "local_server_profiles_equal": True,
              "request_hours": 2856, "asi_hours": 48, "asi_inference_hours": 48,
              "asi_over_inventory_hours_preserved": 4,
              "json_parse": True, "input": "real downloaded data and real SSH readback; no synthetic inputs"}
(output / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"validation": validation, "chutes": chutes}, ensure_ascii=False, indent=2))
