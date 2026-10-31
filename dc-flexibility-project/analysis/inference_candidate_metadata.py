from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess

import pandas as pd
import pyarrow.parquet as pq
import requests

project = Path(__file__).resolve().parents[1]
output = project / "results/inference-data-alternatives-20261031"
snapshots = project / ".codex-temp/inference-sources"
snapshots.mkdir(parents=True, exist_ok=True)


def api(endpoint):
    cache = snapshots / ("api-" + hashlib.sha256(endpoint.encode()).hexdigest()[:16] + ".json")
    if cache.exists():
        return json.loads(cache.read_text("utf-8"))
    response = json.loads(subprocess.check_output(["gh", "api", endpoint], text=True, encoding="utf-8"))
    cache.write_text(json.dumps(response), encoding="utf-8")
    return response


result = {"checked_at": datetime.now(timezone.utc).isoformat(), "github": {}, "hf": {}, "files": []}
for repo in ["Azure/AzurePublicDataset", "HarvardMadSys/chutes_workload", "facebookresearch/distributed_traces"]:
    info = api(f"repos/{repo}")
    tree = api(f"repos/{repo}/git/trees/{info['default_branch']}?recursive=1")
    result["github"][repo] = {"sha": tree["sha"], "license": info["license"], "files": [
        {k: f.get(k) for k in ("path", "size", "sha")} for f in tree["tree"] if f["type"] == "blob"]}
for tag in ["dataset-functions-2019", "dataset-functions-blob-2020"]:
    release = api(f"repos/Azure/AzurePublicDataset/releases/tags/{tag}")
    result["github"][tag] = [{k: f.get(k) for k in ("name", "size", "digest", "browser_download_url")}
                               for f in release["assets"]]
for name in ["chutes_metadata", "freeinference_agentic_trace"]:
    response = requests.get(f"https://huggingface.co/api/datasets/harvardMadsys/{name}?blobs=true", timeout=60)
    response.raise_for_status()
    info = response.json()
    result["hf"][name] = {k: info.get(k) for k in ("sha", "usedStorage", "cardData", "siblings")}
urls = {
    "azure-functions-2021": "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/AzureFunctionsInvocationTrace2021.md",
    "azure-blob-2020": "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/AzureFunctionsBlobDataset2020.md",
    "burst-license": "https://raw.githubusercontent.com/HPMLL/BurstGPT/main/LICENSE",
    "pai-license": "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/LICENSE",
    "servegen-code": "https://raw.githubusercontent.com/alibaba/ServeGen/main/servegen/clientpool.py",
    "servegen-construct": "https://raw.githubusercontent.com/alibaba/ServeGen/main/servegen/construct.py",
    "servegen-chunk": "https://raw.githubusercontent.com/alibaba/ServeGen/main/data/language/m-large/chunk-1-trace.csv",
    "chutes-hour-of-week": "https://raw.githubusercontent.com/HarvardMadSys/chutes_workload/main/data/paper/cache/hour_of_week_requests.parquet",
    "chutes-daily-requests": "https://raw.githubusercontent.com/HarvardMadSys/chutes_workload/main/data/paper/cache/daily_requests.parquet",
    "chutes-daily-tokens": "https://raw.githubusercontent.com/HarvardMadSys/chutes_workload/main/data/paper/cache/daily_tokens.parquet",
}
for name, url in urls.items():
    response = requests.get(url, timeout=60)
    record = {"name": name, "url": url, "status": response.status_code, "bytes": len(response.content),
              "sha256": hashlib.sha256(response.content).hexdigest()}
    if response.ok:
        (snapshots / (name + (".parquet" if url.endswith("parquet") else ".txt"))).write_bytes(response.content)
        if url.endswith("parquet"):
            frame = pq.read_table(io.BytesIO(response.content)).to_pandas()
            record.update(rows=len(frame), columns=list(frame.columns))
            frame.to_csv(output / f"{name}.csv", index=False)
    result["files"].append(record)
(output / "additional-source-audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"files": result["files"], "hf": {k: v["usedStorage"] for k, v in result["hf"].items()}}, indent=2))
