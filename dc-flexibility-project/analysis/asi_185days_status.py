import argparse
import json
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只读显示 ASI 后台处理与汇总验证进度")
    parser.add_argument("--project", type=Path, default=Path("/data3/dataset/ljt/global-grid-flexibility-20261023"))
    args = parser.parse_args()
    directory = args.project / "results/asi-185day-workload-types-20261031"
    processing = json.loads((directory / "status.json").read_text("utf-8"))
    path = directory / "postprocess-status.json"
    postprocessing = json.loads(path.read_text("utf-8")) if path.exists() else {}
    result = {"processing_state": processing["state"], "completed_hours": processing["completed_hours"],
              "pid": processing["pid"], "workers": processing["workers"], "reused_hours": processing["reused_hours"],
              "target_hours": 4440, "updated_at": processing.get("updated_at"),
              "elapsed_seconds": processing.get("elapsed_this_run_seconds"), "jobs_complete": processing.get("jobs_complete"),
              "postprocessing_state": postprocessing.get("state"), "postprocessing_stage": postprocessing.get("stage"),
              "processing_error": processing.get("error"), "postprocessing_error": postprocessing.get("error")}
    last = processing.get("last_completed", {})
    if "index" in last:
        checkpoint = json.loads((directory / "checkpoints" / f"hour-{last['index']:05d}.json").read_text("utf-8"))
        result.update(last_hour_index=last["index"], last_hour_seconds=last["seconds"],
                      last_pod_rows=checkpoint["pod"]["rows"], last_member_bytes=checkpoint["pod"]["bytes"],
                      last_retry_counter=checkpoint["retry_events_total"])
    print(json.dumps(result, ensure_ascii=False))
