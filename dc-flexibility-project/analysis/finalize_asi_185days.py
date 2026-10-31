import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from stream_asi_185days import write_json


def finalize(project):
    code = Path(__file__).resolve().parent
    output = project / "results/asi-185day-workload-types-20261031"
    with (output / "finalize.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"state": "waiting", "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(), "stages": []}
        write_json(output / "postprocess-status.json", state)
        try:
            while True:
                processing = json.loads((output / "status.json").read_text("utf-8"))
                if processing["state"] == "complete":
                    break
                if processing["state"] == "failed":
                    raise RuntimeError(processing["error"])
                time.sleep(30)
            stages = [
                ("summarize", "summarize_asi_185days.py", ["--run-dir", str(output), "--output", str(output), "--baseline", str(code / "baseline")]),
                ("independent_samples", "verify_asi_stream_sample.py", ["--run-dir", str(output)]),
                ("annual_resource_replay", "build_asi_inference_annual.py", ["--project", str(project), "--year", "2025"]),
                ("verify", "verify_asi_185days.py", ["--project", str(project), "--run-dir", str(output)]),
            ]
            for stage, script, arguments in stages:
                state.update(state="running", stage=stage)
                write_json(output / "postprocess-status.json", state)
                before = time.perf_counter()
                subprocess.run([sys.executable, str(code / script), *arguments], check=True)
                state["stages"].append({"stage": stage, "seconds": time.perf_counter() - before,
                                         "script_sha256": hashlib.sha256((code / script).read_bytes()).hexdigest()})
                write_json(output / "postprocess-status.json", state)
            state.update(state="complete", finished_at=datetime.now(timezone.utc).isoformat())
            write_json(output / "postprocess-status.json", state)
        except BaseException as error:
            state.update(state="failed", error=repr(error), traceback=traceback.format_exc())
            write_json(output / "postprocess-status.json", state)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在独立后台会话等待全量完成，然后生成并验证汇总与 INF 年度资源重放")
    parser.add_argument("--project", type=Path, required=True)
    finalize(parser.parse_args().project)
