import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


CODE = ["stream_asi_185days.py", "probe_asi_inference_range.py", "asi_185days_status.py", "summarize_asi_185days.py",
        "annotate_asi_quality.py", "asi_inference_profiles.py", "annual_task_profiles.py", "build_task_annual_loads.py",
        "build_asi_inference_annual.py", "finalize_asi_185days.py", "verify_asi_stream_sample.py", "verify_asi_185days.py",
        "export_asi_summary.py", "audit_asi_execution.py", "verify_asi_delivery.py", "launch_asi_185days.sh"]


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="核验服务器汇总读回、代码指纹与 Git 暂存字节，确保跨系统 SHA 保持一致")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-index", action="store_true")
    args = parser.parse_args()
    directory = args.project / "results/asi-185day-workload-types-20261031"
    manifest = json.loads((directory / "delivery-manifest.json").read_text("utf-8"))
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True, encoding="utf-8").strip()).resolve()
    for item in manifest["files"]:
        path = directory / item["file"]
        assert path.stat().st_size == item["bytes"] and checksum(path) == item["sha256"], item["file"]
        if args.check_index:
            relative = path.resolve().relative_to(repo).as_posix()
            data = subprocess.check_output(["git", "show", ":" + relative])
            assert hashlib.sha256(data).hexdigest() == item["sha256"], relative
    code = {}
    for name in CODE:
        path = args.project / "analysis" / name
        if path.suffix == ".py":
            compile(path.read_text("utf-8"), str(path), "exec")
        code[name] = checksum(path)
    for path in directory.glob("*.json"):
        json.loads(path.read_text("utf-8"))
    source = json.loads((directory / "inf-source-manifest.json").read_text("utf-8"))
    assert checksum(directory / source["input_file"]) == source["input_sha256"]
    assert checksum(directory / source["raw_input_file"]) == source["raw_input_sha256"]
    assert checksum(directory / "quality-summary.json") == source["quality_report_sha256"]
    validation = json.loads((directory / "validation.json").read_text("utf-8"))
    assert validation["code_sha256"] == code["verify_asi_185days.py"]
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "status": "passed",
              "server_summary_files": len(manifest["files"]), "bytes": sum(item["bytes"] for item in manifest["files"]),
              "server_local_sha256_equal": True, "git_index_bytes_equal": args.check_index,
              "python_compile": True, "json_parse": True, "inf_source_hashes": True,
              "code_sha256": code,
              "historical_code_versions": ["8ce7238", "83ea075"],
              "scope": "原始处理与首次年度重放的代码指纹见 processing-audit；质量传播后的当前源码指纹见本记录"}
    (directory / "delivery-validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "code_sha256"}, ensure_ascii=False, indent=2))
