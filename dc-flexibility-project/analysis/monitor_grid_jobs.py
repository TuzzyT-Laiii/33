import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import pandas as pd
import psutil


DEFAULT_ROOT = Path("/data3/dataset/ljt/global-grid-flexibility-20261023")
JOBS = {
    "era5-land": {"script": "acquire_era5_land_weather.py", "target": 26305,
                  "progress": "status/era5-land-acquisition.json", "worker_lock": ".era5-land-acquisition.lock"},
    "worldcover": {"script": "prepare_worldcover_grid.py", "target": 2651,
                   "progress": "data/metadata/worldcover-2021/run-status.json", "worker_lock": None},
    "buildability": {"script": "prepare_buildable_grid.py", "target": 1,
                     "progress": "status/buildability-20261025.json", "worker_lock": None},
    "era5-eligible": {"script": "acquire_eligible_era5_weather.py", "target": 1,
                      "progress": "status/era5-eligible-acquisition.json", "worker_lock": ".era5-eligible-acquisition.lock"},
    "buildability-stream": {"script": "stream_buildable_weather.py", "target": 1,
                            "progress": "status/buildability-stream.json", "worker_lock": None},
}
RESERVE_BYTES = 100 * 1024 ** 3
MAX_DAILY_ACTIONS = 6
MAX_SAME_ERROR_ACTIONS = 3
STALE_SECONDS = 2 * 60 * 60
SHA = re.compile(r"[0-9a-f]{64}")
UNITS = {
    "2m_temperature": "K", "2m_dewpoint_temperature": "K",
    "100m_u_component_of_wind": "m s**-1", "100m_v_component_of_wind": "m s**-1",
    "surface_pressure": "Pa", "mean_surface_downward_short_wave_radiation_flux": "W m**-2",
    "mean_surface_direct_short_wave_radiation_flux": "W m**-2",
}


def now():
    return datetime.now(timezone.utc)


def timestamp(value):
    if not value:
        return None
    parsed = pd.Timestamp(value)
    if pd.isna(parsed) or parsed.nanosecond != 0:
        raise ValueError(f"监控时间戳无效或含非零纳秒，不能截断: {value}")
    parsed = parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")
    return parsed.to_pydatetime(warn=False)


def read_json(path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"监控输入必须为JSON对象: {path}")
    return value


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + f".{os.getpid()}.partial")
    with partial.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def public_text(value):
    text = str(value)
    text = re.sub(r"(https?://[^\s?'\"]+)\?[^\s'\"]+", r"\1?<redacted>", text)
    text = re.sub(r"(?i)((?:password|token|authorization|secret|api[_-]?key)\s*[:=]\s*)[^\s,;]+", r"\1<redacted>", text)
    return text[-1200:]


def error_fingerprint(category, detail):
    text = public_text(detail).lower()
    text = re.sub(r"https?://\S+", "<url>", text)
    text = re.sub(r"[0-9a-f]{32,}", "<digest>", text)
    text = re.sub(r"\b[ns]\d{2}[ew]\d{3}\b", "<tile>", text)
    text = re.sub(r"\d{4}-\d{2}-\d{2}[t ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+z-]\S*)?", "<time>", text)
    text = re.sub(r"\d+", "<number>", text)
    text = re.sub(r"\s+", " ", text).strip()
    # 网络错误按类别计数，时刻、文件名及传输字节不能形成新的恢复预算。
    normalized = category if category.startswith("network_") else category + ":" + text
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def valid_digests(value):
    return isinstance(value, dict) and set(value) == set(UNITS) and all(isinstance(item, str) and SHA.fullmatch(item) for item in value.values())


def completion_records(root, name):
    if name in ("buildability", "era5-eligible", "buildability-stream"):
        return screened_completion_records(root, name)
    errors = []
    latest = None
    sampled = 0
    directory = root / ("checkpoints/era5-land-hourly" if name == "era5-land" else "data/metadata/worldcover-2021/tiles")
    names = []
    if directory.exists():
        with os.scandir(directory) as entries:
            names = sorted(entry.name for entry in entries if entry.name.endswith(".json"))
    valid_names = []
    expected_hours = {}
    source_tiles = None
    index_sha = None
    if name == "era5-land":
        manifest = read_json(root / "metadata/era5/land-acquisition-manifest.json")
        if manifest is None:
            return {"completed": 0, "latest_completion_utc": None, "errors": ["land_manifest_missing"]}
        if manifest.get("units") != UNITS or manifest.get("total_hours") != JOBS[name]["target"]:
            return {"completed": 0, "latest_completion_utc": None, "errors": ["land_manifest_units_or_target_invalid"]}
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        seen = set()
        for filename in names:
            try:
                hour = datetime.strptime(filename, "%Y%m%dT%H.json").replace(tzinfo=timezone.utc)
                index = int((hour - start).total_seconds() / 3600)
                if hour.strftime("%Y%m%dT%H.json") != filename or not 0 <= index < JOBS[name]["target"] or index in seen:
                    errors.append("invalid_weather_checkpoint_filename:" + filename)
                    continue
                seen.add(index)
                valid_names.append(filename)
                expected_hours[filename] = (hour, index)
            except ValueError:
                errors.append("invalid_weather_checkpoint_filename:" + filename)
    else:
        source_index = read_json(root / "data/raw/worldcover-2021/esa_worldcover_grid.geojson")
        if source_index is not None:
            source_tiles = {feature["properties"]["ll_tile"] for feature in source_index["features"]}
            if len(source_tiles) != JOBS[name]["target"] or len(source_index["features"]) != len(source_tiles):
                errors.append("worldcover_source_index_count_or_uniqueness_invalid")
        index_validation = read_json(root / "data/metadata/worldcover-2021/tile-index-validation.json")
        if index_validation is not None:
            index_sha = index_validation.get("source_sha256")
            if index_validation.get("tile_count") != JOBS[name]["target"] or index_validation.get("duplicate_grid_ids") != 0:
                errors.append("worldcover_source_index_validation_invalid")
        seen = set()
        for filename in names:
            tile = filename[:-5]
            if not re.fullmatch(r"[NS]\d{2}[EW]\d{3}", tile) or tile in seen or (source_tiles is not None and tile not in source_tiles):
                errors.append("invalid_worldcover_checkpoint_filename:" + filename)
                continue
            seen.add(tile)
            valid_names.append(filename)
    sample_names = set()
    if valid_names:
        sample_names.update((valid_names[0], valid_names[len(valid_names) // 2], valid_names[-1]))
        sample_names.update(valid_names[-8:])
    progress = read_json(root / JOBS[name]["progress"]) or {}
    if name == "worldcover" and progress.get("last_completed_tile"):
        reported = progress["last_completed_tile"] + ".json"
        if reported in valid_names:
            position = valid_names.index(reported)
            sample_names.update(valid_names[max(0, position - 3):position + 1])
    if len(sample_names) > 16:
        raise ValueError("每项作业的清单内容抽查超过16份上限")
    for filename in sorted(sample_names):
        sampled += 1
        try:
            record = read_json(directory / filename)
            if name == "era5-land":
                hour, index = expected_hours[filename]
                valid = (
                    record.get("status") == "done" and timestamp(record.get("time_utc")) == hour and record.get("hour_index") == index
                    and record.get("source_metadata_sha256") == manifest.get("source_metadata_sha256")
                    and record.get("land_mask_sha256") == manifest.get("land_mask_sha256")
                    and record.get("coordinate_sha256") == manifest.get("coordinate_sha256")
                    and record.get("units") == UNITS and valid_digests(record.get("selected_values_sha256"))
                    and record.get("selected_values_sha256") == record.get("output_values_sha256")
                    and valid_digests(record.get("full_source_values_sha256"))
                )
                verified = timestamp(record.get("verified_utc"))
            else:
                tile = filename[:-5]
                output = root / "data/interim/worldcover-2021/tiles" / (tile + ".parquet")
                valid = (
                    record.get("tile") == tile and record.get("rows") == 36
                    and record.get("all_native_pixels_accounted") is True and record.get("native_pixel_count") == 36000 * 36000
                    and record.get("native_pixel_count_expected") == 36000 * 36000 and record.get("overview_used") is False
                    and isinstance(record.get("output_sha256"), str) and SHA.fullmatch(record["output_sha256"])
                    and isinstance(record.get("source_sha256"), str) and SHA.fullmatch(record["source_sha256"])
                    and (index_sha is None or record.get("index_sha256") == index_sha)
                    and output.is_file() and output.stat().st_size > 0
                )
                verified = timestamp(record.get("completed_utc"))
            if not valid or verified is None:
                errors.append("invalid_sampled_checkpoint:" + filename)
                continue
            latest = max(latest, verified) if latest else verified
        except (json.JSONDecodeError, ValueError, TypeError, OSError) as error:
            errors.append("sampled_checkpoint_read_error:" + filename + ":" + type(error).__name__)
    completed = len(valid_names)
    if completed > JOBS[name]["target"]:
        errors.append("completed_count_exceeds_fixed_target")
    return {"completed": completed, "latest_completion_utc": latest.isoformat() if latest else None,
            "errors": errors[:20], "error_count": len(errors), "sampled_checkpoint_count": sampled,
            "sampled_checkpoint_files": sorted(sample_names), "all_checkpoint_contents_rechecked": False,
            "count_basis": "枚举完成清单文件名总量，检查全部名称的唯一性及时间范围或来源切片；仅抽查首个、中间、末尾八份及状态报告的最近切片邻近清单。科学值与文件SHA由worker写后核验，本次监控不全量重审、不重写结果",
            "latest_completion_basis": "抽查清单中verified_utc或completed_utc的最大值；与真实清单数量及worker心跳交叉核对"}


def screened_completion_records(root, name):
    progress = read_json(root / JOBS[name]["progress"]) or {}
    finished = progress.get("status") == "complete"
    errors = []
    manifest_name = {"buildability": "selection-manifest.json", "era5-eligible": "weather-acquisition-manifest.json",
                     "buildability-stream": "stream-completion.json"}[name]
    manifest_path = root / "data/metadata/buildability-20261025" / manifest_name
    manifest = read_json(manifest_path)
    if finished:
        if manifest is None:
            errors.append("completed_screened_job_manifest_missing")
        elif name == "buildability":
            if manifest.get("status") != "complete" or not manifest.get("protected_areas_applied") or not manifest.get("slope_applied"):
                errors.append("buildability_conditions_not_complete")
            for output in manifest.get("outputs", {}).values():
                path = Path(output["path"])
                if not path.is_file() or path.stat().st_size != output["bytes"]:
                    errors.append("buildability_output_missing_or_size_changed")
        elif name == "buildability-stream":
            if manifest.get("status") != "complete" or manifest.get("completed_cover_tiles") != manifest.get("total_cover_tiles"):
                errors.append("stream_screening_not_complete")
        elif manifest.get("blocks") != progress.get("completed_blocks") or manifest.get("weather_points") != progress.get("completed_points"):
            errors.append("eligible_weather_completed_counts_differ_from_plan")
    directory = root / ("data/interim/buildability-20261025/tiles" if name in ("buildability", "buildability-stream") else "checkpoints/era5-eligible-blocks")
    files = sorted(directory.glob("*.json")) if directory.exists() else []
    samples = sorted(set(files[:1] + files[-8:] + files[len(files) // 2:len(files) // 2 + 1]))
    latest = None
    for path in samples:
        record = read_json(path)
        completed = timestamp(record.get("completed_utc"))
        output_sha = record.get("output_sha256")
        if completed is None or not isinstance(output_sha, str) or not SHA.fullmatch(output_sha):
            errors.append("screened_checkpoint_invalid:" + path.name)
        else:
            latest = max(latest, completed) if latest else completed
        if name == "era5-eligible" and (record.get("status") != "done" or record.get("hours") != 26305):
            errors.append("eligible_weather_checkpoint_not_complete:" + path.name)
    if finished and name == "era5-eligible" and manifest is not None and len(files) != manifest.get("blocks"):
        errors.append("eligible_weather_checkpoint_count_differ_from_plan")
    return {"completed": int(finished), "target": 1, "formal_completed_blocks": len(files) if name == 'era5-eligible' else None,
            "latest_completion_utc": latest.isoformat() if latest else None,
            "errors": errors, "error_count": len(errors), "sampled_checkpoint_count": len(samples),
            "sampled_checkpoint_files": [path.name for path in samples], "all_checkpoint_contents_rechecked": False,
            "actual_progress": {key: progress.get(key) for key in ("status", "required_dem_tiles", "completed_dem_tiles", "total_cover_tiles", "completed_cover_tiles", "eligible_grids", "total_blocks", "completed_blocks", "total_points", "completed_points", "quarantined_blocks", "quarantined_block_count", "other_blocks_target", "other_blocks_complete") if key in progress},
            "count_basis": "完成计数0或1表示整个流程是否完成；actual_progress保存DEM、土地或天气块实际进度；枚举完成记录并抽查最多10份，不重做全量科研计算"}


def running_processes(root, name):
    script = str(root / "analysis" / JOBS[name]["script"])
    manager = str(root / "analysis/grid_job_manager.py")
    workers = {}
    managers = {}
    denied = 0
    for process in psutil.process_iter(["pid", "ppid", "cmdline", "status", "uids"]):
        try:
            info = process.info
            if info["uids"] is None or info["uids"].real != os.getuid() or info["status"] == psutil.STATUS_ZOMBIE:
                continue
            command = info["cmdline"] or []
            root_position = command.index("--root") + 1 if "--root" in command else len(command)
            worker_position = command.index("--worker") + 1 if "--worker" in command else len(command)
            is_worker = script in command and root_position < len(command) and command[root_position] == str(root)
            is_manager = manager in command and worker_position < len(command) and command[worker_position] == name
            if not is_worker and not is_manager:
                continue
            item = {"pid": info["pid"], "ppid": info["ppid"], "pgid": os.getpgid(info["pid"])}
            (workers if is_worker else managers)[item["pid"]] = item
        except (psutil.NoSuchProcess, ProcessLookupError):
            continue
        except psutil.AccessDenied:
            denied += 1
    # fork后的计算子进程继承cmdline；它们不能被当作重复的主worker。
    main_workers = [item for item in workers.values() if item["ppid"] not in workers]
    return {"workers": main_workers, "worker_descendant_count": len(workers) - len(main_workers),
            "managers": list(managers.values()), "access_denied_count": denied}


def probe_existing_lock(path, enabled):
    if not path.exists():
        return "absent"
    if not enabled:
        return "not_probed_readonly"
    try:
        with path.open("r") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return "held"
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return "free"
    except PermissionError:
        return "permission_denied"


def error_detail(root, name, progress):
    parts = [progress.get("error_type", ""), progress.get("error", "")]
    for failure in progress.get("failed_tiles", []):
        parts.append(failure.get("error", ""))
    if any(parts):
        return public_text("\n".join(str(item) for item in parts if item))
    logs = sorted((path for path in (root / "logs").glob(name + "-*.log")
                   if re.fullmatch(re.escape(name) + r"-\d{8}T\d{6}Z\.log", path.name)), key=lambda path: path.stat().st_mtime)
    if not logs:
        return ""
    with logs[-1].open("rb") as stream:
        stream.seek(max(0, logs[-1].stat().st_size - 32768))
        tail = stream.read().decode("utf-8", errors="replace")
    if "Traceback (most recent call last):" in tail:
        return public_text(tail.rsplit("Traceback (most recent call last):", 1)[-1])
    return ""


def classify_error(detail, progress):
    lowered = detail.lower()
    blocked = ("valueerror", "assertionerror", "importerror", "modulenotfounderror", "permissionerror",
               "nameerror", "keyerror", "typeerror", "indexerror", "attributeerror", "syntaxerror", "filenotfounderror",
               "certificate_verify_failed", "certificate verify failed", "sslcertverificationerror",
               "no space left", "permission denied", "可用磁盘不足", "磁盘不足", "checksum", "校验", "单位", "坐标")
    if any(marker in lowered for marker in blocked) or re.search(r"\bsha(?:-?256)?\b|sha256", lowered):
        return "validation_or_environment_error", False
    if progress.get("status") in ("failed_prefetch_reported_failure", "failed_prefetch_idle_timeout"):
        return "prefetch_dependency_recovered", True
    if any(marker in lowered for marker in ("timeout", "timed out", "超时", "curl获取公开对象失败，exit=28")):
        return "network_timeout", True
    if any(marker in lowered for marker in ("connectionerror", "connection reset", "connection aborted", "connection refused", "clientconnectorerror", "chunkedencodingerror", "remotedisconnected", "sslerror", "temporary failure", "network is unreachable", "curl获取公开对象失败，exit=56", "curl获取公开对象失败，exit=35")):
        return "network_connection", True
    if re.search(r"(?:returned error:\s*|http(?:/[\d.]+)?\s+|status(?:_code)?[=:]\s*b?['\"]?)(?:429|502|503|504)\b|\b(?:429|502|503|504) (?:client|server) error", lowered):
        return "network_http_temporary", True
    if detail:
        return "unknown_code_error", False
    return "worker_exited_without_error_detail", True


def inspect_job(root, name, current, allow_lock_probe):
    progress = read_json(root / JOBS[name]["progress"]) or {}
    manager_state = read_json(root / "status" / ("job-" + name + ".json")) or {}
    records = completion_records(root, name)
    processes = running_processes(root, name)
    locks = {"manager": probe_existing_lock(root / (".job-" + name + ".lock"), allow_lock_probe)}
    if JOBS[name]["worker_lock"]:
        locks["worker"] = probe_existing_lock(root / JOBS[name]["worker_lock"], allow_lock_probe)
    heartbeat = timestamp(progress.get("updated_utc") or progress.get("completed_utc") or progress.get("started_utc"))
    sampled_latest = timestamp(records.get("latest_completion_utc"))
    result = {
        "job": name, "target": JOBS[name]["target"], **records, "progress_status": progress.get("status", "missing"),
        "heartbeat_utc": heartbeat.isoformat() if heartbeat else None,
        "heartbeat_age_seconds": max(0, (current - heartbeat).total_seconds()) if heartbeat else None,
        "sampled_latest_completion_age_seconds": max(0, (current - sampled_latest).total_seconds()) if sampled_latest else None,
        "manager_reported_status": manager_state.get("status", "missing"),
        "manager_reported_pid": manager_state.get("manager_pid"), "worker_reported_pid": manager_state.get("worker_pid"),
        "reported_manager_identity_matches": any(item["pid"] == manager_state.get("manager_pid") for item in processes["managers"]),
        "reported_worker_identity_matches": any(item["pid"] == manager_state.get("worker_pid") for item in processes["workers"]),
        "processes": processes, "locks": locks,
    }
    if name == "worldcover":
        relay = read_json(root / "data/metadata/worldcover-2021/prefetch-status.json") or {}
        relay_time = timestamp(relay.get("updated_utc") or relay.get("completed_utc") or relay.get("started_utc"))
        relay_age = max(0, (current - relay_time).total_seconds()) if relay_time else None
        complete = relay.get("status") == "all_selected_originals_on_server_verified" and relay.get("selected_tiles") == 2651 and relay.get("pending_tiles") == 0
        fresh = relay.get("status") == "running" and relay.get("selected_tiles") == 2651 and relay_age is not None and relay_age <= STALE_SECONDS
        result["prefetch"] = {"status": relay.get("status", "missing"), "heartbeat_utc": relay_time.isoformat() if relay_time else None,
                              "age_seconds": relay_age, "ready": bool(complete or fresh)}
    detail = error_detail(root, name, progress)
    category, recoverable = classify_error(detail, progress)
    result.update(error_category=category, error_recoverable=recoverable, error_detail=detail,
                  error_fingerprint=error_fingerprint(category, detail))
    return result


def decide_job(root, observed, previous, current, free_bytes):
    name = observed["job"]
    metric = 'formal_completed_blocks' if name == 'era5-eligible' else 'completed'
    current_count = observed[metric]
    old_count = previous.get(metric)
    if not isinstance(old_count, int):
        old_count = current_count
    advanced = current_count > old_count
    regression = current_count < old_count
    last_progress = current if advanced else timestamp(previous.get("last_progress_utc")) or current
    if name == 'era5-eligible' and observed.get('latest_completion_utc'):
        last_progress = max(last_progress, timestamp(observed['latest_completion_utc']))
    same_checks = 0 if advanced else previous.get("same_count_checks", -1) + 1
    result = {**observed, "last_progress_utc": last_progress.isoformat(), "same_count_checks": same_checks,
              "progress_since_previous_check": advanced, "restart_allowed": False}
    budget = dict(previous.get("recovery_budget", {}))
    day = current.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d")
    history = budget.get("action_history_utc", [])
    if not history and budget.get("actions_today", 0) and budget.get("last_restart_utc"):
        history = [budget["last_restart_utc"]] * budget["actions_today"]
    history = [value for value in history if timestamp(value) > current - timedelta(hours=24)]
    budget.pop("actions_today", None)
    budget.update(day_asia_shanghai=day, action_history_utc=history, actions_last24h=len(history))
    if advanced:
        budget.update(error_action_counts={}, consecutive_same_error=0, last_error_fingerprint=None)
    budget.setdefault("error_action_counts", {})
    result["recovery_budget"] = budget
    policy_path = root / "status/era5-download-policy.json"
    if name == "era5-land" and policy_path.exists():
        policy = read_json(policy_path)
        if policy["mode"] == "candidate_spatial_only":
            result.update(status="disabled_by_landuse_policy", reason="candidate_spatial_downloads_only", restart_allowed=False)
            return result
    active = bool(observed["processes"]["workers"] or observed["processes"]["managers"] or "held" in observed["locks"].values())
    if observed["errors"] or regression:
        result.update(status="needs_attention", reason="checkpoint_validation_failed" if not regression else "completed_count_regressed")
    elif free_bytes < RESERVE_BYTES:
        result.update(status="needs_attention", reason="free_space_below_100_gib")
    elif "permission_denied" in observed["locks"].values():
        result.update(status="needs_attention", reason="job_lock_permission_denied")
    elif observed["completed"] == observed["target"] and (name != "worldcover" or observed["progress_status"] == "complete"):
        result.update(status="complete", reason="all_required_completion_records_present")
    elif name == "worldcover" and not observed["prefetch"]["ready"]:
        result.update(status="waiting_for_local_prefetch", reason="prefetch_missing_failed_or_older_than_two_hours")
    elif active:
        workers = observed["processes"]["workers"]
        managers = observed["processes"]["managers"]
        group_mismatch = len(workers) == len(managers) == 1 and workers[0]["pgid"] != managers[0]["pgid"]
        if len(workers) > 1 or len(managers) > 1 or group_mismatch:
            result.update(status="needs_attention", reason="active_process_identity_or_group_conflict")
        elif same_checks >= 2 and (current - last_progress).total_seconds() >= STALE_SECONDS and observed["heartbeat_age_seconds"] is not None and observed["heartbeat_age_seconds"] >= STALE_SECONDS and (observed["sampled_latest_completion_age_seconds"] is None or observed["sampled_latest_completion_age_seconds"] >= STALE_SECONDS):
            result.update(status="stalled_needs_attention", reason="heartbeat_and_real_completed_count_stale_for_two_hours")
        elif workers:
            pending = budget.get("last_restart_completed_count") is not None and observed["completed"] <= budget["last_restart_completed_count"]
            if name == 'era5-eligible' and observed.get('formal_completed_blocks', 0) > 0:
                pending = False
            if managers and (not observed["reported_manager_identity_matches"] or not observed["reported_worker_identity_matches"]):
                result.update(status="active_protected", reason="active_processes_do_not_match_stale_reported_pids_no_duplicate_start")
            else:
                result.update(status="recovering" if pending else "running", reason="active_worker_pending_progress" if pending else "active_worker_verified")
        else:
            result.update(status="active_protected", reason="manager_or_job_lock_active_no_duplicate_start")
    elif name == 'era5-eligible' and observed['progress_status'] == 'waiting_for_repairs':
        result.update(status='waiting_for_repairs', reason='explicit_quarantine_awaiting_source_repair')
    elif root != DEFAULT_ROOT:
        result.update(status="needs_attention", reason="job_manager_uses_another_fixed_root")
    elif any(not path.is_file() or not os.access(path, os.R_OK) for path in (root / ".venv/bin/python", root / "analysis/grid_job_manager.py", root / "analysis" / JOBS[name]["script"])):
        result.update(status="needs_attention", reason="required_python_or_script_missing_or_unreadable")
    elif not (root / "logs").is_dir() or not os.access(root / "logs", os.W_OK):
        result.update(status="needs_attention", reason="job_log_directory_missing_or_unwritable")
    elif not observed["error_recoverable"]:
        result.update(status="needs_attention", reason=observed["error_category"])
    elif budget["actions_last24h"] >= MAX_DAILY_ACTIONS:
        result.update(status="needs_attention", reason="rolling_24h_recovery_action_limit_reached")
    elif budget["error_action_counts"].get(observed["error_fingerprint"], 0) >= MAX_SAME_ERROR_ACTIONS:
        result.update(status="needs_attention", reason="same_error_recovery_limit_reached")
    else:
        result.update(status="recovery_available", reason=observed["error_category"], restart_allowed=True)
    return result


def restart_once(root, result, current, persist_before_launch):
    name = result["job"]
    if name not in JOBS or not result["restart_allowed"]:
        raise ValueError("恢复请求超出白名单或不满足恢复条件")
    budget = result["recovery_budget"]
    fingerprint = result["error_fingerprint"]
    budget["action_history_utc"].append(current.isoformat())
    budget["actions_last24h"] = len(budget["action_history_utc"])
    budget["error_action_counts"][fingerprint] = budget["error_action_counts"].get(fingerprint, 0) + 1
    budget.update(last_restart_utc=current.isoformat(), last_restart_completed_count=result["completed"],
                  last_error_fingerprint=fingerprint, consecutive_same_error=budget["error_action_counts"][fingerprint])
    result.update(restart_allowed=False, status="recovery_start_unconfirmed", reason="launcher_started_waiting_for_process_confirmation")
    persist_before_launch()
    try:
        process = subprocess.Popen([sys.executable, str(root / "analysis/grid_job_manager.py"), "--start", name],
                                   cwd=root, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as error:
        result.update(status="needs_attention", reason="job_manager_launcher_failed", restart_error=public_text(error))
        return result
    deadline = time.monotonic() + 10
    launcher_output = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            if process.returncode:
                result.update(status="needs_attention", reason="job_manager_start_failed", restart_error=public_text(stderr.decode("utf-8", errors="replace")))
                return result
            try:
                launcher_output = json.loads(stdout.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                result.update(status="needs_attention", reason="job_manager_launcher_response_invalid", restart_error=public_text(error))
                return result
            break
        time.sleep(0.25)
    if launcher_output is None:
        result.update(status="needs_attention", reason="job_manager_launcher_did_not_confirm_within_ten_seconds", launcher_pid=process.pid)
        return result
    while time.monotonic() < deadline:
        observed = inspect_job(root, name, now(), True)
        managers = observed["processes"]["managers"]
        workers = observed["processes"]["workers"]
        expected_manager = launcher_output.get("manager_pid")
        if len(managers) == len(workers) == 1 and managers[0]["pid"] == expected_manager and managers[0]["pgid"] == workers[0]["pgid"] and "held" in observed["locks"].values():
            progressed = observed["completed"] > result["completed"]
            result.update(observed, status="running" if progressed else "recovering",
                          reason="recovery_progress_verified" if progressed else "new_manager_and_worker_verified_waiting_for_real_progress",
                          recovery_budget=budget, restart_allowed=False)
            if progressed:
                budget.update(error_action_counts={}, consecutive_same_error=0, last_error_fingerprint=None)
                result["last_progress_utc"] = now().isoformat()
                result["same_count_checks"] = 0
            return result
        if not managers and not workers and observed["manager_reported_status"] == "failed":
            result.update(observed, status="needs_attention", reason="restarted_worker_exited_before_confirmation", recovery_budget=budget, restart_allowed=False)
            return result
        time.sleep(0.5)
    result.update(status="needs_attention", reason="restart_process_confirmation_timed_out")
    return result


def run(args):
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"监控项目目录不存在: {root}")
    state_path = root / "status/grid-monitor.json"
    old = read_json(state_path) or {}
    current = now()
    free_bytes = shutil.disk_usage(root).free
    report = {"schema_version": 1, "checked_utc": current.isoformat(), "mode": "recover" if args.recover else "check",
              "root": str(root), "allowed_jobs": list(JOBS), "free_bytes": free_bytes,
              "reserve_bytes": RESERVE_BYTES, "reporting_time_zone": "Asia/Shanghai", "recovery_action_limit_per_24h": MAX_DAILY_ACTIONS,
              "jobs": {name: old.get("jobs", {}).get(name, {}) for name in JOBS}}
    events = []
    for name in JOBS:
        previous = old.get("jobs", {}).get(name, {})
        try:
            observed = inspect_job(root, name, current, args.recover)
            result = decide_job(root, observed, previous, current, free_bytes)
        except (json.JSONDecodeError, ValueError, TypeError, OSError) as error:
            result = {"job": name, "status": "needs_attention", "reason": "monitor_input_read_error", "restart_allowed": False,
                      "error_type": type(error).__name__, "error_detail": public_text(error), "completed": previous.get("completed"),
                      "target": JOBS[name]["target"], "recovery_budget": previous.get("recovery_budget", {})}
        if args.recover and result["restart_allowed"]:
            # 先保存预算，再启动进程；监控被中断时仍能保留此次恢复消耗。
            report["jobs"][name] = result
            budget = result["recovery_budget"]
            action = {"event": "recovery_requested", "job": name, "utc": now().isoformat(), "completed": result["completed"], "error_category": result["error_category"]}
            result = restart_once(root, result, current, lambda: atomic_json(state_path, report))
            action.update(outcome=result["status"], reason=result["reason"], actions_last24h=budget["actions_last24h"])
            events.append(action)
        previous_status = previous.get("status")
        if result["status"] != previous_status and (previous_status is not None or result["status"] not in ("running", "active_protected")):
            events.append({"event": "completion" if result["status"] == "complete" else "status_transition", "job": name,
                           "utc": now().isoformat(), "previous_status": previous_status, "status": result["status"],
                           "reason": result["reason"], "completed": result["completed"], "target": result["target"]})
        report["jobs"][name] = result
    if args.check:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return
    atomic_json(state_path, report)
    if events:
        log = root / "logs/grid-monitor-events.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as stream:
            for event in events:
                line = json.dumps(event, ensure_ascii=False, allow_nan=False)
                stream.write(line + "\n")
                print(line, flush=True)
            stream.flush()
            os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description="每次检查一次全球陆地天气与土地处理任务；只对白名单网络暂态故障或已退出worker执行有限恢复。")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="只读取实际状态并输出JSON；不创建文件、不获取锁、不启动任务")
    mode.add_argument("--recover", action="store_true", help="执行一次自检及有预算限制的恢复；健康稳定时不输出事件")
    args = parser.parse_args()
    if args.check:
        run(args)
        return
    if not args.root.is_dir():
        raise FileNotFoundError(f"监控项目目录不存在: {args.root}")
    with (args.root / ".grid-monitor.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        run(args)


if __name__ == "__main__":
    main()
