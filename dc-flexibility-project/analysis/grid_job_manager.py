import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path("/data3/dataset/ljt/global-grid-flexibility-20261023")
JOBS = {
    "era5": ["acquire_era5_grid_weather.py", "--root", str(ROOT), "--workers", "4",
             "--transport", "curl", "--reserve-gb", "100"],
    "era5-land": ["acquire_era5_land_weather.py", "--root", str(ROOT), "--workers", "4",
                  "--reserve-gb", "100", "--timeout-seconds", "180", "--attempts", "5"],
    "worldcover": ["prepare_worldcover_grid.py", "--root", str(ROOT), "--workers", "3",
                   "--prefetched-only", "--watch-prefetched"],
    "buildability": ["prepare_buildable_grid.py", "--root", str(ROOT), "--workers", "8", "--download-workers", "16",
                     "--maximum-slope", "5", "--minimum-area", "1"],
    "era5-eligible": ["acquire_eligible_era5_weather.py", "--root", str(ROOT), "--workers", "4"],
    "buildability-stream": ["stream_buildable_weather.py", "--root", str(ROOT), "--workers", "4"],
}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def job_path(name):
    return ROOT / "status" / f"job-{name}.json"


def check_download_policy(name):
    policy_path = ROOT / "status/era5-download-policy.json"
    if name in ("era5", "era5-land") and policy_path.exists():
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        if policy["mode"] == "candidate_spatial_only":
            raise RuntimeError(f"当前天气获取范围限定为用地候选区域，完整全球及全陆地任务不启动：{policy_path}")


def worker(name):
    import fcntl

    with (ROOT / f".job-{name}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_download_policy(name)
        os.nice(10)
        command = [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "analysis" / JOBS[name][0]), *JOBS[name][1:]]
        state = {"job": name, "status": "running", "manager_pid": os.getpid(), "started_utc": now(), "command": command}
        with (ROOT / f'.job-{name}-start.lock').open('a+') as start_lock:
            fcntl.flock(start_lock, fcntl.LOCK_EX)
            write_state(job_path(name), state)
        while True:
            process = subprocess.Popen(command, cwd=ROOT)
            state.update(status='running', worker_pid=process.pid)
            write_state(job_path(name), state)
            result = process.wait()
            with (ROOT / f'.job-{name}-start.lock').open('a+') as start_lock:
                fcntl.flock(start_lock, fcntl.LOCK_EX)
                resume = ROOT / f'.job-{name}-resume-request.json'
                if name == 'era5-eligible' and result == 75 and resume.exists():
                    resume.unlink()
                    progress = json.loads((ROOT / 'status/era5-eligible-acquisition.json').read_text())
                    proofs = [ROOT / 'data/metadata/buildability-20261025/weather-cds-repairs' / (label + '.json')
                              for label in progress['remaining_blocks']]
                    if any(path.exists() and json.loads(path.read_text())['status'] == 'verified' for path in proofs):
                        continue
                if resume.exists():
                    resume.unlink()
                status = 'waiting_for_repairs' if name == 'era5-eligible' and result == 75 else ('complete' if result == 0 else 'failed')
                fcntl.flock(lock, fcntl.LOCK_UN)
                state.update(status=status, finished_utc=now(), returncode=result)
                write_state(job_path(name), state)
                break
        sys.exit(result)


def start_locked(name, ensure_running=False):
    check_download_policy(name)
    state_path = job_path(name)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        pid = state.get("manager_pid")
        if state["status"] in ('starting', 'running') and pid and Path(f"/proc/{pid}/cmdline").is_file():
            running_command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode()
            if str(Path(__file__).resolve()) in running_command and f"--worker {name}" in running_command:
                if ensure_running:
                    write_state(ROOT / f'.job-{name}-resume-request.json', {'requested_utc': now()})
                    print(json.dumps({'job': name, 'status': 'already_running', 'manager_pid': pid}))
                    return
                raise RuntimeError(f"本任务已在运行：{name}, PID={pid}")
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    log = ROOT / "logs" / f"{name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    with log.open("ab", buffering=0) as stream:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", name],
                                   cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, stdin=subprocess.DEVNULL, env=env)
    write_state(state_path, {'job': name, 'status': 'starting', 'manager_pid': process.pid, 'started_utc': now()})
    print(json.dumps({"job": name, "manager_pid": process.pid, "log": str(log)}, ensure_ascii=False), flush=True)


def start(name, ensure_running=False):
    import fcntl

    with (ROOT / f'.job-{name}-start.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        start_locked(name, ensure_running)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", choices=JOBS)
    parser.add_argument("--worker", choices=JOBS)
    parser.add_argument("--status", action="store_true")
    parser.add_argument('--ensure-running', action='store_true')
    args = parser.parse_args()
    if args.start:
        start(args.start, args.ensure_running)
    elif args.worker:
        worker(args.worker)
    elif args.status:
        result = {name: json.loads(job_path(name).read_text(encoding="utf-8")) if job_path(name).exists() else {"status": "not_started"} for name in JOBS}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        parser.error("需要指定启动、工作进程或状态操作")


if __name__ == "__main__":
    main()
