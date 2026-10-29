import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from run_eight_city_flex_value import DEFAULT_CONFIG, PROJECT, load_input, write_json


def main():
    parser = argparse.ArgumentParser(description='Linux服务器独立后台控制器：等待输入→全年核验→主组合→其余组合')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--pilot-workers', type=int, default=8)
    parser.add_argument('--city', nargs='+')
    parser.add_argument('--scenario', nargs='+')
    args = parser.parse_args()
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    if args.workers < 2 or args.workers > 16:
        raise ValueError('后台总进程数必须位于2至16')
    root = PROJECT / 'data/processed/eight-city-flex-value-20261029'
    output = PROJECT / 'results/eight-city-flex-value-20261029'
    root.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    status_path = root / 'controller-status.json'
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding='utf-8'))
        if previous['pid'] != os.getpid() and psutil.pid_exists(previous['pid']):
            previous_process = psutil.Process(previous['pid'])
            if str(PROJECT) in ' '.join(previous_process.cmdline()):
                raise RuntimeError(f"当前项目已有运行中的控制器PID={previous['pid']}，请查看其进度")

    def state(stage, **extra):
        entry = {'utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(), 'stage': stage, **extra}
        write_json(root / 'controller-status.json', entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    while True:
        ready = [load_input(args.input_dir, 'phoenix', scenario, config) is not None for scenario in ('A-HPC', 'D-TRN')]
        if all(ready):
            break
        state('waiting_pilot_inputs', pilot_ready=ready)
        time.sleep(30)
    state('annual_validation')
    validation = [sys.executable, str(PROJECT / 'analysis/validate_eight_city_flex_value.py'),
        '--config', str(args.config), '--input-dir', str(args.input_dir), '--annual',
        '--workers', str(args.pilot_workers), '--output-dir', str(output), '--details-dir', str(root)]
    legacy = PROJECT / '.codex-temp/legacy-global-grid-flex-solver.py'
    if legacy.exists():
        validation += ['--legacy-solver', str(legacy)]
    subprocess.run(validation, check=True)
    subprocess.run([sys.executable, str(PROJECT / 'analysis/validate_eight_city_physical.py'),
                    '--config', str(args.config), '--input-dir', str(args.input_dir)], check=True)
    gate = output / 'validation.json'
    base = [sys.executable, str(PROJECT / 'analysis/run_eight_city_flex_value.py'),
        '--config', str(args.config), '--input-dir', str(args.input_dir), '--output-dir', str(root),
        '--gate', str(gate), '--wait-inputs']
    if args.city:
        base += ['--city', *args.city]
    if args.scenario:
        base += ['--scenario', *args.scenario]
    physical_workers = max(1, args.workers // 4)
    economic_workers = args.workers - physical_workers
    economic = subprocess.Popen(base + ['--tier', 'main', '--workers', str(economic_workers)])
    physical = subprocess.Popen(base + ['--tier', 'all', '--physical', '--workers', str(physical_workers)])
    state('main_and_physical', economic_pid=economic.pid, physical_pid=physical.pid,
          economic_workers=economic_workers, physical_workers=physical_workers)
    report = [sys.executable, str(PROJECT / 'analysis/report_eight_city_flex_value.py'), '--config', str(args.config)]
    while economic.poll() is None or physical.poll() is None:
        subprocess.run(report, check=True)
        time.sleep(60)
    economic_code, physical_code = economic.returncode, physical.returncode
    if economic_code or physical_code:
        state('failed', economic_returncode=economic_code, physical_returncode=physical_code)
        raise RuntimeError('主批次存在失败，逐算例已保存；排除原因后同命令续跑')
    state('remaining_queue')
    remaining = subprocess.Popen(base + ['--tier', 'remaining', '--workers', str(args.workers)])
    state('remaining_queue', remaining_pid=remaining.pid)
    while remaining.poll() is None:
        subprocess.run(report, check=True)
        time.sleep(60)
    if remaining.returncode:
        raise RuntimeError('后续队列存在失败，检查日志并按相同命令续跑')
    subprocess.run(report, check=True)
    state('completed')


if __name__ == '__main__':
    main()
