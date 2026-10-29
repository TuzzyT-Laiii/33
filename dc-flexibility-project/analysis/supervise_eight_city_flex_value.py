import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from eight_city_onsite_solver import sha256
from run_eight_city_flex_value import DEFAULT_CONFIG, PROJECT, write_json


def main():
    parser = argparse.ArgumentParser(description='接续已经运行的主经济/物理队列，刷新结果并启动后续组合')
    parser.add_argument('--economic-pid', type=int, required=True)
    parser.add_argument('--physical-pid', type=int, required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    root = PROJECT / 'data/processed/eight-city-flex-value-20261029'
    gate = PROJECT / 'results/eight-city-flex-value-20261029/validation.json'
    validation = json.loads(gate.read_text(encoding='utf-8'))
    if not validation['annual_real_input_gate'] or validation['config_sha256'] != sha256(args.config):
        raise ValueError('当前配置的全年核验尚未通过')
    processes = [psutil.Process(pid) for pid in (args.economic_pid, args.physical_pid)]
    for process in processes:
        if str(PROJECT / 'analysis/run_eight_city_flex_value.py') not in process.cmdline():
            raise ValueError(f'目标进程不属于当前项目：{process.pid}')
    def state(stage, **extra):
        write_json(root / 'controller-status.json', {'utc':datetime.now(timezone.utc).isoformat(),
            'pid':os.getpid(), 'stage':stage, 'economic_pid':args.economic_pid,
            'physical_pid':args.physical_pid, 'economic_workers':12, 'physical_workers':4, **extra})
    report = [sys.executable, str(PROJECT / 'analysis/report_eight_city_flex_value.py'), '--config', str(args.config)]
    state('main_and_physical')
    while any(process.is_running() and process.status() != psutil.STATUS_ZOMBIE for process in processes):
        subprocess.run(report, check=True)
        time.sleep(60)
    for label in ('economic-main','physical-all'):
        status = json.loads((root / f'queue-state-{label}.json').read_text(encoding='utf-8'))
        if status['status'] != 'completed':
            raise RuntimeError(f'队列尚未全部完成：{label}, {status}')
    remaining = subprocess.Popen([sys.executable, str(PROJECT / 'analysis/run_eight_city_flex_value.py'),
        '--config', str(args.config), '--input-dir', str(args.input_dir), '--output-dir', str(root),
        '--gate', str(gate), '--tier', 'remaining', '--workers', '16', '--wait-inputs'])
    state('remaining_queue', remaining_pid=remaining.pid)
    while remaining.poll() is None:
        subprocess.run(report, check=True)
        time.sleep(60)
    if remaining.returncode:
        raise RuntimeError('后续队列存在未完成算例，请检查日志')
    subprocess.run(report, check=True)
    state('completed')


if __name__ == '__main__':
    main()
