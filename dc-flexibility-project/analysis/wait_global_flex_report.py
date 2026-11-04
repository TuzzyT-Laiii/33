import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time


def save(path, state, **fields):
    value = {'utc': datetime.now(timezone.utc).isoformat(), 'state': state, **fields}
    temporary = path.with_suffix('.json.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('global-flex-econ-20261103.json'))
    parser.add_argument('--input', default='full')
    parser.add_argument('--output', default='report-full')
    parser.add_argument('--country-audit', type=Path)
    parser.add_argument('--status-root', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text('utf-8'))
    run = Path(config['paths']['run_root'])
    status_root = args.status_root or run
    state_file = status_root / 'report-progress.json'
    lock = (status_root / 'report-controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if state_file.exists():
        previous = json.loads(state_file.read_text('utf-8'))
        if previous['state'] == 'complete' and previous['input'] == args.input and previous['output'] == args.output:
            print('指定全量报告已完成', flush=True)
            return
    progress_path = run / args.input / 'progress.json'
    while True:
        progress = json.loads(progress_path.read_text('utf-8'))
        if progress['scope'] != 'all':
            raise ValueError('自动汇总只接受全量运行')
        if progress['state'] == 'complete':
            assert progress['completed'] == progress['total']
            if args.country_audit:
                country = json.loads((args.country_audit / 'progress.json').read_text('utf-8'))
                if country['state'] == 'complete':
                    assert country['completed'] == country['total']
                    break
            else:
                break
        save(state_file, 'waiting', input=args.input, output=args.output,
             solve_state=progress['state'], completed=progress['completed'], total=progress['total'])
        time.sleep(30)
    save(state_file, 'running', input=args.input, output=args.output)
    started = time.monotonic()
    command = [sys.executable, str(Path(__file__).with_name('report_global_flex_econ.py')),
               '--config', str(args.config), '--input', args.input, '--output', str(run / args.output), '--maps']
    if args.country_audit:
        command.extend(['--country-audit', str(args.country_audit)])
    with (status_root / 'logs' / 'report-full.log').open('a') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        save(state_file, 'error', input=args.input, output=args.output, returncode=result.returncode)
        raise RuntimeError('全量汇总失败，请检查logs/report-full.log')
    save(state_file, 'complete', input=args.input, output=args.output, elapsed_seconds=time.monotonic() - started)
    with (status_root / 'WORK_STATE.md').open('a', encoding='utf-8') as stream:
        stream.write(f'\n- 自动汇总完成 {datetime.now(timezone.utc).isoformat()}：{run / args.output}；统计、来源清单与四幅地图已生成，仍需最终数值检查、图件检查及PR交付。\n')
    print('全量报告已完成', flush=True)


if __name__ == '__main__':
    main()
