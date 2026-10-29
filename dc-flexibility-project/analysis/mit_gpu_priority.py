import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3

import pandas as pd

from mit_gpu_common import atomic_json, identity, load_jobs, process_lock, utc


def window_bounds(start, end, timezone='UTC'):
    begin, finish = pd.Timestamp(start), pd.Timestamp(end)
    begin = begin.tz_localize(timezone) if begin.tzinfo is None else begin
    finish = finish.tz_localize(timezone) if finish.tzinfo is None else finish
    if finish <= begin or begin != begin.floor('h') or finish != finish.floor('h'):
        raise ValueError('优先窗口必须是递增的完整小时边界')
    return begin.tz_convert('UTC'), finish.tz_convert('UTC')


def overlaps(job, begin, finish):
    return job is not None and job['start'] > 0 and job['end'] > job['start'] and job['start'] < finish.timestamp() and job['end'] > begin.timestamp()


def read_snapshot(root):
    db = sqlite3.connect((Path(root) / 'state.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    with db:
        objects = [dict(row) for row in db.execute('SELECT * FROM objects ORDER BY seq')]
        settings = dict(db.execute('SELECT name,value FROM settings'))
    db.close()
    return objects, settings


def candidates(root, output, timezone='UTC'):
    jobs = load_jobs(root)
    objects, _ = read_snapshot(root)
    available = defaultdict(set)
    for obj in objects:
        job_id, node, _, state = identity(obj['key'], jobs)
        if state == 'matched':
            available[job_id].add(node)
    records = []
    for month in range(4, 10):
        start = pd.Timestamp(year=2021, month=month, day=1)
        end = start + pd.offsets.MonthBegin(1)
        begin, finish = window_bounds(start, end, timezone)
        assigned = covered = suspended = 0.0
        days, gpu_days, invalid, empty_nodes = set(), set(), 0, 0
        selected_jobs = set()
        for job_id, job in jobs.items():
            if begin.timestamp() <= job['start'] < finish.timestamp() and job['end'] <= job['start']:
                invalid += 1
            if not overlaps(job, begin, finish):
                continue
            for day in pd.date_range(pd.Timestamp(max(job['start'], begin.timestamp()), unit='s', tz='UTC').tz_convert(timezone).normalize(),
                                     pd.Timestamp(min(job['end'], finish.timestamp()) - 0.000001, unit='s', tz='UTC').tz_convert(timezone).normalize()):
                days.add(day.date().isoformat())
                if job['allocated_gpus']:
                    gpu_days.add(day.date().isoformat())
            if not job['allocated_gpus']:
                continue
            selected_jobs.add(job_id)
            hours = (min(job['end'], finish.timestamp()) - max(job['start'], begin.timestamp())) / 3600
            weight = hours * job['allocated_gpus']
            assigned += weight
            if job['suspended']:
                suspended += weight
            if not job['nodes']:
                empty_nodes += 1
            else:
                covered += weight * len(available[job_id]) / len(job['nodes'])
        selected = [o for o in objects if overlaps(identity(o['key'], jobs)[2], begin, finish)]
        counts = Counter(o['status'] for o in selected)
        records.append({'month': str(start)[:7], 'timezone': timezone, 'start_utc': begin.isoformat(),
            'end_utc': finish.isoformat(), 'calendar_days': int((end-start).days), 'job_active_days': len(days),
            'gpu_job_active_days': len(gpu_days), 'gpu_jobs': len(selected_jobs), 'allocated_gpu_hours': assigned,
            'file_covered_allocated_gpu_hours': covered, 'file_coverage': covered / assigned if assigned else None,
            'invalid_started_job_intervals': invalid, 'gpu_jobs_without_nodes': empty_nodes,
            'suspended_allocated_gpu_hours': suspended, 'objects': len(selected),
            'bytes': sum(o['bytes'] for o in selected), 'completed_objects': counts['done'],
            'pending_bytes': sum(o['bytes'] for o in selected if o['status']=='pending')})
    eligible = [r for r in records if r['job_active_days'] == r['calendar_days'] and r['gpu_job_active_days'] == r['calendar_days'] and r['gpu_jobs_without_nodes'] == 0]
    if not eligible:
        raise ValueError('候选月没有满足逐日作业记录与节点检查的月份')
    chosen = max(eligible, key=lambda r: (r['file_coverage'], -r['invalid_started_job_intervals'], -r['suspended_allocated_gpu_hours']))
    report = {'created_at': utc(), 'chosen_month': chosen['month'], 'timezone': timezone,
        'selection_rule': '逐日存在有效作业及 GPU 作业、GPU 作业节点非空；按分配 GPU 小时文件覆盖率降序，再按异常区间和暂停时长升序',
        'coverage_method': '月内截取分配时长×GPU数×匹配监控文件节点数/作业节点数；假定节点均分GPU，仅表示文件可获得性',
        'completeness_limit': '逐日活动、有效区间及节点检查不能证明 Slurm 发布记录无遗漏；遥测有效时长覆盖率须另行计算',
        'references_sha256': {name: hashlib.sha256((Path(root)/'reference'/name).read_bytes()).hexdigest()
                              for name in ('slurm-log.csv', 'tres-mapping.txt')}, 'candidates': records}
    atomic_json(output / 'month-selection.json', report)
    pd.DataFrame(records).to_csv(output / 'month-candidates.csv', index=False)
    return report


def configure(db, jobs, start, end, timezone='UTC'):
    begin, finish = window_bounds(start, end, timezone)
    window = {'start_utc': begin.isoformat(), 'end_utc': finish.isoformat(), 'timezone': timezone}
    selected = [row['seq'] for row in db.execute('SELECT seq,key FROM objects ORDER BY seq')
                if overlaps(identity(row['key'], jobs)[2], begin, finish)]
    if not selected:
        raise ValueError('优先窗口没有相关监控对象')
    with db:
        db.execute('UPDATE objects SET priority=0 WHERE priority<>0')
        db.executemany('UPDATE objects SET priority=1 WHERE seq=?', [(seq,) for seq in selected])
        db.execute("INSERT OR REPLACE INTO settings(name,value) VALUES('priority_window',?)", (json.dumps(window),))
    return window


def priority_progress(db, rate=0):
    row = db.execute("SELECT value FROM settings WHERE name='priority_window'").fetchone()
    if row is None:
        return None
    window = json.loads(row[0])
    counts = {r['status']: (r['n'], r['b']) for r in db.execute(
        'SELECT status,COUNT(*) n,COALESCE(SUM(bytes),0) b FROM objects WHERE priority=1 GROUP BY status')}
    total = sum(v[0] for v in counts.values())
    pending_bytes = counts.get('pending', (0, 0))[1]
    failed = counts.get('failed', (0, 0))[0]
    eta = pending_bytes / rate if rate > 0 and not failed else (0 if not pending_bytes and not failed else None)
    return {**window, 'completed': counts.get('done', (0, 0))[0], 'total': total,
        'failed': failed, 'pending': counts.get('pending', (0, 0))[0], 'pending_bytes': pending_bytes,
        'total_bytes': sum(v[1] for v in counts.values()), 'completed_bytes': counts.get('done', (0, 0))[1],
        'complete': total > 0 and counts.get('done', (0, 0))[0] == total,
        'eta_seconds': eta, 'estimated_completion_utc': (pd.Timestamp.now(tz='UTC') + pd.Timedelta(seconds=eta)).isoformat() if eta is not None else None,
        'eta_method': '待持久化字节数/本轮完整对象有效吞吐；包含提交等待，吞吐为零或存在失败时不估计'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timezone', default='UTC')
    args = parser.parse_args()
    output = args.output or args.root / 'analysis/priority-month'
    output.mkdir(parents=True, exist_ok=True)
    report = candidates(args.root, output, args.timezone)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
