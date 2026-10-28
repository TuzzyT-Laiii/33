import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from mit_gpu_common import BASE, atomic_json, database, get_retry, identity, load_jobs, utc

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--files', type=int, default=100)
parser.add_argument('--concurrency', type=int, default=16, choices=range(1,17))
parser.add_argument('--proxy', default='http://127.0.0.1:7897')
args = parser.parse_args()
root = args.root
jobs = load_jobs(root)
db = database(root)
objects = [dict(r) for r in db.execute('SELECT * FROM objects ORDER BY seq LIMIT ?', (args.files,))]
db.close()
proxy = args.proxy

def check(obj):
    headers = {'Range': f'bytes=0-{min(4095,obj["bytes"]-1)}', 'If-Match': '"'+obj['etag']+'"'}
    first = get_retry(BASE+obj['key'], proxy, headers=headers)
    first.raise_for_status()
    reader = csv.reader(io.StringIO(first.text))
    columns = next(reader)
    initial = dict(zip(columns, next(reader)))
    begin = max(0,obj['bytes']-4096)
    headers['Range'] = f'bytes={begin}-{obj["bytes"]-1}'
    last = get_retry(BASE+obj['key'], proxy, headers=headers)
    last.raise_for_status()
    if first.status_code != 206 or last.status_code != 206:
        raise ValueError('边界核验 Range 状态不符')
    tail = last.text
    if begin:
        tail = tail.partition('\n')[2]
    final = dict(zip(columns, list(csv.reader(io.StringIO(tail)))[-1]))
    job_id,node,job,status = identity(obj['key'],jobs)
    if job is None:
        return {'seq':obj['seq'],'status':status}
    zone = ZoneInfo('America/New_York')
    shift_start = -datetime.fromtimestamp(job['start'],timezone.utc).astimezone(zone).utcoffset().total_seconds()
    shift_end = -datetime.fromtimestamp(job['end'],timezone.utc).astimezone(zone).utcoffset().total_seconds()
    raw_start,raw_end=float(initial['timestamp']),float(final['timestamp'])
    result = {'seq':obj['seq'],'start_shift_s':shift_start,'end_shift_s':shift_end,
              'raw_start_residual_s':raw_start-job['start'],'raw_end_residual_s':raw_end-job['end'],
              'corrected_start_residual_s':raw_start+shift_start-job['start'],
              'corrected_end_residual_s':raw_end+shift_end-job['end'],
              'dst_transition':shift_start!=shift_end}
    print(json.dumps(result),flush=True)
    return result

with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
    records=list(pool.map(check,objects))
report={'checked_at':utc(),'files':len(records),'method':'S3首尾Range + 标准csv + IANA America/New_York + Slurm epoch',
        'records':records,'max_abs_corrected_start_s':max(abs(r['corrected_start_residual_s']) for r in records),
        'max_abs_corrected_end_s':max(abs(r['corrected_end_residual_s']) for r in records)}
atomic_json(root/'clock-boundary-audit.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
