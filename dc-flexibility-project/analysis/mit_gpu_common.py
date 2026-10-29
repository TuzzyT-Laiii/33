import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import threading
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


BASE = 'https://mit-supercloud-dataset.s3.us-west-2.amazonaws.com/'
PREFIX = 'datacenter-challenge/202201/'
METRICS = ['utilization_gpu_pct', 'utilization_memory_pct', 'power_draw_W',
           'temperature_gpu', 'temperature_memory']
SCHEMA_VERSION = 2
LOCAL = threading.local()


def utc():
    return pd.Timestamp.now(tz='UTC').isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.tmp')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    # Windows 上目标文件被其他进程（监控脚本、杀毒等）短暂打开时替换会被拒绝，稍后重试。
    for attempt in range(20):
        try:
            os.replace(partial, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25 * (attempt + 1))


def database(root):
    db = sqlite3.connect(Path(root) / 'state.sqlite', timeout=60)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS objects (
            seq INTEGER PRIMARY KEY, key TEXT UNIQUE NOT NULL, etag TEXT NOT NULL,
            bytes INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0, reason TEXT, batch TEXT, finished TEXT);
        CREATE TABLE IF NOT EXISTS settings (name TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS objects_status ON objects(status,seq);
    ''')
    # 调度字段独立迁移，保持已有批次 schema、序号及完成状态。
    if 'priority' not in {row['name'] for row in db.execute('PRAGMA table_info(objects)')}:
        with db:
            db.execute('ALTER TABLE objects ADD COLUMN priority INTEGER NOT NULL DEFAULT 0')
    db.execute('CREATE INDEX IF NOT EXISTS objects_priority ON objects(status,priority DESC,seq)')
    return db


def session(proxy):
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
        LOCAL.session.trust_env = False
        LOCAL.session.proxies = {'http': proxy, 'https': proxy}
    return LOCAL.session


def get_retry(url, proxy, **kwargs):
    for attempt in range(5):
        try:
            response = session(proxy).get(url, timeout=(20, 120), **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(min(60, 2 ** attempt))


def process_lock(path):
    handle = Path(path).open('a+b')
    if os.name == 'nt':
        import msvcrt
        handle.seek(0)
        handle.write(b'0')
        handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


class DiskBudget:
    def __init__(self, cache, limit=19_000_000_000, reserve=10_000_000_000):
        self.cache = Path(cache)
        self.limit, self.reserve = limit, reserve
        self.existing = {p.name: p.stat().st_size for p in self.cache.glob('*.part')}
        self.reserved = sum(self.existing.values())
        if self.reserved > self.limit:
            raise OSError('已有下载缓存超过配置硬上限')
        self.lock = threading.Lock()

    def acquire(self, size, cache_name):
        if size > self.limit:
            raise ValueError(f'对象 {size} bytes 超出缓存硬上限 {self.limit}')
        with self.lock:
            existing = self.existing.get(cache_name, 0)
            if existing > size:
                raise ValueError('已缓存对象长度超过清单长度: ' + cache_name)
            additional = size - existing
            if self.reserved + additional > self.limit:
                return False
            if shutil.disk_usage(self.cache).free - self.reserved - additional < self.reserve:
                raise OSError('本机磁盘余量低于预留值，停止调度并保留断点')
            self.reserved += additional
            self.existing.pop(cache_name, None)
            return True

    def release(self, size):
        with self.lock:
            self.reserved -= size


class RangeDownloadError(RuntimeError):
    pass


def fetch_range(obj, begin, stop, proxy, retries):
    for attempt in range(1, retries + 1):
        began = time.monotonic()
        content = bytearray()
        try:
            with session(proxy).get(BASE + obj['key'], timeout=(20, 30), stream=True,
                    headers={'If-Match': '"' + obj['etag'] + '"', 'Range': f'bytes={begin}-{stop}'}) as response:
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {begin}-{stop}/{obj["bytes"]}':
                    raise ValueError('Range 响应边界或状态不符')
                if response.headers.get('ETag', '').strip('"') != obj['etag']:
                    raise ValueError('响应 ETag 与对象清单不符')
                if int(response.headers['Content-Length']) != stop - begin + 1:
                    raise ValueError('响应 Content-Length 与请求范围不符')
                for block in response.iter_content(64 * 1024):
                    content.extend(block)
                    if len(content) > stop - begin + 1:
                        raise ValueError('Range 实际长度超过请求范围')
                    if time.monotonic() - began > 90:
                        raise requests.Timeout('单段请求超过 90 秒')
            if len(content) != stop - begin + 1:
                raise ValueError('Range 下载字节数不完整')
            return bytes(content), attempt - 1
        except (requests.RequestException, ValueError) as error:
            print(json.dumps({'at': utc(), 'event': 'range_retry', 'seq': obj['seq'],
                'begin': begin, 'attempt': attempt, 'reason': str(error)}, ensure_ascii=False) + '\n', end='', flush=True)
            if attempt == retries:
                raise RangeDownloadError(f'Range {begin}-{stop} 连续 {retries} 次失败: {error}') from error
            time.sleep(min(120, 2 ** attempt))


def download(obj, path, proxy, retries=5, range_pool=None):
    path = Path(path)
    part_size = 2 * 1024 * 1024
    if path.exists() and path.stat().st_size > obj['bytes']:
        raise ValueError('缓存文件长度超过源对象长度')
    offset = path.stat().st_size if path.exists() else 0
    if offset != obj['bytes']:
        offset = offset // part_size * part_size
    range_retries = 0
    inflight = {}
    next_offset = offset
    with path.open('r+b' if path.exists() else 'w+b') as output:
        output.truncate(offset)
        output.seek(offset)
        try:
            while offset < obj['bytes']:
                stop = min(offset + part_size, obj['bytes']) - 1
                if range_pool is None:
                    content, retried = fetch_range(obj, offset, stop, proxy, retries)
                else:
                    while next_offset < obj['bytes'] and len(inflight) < 8:
                        next_stop = min(next_offset + part_size, obj['bytes']) - 1
                        inflight[next_offset] = range_pool.submit(fetch_range, obj, next_offset, next_stop, proxy, retries)
                        next_offset = next_stop + 1
                    content, retried = inflight.pop(offset).result()
                output.write(content)
                range_retries += retried
                offset = stop + 1
                output.flush()
        finally:
            for future in inflight.values():
                future.cancel()
    if path.stat().st_size != obj['bytes']:
        raise ValueError('下载字节数与清单不符')
    digest = hashlib.md5()
    sha = hashlib.sha256()
    parts = []
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
            sha.update(block)
            parts.append(hashlib.md5(block).digest())
    actual = (hashlib.md5(b''.join(parts)).hexdigest() + '-' + str(len(parts))
              if '-' in obj['etag'] else digest.hexdigest())
    if actual != obj['etag']:
        path.unlink()
        raise ValueError('完整下载内容的 MD5/multipart MD5 与 ETag 不符')
    return {'sha256': sha.hexdigest(), 'range_retries': range_retries, 'etag_validation':
            'response+length+md5' if '-' not in obj['etag'] else 'response+length+multipart-md5-8MiB'}


def load_jobs(root):
    mapping = pd.read_csv(Path(root) / 'reference/tres-mapping.txt')
    resources = dict(zip(mapping.tres_id, mapping.resource_type.str.strip()))
    if resources.get(1001) != 'gpu:tesla' or resources.get(1002) != 'gpu:volta':
        raise ValueError('官方 TRES GPU 映射发生变化')
    jobs = pd.read_csv(Path(root) / 'reference/slurm-log.csv', usecols=[
        'id_job', 'nodelist', 'time_start', 'time_end', 'time_suspended', 'tres_alloc', 'tres_req'],
        dtype={'id_job': str, 'tres_alloc': str, 'tres_req': str})
    if jobs.id_job.duplicated().any():
        raise ValueError('Slurm id_job 重复，需要重新核定映射')
    result = {}
    for row in jobs.itertuples(index=False):
        alloc, req = {}, {}
        for text, target in [(row.tres_alloc, alloc), (row.tres_req, req)]:
            if pd.notna(text):
                for item in text.split(','):
                    key, value = item.split('=')
                    if int(key) not in resources:
                        raise ValueError('出现未知 TRES 资源 ' + key)
                    target[int(key)] = float(value)
        result[row.id_job] = {'start': row.time_start, 'end': row.time_end,
            'suspended': row.time_suspended, 'nodes': ast.literal_eval(row.nodelist),
            'allocated_gpus': int(alloc.get(1001, 0) + alloc.get(1002, 0)),
            'requested_gpus': int(req.get(1001, 0) + req.get(1002, 0)),
            'gpu_type': '+'.join(resources[i] for i in (1001, 1002) if alloc.get(i, 0)) or 'unknown'}
    return result


def identity(key, jobs):
    match = re.fullmatch(r'(\d+)-(r\d+-n\d+)\.csv', Path(key).name)
    if not match:
        return '', '', None, 'filename_unmapped'
    job_id, node = match.groups()
    job = jobs.get(job_id)
    state = 'matched' if job and node in job['nodes'] else ('job_missing' if not job else 'node_mismatch')
    return job_id, node, job, state


def utilization_bin(values):
    return np.where(values == 0, 0, np.ceil(values / 5)).astype('int16')


def merge_frames(frames, dimensions):
    frame = pd.concat(frames, ignore_index=True)
    operations = {c: ('max' if c.endswith('_max') else 'sum') for c in frame if c not in dimensions}
    return frame.groupby(dimensions, dropna=False, sort=True).agg(operations).reset_index()


def summarize(path, obj, jobs, chunk_rows=100_000):
    job_id, node, job, mapping = identity(obj['key'], jobs)
    start = int(job['start'] * 1_000_000) if job else 0
    end = int(job['end'] * 1_000_000) if job else 0
    shift_us = 0
    clock_status = 'unmapped_source_clock'
    if job and job['start'] > 0 and job['end'] >= job['start']:
        zone = ZoneInfo('America/New_York')
        begin_offset = datetime.fromtimestamp(job['start'], timezone.utc).astimezone(zone).utcoffset()
        end_offset = datetime.fromtimestamp(job['end'], timezone.utc).astimezone(zone).utcoffset()
        if begin_offset == end_offset:
            shift_us = -int(begin_offset.total_seconds()) * 1_000_000
            clock_status = 'new_york_wall_time_inferred'
        else:
            clock_status = 'dst_transition_requires_review'
    pieces = {'hourly': [], 'joint_hist': [], 'util_hist': []}
    dims = {'hourly': ['gpu_index', 'hour_us', 'within_allocation'],
            'joint_hist': ['month', 'within_allocation', 'util_bin', 'power_bin'],
            'util_hist': ['month', 'within_allocation', 'util_value']}
    pending = pd.DataFrame()
    rows = invalid_identity = invalid_values = missing_values = 0
    gpu_ids, cadence = set(), Counter()
    first, last = None, None
    nonpositive = gaps = 0

    def consume(frame, final=False):
        nonlocal pending, first, last, nonpositive, gaps
        if not pending.empty:
            frame = pd.concat([pending, frame], ignore_index=True)
        if frame.empty:
            return
        dt = frame.groupby('gpu_index', sort=False).time_us.shift(-1) - frame.time_us
        tail = ~frame.gpu_index.duplicated(keep='last')
        pending = frame.loc[tail].copy() if not final else pd.DataFrame()
        if not final:
            frame, dt = frame.loc[~tail].copy(), dt.loc[~tail]
        else:
            frame = frame.copy()
        if frame.empty:
            return
        finite_dt = dt.dropna().astype('int64')
        nonpositive += int((finite_dt <= 0).sum())
        gaps += int((finite_dt > 1_000_000).sum())
        # 只保留毫秒间隔频数用于采样诊断；有效时长使用精确微秒，间隙上限 1 秒。
        cadence.update((finite_dt[finite_dt > 0] // 1000).value_counts().to_dict())
        duration = dt.fillna(0).clip(lower=0, upper=1_000_000).astype('int64')
        frame['hour_us'] = frame.time_us // 3_600_000_000 * 3_600_000_000
        frame['within_allocation'] = (frame.time_us >= start) & (frame.time_us < end) if job else False
        # 原始区间时长和分配窗口内时长分别保存；窗口外观测仍然进入独立分组。
        duration = np.where(frame.within_allocation, np.minimum(duration, end - frame.time_us), duration)
        frame['duration_us'] = duration
        frame['sample_count'] = 1
        first_part = np.minimum(duration, frame.hour_us + 3_600_000_000 - frame.time_us)
        spill = frame.duration_us - first_part
        frame['duration_us'] = first_part
        extra = frame.loc[spill > 0].copy()
        extra['hour_us'] += 3_600_000_000
        extra['duration_us'] = spill.loc[spill > 0] if isinstance(spill, pd.Series) else spill[spill > 0]
        extra['sample_count'] = 0
        frame = pd.concat([frame, extra], ignore_index=True)
        month_names = {hour: datetime.fromtimestamp(int(hour) // 1_000_000, timezone.utc).strftime('%Y-%m')
                       for hour in frame.hour_us.unique()}
        frame['month'] = frame.hour_us.map(month_names)
        groups = frame[dims['hourly'] + ['sample_count', 'duration_us']].copy()
        for metric in METRICS:
            valid = frame[metric].notna()
            groups[metric + '_count'] = valid.astype('int64') * frame.sample_count
            groups[metric + '_sum'] = frame[metric].fillna(0) * frame.sample_count
            groups[metric + '_sumsq'] = frame[metric].fillna(0) ** 2 * frame.sample_count
            groups[metric + '_max'] = frame[metric].where(frame.sample_count > 0)
            groups[metric + '_valid_us'] = frame.duration_us * valid
            groups[metric + '_time_sum_us'] = frame[metric].fillna(0) * frame.duration_us
            groups[metric + '_time_sumsq_us'] = frame[metric].fillna(0) ** 2 * frame.duration_us
        pieces['hourly'].append(merge_frames([groups], dims['hourly']))
        valid = frame.utilization_gpu_pct.notna()
        util = frame.loc[valid, ['month', 'within_allocation', 'sample_count', 'duration_us']].copy()
        util['util_value'] = frame.loc[valid, 'utilization_gpu_pct']
        pieces['util_hist'].append(merge_frames([util], dims['util_hist']))
        valid &= frame.power_draw_W.notna()
        joint = frame.loc[valid, ['month', 'within_allocation', 'sample_count', 'duration_us']].copy()
        joint['util_bin'] = utilization_bin(frame.loc[valid, 'utilization_gpu_pct'])
        joint['power_bin'] = np.floor(frame.loc[valid, 'power_draw_W'] / 5).astype('int32')
        pieces['joint_hist'].append(merge_frames([joint], dims['joint_hist']))
        for name in pieces:
            if len(pieces[name]) >= 8:
                pieces[name] = [merge_frames(pieces[name], dims[name])]

    columns = ['timestamp', 'gpu_index', 'utilization_gpu_pct', 'utilization_memory_pct',
               'memory_free_MiB', 'memory_used_MiB', 'temperature_gpu', 'temperature_memory',
               'power_draw_W', 'pcie_link_width_current']
    source_columns = None
    for raw in pd.read_csv(path, dtype=str, chunksize=chunk_rows, on_bad_lines='error'):
        if not set(columns).issubset(raw.columns) or not isinstance(raw.index, pd.RangeIndex):
            raise ValueError('CSV 缺少必需监测列，或行字段数量与文件表头不符')
        source_columns = list(raw.columns)
        rows += len(raw)
        raw = raw[['timestamp', 'gpu_index', *METRICS]]
        numeric = raw.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        invalid_values += int((numeric.isna() & raw.notna()).sum().sum())
        missing_values += int(raw.isna().sum().sum())
        valid = numeric.timestamp.notna() & numeric.gpu_index.notna()
        valid &= numeric.gpu_index.eq(np.floor(numeric.gpu_index)) & numeric.gpu_index.ge(0)
        # 无法定位到 GPU 或时间的行仍计入文件行数和异常计数。
        invalid_identity += int((~valid).sum())
        numeric = numeric.loc[valid].copy()
        for metric in METRICS:
            if metric.startswith('utilization'):
                bad = numeric[metric].notna() & ~numeric[metric].between(0, 100)
            elif metric == 'power_draw_W':
                bad = numeric[metric].lt(0)
            else:
                bad = numeric[metric].lt(-273.15)
            invalid_values += int(bad.sum())
            numeric.loc[bad, metric] = np.nan
        if numeric.empty:
            continue
        numeric['time_us'] = np.rint(numeric.timestamp * 1_000_000).astype('int64') + shift_us
        numeric['gpu_index'] = numeric.gpu_index.astype('int32')
        gpu_ids.update(numeric.gpu_index.unique().tolist())
        tmin, tmax = int(numeric.time_us.min()), int(numeric.time_us.max())
        first, last = min(first, tmin) if first is not None else tmin, max(last, tmax) if last is not None else tmax
        consume(numeric)
    consume(pd.DataFrame(), final=True)
    if not rows or not pieces['hourly']:
        raise ValueError('对象没有可以汇总的 GPU 时间记录')
    eligibility = mapping
    if mapping == 'matched':
        if not job['allocated_gpus'] or not job['requested_gpus']:
            eligibility = 'no_gpu_request_or_allocation'
        elif job['suspended']:
            eligibility = 'suspended_interval_unknown'
        elif end <= start:
            eligibility = 'invalid_job_interval'
        elif clock_status == 'dst_transition_requires_review':
            eligibility = clock_status
        elif len(gpu_ids) * len(job['nodes']) > job['allocated_gpus']:
            eligibility = 'device_count_exceeds_allocation'
        else:
            eligibility = 'eligible'
    common = {'seq': obj['seq'], 'job_id': job_id, 'node': node, 'mapping_status': mapping,
              'eligibility': eligibility, 'allocated_gpus': job['allocated_gpus'] if job else -1,
              'gpu_type': job['gpu_type'] if job else 'unknown'}
    tables = {}
    for name in pieces:
        table = merge_frames(pieces[name], dims[name])
        for key, value in common.items():
            table[key] = value
        tables[name] = table
    hourly = tables['hourly']
    if int(hourly.sample_count.sum()) + invalid_identity != rows:
        raise ValueError('逐小时样本计数不守恒')
    if int(tables['util_hist'].sample_count.sum()) != int(hourly.utilization_gpu_pct_count.sum()):
        raise ValueError('利用率直方图样本计数不守恒')
    metadata = {**common, 'key': obj['key'], 'etag': obj['etag'], 'bytes': obj['bytes'],
        'rows': rows, 'gpu_count': len(gpu_ids), 'gpu_indices': json.dumps(sorted(gpu_ids)),
        'time_min_us': first, 'time_max_us': last, 'invalid_identity_rows': invalid_identity,
        'parse_anomalies': invalid_values, 'missing_fields': missing_values,
        'nonpositive_intervals': nonpositive, 'gaps_over_1s': gaps,
        'cadence_ms_counts': json.dumps(dict(sorted(cadence.items()))),
        'joint_valid_samples': int(tables['joint_hist'].sample_count.sum()),
        'duration_us': int(hourly.duration_us.sum()), 'schema_version': SCHEMA_VERSION}
    metadata.update({'job_start': job['start'] if job else 0, 'job_end': job['end'] if job else 0,
                     'job_nodes': len(job['nodes']) if job else 0,
                     'requested_gpus': job['requested_gpus'] if job else -1,
                     'time_suspended': job['suspended'] if job else -1,
                     'source_columns': json.dumps(source_columns), 'clock_status': clock_status,
                     'clock_shift_us': shift_us, 'raw_time_min_us': first - shift_us,
                     'raw_time_max_us': last - shift_us,
                     'start_residual_s': (first - start) / 1_000_000 if job else None,
                     'end_residual_s': (last - end) / 1_000_000 if job else None})
    tables['files'] = pd.DataFrame([metadata])
    return tables
