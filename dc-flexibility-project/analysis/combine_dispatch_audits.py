import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest, sha256
import json
from pathlib import Path

import pandas as pd


KEYS = ['city', 'scenario', 'alpha', 'flex', 'card_price', 'supply']


def key(row):
    return tuple('economic' if name == 'alpha' and (row[name] is None or pd.isna(row[name])) else str(row[name]) for name in KEYS)


def load_index(path):
    frame = pd.read_csv(path, dtype={'card_price': str}, float_precision='round_trip')
    rows = frame.to_dict('records')
    mapped = {key(row): row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError('索引含重复案例：' + str(path))
    return mapped


def match_records(audit, index):
    by_path = {row['hourly_path']: row for row in index.values()}
    if len(by_path) != len(index):
        raise ValueError('补齐审查元数据前须唯一对应小时文件')
    mapped = {}
    for original in audit['cases']:
        record = dict(original)
        row = by_path[record['hourly_path']]
        if 'supply' in record and record['supply'] != row['supply']:
            raise ValueError('逐例审查与索引供能情境不相同')
        record['supply'] = row['supply']
        if key(record) != key(row):
            raise ValueError('逐例审查与完整索引案例不相同')
        for field in ('input_sha256', 'hourly_sha256'):
            if record[field] != row[field]:
                raise ValueError('逐例审查与索引指纹不相同：' + field)
        if key(record) in mapped:
            raise ValueError('逐例审查重复')
        mapped[key(record)] = record
    return mapped


def verify_file(item):
    path, expected = item
    with path.open('rb') as handle:
        actual = file_digest(handle, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError('最终文件指纹与独立审查不相同：' + str(path))
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-index', type=Path, required=True)
    parser.add_argument('--original-audit', type=Path, required=True)
    parser.add_argument('--refined-index', type=Path, required=True)
    parser.add_argument('--refined-audit', type=Path, required=True)
    parser.add_argument('--final-index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original_index, final_index = load_index(args.original_index), load_index(args.final_index)
    original_audit = json.loads(args.original_audit.read_text('utf-8'))
    refined_audit = json.loads(args.refined_audit.read_text('utf-8'))
    if refined_audit['passed'] is not True:
        raise ValueError('精度重算案例的独立验收尚未全部通过')
    original = match_records(original_audit, original_index)
    refined_index = load_index(args.refined_index)
    refined = match_records(refined_audit, refined_index)
    for audit, path in ((original_audit, args.original_index), (refined_audit, args.refined_index)):
        if audit['index_sha256'] != sha256(path.read_bytes()).hexdigest():
            raise ValueError('独立审查对应的索引已发生变化：' + str(path))
    if len(final_index) != 1296 or set(final_index) != set(original_index) or set(original) != set(original_index):
        raise ValueError('最终案例和原始完整审查必须覆盖同一1296个唯一组合')
    if original_audit['absolute_tolerance_mw_or_mwh'] != refined_audit['absolute_tolerance_mw_or_mwh']:
        raise ValueError('精度重算前后独立验收门槛不同')
    failed = {identity for identity, record in original.items() if not record['passed']}
    if set(refined) != failed or not all(row['passed'] for row in refined.values()):
        raise ValueError('重新审查必须完整且只覆盖所有失败重算案例')
    merged, checks, certified_files = [], {}, set()
    for identity, row in final_index.items():
        record = dict(refined[identity] if identity in refined else original[identity])
        certificate_index = refined_index if identity in refined else original_index
        certificate_root = args.refined_index.parent if identity in refined else args.original_index.parent
        for field, audited_field in [('whole_h', 'whole_deadline_hours'), ('split_h', 'split_deadline_hours')]:
            if float(row[field]) != float(record[audited_field]):
                raise ValueError('最终索引期限与已审查期限不相同')
        for field in ('input_sha256', 'hourly_sha256'):
            if record[field] != row[field]:
                raise ValueError('最终索引指纹未通过独立审查')
        for field, hash_field in [('input_path', 'input_sha256'), ('hourly_path', 'hourly_sha256')]:
            path = (args.final_index.parent / row[field]).resolve()
            expected = row[hash_field]
            if path in checks and checks[path] != expected:
                raise ValueError('同一路径出现冲突指纹')
            certified_path = (certificate_root / certificate_index[identity][field]).resolve()
            if path == certified_path:
                certified_files.add(path)
            else:
                checks[path] = expected
            record[field] = row[field]
        record['verification_reused_unchanged_case'] = identity not in refined
        merged.append(record)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(verify_file, checks.items()))
    report = {'passed': True, 'case_count': len(merged), 'refined_case_count': len(refined),
        'reused_verified_case_count': len(merged) - len(refined),
        'verified_file_count': len(certified_files | set(checks)), 'rehash_file_count': len(checks),
        'method': '完整独立FIFO审查；失败案例提高真实求解精度后重新审查；本次任务保留各源目录文件不变，逐例核对索引、两种SHA256及文件绝对路径；路径改变者重新读取校验',
        'absolute_tolerance_mw_or_mwh': original_audit['absolute_tolerance_mw_or_mwh'],
        'index_sha256': sha256(args.final_index.read_bytes()).hexdigest(),
        'source_audit_sha256': {str(path): sha256(path.read_bytes()).hexdigest() for path in (args.original_audit, args.refined_audit)},
        'cases': merged}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({name: report[name] for name in ('passed', 'case_count', 'refined_case_count', 'reused_verified_case_count')}))


if __name__ == '__main__':
    main()
