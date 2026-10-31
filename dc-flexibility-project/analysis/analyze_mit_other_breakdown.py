import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from audit_research_gpu_demand import tres_count


RULES = [
    ('non_success', '失败、取消或异常退出', 'state低8位不等于3，或存在requeue标志，或exit_code/derived_ec非零',
     '逐作业审阅；成功工作量与失败占用分别考虑', '恢复与重试机制未知'),
    ('array', '数组批量作业', 'id_array_task不等于Slurm NO_VAL=4294967294',
     '成员整体延后候选；须核实依赖和完成期限', '依赖和并发限制核实后可按成员安排；成员内部checkpoint未知'),
    ('large', '多节点或大GPU作业', 'nodes_alloc>=2，或分配GPU数>=8',
     '完整资源同时分配的整体延后候选；须核实用途和期限', '完整资源与checkpoint要求未知'),
    ('short_test', '短作业或测试分区作业', '运行<=600秒，或partition=test',
     '用途审阅后可按完整作业批量改序；测试交互需求未知', '按完整作业运行；checkpoint未知'),
    ('interactive_candidate', '疑似交互式会话', '单节点、GPU<=2、等待<=2秒、600<运行<21600秒、0<有限timelimit<=720分钟',
     '弱规则候选，基准保持原启动时间', '会话状态与checkpoint未知，基准不拆分'),
    ('long', '长时作业，训练候选', '运行>=21600秒',
     '用途与期限核验后的整体延后候选', '需完整checkpoint及恢复成本验证'),
    ('remaining', '中等时长、用途待定', '上述规则均未命中，600<运行<21600秒',
     '用途与期限待审阅，基准保持原启动时间', 'checkpoint未知，基准不拆分'),
]
RULE_NAMES = {row[0]: row[1] for row in RULES}
GPU_IDS = [1001, 1002]
READ_COLUMNS = ['id_job', 'id_array_task', 'nodes_alloc', 'derived_ec', 'exit_code',
                'partition', 'state', 'timelimit', 'time_submit', 'time_start', 'time_end',
                'time_suspended', 'tres_alloc', 'tres_req', 'job_type']


def fingerprint(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'读取期间输入发生变化: {path}')
    return {'file': path.name, 'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns,
            'sha256': digest.hexdigest()}


def family(model):
    if model.startswith('U'):
        return 'U-Net'
    for prefix, name in [('vgg', 'VGG'), ('resnet', 'ResNet'), ('inception', 'Inception'),
                         ('bert-', 'Bert'), ('distilbert-', 'DistilBert')]:
        if model.startswith(prefix):
            return name
    if model in {'dimenet', 'schnet', 'conv', 'pna'}:
        return 'GNN'
    raise ValueError(f'未知模型名称: {model}')


def load_jobs(reference, labels, chunksize):
    seen = set()
    parts = []
    matched_cpu = 0
    row_count = 0
    for chunk in pd.read_csv(reference / 'slurm-log.csv', usecols=READ_COLUMNS,
                             dtype={'tres_req': str, 'tres_alloc': str}, chunksize=chunksize):
        if chunk.id_job.isna().any() or chunk.id_job.duplicated().any() or seen.intersection(chunk.id_job):
            raise ValueError('作业ID缺失或重复')
        seen.update(chunk.id_job)
        row_count += len(chunk)
        requested = sum(tres_count(chunk.tres_req, key).fillna(0) for key in GPU_IDS)
        allocated = sum(tres_count(chunk.tres_alloc, key).fillna(0) for key in GPU_IDS)
        matched_cpu += int((chunk.id_job.isin(labels.id_job) & requested.eq(0)).sum())
        selected = chunk.loc[requested.gt(0)].drop(columns=['tres_req', 'tres_alloc']).copy()
        selected['gpus'] = allocated[requested.gt(0)]
        selected['runtime_s'] = selected.time_end - selected.time_start
        valid = (selected.time_start.gt(0) & selected.time_end.gt(0)
                 & selected.runtime_s.ge(0) & selected.time_suspended.eq(0) & selected.gpus.gt(0))
        if not valid.all():
            raise ValueError('本地GPU执行区间与PR #26全部有效的基线不同')
        wait = selected.time_start - selected.time_submit
        selected['wait_s'] = wait.where(selected.time_submit.gt(0) & wait.ge(0))
        selected['gpu_hours'] = selected.gpus * selected.runtime_s / 3600
        selected['runtime_h'] = selected.runtime_s / 3600
        selected['wait_h'] = selected.wait_s / 3600
        selected['finite_timelimit_min'] = selected.timelimit.where(selected.timelimit.lt(0xfffffffe))
        parts.append(selected)
    if row_count != 395914:
        raise ValueError(f'本地slurm-log行数变化: {row_count}')
    jobs = pd.concat(parts, ignore_index=True)
    del parts, seen
    jobs = jobs.merge(labels, on='id_job', how='left', validate='one_to_one')
    jobs['model_family'] = jobs.model.map(lambda value: family(value) if pd.notna(value) else None)
    return jobs, row_count, matched_cpu


def classify(jobs, include_status=True):
    masks = {
        'non_success': ((jobs.state & 0xff).ne(3) | (jobs.state & 0x400).ne(0)
                        | jobs.exit_code.ne(0) | jobs.derived_ec.ne(0)),
        'array': jobs.id_array_task.ne(0xfffffffe),
        'large': jobs.nodes_alloc.ge(2) | jobs.gpus.ge(8),
        'short_test': jobs.runtime_s.le(600) | jobs.partition.eq('test'),
        'interactive_candidate': (jobs.nodes_alloc.eq(1) & jobs.gpus.le(2) & jobs.wait_s.le(2)
                                  & jobs.runtime_s.gt(600) & jobs.runtime_s.lt(21600)
                                  & jobs.finite_timelimit_min.gt(0) & jobs.finite_timelimit_min.le(720)),
        'long': jobs.runtime_s.ge(21600),
    }
    selected_masks = masks if include_status else {key: value for key, value in masks.items() if key != 'non_success'}
    return pd.Series(np.select(list(selected_masks.values()), list(selected_masks), default='remaining'), index=jobs.index), masks


def summarize(group, total, other):
    hours = float(group.gpu_hours.sum())
    result = {'rows': len(group), 'gpu_hours': hours, 'pct_mit_gpu_hours': 100 * hours / total,
              'pct_other_gpu_hours': 100 * hours / other,
              'wait_valid_rows': int(group.wait_s.notna().sum()),
              'array_rows': int(group.id_array_task.ne(0xfffffffe).sum()),
              'finite_timelimit_rows': int(group.finite_timelimit_min.notna().sum())}
    for name in ['runtime_h', 'wait_h', 'gpus', 'nodes_alloc', 'finite_timelimit_min']:
        for q in [.5, .9, .99]:
            result[f'{name}_p{int(q*100)}'] = group[name].quantile(q)
        result[f'{name}_max'] = group[name].max()
    return result


def emit(output, name, rows):
    pd.DataFrame(rows).to_csv(output / name, index=False, encoding='utf-8', float_format='%.12g')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--chunksize', type=int, default=20000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reference = args.root / 'reference'
    names = ['slurm-log.csv', 'labelled_jobids.csv', 'labelled_job_stats.csv',
             'README.md', 'LICENSE', 'tres-mapping.txt', 'other-label-acquisition-20261031.json']
    inputs = [fingerprint(reference / name) for name in names]
    previous = json.loads((args.baseline / 'mit-audit.json').read_text(encoding='utf-8'))
    expected_hashes = {Path(row['path']).name: row['sha256'] for row in previous['inputs']}
    for row in inputs:
        if row['file'] in expected_hashes and row['sha256'] != expected_hashes[row['file']]:
            raise ValueError(f'PR #26输入指纹变化: {row["file"]}')
    mapping = pd.read_csv(reference / 'tres-mapping.txt')
    resources = dict(zip(mapping.tres_id, mapping.resource_type.str.strip()))
    if {key: resources[key] for key in GPU_IDS} != {1001: 'gpu:tesla', 1002: 'gpu:volta'}:
        raise ValueError('GPU TRES映射发生变化')
    labels = pd.read_csv(reference / 'labelled_jobids.csv', dtype={'id_job': 'int64'})
    if labels.isna().any().any() or labels.id_job.duplicated().any():
        raise ValueError('标签缺失或ID不唯一')
    stats = pd.read_csv(reference / 'labelled_job_stats.csv')
    pd.testing.assert_series_equal(labels.model.value_counts().sort_index(),
                                   stats.set_index('model')['count'].sort_index(), check_names=False)
    jobs, all_rows, matched_cpu = load_jobs(reference, labels, args.chunksize)
    jobs['rule'], masks = classify(jobs)
    jobs['workload_shape'], _ = classify(jobs, include_status=False)
    other_jobs = jobs[jobs.job_type.eq('OTHER')]
    known = other_jobs[other_jobs.model.notna()]
    unknown = other_jobs[other_jobs.model.isna()]
    total, other = float(jobs.gpu_hours.sum()), float(other_jobs.gpu_hours.sum())
    if len(jobs) != 98176 or len(other_jobs) != 76404 or round(total, 2) != 1460190.66:
        raise ValueError('GPU分母与PR #26不一致')
    baseline = pd.read_csv(args.baseline / 'mit-classes.csv')
    reconciliation = []
    for kind, group in jobs.groupby('job_type'):
        old = baseline[(baseline.resource_pool.eq('gpu_hours')) & baseline['class'].eq(kind)].iloc[0]
        if len(group) != old['rows'] or not np.isclose(group.gpu_hours.sum(), old.resource_hours, atol=1e-8, rtol=0):
            raise ValueError(f'PR #26类别口径不一致: {kind}')
        reconciliation.append({'job_type': kind, 'rows': len(group), 'gpu_hours': group.gpu_hours.sum(),
                               'baseline_rows': old['rows'], 'baseline_gpu_hours': old.resource_hours,
                               'gpu_hours_difference': group.gpu_hours.sum() - old.resource_hours})
    emit(args.output, 'baseline-reconciliation.csv', reconciliation)
    coverage = []
    matched = jobs[jobs.model.notna()]
    for kind in sorted(jobs.job_type.unique()):
        group = matched[matched.job_type.eq(kind)]
        coverage.append({'scope': kind, **summarize(group, total, other)})
    emit(args.output, 'label-coverage-by-job-type.csv', coverage)
    coverage_totals = [
        ('label_rows', len(labels)), ('label_unique_job_ids', labels.id_job.nunique()),
        ('official_readme_stated_label_jobs', 3425), ('label_stats_count_sum', stats['count'].sum()),
        ('matched_gpu_jobs', len(matched)), ('matched_no_gpu_request_jobs', matched_cpu),
        ('unmatched_label_jobs', len(labels) - len(matched) - matched_cpu),
        ('matched_other_gpu_jobs', len(known)), ('labelled_other_gpu_hours', known.gpu_hours.sum()),
        ('pct_labels_matched', 100 * (len(matched) + matched_cpu) / len(labels)),
        ('pct_other_gpu_hours_labelled', 100 * known.gpu_hours.sum() / other),
        ('pct_mit_gpu_hours_labelled_other', 100 * known.gpu_hours.sum() / total),
        ('unlabelled_other_jobs', len(unknown)), ('unlabelled_other_gpu_hours', unknown.gpu_hours.sum()),
        ('slurm_rows', all_rows), ('mit_gpu_jobs', len(jobs)), ('other_gpu_jobs', len(other_jobs)),
        ('mit_gpu_hours_exact', total), ('other_gpu_hours_exact', other),
    ]
    emit(args.output, 'coverage-totals.csv', [{'metric': key, 'value': value} for key, value in coverage_totals])
    for col, filename in [('model_family', 'labelled-other-model-families.csv'),
                          ('model', 'labelled-other-model-variants.csv')]:
        emit(args.output, filename, [{col: name, **summarize(group, total, other)}
                                    for name, group in known.groupby(col)])
    emit(args.output, 'rule-definitions.csv', [dict(zip(
        ['rule', 'category', 'condition_after_preceding_rules', 'time_shift_assessment', 'split_assessment'], row),
        precedence=i + 1) for i, row in enumerate(RULES)])
    final = [{'category_id': 'labelled_dnn', 'category': '官方标签DNN训练', 'evidence': '标签实测',
              'time_shift_assessment': '训练用途已知；期限核验后的完整作业延后候选',
              'split_assessment': '须验证checkpoint完整性和恢复成本', **summarize(known, total, other)}]
    for rule, name, _, shift, split in RULES:
        group = unknown[unknown.rule.eq(rule)]
        final.append({'category_id': rule, 'category': name, 'evidence': '规则推断',
                      'time_shift_assessment': shift, 'split_assessment': split, **summarize(group, total, other)})
    emit(args.output, 'other-breakdown.csv', final)
    validation = []
    for scope, selected in [('all_labelled_gpu', matched), ('labelled_other', known)]:
        for rule, name, *_ in RULES:
            group = selected[selected.rule.eq(rule)]
            validation.append({'scope': scope, 'rule': rule, 'category': name,
                               'pct_scope_rows': 100 * len(group) / len(selected),
                               'pct_scope_gpu_hours': 100 * group.gpu_hours.sum() / selected.gpu_hours.sum(),
                               **summarize(group, total, other)})
    emit(args.output, 'label-rule-validation.csv', validation)
    emit(args.output, 'label-model-rule-crosswalk.csv', [
        {'model_family': fam, 'rule': rule, 'category': RULE_NAMES[rule], **summarize(group, total, other)}
        for (fam, rule), group in known.groupby(['model_family', 'rule'])])
    overlaps = []
    for scope, selected in [('unlabelled_other', unknown), ('labelled_other', known)]:
        for rule, mask in masks.items():
            group = selected[mask.loc[selected.index]]
            overlaps.append({'scope': scope, 'rule': rule, 'overlaps_allowed': True,
                             **summarize(group, total, other)})
    emit(args.output, 'rule-overlapping-features.csv', overlaps)
    failed = unknown[unknown.rule.eq('non_success')]
    emit(args.output, 'non-success-workload-shapes.csv', [
        {'rule': rule, 'category': RULE_NAMES[rule], 'pct_non_success_gpu_hours':
         100 * group.gpu_hours.sum() / failed.gpu_hours.sum(), **summarize(group, total, other)}
        for rule, group in failed.groupby('workload_shape')])
    status_names = {0: 'PENDING', 3: 'COMPLETED', 4: 'CANCELLED', 5: 'FAILED',
                    6: 'TIMEOUT', 7: 'NODE_FAIL', 11: 'OUT_OF_MEMORY', 1024: 'PENDING|REQUEUE_FLAG'}
    for column in ['state', 'exit_code', 'derived_ec', 'partition']:
        emit(args.output, f'other-by-{column}.csv', [
            {'evidence': evidence, column: value,
             **({'state_name': status_names[value]} if column == 'state' else {}),
             **summarize(group, total, other)}
            for evidence, selected in [('标签实测', known), ('未标注', unknown)]
            for value, group in selected.groupby(column)])
    # 只检查既有分析目录的文件及schema，不访问流式处理数据库。
    telemetry = []
    for path in sorted((args.root / 'analysis').glob('*/device-hours.parquet')):
        metadata = fingerprint(path)
        parquet = pq.ParquetFile(path)
        telemetry.append({'relative_path': path.relative_to(args.root).as_posix(), **metadata,
                          'rows': parquet.metadata.num_rows, 'columns': ','.join(parquet.schema.names),
                          'job_id_present': 'id_job' in parquet.schema.names,
                          'used_for_classification': False})
    for path in sorted((args.root / 'aggregates').glob('*/summary.json')):
        meta = fingerprint(path)
        telemetry.append({'relative_path': path.relative_to(args.root).as_posix(), **meta,
                          'job_id_present': False, 'used_for_classification': False})
    emit(args.output, 'telemetry-inspection.csv', telemetry)
    acquisition = json.loads((reference / names[-1]).read_text(encoding='utf-8'))
    emit(args.output, 'download-manifest.csv', acquisition['files'])
    after = [fingerprint(reference / name) for name in names]
    if inputs != after:
        raise ValueError('分析前后输入发生变化')
    emit(args.output, 'input-fingerprints.csv', inputs)
    np.testing.assert_allclose(sum(row['gpu_hours'] for row in final), other, atol=1e-8, rtol=0)
    if sum(row['rows'] for row in final) != len(other_jobs):
        raise ValueError('最终类别计数不守恒')
    # 读回交付CSV，核对写入精度及完整性。
    saved = pd.read_csv(args.output / 'other-breakdown.csv')
    np.testing.assert_allclose(saved.gpu_hours.sum(), other, atol=1e-5, rtol=0)
    np.testing.assert_allclose(saved.pct_other_gpu_hours.sum(), 100, atol=1e-8, rtol=0)
    if saved.rows.sum() != 76404:
        raise ValueError('读回计数不守恒')
    checks = [('input_hashes_unchanged', True), ('baseline_all_gpu_classes_reconciled', True),
              ('label_ids_unique', True), ('label_stats_counts_equal_label_file', True),
              ('all_gpu_execution_intervals_valid', True), ('other_counts_conserved', True),
              ('other_gpu_hours_conserved', True), ('saved_csv_totals_reconciled', True),
              ('per_job_output_written', False), ('streaming_database_opened', False)]
    timestamp = datetime.now(timezone.utc).isoformat()
    emit(args.output, 'verification.csv', [{'check': key, 'value': value, 'checked_at_utc': timestamp}
                                         for key, value in checks])
    print(saved[['category', 'rows', 'gpu_hours', 'pct_mit_gpu_hours', 'pct_other_gpu_hours',
                 'runtime_h_p50', 'runtime_h_p90', 'wait_h_p50', 'wait_h_p90']].to_string(index=False))
    print(f'Labels: {len(labels)}, matched GPU: {len(matched)}, OTHER coverage: {100*known.gpu_hours.sum()/other:.6f}%')


if __name__ == '__main__':
    main()
