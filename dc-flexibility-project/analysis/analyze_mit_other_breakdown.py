import argparse
import hashlib
import json
from importlib.metadata import version
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
from scipy.stats import ks_2samp
from sklearn.model_selection import KFold, StratifiedKFold

from mit_other_inference import FEATURE_WEIGHTS, infer, load_jobs


CATEGORIES = [
    ('labelled_dnn', 'DNN训练（官方标签）', '实测', 'DNN用途', '期限和依赖核实后的整体延后候选', '需验证完整checkpoint与恢复成本'),
    ('user_dnn_high', '推断DNN训练（同用户强匹配）', '高', 'DNN用途推断', '用途和期限确认后的整体延后候选', 'checkpoint未观测，暂不拆分'),
    ('user_dnn_medium', '推断DNN训练（同用户近似）', '中', 'DNN用途推断', '人工核实后按完整作业延后', 'checkpoint未观测，暂不拆分'),
    ('similar_dnn', 'DNN训练候选（正例相似）', '中', 'DNN用途推断', '人工核实后按完整作业延后', 'checkpoint未观测，暂不拆分'),
    ('long_single', '长时单任务', '高', '运行形态', '用途和业务期限未知，条件性整体延后', 'checkpoint未观测，暂不拆分'),
    ('array', '数组批量作业（扫描候选）', '高', '运行形态', '核实依赖和并发限制后按成员改序', '成员间可条件安排；成员内部未知'),
    ('large', '多节点大作业', '高', '运行形态', '完整资源同时分配的条件性整体延后', '跨节点状态与恢复未知'),
    ('interactive', '交互式会话候选', '低', '用途弱推断', '基准保持原执行', '会话状态未知，暂不拆分'),
    ('startup', '启动失败或调试候选', '中', '短时异常形态', '排查异常后重新安排', '按完整短作业处理'),
    ('idle', '低GPU活动占用（监测）', '实测', 'GPU活动形态', '需核实后台工作和资源释放权限', '低活动不证明支持暂停恢复'),
    ('short_test', '短时或测试作业', '高', '运行形态', '用途核实后按完整作业改序', '按完整短作业处理'),
    ('unknown', '仍无法判断', '低', '用途与形态待定', '基准保持原执行', '恢复能力未知，暂不拆分'),
]
NAMES = {row[0]: row[1] for row in CATEGORIES}
TIERS = {row[0]: row[2] for row in CATEGORIES}
SCOPES = {row[0]: row[3] for row in CATEGORIES}


def family(model):
    for prefix, name in [('U', 'U-Net'), ('vgg', 'VGG'), ('resnet', 'ResNet'),
                         ('inception', 'Inception'), ('bert-', 'Bert'), ('distilbert-', 'DistilBert')]:
        if model.startswith(prefix):
            return name
    if model in {'dimenet', 'schnet', 'conv', 'pna'}:
        return 'GNN'
    raise ValueError(f'未知模型: {model}')


def emit(output, filename, rows):
    pd.DataFrame(rows).to_csv(output / filename, index=False, encoding='utf-8', float_format='%.12g')


def summary(group, total, other):
    hours = float(group.gpu_hours.sum())
    row = {'rows': len(group), 'gpu_hours': hours, 'pct_mit_gpu_hours': 100 * hours / total,
           'pct_other_gpu_hours': 100 * hours / other}
    for name in ['runtime_h', 'wait_h', 'gpus', 'nodes_alloc', 'cpu_request', 'memory_request_mb']:
        for q in [.5, .9, .99]:
            row[f'{name}_p{int(q*100)}'] = group[name].quantile(q)
    return row


def evaluate(reference, held, protocol, fold):
    if set(reference.id_job).intersection(held.id_job):
        raise ValueError('留出作业混入参考标签')
    prediction = infer(reference, held)
    rows = []
    masks = {'same_user_high': prediction.user_high,
             'same_user_any': prediction.user_near,
             'positive_similarity': prediction.similarity_medium,
             'combined_high_or_medium': prediction.user_near | prediction.similarity_medium}
    for method, found in masks.items():
        rows.append({'protocol': protocol, 'fold': fold, 'method': method,
                     'reference_jobs': len(reference), 'held_jobs': len(held), 'recovered_jobs': int(found.sum()),
                     'held_gpu_hours': held.gpu_hours.sum(), 'recovered_gpu_hours': held.loc[found, 'gpu_hours'].sum(),
                     'recall_jobs_pct': 100 * found.mean(),
                     'recall_gpu_hours_pct': 100 * held.loc[found, 'gpu_hours'].sum() / held.gpu_hours.sum(),
                     'similarity_threshold': prediction.similarity_threshold.iloc[0]})
    return rows


def cross_validation(known, output):
    rows = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261101)
    for fold, (train, test) in enumerate(splitter.split(known, known.model_family), 1):
        rows.extend(evaluate(known.iloc[train], known.iloc[test], 'stratified_random_5fold', fold))
    ordered = known.sort_values(['time_submit', 'id_job'])
    cut = int(len(ordered) * .8)
    rows.extend(evaluate(ordered.iloc[:cut], ordered.iloc[cut:], 'chronological_last20pct', 1))
    for fold, user in enumerate(sorted(known.id_user.unique()), 1):
        rows.extend(evaluate(known[known.id_user.ne(user)], known[known.id_user.eq(user)], 'leave_user_out', fold))
    emit(output, 'label-holdout-validation.csv', rows)
    table = pd.DataFrame(rows)
    totals = table.groupby(['protocol', 'method'])[['held_jobs', 'recovered_jobs', 'held_gpu_hours', 'recovered_gpu_hours']].sum()
    totals['recall_jobs_pct'] = 100 * totals.recovered_jobs / totals.held_jobs
    totals['recall_gpu_hours_pct'] = 100 * totals.recovered_gpu_hours / totals.held_gpu_hours
    totals.reset_index().to_csv(output / 'label-holdout-totals.csv', index=False, float_format='%.12g')
    print(totals[['recall_jobs_pct', 'recall_gpu_hours_pct']].to_string(), flush=True)


def attach_telemetry(jobs, private, output):
    telemetry = pd.read_parquet(private / 'existing-job-telemetry.parquet')
    if telemetry.id_job.duplicated().any():
        raise ValueError('监测结果作业ID重复')
    jobs = jobs.merge(telemetry, on='id_job', how='left', validate='one_to_one')
    jobs['telemetry_coverage'] = jobs.utilization_gpu_pct_observed_gpu_hours / jobs.gpu_hours
    jobs['telemetry_qualified'] = jobs.telemetry_coverage.between(.8, 1.05)
    jobs['telemetry_busy'] = (jobs.telemetry_qualified & jobs.utilization_gpu_pct_mean.ge(50)
                              & jobs.idle_fraction_le5pct.le(.2))
    jobs['telemetry_idle'] = (jobs.telemetry_qualified & jobs.utilization_gpu_pct_mean.le(5)
                              & jobs.idle_fraction_le5pct.ge(.9))
    positive = jobs[jobs.model.notna() & jobs.telemetry_qualified]
    floor = float(positive.utilization_gpu_pct_mean.quantile(.05))
    ceiling = float(positive.idle_fraction_le5pct.quantile(.95))
    compatible = jobs.utilization_gpu_pct_mean.ge(floor) & jobs.idle_fraction_le5pct.le(ceiling)
    jobs['similarity_metadata_match'] = jobs.similarity_medium
    jobs['similarity_rejected_by_telemetry'] = jobs.similarity_medium & jobs.telemetry_qualified & ~compatible
    jobs['similarity_medium'] &= ~jobs.telemetry_qualified | compatible
    checks = []
    for fold, (train, held) in enumerate(KFold(n_splits=5, shuffle=True, random_state=20261101).split(positive), 1):
        reference, target = positive.iloc[train], positive.iloc[held]
        low = reference.utilization_gpu_pct_mean.quantile(.05)
        high = reference.idle_fraction_le5pct.quantile(.95)
        kept = target.utilization_gpu_pct_mean.ge(low) & target.idle_fraction_le5pct.le(high)
        checks.append({'fold': fold, 'held_jobs': len(target), 'retained_jobs': int(kept.sum()),
                       'retention_pct': 100*kept.mean(), 'reference_utilization_p05': low,
                       'reference_idle_fraction_p95': high})
    emit(output, 'telemetry-positive-holdout.csv', checks)
    emit(output, 'telemetry-similarity-screen.csv', [
        {'metric': 'positive_qualified_jobs', 'value': len(positive)},
        {'metric': 'positive_utilization_mean_p05', 'value': floor},
        {'metric': 'positive_idle_fraction_p95', 'value': ceiling},
        {'metric': 'unlabelled_similarity_jobs_reassigned', 'value': int((jobs.model.isna() & jobs.similarity_rejected_by_telemetry).sum())},
        {'metric': 'unlabelled_similarity_gpu_hours_reassigned', 'value': jobs.loc[jobs.model.isna() & jobs.similarity_rejected_by_telemetry, 'gpu_hours'].sum()},
    ])
    return jobs


def classify(jobs):
    abnormal = ((jobs.state & 0xff).isin([5, 7, 11]) | jobs.exit_code.ne(0) | jobs.derived_ec.ne(0))
    startup = jobs.runtime_s.le(600) & abnormal & ~(jobs.state & 0xff).isin([4, 6])
    interaction = (jobs.nodes_alloc.eq(1) & jobs.gpus.le(2) & ~jobs.is_array & jobs.wait_s.le(2)
                   & jobs.runtime_s.gt(600) & jobs.runtime_s.lt(21600)
                   & jobs.finite_timelimit_min.gt(0) & jobs.finite_timelimit_min.le(720)
                   & ~jobs.telemetry_busy)
    jobs['category_id'] = np.select([
        jobs.model.notna(), startup, jobs.telemetry_idle, jobs.user_high,
        jobs.user_near, jobs.similarity_medium, jobs.is_array,
        jobs.nodes_alloc.ge(2) | jobs.gpus.ge(8), interaction,
        jobs.runtime_s.ge(21600), jobs.runtime_s.le(600) | (jobs.partition.eq('test') & jobs.runtime_s.le(3600))],
        ['labelled_dnn', 'startup', 'idle', 'user_dnn_high', 'user_dnn_medium', 'similar_dnn',
         'array', 'large', 'interactive', 'long_single', 'short_test'], default='unknown')
    jobs['category'] = jobs.category_id.map(NAMES)
    jobs['confidence'] = jobs.category_id.map(TIERS)
    jobs['confidence_scope'] = jobs.category_id.map(SCOPES)
    jobs['dnn_evidence'] = np.select([jobs.model.notna(), jobs.user_high, jobs.user_near, jobs.similarity_medium],
                                   ['official_label', 'same_user_high', 'same_user_near', 'positive_similarity'], default='none')
    return jobs


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    emit(args.output, 'analysis-environment.csv', [
        {'component': 'Python', 'version': platform.python_version()},
        {'component': 'OS', 'version': platform.platform()},
        *[{'component': name, 'version': version(name)} for name in
          ['numpy', 'pandas', 'scipy', 'scikit-learn', 'pyarrow', 'requests', 'psutil']]])
    private = args.root / 'derived/other-refine-20261101'
    private.mkdir(parents=True, exist_ok=True)
    source = args.root / 'reference/slurm-log.csv'
    initial = (source.stat().st_size, source.stat().st_mtime_ns)
    expected = json.loads((args.baseline / 'mit-audit.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256()
    with source.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    expected_slurm = next(row['sha256'] for row in expected['inputs'] if Path(row['path']).name == source.name)
    if digest.hexdigest() != expected_slurm:
        raise ValueError('Slurm源文件与PR #26指纹不同')
    all_gpu = load_jobs(args.root)
    total = float(all_gpu.gpu_hours.sum())
    baseline = pd.read_csv(args.baseline / 'mit-classes.csv')
    reconciliation = []
    for name, group in all_gpu.groupby('job_type'):
        old = baseline[(baseline.resource_pool.eq('gpu_hours')) & baseline['class'].eq(name)].iloc[0]
        if len(group) != old['rows'] or not np.isclose(group.gpu_hours.sum(), old.resource_hours, atol=1e-8, rtol=0):
            raise ValueError('PR #26逐类型核验失败')
        reconciliation.append({'job_type': name, 'rows': len(group), 'gpu_hours': group.gpu_hours.sum(),
                               'baseline_rows': old['rows'], 'baseline_gpu_hours': old.resource_hours})
    emit(args.output, 'baseline-reconciliation.csv', reconciliation)
    jobs = all_gpu[all_gpu.job_type.eq('OTHER')].copy()
    del all_gpu
    labels = pd.read_csv(args.root / 'reference/labelled_jobids.csv')
    if labels.id_job.duplicated().any() or len(labels) != 3430:
        raise ValueError('官方标签版本发生变化')
    jobs = jobs.merge(labels, how='left', on='id_job', validate='one_to_one')
    jobs['model_family'] = jobs.model.map(lambda x: family(x) if pd.notna(x) else None)
    known = jobs[jobs.model.notna()]
    if len(known) != 3430 or len(jobs) != 76404:
        raise ValueError('标签匹配或OTHER行数变化')
    other = float(jobs.gpu_hours.sum())
    jobs['has_prior_official_dnn_user'] = False
    for user, references in known.groupby('id_user'):
        selected = jobs.id_user.eq(user)
        times = np.sort(references.time_submit.to_numpy())
        jobs.loc[selected, 'has_prior_official_dnn_user'] = np.searchsorted(
            times, jobs.loc[selected, 'time_submit'].to_numpy(), side='left') > 0
    statistics = pd.read_csv(args.root / 'reference/labelled_job_stats.csv')
    pd.testing.assert_series_equal(labels.model.value_counts().sort_index(),
                                   statistics.set_index('model')['count'].sort_index(), check_names=False)
    print('Feature load complete; evaluating held-out labels', flush=True)
    cross_validation(known, args.output)
    predictions = infer(known, jobs)
    jobs = jobs.join(predictions)
    jobs = attach_telemetry(jobs, private, args.output)
    jobs = classify(jobs)
    result = []
    for category, name, confidence, scope, shift, split in CATEGORIES:
        group = jobs[jobs.category_id.eq(category)]
        result.append({'category_id': category, 'category': name, 'confidence': confidence,
                       'confidence_scope': scope, 'time_shift_assessment': shift, 'split_assessment': split,
                       **summary(group, total, other)})
    emit(args.output, 'other-breakdown.csv', result)
    emit(args.output, 'category-terminal-crosswalk.csv', [
        {'category_id': cat, 'category': NAMES[cat], 'terminal_state': state, **summary(group, total, other)}
        for (cat, state), group in jobs.groupby(['category_id', 'terminal_state'])])
    pd.crosstab(jobs.category, jobs.terminal_state).reindex([row[1] for row in CATEGORIES]).to_csv(
        args.output / 'category-terminal-counts.csv', encoding='utf-8')
    jobs.pivot_table(index='category', columns='terminal_state', values='gpu_hours', aggfunc='sum', fill_value=0).reindex(
        [row[1] for row in CATEGORIES]).to_csv(args.output / 'category-terminal-gpu-hours.csv', encoding='utf-8', float_format='%.12g')
    emit(args.output, 'other-by-state.csv', [
        {'terminal_state': state, **summary(group, total, other)} for state, group in jobs.groupby('terminal_state')])
    emit(args.output, 'category-exit-crosswalk.csv', [
        {'category_id': cat, 'exit_code': ec, 'derived_ec': dec, **summary(group, total, other)}
        for (cat, ec, dec), group in jobs.groupby(['category_id', 'exit_code', 'derived_ec'])])
    confidence_rows = []
    cumulative = jobs.iloc[:0]
    for tier in ['实测', '高', '中', '低']:
        group = jobs[jobs.confidence.eq(tier)]
        cumulative = jobs[jobs.confidence.isin(['实测', '高', '中', '低'][:len(confidence_rows)+1])]
        confidence_rows.append({'confidence': tier, **summary(group, total, other),
                                'cumulative_rows': len(cumulative), 'cumulative_gpu_hours': cumulative.gpu_hours.sum(),
                                'cumulative_pct_mit_gpu_hours': 100*cumulative.gpu_hours.sum()/total,
                                'cumulative_pct_other_gpu_hours': 100*cumulative.gpu_hours.sum()/other})
    emit(args.output, 'confidence-coverage.csv', confidence_rows)
    evidence = []
    for name in ['official_label', 'same_user_high', 'same_user_near', 'positive_similarity', 'none']:
        group = jobs[jobs.dnn_evidence.eq(name)]
        evidence.append({'dnn_evidence': name, **summary(group, total, other)})
    emit(args.output, 'dnn-evidence-coverage.csv', evidence)
    emit(args.output, 'user-propagation-coverage.csv', [
        {'method': name, **summary(jobs.loc[mask], total, other)} for name, mask in [
            ('all_jobs_of_labelled_users', jobs.id_user.isin(known.id_user)),
            ('unlabelled_same_user_high', jobs.model.isna() & jobs.user_high),
            ('unlabelled_same_user_any', jobs.model.isna() & jobs.user_near),
            ('unlabelled_positive_similarity', jobs.model.isna() & jobs.similarity_medium),
        ]])
    diagnostics = []
    for field in ['runtime_h', 'gpus', 'cpu_request', 'memory_request_mb', 'finite_timelimit_min',
                  'wait_h', 'is_array', 'submit_hour_utc', 'submit_weekday_utc', 'has_prior_official_dnn_user']:
        positive = jobs.loc[jobs.model.notna(), field].dropna().astype(float)
        unlabelled = jobs.loc[jobs.model.isna(), field].dropna().astype(float)
        diagnostics.append({'feature': field, 'positive_n': len(positive), 'unlabelled_n': len(unlabelled),
                            'positive_p50': positive.quantile(.5), 'positive_p90': positive.quantile(.9),
                            'unlabelled_p50': unlabelled.quantile(.5), 'unlabelled_p90': unlabelled.quantile(.9),
                            'positive_vs_unlabelled_KS_distance': ks_2samp(positive, unlabelled).statistic})
    emit(args.output, 'positive-unlabelled-feature-diagnostics.csv', diagnostics)
    emit(args.output, 'positive-unlabelled-partitions.csv', [
        {'partition': part, 'positive_fraction': jobs.loc[jobs.model.notna(), 'partition'].eq(part).mean(),
         'unlabelled_fraction': jobs.loc[jobs.model.isna(), 'partition'].eq(part).mean()}
        for part in sorted(jobs.partition.unique())])
    emit(args.output, 'similarity-feature-weights.csv', [{'feature': key, 'weight': value} for key, value in FEATURE_WEIGHTS.items()])
    telemetry_rows = []
    for name, mask in [('all_other', pd.Series(True, index=jobs.index))] + [
            (cat, jobs.category_id.eq(cat)) for cat in NAMES]:
        scope = jobs[mask]
        covered = scope[scope.telemetry_qualified]
        telemetry_rows.append({'scope': name, 'scope_jobs': len(scope), 'scope_gpu_hours': scope.gpu_hours.sum(),
                               'any_telemetry_jobs': int(scope.telemetry_coverage.notna().sum()),
                               'qualified_jobs': len(covered), 'qualified_gpu_hours': covered.gpu_hours.sum(),
                               'qualified_scope_gpu_hour_pct': 100*covered.gpu_hours.sum()/scope.gpu_hours.sum() if len(scope) else np.nan,
                               'utilization_mean_p50': covered.utilization_gpu_pct_mean.quantile(.5),
                               'utilization_mean_p90': covered.utilization_gpu_pct_mean.quantile(.9),
                               'utilization_std_p50': covered.utilization_gpu_pct_std.quantile(.5),
                               'power_W_p50': covered.power_draw_W_mean.quantile(.5),
                               'power_std_W_p50': covered.power_draw_W_std.quantile(.5),
                               'idle_fraction_p50': covered.idle_fraction_le5pct.quantile(.5),
                               'busy_jobs': int(covered.telemetry_busy.sum()), 'idle_jobs': int(covered.telemetry_idle.sum())})
    emit(args.output, 'telemetry-category-validation.csv', telemetry_rows)
    snapshots = pd.read_csv(private / 'existing-telemetry-snapshot.csv')
    first_batch = args.root / 'batches' / snapshots.batch.iloc[0]
    emit(args.output, 'telemetry-inspection.csv', [
        {'source': 'batches/*/' + filename, 'schema_columns': ','.join(pq.ParquetFile(first_batch / filename).schema.names),
         'snapshot_batches': len(snapshots), 'direct_job_id': 'job_id', 'read_only': True,
         'memory_used_MiB_preserved': False}
        for filename in ['files.parquet', 'hourly.parquet', 'util_hist.parquet']])
    emit(args.output, 'rule-definitions.csv', [
        {'precedence': index + 1, 'rule': rule, 'condition': condition} for index, (rule, condition) in enumerate([
            ('labelled_dnn', '官方标签ID精确匹配'),
            ('startup', '运行<=600秒且FAILED/NODE_FAIL/OOM或退出码非零；取消/超时除外'),
            ('idle', '监测覆盖80%-105%，均值利用率<=5%，利用率<=5%的时间占比>=90%'),
            ('user_dnn_high', '同用户/GPU/分区/时限类型；CPU/内存/有限时限差异<=25%；前后90天；正例>=2'),
            ('user_dnn_medium', '同用户/GPU/分区/时限类型；CPU/内存差异<=50%、时限<=2倍；前后90天；正例>=1'),
            ('similar_dnn', '五邻域平均距离<=参考正例留一距离P90，资源兼容；合格监测不得违背正例活动范围'),
            ('array', 'id_array_task!=4294967294'),
            ('large', 'nodes_alloc>=2或GPU>=8'),
            ('interactive', '单节点GPU<=2、非数组、等待<=2秒、600<运行<21600秒、有限时限<=720分钟；排除监测持续繁忙'),
            ('long_single', '运行>=21600秒'),
            ('short_test', '运行<=600秒，或test分区且运行<=3600秒'),
            ('unknown', '其余作业'),
        ])])
    term_rows = []
    for (long, state), group in jobs.groupby([jobs.runtime_s.ge(21600), 'terminal_state']):
        finite = group[group.finite_timelimit_min.notna() & group.finite_timelimit_min.gt(0)]
        ratio = finite.runtime_s / (finite.finite_timelimit_min * 60)
        observed = group[group.telemetry_qualified]
        term_rows.append({'runtime_group': 'at_least_6h' if long else 'under_6h', 'terminal_state': state,
                          **summary(group, total, other), 'official_dnn_jobs': int(group.model.notna().sum()),
                          'inferred_dnn_jobs': int(group.dnn_evidence.isin(['same_user_high', 'same_user_near', 'positive_similarity']).sum()),
                          'finite_timelimit_jobs': len(finite), 'near_limit_95_105pct_jobs': int(ratio.between(.95, 1.05).sum()),
                          'telemetry_qualified_jobs': len(observed), 'telemetry_busy_jobs': int(observed.telemetry_busy.sum()),
                          'telemetry_idle_jobs': int(observed.telemetry_idle.sum()),
                          'observed_mean_utilization_p50': observed.utilization_gpu_pct_mean.quantile(.5),
                          'observed_mean_power_W_p50': observed.power_draw_W_mean.quantile(.5)})
    emit(args.output, 'long-terminal-analysis.csv', term_rows)
    emit(args.output, 'labelled-other-model-families.csv', [
        {'model_family': name, **summary(group, total, other)} for name, group in jobs[jobs.model.notna()].groupby('model_family')])
    emit(args.output, 'labelled-other-model-variants.csv', [
        {'model': name, **summary(group, total, other)} for name, group in jobs[jobs.model.notna()].groupby('model')])
    emit(args.output, 'label-model-rule-crosswalk.csv', [
        {'model_family': name, 'runtime_shape': shape, **summary(group, total, other)}
        for (name, shape), group in jobs[jobs.model.notna()].groupby(['model_family', jobs.runtime_s.ge(21600)])])
    for column in ['partition', 'exit_code', 'derived_ec']:
        emit(args.output, f'other-by-{column}.csv', [{column: name, **summary(group, total, other)} for name, group in jobs.groupby(column)])
    export_columns = ['id_job', 'time_submit', 'time_start', 'time_end', 'runtime_s', 'wait_s', 'gpus',
                      'cpu_request', 'cpu_allocated', 'memory_request_mb', 'nodes_alloc', 'partition', 'is_array',
                      'has_prior_official_dnn_user',
                      'timelimit', 'gpu_hours', 'terminal_state', 'exit_code', 'derived_ec', 'category_id',
                      'category', 'confidence', 'confidence_scope', 'model', 'model_family', 'dnn_evidence',
                      'user_high_support', 'user_near_support', 'similarity_distance', 'similarity_threshold',
                      'similarity_metadata_match', 'similarity_rejected_by_telemetry',
                      'telemetry_coverage', 'telemetry_qualified', 'utilization_gpu_pct_mean', 'utilization_gpu_pct_std',
                      'utilization_memory_pct_mean', 'power_draw_W_mean', 'power_draw_W_std', 'idle_fraction_le5pct']
    exported = jobs[export_columns].copy()
    exported['classification_version'] = 'mit-other-refine-20261101-v2'
    exported.to_parquet(private / 'other-job-categories.parquet', index=False)
    exported.to_csv(private / 'other-job-categories.csv', index=False, encoding='utf-8')
    readback = pd.read_parquet(private / 'other-job-categories.parquet')
    if readback.id_job.duplicated().any() or len(readback) != 76404:
        raise ValueError('逐作业结果读回核验失败')
    saved = pd.read_csv(args.output / 'other-breakdown.csv')
    if saved.rows.sum() != 76404 or not np.isclose(saved.gpu_hours.sum(), other, atol=1e-5, rtol=0):
        raise ValueError('主表计数或GPU小时不守恒')
    if (source.stat().st_size, source.stat().st_mtime_ns) != initial:
        raise ValueError('分析期间输入文件发生变化')
    emit(args.output, 'verification.csv', [{'check': key, 'value': value} for key, value in [
        ('baseline_reconciled', True), ('other_counts_hours_conserved', True), ('private_readback_passed', True),
        ('slurm_source_unchanged', True), ('public_per_job_results', False), ('streaming_database_opened', False),
        ('new_raw_gpu_files_downloaded', 0)]])
    memory = psutil.Process().memory_info()
    peak = memory.peak_wset if os.name == 'nt' else memory.rss
    emit(args.output, 'runtime-verification.csv', [
        {'metric': 'analysis_elapsed', 'value': time.perf_counter()-started, 'unit': 'seconds'},
        {'metric': 'peak_working_set', 'value': peak, 'unit': 'bytes'},
        {'metric': 'csv_chunk_rows', 'value': 20000, 'unit': 'rows'}])
    print(saved[['category', 'confidence', 'rows', 'gpu_hours', 'pct_mit_gpu_hours', 'pct_other_gpu_hours']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
