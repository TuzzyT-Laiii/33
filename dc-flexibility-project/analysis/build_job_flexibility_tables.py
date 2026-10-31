import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def markdown_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def number(value):
    return '缺失' if pd.isna(value) else f'{value:,.2f}'


def triple(row, prefix):
    return '/'.join(number(row[prefix + '_' + q]) for q in ['p50', 'p90', 'p99'])


def build(root):
    inputs = [root / name for name in ['server-classes.csv', 'mit-classes.csv', 'asi-classes.csv',
               'asi-priorities.csv', 'asi-hourly-type-priority.csv', 'asi-workload-scales.csv',
               'asi-training-model-types.csv', 'asi-model-type-priority.csv',
               'asi-training-model-audit.json', 'server-audit.json', 'mit-audit.json']]
    jobs = pd.concat([pd.read_csv(root / f'{prefix}-classes.csv') for prefix in ['server', 'mit']], ignore_index=True)
    sections = ['# 逐类别统计表', '',
        '作业分位数按条数计算，使用 pandas 线性插值；P99 对 SFT 的8条记录等小样本仅为描述值。资源份额按同数据集、同资源池计算，CPU、GPU及GPU宿主CPU分别保留分母。时长和排队单位均为秒，规模为GPU张数或源CPU调度单位。', '']
    for (dataset, pool), group in jobs.groupby(['dataset', 'resource_pool']):
        sections.extend([f'## {dataset} / {pool}', '',
            markdown_table(['原始类别', '条数', '同池条数%', '资源小时', '同池资源%',
                            '运行P50/P90/P99（秒）', '排队P50/P90/P99（秒）'],
                [[r['class'], int(r['rows']), number(r.row_fraction_of_resource_pool * 100),
                  number(r.resource_hours), number(r.work_fraction_of_resource_pool * 100),
                  triple(r, 'runtime_s'), triple(r, 'wait_s')] for _, r in group.iterrows()]), '',
            markdown_table(['原始类别', '资源数量P50/P90/P99/最大', '节点数P50/P90/P99/最大',
                            '有效/无效运行记录', '<60秒条数%', '<60秒资源%', '≥1小时资源%'],
                [[r['class'], triple(r, 'units') + '/' + number(r.units_max),
                  triple(r, 'nodes') + '/' + number(r.nodes_max),
                  f'{int(r.valid_runtime_rows)}/{int(r.invalid_runtime_rows)}',
                  number(r.runtime_under60_row_fraction * 100), number(r.runtime_under60_work_fraction * 100),
                  number(r.runtime_ge1h_work_fraction * 100)] for _, r in group.iterrows()]), ''])
    asi = pd.read_csv(root / 'asi-classes.csv')
    sections.extend(['## Alibaba ASI / 相对日109', '',
        '计数单位为GPU pod小时观测；distinct_pods为窗口内去重pod，无法还原完整作业数。运行时长P50/P90/P99均缺失。申请规模按pod/server/小时记录；排队按每类每pod首次观测的非负schedule_delay_sec计算，不能视为SLA。', '',
        markdown_table(['类型', 'pod小时观测', '观测%', '去重pod', 'GPU小时', 'GPU小时%',
                        '申请GPU P50/P90/P99/最大', '调度延迟P50/P90/P99（秒）'],
            [[r['class'], int(r.pod_hour_observations), number(r.observation_fraction * 100),
              int(r.distinct_pods), number(r.used_gpu_hours), number(r.gpu_hour_fraction * 100),
              triple(r, 'gpu_request_per_pod_server_hour') + '/' + number(r.gpu_request_per_pod_server_hour_max),
              triple(r, 'first_observed_pod_schedule_delay_s')] for _, r in asi.iterrows()]), '',
        'workload同小时聚合的规模另见 [asi-workload-scales.csv](asi-workload-scales.csv)；其中包含多pod、多服务器服务，不能自动解释为同步训练规模。', ''])
    training_models = pd.read_csv(root / 'asi-training-model-types.csv')
    sections.extend(['### ASI training内的模型领域', '',
        'model_type_public是模型领域标签，无法识别预训练、SFT或RL算法。', '',
        markdown_table(['模型领域', 'pod小时观测', 'GPU小时', 'training内GPU小时%'],
            [[r.model_type_public, int(r.pod_hour_observations), number(r.used_gpu_hours),
              number(r.gpu_hour_fraction_within_training * 100)] for _, r in training_models.iterrows()]), ''])
    (root / 'statistics.md').write_text('\n'.join(sections), encoding='utf-8')
    priorities = pd.read_csv(root / 'asi-priorities.csv')
    hourly = pd.read_csv(root / 'asi-hourly-type-priority.csv')
    if set(hourly.hour) != set(range(24)) or hourly.relative_day.ne(109).any():
        raise ValueError('ASI小时汇总必须覆盖相对日109的24小时')
    np.testing.assert_allclose(hourly.used_gpu_hours.sum(), asi.used_gpu_hours.sum(), rtol=1e-11)
    hp = hourly.groupby('hour').used_gpu_hours.sum().rename('all_gpu_hours')
    chosen = hourly.priority_class.eq('LP') & hourly.job_type_public.isin(['training', 'offline_inference'])
    hourly_bounds = hp.to_frame().join(hourly[chosen].groupby('hour').used_gpu_hours.sum().rename('lp_training_offline_gpu_hours'))
    if hourly_bounds.isna().any().any():
        raise ValueError('ASI小时候选汇总出现缺失')
    hourly_bounds['candidate_fraction'] = hourly_bounds.lp_training_offline_gpu_hours / hourly_bounds.all_gpu_hours
    hourly_bounds['source_type'] = '本数据实测汇总+类别筛选定义；可恢复率未知'
    hourly_bounds.reset_index().to_csv(root / 'hourly-candidate-bounds.csv', index=False)
    bounds = []

    def bound(dataset, name, selected_hours, all_hours, assumption):
        bounds.append({'dataset': dataset, 'screen': name, 'candidate_resource_hours': float(selected_hours),
            'resource_pool_hours': float(all_hours), 'candidate_ceiling_fraction': float(selected_hours / all_hours),
            'verified_dispatchable_lower_fraction': 0, 'source_type': '本数据聚合+分类筛选定义',
            'interpretation': assumption})

    total_asi = asi.used_gpu_hours.sum()
    selected = priorities['class'].isin(['training', 'offline_inference'])
    bound('Alibaba_ASI_day109', 'LP_training_and_offline', priorities.loc[selected & priorities.priority.eq('LP'), 'used_gpu_hours'].sum(), total_asi,
          '作为低优先级可回收候选的筛选上界；checkpoint、恢复和期限仍未知')
    bound('Alibaba_ASI_day109', 'all_training_and_offline', priorities.loc[selected, 'used_gpu_hours'].sum(), total_asi,
          '若追加HP/Other业务授权审阅时的类型候选上界；不代表已知允许延期')
    for label in ['training', 'offline_inference']:
        selected_type = priorities['class'].eq(label)
        bound('Alibaba_ASI_day109', 'LP_within_' + label,
              priorities.loc[selected_type & priorities.priority.eq('LP'), 'used_gpu_hours'].sum(),
              priorities.loc[selected_type, 'used_gpu_hours'].sum(), '同类型内LP份额；技术可恢复率未知')
    acme = jobs[jobs.dataset.eq('Acme_Kalos') & jobs.resource_pool.eq('gpu_hours')]
    bound('Acme_Kalos', 'pretrain', acme.loc[acme['class'].eq('Pretrain'), 'resource_hours'].sum(), acme.resource_hours.sum(),
          '整作业延后或完整gang暂停的类型上界；有效截止时间和checkpoint未知')
    bound('Acme_Kalos', 'pretrain_observed_runtime_ge1h',
          (acme.loc[acme['class'].eq('Pretrain'), 'resource_hours'] * acme.loc[acme['class'].eq('Pretrain'), 'runtime_ge1h_work_fraction']).sum(),
          acme.resource_hours.sum(), '运行至少一小时的预训练资源份额；连续段最短长度仍须另设')
    bound('Acme_Kalos', 'eval_and_sft', acme.loc[acme['class'].isin(['Eval', 'SFT']), 'resource_hours'].sum(), acme.resource_hours.sum(),
          '整作业启动延后候选；训练反馈依赖和真实期限未知')
    mit = jobs[jobs.dataset.eq('MIT_Supercloud')]
    for pool, group in mit.groupby('resource_pool'):
        bound('MIT_Supercloud', 'batch_and_map_' + pool,
              group.loc[group['class'].isin(['LLSUB:BATCH', 'LLMAPREDUCE:MAP']), 'resource_hours'].sum(), group.resource_hours.sum(),
              '仅明确批处理和Map提交方式的筛选上界；OTHER不据此推定用途')
    pd.DataFrame(bounds).to_csv(root / 'candidate-bounds.csv', index=False)
    sensitivity = []
    for total_overhead_min in [1, 5, 10, 30]:
        for epsilon in [.01, .05, .10]:
            useful = (1 - epsilon) / epsilon * total_overhead_min
            sensitivity.append({'source_type': '研究假设+公式推导', 'overhead_min': total_overhead_min,
                'max_overhead_fraction': epsilon, 'min_useful_segment_min': useful,
                'min_elapsed_segment_min': useful + total_overhead_min,
                'hourly_elapsed_round_up': int(np.ceil((useful + total_overhead_min) / 60)),
                'definition': 'H含阻塞保存、重启及额外重算；H/(L+H)≤epsilon；L为有效计算分钟'})
    pd.DataFrame(sensitivity).to_csv(root / 'checkpoint-sensitivity.csv', index=False)
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'scope': '基于汇总生成统计表、类别筛选上界、明确标注为假设的开销敏感性；没有生成或发布逐作业数据'}
    (root / 'tables-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(pd.DataFrame(bounds)[['dataset', 'screen', 'candidate_ceiling_fraction']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    build(args.results)
