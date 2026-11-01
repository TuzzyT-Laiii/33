import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compute_load.model import sha256
from load_curve_scenarios_v3 import RUN, catalogue, identity, power


def summarize(project):
    output = project / 'results' / RUN
    configs = catalogue(project)
    ratios = json.loads((project / 'results/typical-load-curves-20261029-v2/mit-power-ratio-summary.json').read_text('utf-8'))['transfer_distribution']
    mit_path = output / 'mit-calibration.json'
    mit = json.loads(mit_path.read_text('utf-8')) if mit_path.exists() else None
    cld = pd.read_csv(project / 'results/alibaba-cpu-cld-20261031/complete-component-window.csv')
    typed = pd.read_parquet(project / 'results/asi-185day-workload-types-20261031/hourly-type-priority-model.parquet')
    inference = typed.job_type_public.isin(['online_inference', 'offline_inference'])
    offline_lp = typed.job_type_public.eq('offline_inference') & typed.priority_class.eq('LP')
    source = pd.read_csv(project / 'results/asi-185day-workload-types-20261031/inf-hourly.csv')
    np.testing.assert_array_equal(np.sort(typed.hour_index.unique()), np.arange(4440))
    combined = typed.loc[inference].groupby('hour_index').used_gpu_hours.sum().reindex(range(4440), fill_value=0)
    np.testing.assert_allclose(combined, source.combined_used_gpu_hours, rtol=1e-12, atol=1e-6)
    lp = typed.loc[offline_lp].groupby('hour_index').used_gpu_hours.sum().reindex(range(4440), fill_value=0).to_numpy()
    inf_hours = float(typed.loc[inference, 'used_gpu_hours'].sum())
    offline_hours = float(typed.loc[typed.job_type_public.eq('offline_inference'), 'used_gpu_hours'].sum())
    lp_hours = float(lp.sum())
    total_core = float(cld.occupied_core_hours.sum())
    source_stats = {'CLD': {'source_core_hours': total_core,
        'online_share': float(cld.online_request_based_core_hours.sum() / total_core),
        'batch_share': float(cld.batch_mean_core_hours.sum() / total_core),
        'unreconciled_signed_share': float(cld.residual_core_hours.sum() / total_core)},
        'INF': {'source_inference_gpu_hours': inf_hours, 'offline_share_of_inference': offline_hours / inf_hours,
                'offline_lp_share_of_offline': lp_hours / offline_hours, 'offline_lp_share_of_inference': lp_hours / inf_hours}}
    durations = pd.read_csv(project / 'results/job-flexibility-constraints-20261030/mit-durations.csv')
    gpu = durations.loc[durations.resource_pool.eq('gpu_hours')]
    batch = gpu['class'].isin(['LLSUB:BATCH', 'LLMAPREDUCE:MAP'])
    long = gpu.runtime_band.isin(['6-24h', '>=24h'])
    all_gpu_hours = float(gpu.resource_hours.sum())
    source_stats['SAI'] = {'reference_gpu_hours': all_gpu_hours,
        'batch_map_share': float(gpu.loc[batch, 'resource_hours'].sum() / all_gpu_hours),
        'runtime_ge6h_share': float(gpu.loc[long, 'resource_hours'].sum() / all_gpu_hours),
        'batch_map_runtime_ge6h_share': float(gpu.loc[batch & long, 'resource_hours'].sum() / all_gpu_hours),
        'scope': '既有MIT GPU作业总体分类；未与237天重放逐小时匹配，不乘成已证实的年度可调功率'}
    server_durations = pd.read_csv(project / 'results/job-flexibility-constraints-20261030/server-durations.csv')
    hpc = server_durations.loc[server_durations.dataset.eq('Lisa') & server_durations.resource_pool.eq('cpu_hours')]
    hpc_screen = hpc.runtime_band.isin(['6-24h', '>=24h']) & ~hpc['class'].str.contains('jupyter')
    source_stats['HPC'] = {'reference_core_hours': float(hpc.resource_hours.sum()),
        'runtime_ge6h_non_jupyter_share': float(hpc.loc[hpc_screen, 'resource_hours'].sum() / hpc.resource_hours.sum()),
        'scope': '既有Lisa CPU作业总体分类；未与309天重放逐小时匹配，运行时长不等于允许推迟时长'}
    bounds = pd.read_csv(project / 'results/job-flexibility-constraints-20261030/candidate-bounds.csv')
    trn_share = float(bounds.loc[bounds.screen.eq('pretrain_observed_runtime_ge1h'), 'candidate_ceiling_fraction'].iloc[0])
    rows = []
    for path in sorted((output / 'hourly').glob('*.parquet')):
        frame = pd.read_parquet(path)
        key, use = frame.server_type.iloc[0], frame.use.iloc[0]
        scale = int(frame.reference_it_mw.iloc[0])
        index = frame.source_hour_index.to_numpy()
        candidate = {'CLD': source_stats['CLD']['batch_share'], 'INF': lp_hours / inf_hours,
                     'SAI': source_stats['SAI']['batch_map_runtime_ge6h_share'], 'TRN': trn_share,
                     'HPC': source_stats['HPC']['runtime_ge6h_non_jupyter_share']}[use]
        row = {**identity(key, use, scale), 'candidate_resource_fraction': candidate,
               'candidate_resource': 'CPU核心小时' if key == 'A' else 'GPU小时',
               'verified_dispatchable_fraction': 0.,
               'screen': {'CLD': '离线实例CPU执行量', 'INF': '离线推理且LP', 'SAI': '明确批处理/Map且运行至少6小时',
                          'TRN': '预训练且运行至少1小时', 'HPC': '非Jupyter队列且运行至少6小时的CPU科研作业'}[use],
               'interpretation': '资源分类候选；期限、依赖、保存恢复成本与操作权限未获验证；未运行优化'}
        if use in ('CLD', 'INF'):
            cpu, gpu = frame.u_cpu_proxy.to_numpy(), frame.allocated_gpu_fraction.to_numpy()
            if use == 'CLD':
                batch_share = cld.batch_share_of_machine.to_numpy()[index]
                if ((batch_share < 0) | (batch_share > 1)).any():
                    raise ValueError('CLD 批处理小时份额超出功率反事实有效范围')
                candidate_cpu, candidate_gpu = cpu * (1 - batch_share), gpu
                row['annual_candidate_resource_fraction'] = float(np.sum(cpu * batch_share) / np.sum(cpu))
            else:
                lp_fraction = lp[index] / float(source.combined_used_gpu_hours.max())
                if (lp_fraction > gpu + 1e-10).any():
                    raise ValueError('离线LP量超过同小时推理资源总量')
                candidate_gpu = np.maximum(gpu - lp_fraction, 0)
                candidate_cpu = .25 * candidate_gpu
                row['annual_candidate_resource_fraction'] = float(np.sum(lp_fraction) / np.sum(gpu))
            before = power(configs[key], key, use, cpu, gpu, ratios, mit)['unit_w']
            after = power(configs[key], key, use, candidate_cpu, candidate_gpu, ratios, mit)['unit_w']
            decrement = (before - after) * int(frame.unit_count.iloc[0]) / 1e6
            if (decrement < -1e-9).any():
                raise ValueError('中央候选功率反事实产生负削减量')
            row.update(illustrative_mean_it_decrement_mw=float(decrement.mean()),
                       illustrative_peak_it_decrement_mw=float(decrement.max()),
                       illustrative_it_energy_decrement_share=float(decrement.sum() / frame.it_mw.sum()),
                       decrement_scope='同小时移出全部候选资源、保持设备通电；推迟后的补做与容量约束未计算，不能解释为节电或可保证削峰')
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / 'flexibility-hints.csv', index=False, encoding='utf-8-sig')
    source_stats['inputs_sha256'] = {str(path.relative_to(project)): sha256(path) for path in [
        project / 'results/alibaba-cpu-cld-20261031/complete-component-window.csv',
        project / 'results/asi-185day-workload-types-20261031/hourly-type-priority-model.parquet',
        project / 'results/job-flexibility-constraints-20261030/mit-durations.csv',
        project / 'results/job-flexibility-constraints-20261030/server-durations.csv']}
    (output / 'flexibility-source-statistics.json').write_text(json.dumps(source_stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(source_stats, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    summarize(parser.parse_args().project.resolve())
