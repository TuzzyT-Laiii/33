import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from prepare_trn_seren import RUN, TYPES


LABELS = {'main': 'Seren同期逐小时，正文', 'idle50': '反推空闲50W', 'idle70': '反推空闲70W',
    'idle_zero_proxy': '反推零利用率代理67.300W', 'coverage75': '报送覆盖≥75%', 'coverage85': '报送覆盖≥85%',
    'clip_0_1': '比例截断0–1', 'full_observed_plus_profile': '完整作业窗口，观测加日形状补缺',
    'seren_constant_080': '相同Seren需求，恒定0.80', 'seren_080_Kalos_shape': '相同Seren需求，Kalos形状且均值0.80',
    'old_Kalos_constant_080': '原Kalos需求，恒定0.80', 'old_Kalos_shape_080': 'PR #36 Kalos需求及形状'}


def table(frame):
    return frame.to_markdown(index=False, floatfmt='.4f')


def report(args):
    output = args.project / 'results' / RUN
    audit = json.loads((output / 'source-audit.json').read_text('utf-8'))
    calc = json.loads((output / 'calculation-audit.json').read_text('utf-8'))
    validation = json.loads((output / 'validation.json').read_text('utf-8'))
    summary = pd.read_csv(output / 'window-summary.csv')
    comparison = pd.read_csv(output / 'comparison.csv')
    composition = pd.read_csv(output / 'task-composition.csv')
    profile = pd.read_csv(output / 'power-profiles.csv')
    source = pd.read_csv(output / 'source-hourly.csv', parse_dates=['source_clock'])
    qualified = source.loc[source.power_quality_80]
    day = profile.loc[profile.grouping.eq('hour_of_source_day')]
    week = profile.loc[profile.grouping.eq('hour_of_source_week')]
    day_ratio = day.gpu_hour_weighted_mean_ratio.max() / day.gpu_hour_weighted_mean_ratio.min()
    week_ratio = week.gpu_hour_weighted_mean_ratio.max() / week.gpu_hour_weighted_mean_ratio.min()
    weekday = week.assign(weekday=week.position // 24).groupby('weekday').apply(
        lambda g: np.average(g.gpu_hour_weighted_mean_ratio, weights=g.allocated_gpu_hours), include_groups=False)
    pd.DataFrame({'weekday_utc': weekday.index, 'gpu_hour_weighted_mean_ratio': weekday.to_numpy()}).to_csv(output / 'power-weekday.csv', index=False)
    rows = []
    for key in ('B', 'C', 'D'):
        for case in ('old_Kalos_constant_080', 'old_Kalos_shape_080', 'main'):
            row = comparison.loc[comparison.server_type.eq(key) & comparison.reference_it_mw.eq(10) & comparison.case.eq(case)].iloc[0]
            rows.append({'情境': key + '-TRN-10MW', '方案': LABELS[case], '小时': int(row.hours),
                'IT均值MW': row.mean_it_mw, 'IT峰值MW': row.peak_it_mw, 'IT最小MW': row.minimum_it_mw,
                'IT日内比': row.it_diurnal_peak_valley, 'IT周内比': row.it_weekly_peak_valley,
                '设施均值MW': row.mean_facility_mw, '设施峰值MW': row.peak_facility_mw, '设施最小MW': row.minimum_facility_mw,
                '设施日内比': row.facility_diurnal_peak_valley, '设施周内比': row.facility_weekly_peak_valley})
    main_table = pd.DataFrame(rows)
    main_table.to_csv(output / 'comparison-10MW.csv', index=False)
    sensitivity = summary.loc[summary.server_type.eq('D') & summary.reference_it_mw.eq(10)].copy()
    sensitivity['方案'] = sensitivity.case.map(LABELS)
    sensitivity = sensitivity[['方案', 'hours', 'allocated_power_ratio_gpu_hour_mean', 'mean_it_mw', 'peak_it_mw',
        'minimum_it_mw', 'it_diurnal_peak_valley', 'it_weekly_peak_valley', 'mean_facility_mw', 'imputed_hours', 'ratio_above_tdp_hours']]
    sensitivity.to_csv(output / 'sensitivity-10MW.csv', index=False)
    split = composition.pivot(index='type', columns='window', values='gpu_share').reindex(TYPES) * 100
    split = split.rename(columns={'full_jobs': '完整窗口GPU小时占比%', 'qualified_overlap': '正文窗口GPU小时占比%'}).reset_index()
    sensitivity_display = sensitivity.rename(columns={'hours': '小时', 'allocated_power_ratio_gpu_hour_mean': 'GPU分配加权比例',
        'mean_it_mw': 'IT均值MW', 'peak_it_mw': 'IT峰值MW', 'minimum_it_mw': 'IT最小MW',
        'it_diurnal_peak_valley': 'IT日内比', 'it_weekly_peak_valley': 'IT周内比', 'mean_facility_mw': '设施均值MW',
        'imputed_hours': '补功率小时', 'ratio_above_tdp_hours': '比例超过1的小时'})
    p = audit['central_power_statistics']
    total_overlap_bins = int(source.overlap_samples.notna().sum())
    text = f'''# 大模型训练/开发情境：Acme Seren原生窗口

本版基于PR #36的`{audit['input_commit']}`，将B/C/D-TRN四档规模的数据源确定为**Acme Seren**。情境名称不变。依赖PR #34的原生窗口与天气链路、PR #36的CPU/基础功率审计；审阅顺序为#34 → #36 → 本PR。INF的#35可按其依赖单独审阅，组合时保留各用途的新输入并重算组合结果。

## 窗口与证据边界

正文采用同期需求可观测且功率监测通过筛选的真实小时。Seren原始作业为{audit['rows']:,}行，其中GPU作业{audit['gpu_jobs']:,}条、独立CPU作业{audit['independent_cpu_jobs']:,}条。作业起止UTC {audit['raw_job_start']}至{audit['raw_job_end']}。全作业表保留{audit['full_source_hours']:,}个小时分组（{audit['full_source_hours']/24:.4f}个等效日），包括首尾部分小时；窗口不循环、不延长、不年化。

监测日历覆盖UTC **{audit['calendar_overlap_start']}至{audit['calendar_overlap_end']}**，约93.595天，涉及{total_overlap_bins:,}个小时分组。按2,288张A100物理库存的报送覆盖至少80%、每小时至少228个15秒采样（95%）、分配量为正且采样中没有库存超额筛选。正文保留**{audit['qualified_hours']:,}个小时分组**（{audit['qualified_hours']/24:.4f}个等效日），实际采样{audit['qualified_sample_equivalent_hours']:.4f}小时。

正文首小时为**{audit['qualified_start']}**，最后小时为**{audit['qualified_end_inclusive']}**。两端之间排除{audit['interior_excluded_hours']}小时，形成{audit['contiguous_segments']}个连续片段，最长{audit['longest_segment_hours']}小时。全部排除小时与原因见[缺口表](body-excluded-hours.csv)，片段见[区间表](body-segments.csv)。缺口不插值、不延续前值、不压缩真实时间；正文电量仅积分保留的小时分组。每小时采样最多允许3分钟缺失，因此小时均值还依赖小时内部缺测的代表性。

本版选择同期合格窗口，是因为每个正文小时都具有同时间的作业分配和功率监测支持。GPU功率仍是**集群条件反推**：报送状态无法确认在线状态，且没有逐作业设备映射。正文不能解释为逐预训练作业功率直接实测。完整作业窗口用于检验更长需求期及任务构成的影响。

## 需求与功率方法

按作业`[start_time,end_time)`与小时的实际交集累计GPU小时、GPU作业宿主CPU核心小时及各任务类型。独立CPU作业单列诊断，保持原Kalos处理方式，不并入TRN功率。GPU需求除以完整作业表瞬时并发峰值**{audit['concurrent_gpu_peak_normalizer']:.0f}张**；同时间事件先合并。该归一化沿用Kalos口径，用于将需求迁移到给定设备规模，区别于反推时采用的2,288张物理库存。

CPU功率仍沿用PR #36：`u_cpu_proxy=0.25×GPU分配份额`及SPECpower平滑曲线。实际宿主CPU分配共{audit['host_cpu_core_hours']:,.2f}核心小时，独立CPU为{audit['independent_cpu_core_hours']:,.2f}核心小时，均已输出；它们是分配量，不是CPU实测利用率。由同一GPU比例构造的CPU代理没有另行添加独立CPU业务。

每个15秒时刻采用`P_est=报送GPU平均功率×2288`，再按小时对齐分配与功率。已分配GPU比例为

`r(t)=[P_est(t)−(2288−N_alloc(t))×P_idle]/[N_alloc(t)×400W]`。

默认空闲60W。主方案直接采用同小时`r(t)`，不使用Kalos 271卡日形状，也不使用文献0.80。源设备为A100-SXM 80GB、400W；目标B/C使用H100 PCIe、350W，D使用H100 SXM、700W，已分配GPU功率为`目标GPU数量×分配份额×目标TDP×r(t)`。未分配目标GPU空闲仍为PCIe 50W、SXM 70W。敏感性50/60/70W改变的是**源集群反推的空闲假设**，不同时改变目标设备空闲端点。

按比例跨A100/H100及PCIe/SXM迁移是工程假设，未保持同等吞吐、运行时长或训练进度。各类型同小时共用集群反推比例；任何类型分项功率都是按分配占比分摊的模型值，无法独立标定预训练、MLLM、SFT或评估功率。

### 逐小时比例及日周形状

GPU采样时间加权均值**{p['gpu_hour_weighted_mean_ratio']:.6f}**；P10/中位数/P90为**{p['p10']:.6f}/{p['median']:.6f}/{p['p90']:.6f}**；小时范围{p['minimum']:.6f}—{p['maximum']:.6f}。{p['above_tdp_hours']}个小时超过1、无负值，正文保留并标记，截断方案单列。它们也可能反映缺测外推或库存假设误差，不能直接证明板卡持续超过额定TDP。

24个UTC小时分组的GPU时间加权比例峰谷比为**{day_ratio:.6f}**，最低UTC{int(day.loc[day.gpu_hour_weighted_mean_ratio.idxmin(), 'position'])}时、最高{int(day.loc[day.gpu_hour_weighted_mean_ratio.idxmax(), 'position'])}时。168个源星期小时分组的峰谷比为**{week_ratio:.6f}**，每个分组有{int(week.hours.min())}—{int(week.hours.max())}个观测小时。按七个星期日汇总的峰谷比为{weekday.max()/weekday.min():.6f}。这些形状是同期混合负载条件统计，包含作业构成与分配变化，不能归因于检查点或认为它们是稳定的预训练自然日周期。主方案逐小时使用原序列，日周分组仅用于展示。

### 任务构成

{table(split)}

完整窗口与正文窗口的预训练份额不同，不能仅用更长窗口的69.5%描述正文。MLLM作为独立层保留；标签不自动赋予可延迟或可暂停权限。[任务构成](task-composition.csv)同时包含宿主CPU分项，[10MW各方案分项](task-power-composition.csv)保留模型GPU分摊与计量边界。

## IT与设施结果

沿用PR #34的源UTC月日小时对应目标气象年：参考网格2025年，八城市2023/2024/2025年，共{calc['location_years']}个地点年份；使用相同冷却v2参数和物理热平衡。Seren日期不移动到旧Kalos窗口。冷却模型逐小时无记忆，新增显式`allow_gaps=True`只允许稀疏真实小时输入，不改变公式；默认连续性检查保留。

下表为参考网格10MW；四档规模、全部地点年份和方案见[全量汇总](window-summary-all-locations.csv)。旧Kalos表使用PR #36已发布的相同天气版本、原生90天结果。两者来源、归一化、窗口和任务构成都变化，因此表中差异是描述性结果。

{table(main_table)}

日内比为24个源UTC小时均值的最高/最低；周内比为168个源星期小时均值的最高/最低。它们与窗口内任意小时最大/最小不同。窗口电量为实际保留小时功率之和，不外推全年。正文是非连续观测样本，不能将其缺口当作零负荷输入下游优化。

## 对照与敏感性

完整作业表方案保留同期合格小时的直接反推；其余小时按合格窗口的GPU采样时间加权UTC小时均值补功率。补功率包括监测窗外及未通过筛选的小时，均有`power_imputed`标记。该方案没有额外星期修正；用同一日形状补缺依赖其对不同任务构成和日期仍有代表性。小时文件中的`power_ratio_above_tdp`保留源60W反推诊断，表内“比例超过1的小时”按各方案实际采用的比例统计。

空闲代理67.300W来自利用率为零的配对观测，设备是否未分配未知。覆盖阈值75%/85%同时改变选中窗口，不能将均值变化全归因于测量精度。比例截断采用`clip(0,1)`，保留所有正文小时。相同Seren窗口下的0.80恒定及Kalos相对形状对照保持需求不变；后者按该窗口GPU小时重新归一化至均值0.80。

{table(sensitivity_display)}

## 图件、核验与文件

- `figure1-source`：原生分配、功率比例及报送覆盖，缺口保留。
- `figure2-comparison`：B/C/D典型日的IT与设施新旧对比。
- `figure3-shapes-sensitivity`：日周功率形状、D-TRN敏感性及包含MLLM的任务构成。

图件为中文PNG（300dpi）和嵌入字体PDF；程序检查字形覆盖、边界和字体嵌入，按用户规定不调用视觉检查。分组条件均值未提供因果或置信区间解释。

验证已读取并核对{validation['hourly_files_sha256_verified']}个小时文件、{validation['saved_hour_rows_verified']:,}条保存小时记录及{validation['all_summary_rows']:,}个汇总行；检查输入哈希、类型GPU/宿主CPU守恒、小时电功率与热平衡、真实日历/缺口、日周汇总及PR #36基础/CPU/设备参数一致。真实旧版全年天气输入复现设施功率，允许缺口的结果与完整序列同小时子集一致，默认接口仍拒绝缺口。

原始作业与GPU监测只留服务器共享卷。Git保存汇总、图件、来源哈希和审计；逐小时IT/设施Parquet留在服务器`results/{RUN}/hourly/`，其SHA-256见[计算清单](calculation-audit.json)。分析一次只处理一个情境、地点、年份和方案，不合并全量小时数据到内存。没有操作MIT源目录、流式进程或STOP文件。

复算入口为`analysis/launch_trn_seren.sh`；流程为`prepare_trn_seren.py` → `build_trn_seren.py` → `verify_trn_seren.py` → `plot_trn_seren.py` → `report_trn_seren.py`。依赖见`analysis/requirements-trn-seren.txt`。原始数据入口和哈希沿用[PR #36证据审计](../trn-power-model-20261101/job-power-evidence/README.md)，官方来源为[AcmeTrace](https://huggingface.co/datasets/Qinghao/AcmeTrace)。HPC、CLD、SAI、INF及既有历史结果保持各自版本。
'''
    (output / 'README.md').write_text(text, encoding='utf-8')
    print(main_table[['情境', '方案', 'IT均值MW', 'IT峰值MW', 'IT日内比', 'IT周内比']].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    report(parser.parse_args())
