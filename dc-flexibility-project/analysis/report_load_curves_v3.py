import argparse
import json
from pathlib import Path

import pandas as pd

from load_curve_scenarios_v3 import RUN


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def report(project):
    output = project / 'results' / RUN
    manifest = json.loads((output / 'manifest.json').read_text('utf-8'))
    summary = pd.read_csv(output / 'annual-summary.csv')
    comparison = pd.read_csv(output / 'v2-v3-comparison.csv')
    sensitivity = pd.read_csv(output / 'sensitivity-summary.csv')
    flex = json.loads((output / 'flexibility-source-statistics.json').read_text('utf-8'))
    publication_path = output / 'server-publication.json'
    publication_text = ''
    if publication_path.exists():
        publication = json.loads(publication_path.read_text('utf-8'))
        publication_text = (f"服务器正式结果目录：`{publication['results_directory']}`；城市小时文件目录："
                            f"`{publication['processed_directory']}`。发布后已读回核对 {publication['hourly_files_checked']} 个小时文件的 SHA-256，"
                            "记录见 [server-publication.json](server-publication.json)。")
    mit_path = output / 'mit-calibration.json'
    mit = json.loads(mit_path.read_text('utf-8')) if mit_path.exists() else None
    selected = summary.loc[summary.reference_it_mw.eq(10)]
    rows = []
    for row in selected.itertuples():
        change = comparison.loc[comparison.scenario_id.eq(row.scenario_id)].iloc[0]
        delta = f'{change.mean_facility_change_pct:+.2f}%' if change.v2_available else 'v2 待数据；新增'
        rows.append([row.scenario_id, f'{row.mean_it_mw:.3f} / {row.peak_it_mw:.3f}',
                     f'{row.mean_facility_mw:.3f} / {row.peak_facility_mw:.3f}',
                     f'{row.it_diurnal_peak_valley:.3f} / {row.it_weekly_peak_valley:.3f}',
                     f'{row.facility_diurnal_peak_valley:.3f} / {row.facility_weekly_peak_valley:.3f}', delta])
    ten_table = table(['情境（10MW）', 'IT 年均 / 峰值 MW', '设施年均 / 峰值 MW', 'IT 日内 / 周内比', '设施日内 / 周内比', '设施年均相对 v2'], rows)
    full_table = table(['情境', 'IT 年均 MW', 'IT 峰值 MW', '设施年均 MW', '设施峰值 MW', '设施日内比', '设施周内比'],
        [[r.scenario_id, f'{r.mean_it_mw:.4f}', f'{r.peak_it_mw:.4f}', f'{r.mean_facility_mw:.4f}', f'{r.peak_facility_mw:.4f}',
          f'{r.facility_diurnal_peak_valley:.4f}', f'{r.facility_weekly_peak_valley:.4f}'] for r in summary.itertuples()])
    (output / 'all-44-scenarios.md').write_text('# 全部情境的年度功率与目标日历形状\n\n参考网格，2025 年 UTC，中央风冷参数；规模为安装参考 IT 预算。日内与周内比均以分组均值的最大值除最小值。\n\n' + full_table + '\n', encoding='utf-8')
    city = pd.read_csv(output / 'annual-summary-all-locations.csv')
    city = city.loc[~city.location_id.eq('reference-grid') & city.reference_it_mw.eq(10)]
    city_rows = []
    for name, group in city.groupby('location_zh', sort=False):
        city_rows.append([name, f'{group.year.min()}–{group.year.max()}',
                          f'{group.mean_facility_mw.min():.3f}–{group.mean_facility_mw.max():.3f}',
                          f'{group.peak_facility_mw.min():.3f}–{group.peak_facility_mw.max():.3f}',
                          f'{group.annual_pue.min():.3f}–{group.annual_pue.max():.3f}'])
    city_table = table(['城市', '气象年', '各情境年均设施 MW 范围', '各情境设施峰值 MW 范围', 'PUE 范围'], city_rows)
    calibration = '科研 AI 等待六月正式月度结果；本阶段仅完成其余 32 个情境。'
    sensitivity_text = ''
    if mit:
        validation_wape = mit['held_out_allocated_comparison']['WAPE']
        validation_wape_text = f'{100*validation_wape:.2f}%' if validation_wape is not None else '无法计算（没有合格验证小时）'
        calibration = f'''科研 AI 使用 **{mit['release_status']}**，`partial_data={str(mit['partial_data']).lower()}`，完成 {mit['completed_objects']:,}/{mit['total_objects']:,} 个文件。前半月标定分配 GPU 的时间加权平均功率为 **{mit['allocated_power_W_per_gpu']:.3f} W/张**，参考 250 W 的比例为 **{mit['measured_power_to_tdp_ratio']:.4f}**；已分配且利用率为零的空转代理为 **{mit['idle_W_per_gpu']:.3f} W/张**。完整月观测范围内，设备小时平均功率低于 50 W 的时间加权占比为 **{100*mit['whole_month_weighted_device_hour_mean_below_50W_share']:.2f}%**，该比例的单位为设备小时平均功率。

有效监测 GPU 小时为 {mit['measured_gpu_hours']:,.3f}，相对月内分配 GPU 小时的覆盖为 {100*mit['monitoring_coverage']:.2f}%；疑似歧义设备小时 {mit['ambiguous_device_hours']:,}。文件完成情况与有效监测覆盖分别记录。后半月达 99% 覆盖的合格小时为 {mit['qualified_hours']}；物理全集群完整实测小时为 {mit['fully_measured_cluster_hours']}。缺失观测和持续未分配 GPU 没有填零。

合格验证仅 {mit['qualified_hours']} 小时，分配功率 WAPE 为 {validation_wape_text}，不足以支持完整日内、周内或全年精度结论。包含同一空载代理的集群误差不用于验证空载假设；目标 H100 仍缺少匹配实测。

六月数据只用于功率标定。科研 AI 仍使用原有 237 天 GPU 与宿主 CPU 分配序列，并分别保持原归一化分母：GPU 小时峰值 428.92 张、CPU 小时峰值 7,394.9089 个 Slurm 调度单位。归一化份额用于参考配置容量映射；没有改为按源集群 448 张物理库存归一化。没有把六月日内/周内功率形状再乘到分配序列上，避免重复叠加需求波动；月度形状的有效范围见聚合 CSV。此前分配量只表示调度占用，v2 已使用 1,574 文件随机前缀的固定中位比例；v3 同时更新窗口、均值统计与空载假设，全部差异不能解释成单一的“实测纠偏”。'''
        sensitivity_rows = []
        for key, group in sensitivity.loc[sensitivity.use.eq('SAI') & sensitivity.cooling_case.eq('central')].groupby('server_type'):
            central = group.loc[group.power_sensitivity.eq('central')].iloc[0]
            sensitivity_rows.append([central.scenario_id, f'{central.mean_it_mw:.3f}',
                f'{group.loc[group.power_sensitivity.isin(["ratio_p10", "ratio_p90"]), "mean_it_mw"].min():.3f}–{group.loc[group.power_sensitivity.isin(["ratio_p10", "ratio_p90"]), "mean_it_mw"].max():.3f}',
                f'{group.loc[group.power_sensitivity.isin(["idle_half", "idle_double"]), "mean_it_mw"].min():.3f}–{group.loc[group.power_sensitivity.isin(["idle_half", "idle_double"]), "mean_it_mw"].max():.3f}',
                f'{group.loc[group.power_sensitivity.eq("legacy_idle"), "mean_it_mw"].iloc[0]:.3f}'])
        sensitivity_text = table(['情境', '中央 IT 年均 MW', '分配功率 P10–P90 端点 MW', '空载×0.5–2 MW', '沿用目标原空载 MW'], sensitivity_rows)
    inf = pd.read_csv(project / 'results/asi-185day-workload-types-20261031/inf-hourly.csv')
    excess = int(inf.combined_used_gpu_hours.gt(inf.inventory_gpu).sum())
    offline = flex['INF']
    text = f'''# 典型负荷曲线 v3：44 个风冷情境的需求与功率重放

输入基线为 `origin/main@{manifest['base_commit']}`，在独立分支合入 #27、#33 与 #32（包含 #30）后接入新数据。当前计算状态：**{manifest['reference_computed_scenarios']} 个参考网格情境、{manifest['location_years']} 个地点×气象年、{manifest['annual_rows']} 行年度汇总、{manifest['hourly_rows']:,} 行小时功率**。全部情境采用[确认命名](../../docs/scenario-naming.md)，范围为 A-HPC、A-CLD、B/C/D-SAI、B/C/D-TRN、B/C/D-INF × 1/10/100/500MW，全部风冷。

## 功率链路及可比范围

每小时需求份额 → CPU 响应 / GPU 分配功率 / 基础 IT 功率 → 风冷模型 → 制冷与配电等其他损耗 → 设施总功率。

`P_IT = P_base + P_CPU + P_GPU`；`P_facility = P_IT + P_cooling + P_other`。基础 IT、CPU 端点和 SPECpower 整机响应迁移沿用 v2，目标 CPU 封装功率仍含假设。每档设备台数按 v2 中央满载参考功率向下取整，实际安装参考功率随表列出；没有按年度峰值重新放大设备数量。GPU 分配份额和目标吞吐等价尚未共同标定。

风冷继续使用 `compute_load/cooling.py` 的 cooling-20261028-v2，沿用机房风机、泵、冷塔、自然冷却、机械补冷、UPS/配电及固定损耗。参考网格为 `g05_r100_c592`，2025 年 8760 小时，固定天气与 v2 相同，气压沿用 101325Pa 假设。八城市使用原工作包冻结的 2023–2025 年温度、露点与气压文件，逐文件 SHA-256 核对；新加坡仍使用柔佛陆地占优格点，代表区域气候。没有重新下载气象或运行能源优化。

HPC 的 Lisa 309 天与 TRN 的 Acme 90 天需求、功率方法保留。INF 保留 v2 的 GPU 功率中位比例与目标空载假设，只更新需求及业务范围。SAI 按下式更新：

`P_GPU = n_GPU × [a × TDP_H100 × r_June + (1−a) × TDP_H100 × P_idle_V100 / 250]`

其中 `a` 为原 237 天分配份额，`r_June` 是六月前半月有效 GPU 时间加权的分配功率/250W。分配功率已包含分配期间空转状态，不再乘利用率或扣除空转占比。

## 10MW 情境的年均、峰值与形状

{ten_table}

日内比为目标 UTC 小时分组均值最大/最小，周内比为目标星期×小时的 168 个分组均值最大/最小。它们描述重放后的目标日历，不能证明来源真实星期或时区。原始小时极值比及排除未解决零资源小时的诊断另见 CSV。[四档规模全部情境](all-44-scenarios.md)、[完整年度数据](annual-summary.csv)、[逐情境 v2 对比](v2-v3-comparison.csv)提供完整精度。规模间保留相同形状，绝对功率按实际整数设备台数变化。

## A-CLD：120 小时整机 CPU 与业务拆分

采用 Alibaba 2018 相对日 2–6 的 **120 小时**连续完整窗口，整机平均 CPU 利用率 **38.056%**，小时极值比 **2.520**。整机有效核心小时为 {flex['CLD']['source_core_hours']:,.3f}，在线服务占 **{100*flex['CLD']['online_share']:.3f}%**、离线实例占 **{100*flex['CLD']['batch_share']:.3f}%**、有符号未对账量占 **{100*flex['CLD']['unreconciled_signed_share']:.3f}%**。未对账量保留原符号及单位边界，没有全部归类为系统开销。

CPU 利用率按有效核心监测时长归一化；缺测不作为空闲。120 小时顺序循环到全年，2025 年恰为 73 个周期；2024 年为 73 个周期加一天。来源没有季节变化、真实星期和时区，五日周期与七日目标周会产生构造性的星期形状。在线分母采用源维护者确认的 request 口径，其他分母的敏感性保留在上游工作包。

## INF：185 天在线与离线推理

采用 ASI **185 天 / 4440 小时**的在线与离线推理归属 GPU 小时之和，二者共用 **{inf.combined_used_gpu_hours.max():,.3f} GPU 小时**的观测峰值分母。2025 年重放 4440+4320 小时，2024 年重放 4440+4344 小时。宿主 CPU 仍为 GPU 份额的 0.25 倍，保留工程代理性质。在线/离线份额与源小时映射随输出提供。

相对日 5 的 6 时（索引 126）全源资源为零但 pod 与库存非零，未确认真实停机，原值保留并逐小时标识；2025 年出现两次。上游全业务有 **409 小时**归属量超过库存，最大比值 1.049695；仅推理组合超过库存的小时为 **{excess}**。v3 分别传递全业务库存异常标识与原始推理需求，归一化没有裁切原值。全年原始小时极值比可能受该零资源小时影响，正观测诊断另外保存。

源时间只有相对日与小时，没有真实日历、时区或可识别季节性。v2 的 ASI 输入为单日混合业务/库存份额，v3 改为长期推理子集/固定峰值份额；差异同时涉及窗口、业务范围、归一化与天气对应相位，不能解释成同一真实设施用电的历史变化。

## SAI：六月 V100 实测功率标定

{calibration}

V100 归一化参考为 250W；目标 H100 PCIe 使用 350W，八卡整机的 H100 使用 700W。将功率/TDP 比及空转/TDP 比迁移到 H100 属于情境假设，未保持等任务吞吐，也没有目标 H100 的同步功率表计。未分配空载代理来自已分配且零利用率的 V100 时段；该状态与未分配 H100 的对应关系仍未知。P10/P90 是有效设备小时功率的描述性端点，空载×0.5/2 是工程敏感性，均不表示年度功率置信区间。低端分配功率可能低于空载代理，保留这种跨状态迁移差异。

{sensitivity_text}

## 八城市年度复算

{city_table}

范围同时跨情境及三个气象年，具体地点×年份×情境见 [annual-summary-all-locations.csv](annual-summary-all-locations.csv)。IT 需求在每个目标年份从相位零开始，气象保留对应年份；2024 年为 8784 小时。地点间差异包括温湿度和气压对风冷的影响。

## 柔性候选量级

CLD 的离线执行量约占整机核心小时 **{100*flex['CLD']['batch_share']:.2f}%**，为候选筛选量；约 29% 未对账部分的业务类型未知。INF 的离线推理占推理 GPU 小时 **{100*offline['offline_share_of_inference']:.2f}%**，其中 LP 占离线推理 **{100*offline['offline_lp_share_of_offline']:.2f}%**；离线且 LP 合计占全部推理 **{100*offline['offline_lp_share_of_inference']:.2f}%**。

SAI 既有作业总体中，明确批处理/Map 占 GPU 小时 **{100*flex['SAI']['batch_map_share']:.2f}%**，运行至少 6 小时占 **{100*flex['SAI']['runtime_ge6h_share']:.2f}%**，两项交集占 **{100*flex['SAI']['batch_map_runtime_ge6h_share']:.2f}%**。该分类汇总与 237 天序列没有逐小时连接，未换算成已证实的年度可调功率。TRN 使用既有预训练且运行至少一小时的资源筛选量。HPC 既有总体中，非 Jupyter 队列且运行至少 6 小时的作业占核心小时 **{100*flex['HPC']['runtime_ge6h_non_jupyter_share']:.2f}%**；该分类与 309 天序列未逐小时连接，完成期限与暂停权限未知。运行时长不作为可推迟时长。

[flexibility-hints.csv](flexibility-hints.csv)覆盖本版全部情境，并为 CLD/INF 列出“同小时移出全部候选、设备仍通电”时的 IT 功率差。这是功率模型反事实，未计算补做、并发容量和服务期限；资源份额不能直接作为设施电量份额。所有来源的已验证可调比例下界保留为零，不给出保证削峰、节电或收益结果。

## 图件与字段

图件均为 10MW 参考情境、参考网格、2025 年 UTC。PNG 为 300dpi，PDF 嵌入中文字体。

1. `figure1-typical-day.*`：全年相同 UTC 小时的 IT、制冷、设施均值。
2. `figure2-typical-week.*`：目标星期×小时均值；不赋予源数据真实星期含义。独立的实际代表周从完整周中按标准化距离选取，起点为 {manifest['representative_week_start_utc']}，逐小时数据及全部评分另存。
3. `figure3-annual-envelope.*`：设施日均及日内最小—最大范围，另绘 IT 日均；循环需求产生的重复结构不解释为真实季节变化。
4. `figure4-power-components.*`：基础 IT、CPU、GPU、制冷、配电及其他损耗的年度电量分项，总和等于设施电量。
5. `figure5-v2-v3-comparison.*`：相同固定天气下年均 IT、年均设施及设施峰值变化；CLD 没有可比 v2 曲线。
6. `figure6-sai-transfer-sensitivity.*`：六月分配功率端点与目标空载假设对 SAI 年均功率的影响。

`hourly/` 保存参考网格全部情境的聚合小时 Parquet，包括 `time_utc`、`it_mw`、`cooling_mw`、`other_mw`、`facility_mw`、CPU/GPU/基础 IT 分项、设备数、来源小时与质量标识。`annual-summary.csv` 为参考网格，`annual-summary-all-locations.csv` 加入八城市三年，`sensitivity-summary.csv` 为 10MW 的功率迁移及风冷参数敏感性。全部城市小时文件留在服务器 `data/processed/{RUN}/`，SHA-256 与路径见 manifest。受限来源的逐作业、逐设备数据及对象快照均未发布；MIT 仅提交聚合统计及情境曲线。

{publication_text}

## 复算与核验

使用 `analysis/requirements-load-curves-v3.txt`。输入包括冻结的 v2 10MW 小时文件、原八城市天气文件，以及本分支已合入的 Alibaba/ASI 聚合结果。先运行 `aggregate_mit_calibration_v3.py --source <月度目录> --output <project>/results/{RUN}`；正式窗口要求 `partial_data=false`。到用户指定的 2026-11-01 14:00 北京时间仍未完成时，使用保留的阶段目录并显式添加 `--allow-partial`。

直接计算命令为 `python analysis/build_typical_load_curves_v3.py --project <project> --baseline <v2小时目录> --weather <八城市天气目录> --base-commit {manifest['base_commit']} --stage full`。

服务器通过 `setsid nohup bash analysis/launch_load_curves_v3.sh <run_root> <python> <base_commit> full` 后台运行，`TMPDIR`、日志和进度均在共享卷。依次执行 `verify_typical_load_curves_v3.py`、`summarize_flexibility_v3.py`、`plot_typical_load_curves_v3.py` 和本报告脚本。核验覆盖小时索引、平闰年、来源映射、质量标识、IT/制冷/设施守恒、独立 GPU 公式、年度积分和 v2 可比结果；HPC/TRN 同时与原八城市年度结果对照。`audit_delivery_v3.py` 检查字体嵌入、情境名称、PDF 文本边界、PNG 分辨率、公开字段和历史文件字节；按用户规范未调用视觉功能。旧结果目录按原字节保留。

## 来源及审阅关系

审阅顺序：先 #27 命名；数据依赖为 **#30 → #32**，#33 可独立审阅；随后审阅本 v3。全部 PR 保持待审，合并需要用户明确批准。

- Alibaba 与 ASI 的输入、许可、归一化及质量说明分别见[云通用计算工作包](../alibaba-cpu-cld-20261031/README.md)与[185 天推理工作包](../asi-185day-workload-types-20261031/README.md)。
- MIT：Samsi 等，*The MIT Supercloud Dataset*，DOI 10.1109/HPEC49654.2021.9622850；[作者论文](https://arxiv.org/abs/2108.02037)、[官方数据与 CC BY-NC-ND 4.0 许可](https://registry.opendata.aws/dcc/)，访问日期 2026-11-01。本仓库发布聚合分析；逐设备、逐作业和原始数据保留私有研究目录。
- [V100 官方规格](https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf)列出 PCIe 250W/SXM2 300W；本次 250W 是归一化参考，源监控未声明实际功率上限。
- [H100 厂商规格](https://lenovopress.lenovo.com/lp1732-thinksystem-nvidia-h100-pcie-gen5-gpu)列出 PCIe 350W、SXM 700W；[DGX H100 官方指南](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html)说明八张 GPU 与双路宿主配置。
'''
    (output / 'README.md').write_text(text, encoding='utf-8')
    print(output / 'README.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    report(parser.parse_args().project.resolve())
