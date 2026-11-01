import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from acquire_alibaba_cpu import now, sha256


def build(project):
    directory=project / "results/alibaba-cpu-cld-20261031"
    metadata=project / "data/metadata/alibaba-cpu-cld-20261031"
    source=json.loads((directory / "source-audit.json").read_text(encoding="utf-8"))
    component=json.loads((directory / "component-audit.json").read_text(encoding="utf-8"))
    validation=json.loads((directory / "validation.json").read_text(encoding="utf-8"))
    small=json.loads((metadata / "download-audit.json").read_text(encoding="utf-8"))
    cache=json.loads((metadata / "large-cache-audit.json").read_text(encoding="utf-8"))
    space=json.loads((metadata / "runtime-space-audit.json").read_text(encoding="utf-8"))
    registry=json.loads((metadata / "public-cpu-cloud-datasets.json").read_text(encoding="utf-8"))
    observed=pd.read_csv(directory / "complete-component-window.csv")
    shapes=pd.read_csv(directory / "shape-comparison.csv").set_index("scenario")
    diurnal=pd.read_csv(directory / "A-CLD-intraday.csv")
    component_diurnal=pd.read_csv(directory / "component-intraday.csv")
    daily=pd.read_csv(directory / "daily-coverage.csv")
    gap=pd.read_csv(directory / "gap-sensitivity.csv")
    units=pd.read_csv(directory / "online-unit-sensitivity.csv")
    totals=component["common_window_core_hours"]
    fractions=component["shares_of_machine"]
    duration=component["batch_duration_quantiles_seconds"]
    rows=[]
    def p(text=""):
        rows.append(text)
    def table(header,values):
        if rows and rows[-1]:
            p()
        p("| "+" | ".join(header)+" |")
        p("| "+" | ".join(["---"]*len(header))+" |")
        for row in values:
            p("| "+" | ".join(str(value).replace("|","/").replace("\n"," ") for value in row)+" |")
        p()
    p("# 纯 CPU 型·云通用计算：Alibaba 2018 需求与较新公开数据核查")
    p()
    p("情境为 **A-CLD**，规模为参考 IT 功率 **1MW、10MW、100MW、500MW**。命名采用 [PR #27](https://github.com/TuzzyT-Laiii/33/pull/27) 的“服务器类型×用途×规模”规则，按当前用户确认的分析范围执行。本工作从 `origin/main` 的 `e7b03cb` 独立建立；已有原稿、历史情境结果和待审命名分支保持各自版本。")
    p()
    p("## 一、Alibaba 2018 的获取、口径与需求结果")
    p()
    p("### 输入与许可")
    p()
    p("[官方说明](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md)、[字段定义](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/schema.txt)及[下载脚本](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/fetchData.sh)已阅读全文。数据为在线 Sigma 服务和离线 Fuxi 批处理的生产混部集群。官方项目允许研究、学习用途，未给出明确的标准数据许可证；不能推定商业使用及原始数据再分发权限。原始数据只在研究服务器保存，仓库提交脚本、统计汇总和来源审计。")
    p()
    table(["压缩包","字节数","SHA-256（已与官方一致）"],[[f"{name}.tar.gz",f"{record['bytes']:,}",f"`{record['sha256']}`"] for name,record in {**small["files"],**cache["files"]}.items()])
    p("四个小表位于 `/data3/dataset/ljt/global-grid-flexibility-20261023/data/raw/alibaba-cpu-2018/`，合计 "+f"{sum(x['bytes'] for x in small['files'].values()):,}"+" 字节。下载前确认可用空间覆盖下载量并额外保留 5 GiB。两张大表的既有下载已复用；系统盘缓存以保留 35 GiB 为约束。全部数据均未解压到磁盘。")
    table(["大表","已校验压缩包位置"],[[name,f"`{record['path']}`"] for name,record in cache['files'].items()])
    p(f"[末次空间核查](../../data/metadata/alibaba-cpu-cld-20261031/runtime-space-audit.json)时间为 {space['checked_at_utc']}，系统盘可用 {space['root_free_bytes']/1024**3:.3f} GiB，共享卷可用 {space['shared_free_bytes']/1024**3:.3f} GiB。"+
      ("该次核查满足系统盘 35 GiB 下限。" if space['root_floor_met'] else "该次核查低于用户要求的系统盘 35 GiB 下限。聚合期间本任务只读取系统盘的既有压缩包，汇总与恢复点写在共享卷；迁移至扩容共享卷或删除已完成缓存的追加授权仍待回复。"))
    p(f"末段空间观测保留 {len(space['late_run_observations'])} 次带时间戳的原始可用字节数，最低系统盘可用空间为 {space['minimum_late_run_root_free_bytes']/1024**3:.3f} GiB；这些观测记录的范围见同一审计文件，不代表对整个运行期的连续监测。")
    p()
    p()
    p("[四表下载审计](../../data/metadata/alibaba-cpu-cld-20261031/download-audit.json)及[系统盘缓存审计](../../data/metadata/alibaba-cpu-cld-20261031/large-cache-audit.json)记录 URL、ETag、字节数、SHA-256、下载日志与空间检查。大表支持 HTTP 或已校验缓存的流式解压；本次容器最终全量结果从系统盘压缩包读取。后台进程使用 `setsid nohup`，定期保存原子检查点和状态；恢复时从压缩流起点解码并跳过已提交行，避免重复累计。检查点只含统计数组、行数、计数和去重恢复位图，不保存逐条 CPU 值或完整 CSV。")
    p()
    p("### 整机逐小时 CPU 与完整日")
    p()
    p(f"`machine_meta` 共 {source['machine_meta_rows']:,} 行、{source['machines']:,} 台；全部 `cpu_num=96`，库存共 {source['inventory_cores']:,.0f} 个源 CPU 单位。官方未区分该数量的物理核心/逻辑线程属性，本文“核小时”均按该源单位计算，不表示现代服务器的等价吞吐。`machine_usage` 共 {source['counts']['raw_rows']:,} 行，重复机器×时间记录 {source['counts']['duplicate_rows']:,} 行。有效 CPU 样本 {source['valid_cpu_unique_samples']:,} 条；CPU 无效/缺失 {sum(source['invalid_cpu_raw_rows'].values()):,} 条，内存非法/缺失 {sum(source['invalid_memory_raw_rows'].values()):,} 条。磁盘字段的 -1 和 101 分别为 {source['invalid_disk_raw_rows'].get('-1',0):,}、{source['invalid_disk_raw_rows'].get('101',0):,} 条，不用于 CPU 计算。")
    p()
    p("机器样本位于 10 秒时间格，文件顺序不代表完整时间顺序。先按机器和时间还原，再采用前值保持至下一条记录、最长 300 秒；超过上限的时段保留为缺测。CPU 仅接受 0—100 的有限值，非法样本终止前一有效区间。每小时占用核小时 = Σ(cpu_num × CPU百分数/100 × 有效区间秒数/3600)。`observed_core_hours` 为有效核时支持，`coverage_fraction` 为其占库存核小时比例；`resource_fraction` 为占用核小时/有效核小时。缺测不填为零。另列按库存作分母的份额，供观察缺测敏感性。")
    p()
    p("小时表包含机器数量、样本均值、机器小时均值及 P05/P25/P50/P75/P95、时间与核数加权均值、占用核小时、覆盖率。完整日要求 24 个小时均落在实际最早/最晚观测边界内，且每小时核时覆盖率至少 95%。最后一个源样本位于相对秒 691190，末日按此保守边界规则未纳入完整窗口。")
    p()
    table(["相对日","最小小时覆盖率","平均覆盖率","完整日","占用核小时"],[[int(r.relative_day),f"{r.minimum_coverage:.4%}",f"{r.mean_coverage:.4%}","是" if r.complete_day else "否",f"{r.occupied_core_hours:,.3f}"] for r in daily.itertuples()])
    p(f"采用相对日 {source['complete_days']}，共 {len(observed)} 小时。平均 CPU 利用率 **{shapes.loc['A-CLD','mean']*100:.2f}%**，小时最低/最高 **{shapes.loc['A-CLD','minimum']*100:.2f}% / {shapes.loc['A-CLD','maximum']*100:.2f}%**，峰谷比 **{shapes.loc['A-CLD','peak_to_trough']:.3f}**；累计有效占用 **{totals['occupied_core_hours']:,.3f} 核小时**。整个八日源窗口的有效监测样本 P05/P25/P50/P75/P95 为 "+"/".join(str(source['sample_cpu_quantiles_pct'][key]) for key in ('p05','p25','p50','p75','p95'))+"%，其统计范围和统计单位与完整窗口的机器小时均值分位数分别记录。")
    p()
    table(["保持上限（秒）","八日平均覆盖率","观测支持上的平均 CPU","占用核小时"],[[int(r.maximum_hold_seconds),f"{r.coverage_fraction:.4%}",f"{r.mean_util_observed_pct:.4f}%",f"{r.occupied_core_hours:,.3f}"] for r in gap.itertuples()])
    p("[全部小时表](alibaba-hourly-cpu-demand.csv)、[完整窗口](complete-window.csv)、[逐机器小时汇总](machine-hourly.parquet)、[样本直方图](cpu-sample-histogram.csv)、[整机来源审计](source-audit.json)支持复算。")
    p()
    p("### 在线、离线与整机对账")
    p()
    online_state=component['large_table_states']['container_usage']
    batch_state=component['large_table_states']['batch_instance']
    p(f"容器表处理 {online_state['processed_rows']:,} 行，时间戳范围 {online_state['timestamp_min']:.0f}—{online_state['timestamp_max']:.0f} 秒。保持原始相对时间，不进行平移；整机八日窗口以外的 {online_state['stats']['outside_machine_window_rows']:,} 行单独计数。每条有效容器监测记录按原始 10 秒格累计；通过容器×时间连接当时的 request/limit 和主机。主解释为 CPU百分数/100 × cpu_request/100 核，同时保留 limit 解释和单核百分数解释。")
    p()
    p("2018 schema 仅列百分比；[2018 官方维护者回复](https://github.com/alibaba/clusterdata/issues/62#issuecomment-476964315)明确分母为容器请求的 CPU 数量，与[2017 官方说明](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2017/trace_201708.md)一致。正式在线份额采用 request 口径，limit 和单核百分数另列敏感性诊断；单位依据独立于对账差额。原始监测时段之间不插值；同容器、同原始时间戳保留文件首条有效记录，不重复计算支持时长。[元数据维护者说明](https://github.com/alibaba/clusterdata/issues/47#issuecomment-448898828)确认少量容器的 limit 随运行改变，本次按事件时间连接该字段。")
    p(f"有效连接记录中去除重复 {online_state['stats']['duplicate_valid_rows']:,} 条，保留 {online_state['unique_valid_slots']:,} 个唯一容器时间格；逐小时计数总和、恢复位图和保留记录数已核对一致，每容器每小时至多 360 个时间格。此处的去重规则不假定重复行的 CPU 数值相同。")
    p(f"容器 CPU 清洗保留有限且位于 [0,100] 的百分数，排除非有限、负数和超过 100 的记录共 {online_state['stats']['invalid_cpu_rows']:,} 条；该范围沿用 2017 官方说明的满额百分比约定，2018 schema 未另列容器 CPU 数值上限。若超过 100 的观测包含真实突发用量，该规则会低估对应部分，应结合计量定义核实。另有未知容器 {online_state['stats']['unknown_container_rows']:,} 行，共同整机窗口内元数据连接失败 {online_state['stats']['unmatched_meta_rows_in_window']:,} 行，均不填补 CPU 用量。")
    p()
    p("[官方维护者关于时间窗口的说明](https://github.com/alibaba/clusterdata/issues/52#issuecomment-455026686)确认：机器表和容器表来自不同的八日窗口，实际包含九个相对日，两者共同覆盖七日；时间戳本身正确，不能将容器时间整体减去一天。本次保留原始相对时间，仅在共同区间中筛选满足覆盖标准的完整日；[维护者回复审计](../../data/metadata/alibaba-cpu-cld-20261031/timestamp-alignment.json)保留答复链接、作者、摘要及校验值。")
    p()
    p(f"批处理实例表处理 {batch_state['processed_rows']:,} 行，正时长、有效 CPU 和可连接主机的执行记录 {batch_state['stats']['valid_positive_duration_rows']:,} 条。平均 CPU 为 `cpu_avg/100` 核，按 start—end 与各小时的精确重叠积分；同样保留 `cpu_max/100` 恒定于整个区间的诊断积分。平均字段超过最大字段的记录有 {batch_state['stats']['average_above_maximum_rows']:,} 条，因此最大字段积分不具有无条件的严格上界含义。小时内活跃形状、右删失与无效区间、重试/备份执行均不能从平均字段完整恢复。")
    p()
    table(["完整窗口部分","核小时","占整机观测量"],[
        ["在线服务（request 主解释）",f"{totals['online_request_based_core_hours']:,.3f}",f"{fractions['online']:.3%}"],
        ["离线实例（平均 CPU 积分）",f"{totals['batch_mean_core_hours']:,.3f}",f"{fractions['batch_candidate']:.3%}"],
        ["有符号未对账差额",f"{totals['residual_core_hours']:,.3f}",f"{fractions['unreconciled']:.3%}"],
        ["整机观测量",f"{totals['occupied_core_hours']:,.3f}","100%"]])
    table(["容器百分比分母解释","在线核小时","未对账差额核小时","集群小时负差额数量"],[[r.basis,f"{r.online_core_hours:,.3f}",f"{r.residual_core_hours:,.3f}",int(r.negative_residual_cluster_hours)] for r in units.itertuples()])
    p(f"完整窗口内有 {component['negative_residual_cluster_hours_complete_window']} 个集群小时、{component['negative_residual_machine_hours_complete_window']:,} 个机器小时的差额为负。所有差额保留原符号，不强制缩放组件或截断负数。差额可能包含系统开销、采样缺口、单位/计量窗口差异和实例均值分摊误差，不能全部认定为系统开销或在线服务。")
    p()
    p(f"完整窗口的容器样本支持为元数据 started 状态容器小时的 {component['online_sample_support_ratio_complete_window']:.3%}，逐小时比值为 {component['online_sample_support_ratio_hour_min']:.3%}—{component['online_sample_support_ratio_hour_max']:.3%}。该比值用原始有效样本数×10 秒与事件生命期对照，受元数据状态及计量粒度影响，不等同于逐时隙完整率；不据此填补在线 CPU 用量。")
    p()
    p("[逐小时对账和份额](component-reconciliation.csv)、[逐机器对账](machine-component-reconciliation.parquet)、[完整业务窗口](complete-component-window.csv)、[业务拆分审计](component-audit.json)为组件入口。另有两小表的[请求/计划包络](online-batch-request-envelope.csv)：在线请求依据容器事件保持，离线为 instance_num×plan_cpu×任务起止包络；它们表示配置需求，不能替代实际 CPU 份额。")
    p()
    p("### 日内、相对日形状与 Lisa 对照")
    p()
    low,high=diurnal.loc[diurnal['mean'].idxmin()],diurnal.loc[diurnal['mean'].idxmax()]
    p(f"完整窗口按相对时刻汇总后，平均需求最低为 {int(low.hour_of_day):02d} 时（{low['mean']*100:.2f}%），最高为 {int(high.hour_of_day):02d} 时（{high['mean']*100:.2f}%），日内平均曲线峰谷比 {high['mean']/low['mean']:.3f}。[在线/离线日内表](component-intraday.csv)及[相对日累计表](component-relative-days.csv)保留分项形状。源记录没有可核查的真实星期与时区，完整窗口只有五日，无法识别工作日/周末效应或稳定周周期；相对日形状不解释为真实周内规律。")
    p()
    shape_rows=[]
    for label,column in (("在线服务（request）","online_request_based_core_hours"),("离线实例（平均 CPU）","batch_mean_core_hours")):
        trough=component_diurnal.loc[component_diurnal[column].idxmin()]
        peak=component_diurnal.loc[component_diurnal[column].idxmax()]
        shape_rows.append([label,f"{int(trough.hour_of_day):02d} 时 / {trough[column]:,.3f}",
                           f"{int(peak.hour_of_day):02d} 时 / {peak[column]:,.3f}",f"{peak[column]/trough[column]:.3f}"])
    table(["组成","日内最低时刻 / 平均核小时","日内最高时刻 / 平均核小时","日内平均峰谷比"],shape_rows)
    relative_days=observed.groupby("relative_day")[["occupied_core_hours","online_request_based_core_hours","batch_mean_core_hours","residual_core_hours"]].sum()
    table(["相对日","整机核小时","在线占整机","离线占整机","有符号差额占整机"],
          [[int(day),f"{r.occupied_core_hours:,.3f}",f"{r.online_request_based_core_hours/r.occupied_core_hours:.3%}",
            f"{r.batch_mean_core_hours/r.occupied_core_hours:.3%}",f"{r.residual_core_hours/r.occupied_core_hours:.3%}"] for day,r in relative_days.iterrows()])
    table(["情境","完整小时数","需求含义与分母","相对波动 CV","P95/P05","小时峰谷比"],[
        ["A-CLD 纯 CPU 型·云通用计算",int(shapes.loc['A-CLD','hours']),"实测利用率；有效源核时",f"{shapes.loc['A-CLD','cv']:.3f}",f"{shapes.loc['A-CLD','p95']/shapes.loc['A-CLD','p05']:.3f}",f"{shapes.loc['A-CLD','peak_to_trough']:.3f}"],
        ["A-HPC 纯 CPU 型·科研计算",int(shapes.loc['A-HPC','hours']),"Lisa 调度分配核心小时；观测小时峰值",f"{shapes.loc['A-HPC','cv']:.3f}",f"{shapes.loc['A-HPC','p95']/shapes.loc['A-HPC','p05']:.3f}","最低为零，未定义"]])
    p("本样本的云负荷相对波动小于 Lisa 长窗口的调度分配曲线。二者窗口长短、分母和“活跃利用率/分配占用”含义不同，不能据此推断云机房的利用效率高低或全部业务的柔性大小。[Lisa 完整小时汇总](lisa-cpu-complete-hourly.csv)及两组日内表保留比较依据。Lisa 的现有官方审计记录许可为 CC BY 4.0，见[来源核查](../../data/metadata/research-gpu-demand-20261028/lisa-gpu-audit.json)。")
    p()
    p("### 年度构造、功率接口与可平移边界")
    p()
    p("复用 `annual_task_profiles.replay_hourly`：选定连续完整日期按原顺序循环，不平滑、不调整均值、不添加季节需求。2025 年 8760 小时 = 120 小时×73；2024 年 8784 小时 = 120 小时×73＋24 小时。源相对午夜映射到目标 UTC 午夜仅为相位设定。**没有季节变化**，不能代表源机房真实全年、节假日或需求增长。[平年映射](A-CLD-2025-replay-map.csv)与[闰年映射](A-CLD-2024-replay-map.csv)列出每个目标小时的源小时和循环位置。")
    p()
    p("`load_alibaba_source(project)` 返回与现有需求源一致的 `profile/clock/metadata`，可交给年度重放及 `ComputeLoadModel.calculate`。已用仓库的实际 2025 年气象小时表验证四档规模的调用和有限输出；该检查使用既有功率接口，目标设备的性能与功率映射仍需按服务器参数单独标定。未用源核心数量替代现代硬件性能换算。")
    p()
    p("在线服务刚性、离线批处理可推迟采用用户指定的情境假设。离线平均 CPU 积分给出候选工作量，原始任务 DAG 可约束执行先后；日志缺少可核查的独立提交时间、业务完成期限和检查点/抢占权限，因而不能将全部离线份额直接作为可移负荷。可平移研究应限于获得延迟窗口和权限的批处理，保持 DAG、资源总量及真实完成约束；在线部分和未对账部分在保守情境中保持原时刻。此处不推定任何实例已有可延后若干小时的权限。")
    p()
    p("离线正时长执行记录的时长分位数（秒）："+"、".join(f"{key.upper()}={value}" for key,value in duration.items())+"。[精确秒级频数](components/batch-instance-duration-histogram.csv)和[分箱表](batch-duration-bins.csv)可复算。分布按执行记录数量统计，包含重试/备份等尝试；实例数量份额与 CPU 工作量份额不同，不能据短作业数量直接推算可移能量。")
    p()
    p("## 二、较新年份公开 CPU 云数据")
    p()
    p("核查日期为 2026-10-31，仅采用官方发布页、官方仓库和作者论文/存档；按版本年与真实采集年分别登记。检索式和全部字段见[机器可读登记表](../../data/metadata/alibaba-cpu-cld-20261031/public-cpu-cloud-datasets.json)，正文/元数据获取的 URL、读取范围、时间与 SHA 见[官方来源核验表](../../data/metadata/alibaba-cpu-cld-20261031/official-source-checks.json)。部分仓库文件的精确大小由[官方目录树元数据](../../data/metadata/alibaba-cpu-cld-20261031/newer-alibaba-repository-files.csv)复算。本部分没有下载任何较新来源的大数据文件。")
    p()
    for dataset in registry['datasets']:
        p(f"### {dataset['id']}")
        p()
        p(f"[官方来源]({dataset['url']})"+(f" / [作者论文]({dataset['paper_url']})" if 'paper_url' in dataset else ""))
        p()
        table(["字段","核查结果"],[
            ["发布/版本年；采集年",f"{dataset['release_year']}；{dataset['collection_year'] if dataset['collection_year'] is not None else '未明确'}"],
            ["时长；规模",dataset['duration']+"；"+dataset['scale']],
            ["时间分辨率",dataset['resolution']],["CPU 指标",dataset['cpu']],["功率",dataset['power']],
            ["业务",dataset['business']],["许可",dataset['license']],["大小",dataset['size']],
            ["获取方式",dataset['access']],["A-CLD 适用性",dataset['cld_fit']]])
    p("### 数据选择")
    p()
    p("**本研究主数据采用 Alibaba 2018**：六表提供同一混部环境的整机实测 CPU、在线资源和离线实例执行，可与当前 A-CLD 的用途定义衔接。完整窗口短、源 CPU 数量的逻辑/物理属性未明确、业务组件存在未对账量，这些限制随需求输入保留。")
    p()
    p("**较新年份优先对照 SAP Cloud Infrastructure 的 2024 年观测、2025 年发布数据**：提供 VM 与物理宿主利用率、30 日窗口和明确 CC BY 4.0，最新 1.0.1 约 1.04 GB。其企业内存密集业务组成与 Alibaba 的互联网混部不同；本轮仅核对元数据和论文，原始缺测、核数定义和连续窗口须在获取后核验，再决定是否升级为较新主样本。")
    p()
    p("**国内年份对照优先补充 ByteDance IaaS 2023 与 PaaS 2024**：前者有 93 个 VM 的 10 分钟 CPU 序列，后者有 426 个服务实例、以毫核计量的 CPU 序列；2025 年公开，数据许可为 CC BY 4.0。它们适合检验较新云需求形状，宿主容量、预处理和业务分类信息仍需核验；本轮没有下载这些 CSV。")
    p()
    p("Google Borg 2019 适合作为进一步的跨厂商混部对照，PowerData2019 还可对照资源与电力形状；应通过 BigQuery 聚合后仅导出小时结果，注意归一化 CPU、功率容量比和包含冷却的测量边界。Alibaba 2022 的 Node 表支持较新国内云负荷形状对照，但 min-max 归一化无法直接恢复绝对利用率及核小时。Azure 2019 是明确的通用云 VM 需求对照，须保留 vCPU 分桶区间和宿主映射缺口。国内还可选 NEP 2020 的生产边缘云 VM CPU；华为 2025 版有 CPU 核心用量和分钟摘要，适合较新 FaaS 对照，须避免并发请求重复归属整个 pod 的平均用量。")
    p()
    p("### 复现与验证")
    p()
    p("在研究服务器 Python 环境安装 `analysis/requirements-alibaba-cpu.txt`。下列 `PROJECT` 指服务器项目目录，`RESULTS` 指本目录对应的服务器位置。")
    p()
    p("""```bash
PROJECT=/data3/dataset/ljt/global-grid-flexibility-20261023
RESULTS=$PROJECT/results/alibaba-cpu-cld-20261031
LARGE=/home/ljt/alibaba-cpu-2018-large
cd "$PROJECT"
python analysis/acquire_alibaba_cpu.py --directory $PROJECT/data/raw/alibaba-cpu-2018
python analysis/audit_alibaba_cpu.py --raw $PROJECT/data/raw/alibaba-cpu-2018 --output $RESULTS --lisa-hourly $PROJECT/data/processed/task-trace-audit-20261026/lisa/lisa-hourly-core-demand.parquet
setsid nohup python analysis/stream_alibaba_online.py --raw $PROJECT/data/raw/alibaba-cpu-2018 --output $RESULTS/components --archive $LARGE/container_usage.tar.gz > $PROJECT/.codex-temp/alibaba-online.log 2>&1 < /dev/null &
setsid nohup python analysis/stream_alibaba_components.py batch_instance --raw $PROJECT/data/raw/alibaba-cpu-2018 --output $RESULTS/components --archive $LARGE/batch_instance.tar.gz > $PROJECT/.codex-temp/alibaba-batch.log 2>&1 < /dev/null &
# 等待两份状态均为 complete 后执行对账和报告核验。
python analysis/reconcile_alibaba_cpu.py --directory $RESULTS
python analysis/verify_alibaba_cpu.py --project $PROJECT --require-components
python analysis/build_alibaba_report.py --project $PROJECT
```""")
    p()
    p("恢复时使用相同参数即可读取检查点；已完成且 SHA 通过的输入直接复用。追加 `--weather <真实全年气象CSV>` 可核验四档功率接口；本次实际输入的 SHA 及检查范围见 [validation.json](validation.json)。[容器恢复独立验证](online-resume-validation.json)使用真实原文前 400 万行，[实例积分与恢复独立验证](batch-resume-validation.json)使用真实前 10 万行，均与独立计算方法比较通过。大表原始压缩包、恢复检查点及逐记录数据不提交 Git。[artifact-manifest.json](artifact-manifest.json)记录交付文件校验值。")
    p()
    p("遗留：业务拆分的计量一致性、源 CPU 的逻辑/物理含义、真实时区和星期、跨硬件吞吐与功率映射、可延迟窗口和抢占权限；较新数据的原始文件尚未下载核验。")
    (directory / "README.md").write_text("\n".join(rows)+"\n",encoding="utf-8")
    paths=[p for p in directory.rglob('*') if p.is_file() and p.suffix in ('.csv','.parquet','.json','.md') and p.name not in ('artifact-manifest.json','container_usage-status.json') and not p.name.endswith('-failure.json')]
    manifest={"created_at_utc":now(),"files":{p.relative_to(project).as_posix():sha256(p) for p in sorted(paths)},
              "scripts_sha256":{p.name:sha256(p) for p in (project / 'analysis').glob('*alibaba*.py')},
              "metadata_sha256":{p.name:sha256(p) for p in metadata.iterdir() if p.suffix in ('.json','.csv')}}
    (directory / "artifact-manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"report":str(directory / 'README.md'),"datasets":len(registry['datasets']),"files":len(paths)}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="由审计及统计表生成 A-CLD 中文报告")
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    build(parser.parse_args().project)
