# 纯 CPU 型·云通用计算：Alibaba 2018 需求与较新公开数据核查

情境为 **A-CLD**，规模为参考 IT 功率 **1MW、10MW、100MW、500MW**。命名采用 [PR #27](https://github.com/TuzzyT-Laiii/33/pull/27) 的“服务器类型×用途×规模”规则，按当前用户确认的分析范围执行。本工作从 `origin/main` 的 `e7b03cb` 独立建立；已有原稿、历史情境结果和待审命名分支保持各自版本。

## 一、Alibaba 2018 的获取、口径与需求结果

### 输入与许可

[官方说明](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md)、[字段定义](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/schema.txt)及[下载脚本](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/fetchData.sh)已阅读全文。数据为在线 Sigma 服务和离线 Fuxi 批处理的生产混部集群。官方项目允许研究、学习用途，未给出明确的标准数据许可证；不能推定商业使用及原始数据再分发权限。原始数据只在研究服务器保存，仓库提交脚本、统计汇总和来源审计。

| 压缩包 | 字节数 | SHA-256（已与官方一致） |
| --- | --- | --- |
| machine_meta.tar.gz | 92,432 | `b5b1b786b22cd413a3674b8f2ebfb2f02fac991c95df537f363ef2797c8f6d55` |
| machine_usage.tar.gz | 1,774,523,160 | `3e6ee87fd204bb85b9e234c5c75a5096580fdabc8f085b224033080090753a7a` |
| batch_task.tar.gz | 130,310,549 | `7c4b32361bd1ec2083647a8f52a6854a03bc125ca5c202652316c499fbf978c6` |
| container_meta.tar.gz | 2,491,220 | `febd75e693d1f208a8941395e7faa7e466e50d21c256eff12a815b7e2fa2053f` |
| container_usage.tar.gz | 29,208,338,602 | `b4bd3b1b82f5407c1b0dd47fe736ff5152c4b016a9c30cb26f988e2d1c5e5031` |
| batch_instance.tar.gz | 21,204,654,955 | `e73e5a9326669aa079ba20048ddd759383cabe1fe3e58620aa75bd034e2450c6` |

四个小表位于 `/data3/dataset/ljt/global-grid-flexibility-20261023/data/raw/alibaba-cpu-2018/`，合计 1,907,417,361 字节。下载前确认可用空间覆盖下载量并额外保留 5 GiB。两张大表的既有下载已复用；系统盘缓存以保留 35 GiB 为约束。全部数据均未解压到磁盘。

| 大表 | 已校验压缩包位置 |
| --- | --- |
| container_usage | `/home/ljt/alibaba-cpu-2018-large/container_usage.tar.gz` |
| batch_instance | `/home/ljt/alibaba-cpu-2018-large/batch_instance.tar.gz` |

[末次空间核查](../../data/metadata/alibaba-cpu-cld-20261031/runtime-space-audit.json)时间为 2026-10-31T12:47:24.999184+00:00，系统盘可用 26.144 GiB，共享卷可用 6036.451 GiB。该次核查低于用户要求的系统盘 35 GiB 下限。聚合期间本任务只读取系统盘的既有压缩包，汇总与恢复点写在共享卷；迁移至扩容共享卷或删除已完成缓存的追加授权仍待回复。
末段空间观测保留 27 次带时间戳的原始可用字节数，最低系统盘可用空间为 26.144 GiB；这些观测记录的范围见同一审计文件，不代表对整个运行期的连续监测。


[四表下载审计](../../data/metadata/alibaba-cpu-cld-20261031/download-audit.json)及[系统盘缓存审计](../../data/metadata/alibaba-cpu-cld-20261031/large-cache-audit.json)记录 URL、ETag、字节数、SHA-256、下载日志与空间检查。大表支持 HTTP 或已校验缓存的流式解压；本次容器最终全量结果从系统盘压缩包读取。后台进程使用 `setsid nohup`，定期保存原子检查点和状态；恢复时从压缩流起点解码并跳过已提交行，避免重复累计。检查点只含统计数组、行数、计数和去重恢复位图，不保存逐条 CPU 值或完整 CSV。

### 整机逐小时 CPU 与完整日

`machine_meta` 共 17,592 行、4,034 台；全部 `cpu_num=96`，库存共 387,264 个源 CPU 单位。官方未区分该数量的物理核心/逻辑线程属性，本文“核小时”均按该源单位计算，不表示现代服务器的等价吞吐。`machine_usage` 共 246,934,820 行，重复机器×时间记录 0 行。有效 CPU 样本 246,934,814 条；CPU 无效/缺失 6 条，内存非法/缺失 0 条。磁盘字段的 -1 和 101 分别为 37,238、260,330 条，不用于 CPU 计算。

机器样本位于 10 秒时间格，文件顺序不代表完整时间顺序。先按机器和时间还原，再采用前值保持至下一条记录、最长 300 秒；超过上限的时段保留为缺测。CPU 仅接受 0—100 的有限值，非法样本终止前一有效区间。每小时占用核小时 = Σ(cpu_num × CPU百分数/100 × 有效区间秒数/3600)。`observed_core_hours` 为有效核时支持，`coverage_fraction` 为其占库存核小时比例；`resource_fraction` 为占用核小时/有效核小时。缺测不填为零。另列按库存作分母的份额，供观察缺测敏感性。

小时表包含机器数量、样本均值、机器小时均值及 P05/P25/P50/P75/P95、时间与核数加权均值、占用核小时、覆盖率。完整日要求 24 个小时均落在实际最早/最晚观测边界内，且每小时核时覆盖率至少 95%。最后一个源样本位于相对秒 691190，末日按此保守边界规则未纳入完整窗口。

| 相对日 | 最小小时覆盖率 | 平均覆盖率 | 完整日 | 占用核小时 |
| --- | --- | --- | --- | --- |
| 0 | 56.0953% | 66.0339% | 否 | 2,037,277.581 |
| 1 | 0.0000% | 59.1610% | 否 | 2,071,804.915 |
| 2 | 99.4584% | 99.5222% | 是 | 3,596,120.027 |
| 3 | 99.5165% | 99.5607% | 是 | 3,499,772.792 |
| 4 | 99.4048% | 99.5315% | 是 | 3,578,319.771 |
| 5 | 99.4719% | 99.5012% | 是 | 3,211,496.501 |
| 6 | 98.1856% | 99.0667% | 是 | 3,697,408.683 |
| 7 | 99.3830% | 99.4771% | 否 | 3,905,566.968 |

采用相对日 [2, 3, 4, 5, 6]，共 120 小时。平均 CPU 利用率 **38.06%**，小时最低/最高 **23.76% / 59.88%**，峰谷比 **2.520**；累计有效占用 **17,583,117.773 核小时**。整个八日源窗口的有效监测样本 P05/P25/P50/P75/P95 为 10/28/38/48/64%，其统计范围和统计单位与完整窗口的机器小时均值分位数分别记录。

| 保持上限（秒） | 八日平均覆盖率 | 观测支持上的平均 CPU | 占用核小时 |
| --- | --- | --- | --- |
| 60 | 89.4067% | 38.1729% | 25,376,599.608 |
| 300 | 90.2318% | 38.1535% | 25,597,767.237 |
| 900 | 91.3164% | 38.1238% | 25,885,316.491 |

[全部小时表](alibaba-hourly-cpu-demand.csv)、[完整窗口](complete-window.csv)、[逐机器小时汇总](machine-hourly.parquet)、[样本直方图](cpu-sample-histogram.csv)、[整机来源审计](source-audit.json)支持复算。

### 在线、离线与整机对账

容器表处理 4,015,763,787 行，时间戳范围 0—777590 秒。保持原始相对时间，不进行平移；整机八日窗口以外的 549,601,032 行单独计数。每条有效容器监测记录按原始 10 秒格累计；通过容器×时间连接当时的 request/limit 和主机。主解释为 CPU百分数/100 × cpu_request/100 核，同时保留 limit 解释和单核百分数解释。

2018 schema 仅列百分比；[2018 官方维护者回复](https://github.com/alibaba/clusterdata/issues/62#issuecomment-476964315)明确分母为容器请求的 CPU 数量，与[2017 官方说明](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2017/trace_201708.md)一致。正式在线份额采用 request 口径，limit 和单核百分数另列敏感性诊断；单位依据独立于对账差额。原始监测时段之间不插值；同容器、同原始时间戳保留文件首条有效记录，不重复计算支持时长。[元数据维护者说明](https://github.com/alibaba/clusterdata/issues/47#issuecomment-448898828)确认少量容器的 limit 随运行改变，本次按事件时间连接该字段。
有效连接记录中去除重复 651 条，保留 3,466,160,781 个唯一容器时间格；逐小时计数总和、恢复位图和保留记录数已核对一致，每容器每小时至多 360 个时间格。此处的去重规则不假定重复行的 CPU 数值相同。
容器 CPU 清洗保留有限且位于 [0,100] 的百分数，排除非有限、负数和超过 100 的记录共 1,238 条；该范围沿用 2017 官方说明的满额百分比约定，2018 schema 未另列容器 CPU 数值上限。若超过 100 的观测包含真实突发用量，该规则会低估对应部分，应结合计量定义核实。另有未知容器 213,791 行，共同整机窗口内元数据连接失败 176 行，均不填补 CPU 用量。

[官方维护者关于时间窗口的说明](https://github.com/alibaba/clusterdata/issues/52#issuecomment-455026686)确认：机器表和容器表来自不同的八日窗口，实际包含九个相对日，两者共同覆盖七日；时间戳本身正确，不能将容器时间整体减去一天。本次保留原始相对时间，仅在共同区间中筛选满足覆盖标准的完整日；[维护者回复审计](../../data/metadata/alibaba-cpu-cld-20261031/timestamp-alignment.json)保留答复链接、作者、摘要及校验值。

批处理实例表处理 1,351,255,775 行，正时长、有效 CPU 和可连接主机的执行记录 1,316,547,080 条。平均 CPU 为 `cpu_avg/100` 核，按 start—end 与各小时的精确重叠积分；同样保留 `cpu_max/100` 恒定于整个区间的诊断积分。平均字段超过最大字段的记录有 419,826 条，因此最大字段积分不具有无条件的严格上界含义。小时内活跃形状、右删失与无效区间、重试/备份执行均不能从平均字段完整恢复。

| 完整窗口部分 | 核小时 | 占整机观测量 |
| --- | --- | --- |
| 在线服务（request 主解释） | 3,479,729.814 | 19.790% |
| 离线实例（平均 CPU 积分） | 8,990,855.225 | 51.133% |
| 有符号未对账差额 | 5,112,532.734 | 29.076% |
| 整机观测量 | 17,583,117.773 | 100% |

| 容器百分比分母解释 | 在线核小时 | 未对账差额核小时 | 集群小时负差额数量 |
| --- | --- | --- | --- |
| request | 3,479,729.814 | 5,112,532.734 | 0 |
| limit | 3,481,895.468 | 5,110,367.080 | 0 |
| one_core | 683,282.059 | 7,908,980.489 | 0 |

完整窗口内有 0 个集群小时、467 个机器小时的差额为负。所有差额保留原符号，不强制缩放组件或截断负数。差额可能包含系统开销、采样缺口、单位/计量窗口差异和实例均值分摊误差，不能全部认定为系统开销或在线服务。

完整窗口的容器样本支持为元数据 started 状态容器小时的 88.535%，逐小时比值为 85.903%—92.131%。该比值用原始有效样本数×10 秒与事件生命期对照，受元数据状态及计量粒度影响，不等同于逐时隙完整率；不据此填补在线 CPU 用量。

[逐小时对账和份额](component-reconciliation.csv)、[逐机器对账](machine-component-reconciliation.parquet)、[完整业务窗口](complete-component-window.csv)、[业务拆分审计](component-audit.json)为组件入口。另有两小表的[请求/计划包络](online-batch-request-envelope.csv)：在线请求依据容器事件保持，离线为 instance_num×plan_cpu×任务起止包络；它们表示配置需求，不能替代实际 CPU 份额。

### 日内、相对日形状与 Lisa 对照

完整窗口按相对时刻汇总后，平均需求最低为 23 时（26.37%），最高为 06 时（57.90%），日内平均曲线峰谷比 2.196。[在线/离线日内表](component-intraday.csv)及[相对日累计表](component-relative-days.csv)保留分项形状。源记录没有可核查的真实星期与时区，完整窗口只有五日，无法识别工作日/周末效应或稳定周周期；相对日形状不解释为真实周内规律。

| 组成 | 日内最低时刻 / 平均核小时 | 日内最高时刻 / 平均核小时 | 日内平均峰谷比 |
| --- | --- | --- | --- |
| 在线服务（request） | 04 时 / 18,187.876 | 10 时 / 36,432.219 | 2.003 |
| 离线实例（平均 CPU） | 23 时 / 43,700.626 | 06 时 / 136,577.544 | 3.125 |

| 相对日 | 整机核小时 | 在线占整机 | 离线占整机 | 有符号差额占整机 |
| --- | --- | --- | --- | --- |
| 2 | 3,596,120.027 | 19.625% | 51.292% | 29.083% |
| 3 | 3,499,772.792 | 19.865% | 48.650% | 31.485% |
| 4 | 3,578,319.771 | 18.952% | 53.621% | 27.426% |
| 5 | 3,211,496.501 | 20.510% | 50.588% | 28.902% |
| 6 | 3,697,408.683 | 20.066% | 51.396% | 28.538% |

| 情境 | 完整小时数 | 需求含义与分母 | 相对波动 CV | P95/P05 | 小时峰谷比 |
| --- | --- | --- | --- | --- | --- |
| A-CLD 纯 CPU 型·云通用计算 | 120 | 实测利用率；有效源核时 | 0.229 | 2.067 | 2.520 |
| A-HPC 纯 CPU 型·科研计算 | 7416 | Lisa 调度分配核心小时；观测小时峰值 | 0.324 | 3.943 | 最低为零，未定义 |

本样本的云负荷相对波动小于 Lisa 长窗口的调度分配曲线。二者窗口长短、分母和“活跃利用率/分配占用”含义不同，不能据此推断云机房的利用效率高低或全部业务的柔性大小。[Lisa 完整小时汇总](lisa-cpu-complete-hourly.csv)及两组日内表保留比较依据。Lisa 的现有官方审计记录许可为 CC BY 4.0，见[来源核查](../../data/metadata/research-gpu-demand-20261028/lisa-gpu-audit.json)。

### 年度构造、功率接口与可平移边界

复用 `annual_task_profiles.replay_hourly`：选定连续完整日期按原顺序循环，不平滑、不调整均值、不添加季节需求。2025 年 8760 小时 = 120 小时×73；2024 年 8784 小时 = 120 小时×73＋24 小时。源相对午夜映射到目标 UTC 午夜仅为相位设定。**没有季节变化**，不能代表源机房真实全年、节假日或需求增长。[平年映射](A-CLD-2025-replay-map.csv)与[闰年映射](A-CLD-2024-replay-map.csv)列出每个目标小时的源小时和循环位置。

`load_alibaba_source(project)` 返回与现有需求源一致的 `profile/clock/metadata`，可交给年度重放及 `ComputeLoadModel.calculate`。已用仓库的实际 2025 年气象小时表验证四档规模的调用和有限输出；该检查使用既有功率接口，目标设备的性能与功率映射仍需按服务器参数单独标定。未用源核心数量替代现代硬件性能换算。

在线服务刚性、离线批处理可推迟采用用户指定的情境假设。离线平均 CPU 积分给出候选工作量，原始任务 DAG 可约束执行先后；日志缺少可核查的独立提交时间、业务完成期限和检查点/抢占权限，因而不能将全部离线份额直接作为可移负荷。可平移研究应限于获得延迟窗口和权限的批处理，保持 DAG、资源总量及真实完成约束；在线部分和未对账部分在保守情境中保持原时刻。此处不推定任何实例已有可延后若干小时的权限。

离线正时长执行记录的时长分位数（秒）：P05=1、P25=3、P50=10、P75=31、P90=99、P95=194、P99=624。[精确秒级频数](components/batch-instance-duration-histogram.csv)和[分箱表](batch-duration-bins.csv)可复算。分布按执行记录数量统计，包含重试/备份等尝试；实例数量份额与 CPU 工作量份额不同，不能据短作业数量直接推算可移能量。

## 二、较新年份公开 CPU 云数据

核查日期为 2026-10-31，仅采用官方发布页、官方仓库和作者论文/存档；按版本年与真实采集年分别登记。检索式和全部字段见[机器可读登记表](../../data/metadata/alibaba-cpu-cld-20261031/public-cpu-cloud-datasets.json)，正文/元数据获取的 URL、读取范围、时间与 SHA 见[官方来源核验表](../../data/metadata/alibaba-cpu-cld-20261031/official-source-checks.json)。部分仓库文件的精确大小由[官方目录树元数据](../../data/metadata/alibaba-cpu-cld-20261031/newer-alibaba-repository-files.csv)复算。本部分没有下载任何较新来源的大数据文件。

### alibaba-cpu-2018

[官方来源](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2018；未明确 |
| 时长；规模 | 8 日；官方约 4000 台；本次核验 4034 台 |
| 时间分辨率 | 秒时间戳，监测位于 10 秒时间格；实际存在缺口 |
| CPU 指标 | 机器利用率、核数；容器利用率与请求；实例平均/最大 CPU |
| 功率 | 无 |
| 业务 | 在线服务与离线批处理混部 |
| 许可 | 官方允许研究和学习；无明确标准数据许可证 |
| 大小 | 六份压缩包合计 52320410918 字节；四小表 1907417361 字节，两大表 50412993557 字节 |
| 获取方式 | 官方 OSS HTTP，fetchData.sh；四小表在共享卷，两大表复用系统盘缓存并流式解压聚合 |
| A-CLD 适用性 | 主数据：真实混部机器 CPU，许可和字段解释须随结果保留 |

### alibaba-microservices-2021

[官方来源](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-microservices-v2021/README.md)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2021；2021 |
| 时长；规模 | 12 小时；1300+ 节点的 CPU 表；90000+ 容器；全调用图 20000+ 微服务 |
| 时间分辨率 | 机器 30 秒；容器 60 秒 |
| CPU 指标 | 有机器和容器 CPU 利用率；未配套核数表 |
| 功率 | 无 |
| 业务 | 生产微服务与混部环境 |
| 许可 | Alibaba program 研究/学习条款；无明确标准数据许可证 |
| 大小 | node 1.1 Gi；其余目录 25/16/19 Gi，官方未统一解释解压状态 |
| 获取方式 | 官方 fetchData.sh 与 OSS |
| A-CLD 适用性 | 可以核验短期 CPU 形状；缺完整日，无法提供周内或年度需求代表性 |

### alibaba-microservices-2022

[官方来源](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-microservices-v2022/README.md)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2022；2022 |
| 时长；规模 | 13 日；40000+ 节点、470000+ 容器、28000+ 微服务 |
| 时间分辨率 | 机器和容器 60 秒 |
| CPU 指标 | 有机器和容器 CPU；官方声明做过 min-max 归一化，未发布还原参数及机器核数 |
| 功率 | 无 |
| 业务 | 生产微服务与混部环境 |
| 许可 | Alibaba program 研究/学习条款；无明确标准数据许可证 |
| 大小 | 全量约 2 TB；Node 压缩约 10 MiB/小时，即 13 日约 3.05 GiB（据官方小时量估算） |
| 获取方式 | 官方 fetchData.sh 可按时间获取 NodeMetricsUpdate 分片；本次只读脚本 |
| A-CLD 适用性 | 较新中国云负载的形状对照；归一化值不可直接当作物理 CPU 份额或占用核小时 |

### alibaba-amtrace-2022

[官方来源](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-microarchitecture-v2022/README.md)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2022；未明确 |
| 时长；规模 | README 未明确，未将版本年份当作采集年份；原始 8577 台、约 98 万容器；公开抽样 1000 台 |
| 时间分辨率 | 按秒时间戳的容器×逻辑 CPU PMU；README 未明确采样间隔 |
| CPU 指标 | instructions、cycles、ref_cycles、核数及 CPU 类型；无直接机器 CPU 利用率列 |
| 功率 | 无 |
| 业务 | 延迟敏感在线业务与批处理混部 |
| 许可 | Alibaba program 研究/学习条款；无明确标准数据许可证 |
| 大小 | 公开抽样约 300 GB；原始超过 2 TB |
| 获取方式 | 官方 get_data.sh |
| A-CLD 适用性 | 支持 CPU 微架构和争用研究；换算整机核小时需要采样窗口、频率及 PMU 覆盖验证 |

### alibaba-gpu-2020

[官方来源](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2020/README.md)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2020；2020 |
| 时长；规模 | 7—8 月，两个月；约 1800 台、6500+ GPU |
| 时间分辨率 | 按实例生命期平均；秒时间戳 |
| CPU 指标 | 机器 CPU 总量与 user/kernel/iowait；按实例生命期平均，缺独立连续机器监测 |
| 功率 | 无 |
| 业务 | PAI 训练与推理 |
| 许可 | CC BY 4.0；该子目录 LICENSE 明确，不推广到其他版本 |
| 大小 | 各表官方压缩量约 53+663+60+198+388+34 MiB，加约 32 KiB 规格表 |
| 获取方式 | 官方 OSS 各表 URL |
| A-CLD 适用性 | 可研究 AI 作业宿主 CPU；同机多实例平均值会重叠，不能直接相加作为 A-CLD |

### alibaba-gpu-2023

[官方来源](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2023/README.md)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2023；未明确 |
| 时长；规模 | README 未给完整观测窗口；1523 个全部节点；8152 个默认任务；GPU 子集约 1200 台 |
| 时间分辨率 | 任务创建/调度/删除事件，秒 |
| CPU 指标 | CPU 库存与请求，无连续 CPU 利用率 |
| 功率 | 无 |
| 业务 | 异构 GPU 调度 |
| 许可 | Alibaba program 研究/学习条款；子项目另有代码许可时不自动推广到数据 |
| 大小 | 仓库 26 份 CSV 合计 12838024 字节，含不同配置变体；来源树元数据已登记 |
| 获取方式 | 官方仓库 csv/ |
| A-CLD 适用性 | CPU-only 子集仍来自 GPU 调度样本；资源请求不足以替代通用云 CPU 利用率 |

### alibaba-gpu-2025

[官方来源](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2025/README.md)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2025；未明确 |
| 时长；规模 | README 未明确；156 个服务，23871 个实例，其中 16485 个 CPU 节点实例 |
| 时间分辨率 | 实例事件，秒 |
| CPU 指标 | CPU request/limit，有 CPU 节点；无实际 CPU 利用率 |
| 功率 | 无 |
| 业务 | GPU 解耦推荐模型推理 |
| 许可 | Alibaba program 研究/学习条款；无明确标准数据许可证 |
| 大小 | 仓库 CSV 为 2118900 字节 |
| 获取方式 | disaggregated_DLRM_trace.csv |
| A-CLD 适用性 | CPU 节点服务于推荐推理，缺实测 CPU 需求，不能仅凭节点名称替代 A-CLD |

### alibaba-asi-2026

[官方来源](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/README.md)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2026；未明确 |
| 时长；规模 | 185 个相对日；峰值 155410 GPU、37707 GPU 服务器 |
| 时间分辨率 | 小时；网络仅相对日 109—115 |
| CPU 指标 | pod 的 cpu_request_cores、avg_cpu_request_util（相对请求）；server 表有 cpu_capacity_cores，无整机 CPU 活跃利用率 |
| 功率 | 未见实测功率表 |
| 业务 | 开发、训练、在线/离线 AI 推理 |
| 许可 | Alibaba program 研究/学习条款；无明确标准数据许可证 |
| 大小 | 四个 ZIP 分别 351803513445、3080121387、203904100、1188295031 字节 |
| 获取方式 | 官方 tre-clusterdata OSS 与 schema 文档 |
| A-CLD 适用性 | 适合 AI 用途；不代表纯 CPU 云通用计算 |

### alibaba-gentd26

[官方来源](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2026-GenAI/README.md)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2026；未明确 |
| 时长；规模 | README 未明确；匿名示例日期不代表采集日期；README 未明确节点总量 |
| 时间分辨率 | 事件/原始采样，按秒匿名时间戳 |
| CPU 指标 | 公开表有容器内存、GPU duty cycle，未见 CPU 利用率表 |
| 功率 | 无 |
| 业务 | Stable Diffusion 生成式图像推理 |
| 许可 | Alibaba program 研究/学习条款；无明确标准数据许可证 |
| 大小 | 仓库请求样例 CSV 为 2012418 字节；全量监测数据大小未汇总 |
| 获取方式 | 官方仓库列出的匿名 CSV |
| A-CLD 适用性 | 适合生成式 AI 服务过程；缺机器 CPU 需求 |

### alibaba-maas-2026

[官方来源](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2026-maas/README.md)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2026；未明确 |
| 时长；规模 | 日表相对日 0—181，缺 172；周表为日 140—146；研究覆盖 56 个集群、超过 100000 GPU |
| 时间分辨率 | 日、5 分钟、应用小时；5 分钟由 20 秒记录汇总 |
| CPU 指标 | 日/周实例表 cpu_usage 为平均占用核数；无纯 CPU 通用云整机时序 |
| 功率 | 未见实测功率表 |
| 业务 | 在线/离线 LLM 推理 |
| 许可 | Alibaba program 研究/学习条款；无明确标准数据许可证 |
| 大小 | ZIP 10528173427、4812189789、99947 字节 |
| 获取方式 | 官方 tre-clusterdata OSS；本次只读 README/schema/下载说明 |
| A-CLD 适用性 | 适用于推理服务用途，无法替代 A-CLD |

### alibaba-spot-gpu-2026

[官方来源](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2026-spot-gpu/README.md)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2026；未明确 |
| 时长；规模 | README 未明确；4278 个 GPU 节点、6 种 GPU 类型 |
| 时间分辨率 | 作业提交/运行时长，秒 |
| CPU 指标 | CPU 库存和请求，未见 CPU 利用率 |
| 功率 | 无 |
| 业务 | 高优先级与 spot AI 作业 |
| 许可 | Alibaba program 研究/学习条款；无明确标准数据许可证 |
| 大小 | 两个 CSV 合计 23134259 字节（23055929＋78330） |
| 获取方式 | node_info_df.csv、job_info_df.csv |
| A-CLD 适用性 | 可以研究 spot GPU 调度；不能替代通用 CPU 云 |

### google-borg-2019

[官方来源](https://github.com/google/cluster-data/blob/master/ClusterData2019.md)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2020；2019 |
| 时长；规模 | 2019 年 5 月，一个月；8 个 Borg cell；README 未给整月唯一机器总数 |
| 时间分辨率 | 5 分钟资源使用分布及事件 |
| CPU 指标 | CPU 请求、使用直方图、机器事件；归一化资源须按 v3 定义处理 |
| 功率 | 独立 PowerData2019 可关联 |
| 业务 | Google 内部混合生产/非生产业务；并非客户公有云 VM 样本 |
| 许可 | CC BY 4.0 |
| 大小 | 约 2.4 TiB 压缩等价数据 |
| 获取方式 | BigQuery 查询后仅导出小时汇总；可能产生查询费用 |
| A-CLD 适用性 | 主要跨厂商混部对照，窗口长于 Alibaba；不能将归一化 CPU 单位直接称为物理核 |

### google-power-2019

[官方来源](https://github.com/google/cluster-data/blob/master/PowerData2019.md)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2024；2019 |
| 时长；规模 | 从 2019-05-01 PT 开始一个月；10 个集群、57 个电力域，其中 8 个 cell 可关联 Borg |
| 时间分辨率 | 5 分钟 |
| CPU 指标 | 通过 machine→PDU 映射关联 Borg |
| 功率 | 总功率/容量比；production 部分为估计，包含机房地板冷却用电；无绝对容量 |
| 业务 | 生产数据中心电力域 |
| 许可 | CC BY 4.0 |
| 大小 | 官方 README/PDF 未汇总文件字节数 |
| 获取方式 | BigQuery 或 gs://powerdata_2019/ 压缩 CSV |
| A-CLD 适用性 | 用于资源与电力形状的对照，不能当作纯 IT 瓦数或实测生产业务功率 |

### azure-vm-2017

[官方来源](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV1.md)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2017；未明确 |
| 时长；规模 | 30 个连续日；2013767 VM、5958 个 subscription |
| 时间分辨率 | 5 分钟 |
| CPU 指标 | VM CPU min/max/mean、vCPU 数和生命期 |
| 功率 | 无 |
| 业务 | 单个地理区域内 first-party Azure VM 工作负载子集 |
| 许可 | 数据 CC BY 4.0；代码 MIT |
| 大小 | 78.5 GB 压缩、117 GB 解压、128 文件 |
| 获取方式 | 官方 Azure Blob URL 列表 |
| A-CLD 适用性 | 通用云 VM 对照；vCPU 与物理核心、宿主空闲和系统开销需另行处理 |

### azure-vm-2019

[官方来源](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2019；未明确 |
| 时长；规模 | 30 个连续日；2695548 VM、6687 个 subscription |
| 时间分辨率 | 5 分钟 |
| CPU 指标 | VM CPU min/max/mean；vCPU 数量分桶，须保留桶区间 |
| 功率 | 无 |
| 业务 | 单个地理区域内 first-party Azure VM 工作负载子集 |
| 许可 | 数据 CC BY 4.0；代码 MIT |
| 大小 | 156 GB 压缩、235 GB 解压、198 文件 |
| 获取方式 | 官方 Azure Blob URL 列表 |
| A-CLD 适用性 | 云通用计算的 VM 层对照；分桶限制精确核小时，不能直接重建宿主机器负荷 |

### azure-packing-2020

[官方来源](https://github.com/Azure/AzurePublicDataset/blob/master/AzureTracesForPacking2020.md)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2020；未明确 |
| 时长；规模 | 14 日收集，历史生命期尾部最多至相对日 90；一个 zone，README 未列 VM 总量 |
| 时间分辨率 | VM 生命期事件；时间为小数日 |
| CPU 指标 | VM 类型 CPU 申请量，以机器份额表示；无实际 CPU 利用率 |
| 功率 | 无 |
| 业务 | VM 放置与高/低优先级分配 |
| 许可 | CC BY 4.0 |
| 大小 | 发布页 README 未列大小 |
| 获取方式 | GitHub release ZIP 内 SQLite 数据库 |
| A-CLD 适用性 | 适合配置与容量研究；仅请求量不足以作为实际 CPU 需求 |

### azure-vm-noise-2024

[官方来源](https://github.com/Azure/AzurePublicDataset/blob/master/AzureVMNoiseDataset2024.md)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2025；2023—2024 |
| 时长；规模 | 483 日；12 个长驻 VM、43617 个短驻 VM；两种 SKU |
| 时间分辨率 | 每次基准测试结果；平均测试套件约 130 分钟 |
| CPU 指标 | CPU 等基准性能指标；无生产机器 CPU 利用率连续需求 |
| 功率 | 无 |
| 业务 | 受控性能与噪声实验 |
| 许可 | CC BY 4.0 |
| 大小 | 277 MB、1520 文件 |
| 获取方式 | 官方仓库 vm-noise-data/ |
| A-CLD 适用性 | 可以研究云性能波动；不适合作为生产负荷需求来源 |

### huawei-private-faas-2023

[官方来源](https://github.com/sir-lab/data-release/blob/main/README_data_release_2023.md)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2023；未明确 |
| 时长；规模 | 141 个有数据日，跨 235 日；论文概述与 README 的日数口径不同，以 README 为准；200 个内部函数，合并多个可用区 |
| 时间分辨率 | CPU/内存/限制/实例数为 1 分钟；调用与延迟有秒级 |
| CPU 指标 | 每函数 CPU 使用百分比与归一化 CPU limit；无机器核数和全机负荷 |
| 功率 | 无 |
| 业务 | 私有云生产 FaaS |
| 许可 | CC BY 4.0（官方仓库总 README 明确） |
| 大小 | README 未汇总，按指标通过 Google Drive 提供 |
| 获取方式 | 官方 SIR lab README 的 Google Drive 链接 |
| A-CLD 适用性 | 国内较新 serverless 子用途对照；有长时间缺口和归一化容量，不能代替整机总 CPU |

### huawei-public-faas-2023

[官方来源](https://github.com/sir-lab/data-release/blob/main/README_data_release_2023.md)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2023；未明确 |
| 时长；规模 | 26 日；5093 个函数、一个可用区，按官方 README；论文写 5019，保留差异 |
| 时间分辨率 | 1 分钟 |
| CPU 指标 | 公开表只有请求计数，未公开 CPU 使用率 |
| 功率 | 无 |
| 业务 | 面向客户的 FaaS |
| 许可 | CC BY 4.0（官方仓库总 README 明确） |
| 大小 | README 未列大小 |
| 获取方式 | 官方 Google Drive 链接 |
| A-CLD 适用性 | 请求形状可以作业务对照；缺 CPU 服务需求换算依据 |

### huawei-public-faas-2025

[官方来源](https://github.com/sir-lab/data-release/blob/main/README_data_release_2025.md)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2025；未明确 |
| 时长；规模 | 冷启动和聚合时序 31 日；详细请求子集 1 日；5 个区域；官方总说明报告约 850 亿请求 |
| 时间分辨率 | 请求事件；每函数每分钟 avg/std/分位数 |
| CPU 指标 | cpu_usage 为请求执行期间整个 pod 的平均 CPU 核数；分钟摘要也有 CPU 字段，另有 request/limit 配置 |
| 功率 | 无 |
| 业务 | FaaS 冷启动和请求处理 |
| 许可 | CC BY 4.0（官方仓库总 README 明确） |
| 大小 | README 未汇总，按区域/文件提供 |
| 获取方式 | 官方 SIR lab Google Drive 链接 |
| A-CLD 适用性 | 较新国内 FaaS CPU 使用对照；并发请求的 pod 平均量会重叠，分钟摘要不是全机平均，不能直接相加当作集群 CPU 需求 |

### tencent-cloud-volume-2025

[官方来源](https://github.com/Tencent/Cloud-Volume-Workload-Traces)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2025；未明确 |
| 时长；规模 | 最多两个月；README 未明确采集年，示例时间戳不作为年份依据；上海 10 个 warehouse、约 16000 云盘 |
| 时间分辨率 | 300 秒 I/O |
| CPU 指标 | vm_cpu 只有 VM 规格，无 CPU 利用率 |
| 功率 | 无 |
| 业务 | 云块存储 |
| 许可 | CC BY 4.0；README 的 CC-4.0 链接指向 /licenses/by/4.0/ |
| 大小 | 约 1.3 GB 压缩、7.3 GB 解压 |
| 获取方式 | 官方 GitHub Cloud_Disk_dataset.zip |
| A-CLD 适用性 | 不能从云盘 I/O 或静态 vCPU 数构造 CPU 需求；本次范围内未找到腾讯官方公开 CPU 利用率生产时序 |

### bytedance-iaas-2023

[官方来源](https://huggingface.co/datasets/ByteDance/CloudTimeSeriesData) / [作者论文](https://www.vldb.org/pvldb/vol18/p3812-gao.pdf)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2025；2023 |
| 时长；规模 | 3456 点，按 10 分钟折算 24 日；论文列 2023-06-30—2023-07-24，原始端点尚未下载核验；93 个 VM 序列 |
| 时间分辨率 | 10 分钟 |
| CPU 指标 | 论文明确为 VM CPU usage；数据卡只有 date/data/cols，未配套 vCPU、宿主与容量表，绝对核小时须补充规格和单位定义 |
| 功率 | 无 |
| 业务 | ByteDance 生产 IaaS 虚拟机 |
| 许可 | 数据 CC BY 4.0（官方数据卡）；论文许可与数据许可分别标注 |
| 大小 | iaas.csv 为 15185970 字节，来自官方文件元数据；没有下载 CSV |
| 获取方式 | ByteDance 官方 Hugging Face CloudTimeSeriesData/Datasets/iaas.csv |
| A-CLD 适用性 | 优先推荐的较新国内云 VM 形状对照；样本较小且经过预测数据预处理，缺宿主总量及在线/批处理标签 |

### bytedance-paas-2024

[官方来源](https://huggingface.co/datasets/ByteDance/CloudTimeSeriesData) / [作者论文](https://www.vldb.org/pvldb/vol18/p3812-gao.pdf)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2025；2024 |
| 时长；规模 | 7776 点，按 10 分钟折算 54 日；论文列 2024-09-01—2024-10-24，原始端点尚未下载核验；426 个服务实例序列 |
| 时间分辨率 | 10 分钟 |
| CPU 指标 | 服务 CPU 使用量，单位 milli-cores；除以 1000 可换算源 CPU 核数，无宿主总利用率 |
| 功率 | 无 |
| 业务 | ByteDance 生产 PaaS 服务 |
| 许可 | 数据 CC BY 4.0（官方数据卡）；论文许可与数据许可分别标注 |
| 大小 | paas.csv 为 168895841 字节，来自官方文件元数据；没有下载 CSV |
| 获取方式 | ByteDance 官方 Hugging Face CloudTimeSeriesData/Datasets/paas.csv |
| A-CLD 适用性 | 较新国内在线服务 CPU 需求对照；服务粒度和预处理范围须保留，缺整机库存与离线执行信息 |

### nep-edge-cloud-2020

[官方来源](https://github.com/xumengwei/EdgeWorkloadsTraces)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2021；2020 |
| 时长；规模 | 2020 年 6 月；额外三个月数据仅为小时带宽；NEP-small 14 个站点；NEP-large 139 个边缘站点，README 未列 VM 总数 |
| 时间分辨率 | VM CPU 5 分钟 |
| CPU 指标 | VM_CPU.csv 的 cpu_rate（0.6=60%），有 VM→PM 连接及物理机 cores 规格 |
| 功率 | 无 |
| 业务 | 中国生产公共边缘云 VM |
| 许可 | 仅限研究；官方禁止 offline sharing |
| 大小 | README 未列完整字节数 |
| 获取方式 | 作者官方仓库链接 Google Drive，本次未下载 |
| A-CLD 适用性 | 较新国内 VM CPU 对照；边缘站点业务与集中式云不同，缺测、VM 容量和聚合口径须核验 |

### sap-cloud-2024

[官方来源](https://zenodo.org/records/17357494) / [作者论文](https://arxiv.org/html/2510.23911v1)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2025；2024 |
| 时长；规模 | 30 日；论文标注 2024-07-31 00:00 UTC，精确起止须核对数据时间戳；约 1800 hypervisor、48000 VM |
| 时间分辨率 | 30—300 秒，随组件而异 |
| CPU 指标 | VM 与物理宿主 CPU 利用率、调度事件及内存网络指标 |
| 功率 | 未见实测功率字段 |
| 业务 | SAP 企业云，含 S/4HANA 和通用业务 |
| 许可 | CC BY 4.0 |
| 大小 | 最新 1.0.1 ZIP 为 1035045907 字节；旧 1.0.0 为 1032982604 字节 |
| 获取方式 | 作者 Zenodo，最新记录 17357494；本次只取记录元数据及论文 |
| A-CLD 适用性 | 优先推荐的较新年份对照；具备机器 CPU 和明确许可，但企业内存密集业务组成、采样覆盖须验证 |

### unimelb-ucc2023

[官方来源](https://zenodo.org/records/10069402) / [作者论文](https://hpc.ec.tuwien.ac.at/files/UCC_23_data_center_analysis.pdf)

| 字段 | 核查结果 |
| --- | --- |
| 发布/版本年；采集年 | 2023；未明确 |
| 时长；规模 | D1 约 9 个月、D2 约 5 个月；正文未明确采集年份；80+64=144 台物理机，约 1338 个常驻 VM |
| 时间分辨率 | 10 分钟 |
| CPU 指标 | 物理机 CPU 利用率、核心数量等 |
| 功率 | 物理机功率 W、温度和部分风扇指标 |
| 业务 | OpenStack 研究云，含科研、web、桌面等混合 VM |
| 许可 | CC BY 4.0 |
| 大小 | 613212066+184178675=797390741 字节，两份 CSV |
| 获取方式 | 作者 Zenodo 10069402；本次只取元数据及全文论文 |
| A-CLD 适用性 | 可作为较长时间窗和 CPU→功率外部对照；发布年不能证明采集年更近，研究云代表性有限 |

### 数据选择

**本研究主数据采用 Alibaba 2018**：六表提供同一混部环境的整机实测 CPU、在线资源和离线实例执行，可与当前 A-CLD 的用途定义衔接。完整窗口短、源 CPU 数量的逻辑/物理属性未明确、业务组件存在未对账量，这些限制随需求输入保留。

**较新年份优先对照 SAP Cloud Infrastructure 的 2024 年观测、2025 年发布数据**：提供 VM 与物理宿主利用率、30 日窗口和明确 CC BY 4.0，最新 1.0.1 约 1.04 GB。其企业内存密集业务组成与 Alibaba 的互联网混部不同；本轮仅核对元数据和论文，原始缺测、核数定义和连续窗口须在获取后核验，再决定是否升级为较新主样本。

**国内年份对照优先补充 ByteDance IaaS 2023 与 PaaS 2024**：前者有 93 个 VM 的 10 分钟 CPU 序列，后者有 426 个服务实例、以毫核计量的 CPU 序列；2025 年公开，数据许可为 CC BY 4.0。它们适合检验较新云需求形状，宿主容量、预处理和业务分类信息仍需核验；本轮没有下载这些 CSV。

Google Borg 2019 适合作为进一步的跨厂商混部对照，PowerData2019 还可对照资源与电力形状；应通过 BigQuery 聚合后仅导出小时结果，注意归一化 CPU、功率容量比和包含冷却的测量边界。Alibaba 2022 的 Node 表支持较新国内云负荷形状对照，但 min-max 归一化无法直接恢复绝对利用率及核小时。Azure 2019 是明确的通用云 VM 需求对照，须保留 vCPU 分桶区间和宿主映射缺口。国内还可选 NEP 2020 的生产边缘云 VM CPU；华为 2025 版有 CPU 核心用量和分钟摘要，适合较新 FaaS 对照，须避免并发请求重复归属整个 pod 的平均用量。

### 复现与验证

在研究服务器 Python 环境安装 `analysis/requirements-alibaba-cpu.txt`。下列 `PROJECT` 指服务器项目目录，`RESULTS` 指本目录对应的服务器位置。

```bash
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
```

恢复时使用相同参数即可读取检查点；已完成且 SHA 通过的输入直接复用。追加 `--weather <真实全年气象CSV>` 可核验四档功率接口；本次实际输入的 SHA 及检查范围见 [validation.json](validation.json)。[容器恢复独立验证](online-resume-validation.json)使用真实原文前 400 万行，[实例积分与恢复独立验证](batch-resume-validation.json)使用真实前 10 万行，均与独立计算方法比较通过。大表原始压缩包、恢复检查点及逐记录数据不提交 Git。[artifact-manifest.json](artifact-manifest.json)记录交付文件校验值。

遗留：业务拆分的计量一致性、源 CPU 的逻辑/物理含义、真实时区和星期、跨硬件吞吐与功率映射、可延迟窗口和抢占权限；较新数据的原始文件尚未下载核验。
