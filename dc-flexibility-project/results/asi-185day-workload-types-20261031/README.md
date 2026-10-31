# ASI 185 天全部任务类型与推理服务需求

数据版本：Alibaba `cluster-trace-gpu-v2026`。业务类型、优先级和模型领域均使用官方公开字段。服务器类型与用途采用用户确认的命名：CPU 主导型·推理服务（B-INF）、均衡型·推理服务（C-INF）、GPU 主导型·推理服务（D-INF），规模为 1MW、10MW、100MW、500MW。

本交付依赖[数据调研与 Range 可行性验证 PR #30](https://github.com/TuzzyT-Laiii/33/pull/30)，审阅顺序为先 #30、再本 PR。本分支从 #30 的 `d65213bf503d1d4b08eccbb108cad5dc5a8611aa` 派生；PR 保持待审，合并须经用户明确批准。

## 已完成覆盖与任务构成

**全部 185 天、4,440 小时已完成，主汇总有 197,473 个非空“小时×业务类型×优先级×模型领域”组合。** 读取 6,570,124,684 条 pod 小时行，其中申请 GPU 的行数为 524,624,662；另读取 40,522,321 条执行跨度。每个 pod/server 小时分区及执行跨度表均校验成员 CRC 和 SHA-256。

| 实测项目 | 结果 |
| --- | --- |
| 完成成员 | pod 4,440 + server 4,440 + execution summary 1，共 8,881 |
| 成员载荷字节 | **356,070,137,323 B**，约 356.07 GB |
| 已记录成功 Range 字节 | **356,100,010,415 B**；含定位/中央目录开销，不含独立复核、未提交在途读取与超时部分传输 |
| 全量后台阶段 | 2026-10-31 17:25:50–19:25:42 北京时间，**7,191.34 秒，即 1 小时 59 分 51 秒**；包含并发切换与续跑等待 |
| 检查点续跑 | 复用 **2,198 小时**；八个工作进程切换为十六个 |
| 传输重试 | 一次 120 秒读取超时后恢复；未发生最终失败 |
| 单进程最高 RSS | **10,103,226,368 B，约 9.41 GiB**，来自执行跨度表处理 |
| 原始文件写盘 | **0 B**；服务器只保留汇总、检查点与日志 |
| 检查点逻辑体积 | **481,798,134 B** |
| 运行目录实测占用 | 约 **483 MiB** 实际分配空间；核验时共享卷仍可用约 **6.48 TB** |
| 下载到工作区的服务器汇总 | **33 个文件、13,425,175 B**，逐文件字节与 SHA 全部匹配 |

处理审计见 [processing-audit.json](processing-audit.json)，日志/磁盘读回见 [execution-audit.json](execution-audit.json)，传输清单见 [delivery-manifest.json](delivery-manifest.json)。工作区另提交必要的说明与验证记录；没有提交原始 pod/workload 标识。

全期归属 GPU 资源为 **558,854,035.50 GPU·h**。占比按小时表归属资源计算：

| 业务类型 | GPU·h，百万 | GPU 小时占比 | 该类型中 LP 占比 | 该类型主要模型领域，按 GPU 小时 |
| --- | ---: | ---: | ---: | --- |
| online_inference 在线推理 | 299.739 | **53.635%** | 1.426% | rec 61.808%；genai 29.473% |
| training 训练 | 117.472 | **21.020%** | 6.764% | genai 59.250%；rec 29.646% |
| offline_inference 离线推理 | 116.113 | **20.777%** | 95.300% | genai 91.570% |
| dev 开发 | 19.300 | **3.454%** | 1.241% | dev 74.395%；rec 21.101% |
| other 其他已知类型 | 4.341 | **0.777%** | 0.484% | genai 68.329%；rec 31.263% |
| unknown 未知类型 | 1.889 | **0.338%** | 19.630% | unknown 71.223%；genai 28.777% |

优先级总体构成为 HP **77.144%**、LP **22.100%**、Other **0.756%**。模型领域总体构成为 genai **48.009%**、rec **40.496%**、unknown **5.252%**、dev **2.569%**、cv **2.175%**、embedding **1.499%**。在线推理的大部分资源属于推荐模型，不能将整条在线曲线称为纯 LLM 负荷。训练标签同样包含多种模型领域。

## 数据与统计口径

官方入口：[发布说明](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2026)、[字段定义](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/docs/schema.md)、[下载清单](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/docs/data_download.md)。来源允许研究和学习使用；该子目录未单列标准数据许可证，沿用 PR #30 的许可核验范围。

本次读取三个归档：

| 表 | 官方归档字节数 | 用途 |
| --- | ---: | --- |
| pod_hourly | 351,803,513,445 | 全部 185 天的任务类型、优先级、模型、GPU 资源和 pod/workload 计数 |
| server_hourly | 3,080,121,387 | 同小时全机群 GPU 库存与型号 |
| job_execution_summary | 1,188,295,031 | 已观测执行跨度的运行时长及申请 GPU 规模 |

主汇总粒度为 **相对日×小时×业务类型×优先级×模型领域**。逐小时记录归属 `used_gpu_hours`、申请 GPU 等效数量、申请 CPU 核数、pod 行数、该组合内精确去重 pod/workload 数；另外分别记录申请 GPU 大于零的 pod、workload 和行数。全部 pod 包含 GPU 服务器上的零 GPU 辅助容器。

`hourly-type-priority-model.parquet` 中的去重计数只适用于该小时与该字段组合。跨小时或跨组合相加，得到的是相应观察次数之和，不能解释为全期唯一任务数；汇总 CSV 的相关字段以 `sum_group_hourly_` 标明。`hourly-total.csv` 另保留每小时全表精确去重 pod/workload 数。缺失 workload 标识单独计数，未补造标识。

规模分布分为三种：小时 pod 行的申请 GPU 分布、同小时同组合内 workload 汇总申请 GPU 分布，以及执行跨度表每行的申请 GPU 分布。它们分别是 pod 小时行、组合内 workload 小时、执行跨度口径；不据此推断未观测的全期唯一作业规模。

## 运行时长与规模

官方执行跨度表只涵盖请求 GPU 的执行跨度；零 GPU 辅助 pod 的运行时长没有由该表提供。跨度计数与 185 天小时表的 pod 小时计数分别使用。

执行跨度表包含 **40,522,321** 行，全部用于时长与规模统计。`execution-span-metrics.csv` 提供类型×优先级×模型的精确分位数；`execution-duration-histogram.csv` 提供时长直方图；`execution-by-type-summary.csv` 提供各类型均值、最长跨度、长任务比例和经验分位数所在区间。跨组合汇总的分位数使用完整直方图定位区间，不平均各组分位数。

| 业务类型 | 执行跨度条数 | 平均时长，小时 | 中位数所在区间 | P90 所在区间 | 平均申请 GPU 等效数 |
| --- | ---: | ---: | --- | --- | ---: |
| training | 2,999,580 | 4.503 | 1–3 小时 | 6–12 小时 | 2.430 |
| online_inference | 5,054,990 | 28.847 | 1–3 小时 | 12–24 小时 | 0.762 |
| offline_inference | 32,194,936 | 1.416 | 10–30 分钟 | 3–6 小时 | 1.152 |
| dev | 31,708 | 91.321 | 6–12 小时 | 7–30 天 | 2.865 |
| other | 70,130 | 3.527 | 10–30 分钟 | 6–12 小时 | 10.343 |
| unknown | 170,977 | 4.390 | 10–30 分钟 | 30–60 分钟 | 1.225 |

分位数区间左闭右开。各类型每条执行跨度的最大申请量均为 16 个 GPU 等效资源；这不代表整个多 pod 工作负载的最大规模。在线推理和开发类别的均值明显高于中位数区间，表明存在较长的运行跨度；长期运行的服务 pod 不等同于一条持续数十小时的推理请求。

官方 `duration_hours` 表示观测到的执行跨度。该表没有起止日期与完成/截尾标志，不能核实每条跨度对应小时表的哪些日期，也不能将它当作请求延迟、业务截止时间或完整生命周期。最长跨度约 1,824 小时（76 天）不证明数据窗口只有 76 天。`gpu_request × duration_hours` 在分布表中明确命名为申请资源小时代理，与小时表的归属资源分别使用。

多 pod 工作负载的小时汇总规模与单条执行跨度不同。以下仅统计非空 workload 标识、正 GPU 申请的“组合内 workload 小时”观测，区间单位为申请 GPU 等效数量：

| 业务类型 | workload 小时观测数 | P50 区间 | P90 区间 | 最大有观测区间 |
| --- | ---: | --- | --- | --- |
| training | 5,192,641 | [8,16) | [64,128) | [1024,∞) |
| online_inference | 28,720,224 | [1,2) | [8,16) | [1024,∞) |
| offline_inference | 3,421,068 | [1,2) | [8,16) | [1024,∞) |
| dev | 5,357,395 | [2,4) | [8,16) | [256,512) |
| other | 250,740 | [8,16) | [64,128) | [512,1024) |
| unknown | 110,686 | [2,4) | [8,16) | [1024,∞) |

最高区间只给出已发布直方图的边界，不提供区间内精确最大值；小时汇总申请量也不保证所有 pod 在同一瞬间并发。详见 [workload-hour-scale-histogram.csv](workload-hour-scale-histogram.csv)、[pod-scale-histogram.csv](pod-scale-histogram.csv)与[执行跨度规模分布](execution-scale-histogram.csv)。

## 时间形态与柔性含义

日内形状按 185 天同一相对小时的均值计算；相对七日形状使用 `day % 7`，各分组按实际观测日数取平均。公开数据没有原始日历、星期或时区，不能将分组直接称为星期一或周末。跨日变异系数包括趋势和机群扩容影响，另提供按同小时库存归一化后的日均份额变异系数。

| 业务类型 | 日内均值峰谷比 | 相对七日均值峰谷比 | 日总量 CV | 末 14 日 / 首 14 日日均量 |
| --- | ---: | ---: | ---: | ---: |
| online_inference | 1.012 | 1.022 | 0.131 | 1.441 |
| offline_inference | 1.365 | 1.022 | 0.187 | 1.307 |
| training | 1.083 | 1.090 | 0.128 | 1.464 |
| dev | 1.023 | 1.036 | 0.119 | 1.382 |
| other | 1.232 | 1.101 | 0.636 | 1,012.574 |
| unknown | 8.798 | 1.232 | 0.397 | 0.627 |

在线推理的平均日内形状较平，离线推理的日内与跨日波动更大。other 的增长倍率受到首期极小基数影响；unknown 的形状也受到未识别业务构成影响，均不能据此认定存在可移负荷或季节效应。库存端点从 111,321 增至 155,410；在线推理按库存归一化后的日均份额 CV 为 0.061，低于原始日总量 CV，说明描述增长时须同时考虑机群规模。

| 业务类型 | 数据支持的柔性审阅方向与边界 |
| --- | --- |
| online_inference | 98% 以上归属量为 HP；补充请求延迟、尾延迟与可用性约束后研究放置和容量策略，不能由 pod 长运行推断请求可延期 |
| offline_inference | LP 占 95.30%、跨度中位数为 10–30 分钟，可作为批推理候选；还需完成期限、批次和数据依赖 |
| training | 训练跨度中位数为 1–3 小时，但 LP 仅占 6.76%；检查点、恢复开销、通信与最终期限需要另行确认 |
| dev | 存在较长常驻跨度；交互式会话、真正空闲与允许休眠的边界未知 |
| other | 业务含义与增长来源需进一步识别后才能设定约束 |
| unknown | 保留未识别状态，不自动赋予可中断或可延后权限 |

训练和离线推理可进入截止时间、检查点、吞吐和数据依赖的进一步审阅；较长运行跨度或 LP 标签均不足以直接授予迁移、暂停或延后权限。在线推理需要优先约束请求与尾延迟目标；小时级资源表无法恢复这些目标。开发任务可能包含交互式会话；other、unknown 保留分类不确定性。

## INF 185 天曲线与年度接入

`inf-hourly.csv` 分别保存在线、离线与组合推理的 4,440 小时资源需求、GPU 库存及申请 CPU。推理曲线只选 `online_inference` 和 `offline_inference`，训练、开发及其他类别用于全部任务的独立统计。模型类别保留源定义：`genai` 可能包含 LLM、视觉语言模型和图像生成，不能单独等同于纯 LLM。

在线与离线合计 **415,851,928.88 GPU·h，占全期归属量 74.412%**。

| 曲线 | 全期 GPU·h，百万 | 平均每小时 GPU·h | 最大单小时 GPU·h | 正观测小时峰谷比 | 日内均值峰谷比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 在线推理 | 299.739 | 67,508.72 | 85,975.76 | 1.599 | 1.012 |
| 离线推理 | 116.113 | 26,151.62 | 44,320.84 | 7.640 | 1.365 |
| 组合推理 | 415.852 | 93,660.34 | 124,677.70 | 1.786 | 1.102 |

全时段包含一个共同零资源小时，因此全时段最大/最小值之比未定义；“正观测小时峰谷比”只使用大于零的观测，没有替换或插值。组合推理日总量 CV 为 **0.124**，相对七日均值峰谷比为 **1.018**，末 14 日日均归属资源为首 14 日的 **1.404 倍**。所有趋势均是相对窗口内的描述。

三条形状共用**组合推理的最高小时归属 GPU 资源量**作为参考分母。原始 GPU 小时、同小时库存和库存比值完整保留；形状归一化不改变异常记录，也不将归属量称为物理利用率。异构 GPU 尚未换算为指定硬件的吞吐等价；B-INF、C-INF、D-INF 的性能、CPU 配套和设备功率须按对应配置另行标定。

`asi_inference_profiles.load_asi_sources()` 读取并校验来源 SHA，提供 `asi_gpu`（组合）、`asi_online_gpu`、`asi_offline_gpu`。年度资源入口复用既有 `annual_task_profiles.replay_hourly()`；`build_task_annual_loads.py` 的 ASI 输入改为上述三条 185 天曲线，结果写入独立的 `task-annual-loads-asi185-20261031` 目录。历史曲线及其按版本复现的入口保留。

默认读取带质量标识的 [inf-hourly-quality.csv](inf-hourly-quality.csv)，其资源数值与 [inf-hourly.csv](inf-hourly.csv)逐项完全一致。相对小时 126 的共同零资源观测通过 `source_quality_flag` 传到年度来源映射和后续功率输出；未自动认定为停机或可调节空间。

平年 8,760 小时 = 4,440 + 4,320 小时：先保留全部 185 天，再接相同窗口的前 180 天。因此相对日 0–179 重复两次，180–184 只出现一次。连接处保留原始边界跳变；185 不能被 7 整除，循环边界会重置相对七日相位。目标 UTC 午夜和源午夜的对应关系是建模设定，源日期与季节未被识别；该重放不能恢复缺失的全年季节变化、节假日或真实年度增长。

`build_asi_inference_annual.py` 提供 `--phase-days 0..184`，输出来源小时映射和三条资源需求。在线、离线与组合曲线的资源可相加；分别计算的设施功率含各自空闲和固定组件，不能直接相加作为同一机群功率。本次验证资源重放接口，未运行年度气象、设备功率与柔性优化全流程。

已提交 [2025 年 8,760 小时资源重放](inf-annual-replay-2025-phase000.parquet)及[来源清单](inf-annual-replay-2025-phase000.json)。按原值重放，在线、离线、组合推理的年度积分分别为 **589,609,112.89 / 228,904,465.01 / 818,513,577.90 GPU·h**，属于年度构造量。共同零资源观测在目标年出现两次，均保留质量标识。日序列相位检查不等于已解决源时区与目标天气之间的小时相位。

## 数据质量与适用限制

- **共同零资源小时：**相对日 5、6 时（小时索引 126）归属 GPU、申请 GPU 与申请 CPU 均为零，但仍有 **1,123,104** 条 pod 观测和 **113,047** 张库存 GPU。原始字段无空值，仍不能确认这是真实停机；[quality-hour-audit.csv](quality-hour-audit.csv)和[质量报告](quality-summary.json)保留原值与标识，没有插值。量化真实功率或柔性收益前，需要核验该小时或明确缺测处理及敏感性政策。
- **归属量与库存：**有 **409 小时**归属 GPU 小时/同小时库存大于 1，最高 **1.049695**；申请量/库存有 **4,293 小时**大于 1，最高 **1.376936**。原值未截断，逐小时记录见 [inventory-attribution-anomalies.csv](inventory-attribution-anomalies.csv)。小时内 pod 更替及来源归属口径仍需审计，不能把这些比值直接称为物理利用率。
- **标识覆盖：**workload 标识缺失 **2,513,855,734** 行，约占全部 pod 小时行 38.26%；相关唯一计数与 workload 规模分布仅使用已知标识。全期逐小时行数与全表精确去重 pod 数之差累计 **6,987**，资源量继续按来源行累计，不未经依据删除观测。
- **资源字段：**GPU 申请量和归属 GPU 小时无空值；正申请以外没有超过数值容差的归属量。辅助 pod、申请资源与实际忙碌资源分别解释。异构卡、CPU 实际忙碌时间、硬件功率和 SLA 没有由该数据完整确定。
- **时间覆盖：**185 天小时表、无起止日的执行跨度表、相对七日分组及年度重放各有不同证据范围；不把跨度数当作已识别的 185 天完整到达任务数，不将六个月形状解释为全年季节性。

## 验证与产物入口

[validation.json](validation.json)记录真实输入验证：8,881 个成员、4,440 小时键、资源与计数守恒、pod/workload 规模直方图、GPU 型号与状态分组守恒；相对小时 0 与 4,439 独立重读并用 pandas 对照 Arrow，另与 PR #30 的 48 个端点小时核对。平年重放资源积分、质量标识传播，以及三条源曲线在闰年两种日相位下的六个 8,784 小时映射均通过。

[交付字节核验](delivery-validation.json)确认 33 个服务器汇总在本地和 Git 暂存区的 SHA-256 一致，并记录当前代码指纹。原始处理与首次年度重放的代码版本保留于处理审计；质量标识后的代码与验证版本另行记录。

- [小时×类型×优先级×模型主表](hourly-type-priority-model.parquet)、[类型×优先级×模型总量](type-priority-model-total.csv)、[类型统计](task-type-summary.csv)。
- [逐小时类型资源](hourly-by-type.csv)、[逐日类型资源](daily-by-type.csv)、[日内形状](diurnal-by-type.csv)、[相对七日小时形状](relative-week-by-type.csv)、[相对七日日均量](mod7-by-type.csv)。
- [执行跨度分布](execution-duration-histogram.csv)、[类型跨度统计](execution-by-type-summary.csv)、[GPU 型号构成](gpu-model-total.csv)、[来源状态构成](state-total.csv)。
- [逐成员审计](member-audit.parquet)、[检查点 SHA 清单](checkpoint-manifest.csv)、[续跑验证](resume-validation.json)、[独占锁验证](exclusive-lock-validation.json)。

## 后台处理、检查点与复现

服务器项目：`/data3/dataset/ljt/global-grid-flexibility-20261023`。本任务代码位于 `analysis/asi-185day-workload-types-20261031/`，运行与汇总位于 `results/asi-185day-workload-types-20261031/`。

程序使用 `setsid nohup`，stdin 指向 `/dev/null`，日志写入 `run.log`；启动后确认 PPID 1、独立 SID 和无终端。工作进程各自以 HTTP Range 定位 ZIP_STORED 成员，由标准 `zipfile` 处理 ZIP/ZIP64 和 CRC，再由 Arrow 解析 Parquet。请求要求精确 HTTP 206、Content-Range 与长度，并用 ETag 固定对象版本；成员 SHA 随审计保存。原始 ZIP 和 Parquet 均只在内存中读取。本次前期使用八个进程，后续在共享 CPU 竞争下按检查点续跑并调整为十六个进程，调整前状态与时间写入 `restart-events.json`。

每小时完成后以临时汇总文件、fsync 和原子重命名写入 `checkpoints/hour-XXXXX.json`；执行跨度另存 `checkpoints/jobs.json`。`status.json`、`progress.jsonl`、`runs.jsonl` 记录状态与进度；独占文件锁阻止重复运行。续跑读取已有检查点并校验对象版本，跳过已完成小时。已用真实首小时验证复用行为。传输异常有限重试并记入日志，最终失败保留可定位错误和已完成检查点。

全量完成后，另一独立后台进程依次生成汇总、重读端点样本、构造年度资源重放并执行验证，状态写入 `postprocess-status.json`，日志为 `postprocess.log`。这些处理均不依赖本地前台连接。

共享卷以 `/data3/dataset` 为检查对象，运行前后及运行期间保留 30 GiB 空间加 2 GiB 并发余量。运行时长表的单成员约 1.19 GB，原始内容只在内存，因此其内存峰值高于 PR #30 的小时分区抽样；审计保留本轮实测峰值。

依赖见 `analysis/requirements-asi-185days.txt`。在服务器项目配置好代码路径后：

服务器独立代码目录需要本 PR 的 `analysis/*asi*.py`、`annual_task_profiles.py`、`build_task_annual_loads.py` 和 `launch_asi_185days.sh`；将 PR #30 的 `results/inference-data-alternatives-20261031/asi-hourly.csv` 复制到该代码目录的 `baseline/asi-hourly.csv`，供端点复核使用。既有 `compute_load` 等模块通过项目 `analysis` 路径导入，启动脚本为后台汇总设置该 `PYTHONPATH`。

```bash
bash analysis/asi-185day-workload-types-20261031/launch_asi_185days.sh
cat results/asi-185day-workload-types-20261031/status.json
python analysis/summarize_asi_185days.py --run-dir results/asi-185day-workload-types-20261031 --output results/asi-185day-workload-types-20261031 --baseline results/inference-data-alternatives-20261031
python analysis/annotate_asi_quality.py --directory results/asi-185day-workload-types-20261031
python analysis/build_asi_inference_annual.py --year 2025 --phase-days 0
python analysis/verify_asi_185days.py --run-dir results/asi-185day-workload-types-20261031
```

复现时先检查 `launch.pid`、`status.json` 与检查点。进程仍运行时读取进度；进程已结束时使用相同输出目录续跑。`checkpoint-manifest.csv` 提供检查点字节与 SHA；`member-audit.parquet` 提供 8,881 个成员的名称、CRC、SHA、行数与字段。公开提交仅包含脚本、汇总和审计，不含原始记录或 pod/workload 标识。
