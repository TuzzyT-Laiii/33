# MIT GPU OTHER 的用途证据、运行形态与终态交叉

分析日期：2026-11-01；分类版本：`mit-other-refine-20261101-v2`。本需求继续维护 [PR #28](https://github.com/TuzzyT-Laiii/33/pull/28)，依赖 [PR #26](https://github.com/TuzzyT-Laiii/33/pull/26) 的 GPU 需求口径，审阅时先阅读 PR #26。原始口径代码基线为 `job-flexibility-constraints-20261030@70e468a`；本次深化从 PR #28 的 `5b4a747` 开始。原 PR 的分析文件保持原样。

**OTHER 的 73.6338% MIT GPU 小时可以按下面十二行使用。官方标签与 DNN 候选四行合计 49,862.30 GPU 小时，占 MIT 3.4148%、OTHER 4.6375%；长时、数组和多节点类别只确认运行形态。仍无法判断为 OTHER GPU 小时的 5.0618%，不能将其补集解释为已经识别了真实用途。** 终态作为独立交叉属性保存。

## 主表：互斥类别、证据范围与资源占用

运行、排队单位为小时；P50/P90 按作业等权计算。GPU 数为分配数量。占比分母保持 PR #26：MIT **1,460,190.6641666667 GPU 小时**（展示值 1,460,190.66），OTHER **1,075,193.2205555556 GPU 小时**。GPU 小时包含完成、取消、超时、失败和低活动分配。类别按下文优先顺序互斥，表格按用途与形态排列。

| OTHER 细分类别 | 条数 | GPU 小时 | 占 MIT | 占 OTHER | 置信度及范围 | 运行 P50/P90 | 排队 P50/P90 | GPU 数 P50/P90 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| DNN训练（官方标签） | 3,430 | 40,071.89 | 2.7443% | 3.7269% | 实测：DNN用途 | 2.732 / 10.858 | 14.624 / 47.842 | 2 / 4 |
| 推断DNN训练（同用户强匹配） | 578 | 4,407.83 | 0.3019% | 0.4100% | 高：DNN用途推断 | 0.809 / 6.403 | 4.638 / 37.170 | 2 / 4 |
| 推断DNN训练（同用户近似） | 20 | 333.09 | 0.0228% | 0.0310% | 中：DNN用途推断 | 1.513 / 4.753 | 20.555 / 103.688 | 8 / 8 |
| DNN训练候选（正例相似） | 895 | 5,049.49 | 0.3458% | 0.4696% | 中：DNN用途推断 | 4.271 / 10.282 | 7.702 / 21.461 | 1 / 2 |
| 长时单任务 | 13,841 | 577,965.49 | 39.5815% | 53.7546% | 高：运行形态 | 15.566 / 72.004 | 0.001 / 13.789 | 1 / 2 |
| 数组批量作业（扫描候选） | 25,471 | 209,225.46 | 14.3286% | 19.4593% | 高：数组形态 | 1.563 / 14.932 | 0.736 / 30.238 | 1 / 1 |
| 多节点大作业 | 878 | 161,427.17 | 11.0552% | 15.0138% | 高：运行形态 | 1.365 / 50.407 | 0.002 / 14.993 | 8 / 16 |
| 交互式会话候选 | 664 | 1,326.62 | 0.0909% | 0.1234% | 低：用途弱推断 | 0.918 / 4.013 | 0.000278 / 0.000556 | 1 / 2 |
| 启动失败或调试候选 | 831 | 148.89 | 0.0102% | 0.0138% | 中：短时异常形态 | 0.108 / 0.146 | 0.003 / 2.339 | 1 / 2 |
| 低GPU活动占用（监测） | 1,124 | 20,528.24 | 1.4059% | 1.9093% | 实测：GPU活动形态 | 5.391 / 27.608 | 0.000278 / 4.331 | 1 / 2 |
| 短时或测试作业 | 1,662 | 284.51 | 0.0195% | 0.0265% | 高：运行形态 | 0.122 / 0.159 | 0.001 / 0.482 | 1 / 2 |
| 仍无法判断 | 27,010 | 54,424.53 | 3.7272% | 5.0618% | 低：用途与形态待定 | 1.129 / 4.134 | 1.107 / 10.039 | 1 / 2 |
| 合计 | 76,404 | 1,075,193.22 | 73.6338% | 100.0000% | — | — | — | — |

[other-breakdown.csv](other-breakdown.csv)保存完整精度、P99、CPU/内存/节点分位数及每类迁移与拆分判断。强匹配表示相似资源配置与同用户训练历史的联合证据，不构成官方用途标签；数组参数扫描仅为用途候选。低 GPU 活动不说明 CPU、节点或业务整体闲置。

| 置信度档 | 条数 | GPU 小时 | 本档占 OTHER | 累计占 OTHER | 累计占 MIT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 实测 | 4,554 | 60,600.13 | 5.6362% | 5.6362% | 4.1502% |
| 高 | 42,430 | 953,310.47 | 88.6641% | 94.3003% | 69.4369% |
| 中 | 1,746 | 5,531.47 | 0.5145% | 94.8148% | 69.8157% |
| 低 | 27,674 | 55,751.15 | 5.1852% | 100.0000% | 73.6338% |

这些档位对应**各行声明的证据范围**：实测同时包含标签用途和监测活动；高档的大部分为运行形态。它们不代表 DNN 概率或分类准确率。[confidence-coverage.csv](confidence-coverage.csv)提供累计条数与资源小时。若只计主表前四行，DNN 用途证据累计覆盖 OTHER 4.6375%；官方标签自身覆盖 3.7269%。

独立于主类顺序的 [dnn-evidence-coverage.csv](dnn-evidence-coverage.csv)采用“标签→强匹配→近似→相似度”的证据顺序，共覆盖 OTHER 4.6549%。其中同用户强匹配 582 条，4 条在主表优先归入短时异常或低活动形态。此表与主表不能相加。

## 官方来源、输入与许可证

输入为 `D:/mit-supercloud-gpu-summary-20261028/reference/slurm-log.csv`，395,914 条唯一作业、98,176 条请求 GPU。请求筛选为 `tres_req` 中 1001、1002 数量和大于零；GPU 数为 `tres_alloc` 中两者之和。运行时长为 `time_end-time_start`，排队为 `time_start-time_submit`，资源小时为分配 GPU 数乘运行秒数再除以 3600。GPU 区间和等待时间均有效，`time_suspended=0`。[baseline-reconciliation.csv](baseline-reconciliation.csv)逐类核对 PR #26 的四种提交工具类别。

官方前缀为 `s3://mit-supercloud-dataset/datacenter-challenge/202201/`。2026-10-31 已通过 `http://127.0.0.1:7897` 执行 HEAD 并流式下载标签表；五个文件均小于 2,000,000,000 字节。本次使用相同文件。

| 官方文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| [labelled_jobids.csv](https://mit-supercloud-dataset.s3.us-west-2.amazonaws.com/datacenter-challenge/202201/labelled_jobids.csv) | 82,332 | `bd00b9e572442ed9ad900c8919de789e0a4bf225df8a657be9c7b389d7ca6d82` |
| [labelled_job_stats.csv](https://mit-supercloud-dataset.s3.us-west-2.amazonaws.com/datacenter-challenge/202201/labelled_job_stats.csv) | 339 | `3c23d07cc0f923a7ec0415f301d9243b83bf36fdb8b30a6774f9503ff3731903` |
| [README.md](https://mit-supercloud-dataset.s3.us-west-2.amazonaws.com/datacenter-challenge/202201/README.md) | 7,616 | `915b12d2110266fe4cad8a8d0f831f75d3ae08cf9ba42d106ac82a5a5f84c771` |
| [LICENSE](https://mit-supercloud-dataset.s3.us-west-2.amazonaws.com/datacenter-challenge/202201/LICENSE) | 285 | `a68e8239ee146c3e58c8988b66b176708310d2595095bf8d16ebb65da728dd0f` |
| [tres-mapping.txt](https://mit-supercloud-dataset.s3.us-west-2.amazonaws.com/datacenter-challenge/202201/tres-mapping.txt) | 111 | `4897c5cf1575664217f93d9c66cda3c06cae9d90f827a5dadc5483044dd64a51` |

[download-manifest.csv](download-manifest.csv)含 URL、ETag、Last-Modified；[input-fingerprints.csv](input-fingerprints.csv)保留输入指纹。Slurm SHA-256 为 `27259e5b8e6daecd80b7853126af40ba759bc25edc9053fc174ef7a4f186aa01`，主分析前核对 PR #26 指纹，运行后核对大小与修改时间。

许可证为 **CC-BY-NC-ND 4.0**；原始数据、用户关联、逐文件清单和逐作业结果仅存本机，仓库只提交脚本与汇总。署名：Samsi et al., *The MIT Supercloud Dataset*, HPEC 2021，[DOI:10.1109/HPEC49654.2021.9622850](https://doi.org/10.1109/HPEC49654.2021.9622850)。[MIT 官方说明](https://github.com/MIT-AI-Accelerator/MIT-Supercloud-Dataset)描述已知模型训练，其中 Bert/DistilBert 为微调；模型家族不能证明预训练、RL 或在线推理用途。

## 标签覆盖与模型家族

标签文件实际 **3,430 行、3,430 个唯一 ID**，全部关联到 GPU OTHER，匹配率 **100%**，40,071.8875 GPU 小时。其他 job_type、无 GPU 请求和未匹配均为零。[coverage-totals.csv](coverage-totals.csv)和[label-coverage-by-job-type.csv](label-coverage-by-job-type.csv)保存结果。官方 README 的 3,425 与下载文件记录数不同，本次采用实际记录。`labelled_job_stats.csv` 是 26 个模型变体计数，逐模型核对一致，没有逐作业监测统计。

| 模型家族 | 条数 | GPU 小时 | 占 MIT | 占 OTHER | 运行 P50/P90 | 排队 P50/P90 | GPU 数 P50/P90 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| U-Net | 1,431 | 15,495.24 | 1.0612% | 1.4412% | 2.059 / 10.953 | 27.999 / 60.265 | 2 / 4 |
| VGG | 560 | 6,923.07 | 0.4741% | 0.6439% | 4.777 / 13.588 | 9.691 / 35.723 | 2 / 4 |
| Inception | 484 | 5,350.45 | 0.3664% | 0.4976% | 4.920 / 11.484 | 11.428 / 31.588 | 2 / 4 |
| Bert | 189 | 4,938.97 | 0.3382% | 0.4594% | 1.368 / 5.240 | 9.050 / 17.318 | 2 / 64 |
| ResNet | 463 | 3,713.62 | 0.2543% | 0.3454% | 2.823 / 8.332 | 15.851 / 38.268 | 2 / 4 |
| GNN | 131 | 3,203.93 | 0.2194% | 0.2980% | 1.664 / 11.159 | 0.004 / 0.727 | 2 / 256 |
| DistilBert | 172 | 446.60 | 0.0306% | 0.0415% | 0.752 / 3.050 | 4.711 / 12.969 | 2 / 2 |

[模型家族](labelled-other-model-families.csv)与[26 个变体](labelled-other-model-variants.csv)提供完整分位数。映射为 `U*→U-Net`、`vgg*→VGG`、`resnet*→ResNet`、`inception*→Inception`，Bert/DistilBert 分开，`dimenet/schnet/conv/pna→GNN`。标签只有 **3 位用户**；OTHER 有 **246 位用户**。标签用户共 4,255 条 OTHER、47,164.18 GPU 小时（OTHER 4.3866%），限制了同用户传播的潜在覆盖。仓库不发布用户标识。

## 同用户传播与正例相似度

采用**正例与无标签集合的相似度检索**。未标注集合包含训练，不当作真实负例；不训练二元用途分类器，不报告精确率或总体准确率。分类为回顾性描述，包含运行时长等事后字段，不能直接用作提交时在线预测。

同用户传播要求 GPU 数、分区、有限/非有限时限类型相同，提交时间相差最多 90 天。强匹配要求 CPU、总内存和 `1+有限时限分钟` 的较大值/较小值均不超过 1.25，至少两个官方正例支持；近似要求 CPU/内存比不超过 1.5、时限比不超过 2，至少一个正例支持。全量回顾允许前后 90 天标签；时间检验只使用较早标签。

强匹配新增 **582 条、4,594.23 GPU 小时**（OTHER 0.4273%）；合并近似后新增 **602 条、4,927.32 GPU 小时**（OTHER 0.4583%）。此为用途证据标记，含主类分到异常/低活动的 4 条；主表强匹配为 578 条。[user-propagation-coverage.csv](user-propagation-coverage.csv)保存完整覆盖。

资源比较使用 `tres_req` 的 CPU=1、总内存=2；原始 `mem_req` 包含 MEM_PER_CPU 高位，不直接作为容量。4 条 GPU 作业缺失内存请求，保留缺失并排除资源匹配。时限单位分钟，`4294967295` 为 INFINITE、`4294967294` 为 NO_VAL，合并为非有限类型；不能当作业务期限。数组使用 `id_array_task != 4294967294`。

跨用户相似度为五个最近正例的平均加权 Manhattan 距离。资源数量采用 log2，内存/等待/有限时限采用 `log2(1+x)`，提交小时与星期采用周期 sin/cos。权重：GPU=1，CPU/内存各 0.25，内存缺失=4，节点/运行时长各 0.5，有限时限=0.1，非有限标记=0.5，等待=0.025，数组=4，分区独热各 1，小时 sin/cos 各 0.1，星期 sin/cos 各 0.05。[完整权重](similarity-feature-weights.csv)与脚本逐项公开。

距离阈值为参考正例内部留一五邻域均值的 P90，每折只从参考集计算。还要求与最近正例的 GPU 数、分区、有限类型相同，CPU/内存比不超过 1.25、运行时长比不超过 2、非数组。无足够监测时只保留中等置信候选；有合格监测时须符合下文正例活动范围。

正例与无标签的分布 KS 距离包括 CPU **0.7252**、内存 **0.7212**、有限时限 **0.6613**、GPU **0.5787**、等待 **0.4617**、数组 **0.3573**、提交小时 **0.2904**、星期 **0.2639**、运行时长 **0.2230**。[诊断表](positive-unlabelled-feature-diagnostics.csv)还含严格较早的同用户训练历史（KS 0.9892）与样本量；[分区表](positive-unlabelled-partitions.csv)提供频率。差异受三位用户的提交模板影响，不能解释为独立的用途识别能力；用户历史由传播规则使用，不重复计入跨用户距离。

### 隐藏标签后的找回检验

从参考集完全删除检验作业，再匹配 Slurm 特征。随机检验按模型家族分层五折，时间检验使用较早 80% 预测最后 20%，用户检验每次整位用户留出。联合表示同用户近似或相似度命中，以下均为**元数据方法**找回率；监测筛选另行检验。

| 留出方案 | 隐藏条数 | 联合找回条数 | 联合条数找回率 | 联合 GPU 小时找回率 | 同用户强匹配找回率 | 仅相似度找回率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 随机分层五折 | 3,430 | 3,421 | 99.7376% | 99.5304% | 99.4169% | 90.0583% |
| 最后 20% 时间留出 | 686 | 484 | 70.5539% | 43.3586% | 69.3878% | 46.7930% |
| 整位用户留出 | 3,430 | 1,096 | 31.9534% | 22.4904% | 0.0000% | 31.9534% |

[逐折检验](label-holdout-validation.csv)及[合并检验](label-holdout-totals.csv)保留每个方法和 GPU 小时找回率。随机检验共享用户和近似配置，结果较乐观；时间和用户留出显示较弱的推广能力。没有真实负例，无法换算新增候选的 DNN 正确率。已知 DNN 中 793 条运行至少 6 小时（标签条数 23.1195%、标签 GPU 小时 48.0106%）；[模型家族与长时形态交叉](label-model-rule-crosswalk.csv)中 `runtime_shape=True` 表示至少 6 小时。单靠长时阈值会遗漏多数已知训练作业。

## 监测组织、下载成本与既有观测复用

官方 GPU 对象位于 `datacenter-challenge/202201/gpu/0000/` 至 `0099/`，文件名为 **`<id_job>-<node>.csv`**，按作业和节点组织。独立 LIST 元数据得到 **109,946 个文件、2,436,905,883,610 字节**，本机清单保留键、大小及 ETag。

计划包括全部 3,430 条标签和六种未标注形态各 200 条，层内按 GPU 小时四分位等量抽样，种子 20261101。全部 4,630 条均找到监测对象。

| 下载层 | 作业数 | 文件数 | 十进制 GB | 3–1 MiB/s 理想下载小时 |
| --- | ---: | ---: | ---: | --- |
| 全部官方标签 | 3,430 | 10,201 | 68.7784 | 6.073 / 18.220 |
| 数组 | 200 | 200 | 3.3678 | 0.297 / 0.892 |
| 多节点或大 GPU | 200 | 1,085 | 29.8366 | 2.635 / 7.904 |
| 短作业 | 200 | 200 | 0.0538 | 0.005 / 0.014 |
| 交互候选 | 200 | 200 | 0.6166 | 0.054 / 0.163 |
| 长时 | 200 | 200 | 15.1448 | 1.337 / 4.012 |
| 其余 | 200 | 200 | 0.7096 | 0.063 / 0.188 |
| 合计 | 4,630 | 12,286 | 118.5077 | 10.465 / 31.394 |

合计 **118,507,676,143 字节**，超过 30 GB 预算，**本次没有新增下载原始 GPU 文件**。全部标签本身也超过预算。时间未计请求延迟和重试，详见 [gpu-telemetry-download-cost.csv](gpu-telemetry-download-cost.csv)。后续如需补齐显存占用，建议另行按作业/节点文件字节数封顶抽样，保留大规模作业层并使用抽样权重；当前没有执行额外下载。

既有节点小时聚合没有直接作业关联，但 **`batches/*/files.parquet`、`hourly.parquet`、`util_hist.parquet` 保留 job_id**。只读冻结 **2,391 个完整批次**的 manifest 清单与 SHA-256，重复 seq 取最新完整批次，仅使用 matched、eligible、within_allocation。去重 11,230 个源文件，其中 11,226 个符合条件，可提取 **10,691 个作业**的监测特征，OTHER 有任意观测 7,607 条。

按 GPU 时间加权合并有效时长、一阶和二阶矩，计算利用率/功率均值与标准差；标准差同时包含时间和设备间变化。低活动比例来自利用率不超过 5% 的有效时长。覆盖定义为有效 GPU 利用率小时/分配 GPU 小时；只有 **80%–105%** 参与活动分类。OTHER 合格 **7,221 条、168,182.47 GPU 小时**，占 OTHER **15.6421%**。105% 容纳已有边界计时误差，超出者不用于分类。

| 覆盖合格的监测集合 | 条数 | 每作业利用率均值 P50 | 每作业利用率标准差 P50 | 每作业功率均值 P50 |
| --- | ---: | ---: | ---: | ---: |
| 官方标签 DNN | 934 | 92.6939% | 13.7266 个百分点 | 182.78 W |
| 同用户强匹配主类 | 11 | 97.3934% | 11.6265 个百分点 | 150.26 W |
| 正例相似主类 | 4 | 89.7140% | 21.6613 个百分点 | 154.08 W |
| 交互式会话候选 | 39 | 17.3796% | 9.2443 个百分点 | 43.26 W |
| 低 GPU 活动主类 | 1,124 | 0.000541% | 0.0583 个百分点 | 38.02 W |

标签正例利用率均值 P05 为 **72.2679%**，低活动比例 P95 为 **0.118647**。合格监测的元数据相似候选若低于利用率下限或超过低活动比例上限，撤除相似用途标记并按其他形态归类：**94 条、725.94 GPU 小时**。官方标签优先；短时异常与实测低活动优先于用户传播主类。监测正例五折留出保留 **862/934=92.2912%**，每折只从参考部分设活动范围。[筛选记录](telemetry-similarity-screen.csv)、[监测留出](telemetry-positive-holdout.csv)、[分类监测检验](telemetry-category-validation.csv)提供细项。

低活动主类要求覆盖合格、平均利用率不超过 5%、至少 90% 观测时间利用率不超过 5%。其功率中位数 38.02 W，低活动时间比例中位数 99.9986%。持续繁忙定义为均值至少 50%、低活动比例不超过 20%，用于排除交互候选及终态检验，不直接作为 DNN 标签。

既有聚合保留 `utilization_memory_pct`（显存读写活动），**没有 `memory_used_MiB/memory_free_MiB`，无法计算显存容量占用**。没有为补齐此字段下载超预算数据。功率受 GPU 型号影响，未用统一瓦数阈值推断用途。快照以已有六月优先处理数据为主，并非随机样本；相似度主类仅 4 条监测合格，不能据此估计总体精确率或外推空闲比例。[提取摘要](existing-telemetry-extraction.csv)与[schema 检查](telemetry-inspection.csv)记录范围。没有打开流式数据库，没有改变进程、锁、STOP 文件或已有批次。

## 分类优先顺序与终态交叉

依次取第一个命中项，[rule-definitions.csv](rule-definitions.csv)保存条件：

1. 官方标签精确匹配。
2. 启动失败或调试：运行不超过 600 秒且 FAILED/NODE_FAIL/OUT_OF_MEMORY，或退出码/步骤退出码非零；CANCELLED、TIMEOUT 除外。实际 GPU 作业最短约 5 分钟，本类可见范围约 5–10 分钟，不能确认故障发生时刻或调试意图。
3. 合格监测的低 GPU 活动。
4. 同用户强匹配，然后同用户近似。
5. 正例相似且监测活动不冲突。
6. 数组成员：`id_array_task != 4294967294`。
7. 多节点或大 GPU：节点至少 2，或 GPU 至少 8。
8. 交互候选：单节点、GPU 不超过 2、非数组、等待不超过 2 秒、600 秒小于运行时长且小于 6 小时、有限时限大于零且不超过 720 分钟，排除监测持续繁忙。快速启动也可能是批处理，因此只标低置信候选。
9. 长时单任务：运行至少 6 小时，未被此前类别覆盖。
10. 短时或测试：运行不超过 600 秒，或 test 分区且不超过 1 小时。
11. 仍无法判断。

“高”表示可观测形态明确。长时作业仍可能包含训练、模拟、数据处理和未被监测识别的低活动分配。时长、分区和终态不单独证明训练用途。

终态按 [Slurm 20.11 头文件](https://github.com/SchedMD/slurm/blob/slurm-20-11-9-1/slurm/slurm.h.in)兼容读取 `state&0xff`，实际采集版本未公开。分别保留 COMPLETED、CANCELLED、TIMEOUT、FAILED、NODE_FAIL、OUT_OF_MEMORY，以及 1 条 `PENDING|REQUEUE_FLAG`；退出码、步骤退出码另存。

- [category-terminal-counts.csv](category-terminal-counts.csv)：十二类×终态的条数宽表。
- [category-terminal-gpu-hours.csv](category-terminal-gpu-hours.csv)：同一交叉的 GPU 小时宽表。
- [category-terminal-crosswalk.csv](category-terminal-crosswalk.csv)：交叉长表，含占比与分位数。
- [category-exit-crosswalk.csv](category-exit-crosswalk.csv)：类别、退出码、步骤退出码交叉。
- [other-by-state.csv](other-by-state.csv)、[退出码](other-by-exit_code.csv)、[步骤退出码](other-by-derived_ec.csv)、[分区](other-by-partition.csv)：边际汇总。

### 长时取消或超时能否解释为训练正常结束

以下统计运行至少 6 小时的全部 OTHER，不按终态排除其他形态。

| 长时终态 | 条数 | GPU 小时 | 占 MIT | 占 OTHER | 有限时限/接近时限条数 | 监测合格/繁忙/低活动条数 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| CANCELLED | 4,430 | 398,143.88 | 27.2666% | 37.0300% | 2,600 / 82 | 793 / 309 / 155 |
| TIMEOUT | 2,468 | 155,053.97 | 10.6187% | 14.4210% | 2,468 / 2,468 | 346 / 11 / 283 |

接近时限指运行/请求时限在 95%–105%。所有长时超时均接近时限，说明由时限终止；Slurm 没有收敛、训练完成或保存 checkpoint 信息。长时取消监测样本中 309/793 持续繁忙，与用户主动结束有效计算相容，另有 155 条长期低活动。长时超时样本中 **283/346=81.79%** 长期低活动，仅 **11/346=3.18%** 持续繁忙，每作业平均利用率中位数约 **0.000745%**、功率 **39.33 W**。现有证据不足以将长时取消或超时整体解释为训练正常结束，也不能外推到全部未监测作业。

官方 DNN 的长时取消/超时均为零，全部标签中有 6 条较短取消、10 条较短超时。长时取消有 33 条保留 DNN 推断证据。[long-terminal-analysis.csv](long-terminal-analysis.csv)还提供完成、失败、节点故障、内存不足的长短时对照。

## 柔性简表与 PR #31 的使用方式

原“用途未知 MIT OTHER 73.6%”可拆为主表十二行，保留证据范围。不能将资源小时直接当作可迁移电量，历史排队也不提供允许延后的期限。

| 可使用的类别 | 时间上能否迁移 | 能否中途拆分 |
| --- | --- | --- |
| 官方标签 DNN | 核实业务期限与依赖后，作为完整作业延后候选 | 须验证完整 checkpoint、恢复成功率及成本，当前无恢复实测 |
| 同用户强匹配 DNN | 核实用途和期限后整体延后 | checkpoint 未观测，基准暂不拆分 |
| 同用户近似 DNN | 人工确认用途后按完整作业延后 | 恢复能力未验证，基准暂不拆分 |
| 正例相似 DNN 候选 | 人工确认用途后按完整作业延后 | 恢复能力未验证，基准暂不拆分 |
| 长时单任务 | 核实用途及期限后，设置整体延后情境 | 无 checkpoint 证据，基准暂不拆分 |
| 数组批量 | 核实成员依赖、并发限制和期限后按成员改序 | 成员之间可条件性分别安排，成员内部未知 |
| 多节点大作业 | 完整资源同时分配的整体延后情境 | 跨节点状态与恢复未验证 |
| 交互会话候选 | 基准保持原执行时间 | 会话状态未知，暂不拆分 |
| 启动失败或调试候选 | 排查异常后重新安排 | 按完整短作业处理 |
| 监测低 GPU 活动 | 核实后台工作、资源释放权限后再判断 | 低 GPU 活动不证明可暂停恢复 |
| 短时或测试 | 用途确认后按完整作业改序 | 按完整短作业处理 |
| 仍无法判断 | 基准保持原执行时间 | 恢复能力未知，暂不拆分 |

本机结果供 [PR #31](https://github.com/TuzzyT-Laiii/33/pull/31) 后续按原执行区间拆分 OTHER 小时曲线，本 PR 没有修改 PR #31 或重算其输出：

- `D:/mit-supercloud-gpu-summary-20261028/derived/other-refine-20261101/other-job-categories.parquet`
- 同目录 `other-job-categories.csv`，76,404 条唯一作业，不含用户标识。

| 字段 | 含义及单位 |
| --- | --- |
| `id_job`, `time_submit`, `time_start`, `time_end` | 关联键与原始 UTC epoch 秒，保持执行区间 |
| `runtime_s`, `wait_s`, `gpus`, `gpu_hours` | 运行/排队秒数，分配 GPU 数与 GPU 小时 |
| `cpu_request`, `cpu_allocated`, `memory_request_mb`, `nodes_alloc`, `partition`, `is_array`, `timelimit` | 请求/分配资源；内存为 TRES 总内存 MB，时限分钟含特殊值 |
| `category_id`, `category`, `confidence`, `confidence_scope`, `classification_version` | 互斥主类、名称、置信档、证据范围、版本 |
| `model`, `model_family`, `dnn_evidence` | 官方模型/家族及独立用途证据等级 |
| `terminal_state`, `exit_code`, `derived_ec` | 终态与退出码交叉属性 |
| `has_prior_official_dnn_user`, `user_high_support`, `user_near_support` | 严格较早标签历史与前后 90 天匹配支持数；没有用户 ID |
| `similarity_distance`, `similarity_threshold`, `similarity_metadata_match`, `similarity_rejected_by_telemetry` | 五邻域距离、阈值、元数据匹配与监测冲突 |
| `telemetry_coverage`, `telemetry_qualified` | GPU 利用率有效时长覆盖和资格 |
| `utilization_gpu_pct_mean`, `utilization_gpu_pct_std`, `idle_fraction_le5pct` | GPU 时间加权均值、标准差（百分点）与低活动比例 |
| `utilization_memory_pct_mean`, `power_draw_W_mean`, `power_draw_W_std` | 显存读写活动百分比，功率均值/标准差 W；没有显存容量字段 |

小时曲线按 `id_job` 关联，用 `[time_start,time_end)` 与小时窗口的实际重叠秒数乘 GPU 数；不要将全程资源小时分给启动小时。分类版本为 `mit-other-refine-20261101-v2`。原始数据、私有监测特征、用户匹配及逐作业结果不进入公开仓库。

## 复现、内存与验证边界

依赖见 `../../analysis/requirements-mit-other-refinement.txt`，本次实际版本见 [analysis-environment.csv](analysis-environment.csv)。主分析按 20,000 行读取必要 Slurm 列，仅保留 GPU 特征，近邻分批查询，不加载原始 GPU 大文件。此次峰值 working set **398,598,144 字节，约 380.13 MiB**，运行 **11.55 秒**，见 [runtime-verification.csv](runtime-verification.csv)。监测提取一次只读取一个完整批次的必要列。

从 worktree 根目录运行：

```powershell
python -m pip install -r dc-flexibility-project/analysis/requirements-mit-other-refinement.txt
python -X utf8 dc-flexibility-project/analysis/download_mit_other_labels.py --reference D:\mit-supercloud-gpu-summary-20261028\reference
python -X utf8 dc-flexibility-project/analysis/inventory_mit_other_telemetry.py --root D:\mit-supercloud-gpu-summary-20261028 --output dc-flexibility-project/results/mit-other-breakdown-20261031
python -X utf8 dc-flexibility-project/analysis/extract_mit_other_existing_telemetry.py --root D:\mit-supercloud-gpu-summary-20261028 --output dc-flexibility-project/results/mit-other-breakdown-20261031
python -X utf8 dc-flexibility-project/analysis/analyze_mit_other_breakdown.py --root D:\mit-supercloud-gpu-summary-20261028 --baseline dc-flexibility-project/results/job-flexibility-constraints-20261030 --output dc-flexibility-project/results/mit-other-breakdown-20261031
python -X utf8 dc-flexibility-project/analysis/verify_mit_other_breakdown.py --reference D:\mit-supercloud-gpu-summary-20261028\reference --output dc-flexibility-project/results/mit-other-breakdown-20261031
```

清单脚本只做 LIST，复用本机缓存，不执行原始 GPU GET。提取器默认复用本机 manifest 快照并校验哈希，显式 `--refresh-snapshot` 才采集更多完整批次；刷新会改变覆盖，应重跑主分析并记录新分类版本。没有已有批次时无法复现本次监测增强结果，须保留该证据缺口。

真实数据验证包括：官方标签逐模型核对、PR #26 四类资源核对、十二类与终态交叉守恒、逐作业读回、独立标准 CSV 库扫描原始 Slurm，逐条比对 76,404 条 OTHER 的执行区间和 GPU 数。另检查留出集与参考集不交叉、下载成本超过预算、公开 CSV 不含逐作业/用户标识。[verification.csv](verification.csv)记录主分析检查。缺少真实负标签、代表性完整监测、业务期限、checkpoint 和恢复记录，不声称用途准确率、总体空闲率、可中断比例或实际电量柔性收益。
