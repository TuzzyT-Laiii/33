# MIT GPU OTHER 的官方模型标签与运行特征细分

分析日期：2026-10-31。依赖 [PR #26](https://github.com/TuzzyT-Laiii/33/pull/26)，输入代码基线为 `job-flexibility-constraints-20261030@70e468a`。审阅顺序为 PR #26，然后本次细分。原 PR 的报告与文件保留原样。

**官方标签确认了 3,430 条 OTHER 作业的 DNN 训练用途，共 40,071.8875 GPU 小时，占全部 MIT GPU 小时 2.7443%，覆盖 OTHER GPU 小时 3.7269%。其余 70.8895% 的 MIT GPU 小时仍缺少用途标签，可以继续按可核查的运行特征分类。** 以下类别区分“标签实测”和“规则推断”，分类占比均描述分配资源占用。

## 输入、下载与许可

本次使用本地 `D:/mit-supercloud-gpu-summary-20261028/reference/slurm-log.csv`，395,914 条唯一作业，98,176 条请求 GPU。请求筛选为 `tres_req` 中 1001、1002 的数量和大于零；资源数量使用 `tres_alloc` 中两者之和。未请求 GPU 的整节点预留不纳入 GPU 业务需求。GPU 运行区间全部有效。

运行时长为 `time_end-time_start`，排队时间为 `time_start-time_submit`，GPU 小时为分配 GPU 数乘运行秒数再除以 3600。包含取消、失败、超时与空闲分配。全部 MIT GPU 小时精确累计为 **1,460,190.6641666667**，与 PR #26 一致，展示值为用户指定的 **1,460,190.66**；OTHER 为 **1,075,193.2205555556**，占 **73.6337552993%**。百分比使用未提前舍入的分母。[四类 job_type 逐类核对](baseline-reconciliation.csv)的计数完全相同，资源小时差异只有浮点累计误差。

官方发布前缀为 `s3://mit-supercloud-dataset/datacenter-challenge/202201/`，HTTPS 前缀为 [MIT S3 202201](https://mit-supercloud-dataset.s3.us-west-2.amazonaws.com/datacenter-challenge/202201/README.md)。本机通过用户指定代理执行 HEAD 后再执行流式 GET；五个文件均小于 2,000,000,000 字节，未触发停止条件。两个标签文件已下载到上述 reference 目录；既有说明、许可和 TRES 映射与本次重新下载的字节完全相同，保持原文件不变。

| 官方文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| [labelled_jobids.csv](https://mit-supercloud-dataset.s3.us-west-2.amazonaws.com/datacenter-challenge/202201/labelled_jobids.csv) | 82,332 | `bd00b9e572442ed9ad900c8919de789e0a4bf225df8a657be9c7b389d7ca6d82` |
| [labelled_job_stats.csv](https://mit-supercloud-dataset.s3.us-west-2.amazonaws.com/datacenter-challenge/202201/labelled_job_stats.csv) | 339 | `3c23d07cc0f923a7ec0415f301d9243b83bf36fdb8b30a6774f9503ff3731903` |
| [README.md](https://mit-supercloud-dataset.s3.us-west-2.amazonaws.com/datacenter-challenge/202201/README.md) | 7,616 | `915b12d2110266fe4cad8a8d0f831f75d3ae08cf9ba42d106ac82a5a5f84c771` |
| [LICENSE](https://mit-supercloud-dataset.s3.us-west-2.amazonaws.com/datacenter-challenge/202201/LICENSE) | 285 | `a68e8239ee146c3e58c8988b66b176708310d2595095bf8d16ebb65da728dd0f` |
| [tres-mapping.txt](https://mit-supercloud-dataset.s3.us-west-2.amazonaws.com/datacenter-challenge/202201/tres-mapping.txt) | 111 | `4897c5cf1575664217f93d9c66cda3c06cae9d90f827a5dadc5483044dd64a51` |

来源 URL、ETag、Last-Modified、大小和哈希见 [download-manifest.csv](download-manifest.csv)。本机额外保存获取记录 `reference/other-label-acquisition-20261031.json`。全部输入的指纹见 [input-fingerprints.csv](input-fingerprints.csv)，包括 slurm-log SHA-256 `27259e5b8e6daecd80b7853126af40ba759bc25edc9053fc174ef7a4f186aa01`。分析前后七份输入的哈希和元数据相同，四份既有 reference 输入与 PR #26 的指纹相同。

S3 数据许可为 **CC-BY-NC-ND 4.0**。原始数据保留在本机 reference，仓库仅提供分析脚本、说明与汇总 CSV；未输出或提交逐作业结果。署名：Samsi et al., *The MIT Supercloud Dataset*, HPEC 2021, [DOI:10.1109/HPEC49654.2021.9622850](https://doi.org/10.1109/HPEC49654.2021.9622850)。[MIT 官方说明](https://github.com/MIT-AI-Accelerator/MIT-Supercloud-Dataset)描述模型训练来源，其中 Bert/DistilBert 为微调任务；本次不将模型家族推断为预训练、RL 或在线推理。

## 官方标签关联与模型分布

`labelled_jobids.csv` 实际包含 **3,430 行、3,430 个唯一 ID**，无缺失或重复。`labelled_job_stats.csv` 是 26 个模型变体的计数表，计数和为 **3,430**，逐模型与标签表一致；它没有逐作业 GPU 利用率或功率。202201 README 写有 3,425 个已知作业，该文字数字与下载文件实际条数有差异，本次使用文件实际记录。

| 标签关联去向 | 匹配条数 | GPU 小时 | 占 MIT GPU 小时 | 占 OTHER GPU 小时 |
| --- | ---: | ---: | ---: | ---: |
| OTHER | 3,430 | 40,071.8875 | 2.7443% | 3.7269% |
| LLSUB:BATCH | 0 | 0 | 0 | 0 |
| LLSUB:INTERACTIVE | 0 | 0 | 0 | 0 |
| LLMAPREDUCE:MAP | 0 | 0 | 0 | 0 |
| 无 GPU 请求作业 | 0 | 0 | 0 | 0 |
| 未匹配 | 0 | 不适用 | 不适用 | 不适用 |

标签 ID 匹配率为 **100%**，未匹配为零。[覆盖总表](coverage-totals.csv)、[按提交类型的覆盖表](label-coverage-by-job-type.csv)提供机器可读结果。未标注 OTHER 共 **72,974 条、1,035,121.3330555556 GPU 小时**。

下表均为 OTHER 内的**标签实测**，运行和排队单位为小时，`P50/P90` 按作业等权计算，GPU 数为分配数量。模型变体的归并规则是：`U*` 为 U-Net，`vgg*` 为 VGG，`resnet*` 为 ResNet，`inception*` 为 Inception；Bert 与 DistilBert 分开；`dimenet/schnet/conv/pna` 为 GNN，其中官方说明将 `conv` 对应 NNConv。标签来自已知训练活动，资源占用仍包含未成功退出的记录。

| 模型家族 | 条数 | GPU 小时 | 占 MIT | 占 OTHER | 运行 P50/P90 | 排队 P50/P90 | GPU 数 P50/P90 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| U-Net | 1,431 | 15,495.24 | 1.0612% | 1.4412% | 2.059 / 10.953 | 27.999 / 60.265 | 2 / 4 |
| VGG | 560 | 6,923.07 | 0.4741% | 0.6439% | 4.777 / 13.588 | 9.691 / 35.723 | 2 / 4 |
| Inception | 484 | 5,350.45 | 0.3664% | 0.4976% | 4.920 / 11.484 | 11.428 / 31.588 | 2 / 4 |
| Bert | 189 | 4,938.97 | 0.3382% | 0.4594% | 1.368 / 5.240 | 9.050 / 17.318 | 2 / 64 |
| ResNet | 463 | 3,713.62 | 0.2543% | 0.3454% | 2.823 / 8.332 | 15.851 / 38.268 | 2 / 4 |
| GNN | 131 | 3,203.93 | 0.2194% | 0.2980% | 1.664 / 11.159 | 0.004 / 0.727 | 2 / 256 |
| DistilBert | 172 | 446.60 | 0.0306% | 0.0415% | 0.752 / 3.050 | 4.711 / 12.969 | 2 / 2 |

[模型家族完整分位数](labelled-other-model-families.csv)及[26 个模型变体](labelled-other-model-variants.csv)还提供 P99、最大值、节点数和有限 timelimit 的分布。大 GPU 数直接来自 TRES 分配，Bert 最大 424、GNN 最大 416；保留原始分配，不将其等同于等量有效训练吞吐。

## 未标注 OTHER 的规则及互斥结果

先将官方标签作业单列，然后对未标注 OTHER 按以下顺序取**第一个命中的规则**。阈值是研究分类选择，保持透明，不作为官方用途标签。状态及时间限制沿用 [Slurm 20.11 官方头文件](https://github.com/SchedMD/slurm/blob/slurm-20-11-9-1/slurm/slurm.h.in)的兼容定义；实际采集版本未公开。

1. **失败、取消或异常退出**：`state & 0xff != 3`，或 `state & 0x400 != 0`，或 `exit_code != 0`，或 `derived_ec != 0`。`derived_ec` 表示步骤的最高退出码；因此总体 COMPLETED 也可能进入此类别。非零退出码仅标记异常，不推断失败原因。
2. **数组批量作业**：`id_array_task != 4294967294`。使用 Slurm `NO_VAL`，不根据匿名化后非零的 `id_array_job` 判断数组。
3. **多节点或大 GPU 作业**：`nodes_alloc >= 2` 或分配 `GPU >= 8`。
4. **短作业或测试分区作业**：运行 `<= 600 秒`，或 `partition == test`。测试分区可以包含数小时训练，名称保留这一差别；不自动判定为调试。
5. **疑似交互式会话**：单节点、`GPU <= 2`、排队 `<= 2 秒`、`600 < 运行 < 21600 秒`，且 `0 < 有限 timelimit <= 720 分钟`。这是弱证据候选；快速启动也可能是批处理。
6. **长时作业，训练候选**：运行 `>= 21600 秒`，即至少 6 小时。长运行还可能是模拟、数据处理或空闲会话，真实用途保持待核实。
7. **中等时长、用途待定**：剩余记录，运行在 10 分钟至 6 小时之间。

`timelimit` 单位为分钟；`4294967295` 为 INFINITE，`4294967294` 为 NO_VAL，两者均不作为有限时限。timelimit 描述最大允许运行时长，不提供到达后的业务完成期限。排队只统计启动和提交均为正且差值非负的记录；本次 GPU OTHER 的等待值全部有效。`partition=normal/gaia` 不额外赋予用途。

下面八个类别互斥，条数和 GPU 小时均守恒。前一行为**标签实测**，后七行为**规则推断**。运行与排队单位为小时。

| OTHER 细分类别 | 条数 | GPU 小时 | 占 MIT | 占 OTHER | 运行 P50/P90 | 排队 P50/P90 | GPU 数 P50/P90 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 官方标签 DNN 训练 | 3,430 | 40,071.89 | 2.7443% | 3.7269% | 2.732 / 10.858 | 14.624 / 47.842 | 2 / 4 |
| 失败、取消或异常退出 | 18,569 | 649,928.79 | 44.5099% | 60.4476% | 4.278 / 59.685 | 0.000833 / 6.009 | 1 / 2 |
| 数组批量作业 | 20,881 | 117,376.08 | 8.0384% | 10.9167% | 1.384 / 12.060 | 1.128 / 29.103 | 1 / 1 |
| 多节点或大 GPU 作业 | 531 | 22,924.22 | 1.5699% | 2.1321% | 1.890 / 21.374 | 8.021 / 61.590 | 4 / 8 |
| 短作业或测试分区作业 | 1,052 | 765.46 | 0.0524% | 0.0712% | 0.121 / 0.160 | 0.001667 / 2.748 | 1 / 2 |
| 疑似交互式会话 | 223 | 236.10 | 0.0162% | 0.0220% | 0.468 / 2.638 | 0.000278 / 0.000556 | 1 / 1 |
| 长时作业，训练候选 | 7,707 | 194,082.49 | 13.2916% | 18.0509% | 12.706 / 43.555 | 0.004444 / 21.127 | 1 / 2 |
| 中等时长、用途待定 | 24,011 | 49,808.20 | 3.4111% | 4.6325% | 1.205 / 4.224 | 1.333 / 12.077 | 1 / 2 |
| 合计 | 76,404 | 1,075,193.22 | 73.6338% | 100.0000% | — | — | — |

[other-breakdown.csv](other-breakdown.csv)保存未提前舍入的汇总精度、完整分位数和每类的柔性判断；[rule-definitions.csv](rule-definitions.csv)保存条件和优先顺序。小数占比逐项舍入后可能与合计相差末位。

第一条规则优先保留终态事实，因此长时和数组作业仍可能出现在失败类别。未标注 OTHER 的原始取消状态有 **10,997 条、410,841.04 GPU 小时**，占 MIT **28.1361%**；超时 **2,498 条、155,150.47 GPU 小时**，占 **10.6254%**。总体 COMPLETED 但退出码或步骤退出码异常的未标注记录有 **61 条**。这些占用都未从分母删除。[终态](other-by-state.csv)、[退出码](other-by-exit_code.csv)、[步骤退出码](other-by-derived_ec.csv)、[分区](other-by-partition.csv)保留对应汇总。

对上述 18,569 条失败、取消或异常记录进一步暂不使用终态规则，按其余同序规则划分，得到下面的内部结构；这些行是失败类别的子集，不能再与主表相加。

| 失败类别中的运行特征 | 条数 | GPU 小时 | 占失败类别 GPU 小时 |
| --- | ---: | ---: | ---: |
| 数组批量 | 5,190 | 94,319.77 | 14.5123% |
| 多节点或大 GPU | 650 | 142,816.19 | 21.9741% |
| 短作业或测试分区 | 1,217 | 265.07 | 0.0408% |
| 疑似交互式 | 486 | 1,188.76 | 0.1829% |
| 长时、训练候选 | 6,870 | 403,695.30 | 62.1138% |
| 中等时长、用途待定 | 4,156 | 7,643.69 | 1.1761% |

[失败类别的运行特征](non-success-workload-shapes.csv)提供分位数及两种总体分母占比。[允许重叠的特征表](rule-overlapping-features.csv)另外统计所有特征：未标注 OTHER 全部数组成员为 26,071 条、211,695.85 GPU 小时；运行至少 6 小时为 19,958 条、936,219.76 GPU 小时。重叠表不提供可直接相加的用途占比。

## 已知 DNN 对规则的检验

将完全相同的规则应用于全部 3,430 条已知 DNN，先不使用其模型标签选择类别，结果如下。分母分别为标签作业条数与标签 GPU 小时。

| 规则去向 | 标签作业条数 | 占标签条数 | 标签 GPU 小时 | 占标签 GPU 小时 |
| --- | ---: | ---: | ---: | ---: |
| 失败、取消或异常退出 | 42 | 1.2245% | 4,699.90 | 11.7287% |
| 数组批量 | 0 | 0 | 0 | 0 |
| 多节点或大 GPU | 1,029 | 30.0000% | 17,413.23 | 43.4550% |
| 短作业或测试分区 | 13 | 0.3790% | 93.96 | 0.2345% |
| 疑似交互式 | 1 | 0.0292% | 5.50 | 0.0137% |
| 长时、训练候选 | 637 | 18.5714% | 11,166.54 | 27.8663% |
| 中等时长、用途待定 | 1,708 | 49.7959% | 6,692.76 | 16.7019% |

“多节点或大 GPU”与“长时”两类合计容纳 **48.5714%** 的标签作业、**71.3212%** 的标签 GPU 小时；它们描述与已知训练样本相容的运行形态。单独使用最终长时类别只能容纳 18.5714% 的已知训练作业。忽略规则优先顺序时，直接运行至少 6 小时的标签作业有 793 条，占标签条数 23.1195%、标签 GPU 小时 48.0106%。

已知 DNN 中没有数组成员；有 1 条训练进入最终疑似交互式类别，另有 1 条先被异常规则捕获、同时满足交互式特征。13 条进入短作业或测试分区类别的标签记录说明测试分区也可运行训练。42 条异常中有 6 条取消、10 条超时及 26 条总体完成但步骤退出码非零。这些交叉结果显示用途、运行规模和终态需要分开解释。

本标签集只有已知 DNN 正例，没有经核实的非 DNN、交互式或调试对照样本。因此可报告已知训练的规则去向和容纳程度，**无法计算总体用途分类准确率、精确率或交互式识别率，也无法将长时规则的训练占比外推到未标注集合**。[规则检验表](label-rule-validation.csv)及[模型家族与规则交叉表](label-model-rule-crosswalk.csv)提供完整结果。

## 对任务柔性简表的修改含义

PR #26 中“用途未知 MIT OTHER 73.6%”可替换为主表八行，并在简表保留“证据”一列。原有 MIT `LLSUB:INTERACTIVE` 的 4.1828% 继续单列；其用途来自提交工具，本报告的疑似交互式候选不能代表 OTHER 内全部交互作业。

| 可替换的行与占 MIT GPU 小时 | 时间上能否迁移 | 能否中途拆分 |
| --- | --- | --- |
| 标签实测：DNN 训练 2.7443% | 有条件的完整作业延后候选；须核实业务完成期限与数据依赖 | 有条件；须验证完整 checkpoint、恢复成功率和成本。当前没有恢复实测 |
| 规则推断：失败、取消或异常 44.5099% | 用途及成功工作量未知；需审阅取消原因、故障和重试关系，基准保持既有执行 | 未知；终态和步骤错误不证明支持恢复 |
| 规则推断：数组批量 8.0384% | 成员可作为完整作业延后候选，先核实成员依赖、并发限制和期限 | 成员之间可在上述条件下分别安排；成员内部的时间拆分未知 |
| 规则推断：多节点或大 GPU 1.5699% | 完整资源同时分配的整体延后候选，须核实用途和期限 | 未知；跨节点状态保存、通信与恢复未观测 |
| 规则推断：短作业或测试分区 0.0524% | 用途审阅后按完整作业改序；保留交互需求 | 按完整作业运行；短时长本身不提供中断能力证据 |
| 规则推断：疑似交互式 0.0162% | 基准保持原启动时间；证据不足以指定延后窗口 | 基准不拆分；会话状态与恢复能力未知 |
| 规则推断：长时、训练候选 13.2916% | 用途和期限核验后的整体延后候选 | 只有在 checkpoint 和恢复成本验证后才可设置拆分情境 |
| 规则推断：中等时长用途待定 3.4111% | 基准保持既有执行，等待用途与期限核实 | 未知，基准不拆分 |

“有条件”描述下一步需核实的调度方式，不能把对应 GPU 小时占比直接设为可调电量比例。排队时间反映历史等待，不能作为用户允许延后的期限。数组成员之间的批量安排也不等于把单个训练过程按小时拆开。没有新增实测可中断比例；沿用 PR #26 的证据基准，未知控制权限保留为未验证。

## 利用率、功率与证据局限

只读检查了既有 `aggregates/` 与 `analysis/` 的汇总及 schema。两个 `device-hours.parquet` 为节点、GPU、小时粒度，包含功率有效时间、能量和覆盖率，没有 `id_job`，也没有逐作业利用率。`aggregates` 中的利用率分布和利用率—功率曲线为整体或分组汇总；无法直接关联到特定 OTHER 作业。本次未将它们用于交互式或训练分类。[检查记录](telemetry-inspection.csv)保留所检查文件的指纹和 schema。

较新的既有月度部分分析是 `analysis/priority-month-partial-20261031`，窗口为 2021 年 6 月，记录的监测覆盖率约 27.66%，且存在设备小时关联歧义。将整体低利用率或部分功率观测赋给某个作业会混入覆盖和设备关联的不确定性。本次没有打开 `state.sqlite`，没有触碰运行进程、锁或 STOP 文件，没有重新计算原始遥测。

官方标签来自有选择的已知模型训练活动，覆盖 OTHER GPU 小时仅 3.7269%，不构成 OTHER 的随机样本。模型家族不揭示具体训练阶段、checkpoint 或实际期限。终态是最终记录，1 条 requeue 标志也没有分段恢复证据。全部占比均为异构 GPU 的分配小时，不能等同于设备电量、成功计算工作量或可实现的柔性收益。

## 复现与验证

主脚本每次只读取 20,000 行和所需列，仅保留请求 GPU 的作业特征；未加载原始 GPU 大文件。此次 Windows 主分析进程峰值 working set 约 **209.40 MiB**，主分析运行约 **5.20 秒**，实测记录见 [runtime-verification.csv](runtime-verification.csv)。运行时和内存随环境变化，该文件记录本次执行。

依赖：Python、pandas、numpy、requests、pyarrow。下载器先 HEAD 所有目标文件，再按 1 MiB 块下载并核对字节数和 SHA-256。重跑下载器时会重新核验已有说明文件；内容不同会停止并保留原文件。

从本 worktree 根目录运行：

```powershell
python -X utf8 dc-flexibility-project/analysis/download_mit_other_labels.py --reference D:\mit-supercloud-gpu-summary-20261028\reference
python -X utf8 dc-flexibility-project/analysis/analyze_mit_other_breakdown.py --root D:\mit-supercloud-gpu-summary-20261028 --baseline dc-flexibility-project/results/job-flexibility-constraints-20261030 --output dc-flexibility-project/results/mit-other-breakdown-20261031
python -X utf8 dc-flexibility-project/analysis/verify_mit_other_breakdown.py --reference D:\mit-supercloud-gpu-summary-20261028\reference --output dc-flexibility-project/results/mit-other-breakdown-20261031
```

已完成真实输入运行、标签计数与官方统计表逐模型核对、PR #26 四类 GPU 作业计数和 GPU 小时核对、最终八类计数与 GPU 小时守恒、交付 CSV 读回、输入分析前后指纹核验，以及独立使用标准 CSV 库的逐行资源累计和关联复核。独立核验器同时检查交付 CSV 不含逐作业标识列。[verification.csv](verification.csv)记录主分析检查结果。
