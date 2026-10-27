# CPU GPU 任务记录的时间粒度与柔性约束

任务记录可以提供提交、执行和资源分配信息，用于计算 CPU/GPU 任务需求及小时负荷。真实柔性范围还取决于完成期限、暂停权限和服务质量。任务类型与优先级保留为筛选依据，缺失的业务约束保持未知。

[任务来源与柔性文献历史报告](../results/task-driven-load-report-20261026-v2/README.md)保存来源波动、功率示例和柔性文献对照；[初始报告](../results/task-driven-load-report-20261026/README.md)保留。

[单机 CPU/GPU 数量情境与来源说明](../results/server-count-scenarios-20261027/README.md)定义 2∶1、2∶2、2∶8 三种厂商配置实例，并说明联合 CPU/GPU 需求、ASI 作用及其他公开来源。[年度构造工作包](../results/scenario-load-methods-20261026-v2/README.md)保存既有 16 条全年参考路径和 490 组相位检查。

ASI 提供生产 AI 小时业务类型、优先级、GPU 归属资源与库存，可帮助构造生产需求和筛选柔性候选；CPU 物理插槽数量及业务截止权限仍需其他依据。Lisa、Acme、ASI 是工作区当前已详细核验主样本。Microsoft Philly 的训练任务与同服务器 CPU/GPU 分钟监测、Alibaba CPU 2018 的在线/批处理混部、Google 2019 PowerData 的关联电力域功率，也分别适合对应研究问题；本轮只核对这三项的官方说明，未下载计算。

## 数据与适用时间尺度

| 数据 | 原始粒度及覆盖 | 可计算内容 | 柔性约束 |
| --- | --- | --- | --- |
| SURF Lisa | 作业时间记录到秒；官方节点监测每 30 秒；作业约 10 个月、监测约 5 个月 | CPU/GPU 分区作业、节点和核心数量；监测数据可核对实际功率 | 作业文件没有业务截止时间和抢占权限，分区也没有给出训练/推理标签 |
| AcmeTrace Kalos | 提交、启动、结束时间记录到秒；本轮文件覆盖 2023 年 5 月 17 日至 8 月 16 日 | 独立 CPU 作业和 GPU 作业的运行区间、分配数量、排队、执行状态，以及 GPU 的预训练、微调、评估等标签 | 没有业务截止时间、检查点和抢占权限；CPU 作业标签只表示 CPU |
| Alibaba GPU 2026 ASI | 逐 pod、逐服务器、逐小时；相对日 0 至 184，共 4440 小时 | 训练、在线推理、离线推理及开发任务的小时申请量、归属 GPU 小时、库存和优先级 | 没有业务截止时间及明确的逐任务抢占权限；小时数据无法恢复小时内部的执行顺序 |
| Alibaba CPU 2018 | 任务与实例的启动、结束时间记录到秒，另有机器和容器监测；约 8 天 | 在线服务与批处理分别统计，保留部分任务依赖 | 公开批处理表没有完整的独立提交时间字段和业务期限，不能从执行时刻恢复真实到达窗口 |
| PWA 标准作业记录 | 各作业的提交、等待、运行时长通常以秒表示；覆盖期依具体日志 | 处理器数量、队列、运行区间；部分日志提供分段执行记录 | 要逐日志读取队列和抢占说明；Requested Time 是运行时间请求或上限，不能直接解释为完成期限 |

Acme 的 `Eval` 保持“评估任务”的含义，不能直接改成在线推理。ASI 的 `LP` 保持低优先级/spot 的来源含义，不能直接换算成允许延后若干小时。两套数据的任务标识互不关联。

CPU 超算的主要任务来源可采用 [SURF Lisa 作业与监测数据](https://zenodo.org/records/13685426)。本轮读取并通过官方 MD5 核对了全部 Slurm 作业文件和节点硬件表。30 秒节点监测的粒度由官方说明及整合代码确认，完整监测尚未进入本次功率标定。任务调度与监测覆盖期不同，联合分析需要使用实际重叠区间。

Lisa 原始时间没有携带时区。当前输出保留源时间及相对小时，不直接与全球 UTC 天气连接；来源时区需要核实。GPU 分区表示分区类别，不能据此确定每个任务实际使用的卡数或业务类型。

## 真实任务记录核对

Lisa 文件包含 1596963 条唯一作业，其中 CPU 分区 1395625 条，GPU 分区 201338 条。源时间覆盖 2021 年 12 月 25 日提交记录至 2022 年 11 月 1 日结束记录。CPU 分区运行时长中位数为 25 秒，90% 分位为 3280 秒；GPU 分区中位数为 274 秒，90% 分位为 18017.3 秒。作业条数分布与资源小时分布分别统计。

按作业实际执行区间和分配核心数量累计，Lisa CPU 分区约有 2604.02 万核心小时，GPU 分区约有 384.48 万宿主核心小时。GPU 分区的核心小时不能改成 GPU 卡小时。[audit_lisa_tasks.py](../analysis/audit_lisa_tasks.py)已输出任务表、5 分钟及小时核心需求，核对原始总量及两种粒度间的守恒。

本轮读取 AcmeTrace 官方 `trace_kalos.csv`，共 62413 条作业：42506 条独立 CPU 作业、19907 条 GPU 作业。GPU 标签包括 18471 条评估、641 条预训练、537 条开发调试、250 条其他和 8 条微调。上述数量是任务条数，不能作为各类型的耗电比例。

独立 CPU 作业运行时长中位数为 8 秒，97.30% 少于 60 秒，99.98% 少于一小时。GPU 有效执行区间的运行时长中位数为 48 秒，其中 98.18% 少于一小时；该分布受到大量评估任务影响。GPU 有 3 条记录没有结束时间，保留在任务表中，并在执行区间累计时明确排除。零时长区间保留零资源小时。

这个 CPU 样本来自 LLM 开发集群，标签没有给出 CPU 作业的业务用途，不能据此规定全部 CPU 超算的业务结构或柔性比例。

在这批文件中，62410 条具有完整结束时间的作业，其原始 `duration` 等于提交至结束的总时长。运行负荷统一使用 `end_time-start_time`，排队使用 `start_time-submit_time`。原始 `duration`、`queue` 和 `gpu_time` 字段保留供核对，运行资源小时由实际执行区间重新累计。

## 从秒级作业累计资源需求

对作业 i，保留实际启动时刻 s_i、结束时刻 e_i 和分配资源数量 n_i。时间段 h 内的资源小时为：

\[
W_h=\sum_i n_i\frac{|[s_i,e_i)\cap[h,h+\Delta t)|}{3600},
\]

其中交集长度以秒计算。小时平均分配数量为 W_h/Δt_h，Δt_h 以小时表示。一个占用 100 个核心、运行 30 秒的作业贡献 0.8333 核心小时。

[task_trace_inputs.py](../analysis/task_trace_inputs.py) 使用启动和结束事件积分，同时输出 5 分钟与小时资源需求，保留作业原始时间。CPU 独立需求只统计 `gpu_num=0` 的作业；GPU 作业的宿主 CPU 随 GPU 整机处理，不能再作为另一套独立 CPU 中心重复累计。

ASI 使用同小时服务器库存作为 GPU 数量分母，分开输出申请份额和归属 GPU 小时份额。`avg_gpu_sm_util` 保留原始字段；真实样本存在大于 100 的记录及缺失值，字段口径尚不足以把全部记录直接解释为单卡 0 至 100% 利用率。本接口不对它截断或补零。

本轮读取 ASI 相对日 109 的完整 24 小时，共 36264315 条 pod 小时观测；筛选申请 GPU 大于零的记录后，得到 2816548 条 GPU pod 小时观测。这些数量按小时累计，同一 pod 可以重复出现。样本中申请量/库存为 104.63% 至 116.62%，不能直接将该比例作为物理利用率。固定路径功率接口使用另行保留的归属 GPU 小时份额。

## 柔性任务的识别

任务表保留 `task_id`、任务类别、资源数量、提交和执行时间，以及 `deadline_utc`、`preemption_allowed`、`flexibility_status`。这两套公开记录没有提供的期限和权限为空，`flexibility_status=unknown`。

训练、离线推理、微调和评估可进入服务约束审阅候选。CPU 标签、GPU 类别、LP 优先级、较低使用率或历史排队，均不能单独证明任务允许延期或中断。任务被取消或执行失败，也不能单独证明它支持保存状态后继续。

| 调节方式 | 必需约束 | 对时间粒度的要求 |
| --- | --- | --- |
| 延后启动并连续执行 | 数据就绪时间、最晚完成时间、运行时长和并行资源 | 保留秒级作业；调度后再汇总小时负荷 |
| 暂停后恢复 | 明确暂停权限、检查点间隔、恢复时间及额外工作量 | ASI 小时记录不能验证小时内的暂停过程 |
| 减少资源或降低功率 | 任务性能与资源/功率关系、服务时延或完成期限 | 需要对应设备及任务的性能和功率记录 |

小时能源模型可读取累计后的功率，但短任务仍须遵守原始执行时长和不可中断等约束。对没有业务期限的数据，允许延后 1、6、24 小时等数值只能作为明确标注的研究情境。若采用“不晚于历史完成时间”的服务基准，也须注明该基准是研究定义，并检查数据就绪与依赖关系。

## 接入 IT 散热与气象模块

`resource_profile(demand, capacity, hardware, source, capacity_basis)` 将资源需求除以明确的参考资源容量，构造现有 `ResourceProfile`。容量必须为正且有依据；需求超过容量会报错，不会自动截断。

这里的资源量是分配或归属资源的代理。映射到 SPEC CPU 或 H100 整机功率时，保留跨数据集映射的含义。CPU/GPU 硬件、任务性能、机内组件和外部冷却边界仍遵循[通用负荷模块说明](compute-load-module.md)。先得到 IT 功率与散热，再结合当地温湿度计算冷却及设施用电。

本次 CPU 功率接口检查以全部有效 CPU 作业的观测并发峰值作为规模参考，验证窗口选择 CPU 资源小时最高的日期。这个分母用于参考规模，不能解释为原集群安装容量。GPU 使用同小时库存，异构卡数也不能直接解释为相同数量 H100 的吞吐能力。资源相对小时与已有真实天气的配对属于接口验证情境。

年度入口 [build_task_annual_loads.py](../analysis/build_task_annual_loads.py)读取已核验的需求表，按完整连续观测窗口重放到目标日历年，并通过现有全球网格气象适配器计算冷却。已处理 Lisa CPU 309 日、Acme CPU/GPU 90 日和 ASI GPU 1 日；2025 年四档规模共 16 条、140160 条小时记录。ASI 单日仅演示，Acme CPU 作为开发样本诊断；所有结果保留来源、分母、相位及天气外推标记。

每小时记录的源索引由 [annual_task_profiles.py](../analysis/annual_task_profiles.py) 保存。基准采用 j(h)=h mod L，相位敏感性检查全部完整日偏移；不平滑、不调整均值或添加未知季节需求。闰年采用 8784 小时。原有资源曲线库保留独立输入版本；任务级调度仍需真实业务约束及跨年执行状态，不能从小时重放推定权限。

## 复现与输出

依赖见 [task-trace-requirements.txt](../analysis/task-trace-requirements.txt)。在项目根目录执行：

```bash
python -m pip install -r analysis/task-trace-requirements.txt
python analysis/fetch_task_trace_samples.py --output-dir data/raw/task-trace-audit-20261026 --days 109
python analysis/audit_task_trace_sources.py --raw-dir data/raw/task-trace-audit-20261026 --output-dir data/processed/task-trace-audit-20261026
python analysis/verify_task_trace_inputs.py --raw-dir data/raw/task-trace-audit-20261026 --output data/processed/task-trace-audit-20261026/independent-checks.json
```

Lisa 的输入为官方 `slurm_table_cleaned.zip` 和 `node_hardware_info.parquet`，从上述 Zenodo 页面下载后运行：

```bash
python analysis/audit_lisa_tasks.py --jobs-zip data/raw/task-trace-audit-20261026/lisa/slurm_table_cleaned.zip --hardware-parquet data/raw/task-trace-audit-20261026/lisa/node_hardware_info.parquet --output-dir data/processed/task-trace-audit-20261026/lisa
```

审核程序另支持 `--weather-parquet`，使用已有真实天气与两条资源路径检查 CPU/GPU、1/10/100/500 MW 的功率接口。这里计算固定路径，未赋予缺失的调度权限。

Acme 和 ASI 已完成上述资源累计与 8 条各 24 小时的功率接口检查，独立逐作业交集计算核对 48 个时间段，另通过 5 项无效输入检查。Lisa 通过官方文件校验、标识唯一性、运行时间与等待时间非负、核心小时累计及 5 分钟到小时的守恒检查。汇总记录见 [任务来源核验目录](../data/metadata/task-trace-audit-20261026)。

输出包括任务表、5 分钟及小时资源量、ASI 类型与优先级统计、逐小时质量记录，以及输入输出校验值。原始数据与带任务标识的派生表保留在研究数据目录，公开仓库保存代码、文档和汇总验证记录。

服务器目录为 `/data3/dataset/ljt/global-grid-flexibility-20261023`。采集按远程 ZIP 成员读取，不要求下载全部 ASI 压缩包；每个样本都登记相对日、小时、官方 URL、成员 CRC 和本地列选取文件的 SHA-256。

## 来源

- [AcmeTrace 官方作业和字段说明](https://github.com/InternLM/AcmeTrace)
- [Alibaba GPU 2026 官方字段说明](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/docs/schema.md)及[下载分片说明](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/docs/data_download.md)
- [Alibaba CPU 2018 官方字段](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/schema.txt)及[任务依赖说明](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md)
- [PWA Standard Workload Format](https://www.cs.huji.ac.il/labs/parallel/workload/swf.html)
- [SURF Lisa 作业及 30 秒监测](https://zenodo.org/records/13685426)
