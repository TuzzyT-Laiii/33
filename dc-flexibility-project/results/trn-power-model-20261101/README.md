# 大模型训练 GPU 功率与整套功率模型核查

输入版本：PR [#34](https://github.com/TuzzyT-Laiii/33/pull/34)，分支 `codex/load-curves-v3-20261101`，提交 `ab07a41157442e837659535e1a33579ced5af8fa`。本任务分支为 `codex/trn-power-model-20261101`。需求、天气、时间索引、服务器数量及安装参考 IT 容量沿用 v3；功率结果采用独立目录。年均、峰值均指按公开轨迹生成的年度情境结果。

TRN 已分配 GPU 的中央功率比例采用 **TDP 的 0.80**，低、高敏感性为 **0.65、0.95**。这是 H100 训练文献支持的情境参数；公开证据尚不能提供 Acme 预训练作业在整个分配生命周期内的 H100 平均功率实测。SAI 保持 v3 的 MIT 标定方法；INF 的需求及 GPU 功率方法沿用 v3。本次统一修改 CPU 响应曲线，因此同时更新 HPC、CLD、SAI、TRN、INF 的 IT 和设施功率。

## 数据与证据范围

Acme 官方 [GitHub 仓库](https://github.com/InternLM/AcmeTrace/tree/6d6ac9cd15f3b01151ac6ba00ecddd9156f0dc73) 的 README 明确指向 [Hugging Face 完整监测](https://huggingface.co/datasets/Qinghao/AcmeTrace)。本次取得两个集群的 `GPU_UTIL.csv`、`FB_USED.csv`、`POWER_USAGE.csv` 及三份 IPMI，共九份监测文件、21,703,979,345 字节；每份小于 10GB。下载记录逐项保留来源、字节数和 SHA-256，并核对 Hugging Face LFS 对象哈希。原始文件保留在服务器共享卷的 `data/raw/trn-power-model-20261101/`；Git 只保存统计和来源元数据。原件、官方处理代码及文献另有本地副本和哈希。

监测宽表的字段为 `Time` 和设备列；作业表有 `job_id`、资源数量、`type` 和时间，但缺少作业到节点/GPU 的分配映射。**各类作业的 GPU 利用率、显存、功率分布无法从公开表直接关联，保留 unknown。** 集群层面统计包括零值和空闲设备，均值按有效设备样本计权；空白不填零，时间间断不延续前值。设备列数量表示整个窗口出现的列，不能解释为同时在线的物理设备数量。各文件的时间范围、缺测、间隔、分位数精度和日内变化见 [Acme 统计](acme-statistics.md) 和 `acme/`。

Kalos 三个 GPU 指标覆盖 2023-08-15 至 2023-08-31，Seren 覆盖 2023-05-16 至 2023-08-18，均有时间间断。Kalos/Seren 的 GPU 利用率样本均值为 **62.01%/60.95%**，功率均值为 **191.37/195.49W**；超过 400W 的功率样本占 **12.53%/14.30%**。按 A100 400W 归一化的全部监测卡均值为 **0.478/0.489**，包含未分配设备，不能直接替代已分配 H100 预训练比例。发布表的设备列为 2,344/1,992，低于论文所述物理库存 2,416/2,288；当前统计不代表完整集群的纯实测总电量。官方论文与本次下载窗口不同，分布不要求完全一致。

公开作业表的实际行数为 Kalos 62,413、Seren 818,327，合计 880,740；唯一作业标识数量守恒，`type` 均无空值。官方 `cluster_summary.csv` 的 Seren 行数 1,031,550 与公开作业表版本不同，统计以下载原件为准。Kalos 预训练为 641 个作业，按 `start_time` 至 `end_time` 计算占 GPU 时间 **95.55%**；Seren 为 4,026 个、**69.49%**。全部 CPU、Debug、Eval、Other、Pretrain、SFT、MLLM 分类均列在 [分类统计](acme/job-type-summary.csv)。CPU 作业的原始 `gpu_time` 为空，不作为 GPU 作业功率样本。

Kalos 的原始 `duration` 有 62,409 行与 `end_time-start_time` 不同，另有三行缺少结束时间；Seren 两者一致。分类表分别保留官方 `gpu_time` 及时间戳计算的 GPU 小时，未将两种口径混用。本次功率重算继续使用 v3 已冻结的需求轨迹。

## GPU 功率参数

证据按与目标量的接近程度使用；完整来源、硬件、单位、计量边界及原件哈希分别见 [文献依据表](literature-evidence.csv) 和 [获取记录](literature-acquisition.json)。

| 依据 | 可核查结果 | 对 TRN 参数的用途 |
| --- | --- | --- |
| [Acme 官方监测与 NSDI 2024 论文](https://www.usenix.org/system/files/nsdi24-hu.pdf) | A100 混合开发负载的 GPU 利用率中位数约 97%/99%；低功率模式约 60W；大量样本接近或超过 400W | 直接匹配任务来源，硬件为 A100；监测缺少作业关联，不能直接标定 H100 已分配预训练功率 |
| [OLMo 环境影响研究，ICLR 2025](https://arxiv.org/html/2503.05804v1) | OLMo2 7B 的 H100 活跃训练每卡平均功率超过 600W；检查点阶段降到略高于 100W；按 700W 归一化约为 >0.857、0.143 | 最接近 H100 预训练的直接 GPU 功率依据；图中为一个节点最初 300 个记录步的序列，不能替代整个作业生命周期均值 |
| [MLPerf Training v4.0 GPT-3 功耗结果](https://smc.co/mlperfv40/results/gpt3) | 64 节点、512 H100，运行 56.86942 分钟；节点 DC 平均 6.747kW，分摊网络后 7.678kW/节点 | 页面测试说明称微调；用于整机数量级参照。浸没冷却、节点 DC 和分摊网络边界须区分，不能反推出预训练单卡功率比例 |
| [HGX H100 实测研究](https://arxiv.org/html/2412.08602v2) | 八卡节点空闲 1.86kW，Llama2-13B VQA 微调整机功率中位数 7.92kW、峰值 8.42kW，GPU 平均利用率 93% | 独立 PDU AC 整机参照；微调、AMD CPU 和其他配置与目标 DGX 不同 |
| [微软 ASPLOS 2024](https://www.microsoft.com/en-us/research/uploads/prod/2024/03/GPU_Power_ASPLOS_24.pdf) | A100 的 LLM 训练功率随前向、反向、梯度通信周期变化，训练与推理差异明显 | 支持按用途区分；硬件及短时微调实验限制其对 H100 生命周期均值的迁移 |
| [Llama 3 论文](https://arxiv.org/html/2407.21783v3) | 使用 700W H100；检查点、集合通信和大量 GPU 同步行为会引起大幅整群功率变化 | 700W 是配置 TDP；该描述未给出可直接采用的全程平均 GPU/TDP 比例 |

训练比例的解释采用 `r = (1-f) r_active + f r_wait`。例如 `r_active=0.86`、`r_wait=0.15` 时，`r=0.80` 对应约 8.5% 的等待占比；`r=0.65` 对应约 29.6%。这里的等待占比用于说明参数含义，**未作为观测事实**。0.95 表示更接近功率上限的训练实现，包含跨模型、并行策略和硬件配置差异；三档组成情境范围，不构成统计置信区间。

检查点写出时 GPU 可降低到低功率状态；评估可能运行生成计算，也可能等待 CPU 指标处理；同步等待会使计算间歇及整群同步降功率。GPU 利用率表示采样区间内是否存在内核活动，不等于 SM 活跃比例、MFU 或功率/TDP。暂不把这些阶段强行增加为每日固定时段；文献没有支持 Acme 存在某个固定小时的训练功率周期。

| 用途 | 已分配 GPU 功率/TDP | 未分配 GPU |
| --- | --- | --- |
| TRN | 低 0.65；中央 0.80；高 0.95 | H100 PCIe 中央 50W、SXM 中央 70W；独立空闲敏感性 |
| SAI | 保持 MIT 六月标定均值 97.674W/250W = 0.390697 | 保持 v3：V100 零利用率功率代理 41.572W/250W 迁移到目标 TDP |
| INF | 保持 v3 的 MIT 迁移参数 | 保持 v3 |

TRN 的 GPU 公式为 `N_GPU × [a × TDP × r_TRN + (1-a) × P_GPU_idle]`，其中 `a` 是已分配 GPU 比例。GPU 板卡功率已包含卡内显存等用电，不另加显存功率。B/C 使用 350W H100 PCIe，D 使用 700W H100 SXM；把 H100 SXM 的训练比例迁移到 PCIe 仍存在硬件差异。

## 基础、CPU、空闲及设施模型

`P_IT = P_base + P_CPU + P_GPU`，`P_facility = P_IT + P_cooling + P_other`。`P_base` 为内存、存储、网络、内部风机和 PSU 损耗等集总量；CPU、GPU 分项分别含自身空闲点。基础与空闲 IT 为同样服务器数量下所有任务需求为零时的 IT 功率，任务层只计算超过该点的增量。不能把整个 CPU/GPU 分项都称为任务增量。

| 类型 | P_base（W/台） | CPU 空闲（W/台） | GPU 空闲（W/台） | 无任务整机中央值（W） | 空闲敏感性低—高（W） |
| --- | --- | --- | --- | --- | --- |
| A | 150 | 40 | 0 | 190 | 120—320 |
| B | 200 | 40 | 50 | 290 | 195—500 |
| C | 250 | 40 | 100 | 390 | 270—680 |
| D | 1,800 | 100 | 560 | 2,460 | 1,370—4,110 |

该表适用于 HPC/CLD/TRN/INF。SAI 保持自己的 GPU 空闲代理，因此中央无任务整机为 B 298.20W、C 406.40W、D 2,831.53W。SAI 的空闲敏感性只变基础与 CPU 空闲，GPU 方法保持 v3。

三份 SPECpower 的整机 AC 空闲点为：单路 6448Y 76.2W、双路 6448Y 122W、双路同代 8490H 163W。A 型 190W 高于这些公开空闲点，但目标内存、存储、风机和 PSU 完整配置尚未确定；B/C 增加 GPU 与机箱用电的参数也缺少匹配实测。DGX H100 [官方规格](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html) 给出最大功率 10.2kW，未给出空闲功率。另一款 HGX H100 的 1.86kW 实测比模型 D 型 2.46kW 低约 24.4%；按实测为基准，模型高约 32.3%。该差异不足以直接替换目标 DGX 参数。

**本次保留基础及空闲中央值，明确列为未匹配配置的工程假设，并增加独立敏感性。** 未分配 H100 的 50/70W 没有本次取得的同卡型直接实测；Acme A100 约 60W 只能作跨代参照。SAI 的空载来自已分配且零利用率 V100 时段，迁移为未分配 H100 同样属于假设。P_base 固定值无法完整描述风机、PSU、内存和网络的负载依赖；这是整套模型仍需目标设备实测解决的主要不确定性。

CPU 响应使用以下三份整机测量，分别归一化为 `(P(u)-P_idle)/(P_full-P_idle)` 后等权平均，再拟合固定端点、单调三次 Bernstein 曲线。原始 33 个点见 [SPEC 数据](specpower-evidence.csv)，系数和拟合误差见 [cpu-shape.json](cpu-shape.json)。

- [Dell HS5610 单路 6448Y，01377](https://www.spec.org/power_ssj2008/results/res2024q1/power_ssj2008-20240227-01377.html)：空闲 76.2W，满载 318W。
- [Dell HS5610 双路 6448Y，01376](https://www.spec.org/power_ssj2008/results/res2024q1/power_ssj2008-20240227-01376.html)：空闲 122W，满载 602W。
- [Lenovo SR650 V3 双路 8490H，01282](https://www.spec.org/power_ssj2008/results/res2023q3/power_ssj2008-20230619-01282.html)：空闲 163W，满载 915W。

曲线为 `g(u)=3c1*u*(1-u)^2+3c2*u^2*(1-u)+u^3`，`c1=0.3110172232`、`c2=0.6476401659`。CPU 功率仍按原来的空闲和满载端点缩放，A/B/C 双路动态幅度 410W；50%→60% 增量从约 125.48W 变为 41.35W。三条结果涉及同型号或同代不同配置，并非独立硬件重复；SPEC 为 Java 负载的整机响应，迁移到 CPU 分项及 HPC/CLD 工作负载仍是代理假设。

纯 CPU 云计算 A-CLD 的峰值 IT 从 8.137MW 降至 7.158MW，年均从 5.639MW 变为 5.669MW；A-HPC 的年均从 8.384MW 降至 7.537MW，满载端点仍为 10.000MW。CPU 平滑改变不同利用率区间的功率，年均变化方向取决于需求分布。

冷却与设施模型保持 `cooling-20261028-v2`。抽查 [DOE 2024 官方指南](https://www.energy.gov/sites/default/files/2024-07/best-practice-guide-data-center-design_0.pdf) 支持按风机、泵、冷机、经济器、UPS 和配电分别计量，现代双变换 UPS 效率可达 95% 以上。中央送风 22°C、UPS 96%、配电 98%、参考 COP 6.5 的数量级可用于工程情境；具体压降、温差、部分负载惩罚和 COP 温度迁移未做目标设施标定。6.5 来自现有模型所列厂商额定点，未把单点当作完整性能实测曲线。内部 PSU 在 IT，设施 UPS/配电位于上游；IT 内部风机与机房风机分别计入。PUE 由年度设施电量/IT 电量得出，不再额外乘一次固定 PUE；1,100 个重算情境的年化 PUE 范围为 1.279—1.453。

## 重算结果、图件和不确定性

完成参考网格 2025 年 44 个情境，以及八城市 2023—2025 年 1,056 个情境，共 **1,100 个、9,644,448 小时**。全部需求输入 SHA-256 与 v3 清单核对；逐小时结果保留非负、分项及热平衡检查。八城市的大批小时文件留在服务器 `data/processed/trn-power-model-20261101/`，参考网格小时文件保留在本地及服务器 `results/trn-power-model-20261101/hourly/`。Git 只保存全部地点汇总、图件与审计元数据，逐小时文件已加入忽略规则。

[10MW 新旧完整对照](comparison.md) 同时给出 IT/设施年均、全年小时峰值、日内峰谷比、任务层占比和 PUE；[44 情境对照](v3-comparison.csv)、[1,100 情境对照](v3-comparison-all-locations.csv) 可直接复核。D-TRN-10MW 年均 IT **3.860→5.197MW**，年峰值 **4.433→6.811MW**，年均设施 **5.171→6.901MW**；任务增量占 IT **21.35%→41.59%**，基础与空闲功率仍为 **3.03564MW**。v3 的约 3.70MW 对应年度小时 IT 中位数，不能与年度均值混用。

D-TRN 的全年平均 GPU 分配比例仅约 43.1%。提高已分配卡功率后，典型日 IT 峰谷比从 **1.016→1.034**；当前源轨迹在 24 小时平均后依然较平坦，修正功率比例不会产生额外的日内需求周期。

[敏感性表](sensitivity.md) 显示 D-TRN-10MW 的 GPU 比例从 0.65 到 0.95 时，年均 IT 为 **4.750—5.644MW**；固定 GPU 比例 0.80、改变基础与空闲参数时为 **3.980—7.020MW**。空闲功率的不确定性仍大于当前 GPU 比例区间引起的变化。所有敏感性使用冻结的服务器数量；参考 IT 容量用于计数及冷却参数，未用来截断功率。其数值不代表设施经过验证的供电上限。

图件均提供 300dpi PNG 和嵌入中文字体的 PDF：

1. [TRN 典型日新旧对照](figure1-trn-typical-day.png)：B/C/D，分别展示 IT 和设施，纵轴从零开始。
2. [功率比例与整机依据](figure2-power-evidence.png)：GPU 功率/TDP 与整机 kW 分开显示，标明实测、情境、TDP 和计量边界。
3. [敏感性](figure3-sensitivity.png)：GPU 比例和基础空闲分别改变。
4. [CPU 曲线](figure4-cpu-response.png)：原始归一化曲线、单调平滑结果及双路动态增量。
5. [Acme 监测日内变化](figure5-acme-diurnal.png)：源数据北京时间小时的集群样本均值。

图件采用 Microsoft YaHei，已检查字符覆盖、文本边界、PDF 字体嵌入和可提取文本。遵照当前用户限制，未做视觉检查，验证范围见 `figure-qa.json`。

公开轨迹的有效窗口差异、年度循环合成、CPU/分配代理到实际功率的迁移、目标服务器完整配置缺失、未分配卡功率、跨代硬件和分布式训练策略，都会影响结果。当前参数与敏感性支持情境比较，尚不能称为目标数据中心实测校准模型。检查点频率、评估持续时间、同步效率以及逐作业 H100 遥测仍是缩小 TRN 区间所需证据。

## 复现及与其他 PR 的关系

脚本均在 `analysis/`，依赖见 `requirements-trn-power-model.txt`。文献原件、监测和作业原件不进入版本库。服务器运行将 `TMPDIR`、`NUMBA_CACHE_DIR`、`MPLCONFIGDIR` 设置到本任务共享卷临时目录；处理器每块 2,048 行，每个进程只处理一个指标表，最多三个指标进程并行。三进程峰值 RSS 分别约 360、360、374MiB，运行记录见 `acme/resource-*.txt`。未操作 MIT 流式处理进程及任何 STOP 文件。

执行顺序：

```text
python analysis/acquire_trn_power_evidence.py --directory <raw> --github-tree <github-tree.json> --proxy <proxy>
python analysis/acquire_trn_power_evidence.py --directory <raw> --hf-only --proxy <proxy> --direct-cdn
python analysis/prepare_trn_power_evidence.py --raw <raw>/literature --output results/trn-power-model-20261101
python analysis/summarize_acme_telemetry.py --raw <raw> --output results/trn-power-model-20261101/acme
python analysis/summarize_acme_telemetry.py --raw <raw> --output results/trn-power-model-20261101/acme --jobs-only
python analysis/build_trn_power_model.py
python analysis/recalculate_trn_cities.py --project <project> --source <shared-v3-root>
python analysis/recalculate_trn_cities.py --project <project> --source <shared-v3-root> --finalize-only
python analysis/verify_trn_power_model.py --project <project> --all-locations
python analysis/verify_acme_telemetry.py
python analysis/summarize_trn_results.py
python analysis/plot_trn_power_model.py
python analysis/finalize_trn_audit.py --raw <raw>
python analysis/publish_trn_power_model.py --source-project <project> --destination-root <shared-root>
```

参考结果在本地和服务器的 Parquet 序列化环境不同，最终采用服务器城市重算产物作为统一的小时文件，`--finalize-only` 更新其实际文件哈希；数值、冻结输入与年度汇总均另行验证。`all-locations-manifest.json` 保存 1,100 份小时结果哈希，`manifest.json` 保存输入和方法，验证记录为 `validation.json`、`validation-all-locations.json`。

本 PR 的基分支为 #34 分支，审阅顺序为 **#34 → 本 PR**。推理日内起伏由 [#35](https://github.com/TuzzyT-Laiii/33/pull/35) 处理，该 PR 另依赖 #30；满足各自前置依赖后，#35 与本 PR 可并行审阅，两项工作互不依赖。本次 INF 的变化完全来自 CPU 平滑，需求和 GPU 分项逐小时保持 v3。组合两项工作时，应保留 #35 的 INF 需求及 GPU 方法，应用本 PR 的 CPU 平滑曲线并重算组合结果，避免用本任务的冻结 INF 小时表覆盖 #35 输出。INF 的 GPU 功率后续应使用推理模型、批量、吞吐和 SLO 对应的测量；本 PR 保留其现有参数。

本次未合并 PR，未启用自动合并。
