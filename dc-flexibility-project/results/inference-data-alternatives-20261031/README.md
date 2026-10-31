# 推理服务需求数据：185 天 GPU 资源主数据与数月请求对照

核验日期：2026-10-31。适用情境为 **CPU 主导型·推理服务（B-INF）、均衡型·推理服务（C-INF）、GPU 主导型·推理服务（D-INF）**，规模分别为 1MW、10MW、100MW、500MW，例如 `D-INF-100MW`。命名依据[情境命名 PR #27](https://github.com/TuzzyT-Laiii/33/pull/27)及本次用户要求；本报告采用 A–D 四类服务器的范围。

## 1. 数据选择与可回答的问题

**现有 INF 情境的主数据推荐扩展为 ASI 185 天，仅选择 `online_inference` 与 `offline_inference`，分别保留优先级；请求侧对照采用 BurstGPT 前两份文件，Azure 2024 用于另一类生产 LLM 服务的七日对照。** ASI 可直接累计归属 GPU 小时；全类型聚合用于资源核验，训练、开发及未知类别单独保存。BurstGPT 已在服务器完成 121 个相对日期的文件连接和 119 个完整日的小时汇总。Azure 两份文件的首尾记录分别覆盖 2024-05-10–16 和 05-12–18，各约七日。

可获得多天、周内乃至数月的逐小时曲线。ASI 有 4,440 个连续小时分区；BurstGPT 前两份文件有约四个月的请求时间轴；Chutes 另提供一整年的请求级记录。ASI 和 BurstGPT 均缺少可核验的原始日历起点，支持跨日、相对七日和长期变化研究；年内变化、业务增长、硬件扩容与季节效应仍须区分。仅有六个月或四个月记录，无法识别完整年度季节规律；一整年数据也需要控制平台增长和模型更替。

采用 ASI 的理由是其资源字段和业务分类更接近本项目现有输入。其推理类别涵盖多种模型；若研究范围进一步限定为纯 LLM 在线推理，则 BurstGPT / Azure / Chutes 的业务含义更直接，但必须增加 token 到指定硬件 GPU 小时的标定。ASI 的 `model_type_public=genai` 还可能包括图像生成，不能直接等同为纯 LLM。不同业务边界分别说明。

本次完成调研与可行性核验，保留[既有单日演示](../scenario-load-methods-20261026-v2/README.md)及原始输入版本。没有更新年度情境曲线或计算柔性收益。

## 2. 检索与证据范围

检索覆盖厂商官方仓库、作者或机构数据仓库、原论文及数据卡。查询主题包括 Azure LLM / Splitwise / DynamoLLM、BurstGPT、Mooncake、Alibaba ASI / ServeGen / PAI / MaaS / GenAI / microservices、Google cluster/power、Meta distributed traces、公开长期 LLM serving / agent traces。下表列出与需求曲线、资源换算或直接对照有关的发布物。

官方说明正文、字段、下载说明与许可已读取；来源 URL、正文 SHA-256、GitHub 树版本、资产大小见[来源审计](../../data/metadata/inference-alternatives-20261031/source-audit.json)及[补充元数据](additional-source-audit.json)。审计 JSON 保留核验时版本。未读取或下载全量记录的项目，表内明确限定证据范围。

### 2.1 时间、粒度、GPU 与业务

“可换算”表示可以在明确假设下建立代理，不表示已经取得物理 GPU 忙碌时间或实测功率。年份分别注明采集或发布年份；无法确认原始采集日历时保留未知。

| 数据集与官方入口 | 年份与时长 | 时间分辨率、粒度 | GPU 用量 / 功率 | 小时 GPU 需求与业务边界 |
| --- | --- | --- | --- | --- |
| [Alibaba ASI](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2026) | 2026 发布；相对日 0–184，185 天；原始日历不公开 | pod×小时、服务器×小时；另有作业汇总 | 有申请 GPU、归属 GPU 小时、SM 指标；无功率 | 可直接累计资源小时。在线/离线推理、训练、开发等生产 AI 混合业务；异构 GPU 需另行换算性能 |
| [BurstGPT](https://github.com/HPMLL/BurstGPT) | 2024 预印本 / KDD 2025；前两文件 121 天，第三文件另 110 天；所读发布说明未给出可核验日历起点 | 请求；相对秒；输入/输出 token，模型与 API/对话类别 | 无 GPU、无功率；本次基础 CSV 也没有新增延迟列 | 可按小时累计请求和 token；GPU 小时需标定。Azure 承载的 GPT-3.5/GPT-4 服务 |
| [Azure 2023 / Splitwise](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md) | 2023-11-11 单日；ISCA 2024 | 请求时间戳、上下文与生成 token | 无 GPU、无功率 | 需性能标定；代码补全与对话。Splitwise 对应这一发布物，不单独重复计为另一数据集 |
| [Azure 2024 / DynamoLLM](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md) | 2024 采集；HPCA 2025；两个文件各约七日，具体边界见第 4 节 | 请求时间戳带 UTC 偏移、输入/输出 token | 无 GPU、无功率；论文的能耗实验不能自动附到每条请求 | 支持小时、跨日、七日对照，需性能标定；代码补全与对话 |
| [Azure LMM 2025 / ModServe](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLMMInferenceDataset2025.md) | 2025 发布；采集说明为 2024-10-15–22 的一周样本 | 请求、图片数、上下文和生成 token | 无 GPU、无功率 | 多模态对照；另需视觉编码器成本，图片 token 不能与文本按统一系数直接相加 |
| [Azure Functions 2019](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md) | 2019-07，14 天 | 每函数每分钟调用数；每日执行时间与内存分布 | 无 GPU、无功率 | 可作服务到达形状敏感性对照；函数业务没有推理标签，不能据此恢复真实 LLM 需求 |
| [Azure Functions 2021](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsInvocationTrace2021.md) | 2021-01-31 起两周；时间已修改 | 函数调用结束秒与时长，可推算开始时刻 | 无 GPU、无功率 | 可聚合小时调用或函数执行秒；需业务映射后才能作为推理代理 |
| [Azure Functions Blob 2020](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsBlobDataset2020.md) | 2020-11-23–12-06 | 毫秒级存储访问、字节、读写、调用标识 | 无 GPU、无功率 | 存储侧辅助证据；存储访问数量没有直接的 GPU 服务量换算关系 |
| [Mooncake / Kimi](https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release) | 2024 技术报告 / FAST 2025 | 当前真实对话、工具/agent 各采样一小时；请求到达毫秒、输入/输出长度、前缀块哈希 | 无 GPU、无功率 | 可校验 KV cache 与短时请求特征；合成文件含 Poisson 到达；公开真实样本无法提供跨日曲线 |
| [ServeGen 在线](https://github.com/alibaba/ServeGen) | NSDI 2026；论文基于长期生产分析；核验的公开 m-large 客户窗口为 14 天 | 客户×600 秒的速率、CV、到达分布，以及 token 分布；另有选择性会话样本 | 无 GPU、无功率 | 可用速率×窗口时长构造到达量，并匹配对应 token 分布；生成结果须标为模型生成。`chunk-i` 是客户编号，不可顺次连接成更长时间轴 |
| [ServeGen 离线 / ACDC](https://github.com/alibaba/ServeGen/tree/main/data/offline) | SOSP 2026；六个完整批任务 | 每请求输入长度/哈希，或输入输出长度与执行毫秒 | 无 GPU、无功率 | 翻译、生成、分类、图片描述等批推理；没有请求到达时间，不能自行恢复跨日批任务到达曲线 |
| [Alibaba xMaaS 2026](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-maas) | 日 0–181，缺日 172；五分钟及应用案例仅日 140–146 | 六个月每日实例片段；一周实例五分钟均值；少量应用小时指标 | 有分配 GPU、GPU 小时、SM/显存、CPU；无功率 | 一周可聚合小时资源；六个月日表缺小时内位置，不能拆成六个月真实小时曲线。在线/离线 LLM，含 prefill/decode/fused 角色 |
| [Alibaba PAI GPU 2020](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020) | 2020-07–08，两个月；NSDI 2022 | 作业/任务/实例的秒级启动结束，资源监测与语义表 | 有 GPU 申请及使用指标；无功率 | 可积分实例占用并核对监测；训练与推理混合，语义标签覆盖有限，不能将 evaluator 或全部任务当作在线推理 |
| [Alibaba GenTD26](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI) | 2026 发布；论文含 2025/2026；所读说明未承诺多月连续窗口 | 生成请求、QPS、容器 GPU duty cycle、显存与时延；时间平移、部分指标缩放 | 有 GPU 活动指标；无功率 | Stable Diffusion / LoRA 图像推理；需核对实际文件覆盖、采样间隔和缩放口径，不能由说明推断全年规模 |
| [Alibaba 微服务 2021](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2021) | 2021，12 小时 | 节点 30 秒、容器 60 秒、调用指标与抽样调用图 | 仅 CPU/内存等，无 GPU/功率 | 通用服务对照；缺少推理业务标签与跨日覆盖 |
| [Google Borg / PowerData 2019](https://github.com/google/cluster-data) | 2019-05，八个 cell；57 个电力域 | 资源记录含五分钟 CPU 分布；另有电力域监测 | 有电力域功率；缺少可关联的推理 GPU 需求 | 可用于资源与功率边界比较；电力域功率不能直接分配给推理。没有公开的逐推理请求标签 |
| [Meta distributed_traces](https://github.com/facebookresearch/distributed_traces) | ATC 2023 发布；公开的是微服务统计表，所读 README 未定义推理连续日窗口 | 服务拓扑历史、请求工作流与结构统计 | 无可对应的 GPU/功率 | 可研究服务结构；不能从汇总拓扑恢复多月推理需求曲线 |
| [Harvard / Chutes](https://huggingface.co/datasets/harvardMadsys/chutes_metadata) | 2026 发布；作者称 2025-04–2026-04 一年；发布时钟归一化 | 请求级开始/完成、模型、实例、输入/输出/缓存 token、TTFT | 无 GPU 时间和功率；实例标识不等于卡数 | 可生成一年逐小时请求/token 曲线；缓存 token 仅最后约十周提供。多模型生产 LLM，有平台增长与模型更替 |
| [FreeInference Agentic](https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace) | 2026-05-17–09-05，16 周 | 请求/会话毫秒、token、缓存、工具调用与延迟 | 无直接 GPU/功率 | 可形成数月 agent 请求负荷；工具等待不能全部计入 GPU 活跃时间；限所观测用户/平台 |
| [UW TraceLab](https://github.com/uw-syfi/TraceLab) | 2026 发布；README 的 v0.0.1 统计与最新 v0.0.2 区分保存；连续覆盖未核验 | 开发者 Claude/Codex 会话、LLM 调用、缓存及工具记录 | 无 GPU、无功率 | 可用于编码 agent 需求组成；需先核对连续覆盖和采集选择，不能直接当作机房全年总需求 |

### 2.2 大小、许可与获取

字节数为本次官方资产 / GitHub 树 / 数据卡核验值；近似值沿用官方说明。未明确授予统一许可时不套用其他子目录许可。

| 数据集 | 大小 | 许可 | 获取方式与本次处理 |
| --- | --- | --- | --- |
| ASI | pod ZIP **351,803,513,445 B**；server ZIP **3,080,121,387 B** | clusterdata 总说明允许研究/学习；ASI 子目录未发现独立标准许可证 | 官方 OSS，支持 Range；本次只验证两天，原始记录不持久化 |
| BurstGPT | 前两基础 CSV **197,102,320 B**；第三 **231,682,327 B** | [CC BY 4.0](https://github.com/HPMLL/BurstGPT/blob/main/LICENSE) | GitHub v2.0 release；本次保存前两份基础文件，保留零输出记录 |
| Azure 2023 | Code **320,117 B**；Conversation **719,188 B** | CC BY 4.0 | 官方仓库 `data/`；本次读取说明与元数据 |
| Azure 2024 | Code **691,989,454 B**；Conversation **1,135,195,393 B** | CC BY 4.0 | 官方 release；各抽首尾 64 KiB，未保存整表 |
| Azure LMM 2025 | gzip **8,524,196 B** | CC BY 4.0 | 官方仓库 `data/`；本次读取说明与元数据 |
| Functions 2019 / 2021 / Blob 2020 | 分别 **142,968,140 B** tar.xz、**18,444,269 B** RAR、**900,450,549 B** bz2 | CC BY 4.0 | Azure 官方仓库及 releases；本次核验说明和大小 |
| Mooncake | 真实对话 **3,029,533 B**，真实工具/agent **4,415,857 B**；合成 **1,136,313 B**；历史 trace **4,446,104 B** | 仓库 Apache-2.0；trace 说明未另列独立数据许可 | 官方仓库 JSONL；核验 README、字段及文件大小 |
| ServeGen 在线 / 离线 | 当前仓库 `data/` 合计 **915,271,256 B** | 仓库 Apache-2.0；保留论文归属 | 官方 GitHub；读取 m-large 代表 CSV、加载代码与离线说明，不下载整套 |
| xMaaS | daily ZIP **10,528,173,427 B**；week ZIP **4,812,189,789 B**；app ZIP **99,947 B** | clusterdata 研究/学习说明；该子目录未发现独立标准许可证 | 官方 OSS；本次核验 schema/下载说明，未下载归档 |
| PAI 2020 | 七个压缩表合计约 **1.4 GiB**，实例表约 663 MiB | 该子目录明确 [CC BY 4.0](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2020/LICENSE) | 官方 OSS / 作者数据仓库；下载说明提供 SHA-256；本次核验说明 |
| GenTD26 | 本次未核验完整发布包字节数，README 未列出归档总量 | clusterdata 研究/学习说明；该子目录未发现独立标准许可证 | 从官方子目录/其发布入口追踪文件；未实证其多月覆盖 |
| Alibaba 微服务 2021 | 官方列 node 1.1 GiB、调用图 25 GiB、资源 16 GiB、调用指标 19 GiB | clusterdata 研究/学习说明 | 官方 `fetchData.sh` 指向分表归档；本次仅文档核验 |
| Google Borg / PowerData | Borg 官方约 **2.4 TiB 压缩**；PowerData 大小本次未核验 | CC BY 4.0 | 官方 BigQuery 和关联说明；未发起计费查询 |
| Meta | 公开 `summary_data_atc23/data/` 文件合计 **35,424,362 B** | CC BY-NC 4.0 | 官方仓库汇总表；本次读取说明与文件树 |
| Chutes | HF 资产约 **94.26 GB**，trace 45 个 Parquet 分片；另有官方小型汇总 | CC BY 4.0 | 官方 HF / S3；本次仅保存 3 个汇总 Parquet，共 **15,587 B**，不下载整年原表 |
| FreeInference Agentic | HF 当前存储约 **9.50 GB**；同时提供不同格式/视图，逐请求 Parquet 合计约 8.35 GB | CC BY 4.0 | 官方 HF 按周文件；可选列流式查询，本次仅元数据核验 |
| TraceLab | v0.0.2 JSONL gzip **100,939,722 B**；DuckDB **160,182,272 B**，是同一数据的两种格式 | 数据 CC BY 4.0；代码 Apache-2.0 | 官方 releases；本次读取版本与资产，不重复合并格式 |

## 3. 已落盘小数据与实际形状

服务器原始数据目录：`/data3/dataset/ljt/global-grid-flexibility-20261023/data/raw/inference-alternatives-20261031/`。先在本地完成官方下载和校验，再复制到服务器；服务器重新计算 SHA-256。五个文件共 **197,117,907 B**，URL、字节与哈希见[服务器下载清单](../../data/metadata/inference-alternatives-20261031/server-download-manifest.json)。

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `BurstGPT_1.csv` | 52,283,111 | `4bb3783693d0a435686fbfc885615d2349bd067239079fa4b749f2e679e12122` |
| `BurstGPT_2.csv` | 144,819,209 | `44bf5942b03fca42c01a226a545ad3a750ec62688faecd67b6831c56a8928bf7` |
| `chutes-hour-of-week.parquet` | 3,940 | `f33e8d4de0e62c95def486a96f6514a0580f47ad58ab020692cb727e2da842ea` |
| `chutes-daily-requests.parquet` | 5,552 | `a2e73c428bd986ced90282d6c8ca53dc7f596196774a1b04badb4fb681c4d720` |
| `chutes-daily-tokens.parquet` | 6,095 | `f263a2e4522a171e41a33e3b142bc79461eb2c0179f5ce5512c98fb8bb2d616f` |

BurstGPT 两份表共 **5,288,173** 条请求；前一文件时间范围 5–5,269,973 秒，后一文件为 5,270,414–10,454,395 秒，覆盖相对日 0–120，文件之间没有时间重叠。原表没有请求 ID，同一秒同一字段的记录可能是真实并发请求，不做没有依据的去重。零输出记录合计 **99,666** 条，按发布说明保留为失败标记；输入 token、输出 token、零输出数量及成功记录输入量分别汇总。

本次排除首尾不完整日，分析相对日 **1–119**，共 **119 天 / 2,856 小时**。所有服务类别使用同一支持窗口。窗口内没有记录的小时填为“零观测”；采集完整性未知，不能把它们全部解释为真实服务停机。总计有 **149** 个零观测小时，因此全时序最大值/最小值之比未定义，不报告有限的全时序峰谷比。[小时表](request-hourly.csv)、[日表](request-daily.csv)、[日内均值](request-diurnal.csv)、[相对七日均值](request-mod7.csv)和[详细统计](request-summary.json)可直接复核。

| 实际观测指标 | BurstGPT 前两文件，119 个完整日 |
| --- | ---: |
| 窗口内请求 | 5,252,477 |
| 输入 / 输出 token | 1,853,340,790 / 309,164,730 |
| 平均日内请求高峰 / 低谷时刻 | 相对当地时钟 16 时 / 5 时 |
| 平均日内请求峰谷比 | 2.455 |
| 相对七日请求均值峰谷比 | 1.662 |
| 平均日内输出 token 峰谷比 | 5.677 |
| 相对七日输出 token 均值峰谷比 | 1.800 |
| 请求序列滞后 24 / 168 小时相关系数 | 0.389 / 0.338 |

各模型/API/对话类别分别计算于统计 JSON。相对日模 7 分组保留七日形状，未将起始日指定为星期一；原始日历不公开。上述统计是描述性证据，受到增长、服务构成和零观测窗口影响，不单独证明稳定周周期或季节性。

Chutes 保存的是作者已经发布的 **367 行每日汇总与 168 行 hour-of-week 汇总**，对应 CSV 一同提交。该汇总共 6,122,413,756 次请求；按发布时钟汇总的日内请求峰谷比约 **1.185**，相对七日请求总量峰谷比约 **1.065**；每日请求最大/最小值之比约 **21.30**，每日 token 约 **18.67**。每日极值受部分日期、平台增长和模型变化影响，不能解释为季节效应。[计算结果](chutes-summary.json)保留具体口径。

这些小表适合核查长期变化与周内平均形状；它们没有保留逐日逐小时的联合变化，不能组合还原完整的 8,760 小时时序。取得一年真实逐小时序列，需要另行读取请求原表的时间与 token 列。

## 4. Azure 2024 首尾验证

[Range 验证记录](azure-2024-range-probe.json)只传输了 **262,144 B**，核对三列字段 `TIMESTAMP,ContextTokens,GeneratedTokens`：

| 文件 | 第一条记录 UTC | 最后一条记录 UTC | 覆盖 |
| --- | --- | --- | --- |
| Code | 2024-05-10 00:00:00.009930 | 2024-05-16 23:59:59.929501 | 约七日 |
| Conversation | 2024-05-12 00:00:00.001163 | 2024-05-18 23:59:59.995460 | 约七日 |

两套服务的起止日期不同，共同时段只有 5 月 12–16 日。各自可作七日对照；将两份直接首尾连接会重复部分日期。首尾读取验证时间边界和字段，不验证中间缺失情况，也没有计算完整文件 SHA-256。官方说明页给出的 5 月 10–19 日属于描述范围；分析使用文件中的实值边界。

## 5. ASI 185 天远程聚合可行性

### 5.1 本次实测

在 `ssh pku-test` 上执行[验证脚本](../../analysis/probe_asi_inference_range.py)，以 `requests` 实现带边界的 HTTP Range 传输，由 Python `zipfile` 解析 ZIP/ZIP64，由 Arrow 读取 Parquet。每次要求 HTTP 206、精确 Content-Range 与字节数，并用 ETag 的 If-Match 固定对象版本；ZIP 读取校验成员 CRC，成员另计算 SHA-256。

两个归档均找到 **185×24=4,440** 个准确小时分区。中央目录 Range 读取量分别为 **512,078 B**（pod）和 **490,778 B**（server）。ZIP 成员为 **ZIP_STORED**；内部 Parquet 保留其原有列压缩，ZIP 外层没有再压缩。可直接定位 `day=<d>/hour=<hh>/part-000.parquet`，无需读取其他日期。

抽取相对日 **0 和 184**，每个日期各读取 24 个 pod 文件及 24 个 server 文件，总共 **96** 个成员。共检视 **69,844,911** 条 pod 小时行，传输 **3,803,826,525 B**；端到端约 **84.70 秒**，最高常驻内存 **298,291,200 B（284.5 MiB）**。原始 Parquet 在内存中处理后释放，原始数据写盘 **0 B**。中央目录、Range 位置、时间、字段及全部成员 SHA 见[抽取审计](asi-range-probe.json)。

脚本筛选 `gpu_request > 0` 后，按**业务类型×优先级×小时**累计申请量、`used_gpu_hours`、宿主 CPU 请求和行数，同时额外保留 GPU 型号与状态维度。样本中归属 GPU 小时没有缺失，正申请筛选外的归属 GPU 小时合计为零。使用 `ParquetFile.iter_batches` 限制解码批量；每完成一个成员就释放原文件内容。

| 端点日 | GPU 库存 | 日归属 GPU 小时 | 总量日内峰谷比 | 在线 / 离线推理归属份额 |
| --- | ---: | ---: | ---: | --- |
| 0 | 111,321 | 2,410,951.34 | 1.080 | 54.06% / 21.88% |
| 184 | 155,410 | 3,620,551.85 | 1.106 | 54.58% / 20.39% |

这些份额仅代表两个被抽取日期，不作为全部 185 天的固定比例。跨日总量变化同时包含库存扩张。两天抽样不能测量七日周期。[推理类别小时表](asi-inference-hourly.csv)只含在线与离线推理；[类型×优先级小时表](asi-hourly-by-type-priority.csv)、[型号/状态细分](asi-hourly-by-type-priority-gpu-state.csv)、[小时总表](asi-hourly.csv)与[估算汇总](asi-summary.json)保留全类型核验依据。

仅在线与离线推理的日归属 GPU 小时分别为 **1,830,896.54** 与 **2,714,380.65**，日内峰谷比分别约 **1.166** 与 **1.124**。这些数值使用推理分类子集，GPU 库存仍是全机群库存，不能据此计算推理专属服务器利用率。

**资源口径须保留限制：**申请 GPU / 库存最高约 **1.196**；末日的 2–5 时，归属 GPU 小时 / 库存分别约 **1.025、1.036、1.019、1.025**。记录保持原值。大部分状态为 `Unknown`，不能用 `Running` 筛选替代完整资源核算。归属量是来源定义的资源代理，其与同小时库存的偏差需进一步审计，不能直接输入要求 0–1 的物理利用率接口。GPU 活动、分配与功率仍分别处理。

### 5.2 全量流式成本：阶段估计

全量需传输 pod+server 两表约 **354.88 GB / 330.51 GiB**，无需 network 表或 job summary 才能得到本任务的小时类型/优先级曲线。即使只保留少数列，本次验证方法仍读取完整的单小时 Parquet 成员，因此不能声称传输量降至汇总 CSV 的大小。

| 估算依据 | 单进程时间 / 占用 |
| --- | --- |
| 48 MB/s，十进制吞吐的纯传输下限 | 约 2.05 小时 |
| 按实测 96 个成员处理速率外推 | 约 2.13 小时 |
| 按本次端到端字节吞吐外推 | 约 2.19 小时 |
| 运行安排建议 | **预留 3–4 小时**，覆盖网络、其他任务竞争与检查点开销；这是阶段估计 |
| 单小时最大 Parquet 成员 | 111,942,754 B，约 106.8 MiB |
| 内存方式 | 实测峰值约 285 MiB；建议为单进程预留至少 1 GiB 内存 |
| 必需的类型×优先级小时 CSV | 按抽样密度外推约 3.75 MB |
| 保留 GPU 型号和状态的细分 CSV | 按抽样密度外推约 34.12 MB |
| 磁盘安排 | 内存原始成员 + 增量聚合：预留 **1 GiB** 给汇总、清单和检查点；若使用单成员磁盘暂存，另留约 0.11 GiB；不保存 330 GiB ZIP |

全量实施应按小时写入可重用的小型聚合及完成清单，保留对象 ETag、成员 CRC/SHA、缺失和异常计数，完成后核对 4,440 小时覆盖与资源守恒。原始成员处理后立即释放。网络或字段异常时在相应小时明确失败，后续从已完成清单继续。**本次没有运行或启动 ASI 全量任务。**验证脚本将日期限制为最多两日，防止把本次抽样入口当成全量下载器。

## 6. token / 请求到 GPU 小时的假设

首先保留每小时输入 token `I[m,h]`、输出 token `O[m,h]`、请求数量 `N[m,h]`、失败标记和模型类别。给定模型 m 和硬件 g 的有效 GPU 秒/token 系数 `a[m,g]`、`b[m,g]`，以及每请求额外 GPU 秒 `c[m,g]`，一种明确标记的代理为：

\[
W_{g,h}=\frac{1}{3600}\sum_m\left(a_{m,g}I_{m,h}^{uncached}+a^{cache}_{m,g}I_{m,h}^{cached}+b_{m,g}O_{m,h}+c_{m,g}N_{m,h}\right).
\]

如果系数来自多卡副本的每 token 墙钟时长，需要乘以副本卡数；如果吞吐已经按 GPU 秒给出，不能再次乘以卡数。`输入+输出 token` 乘单一系数只适合明确声明的简化敏感性情境。实际标定至少需要：

1. 模型、激活参数量及 MoE 路由，量化精度、上下文长度与图片编码成本；GPT 服务内部硬件和模型细节未知时保留标定范围。
2. prefill / decode 的不同成本，batch size、连续批处理、张量/流水并行、推测解码和运行软件；相同 token 数不保证相同执行时间。
3. prefix cache 命中及共享上下文。Chutes 的缓存字段早期缺失，BurstGPT 基础表没有缓存字段；缺失不能直接补为确定零缓存。
4. TTFT / TPOT 与尾延迟目标、排队、预留冗余和模型常驻显存。实际 GPU 容量需求通常高于纯计算忙碌小时；模型加载与空闲卡功率另算。
5. 失败请求、重试和取消的已消耗工作量。零输出不表示未消耗 GPU；请求结束延迟也不能整段作为 GPU 独占执行时间。
6. 对长请求按实际执行区间切分小时。只有到达/token 时，将 token 全归于到达小时属于需求到达代理，不能称为执行功率曲线。

应用到 B-INF / C-INF / D-INF 时，应对同一服务工作量分别标定相应硬件与服务器配置，再计算 CPU、GPU、主机其他组件和冷却；不能仅按卡数把异构卡小时等同为 H100 卡小时。四档 MW 规模需要明确容量与归一化分母。ASI 的原始库存、请求/归属量、归一化后的设计容量各自保存，不将大于库存的原始比值截断到 1。

在线推理的延迟预算通常远小于一小时；数据中的 `LP`、低利用率或离线标签均不自动授予延后/暂停权限。需求统计与柔性 SLA 约束分别进入模型。

## 7. 磁盘、复现与待定事项

初次核验 `/data3/dataset` 可用 **23,705,219,072 B**，低于 30 GiB。其间 ASI 采用内存读取、小数据先在本地保存。后续同一共享卷经外部扩容恢复大量可用空间，本任务随后才创建服务器原始数据新目录并传入小数据。交付核验时可用约 **6.49 TB**，文件合计仅 197,117,907 B；[服务器空间与哈希读回](../../data/metadata/inference-alternatives-20261031/server-storage-verification.json)记录实测时刻与精确字节，[验证记录](validation.json)记录跨环境比较。空间变化不归因于本任务。本任务没有操作其他任务的分支、目录或下载进程。

下载入口为原文件大小加 **30 GiB 保留量 + 2 GiB 并发余量**预检，并在下载期间继续检查可用空间；以 `df /data3/dataset` 对应文件系统为准。本次提交限于调研脚本、元数据、汇总与项目入口链接，原始 CSV/Parquet 留在已忽略的 `data/raw/` 或服务器。

在项目根目录安装 `analysis/inference-data-requirements.txt` 中依赖，另需 `curl`、`gh` 和可用的 SSH。`gh` 使用用户指定的 `HTTPS_PROXY=http://127.0.0.1:7897`。

```bash
python analysis/inference_data_sources.py sources --output .codex-temp/inference-sources
python analysis/inference_candidate_metadata.py
python analysis/summarize_inference_requests.py --datasets burst-1 burst-2 --raw-dir data/raw/inference-alternatives-20261031 --output results/inference-data-alternatives-20261031
python analysis/probe_inference_small_sources.py --output results/inference-data-alternatives-20261031/azure-2024-range-probe.json
python analysis/build_inference_probe_summary.py --output results/inference-data-alternatives-20261031
python analysis/verify_inference_delivery.py
```

ASI 验证须在能访问官方 OSS 的服务器运行，stdout 输出 JSON，可以通过 SSH stdin 传入脚本并将 stdout 保存到本地；命令本身不创建服务器原始数据文件：

```bash
python analysis/probe_asi_inference_range.py --days 0 184 --disk-path /data3/dataset > asi-range-probe.json
```

服务器上的 BurstGPT 汇总入口与本地相同，将 `--raw-dir` 和 `--output` 替换为服务器路径。本次在服务器真实运行，使用已下载文件重新核对哈希，并与本地的小时聚合进行读回比较。验证覆盖数据传输、ZIP/Parquet/CSV 解析、小时覆盖、分组守恒、统计与跨环境读回；未运行年度气象、能耗或优化全流程。

待用户决定的事项：

- 是否按本报告方案启动 ASI 全部 185 天的逐小时资源聚合；本次授权止于可行性验证。
- INF 是否进一步限定为纯 LLM 在线推理。当前推荐覆盖 ASI 的在线与离线推理，包含多种模型；若限定为纯 LLM 在线服务，应以请求数据重新定义需求，并完成 GPU 性能标定。
- 是否另行处理 Chutes 一年原表以检验长期变化。其体量约 94 GB，超出本次小数据下载范围；已保存的小型汇总无法替代逐小时原表。

本任务 PR 保持待审，未经用户明确批准不合并。
