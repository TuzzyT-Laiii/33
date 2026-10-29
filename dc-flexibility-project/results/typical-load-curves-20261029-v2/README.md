# 方向一典型负荷曲线 v2

输入基线为 `origin/main` 的 `906928d`，采用 PR #14 的 52 单元需求矩阵、PR #15 的冷却 v2 和 PR #19/#20 的 MIT GPU 监测批次。与 [v1](../typical-load-curves-20261028/README.md) 使用相同的 `g05_r100_c592` 中心节点、2025 年 8760 小时天气，气象文件 SHA-256 与 v1 一致。

四档规模为 1、10、100、500 MW 安装参考 IT 预算。每档 13 个允许组合，12 个已有输入，C20×Alibaba 标为**待数据**并跳过；共计算 48 个设计单元。图件展示 10 MW，台数/柜数沿用 v1 中央满载参考功率并向下取整。不同规模的需求份额形状、PUE 和负荷率一致，绝对功率随设备数量缩放，已调用冷却模型核验。

## MIT 固定快照与统计

快照位于 `D:\mit-supercloud-gpu-summary-20261028\aggregates\snapshot-20261029T014000Z\`。它固定已提交批次中 seq<1574 的连续随机前缀：**1,574 文件**、844,961,471 行，源对象约 41.70 GB。有效文件 1,571 个，另 3 个跨夏令时转换的文件待核查并排除。统计覆盖 **24,235.4047 个有效分配 GPU 小时**、26,270 个 GPU 小时片段。样本内监测时长/对应有效文件调度 GPU 时长为 99.993%；该比例不证明全数据集采集完整。

源 GPU 按同平台资料识别为 **Tesla V100 PCIe 32GB，官方 TDP 250 W**。平台识别依据为 Samsi 等的 [Supercloud 实验配置及表 I](https://arxiv.org/pdf/2310.03003)，250 W 另与 [NVIDIA V100 PCIe 官方说明](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/Tesla-V100-PCIe-Product-Brief.pdf)核对。监测文件仅有 TRES `gpu:volta` 归属，未逐文件独立记录板卡型号、实际设置的功率上限或 DVFS 状态；型号识别及跨代使用保留这一边界。

每个作业文件、GPU、UTC 小时先计算 `有效功率时间积分 / 有效覆盖时间`，再按有效 GPU 小时加权统计。边界小时按实际覆盖时长计权；相邻观测向前保持最多 1 秒，末端不外推，缺失监测不补零。小时内波动来自时间加权二阶矩。秒级直方图仅辅助报告，采用 5 W 箱中心，误差上限 2.5 W。方法与完整分位数见 [MIT 统计](mit-power-ratio-summary.json)。

| GPU 小时平均功率统计 | 均值 | P10 | P25 | 中位 | P75 | P90 | P95 | P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 功率（W） | 93.87 | 27.30 | 43.69 | 83.97 | 147.12 | 170.19 | 198.51 | 211.63 |
| 功率/TDP | 0.3755 | 0.1092 | 0.1748 | 0.3359 | 0.5885 | 0.6808 | 0.7940 | 0.8465 |

空载定义为已分配观测区间内 `nvidia-smi utilization=0`，占 **29.401%**。这些区间的功率中位约 37.5 W。利用率时间加权均值为 50.97%。小时内功率标准差的均值、中位和 P90 分别为 **22.89、12.84、56.47 W**。空载占比未包含未分配 GPU 的时间。

| 固定随机前缀文件数 | 有效 GPU 小时 | 均值/TDP | P10/TDP | 中位/TDP | P90/TDP | 空载占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 9,442.03 | 0.3099 | 0.1062 | 0.2497 | 0.5949 | 32.19% |
| 1,000 | 15,290.03 | 0.3455 | 0.1070 | 0.2924 | 0.6642 | 28.37% |
| 1,574 | 24,235.40 | 0.3755 | 0.1092 | 0.3359 | 0.6808 | 29.40% |

前 1,000 到 1,574 文件的均值和中位仍分别增加 **8.66% 和 14.86%**，P90 增加 2.49%；500 到 1,000 文件的均值和中位增加 11.50% 和 17.10%。按诊断用的 5% 变化阈值，**分布尚未稳定**。文件简单随机采样后按 GPU 时间加权，长作业会明显影响结果；嵌套样本的接近程度不构成总体收敛或代表性证明。完整结果见[稳定性表](mit-power-ratio-stability.csv)。后台下载和聚合进程继续运行，本次没有修改其状态、日志或运行文件，也未改写 `aggregates/current`。

## 功率与冷却方法

令每小时分配份额为 `a`，每台 GPU 数为 `n`，小时平均实测功率/TDP 的选定分位数为 `r`：

`P_GPU = n × [a × TDP_target × r + (1−a) × P_idle_target]`

`P_IT = P_base + P_CPU + P_GPU`。比例分布已经包含分配期间空载状态，不再乘以 50.97% 利用率或扣除空载占比。每档分位数在全年保持固定，形成可核查的敏感性情境；没有人为生成秒级波动、随机年景或跨 GPU 同步峰值。

| 目标 GPU | TDP（W/颗） | 分配期间 P10 / 中位 / P90（W/颗） | 未分配空载（W/颗） | 依据 |
| --- | ---: | --- | ---: | --- |
| C21/C22：H100 PCIe | 350 | 38.22 / 117.56 / 238.26 | 50 | TDP 为厂商规格；空载延续 PR13 中央假设 |
| C28：H100 SXM | 700 | 76.44 / 235.12 / 476.53 | 70 | TDP 为厂商规格；空载延续 PR13 中央假设 |
| R-GB200：NVL72 中的 Blackwell GPU | 1,200 | 131.04 / 403.06 / 816.90 | 200 | NVIDIA DPS 0.8 设备模板的 maxLoadWatts/minLoadWatts；空载为工程代理 |

H100 空载值尚无同配置独立实测，GB200 的 200 W 是官方软件设备模板参数，未视为目标机柜电表测量。P10 的分配期间功率可低于 C21/C22 和 GB200 的空载代理，汇总表用 `allocated_power_below_idle` 明示；该组合描述跨工作状态、跨代的低功率情境，不能推导单卡增加分配后的因果响应。没有人为裁切分位数。

方法档位为 `v1_utilization`、`ratio_p10`、`ratio_p50`、`ratio_p90`；C20 只有 `cpu_spec`。`v1_utilization` 使用**相同 v2 需求**和 v1 中央功率函数，便于区分 GPU 方法变化；它与历史 v1 曲线的对照各自保存。主图与 `annual-summary.csv` 采用 `ratio_p50` 和冷却中央参数。中位情境不解释为分布均值对应的期望电量，P10/P90 也不解释为整座设施年度电量或峰值的概率区间。

CPU 沿用 PR13 的 SPECpower 整机测量归一化响应及中央端点，`P_base` 和单 CPU 功率端点仍含假设，未把整机 AC 测量当作 CPU 封装测量。GB200 的 Grace CPU 继续使用原中央线性代理。未分配 GPU 的功率处理仅进入 GPU 分项，机内风机、存储等固定部分留在 `P_base`。

科研 GPU 的 GPU 与同作业宿主 CPU 分配曲线来自 v2 矩阵的 MIT 237 日输入，分别按小时分配峰值 428.92 张 GPU、7,394.9089 个 Slurm CPU 调度单位归一化。分资源归一化保留各自形状，未建立实际宿主核心/卡比例或跨硬件吞吐等价。C20×Lisa 继续用 309 日 CPU 分配代理。Acme 为 90 日原序循环，ASI 维持**单日重复演示**；二者只有分配信息，同样借用 MIT 功率比例分布，属于跨工作负载假设。Acme CPU 与宿主 CPU 的合成继续采用 PR13 中央规则；ASI 宿主 CPU 份额继续为 GPU 份额的 0.25 倍。

风冷配置采用 `CoolingModel(cooling_type="air")`，GB200 采用 `direct_liquid`，每种 GPU 方法均计算 `low/central/high` 三套冷却参数。显式包含机房风机、泵、CDU、自然冷却、机械补冷、UPS/设施配电与其他损耗；参数与适用范围见[冷却来源说明](../cooling-recalibration-20261028-v2/README.md)和[本次冷却参数表](cooling-parameters.csv)。高低参数为工程敏感性，尚无目标设施联合电表和供回液验证。

## v1 → v2：10 MW 中央结果

年电量为 GWh，峰值为 MW，PUE 为全年设施电量/全年 IT 电量。三组“Lisa→MIT”同时改变需求来源、GPU 功率和冷却模型，不能将变化全部归因于其中一项。

| 配置×需求 | 年电量 v1→v2 | 峰值 v1→v2 | PUE v1→v2 |
| --- | ---: | ---: | ---: |
| C20×Lisa | 78.93→96.30 | 10.99→14.28 | 1.075→1.311 |
| C21×Lisa→MIT | 57.76→61.20 | 7.82→11.00 | 1.093→1.333 |
| C21×Acme | 47.28→45.32 | 6.81→6.30 | 1.110→1.340 |
| C21×ASI | 64.56→53.64 | 7.73→6.79 | 1.086→1.333 |
| C22×Lisa→MIT | 48.69→55.29 | 6.46→9.61 | 1.107→1.336 |
| C22×Acme | 49.51→43.26 | 7.42→6.07 | 1.106→1.342 |
| C22×ASI | 71.70→52.00 | 8.61→6.59 | 1.080→1.334 |
| C28×Lisa→MIT | 35.92→50.27 | 4.41→7.82 | 1.139→1.338 |
| C28×Acme | 56.05→45.30 | 8.68→6.41 | 1.096→1.340 |
| C28×ASI | 85.25→55.52 | 10.26→7.05 | 1.071→1.331 |
| GB200×Acme | 50.19†→40.64 | 7.86†→5.39 | 待定→1.175 |
| GB200×ASI | 79.74†→47.59 | 9.30†→5.98 | 待定→1.169 |

† GB200 的 v1 仅为 IT，v2 列为设施，设施增幅保留为空。可比 IT 年电量分别从 50.19、79.74 降至 34.58、40.71 GWh。完整精度、设施可比标记及 IT 单独变化见 [v1-v2-comparison.csv](v1-v2-comparison.csv)。

## 图件及图注

所有图件为 300 dpi PNG 和嵌入中文字体的 PDF，五张图均使用中央功率比例和中央冷却参数。H100 的未分配空载、CPU、基础功率及跨代转换含假设。

1. [典型工作日 PNG](figure1-typical-weekday.png) / [PDF](figure1-typical-weekday.pdf)：按目标 2025 年周一至周五的 UTC 小时求均值，展示全部 12 个可计算组合。该工作日定义只作用于重放后的目标日历。
2. [典型一周 PNG](figure2-typical-week.png) / [PDF](figure2-typical-week.pdf)：共同代表周为 **2025-10-20 至 2025-10-26 UTC**。完整周一至周日候选周按 Lisa CPU、MIT GPU/宿主 CPU、Acme GPU、ASI GPU、干球温度六个标准化特征距平均周模板的均方距离选择。逐小时评分只保存本地。
3. [全年包络 PNG](figure3-annual-envelope.png) / [PDF](figure3-annual-envelope.pdf)：实线为日均设施功率，浅色为日内逐小时最小至最大值。**Acme 90 日循环可产生重复的需求方块，不能作为真实季节性证据；ASI 为单日重复演示。** MIT 的 237 日、Lisa 的 309 日原窗口同样按原顺序循环，样本外季节性、增长与节假日未识别。
4. [IT/冷却构成 PNG](figure4-power-components.png) / [PDF](figure4-power-components.pdf)：年电量分解为基础 IT、CPU、GPU、冷却及 UPS/配电与其他损耗；堆积总量等于设施年电量。
5. [v1→v2 对比 PNG](figure5-v1-v2-comparison.png) / [PDF](figure5-v1-v2-comparison.pdf)：按组合比较年电量、峰值和年均 PUE。灰色方形表示 v1、蓝色圆形表示 v2，灰色三角形表示 GB200 的 v1 IT；原版未提供 GB200 PUE，因此该位置只有 v2 结果。

## 文件与字段

- [annual-summary.csv](annual-summary.csv)：48 个有效设计单元的中央结果，包括年 IT/冷却/设施电量、峰值、平均功率、平均负荷率、峰谷比、年均 PUE、冷却占比及可信度标记。
- [sensitivity-summary.csv](sensitivity-summary.csv)：540 行，覆盖四档规模、GPU 四种方法及冷却三档；CPU 配置的 GPU 方法不重复计数。平均负荷率=`年平均设施功率/设施峰值`；峰谷比=`设施最大值/最小值`；冷却占比=`冷却电量/设施电量`。
- [design-cell-status.csv](design-cell-status.csv)：52 单元完整状态，包括 4 个 Alibaba 待数据单元。
- [power-parameters.csv](power-parameters.csv)、[cooling-parameters.csv](cooling-parameters.csv)、[specpower-evidence.csv](specpower-evidence.csv)、[source-register.csv](source-register.csv)：功率、冷却、CPU 测量参考与来源/假设属性。
- [validation.json](validation.json)、[manifest.json](manifest.json)：输入指纹、代码与产物 SHA-256、数值检查、图件核验和本地逐小时文件清单。

540 条逐小时曲线共 4,730,400 行，连同需求重放映射保存在工作树 `data/processed/typical-load-curves-20261029-v2/`，不提交 Git。10 MW 文件名为 `配置-需求-方法-冷却档.parquet`，其他规模增加 `1MW-`、`100MW-` 或 `500MW-` 前缀。按实际台数缩放绝对电功率、热功率和参考容量，比率及时间坐标保留原值。

本机实际存储位于 `D:\dc-typical-load-curves-20261029-v2\data\processed\typical-load-curves-20261029-v2\`，工作树的上述 `data/processed` 路径为目录连接，两处访问同一份已校验文件。物理存储路径另记入 manifest。

MIT 数据许可为 **CC-BY-NC-ND 4.0**，引用 Samsi 等 (2021), *The MIT Supercloud Dataset*, DOI [10.1109/HPEC49654.2021.9622850](https://doi.org/10.1109/HPEC49654.2021.9622850)。本目录仅发布聚合统计、参数、来源和图件；逐作业及逐小时 MIT 衍生表均留在已忽略本地目录。

## 复现

依赖见 `analysis/requirements-typical-load-curves-v2.txt`。需恢复已校验的天气和 Lisa/Acme/ASI 输入，以及 PR14 的 `research-gpu-observed-hourly.csv`；读取路径由参数明确指定。MIT 状态数据库始终只读打开，聚合只读取已提交批次。下面的快照目录必须选用新的名称：

```powershell
python analysis/aggregate_mit_gpu_telemetry.py --root D:/mit-supercloud-gpu-summary-20261028 --output D:/mit-supercloud-gpu-summary-20261028/aggregates/snapshot-reproduce-<时间> --max-prefix 1574
python analysis/summarize_mit_power_ratios.py --root D:/mit-supercloud-gpu-summary-20261028 --snapshot D:/mit-supercloud-gpu-summary-20261028/aggregates/snapshot-reproduce-<时间> --output results/typical-load-curves-20261029-v2 --tdp-w 250 --model "Tesla V100 PCIe 32GB"
python analysis/build_typical_load_curves_v2.py --input-project <已恢复天气与Lisa-Acme-ASI输入的项目目录> --mit-source-dir <PR14已核验小时输入目录>
python analysis/verify_typical_load_curves_v2.py
```

完成实际图件查看后，使用 `--visual-review-complete` 记录验收状态。该标记要求实际查看 PNG 与 PDF 渲染结果，不能由脚本运行成功替代。研究结果支持已声明需求与工程参数下的负荷情境比较；目标设备实测电耗、跨代性能/服务量匹配、完整采集代表性和真实年度需求仍待验证。
