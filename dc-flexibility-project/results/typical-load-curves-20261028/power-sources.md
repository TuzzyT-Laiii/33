# 功率模型、来源与假设

核查日期：2026-10-28。设备定义采用 [PR #12](https://github.com/TuzzyT-Laiii/33/pull/12) 的 `server-count-scenarios-20261028/server-configurations.json`。本目录曲线用于查看设备组成、需求形状与天气如何共同影响功率。目标设备的联合 CPU/GPU 实测响应尚未取得。

## 整机边界

每台服务器或每个完整 GB200 NVL72 机柜采用：

`P_unit = P_base + n_cpu × P_cpu(u_cpu) + n_gpu × P_gpu(u_gpu)`。

`P_base` 包括内存、主板、机内存储、风扇、网卡及电源转换损耗；DGX 的 NVSwitch、GB200 的柜内交换、管理网络和电源架也归入此项。它是固定的有效交流功率残差假设，未将每个部件分别测量；电源损耗随负载变化的真实关系也包含在这个近似中。机房冷却与其他设施用电另行计算。外部共享存储、柜外业务网络、UPS/配电的额外损耗未另行标定，当前设施结果属于这一限定边界内的演示。

## 已核查的公开依据

| 来源 | 可核查信息 | 本轮使用范围 |
| --- | --- | --- |
| [SPECpower：SR650 V3，20230619-01282](https://www.spec.org/power_ssj2008/results/res2023q3/power_ssj2008-20230619-01282.html) | 双路 Xeon Platinum 8490H、256 GB、16 DIMM、1×1100 W PSU；Active Idle 为 163 W，100% 目标负载为 915 W | 同平台的整机响应形状及功率量级；CPU型号、内存和电源配置与目标配置有差异 |
| [SPECpower：HS5610，20240227-01377](https://www.spec.org/power_ssj2008/results/res2024q1/power_ssj2008-20240227-01377.html) | 单路 Xeon Gold 6448Y、128 GB、8 DIMM、1×1400 W PSU；Active Idle 为 76.2 W，100% 目标负载为 318 W | 与 C20/C21/C22 相同 CPU 的归一化响应形状；整机平台和 CPU 颗数不同 |
| [Lenovo SR650 V3 产品指南](https://lenovopress.lenovo.com/lp1601-thinksystem-sr650-v3-server) | 双路平台、6448Y 每颗 225 W TDP、H100 受支持选件及热规则 | 设备组成及 CPU 中央端点的参考规格 |
| [Lenovo H100 产品指南](https://lenovopress.lenovo.com/lp1732-thinksystem-nvidia-h100-pcie-gen5-gpu) | H100 80 GB PCIe 350 W、HGX H100 80 GB 700 W 八卡板 | GPU 满载演示端点的规格上限；TDP 不能视为任务实测耗电 |
| [NVIDIA DGX H100 官方指南](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html) | 双路 Xeon 8480C、8 张 H100、最大整机 10.2 kW | C28 的设备边界与最大规格校验；中央参考功率 8.1 kW 为假设 |
| [NVIDIA GB200 官方硬件指南](https://docs.nvidia.com/dgx/dgxgb200-user-guide/hardware.html) | 18 个计算托盘，每托盘 2 Grace + 4 Blackwell；9 个 NVLink 交换托盘、2 个管理 TOR；整柜功耗约 120 kW | 完整机柜组成和台数换算的约值依据；120 kW 的公开值没有确定的任务工况，仍以演示参考使用 |
| [Supermicro GB200 官方公告](https://www.supermicro.com/en/pressreleases/supermicro-grows-ai-optimized-product-portfolio-new-generation-systems-and-rack) | GB200 机柜中的 72 个 Blackwell GPU，单 GPU 为 1200 W TDP | GB200 所用 Blackwell 的规格上限；不与独立 HGX B200 的功率规格混用 |
| [NVIDIA Grace 电源与热管理](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/power-thermals.html) | CPU/GPU 模块间有动态功率分配；Grace 部分遥测边界含内存和调压损耗 | 说明简单可加模型的局限；没有从这里取得匹配 GB200 的 Grace 空载或满载实测端点 |
| [Latif 等，完整可读预印本](https://arxiv.org/html/2412.08602v2)，对应 [IEEE Access 2025 DOI](https://doi.org/10.1109/ACCESS.2025.3554728) | Table 1：另一台 8×H100 HGX 节点空载 1.86 kW、仅 GPU 压力测试 7.97 kW、GPU+CPU 压力测试 8.43 kW；训练工况中位数 4.68/6.26/7.92 kW | 外部整机量级比较；这些测量不属于本轮 DGX，也没有分离单 GPU 空载功率。本轮读取预印本全文并核对发表 DOI，未用这些测量标定 DGX 的组件 |

两项 SPECpower 的全部 11 档功率整理于 [specpower-evidence.csv](specpower-evidence.csv)。这些值均为测试整机的交流输入，负载档表示 ssj 吞吐目标比例。它们没有提供单独的 CPU package 功率，也没有直接测量任务迹对应的 CPU 利用率。

## 中央端点及假设区间

以下单位均为 W；CPU/GPU 列为每个器件的值。括号为低、高两档端点，全部工作端点与 `P_base` 分解均标记 `assumption`。GPU 满载中央值取公开 TDP，CPU 6448Y 中央满载值取公开 225 W TDP；这些选择的状态仍是工作功率假设。

| 配置 | CPU / GPU 数量 | P_base | CPU 空载 → 满载 | GPU 空载 → 满载 | 中央整机空载 / 参考满载 | 10 MW 数量 |
| --- | --- | --- | --- | --- | --- | --- |
| C20 | 2 / 0 | 150（100–250） | 20→225（10→180；35→270） | 无 GPU | 190 / 600 | 16666 台 |
| C21 | 2 / 1 | 200（150–350） | 20→225（10→180；35→270） | 50→350（25→280；80→350） | 290 / 1000 | 10000 台 |
| C22 | 2 / 2 | 250（200–450） | 20→225（10→180；35→270） | 50→350（25→280；80→350） | 390 / 1400 | 7142 台 |
| C28 | 2 / 8 | 1800（1000–3000） | 50→350（25→280；75→400） | 70→700（40→560；120→700） | 2460 / 8100 | 1234 台 |
| GB200 NVL72 | 36 / 72 | 24600（18000–30000） | 40→250（20→180；80→350） | 120→1200（80→960；200→1200） | 34680 / 120000 | 83 柜 |

C20 的基础功率和空载值在公开 SR650 量级附近，适度增加 GPU 配置的机内风扇、电源和配套部件预算。C28 的中央满载量级接近外部 HGX 压力实验，但中央空载及分解没有匹配测量，因此使用宽区间；三档满载为 6.04、8.10、9.40 kW，均低于 DGX 官方 10.2 kW 最大规格。该量级比较不能确认目标 DGX 的实际工况。

GB200 取每 Grace 250 W、每 GPU 1200 W 后，以 `120000 - 36×250 - 72×1200 = 24600 W` 得到基础残差。交换托盘、柜内网络、风扇、电源等的实际分项和空载功率均未测量。低、高满载约为 93.60、129.00 kW。Grace 与 GPU 存在动态功率分配，独立相加仅适合演示。本轮没有获得匹配器件的单 GPU 空载、Grace 空载或完整机柜联合实测响应。

区间用于覆盖内存配置、电源与风扇状态、器件功率管理和任务类型差异。区间由研究假设指定，没有统计概率或置信含义。低档 GPU 工作端点取 TDP 的 80%；高档取 TDP。Xeon Gold 的高档有效端点使用 270 W，用于保留功率转换及强负载的不确定性，不能解释为厂商 6448Y 的新 TDP 规格。

## 响应曲线

CPU 中央响应使用同 CPU 的 SPECpower 曲线，按其整机空载和满载归一化：

`f_Dell(u) = interp(u, 目标负载档, (P_SPEC - 76.2) / (318 - 76.2))`。

`P_cpu(u) = P_cpu_idle + (P_cpu_full - P_cpu_idle) × f_Dell(u)`。

同平台 SR650 的曲线按 `(P_SPEC - 163)/(915 - 163)` 归一化为 `f_Lenovo(u)`。低档使用 `min(f_Dell, f_Lenovo)`，高档使用 `max(f_Dell, f_Lenovo)`。C28 转移同代 Xeon 形状；Grace 使用线性响应，均为假设。跨平台的归一化形状不能解释为目标 CPU 的 package 实测曲线。

GPU 使用 `P_gpu(u) = P_idle + (P_full - P_idle) × u^a`，中央 `a=1`，低档 `a=1.2`，高档 `a=0.8`。没有用 GPU TDP 乘占用率替代整个服务器的功率。

## 需求到 CPU/GPU 的映射

- Lisa 使用 309 个完整日的 CPU 分配核心小时形状；`u_cpu` 为既有算法的峰值归一化资源份额，`u_gpu=0`。GPU 保持通电空载。Lisa 没有提供 GPU 任务证据。
- Acme 同时重放 90 日 CPU 与 GPU 分配资源记录。`u_gpu` 为既有 GPU 资源份额；`u_cpu=max(u_Acme_CPU, k×u_gpu)`。CPU 作业与 GPU 作业的宿主开销不等价，因此额外设置宿主代理。
- ASI 仅有 24 小时，全年严格重复。`u_gpu` 为 GPU 小时/异构库存的份额；`u_cpu=k×u_gpu`。
- 中央 `k=0.25`，低档 `k=0.10`，高档 `k=0.40`，全部为假设。资源分配/归属份额与芯片活跃利用率的转换、CPU 架构转移和 GPU 型号转移均未标定。

相同形状在不同配置上重放，预算按各自完整设备参考功率换算数量。CPU 核心数、GPU 数量与吞吐量随配置变化。曲线比较没有固定实际完成工作量，不能据此形成设备能效排名或投资结论。

## 冷却、PUE 与敏感性边界

风冷四配置直接使用仓库 `ComputeLoadModel` 的冷却系数、干球/露点分段特征及设备参考容量，替换其 IT 输入。数值核验将新接口与原接口逐小时比较。设施功率为 `IT + cooling_mw + other_mw`；PUE 为全年设施电量/全年 IT 电量，冷却占比为冷却/设施电量。模型基于 ESIF 数据，跨设施及不同 IT 份额的外推小时单独计数；未加入对不同目标机房的性能保证。

GB200 官方文档说明 CPU/GPU 使用液冷，网络、存储及电源等仍有风冷。现有房间风冷模型缺少供回液温度、流量、CDU 和残余风冷数据。本轮采用用户允许的 IT-only 路径：逐小时冷却、其他设施、设施总功率和 PUE 为空，图中虚线或斜线柱表示 IT。缺失值不计为零，不计算 GB200 设施电量或冷却占比。

低、中央、高三档使用固定中央台数，同时改变端点、响应形状和宿主代理，保留小时结果与 [sensitivity-summary.csv](sensitivity-summary.csv)。这是参数情景范围；它没有覆盖所有需求迁移与季节性不确定性。高档可能超过原 10 MW 参考预算，汇总表单列超预算小时，未截断功率。
