# CPU/GPU 服务器与整柜情境（20261028）

[Markdown 说明](算力中心_CPU_GPU服务器与整柜情境及需求来源说明_20261028.md)与[PDF 说明](算力中心_CPU_GPU服务器与整柜情境及需求来源说明_20261028.pdf)提供五种配置及允许配对的 52 个设计单元。

| 配置 | 基本单元 | CPU∶GPU | 需求配对 | 四档合计 | cooling_type |
| --- | --- | --- | --- | --- | --- |
| C20 纯 CPU | SR650 V3 完整服务器，2U | 2∶0 | Lisa CPU 作业、Alibaba Cluster Trace 2018 | 8 | air |
| C21 | SR650 V3 完整服务器，2U | 2∶1 | Lisa、Acme、ASI | 12 | air |
| C22 | SR650 V3 完整服务器，2U | 2∶2 | Lisa、Acme、ASI | 12 | air |
| C28 | DGX H100 完整服务器，8U | 2∶8 | Lisa、Acme、ASI | 12 | air |
| R-GB200 基本纯 GPU | GB200 NVL72 完整机柜 | 36∶72 | Acme、ASI | 8 | direct_liquid |
| 合计 | 1 / 10 / 100 / 500 MW | 每档 13 个允许组合 | 4×（9＋2＋2） | 52 | 44 air＋8 direct_liquid |

C20 采用双路 Xeon Gold 6448Y，每颗 32 核、225 W TDP，共 64 核；不装计算 GPU。厂商支持选配依据见 [Lenovo 产品指南](https://lenovopress.lenovo.com/lp1601-thinksystem-sr650-v3-server)标准规格和第 19 表。管理控制器集成显示不计入计算 GPU，完整内存/存储/风扇/电源 BOM 与运行功率仍待核定。

R 表示 rack，R-GB200 按整柜计量。每柜有 18 个计算托盘（每个 2 颗 Grace＋4 个 Blackwell B200）及 9 个 NVLink 交换托盘（每个 2 个 NVSwitch），36 颗 Grace 每颗 72 核，共 2592 核。GPU 显存整柜为官方约数 13.4 TB HBM3E；双 GPU Superchip 为 372 GB，单 GPU 的 186 GB 由其除以 2 得到。规格见 [NVIDIA 产品页](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)、[硬件指南](https://docs.nvidia.com/dgx/dgxgb200-user-guide/hardware.html)和[参考架构](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/dgx-superpod-components.html)。

GPU 服务器需要宿主 CPU。“基本纯 GPU”定义为 CPU 仅承担主机及 GPU 配套工作、电耗以 GPU 为主的研究形态，实际 CPU/GPU 电力份额保留待测。整柜计算 CPU 不包括 DPU/BMC 或交换管理处理器。

硬件指南的整柜功耗约 120 kW 单独记为 `vendor_approx_rack_power_w`；任务条件与测量边界未明确。所查 [NVIDIA 数据表](https://resources.nvidia.com/en-us-dgx-systems/dgx-superpod-gb200-datasheet)及硬件指南未列外部机械尺寸，整柜高/宽/深和高度 U 数留空，待官方机械规格。不得用计算托盘与交换托盘的 U 数之和代替整柜高度。

- [设备参数 CSV](server-configurations.csv)与[JSON](server-configurations.json)：`research_unit=server/rack`、统一的 `per_unit` 数量字段、规格、来源和核验日期。旧 `per_node` 字段仅用于服务器，GB200 对应字段为空。
- [52 个设计单元 CSV](scenario-matrix.csv)与[JSON](scenario-matrix.json)：每个单元具有 `cooling_type`、`demand_readiness` 和功率/冷却模型状态。`scenario_id` 沿用 `规模MW_配置编号_来源首词小写`；原有 36 个标识与既有字段保留。
- [生成核验](validation.json)与[文件 SHA-256](manifest.json)：含旧版全部文件的逐项 SHA-256，旧版 [20261027](../server-count-scenarios-20261027/README.md)保持原样。

Alibaba 2018 只用于 C20 的 CPU 云混部设计。本次只核对[官方说明](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md)与 [schema](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/schema.txt)，未下载原始数据；下载、校验与字段实值核验仍待完成。Lisa、Acme、ASI 的既有核验状态保留，向新硬件的联合占用与性能映射仍需核对。

GB200 的 CPU/GPU 冷板必须接入直接液冷，部分网络、存储及电源部件仍保留风冷。现有冷却/气象模块按风冷机房设计，液冷单元需另建并校准供液温度、CDU/泵/风机和残余风冷热量参数；干冷器与冷却塔分别受干球、湿球条件影响。本次仅登记待办，未实现新模型或赋予供液温度、流量、COP/PUE、气象系数数值。

五类配置的匹配整机/整柜运行功率均未标定。`measured_reference_unit_w`、`measured_reference_node_w`、CPU/GPU 功率份额、`unit_count`/`node_count`/`rack_count` 及累计硬件数量均为空。10.2 kW 为 DGX H100 最大铭牌，约 120 kW 为 GB200 厂商描述，均不作为运行参考功率。未计算新增年曲线；[既有年度工作包](../scenario-load-methods-20261026-v2/README.md)按原版本使用。

在项目根目录运行 `python analysis/build_server_count_scenarios.py` 可生成本版参数、说明、渲染图片及机器核验；图片保存于已忽略的 `.codex-temp/server-count-scenarios-20261028/`。来源核验日期为 2026-10-28，旧配置与样本按 2026-10-27 已核验版本保留。
