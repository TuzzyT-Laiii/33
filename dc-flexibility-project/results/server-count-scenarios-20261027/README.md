# 单机 CPU/GPU 三种数量情境

[PDF 说明](算力中心_单机CPU_GPU典型数量配比与需求来源说明_20261027.pdf)按用户确认的单台服务器物理 CPU 颗数与 GPU 数量，定义三种有厂商配置依据的情境：

| 情境 | 每台 CPU∶GPU | 参考平台与证据 |
| --- | --- | --- |
| CPU 数量较多 C21 | 2∶1 | Lenovo SR650 V3 双路＋1 张 H100 PCIe，依据厂商受支持选件与插槽规则选取 |
| 数量相等 C22 | 2∶2 | 同平台双路＋2 张 H100 PCIe，保留两卡插槽与服务器进风约束 |
| GPU 数量较多 C28 | 2∶8 | NVIDIA DGX H100 官方固定整机：2 颗 Xeon 8480C＋8 张 H100 |

单卡与双卡为厂商支持选件构成的研究配置，八卡为官方固定整机。数量比例用于定义每台设备组成，功率、算力及全年电量分别核算；当前没有三类配置市场占比的统计依据。

研究基本单元为完整机架式服务器：C21/C22 均为 2U、双路共 64 个 CPU 物理核心；C28 为 8U、双路共 112 个 CPU 物理核心。CPU 颗数按物理处理器封装计数，GPU 按 PCIe 卡或 SXM 模块计数。整机 IT 包含 CPU、GPU、内存、主板、网卡、机内存储、风扇及电源转换损耗，机房冷却耗电单独计算。

四档 MW 规模表示完整 IT 预算；服务器台数使用分配给计算节点的预算与匹配整机参考功率换算，公共网络及独立存储的预算另行计入。CPU 总颗数、总物理核心数与 GPU 总数量随服务器台数累计。

- [单机配置 CSV](server-configurations.csv)与[JSON](server-configurations.json)：机架形态、U 数、CPU/GPU 数量、核心、型号、显存、部分器件 TDP、证据类型与功率标定状态。
- [36 个设计单元 CSV](scenario-matrix.csv)与[JSON](scenario-matrix.json)：四档规模×三种单机数量配置×三类需求导向，保留气象输入与独立柔性控制定义。
- [生成核验](validation.json)与[文件校验值](manifest.json)。

新三类配置的匹配整机联合功率曲线尚未标定，台数、CPU/GPU 功率份额及新全年负荷保持为空。10.2 kW 是 DGX H100 最大铭牌值，既有 8.43 kW 实测属于另一台 AMD HGX 节点。现有全年算法和已计算示例见[年度构造工作包](../scenario-load-methods-20261026-v2/README.md)。

说明同时回答 ASI 的用途及其他数据来源：Lisa、Acme、ASI 为工作区已详细核验主样本；Philly、Alibaba CPU 2018 和 Google 2019 PowerData 可分别补充联合占用、CPU 混部与功率验证，本次只核对官方字段，未下载计算。

在项目根目录运行 `python analysis/build_server_count_scenarios.py` 可生成说明和参数表。源程序见 [build_server_count_scenarios.py](../../analysis/build_server_count_scenarios.py)。
