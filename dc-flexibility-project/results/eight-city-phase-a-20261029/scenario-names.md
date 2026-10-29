# 服务器类型 × 用途情境命名

情境名称由服务器类型和用途组成，规模后缀为 `-10MW`。数据来源保留在单独字段中。

| 情境 | 服务器类型 | 用途 | CPU:GPU | 旧配置 | 需求来源 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| A-HPC-10MW | 纯 CPU 型 | 科研计算 | 2:0 | C20 | Lisa | 可计算 |
| A-CLD-10MW | 纯 CPU 型 | 云通用计算 | 2:0 | C20 | Alibaba | 待数据 |
| B-SAI-10MW | CPU 主导型 | 科研 AI | 2:1 | C21 | MIT | 可计算 |
| B-TRN-10MW | CPU 主导型 | 大模型训练/开发 | 2:1 | C21 | Acme | 可计算 |
| B-INF-10MW | CPU 主导型 | 推理服务 | 2:1 | C21 | ASI | 可计算 |
| C-SAI-10MW | 均衡型 | 科研 AI | 2:2 | C22 | MIT | 可计算 |
| C-TRN-10MW | 均衡型 | 大模型训练/开发 | 2:2 | C22 | Acme | 可计算 |
| C-INF-10MW | 均衡型 | 推理服务 | 2:2 | C22 | ASI | 可计算 |
| D-SAI-10MW | GPU 主导型 | 科研 AI | 2:8 | C28 | MIT | 可计算 |
| D-TRN-10MW | GPU 主导型 | 大模型训练/开发 | 2:8 | C28 | Acme | 可计算 |
| D-INF-10MW | GPU 主导型 | 推理服务 | 2:8 | C28 | ASI | 可计算 |
| E-TRN-10MW | 纯 GPU 型（整柜液冷） | 大模型训练/开发 | 36:72 | R-GB200 | Acme | 可计算 |
| E-INF-10MW | 纯 GPU 型（整柜液冷） | 推理服务 | 36:72 | R-GB200 | ASI | 可计算 |

E 型的“纯 GPU 型（整柜液冷）”按用户指定表示 GPU 主导的机柜形态。GB200 NVL72 含 72 颗 GPU 和 36 颗 Grace CPU。
NVIDIA DPS 0.8 设备模板给出每颗 Blackwell GPU 的 maxLoadWatts=1200；NVIDIA GB200 NVL 分区指南给出约120 kW整柜功率。
据此核算 GPU 参考最大功率为72×1.2=86.4 kW，占120 kW整柜参考功率的72.0%；其余28.0%包含CPU、内存、网络、交换和供电等。
72.0%是不同官方设计参数组合后的参考功率占比，随负载变化；年度功率分项继续按v2模型计算。Grace保留主机计算功能，液冷机柜仍包含空气冷却部件。

- [NVIDIA DPS 0.8 设备参数](https://docs.nvidia.com/datacenter/dps/versions/0.8/guides/concepts/device-specifications/)
- [NVIDIA GB200 NVL 分区指南](https://docs.nvidia.com/multi-node-nvlink-systems/partition-guide-v1-2.pdf)
- [NVIDIA GB200 系统架构](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/system.html)
