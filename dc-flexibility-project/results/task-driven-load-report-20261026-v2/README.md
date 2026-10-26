# CPU GPU 算力中心需求模式与柔性约束负荷报告 v2

日期：2026 年 10 月 26 日。[初始报告与生成程序](../task-driven-load-report-20261026/README.md)保留原文件。

- [完整报告 PDF](CPU_GPU算力中心_需求模式与柔性约束负荷报告_20261026_v2.pdf)：16 页、六组图。
- [可审阅正文](CPU_GPU算力中心_需求模式与柔性约束负荷报告_20261026_v2.md)。
- [来源波动汇总](demand-shape-summary.csv)、[统计结果](report-statistics.json)与[文件校验清单](report-manifest.json)。

## 需求模式与情境

| 需求模式 | 来源 | 使用边界 |
| --- | --- | --- |
| 科研超算型 | SURF Lisa | 以科研 CPU 作业为主；GPU 加速分区保留其标签和卡数缺口 |
| 大模型开发型 | AcmeTrace Kalos | 预训练、微调、评估和开发调试；原论文不包含线上部署服务 |
| 生产 AI 服务型 | Alibaba ASI | 训练、在线推理、离线推理、开发和优先级；当前本地核验样本为一天 |

设备构成与需求模式分别登记。用于研究比较的 CPU/GPU 参考 IT 容量配比分别为 CPU 主导 75/25、GPU 主导 25/75、均衡混合 50/50；这些是研究设计值，平均资源占用仍由记录累计。1、10、100、500 MW 定义总 IT 参考预算；GPU 整机包含宿主 CPU。

报告说明来源差异、连续周/月重放、混合中心共时关系、资源分母、共享冷却与未知业务权限。新增混合组合尚未运行；图中八条 24 小时功率示例保留其已有数值，年度部分提供构造与累计方法。

## 数据统计与文献

来源波动分别基于 Lisa 的 309 个完整日、Acme 的 90 个完整日和 ASI 相对日 109 的 24 小时。变异系数使用总体标准差，资源统计口径和观察长度分别说明；ASI 单日结果不解释为全年波动。

参考资料包含 Lisa 的 ICPADS 2024、Acme 的 NSDI 2024、ASI 的 OSDI 2026 原始论文。ASI 电力柔性结果另据 [arXiv:2609.05406v1](https://arxiv.org/abs/2609.05406v1)作为预印本文献对照，保留模型功率、候选权限、实现比例和持续可用性的限制。

## 复现与核验

在仓库根目录运行：

```powershell
python dc-flexibility-project/analysis/build_task_driven_load_report_v2.py
```

依赖与 Windows Microsoft YaHei 字体沿用初始报告。输入为 `data/processed/task-trace-audit-20261026/` 中已登记 SHA-256 的任务和小时文件。脚本重新累计报告统计，复用并校验初始报告的六组 PNG 和矢量 PDF 图件，生成正文、PDF 与文件清单；不重新运行科研负荷模型。

服务器项目目录为 `/data3/dataset/ljt/global-grid-flexibility-20261023`，本报告位于其 `results/task-driven-load-report-20261026-v2/`。算法说明见[任务记录文档](../../docs/task-trace-flexibility.md)与[通用负荷模块文档](../../docs/compute-load-module.md)。
