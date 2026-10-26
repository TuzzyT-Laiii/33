# CPU GPU 算力中心任务需求与柔性约束负荷报告

日期：2026 年 10 月 26 日。根据已核验的 Lisa、AcmeTrace Kalos 和 Alibaba ASI 数据新建；20261025-v4 报告保留原文件。

- [完整报告 PDF](CPU_GPU算力中心_任务需求与柔性约束负荷报告_20261026.pdf)：12 页，六组图。
- [可审阅正文](CPU_GPU算力中心_任务需求与柔性约束负荷报告_20261026.md)。
- [统计结果](report-statistics.json)与[文件校验清单](report-manifest.json)。

## 数据与计算范围

| 内容 | 输入与范围 | 本报告输出 |
| --- | --- | --- |
| CPU 作业 | Lisa 全部 1596963 条作业中的 CPU 分区；源时间约十个月 | 5 分钟、小时与日均分配核心需求；作业运行时长分布 |
| GPU 作业 | Acme Kalos 的 19907 条 GPU 作业，19904 条有完整执行区间 | 按预训练、评估、微调、调试等类型统计作业数与分配 GPU 小时 |
| GPU 运营构成 | ASI 相对日 109 的完整 24 小时 | 训练、在线推理、离线推理等类型的归属 GPU 小时和申请份额 |
| 参考用电 | Acme CPU 峰值日期、ASI 一天资源代理与真实网格天气 | CPU/GPU 各 1、10、100、500 MW IT 预算的 8 条 24 小时示例 |
| 柔性约束 | 记录中的任务类型、优先级、执行和排队信息 | 明确期限、暂停恢复与服务质量的证据要求；未知权限保留未知 |

功率采用既有整机功率锚点与 ESIF 温湿度响应。示例为跨数据集参考计算，不能视为来源机房实测功率；资源小时也不能直接解释为电量。当前报告给出完整年度的累计和重放条件，没有将一天样本扩充为实测全年。

## 图表和汇总

六组图分别为作业时长分布、Lisa CPU 日与周需求、Acme GPU 类型构成、ASI 小时类型构成、参考节点功率、参考设施功率。每组同时保存 300 dpi PNG 与矢量 PDF。

`acme-gpu-type-summary.csv` 和 `asi-gpu-type-summary.csv` 保留任务类型统计；`lisa-cpu-daily-core-demand.csv` 保存完整日期的资源需求；`power-24h-summary.csv` 保存八种参考规模的电量及峰值。带任务标识的逐作业记录存放于研究数据目录，不随报告提交。

## 复现与输入版本

在仓库根目录运行：

```powershell
python dc-flexibility-project/analysis/build_task_driven_load_report.py
```

生成脚本使用 Python、NumPy、pandas、PyArrow、matplotlib、Pillow、ReportLab 和 PyMuPDF，以及 Windows 的 Microsoft YaHei 字体路径。报告在 Windows 环境生成，服务器保存报告、输入和代码；更换运行平台时须提供相应字体路径。

输入位于 `data/processed/task-trace-audit-20261026/`，由[任务核验记录](../../data/metadata/task-trace-audit-20261026/asi-acme-validation.json)固定文件 SHA-256。脚本先核对输入校验值，再累计统计、绘图、生成正文与 PDF。`report-manifest.json` 固定生成脚本、验证记录及交付文件的校验值。

服务器项目目录为 `/data3/dataset/ljt/global-grid-flexibility-20261023`，报告位于其 `results/task-driven-load-report-20261026/`。算法接口及使用条件见[任务记录文档](../../docs/task-trace-flexibility.md)和[通用负荷模块文档](../../docs/compute-load-module.md)。
