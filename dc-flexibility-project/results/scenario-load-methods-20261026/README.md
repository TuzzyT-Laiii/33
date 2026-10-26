# 算力中心情境设计与负荷计算说明

2026 年 10 月 26 日。按规模、数据中心模式、需求导向组织的简明计算说明；详细分析保留在[需求模式报告 v2](../task-driven-load-report-20261026-v2/README.md)。

- [PDF 文档](算力中心_情境设计与负荷计算说明_20261026.pdf)：6 页。
- [可审阅正文](算力中心_情境设计与负荷计算说明_20261026.md)。
- [36 组情境表 CSV](scenario-matrix.csv)与[JSON](scenario-matrix.json)。
- [生成文件与参数校验值](manifest.json)。

设计为 1/10/100/500 MW × CPU 主导/GPU 主导/均衡 × 科研超算/大模型开发/生产 AI 服务，每个网格、每年 36 组。CPU 容量份额为 75%/25%/50%，均为研究比较点。外生气象输入为当地逐小时温度与相对湿度或露点；资源占用由有来源的需求序列累计。

文档逐步说明资源累计、容量分配、节点取整、IT 功率、散热、共享冷却和全年电量，并区分已核验输入与缺少的科研 GPU、生产独立 CPU 等资料。36 组为实验设计，尚未运行其联合负荷计算；情境表只列容量和参考设备数量。

柔性权限在第 5 节独立定义为控制设计，说明允许的操作、求解变量和服务约束，与基础情境三项维度分别记录。外生输入为气象；文档围绕需求、设备与负荷计算组织。

复现命令在仓库根目录运行：

```powershell
python dc-flexibility-project/analysis/build_scenario_load_methods_note.py
```

生成依赖为 Python、pandas、PyArrow、ReportLab、PyMuPDF 及现有负荷模块依赖；字体沿用 Windows Microsoft YaHei。图形化版面在 Windows 生成，服务器保存文档和代码。来源及参数见文档第 6 节与[模块说明](../../docs/compute-load-module.md)。
