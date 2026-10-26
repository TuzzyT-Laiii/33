# 实际设备配置与全年小时负荷

[PDF：实际配置与 8760 小时负荷计算说明](算力中心_实际配置与8760小时负荷计算说明_20261026_v2.pdf)按真实超算架构定义 CPU 计算型、GPU 加速型和混合计算型，说明三项基础情境维度、需求来源、全年重放、温湿度与冷却、柔性控制以及本次实际结果。

- [36 个设计单元](scenario-matrix.csv)：1、10、100、500 MW × 三类设备模式 × 科研/开发/生产需求；逐项注明已计算、诊断、演示和待补数据。CPU 池与 GPU 整机池分开核算，宿主 CPU 计入 GPU 整机。
- [实际硬件依据](hardware-evidence.json)：ARCHER2、Frontier、Perlmutter 与 Lisa。混合功率份额由对应节点清单及匹配整机功率推导，未识别的份额保留为空。
- [全年汇总](annual-load-summary.csv)：16 条 × 8760 小时，共 140160 条负荷记录，分别汇总资源小时、IT/冷却/设施电量、最高小时功率与外推标记。
- [相位敏感性](annual-phase-sensitivity.csv)：490 组完整日相位，在 100 MW 下检查需求与天气配对、年末截尾对电量和峰值的影响。
- [全年来源与校验清单](annualization-manifest.json)、[计算核验](validation.json)及[交付校验](manifest.json)。

基准输入为已核验 Lisa CPU 309 个完整日、Acme CPU/GPU 90 个完整日、ASI 相对日 109 的 24 小时。按完整连续窗口顺序循环到 2025 年，再使用网格 g05_r100_c592 中心 ERA5 节点全年天气计算冷却。短窗口重放属于全年构造；样本外季节性和需求增长未识别。Acme CPU 保留为开发样本诊断；ASI 单日仅作为扩展方法演示。

完整逐小时负荷、资源来源映射保存在 `data/processed/task-annual-loads-20261026-v2/g05_r100_c592-2025/`，随本地与服务器研究数据保存。仓库提交代码、图件及汇总。跨数据集资源到参考整机功率映射保留代理性质；公共网络和独立存储电耗须单独加入。

在项目根目录运行：

```powershell
python analysis/build_task_annual_loads.py --year 2025 --grid-id g05_r100_c592
python analysis/verify_task_annual_loads.py
python analysis/build_scenario_load_methods_note_v2.py
```

实际全年重放由 [annual_task_profiles.py](../../analysis/annual_task_profiles.py) 和 [build_task_annual_loads.py](../../analysis/build_task_annual_loads.py) 实现。可通过 `--weather-path` 指定同结构 ERA5 数据；其他网格/年份汇总保存在本目录 `annual-runs/` 下，逐小时结果按网格与年份独立保存。闰年按 8784 小时处理。

柔性权限作为独立控制设计，调度后重新计算资源、IT 和冷却。当前已实现固定路径年度重放；任务级全年复制和混合设备联合调度须满足文中输入要求。
