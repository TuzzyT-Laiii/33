# 第二部分任务容量与八城市供能复核

输入基于 PR #62 的 `01dc316a5e8678a5f097b07cc3ce582e507cf36b`。任务量、设备规格、价格、原生观测时间、八城市气象和供能参数沿用该批次。容量优化移除地点和能源目标；原全球 MIP 的已有记录继续保留。

## 容量问题与证明范围

本次主期限 11 个情境中有 10 个已证明最优。三档期限合计 33 项，29 项上下界一致，B-SAI 的 24／96 小时相差 1 台；A-HPC 三项仍未达到证明精度。`final.md` 列出当前有效区间，不能将 HPC 的可行上界记为最少台数证明。

原时序的 A 台数为小时 CPU 与 GPU 资源需求对应整机数的最大值向上取整。C 最小化整机台数，同时完成相同工作量。整体作业保留实际运行时长，按整数小时平移起始时间，启动后连续运行；同类作业可以选择不同起始时间。主情境的整体／可拆分推迟期限为 12／48 小时，敏感性为 24／96 和 48／168 小时。

同类作业的合并要求原始小时占用形状、CPU/GPU 资源数量和起止小时相同。整数变量记录选择各起始时间的作业数量，可以还原为原始作业。观测不完整、跨缺失时段以及不能完整放入观测窗口的作业保留原时序。缺失时段不增加执行变量，推迟期限按 `source_hour_index` 的真实小时差核验。

A-CLD 与 B/C/D-INF 缺少逐作业记录的可推迟工作量，按小时工作量在期限内转移，同时移动对应 CPU/GPU 资源与任务电量。A-CLD 只有 120 个观测小时，窗口末工作量清空影响期限敏感性；这一行用于展示模型边界。

服务器台数证明适用于冻结输入的小时资源投影。模型没有逐节点显存、内存、网络拓扑和逐秒资源装箱约束，因而这些台数不能直接解释为实际设备清单中已核实可撤走的设备数量。

独立资源 LP 允许整体作业拆分，分别求 CPU/GPU 的期限可行流，给出有效下界。进一步的作业模式 LP 保留连续运行形状，将起始时间的整数选择放松为连续数量。比例相同的资源形状可以在这一阶段聚合；完整起始候选的价格下界始终用原始同类作业重新计算。整数调度上界也还原至原始作业后核验。聚合 LP 的台数不会直接作为已实现的整数台数。

资源差分模型用逐小时占用量的变化表示作业首小时、中间持续占用及末小时。这个等价表示减少长作业重复写入的非零系数。真实 336 小时 HPC 窗口上的随机可行调度和最优 LP 值，与逐小时展开模型分别比较，证据见 `compact-model-verification.json`。各情境只有在整数上、下界相差不超过一台时才能列为达到证明精度；区间上下界相同表示已经证明最优。

B/C/D 同一用途的冻结资源曲线和逐作业资源形状，经参考台数归一化后相同。程序先验证全部小时、作业编号和形状，再按连续资源量映射上下界，最后分别向上取整；不会直接按比例缩放已经取整的台数。

Claude 的 `C_min/K` 使用任务功率峰值作为分母，设备 `C/A` 使用整机台数作为分母。结果同时列出两项比例，并把完整可行调度重新换算为任务功率峰值后核对功率下界。这两个比例不能直接作大小判定。

## 能源费用与节省区间

C 台数固定为容量阶段核验的可行台数，分别计算八城市的经济最优供能和 100% 本地无碳匹配。后者要求市电用量为零。风电、光伏、储能和电网接入的费用、寿命、效率与容量上限沿用冻结配置。

云计算和推理使用连续的小时工作量模型。整体作业情境同时保留：完整任务容量调度下的供能可行费用上界，以及允许整体作业拆分、忽略 CPU/GPU 资源限制的能源 LP 下界。两者形成固定 C 台数下能源最优费用的有效区间。原 MIP 中台数相同且通过输入版本和可行性检查的方案，可用于收紧上界；其来源单独记录。固定任务调度的供能子问题收敛，并不表示已经证明联合任务时序与能源规划的全局最优。

主表中的方案费用使用已经核验的可行上界，相对最优 A 的净节省因而是保守结果。另列净节省上界及能源区间宽度。八城市范围和求解不确定区间分别报告。B 到 C 的增量同时给出 B、C 费用下界与上界组合而成的区间。

原 MIP 对照保留原状态、可行台数、有效求解间隙和方案费用。没有有效下界的原结果不会用于证明“无法减配”。重复使用原可行方案收紧费用上界，与独立容量调度对原 MIP 的误差检验分别记录。尚未满足八城市精度条件时，全球 LP 补充计算不启动。

设备采购节省等于减少台数乘以冻结整机价格；GPU 减少量等于减少台数乘以每台 GPU 数量。设备年化采用冻结折现率、寿命及固定运维比例，系数为约 0.3319208。能源窗口费用乘以 `8760 / 观测小时数`。这是原生观测窗口的等效年化估算，不是全年实测现金流。

净年化节省 = 设备年化节省 + A 能源费用 − C 能源费用。占机群年化成本比例以 A 的设备年化成本为分母。B 到 C 的增量另用 B 能源费用替代 A 能源费用。最少容量策略下经济与无碳两行的设备数相同；全生命周期经济最优设备数可能略多，原 MIP 的 100% 无碳结果即包含这种权衡。

5% 和 10% 余量按最少可行台数再向上配置设备，单独计算设备费用。设备价格 0.5／1／2／3 倍情境保持相同容量策略及任务调度。两项敏感性没有重新宣称供能资产已按新增设备优化。等待分布和利用率明确标注所对应的可行任务调度；没有保存 B 逐作业时序时，逐作业等待指标留空。

## 运行与数据文件

计算服务器为 `pku-solar-pro`（192.168.2.204）。计算进程单线程，并发不超过四个，使用用户级 systemd 的 `MemoryMax`、`MemorySwapMax=0` 和虚拟内存限制。容量服务按模型规模分配不超过 8 GiB 物理内存上限；城市服务单独限额。运行不修改其他用户的进程。

`capacity/` 保存逐情境容量证书、原始同类作业的整数平移方案及资源调度；派生硬件情境注明基础资源模型。`lower-bounds/` 保存独立资源 LP 下界。`collected-cities/` 保存城市费用、来源、原 MIP 对照和容量调度指标。汇总表由 `analysis/report_minimum_server_capacity.py` 生成，输入不完整或台数证明精度不达要求时会直接报错。

本次使用显式参数 `--allow-open-bounds` 输出尚有 HPC 开放区间的研究记录，`capacity_proof_complete` 保留证明状态。这一参数不会把开放区间改为最优。汇总表与全部明细同时包含能源费用下界和可行上界，最大单城市区间宽度为 7.85 百万美元／年。

代码入口依次为：`joint_capacity_whole_jobs.py` 编译原始同类作业，`resource_capacity_lower_bound.py` 给出独立资源下界，`minimum_server_capacity.py` 与 `capacity_plan.py` 求解并还原容量方案，`bounded_city_energy.py`、`fixed_capacity_city_energy.py` 和 `collect_city_energy.py` 形成城市费用区间，`audit_capacity_results.py` 检查 33 项资源与硬件元数据，最后生成汇总表。差分压缩、完整模式价格核验、分段整数模型和整数修正分别保留独立模块与真实输入验证记录。

在项目根目录，已有冻结输入和容量、城市记录时，可运行：

```powershell
python dc-flexibility-project/analysis/report_minimum_server_capacity.py `
  --project dc-flexibility-project --resources INPUTS/resources `
  --capacities dc-flexibility-project/results/minimum-server-capacity-20261108/capacity `
  --cities dc-flexibility-project/results/minimum-server-capacity-20261108/collected-cities `
  --city-inputs INPUTS/cities `
  --claude-lower dc-flexibility-project/results/minimum-server-capacity-20261108/claude-power-lower-bounds.csv `
  --resource-lower dc-flexibility-project/results/minimum-server-capacity-20261108/lower-bounds/resource-lower-bounds.csv `
  --first-part dc-flexibility-project/results/flex-card-cost-calendar-20261102/runs-cards.csv `
  --output dc-flexibility-project/results/minimum-server-capacity-20261108 --allow-open-bounds
```

`INPUTS` 代表服务器保留的冻结输入目录。`resource-manifest.json` 也随结果交付。临时检查点和过期试算方案不纳入交付；全局旧 MIP 原始记录继续保留在原位置。

容量证书、城市记录和 CSV 按存档原字节保存，目录属性关闭自动换行转换，支持 Windows 与 Linux 文件 SHA-256 核对。`delivery-verification.json` 记录 22 行主表、33 项期限、176 项城市与功率核查、12 项 COPT／HiGHS 小时模型交叉检查。HPC 最后单进程分段任务在 17:39:48–17:46:43 运行，journal 记录 CPU 412.474 秒、内存峰值约 4.6G；区间保持 13,195–13,620，证据见 `hpc-final-segmented-service.json`。

代码需要 NumPy、pandas、PyArrow、SciPy、HiGHS 和 COPT，以及有效 COPT 许可。服务器本次使用独立 Python 3.12 环境，避免修改原 Conda 环境。完整原始逐作业输入保留在研究服务器；结果中的输入 SHA-256、编译清单和参数用于确认版本。
