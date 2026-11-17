# 既有闲置 GPU 出租结果的导入与对照

依赖 [容量与八城市复核 PR #65](https://github.com/TuzzyT-Laiii/33/pull/65)。遵循 [PR #62 中 Claude 的最终指示](https://github.com/TuzzyT-Laiii/33/pull/62#issuecomment-6719112086)，引用其在 `/data3/dataset/ljt/claude-sell-idle-20261108/` 已完成的计算。本次没有重新运行出租调度或能源求解。

## 输入与核验

`source/` 保留 Claude 提供的逐城市表和中位数表。`source-audit.json` 记录原始 SHA-256、72 项城市记录、1,080 个价格与可售比例组合，以及 `dispatch_recomputed=false`。导入程序逐项核对 A 的 GPU 总量、设备年化费用、冻结市电碳因子、C 的新最少台数、剩余 GPU 小时、全部收入恒等式及来源中位数。

B→A 与 C→B 的对照引用 #65 的当前费用区间，设备节省采用新容量结果。九个 GPU 情境的主期限台数均已证明最优。`part2-main-with-resale.csv` 为 22 行第二部分主表增加出租列；出租适用的九个经济目标行填入结果。纯 CPU 与 100% 本地无碳目标的出租列留空。

## 计算口径

出租对照 D 保持 A 机群、原任务时序和 A 的供能资产。可售 GPU 小时来自 A 的 GPU 总数减去逐小时已分配 GPU 小时。参数为价格 0.5／1.0／1.5／2.2／2.9 美元/GPU·小时、可售比例 25%／50%／75%。金额按原生观测窗口乘 `8760/观测小时数` 等效年化。

采用 Claude 最终结果的平均电耗参数：窗口任务动态功率之和除以任务 GPU 小时之和，得到每已分配 GPU 小时对应的 IT 电耗；额外设施用电乘逐小时 PUE，按 A 原时序闲置 GPU 小时加权。额外电费与碳费采用市电 100 美元/MWh＋碳价 100 美元/t × 当地碳因子。净收入为可售 GPU 小时 × 可售比例 ×（价格－每售出 GPU 小时能源费用）。该平均电耗参数属于情景假设，未测量各推理服务的实际满载功耗。

2.9 美元是 2026-11-08 查验的 [DeepInfra 官方 GPU 价格页](https://deepinfra.com/pricing)中 H100 80GB 的公开零售价。2.2 美元是 [UsagePricing 的 2026-07-14 第三方历史记录](https://www.usagepricing.com/blueprint/activity/deepinfra-2026-07-14-gpu-hour-token-hike)。两项作为价格情景，运营方实现的转售价需要单独估计。

## 八城市对照

金额为万美元／年，均为八城市中位数；出租比例为 50%。完整城市范围、B/C 费用最优性区间和全部 15 个价格比例组合保留在 CSV。

|情境|B→A 节省|C→B 增量节省|D 收入 p=1|占 A 设备年化费用|D 收入 p=2.2|占 A 设备年化费用|
|---|---:|---:|---:|---:|---:|---:|
|B-SAI|64.2|2,509.6|1,627.2|10.32%|3,740.7|23.73%|
|C-SAI|55.8|2,817.8|2,400.6|13.50%|5,419.5|30.48%|
|D-SAI|35.9|2,298.5|1,666.2|11.62%|3,752.6|26.18%|
|B-TRN|60.6|3,356.1|1,137.2|7.79%|2,594.9|17.77%|
|C-TRN|74.7|3,760.8|1,641.4|9.97%|3,723.5|22.61%|
|D-TRN|94.8|3,044.3|1,083.0|8.16%|2,522.9|19.00%|
|B-INF|16.0|1,477.0|1,054.9|6.69%|2,362.4|14.98%|
|C-INF|16.0|1,660.3|1,521.4|8.56%|3,389.1|19.06%|
|D-INF|17.3|1,345.0|1,037.6|7.24%|2,328.5|16.24%|

上述 B→A 和 C→B 点值比较保存的可行供能方案；最优费用区间在 `B-C-D-city-comparison.csv` 中另列。D 只在既有 A 经济供能基础上估算额外市电与碳费用，没有给新增出租用电重新施加 100% 本地无碳匹配。

D、B、C 作为互斥对照比较，收入不能直接相加：同一空档售出后会占用供柔性调度使用的资源；C 减少设备也会减少可售空档。`CD_extra_annual_usd` 单列减配后剩余总 GPU 小时的出租上限情景，使用 C GPU 总小时减去原有 GPU 工作量，并沿用 A 的平均能源扣除参数。该项没有为 C 的具体空档重新匹配出租任务。

未建模推理市场需求、价格形成、显存装箱、模型加载、逐卡碎片、抢占许可及新增电网接入容量。可售比例是上述限制的情景参数。原始任务分配的 GPU 小时不构成已核实可出租的逐卡清单。

## 复现导入

```powershell
python dc-flexibility-project/analysis/integrate_idle_gpu_resale.py `
  --project dc-flexibility-project `
  --source dc-flexibility-project/results/idle-gpu-resale-import-20261108/source/sell-idle-city.csv `
  --source-summary dc-flexibility-project/results/idle-gpu-resale-import-20261108/source/sell-idle-summary-median.csv `
  --part2 dc-flexibility-project/results/minimum-server-capacity-20261108 `
  --output dc-flexibility-project/results/idle-gpu-resale-import-20261108
```

导入仅读取和核验既有结果，不调用调度求解器。原始表按原字节保存，避免换行转换改变来源指纹。
