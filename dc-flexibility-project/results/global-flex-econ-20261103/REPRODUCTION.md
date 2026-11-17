# 全球网格经济优化复现材料

同一运行批次的复现输入与详细结果存放于项目百度网盘目录 `/科研数据/数据中心研究/20261019/global-flex-econ-20261103/`。公开仓库提供[方法与验证](METHODS.md)、[逐格汇总](grid-summary.parquet)和[来源记录](summary-manifest.json)。

| 文件 | 内容 |
| --- | --- |
| `g05_r101_c205-D-TRN-reproduction.zip` | 阿什本代表网格 D-TRN 原始任务模板、网格输入、合并后的逐时输入、两条原规划结果及对应审计和运行配置 |
| `runs-full.parquet` | 最终全量结果宽表，1,293,600 条记录，含每个算例的状态、成本分项及容量 |
| `hourly-and-audits-10-grids.zip` | 原运行保存的 10 个代表网格的 240 份逐时调度与 240 份审计记录 |
| `report-delivery.tar.gz` | 原运行的报告、汇总表、图件及来源记录 |

## 阿什本代表网格 D-TRN

网格 `g05_r101_c205` 的中心为北纬 39.0°、西经 77.5°。包内两份原输入的 SHA-256 与[模板索引](provenance/template-index.csv)及[网格索引](provenance/grid-input-index-all.csv)一致。

| 文件 | SHA-256 |
| --- | --- |
| `inputs/templates/D-TRN.parquet` | `01b907b8a12205931a6c3a4f6647beacbab81a0087f0c64eb47939a553b62a57` |
| `inputs/grids/g05_r101_c205.npz` | `841690586c16d276b13d5fc5effb5d85581539c9f4215a7cbbeb49a167df7699` |

`hourly-inputs.parquet` 保留 2,057 条原生小时记录及原顺序，覆盖 2025-05-16 10:00 至 2025-08-17 23:00 UTC，按原数据保留观测间隔。机房总负荷 `facility_mw` 取自 `aNone-planning-none` 的原始逐时结果，含冷却用电，单位为 MW。光伏与风电容量因子分别从网格文件中的 `cf_pv`、`cf_wind` 按模板的 `weather_hour_index` 提取。

固定原任务时间重算禁用经营性储能后的成本时，以 `aNone-planning-none` 为基线。沿用 `alpha=None`、`whole_h=split_h=0`、零新增算力和同一批小时输入，使用包内冻结配置，仅将 `battery_power_max_mw` 与 `battery_energy_max_mwh` 设为零，继续优化风电、光伏及市电接入容量。比较两次求解的 `total_cost_usd`。固定资产成本按 2,057/8,760 折算，购电与碳成本按观测窗口核算。原运行的成本、容量及调度边界见[模型实现](../../analysis/flex_econ_copt.py)。
