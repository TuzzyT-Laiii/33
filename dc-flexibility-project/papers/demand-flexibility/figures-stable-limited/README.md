# 《算力中心算电协同收益稳定但有限》结果图 1–3（英文版，2026-11-15）

对应 OneDrive《结果框架_算力中心算电协同收益稳定但有限_20261115_v15》的结论一至三；版式参照 GeoSciPlot 图库（https://geosciplot.zbhgis.com/）的多类型组图。三张图共用一套配色：五类用途色（同用途内 CPU 主导/均衡/GPU 主导由浅到深），数值色带米色→青绿→深蓝，发散色带赤陶红↔深青。子图只标字母，标题与说明见图注；情境标签分两级（左用途、右配置）。图 4 对应结论四（碳价与技术成本）。

180 mm 双栏宽，高 215/205/195 mm；PDF（地图、热图、散点栅格化，其余矢量）与 600 dpi PNG。字体 Arial。

金额统一为 **年收益或年节省 ÷ 数据中心年用电量（不卖推理时的基准用电）**，单位 USD/MWh。

| 图 | 面板 | 对应子论点 |
| --- | --- | --- |
| [图1](fig1.pdf) | a 训练·GPU 主导协同节省地图 + 纬度条；b 不可推迟/可推迟功率与峰值余量；c 节省分布（山脊图）；d 现有协同 vs 减配上限；e 成本嵌套方块；f 节省 vs 市电占比（点色风电占比）；g 上下限分解：节省比例 × 成本基数（等节省额线），曲线上每点多减 2.5% 服务器 | 1.1 b；1.2 a、c、f；1.3 e；1.4 d、g |
| [图2](fig2.pdf) | a 环形分组柱：现有协同、减配上限、真实需求下卖推理、全卖四档（径向对数）；b 盈亏平衡售价区间条，标出市场价带与市场价 ÷ 减配盈亏平衡价；c 2025-06-02 至 06-15 逐时曲线（闲置 GPU vs 同均值外部需求，红区为需求超过闲置）；d 闲置 GPU 比例 × 每 MWh GPU 小时（等收益分区）；e 电价梯度：协同节省偏离的六边形密度与中位线 vs 全卖收入，右侧放大全卖 | 2.1 a、b、c；2.2 d；2.3 e |
| [图3](fig3.pdf) | a 上限 75% 时每个网格 11 类机群中最低的协同节省（六边形格网，0–6 美元/MWh 每 0.5 一档离散色；海域内嵌直方图兼色标；下方为 14 个合并分区的箱线图，分区由联合国子区域合并，见 source-data/fig3a-regions.csv）；b 实施成本存活曲线（2、3 美元/MWh 处 97%、90%）；c 争用与互补（占用闲置 vs 保留率，线尾直接标注）；d 保留率分解：转移电量比 × 每 MWh 价值比，背景为等保留率线；e 训练机群保留率 vs 风电占比 + 11 类机群秩相关 | 3.1 a、b；3.2 c、d；3.3 e |
| [图4](fig4.pdf) | a 双变量地图：电网碳因子 × 碳价 0→200 放大倍数（Equal Earth，3×3 分位）；b 云雨图：推理 75% 时五类用途在碳价 0/100/200 下的节省分布；c 2035 乐观路径协同节省倍数地图 + 经度统计条（三条路径）；d 六边形密度：顺序调度储能时长 vs 协同节省倍数 | 4.1 a；4.2 b；4.4 c、d；4.3 见正文（八城市） |

`source-data/` 为各面板数据（UTF-8 BOM CSV）。

## 输入

- 图 4 使用第四节汇总表（PR #80 `results/sec4-carbon-learning-20261116/` 与 `analysis/learning-curves-atb2025.json`），原样复制到 `results/paper-figs-stable-limited-20261115/sec4/`；逐网格小表 `fig4-grid-carbon.parquet`、`fig4-grid-lc.parquet` 由 `plot_stable_limited_fig4.py` 的 prepare() 从 #80 服务器结果的本地副本提取。

- `results/residual-flex-global-20261115/paired.parquet`、`results/sell-inference-tiers-20261115/`（PR #77、#78）。
- `results/paper-figs-stable-limited-20261115/`：`baseline-energy.parquet`、`moved-energy.parquet`（由 `prepare_stable_limited_figs.py` 从第三节 runs-compact 生成）；`part1-summary.csv`（PR #60）；`downsizing-part2-main.csv`（PR #67 减配主表，只取与地点无关的设备年化节省与机群年化设备成本）。
- `results/global-flex-econ-20261103/grid-summary.parquet`（图 1d）；`results/flex-class-20261101/native-hourly/`（图 2c）。

- 附录图 S1（供能侧节省分项）使用 `results/supply-cost-decomposition-20261119/`（`analysis/supply_cost_decomposition.py` 由第三、四节服务器结果的本地压缩表生成），作图脚本 `plot_supply_cost_decomposition.py`。

## 口径与提示

1. 减配上限 = 按最少台数少买设备的年化节省（与地点无关）÷ 全球基准年用电中位数，未计供能侧的额外节省（占减配净节省 2%–4%）。
2. 图 2c 外部推理需求与推理机群自身负荷同为 ASI 曲线，日内错位被同源放大；两周窗口按训练曲线缺测最少选取；训练的 78% 来自任务推迟，曲线为挪动前的原始闲置。
3. 图 2e 横轴电价 = 100 美元/MWh + 碳价 100 美元/t × 当地碳因子；跨地点偏离用年收益：全卖 ±0.2–0.7%，协同 −33% 至 +42%。按 USD/MWh 计全卖因各地 PUE 变为 ±5–7%。
4. 图 3c 科研 AI · GPU 主导（0.819）落在完全争用线（0.826）附近，其余情境均在线上方。
5. 框架核对：3.5 节（旧）50% 档应为 1.7–4.9%；“下十分位 2.3–9.0”对应强度 50%，75% 为 1.8–8.3；3.2 “转移电量不变”需计入离线推理。

## 复现

```powershell
# 地图底图（PR #63 map-assets）放到已忽略的 .codex-temp/
git archive origin/codex/paper-results1-figures-20261107 dc-flexibility-project/papers/demand-flexibility/figures/map-assets | tar -x -C dc-flexibility-project/.codex-temp/figs-en --strip-components=4
python dc-flexibility-project/analysis/prepare_stable_limited_figs.py --runs <第三节 runs-compact.parquet> --part1 <PR #60 part1-summary.csv>
cd dc-flexibility-project/analysis
python plot_stable_limited_fig1.py; python plot_stable_limited_fig2.py; python plot_stable_limited_fig3.py; python plot_stable_limited_fig4.py
python plot_supply_cost_decomposition.py   # 附录图 S1
```

需要 matplotlib、cartopy、geopandas、shapely、pandas、scipy、pyarrow。
