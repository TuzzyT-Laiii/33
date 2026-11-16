# 《算力中心算电协同收益稳定但有限》结果图 1–3（英文版，2026-11-15）

对应 OneDrive《结果框架_算力中心算电协同收益稳定但有限_20261115_v15》的结论一至三；版式参照 GeoSciPlot 图库（https://geosciplot.zbhgis.com/）的多类型组图。三张图共用一套配色：五类用途色（同用途内 CPU 主导/均衡/GPU 主导由浅到深），数值色带米色→青绿→深蓝，发散色带赤陶红↔深青。子图只标字母，标题与说明见图注；情境标签分两级（左用途、右配置）。碳价图（图 4）待数据。

180 mm 双栏宽，高 215/205/195 mm；PDF（地图、热图、散点栅格化，其余矢量）与 600 dpi PNG。字体 Arial。

金额统一为 **年收益或年节省 ÷ 数据中心年用电量（不卖推理时的基准用电）**，单位 USD/MWh。

| 图 | 面板 | 对应子论点 |
| --- | --- | --- |
| [图1](fig1.pdf) | a 训练·GPU 主导协同节省地图 + 纬度条；b 不可推迟/可推迟功率与峰值余量；c 节省分布（山脊图）；d 现有协同 vs 减配上限；e 成本嵌套方块；f 节省 vs 市电占比（点色风电占比）；g 上下限分解：节省比例 × 成本基数（等节省额线），曲线上每点多减 2.5% 服务器 | 1.1 b；1.2 a、c、f；1.3 e；1.4 d、g |
| [图2](fig2.pdf) | a 现有协同、减配上限、真实需求下卖推理、全卖四档（对数轴）；b 盈亏平衡售价；c 日内热条（推理需求与三类机群闲置）；d 闲置 GPU 比例 × 每 MWh GPU 小时（等收益分区）；e 跨地点偏离（半边小提琴：协同 vs 全卖） | 2.1 a、b、c；2.2 d；2.3 e |
| [图3](fig3.pdf) | a 上限 75% 时节省地图 + 直方图色标；b 协同节省占“卖推理净收入 + 协同节省”（环形图）；c 争用与互补（占用闲置 vs 保留率，虚线为完全争用）；d 保留率 vs 风电占比；e 转移电量与每 MWh 价值变化 | 3.1 a、b；3.2 c、e；3.3 d |

`source-data/` 为各面板数据（UTF-8 BOM CSV）。

## 输入

- `results/residual-flex-global-20261115/paired.parquet`、`results/sell-inference-tiers-20261115/`（PR #77、#78）。
- `results/paper-figs-stable-limited-20261115/`：`baseline-energy.parquet`、`moved-energy.parquet`（由 `prepare_stable_limited_figs.py` 从第三节 runs-compact 生成）；`part1-summary.csv`（PR #60）；`downsizing-part2-main.csv`（PR #67 减配主表，只取与地点无关的设备年化节省与机群年化设备成本）。
- `results/global-flex-econ-20261103/grid-summary.parquet`（图 1d）；`results/flex-class-20261101/native-hourly/`（图 2c）。

## 口径与提示

1. 减配上限 = 按最少台数少买设备的年化节省（与地点无关）÷ 全球基准年用电中位数，未计供能侧的额外节省（占减配净节省 2%–4%）。
2. 图 2c 外部推理需求与推理机群自身负荷同为 ASI 曲线，反相关被同源放大。
3. 图 2e 跨地点偏离用年收益：全卖 ±0.2–0.7%，协同 −33% 至 +42%。按 USD/MWh 计全卖因各地 PUE 变为 ±5–7%。
4. 图 3c 科研 AI · GPU 主导（0.819）落在完全争用线（0.826）附近，其余情境均在线上方。
5. 框架核对：3.5 节（旧）50% 档应为 1.7–4.9%；“下十分位 2.3–9.0”对应强度 50%，75% 为 1.8–8.3；3.2 “转移电量不变”需计入离线推理。

## 复现

```powershell
# 地图底图（PR #63 map-assets）放到已忽略的 .codex-temp/
git archive origin/codex/paper-results1-figures-20261107 dc-flexibility-project/papers/demand-flexibility/figures/map-assets | tar -x -C dc-flexibility-project/.codex-temp/figs-en --strip-components=4
python dc-flexibility-project/analysis/prepare_stable_limited_figs.py --runs <第三节 runs-compact.parquet> --part1 <PR #60 part1-summary.csv>
cd dc-flexibility-project/analysis
python plot_stable_limited_fig1.py; python plot_stable_limited_fig2.py; python plot_stable_limited_fig3.py
```

需要 matplotlib、cartopy、geopandas、shapely、pandas、scipy、pyarrow。
