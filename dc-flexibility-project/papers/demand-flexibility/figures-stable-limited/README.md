# 《算力中心算电协同收益稳定但有限》结果图 1–3（英文版，2026-11-15）

按 OneDrive《结果框架_算力中心算电协同收益稳定但有限_20261115_v2》的图 1–3 绘制，版式参照 GeoSciPlot 图库（https://geosciplot.zbhgis.com/）的多类型组图：三张图之间不重复图型，地图分别用 Robinson + 经纬度统计条、六边形格网、Mollweide 双变量三种形式。子图只标字母，标题与说明放在图注；情境标签分两级（左侧用途、右侧配置）。图 4（碳价、学习曲线）数据未算完，暂不画。

180 mm 双栏宽，高 205/200/215 mm；PDF（地图与热图栅格化，其余矢量）与 600 dpi PNG。字体 Arial；五类用途配色与 PR #63/#70 一致。

金额统一为 **年收益或年节省 ÷ 数据中心年用电量（不卖推理时的基准用电）**，单位 USD/MWh。窗口内节省与窗口内基准用电直接相除。

| 图 | 面板（图型） | 对应子论点 | 数据 |
| --- | --- | --- | --- |
| [图1](fig1.pdf) | a 训练·GPU-heavy 协同节省地图 + 经度、纬度统计条；b 峰值余量（环形柱图）；c 节省分布（山脊图）；d 两种分母（斜率图）；e 市电占比 × 风电占比归因（六边形分箱） | 1.1 b；1.2 a、c、e；1.3 d | a–c、e：第三节 I=0（`results/residual-flex-global-20261115/paired.parquet`、`baseline-energy.parquet`）；b：PR #60 part1-summary；d：第一节原生窗口 `results/global-flex-econ-20261103/grid-summary.parquet` |
| [图2](fig2.pdf) | a 9 个 GPU 情境 × 6 档收益（气泡矩阵）；b 真实上限/理想（半环）；c 逐时闲置（日历热图）；d 地点偏离（六边形格网地图）；e 峰谷计价 vs 地点离散度（散点） | 2.1 a；2.2 b、c；2.3 d、e | `results/sell-inference-tiers-20261115/`、`max-intensity.csv`、`results/flex-class-20261101/native-hourly/` |
| [图3](fig3.pdf) | a 保留率曲线；b 保留率 × 风电占比（Mollweide 双变量地图）；c 转移电量 vs 每 MWh 价值（箭头图）；d 离线推理可推迟、需求波动（三城市注释热图）；e 强度 75% 节省（累积分布）；f 协同节省/卖推理净收入（分组柱） | 3.1 e；3.2 a、c；3.3 b；3.4 d；3.5 f | `paired.parquet`、`moved-energy.parquet`、`results/residual-flex-pilot-20261114/`、`sell-inference-global.parquet` |

`source-data/` 为各面板数据（UTF-8 BOM CSV）。

## 与框架 v2 的核对

全部重算，与框架一致的：节省 3.5–15.4 USD/MWh、供能成本 103–116 USD/MWh、占供能成本 3.3–13.8%、占总成本中位数 0.10–0.94%（单格最高 1.68%）、峰值余量 27–52%；卖推理理想 934–2,922、真实上限 197–1,240 USD/MWh，真实/理想 训练 78%、科研 AI 23–28%、推理 15–18%；峰谷计价 −2.4 至 −2.6%；强度 75% 保留率中位数 训练 0.54–0.55、云 0.56、HPC 0.82、科研 AI 0.82–0.86、推理 0.98–1.01；训练保留率与风电占比 Spearman 0.77–0.79；协同节省/卖推理净收入 强度 75% 为 0.8–3.0%；转移电量（原任务 + 离线推理）在强度 75% 时为 I=0 的 0.95–1.08 倍。

需要注意：

1. **框架 3.5 “上限 50% 时占 2.0%–4.9%”应为 1.7%–4.9%**（C-TRN 1.68%、B-TRN 1.95%）。
2. **“全球各网格之间相差不到 1%”只对年净收入（美元）成立**：P10–P90 离散度 0.3–1.4%。换成 USD/MWh 后，分母（含制冷的设施用电，随各地 PUE 变化）使离散度变为约 11%，所以图 2d 用年净收入的偏离作图。
3. 强度 75% 时 213,048 对全部为正，但全局最小值只有 0.03 USD/MWh（云计算，120 h 窗口）；框架“下十分位 2.3–9.0 USD/MWh”对应强度 50%（2.30–9.09），强度 75% 为 1.75–8.26。
4. 3.2 “转移的电量基本不变”只在把离线推理的转移电量计入时成立；只看原任务，训练机群在强度 75% 时降到 0.82–0.83 倍。
5. 3.4 只有三城市 × 训练机群试算（需求强度以平均闲置为 1，与全球以上限为 1 不同）：离线推理不可推迟时节省减少 0–11%（利雅得 GPU-heavy 强度 0.75 为 10.9%）；波动放大一倍保留率变化 −1 至 +9 个百分点。框架“推理机群最多 14%”在现有结果里找不到出处。
6. 1.2 归因：训练机群节省与不做协同时的市电占比 ρ = −0.81，与风电装机占比 ρ = 0.61。

口径说明：图 1 的 USD/MWh 面板来自第三节 I=0，科研 AI 与 HPC 只用与 ASI 推理重叠的窗口（4,320/4,440 h），其余情境与第一节窗口相同；图 1d 的百分比用第一节原生窗口。图 2c 的闲置按任务功率计（K − 原任务功率），与可承接上限口径一致。地图灰色为没有合格 0.5° 网格的 2° 区块。

## 复现

```powershell
# 1) 地图底图（PR #63 已提交的地图资产）放到已忽略的 .codex-temp/
git archive origin/codex/paper-results1-figures-20261107 dc-flexibility-project/papers/demand-flexibility/figures/map-assets | tar -x -C dc-flexibility-project/.codex-temp/figs-en --strip-components=4
# 2) 小体量输入（已提交在 results/paper-figs-stable-limited-20261115/，仅在重建时需要）
python dc-flexibility-project/analysis/prepare_stable_limited_figs.py --runs <第三节 runs-compact.parquet> --part1 <PR #60 part1-summary.csv>
# 3) 作图
cd dc-flexibility-project/analysis
python plot_stable_limited_fig1.py; python plot_stable_limited_fig2.py; python plot_stable_limited_fig3.py
```

`runs-compact.parquet` 为服务器 `/data3/dataset/ljt/residual-flex-global-20261114/full` 的压缩汇总（本机 `D:/codex-runs/residual-flex-pilot-20261114/global/`）。需要 matplotlib、cartopy、geopandas、pandas、scipy、pyarrow。
