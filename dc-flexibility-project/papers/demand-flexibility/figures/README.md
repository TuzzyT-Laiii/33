# Results 第一部分图件

两张主图的绘图、图注草稿和源数据。仅复用既有求解结果，没有修改或重新执行优化模型。

运行位置为 `dc-flexibility-project/`，安装 `numpy pandas pyarrow matplotlib geopandas shapely cartopy PyMuPDF Pillow` 后执行：

```powershell
python analysis/plot_paper_fig1.py
python analysis/plot_paper_fig2.py
python analysis/validate_paper_results1_figures.py
```

无需服务器即可重建。`map-assets/` 保存从研究服务器运行目录复制的两份原始地图依赖，其 SHA-256 已与服务器逐字节核对，校验值见 `validation.json`。不在运行时下载国界数据。

输出为 180 mm 宽、600 dpi PNG 及嵌入 Arial 的 PDF。Fig. 1 高 160 mm，Fig. 2 高 154 mm。地图图层在 PDF 中栅格化；其余图元、文字及 Fig. 2 散点保留矢量。中间预览在项目内已忽略的 `.codex-temp/paper-results1/`。

## 源数据

| 文件 | 内容 |
| --- | --- |
| `source-data/fig1a.csv` | 4,842 个有效网格的坐标、区块、情境与能源侧收益 |
| `source-data/fig1a_continents.csv` | 大洲网格数与未加权收益中位数 |
| `source-data/fig1a_geometry_audit.csv` | 逐网格地图绘制标志 `map_drawn` |
| `source-data/fig1b.csv` | 11 情境的原生窗口、任务峰值 K、平均负荷比例、余量及 >90% K 小时数 |
| `source-data/fig1c.csv` | 11 情境 × 两种边界的成本和收益中位数，金额单位为美元／年 |
| `source-data/fig1d.csv` | 106,524 个经济最优配对的窗口成本、等效年化节省与机群成本比值 |
| `source-data/fig1d_summary.csv` | 箱线图的样本数、最小值、四分位数、中位数与最大值 |
| `source-data/fig2a.csv` | 五代表情境 × 九风电分档的样本数、中位数和四分位数 |
| `source-data/fig2a_gridvalues.csv` | 五代表情境的逐网格风电出力系数、收益及分档编号 |
| `source-data/fig2b.csv` | 训练情境逐网格市电比例、风电出力系数和收益；同时提供 0–1 与百分比的市电比例 |
| `source-data/fig2c.csv` | 80 个回归系数及原回归表的 HC1 t 值、删变量 R² 降幅、拟合 R²、样本数和加粗标志 |
| `source-data/fig1_metadata.json`、`fig2_metadata.json` | 图形与统计口径 |
| `validation.json` | 输入 SHA-256、数字复核、源数据一致性及文件格式检查 |

`fixed` 在图中标为 **As-built**；`alpha` 为空的配对表示经济最优，不施加额外本地无碳匹配比例。每个情境和边界有 4,842 个有效最优配对，统计量对网格等权，没有按面积或实际数据中心规模加权。所有颜色跨图一致：科研计算紫色、云通用赭色、科研 AI 蓝色、训练绿色、推理橙色。

## 口径与证据范围

- K 为调度前原生窗口任务功率最大值，分母不包含空闲 IT 和冷却功率；这是本研究容量约束的定义，不能据此推断实际机群物理库存或操作权限。`fig1b.csv` 同时提供观察时长和 >90% K 小时数，绝对小时数须结合各自窗口阅读。
- 机群年化费率精确采用 `0.08×1.08⁴/(1.08⁴−1)+0.03`；能源成本和节省乘以 `8760/原生窗口小时数`。后者是等效年化，不是完整年度负荷仿真。各网格先计算比例，再取中位数；分位数中位数不能用两个独立中位数的比值替代。
- 规划边界可分别优化两侧供能容量；已建成边界固定无柔性基准的能源资产。机群在两侧相同，没有新增任务容量。无柔性方案的市电比例来自原归因表，与每一网格、情境配对。
- `map-assets/map-block-land-cells.parquet` 是原运行缓存，WGS84，经纬度 2° 区块与 Natural Earth 1:10m admin-0 陆地交集。原始 Natural Earth shapefile 包位于 `map-assets/countries-10m.zip`。地图用 Robinson 投影，不进行空间插值。没有有效配对的陆地为灰色。
- 10 个有效网格的原缓存几何为空。已用同一国界数据重新求交，仍为空；地图保留空白，全部 4,842 个网格继续进入统计。10 个网格 ID 及绘制标志见 `fig1a_geometry_audit.csv`。地图范围是代表网格所在区块，不能用于宣称全球每个区块都经过求解。
- 回归复用 #51 的外生变量口径：风光年均出力系数、市电碳强度、风光小时相关、风电逐日变异系数、光伏月均最大／最小比、年均气温及纬度绝对值。特征为 2025 全年，负荷按各自原生窗口。只标准化自变量，系数为每一个标准差对应的收益百分点。粗体表示删变量后 R² 降幅至少 0.02；不用于显著性或因果判断，也未处理空间自相关。

## 输入版本与审阅关系

从 `origin/main` 的 `12a5241` 新建任务分支，合入 #60 的 `ab72545` 和 #51 的 `78abdb9`，保留其祖先依赖。#51 链包含 #47–#50；建议先审阅该依赖链与 #60，再审阅本次新增的 `plot_paper_fig*.py`、共享绘图模块、验证脚本和本目录。依赖分支原有文档／图表随历史合入，本任务未继续修改这些文件。
