# 论文结果部分中文主图

图1–图4均为 180 mm 双栏宽，分别高 162、150、174、188 mm；提供 PDF 与 600 dpi PNG。中文使用本机微软雅黑，数字及拉丁字母使用 Arial；正文与说明为 6–7 pt，小写粗体面板字母为 8 pt。PDF 中仅图1地图层包含栅格图像，其余文字、曲线、散点、箱线、条形与色标保留矢量。

| 图件 | 面板 | 文件 |
| --- | --- | --- |
| 图1 | a 全球地图及大洲中位数；b 峰值余量；c 成本量级；d 相对机群费用分布 | [PDF](fig1.pdf) · [PNG](fig1.png) · [图注](fig1_legend.md) |
| 图2 | a 风电出力系数分档；b 市电占比与训练节省；c 回归系数 | [PDF](fig2.pdf) · [PNG](fig2.png) · [图注](fig2_legend.md) |
| 图3 | a 主期限11情境减配率；b 期限敏感性；c 等待与占用率；d 冗余敏感性 | [PDF](fig3.pdf) · [PNG](fig3.png) · [图注](fig3_legend.md) |
| 图4 | a 设备与能源节省；b 设备价格敏感性；c 出售空闲算力 | [PDF](fig4.pdf) · [PNG](fig4.png) · [图注](fig4_legend.md) |

`source-data/figNa.csv` 等十四个 CSV 对应全部主面板，补充 CSV 保留逐城市、逐网格、箱线摘要、持平价格与报价出处。CSV 使用 UTF-8 BOM，`scenario` 保留英文情境标识，同时添加中文 `情境` 列。`figN_metadata.json` 说明编码和科学边界，`validation.json` 记录输入 SHA-256、来源提交、数值核对和导出检查。

## 输入版本与复现

本分支由 `origin/main` 的 `12a52412fdeb9aa3473758889a6138a226c46d25` 创建，通过读取以下依赖的文件完成绘图，未合并上游 PR。

| 依赖 | 输入版本 | 使用内容 |
| --- | --- | --- |
| [PR #60](https://github.com/TuzzyT-Laiii/33/pull/60) | `ab7254543184ba1e55a4de5b26174e57e3099a9e` | 第一部分机群费用比例中位数 |
| [PR #63](https://github.com/TuzzyT-Laiii/33/pull/63) | `312022e2adb023d04e35e139e646e606f5f256f4` | 英文版逐面板数据与地图缓存 |
| [PR #65](https://github.com/TuzzyT-Laiii/33/pull/65) | `25a494f0054974cecd81381ffcc88d56792b94fa` | 主表、期限、等待、设备冗余及价格敏感性 |
| [PR #67](https://github.com/TuzzyT-Laiii/33/pull/67) | `08abb1fc96942e341a33ca31728f54ecbc80ce21` | 既有空闲GPU出售结果 |

需要 Python、matplotlib、pandas、numpy、pyarrow、geopandas、cartopy、PyMuPDF 和 Pillow，并安装 Arial 与 Noto Sans CJK SC 或微软雅黑。绘图读取已完成的科研计算，未启动优化求解。

尚未合并依赖时，可在仓库根目录运行下列 PowerShell 命令，按固定提交读取所需输入。存档放在已忽略的 `.codex-temp/`。所读入的依赖文件用于本地复现，不属于本 PR 新增内容。

```powershell
git fetch origin
New-Item -ItemType Directory -Force .codex-temp | Out-Null
git archive --format=tar --output=.codex-temp/fig-zh-input63.tar 312022e2adb023d04e35e139e646e606f5f256f4 dc-flexibility-project/papers/demand-flexibility/figures
tar -xf .codex-temp/fig-zh-input63.tar
git archive --format=tar --output=.codex-temp/fig-zh-input65.tar 25a494f0054974cecd81381ffcc88d56792b94fa dc-flexibility-project/results/minimum-server-capacity-20261108
tar -xf .codex-temp/fig-zh-input65.tar
git archive --format=tar --output=.codex-temp/fig-zh-input67.tar 08abb1fc96942e341a33ca31728f54ecbc80ce21 dc-flexibility-project/results/idle-gpu-resale-import-20261108
tar -xf .codex-temp/fig-zh-input67.tar
git archive --format=tar --output=.codex-temp/fig-zh-input60.tar ab7254543184ba1e55a4de5b26174e57e3099a9e dc-flexibility-project/results/global-flex-econ-20261103/part1
tar -xf .codex-temp/fig-zh-input60.tar
python dc-flexibility-project/analysis/plot_paper_fig1_zh.py
python dc-flexibility-project/analysis/plot_paper_fig2_zh.py
python dc-flexibility-project/analysis/plot_paper_fig3_zh.py
python dc-flexibility-project/analysis/plot_paper_fig4_zh.py
python dc-flexibility-project/analysis/validate_paper_figures_zh.py
```

图1、图2与英文版全部数值列逐项一致。图3、图4采用的八城市可行费用、科研计算区间、全局中位数与出租情景边界见各图注及根目录 `final.md`。
