# 论文结果部分中文主图交付

四张主图及其中文图注、面板源数据和复现脚本已完成。图件目录为 [`dc-flexibility-project/papers/demand-flexibility/figures-zh/`](dc-flexibility-project/papers/demand-flexibility/figures-zh/README.md)。本次交付仅包括图件、图注和核验说明。

## 图件与各面板

| 图件 | 面板内容 | 文件 |
| --- | --- | --- |
| 图1：机群余量充足，但相对算力成本价值很小 | a GPU主导型训练规划阶段经济最优的全球能源节省地图与大洲中位数；b 11情境峰值余量；c 机群、无柔性能源费用及两种边界节省的对数量级；d 节省占机群费用的网格分布、最大值与3%参考线 | [PDF](dc-flexibility-project/papers/demand-flexibility/figures-zh/fig1.pdf) · [PNG](dc-flexibility-project/papers/demand-flexibility/figures-zh/fig1.png) · [图注](dc-flexibility-project/papers/demand-flexibility/figures-zh/fig1_legend.md) |
| 图2：风电资源与市电依赖决定柔性在哪里有价值 | a 五类用途的风电出力系数分档中位数；b 训练节省与无柔性市电占比，颜色为风电出力系数；c 两种边界、五类用途的八自变量回归系数与R² | [PDF](dc-flexibility-project/papers/demand-flexibility/figures-zh/fig2.pdf) · [PNG](dc-flexibility-project/papers/demand-flexibility/figures-zh/fig2.png) · [图注](dc-flexibility-project/papers/demand-flexibility/figures-zh/fig2_legend.md) |
| 图3：机群规模参与规划时可少配的服务器 | a 11情境12/48小时减配率、GPU或服务器少配数及科研计算区间；b 三档推迟期限；c 可推迟电量加权平均推迟、资源占用率及设备节省；d 0/5/10%冗余下的设备节省 | [PDF](dc-flexibility-project/papers/demand-flexibility/figures-zh/fig3.pdf) · [PNG](dc-flexibility-project/papers/demand-flexibility/figures-zh/fig3.png) · [图注](dc-flexibility-project/papers/demand-flexibility/figures-zh/fig3_legend.md) |
| 图4：空闲容量的三种用法 | a 联合规划设备与能源节省的堆叠条形及第一部分仅调度比例；b 设备价格倍数下的净节省金额与仅调度比例；c 三类GPU主导用途的出售空闲算力收入、可售比例阴影、设备节省持平价与公开零售价 | [PDF](dc-flexibility-project/papers/demand-flexibility/figures-zh/fig4.pdf) · [PNG](dc-flexibility-project/papers/demand-flexibility/figures-zh/fig4.png) · [图注](dc-flexibility-project/papers/demand-flexibility/figures-zh/fig4_legend.md) |

四张图宽度均为180 mm；微软雅黑承担中文，Arial承担数字和拉丁字母，正文6–7 pt，面板字母为8 pt粗体小写。五类用途颜色在四图间一致，三类CPU/GPU配置按由浅至深编码。PDF只将地图层栅格化，其余图形和文字为矢量；PNG为600 dpi。十四个主面板均提供独立CSV，并保留逐网格、逐城市、箱线统计和报价出处等补充数据。

## 数字核对

以下数值从实际绘图输入和导出的源数据核对，完整精度及输入SHA-256见 [`validation.json`](dc-flexibility-project/papers/demand-flexibility/figures-zh/validation.json)。图1、图2的全部数值列与PR #63英文版逐项一致。

| 必须核对的项目 | 实际数值与图中表达 | 核对结果 |
| --- | --- | --- |
| 峰值余量27–52% | 27.3298–51.9547%；图1显示27–52% | 一致 |
| 规划能源节省中位3.3–13.8% | 3.3292–13.8393% | 一致 |
| 已建成能源节省中位2.2–7.3% | 2.1710–7.3228% | 一致 |
| 占机群成本最高1.87% | 1.866641% | 一致 |
| 风电首末档9.4%→20.4% | GPU主导型训练9.3598%→20.3526% | 一致 |
| 市电依赖相关系数−0.88 | −0.8772819 | 一致 |
| 训练减配22.6% | CPU主导型22.6433%、均衡型22.6338%、GPU主导型22.6597%；图3a分别显示22.6%、22.6%、22.7% | GPU主导型按一位小数四舍五入为22.7%，保留精确来源 |
| 科研AI减配15.7–15.8% | 三种配置15.7212–15.7800% | 一致 |
| 推理减配9.2% | 三种配置9.1572–9.1711% | 一致 |
| 云通用减配31.6% | 31.6132% | 一致 |
| 科研计算减配18.3–20.8% | 18.2767–20.8268%；最少台数13,195–13,620 | 一致，图中显示求解区间 |
| 48/168小时训练27.6%、科研AI22.4% | GPU主导型分别27.5591%、22.3663% | 一致 |
| 训练平均推迟17小时、平均占用91% | 可推迟电量加权17.2243小时，瓶颈资源平均占用90.5605% | 一致；图中显示17.2小时、90.6% |
| 训练设备年化节省3,009万 | 30.088621百万美元/年，即3,008.862万 | 一致 |
| 5%冗余2,486万、10%冗余1,975万 | 24.860868、19.749288百万美元/年 | 一致 |
| A→C净节省占机群9.5–33.0% | 实际为9.446199–32.978176%；最小值来自均衡型推理，图4a总范围标9.45–33.0%，各条按一位小数标注 | 精确下界9.446199%按一位小数为9.4%；用户给出的9.5%为约数 |
| 能源部分0.49–1.40百万美元 | 八城市中位数在11情境间为0.489107–1.398242百万美元/年 | 一致 |
| 设备价格翻倍，训练净节省31.49→61.58百万美元 | GPU主导型31.486863→61.575484百万美元/年 | 一致 |
| 出售空闲算力，s=50%、p=1为10.38–24.01百万美元 | 全部九种GPU情境、各取八城市中位数，范围10.376250–24.006019百万美元/年 | 一致；图4c展示其中三种GPU主导用途，全部1,080个城市价格比例组合另存CSV |
| s=50%、p=2.2为23.28–54.20百万美元 | 同口径23.284773–54.195162百万美元/年 | 一致 |
| 训练/科研AI/推理持平售价约2.6/1.3/1.3 | GPU主导型分别2.605015、1.337931、1.255758美元/GPU·小时 | 一致；图中标2.61、1.34、1.26 |

图4的2.2美元竖线采用[UsagePricing 2026-07第三方历史记录](https://www.usagepricing.com/blueprint/activity/deepinfra-2026-07-14-gpu-hour-token-hike)；2.9美元竖线采用[DeepInfra官方H100 80GB当前价格](https://deepinfra.com/pricing)，于2026-11-08读取核验。

## 验证与研究边界

已按每张图完成两轮排版修正，并复核导出PDF的实际渲染，包括字体、字号、面板字母、文字间距、对齐、色标、图例与留白。图1d的最大值说明位于分布面板内部独立空白区域。程序检查覆盖14个主面板源数据、第一部分数值一致性、设备与能源节省恒等式、图4持平价格计算、字体嵌入、180 mm宽度、600 dpi元数据及文字画布边界。图2–图4的PDF无栅格图像；图1的栅格图像均属于地图层。

研究输入仍有以下边界，已在图内说明、图注或源数据中保留：

1. 科研计算台数仍在13,195–13,620之间，尚未证明唯一最优解；三个期限均显示区间，金额采用主期限13,620台可行方案。
2. 整体作业情境的能源费用最优性尚未证明；图4报告八城市可行策略节省中位数，完整费用上下界保留在面板源数据。价格与冗余敏感性没有重新优化供能。
3. 云通用计算源表缺少平均推迟记录；图3c单列其92.93%资源占用率，平均推迟保持缺失。训练17.22小时采用电量加权，逐作业平均等待采用另一分母。
4. 第一部分是4,842个全球有效网格的中位数，第二部分能源结果来自八城市；图4用于量级比较。机群峰值余量和资源占用率均不构成逐卡可售容量清单。
5. 出售收入保留原机群与原供能资产，扣除额外市电和碳费用；可售比例是情景参数，未模拟市场需求、显存装箱、逐卡碎片和模型加载开销。三种用法收益不可直接相加，减配后出售的 `CD_extra` 上限情景另存源数据。

本次没有启动新的服务器容量或能源优化求解。

## 版本、提交与PR

分支：`codex/paper-figures-zh-20261108`；由`origin/main`的`12a52412fdeb9aa3473758889a6138a226c46d25`创建，PR目标为`main`。数据依赖[#60](https://github.com/TuzzyT-Laiii/33/pull/60)、[#63](https://github.com/TuzzyT-Laiii/33/pull/63)、[#65](https://github.com/TuzzyT-Laiii/33/pull/65)、[#67](https://github.com/TuzzyT-Laiii/33/pull/67)，固定输入提交见[图件README](dc-flexibility-project/papers/demand-flexibility/figures-zh/README.md)。本分支仅读取这些来源文件，未执行上游PR合并。
