# 八城市现场供电柔性收益

本目录计算同一城市点位新增光伏、陆上风电及本地风光充电储能下的需求侧柔性净收益。主用2024天气、10 MW参考IT设备、12个服务器类型×用途情境。第一阶段负责天气、容量因子及原始负荷/PUE；第二阶段只读其已落盘文件。完整城市输入尚在生成时，服务器控制器等待输入并保持全年核验门槛。

## 运行状态与文件

- `validation.json`：数学结构核验、与原168小时求解器对照，以及真实全年核验是否通过。`annual_real_input_gate=false`期间不放开批量。
- `main-results.csv`：同S、同α或同碳价的经济配对结果；保留不可行与未完成求解状态。
- `physical-results.csv`：固定风光与储能容量后的匹配率、购电、弃电变化。
- `delay-thresholds.csv`：D50/D90；只有十档D全部完成且V(168)>0时才计算。
- `infeasible-summary.csv`：不可行及未完成状态汇总。不存在的成本差额保留空白。
- `manifest.json`：结果覆盖数、参数与模型SHA、输入来源与SHA、输出SHA。
- 三组诊断图：`value-vs-delay`、`delay-thresholds`、`storage-delay-heatmap`，均输出300 dpi PNG与PDF。灰色单元表示配对收益不可用，包括固定参照不可行或曲线尚未完成。图仅展示已核验结果的范围。
- 逐小时明细与逐算例JSON位于 `data/processed/eight-city-flex-value-20261029/`，不提交Git；服务器有独立副本。

逐小时明细保留执行工作量、IT与设施功率、积压、直接供电、购电、充放电、SOC及弃电；天气、原始负荷和PUE按输入SHA独立保存，Zstandard压缩保持数值无损。服务器法兰克福、吕勒奥、乌兰察布、贵阳、利雅得和新加坡的输出目录映射至 `/home/ljt/eight-city-flex-value-20261029/details/`，逻辑路径保持上述`data/processed`目录；路径清单见`storage-layout.json`。后续队列写入前保留至少64 GiB的文件系统余量（`--minimum-free-disk-gb`可调整），空间不足时明确停止并保留完成算例，保障ERA5所在数据盘的剩余空间。

## 模型与任务权限

令原始IT功率为P⁰_t，硬件空闲功率为I，动态到达工作量q_t=P⁰_t−I。q以等效动态IT MWh表示，在本线性模型中假设相同工作量具有固定电力效率。不可推迟部分为(1−φ)q_t，柔性完成量为y_t，新增容量倍数u由模型在[1,r]内选择：

\[
b_t=b_{t-1}+\phi q_t-y_t,\quad
0\le b_t\le\sum_{a=\max(0,t-D+1)}^t\phi q_a,\quad b_{-1}=b_{T-1}=0,
\]
\[
(1-\phi)q_t+y_t\le uK,\qquad
P^{IT}_t=uI+(1-\phi)q_t+y_t,\qquad L_t=PUE_tP^{IT}_t.
\]

K使用第一阶段同一硬件、同一功率方法的满资源动态功率端点。GPU的`ratio_p50`方法按其迁移功率比例标定K；10 MW为设备定容参考，不能将未使用的GPU TDP直接解释为额外任务吞吐能力。闲置服务器、CPU、内存、网络与机房费用包含在统一算力价格代理中；新增空闲用电uI完整计入。动态IT电量守恒；额外总IT电量为(u−1)IT。

执行时刻的PUE取第一阶段该小时数值，推迟任务的冷却随执行时刻天气变化。此处冻结PUE对IT负载率的反馈；CPU非线性功率与实际作业时长折合为等效动态IT工作量。结果为可中断、可分割技术潜力，未加入作业连续性、检查点开销、异构任务吞吐约束。INF情境下φ是反事实服务权限，不能解释为已有在线服务SLA。

2024实际为8784小时。本轮按配置删除2月29日，保留原UTC时间戳，模型按连续8760个研究小时计时；跨删除日期的D按研究小时计算。年初积压为零，年末全部任务完成，最后D小时的任务完成窗口受到年末收紧；任务、能耗和费用均保留在报告期。该保守有限年边界区别于跨年滚动运行，D较长时需结合边界敏感性解释。

### 积压形式的等价性

定义累计到达A_t=Σ_{a≤t}φq_a、累计完成Y_t=Σ_{s≤t}y_s，则上述约束等价于A_{t−D}≤Y_t≤A_t，其中负数索引A为零。逐批x(a,t)形式显然满足这些累计上下界。反方向按最早到达/最早截止时间执行：Y_t≤A_t禁止提前执行；Y_t≥A_{t−D}保证截止到t的到达批次全部完成，因此可重建仅在a≤t≤min(a+D,T−1)内非零的x。两种形式的设施与容量约束仅依赖逐小时y，故可行投影与最优目标一致。

代码独立重建EDF，核验逐批期限、等待积分和期末遗留。逐批对照移除了积压上界，避免把待验证约束重复附加到参照模型。全年主体为70085个变量，D不改变变量数；逐批形式只用于≤336小时核验。

## 供电、储能与目标函数

\[
K_{PV}CF^{PV}_t+K_WCF^W_t=h_t+c_t+z_t,\quad h_t+g_t+d_t=L_t,
\]
\[
e_t=(1-s)e_{t-1}+\eta_cc_t-d_t/\eta_d,\quad e_{-1}=e_{T-1},
\]
\[
0\le c_t,d_t\le P_S,\quad0\le e_t\le E_S,\quad
\sum_tg_t\le(1-\alpha)\sum_tL_t.
\]

h为现场绿电直接供电，g为电网补电，c/d为母线侧充放电，z为弃电。独立可再生母线禁止电网给储能充电。α只计入实际供应设施的本地绿电，储能输入和弃电不重复计入。无α时删除最后一项约束。逐小时核验充放电互斥；正的放电运行费用与免费弃电消除有损循环的收益，若出现超过2×10⁻⁵ MW的同小时充放电则立即报错。

年化目标为算力、风光及储能的资本年金与固定运维，加上Σ_tg_t(p_grid+p_CO₂EF)、储能微小正运行费用。设备年化系数CRF=i/[1−(1+i)^−L]，分别使用资产寿命。风光投资使用总装机成本，现场土地、土建和连接采用该成本及情景运维代理，没有逐城市用地或接入报价；10000 MW风光上限属于数值上限，不能据此宣称实际现场土地可建设。电网接入上限100 MW，统一情景不另收容量/需量费用。

给定α时，首先单独求解最小约束残差ρ=min[Σg−(1−α)ΣL]，这一阶段不附加α行。发电与储能容量增加可以通过弃电或闲置保持原运行方案，因此该阶段将这四项容量固定至各自允许上限，与在全部容量范围内求最小ρ等价。若ρ>10⁻⁴ MWh，即得到该任务权限、S与容量上限下α不可达的证书；若ρ≤容差，再求解带α的完整成本优化。证书阶段的容量不作为经济结果，最终经济装机由成本优化决定。各阶段的原始HiGHS状态、残差、耗时均保存。

S六档为0、1/2/4/8小时、优化参照。固定档功率等于该情境原始年平均IT功率，电量为功率×小时数；固定S成本即使未使用仍计入。配置S也支持`{"power_mw":数值,"energy_mwh":数值}`。优化档分别选择MW/MWh；ATB是60 MW、2–10小时系统参考，1小时及更小系统为线性成本外推。

## 两种收益口径

经济收益：V(D,φ,r|S,α,p_CO₂)=C_fixed(S,α,p_CO₂)−C_flex(D,φ,r,S,α,p_CO₂)。两侧风光均可优化，额外算力费用已扣除。单位收益分母固定为原始年IT电量；另给出V/C_fixed百分比。

物理收益：给定同一城市点位光伏30 MW、风电30 MW，作为明确的固定装机情景；固定储能档按定义给定MW/MWh，优化储能档使用该档固定任务经济参照的MW/MWh。固定容量后仅以减少购电为调度目标，比较CFE score、购电及弃电；不在物理CSV中报告经济收益。固定任务α95%不可行时仍可进行这一固定装机物理比较。

`local_cfe_hourly_mean`是平均逐小时本地供电比例，`local_cfe_energy`是年电量加权本地比例；α作用于后者。电网自身清洁份额不计入现场匹配率。移动电量为Σmax(q_t−q̂_t,0)，等于动态曲线差绝对值之和的一半；EDF另报告实际被延期批次的电量与最长延迟。每MWh移动电量收益在移动量为零时保留空白。最优调度可能不唯一，移动量等运行指标对应保存的具体最优解。

边际时间价值按相邻测试D的ΔV/ΔD计算；D50/D90为测试网格上首次达到V(168)的50%/90%的D，不插值、不推断未完成曲线。`storage_interaction_usd_year=V(S)−V(S=0)`，正值表示该设置下互补，负值表示替代；优化档另报告柔性减少的MW/MWh。

## 集中参数

编辑 [analysis/eight-city-flex-value-20261029.json](../../analysis/eight-city-flex-value-20261029.json)。每项成本及情景参数附来源或情景假设；所有价格统一到2024 USD。原始来源核验见 [cost-source-validation.json](../../data/metadata/eight-city-flex-value-20261029/cost-source-validation.json)。

| 参数 | 主值 | 来源及边界 |
|---|---:|---|
| 光伏总装机成本 | 691 USD/kW | [IRENA 2024全球加权平均](https://www.irena.org/Digital-Report/Renewable-Power-Generation-Costs-in-2024) |
| 陆上风电总装机成本 | 1041 USD/kW | 同上；2024 USD |
| 储能功率投资 | 361.34 2023USD/kW → 371.9975 2024USD/kW | [ATB2025工作簿](https://data.openei.org/files/8759/2025_v1_Workbook_08-27-2026.xlsx)，Utility-Scale Battery - R&D!G26，2024技术年 |
| 储能电量投资 | 234.01 2023USD/kWh → 240.9120 2024USD/kWh | 同工作表G20；功率费用另计 |
| 储能寿命、FOM、往返效率 | 15年、资本额4%/年、85% | [ATB2025](https://atb.nlr.gov/electricity/2025/utility-scale_battery_storage)，FOM含增容；η_c=η_d=√0.85 |
| 价格年份换算 | 313.689/304.702 | [BLS CPI-U年度平均](https://www.bls.gov/regions/mid-atlantic/data/consumerpriceindexannualandsemiannual_table.htm)，2024/2023 |
| 折现率 | 8% | 情景假设；敏感性5%/12% |
| 光伏、风电寿命 | 均25年 | 情景假设 |
| 光伏、风电年运维 | 资本额1.5%、2.5% | 情景假设 |
| 算力及机房资本额 | 600万 USD/MW参考IT | 情景假设；敏感性150万/600万/2400万 |
| 算力寿命、运维 | 4年、资本额3%/年 | 情景假设；寿命敏感性3/5年 |
| 算力年化费用 | 199.1525万 USD/MW·年 | 以上主值推导，包含基准与新增设备 |
| 购电价格 | 100 USD/MWh | 全球统一情景值；城市差异需修改配置 |
| 碳价 | 0/50/100/200 USD/tCO₂e | 情景假设；主算100 |
| 储能放电运行费用 | 0.01 USD/MWh | 数值运行情景；ATB增容费用已在FOM内 |

排放因子来自[Ember年度数据](https://ember-energy.org/data/yearly-electricity-data/)在2026-10-29下载的最新完整2025年`Total generation`行，原始`gCO₂e/kWh`除以1000。Ember采用CO₂e边界，本轮碳税按该强度作为计价代理；该值属于国家平均发电强度，不能解释为城市边际减排或法定直接CO₂排放税基。天气年与排放因子年份分别记录。

| 城市 | 电网代理 | tCO₂e/MWh |
|---|---|---:|
| 菲尼克斯、阿什本 | 美国国家平均 | 0.384403 |
| 法兰克福 | 德国国家平均 | 0.334117 |
| 吕勒奥 | 瑞典国家平均 | 0.035390 |
| 乌兰察布、贵阳 | 中国国家平均 | 0.528278 |
| 利雅得 | 沙特阿拉伯国家平均 | 0.679828 |
| 新加坡 | 新加坡国家平均 | 0.497047 |

## 全部与单城市重跑

Linux服务器既有环境含SciPy/HiGHS、pandas及pyarrow。新增环境安装 `pip install -r analysis/requirements-eight-city-flex-value.txt`。从项目目录执行一条命令，自动等待输入、完成真实全年核验并依次启动主组合与后续队列：

```bash
python analysis/launch_eight_city_flex_value.py --input-dir /data3/dataset/ljt/eight-city-flex-value-20261029/inputs/phase-a --workers 16
```

只计算一个城市及情境：

```bash
python analysis/launch_eight_city_flex_value.py --input-dir /data3/dataset/ljt/eight-city-flex-value-20261029/inputs/phase-a --city frankfurt --scenario D-TRN --workers 4
```

已有同配置全年核验时，可直接续跑全部或筛选城市/情境；配置或模型变化时门槛立即失效，使用上述控制器重建核验：

```bash
python analysis/run_eight_city_flex_value.py --input-dir /data3/dataset/ljt/eight-city-flex-value-20261029/inputs/phase-a --gate results/eight-city-flex-value-20261029/validation.json --tier main --city phoenix --scenario A-HPC --workers 4
```

主组合：96个城市×情境单元，φ=100%、r≤2、碳价100、α为无约束/95%，六档S、十档D，共11520个经济算例。全部经济网格共276480个算例，后续264960个排队；完整物理网格为11520个算例。2023/2025及r=1/1.5作为配置敏感性，默认不混入主批次。逐算例原子保存，重新运行跳过已经完成的相同输入/配置/模型，`not_solved`保留状态并可重新计算。

控制器运行目录 `/data3/dataset/ljt/eight-city-flex-value-20261029/`。服务器监督进程PID为2545856，经济/物理队列PID分别为2511553/2511554。第一阶段输入已完成只读同步。主批次运行期间每分钟刷新服务器CSV和诊断图。日志和进度：

```bash
tail -n 30 /data3/dataset/ljt/eight-city-flex-value-20261029/logs/controller.log
cat /data3/dataset/ljt/eight-city-flex-value-20261029/project/data/processed/eight-city-flex-value-20261029/controller-status.json
```

重新生成主CSV、派生指标、来源清单与诊断图：

```bash
python analysis/report_eight_city_flex_value.py
```

## 审阅依赖

本分支从origin/main `6ffda9b`建立。依赖[第一阶段PR #23](https://github.com/TuzzyT-Laiii/33/pull/23)（分支`codex/eight-city-phase-a-20261029`）的天气、风光容量因子、情境命名和逐小时PUE/负荷输出；建议先审阅第一阶段的数据与物理校验，再审阅本阶段收益模型与结果。原168小时求解器接口保留，本阶段新增独立模块。PR保持待审，不自动合并。
