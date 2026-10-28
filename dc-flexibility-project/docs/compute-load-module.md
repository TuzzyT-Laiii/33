# 任意网格 CPU/GPU 逐小时负荷模块

模块版本：`compute-load-20261025-v1`。功率标定与冷却参数版本：`20261025-v4`。代码入口为 [compute_load](../analysis/compute_load/__init__.py)，全球网格入口为 [global_grid_compute_load.py](../analysis/global_grid_compute_load.py)。

本模块给定一个网格完整 UTC 年的逐小时温湿度、CPU/GPU 资源需求曲线和 IT 参考规模，输出每小时的 IT、散热需求、冷却及设施总负荷。默认规模为 1、10、100、500 MW，每个网格、每年输出 8 条全年曲线。同一套资源需求与设备可以在不同网格比较，气象改变冷却和配套用电。

## 1. 输入、输出与单位

天气采用 CSV 或 Parquet，字段如下。Python API 接受同样结构的 pandas DataFrame。

| 字段 | 单位及要求 |
| --- | --- |
| `grid_id` | 网格标识；批处理必须提供，一次底层计算只处理一个网格 |
| `time_utc` | 每小时整点，完整 UTC 日历年，无重复、缺口；2024 等闰年为 8784 小时 |
| `temperature_c` | 室外干球温度，℃；必须显式将开尔文减去 273.15 |
| `relative_humidity_pct` | 相对湿度，百分数，满足 0 < RH ≤ 100 |
| `dew_point_c` | 露点，℃；可用它代替相对湿度。近饱和的微小数值超差按下述规则处理 |

湿度与露点至少提供其中一个。两者同时提供时，按 PsychroLib 换算的露点差异须在 0.05℃ 内。天气记录代表对应 UTC 小时的运行环境；上游负责明确天气是小时均值还是该小时的代表时刻值。模块不填补缺失天气，超出冷却校准范围的天气保留外推标记。

近饱和数值超差采用明确的 0.01℃ 容差：若露点高于干球且差值不超过此值，保留 `input_dew_point_c` 原值和 `dew_point_roundoff_adjusted` 标记，用干球值作为饱和边界处的计算露点；超过容差立即报错。这个容差是本模块的数据质量规则。既有全球 ERA5 样本网格的 26305 条原生记录中出现过两条此类超差，最大约 0.00098℃。严格 `extrapolation="error"` 策略同时拒绝需要此项处理的小时。

资源输入使用 `ResourceProfile`，包括硬件类型、逐小时份额和来源信息。CPU 默认使用 Bitbrains fast 的使用率；GPU 使用 Alibaba 的申请资源份额。资源曲线保持现有观测重放路径，全年均值由曲线自然形成。配置好的资源库包含平年与闰年各一套曲线，按相对小时顺序应用到请求年份。这个重放定义了统一研究情境，目标年份的真实业务需求需由用户提供。

[任务记录接口与柔性约束](task-trace-flexibility.md)支持从有秒级启动和结束时间的 CPU/GPU 作业累计资源小时，以及读取带任务类型和优先级的 ASI 小时记录。任务类别、截止时间和抢占权限分别保留；缺失权限维持未知。资源累计可以接入 `ResourceProfile`，需要先明确参考资源容量及跨设备功率映射的适用范围。

| 主要输出 | 单位及含义 |
| --- | --- |
| `resource_fraction` | CPU 使用率或 GPU 申请份额，0～1 |
| `node_count`、`gpu_count` | 安装节点数量与参考 GPU 数量 |
| `it_mw` | 整机 IT 电功率，MW电 |
| `heat_demand_mw_th` | 设备散热需求，MW热 |
| `outdoor_cooling_mw`、`hvac_mw`、`pump_mw` | 室外冷却、HVAC 和泵电功率 |
| `cooling_mw` | 上述三个冷却分项之和，MW电 |
| `other_mw` | ESIF 其余配套响应，MW电 |
| `facility_mw` | IT、冷却和其他配套电功率之和 |
| `pue` | 当小时设施总功率 / IT 功率 |
| `cumulative_*_mwh` | 从目标年起点累计的电量 |
| `outside_calibration_*` | 天气与 IT 负载超出冷却校准范围的标记 |
| `outside_power_anchor_range` | 资源份额超出实测节点功率点范围的标记 |

CPU 另输出核心占用代理数与累计核心代理小时；GPU 输出申请的等效卡数与累计等效卡小时。它们保留各自的数据含义，不解释为实测 FLOP 或已经执行的任务量。

## 2. 设备规模与 IT 功率

当前设备情境见[五种 CPU/GPU 服务器与整柜配置](../results/server-count-scenarios-20261028/README.md)：C20/C21/C22/C28 每台为 2∶0、2∶1、2∶2、2∶8，R-GB200 每柜为 36∶72。四档规模按允许需求配对，共 52 个单元。`research_unit` 与 `per_unit` 字段明确完整服务器/机柜；数量、功率及算力分别核算，每类需要匹配的完整单元功率标定。运行功率和台数/柜数保持为空，下面的纯设备池参考实现及既有结果按原版本使用。

CPU 参考硬件为 HPE DL385 Gen10 Plus，2×EPYC 7763、128 物理核心、256 GB 内存。GPU 参考硬件为 8×H100 HGX 整机，含宿主 CPU 与内存。CPU 采用 SPECpower 的 11 个实测点；GPU 采用 Latif 等 Table I 的空闲、实际任务和 CPU/GPU 联合压力点。标定表随模块保存在 [assets](../analysis/compute_load/assets/)，计算时不访问网络。

设 IT 参考规模为 S MW，单节点高负载参考功率为 p_ref W：

\[
N=\left\lfloor\frac{10^6 S}{p_{ref}}\right\rfloor,\qquad C=\frac{N p_{ref}}{10^6}.
\]

CPU 的 p_ref=546 W；GPU 的 p_ref=8430 W。C 是整机取整后的安装参考 IT 功率。

| IT 参考规模 | CPU 节点 | GPU 节点 | H100 参考卡 |
| --- | ---: | ---: | ---: |
| 1 MW | 1831 | 118 | 944 |
| 10 MW | 18315 | 1186 | 9488 |
| 100 MW | 183150 | 11862 | 94896 |
| 500 MW | 915750 | 59311 | 474488 |

给定资源份额 u_t，在实测点之间作线性插值，得到单节点功率 p(u_t)，再计算：

\[
P_{IT,t}=N p(u_t)/10^6.
\]

CPU 空闲为 120 W，最高标定点的相对负载为 0.995、功率 546 W；GPU 空闲为 1.86 kW，36%、77%、93%、100% 的标定功率为 4.68、6.26、7.92、8.43 kW。GPU-only 的 7.97 kW 压力点单独保留在来源表中，整机映射采用包含 CPU 的联合压力点。

模块默认各节点持续通电，并将平均资源份额均匀映射到节点。节点装箱、关闭空闲节点或不同硬件的功率需要对应的观测或运行方案。插值范围外采用端点功率并添加标记；`extrapolation="error"` 会拒绝这类输入。

这里的 MW 表示实测高负载 IT 预算。工程设计还要核对设备铭牌、配电和冷却容量。例如 500 MW GPU 情境的 59311 个节点，按 10.2 kW/节点的厂商参考额定值约为 604.97 MW，配套用电另计。

## 3. 散热、温湿度与冷却电功率

新版矩阵逐单元记录 `cooling_type`：44 个为 `air`，8 个 GB200 单元为 `direct_liquid`。本节现有冷却/气象耦合按风冷机房设计；GB200 的 CPU/GPU 采用冷板直接液冷，仍有残余风冷部件，需要独立的供液、CDU/泵、风机与换热参数。液冷可采用更高供液温度，干冷器和冷却塔对干球/湿球条件的响应不同，具体数值待官方规格或测量核定。本次未实现新液冷模型，现有系数不能直接用于 GB200 的设施负荷计算。

小时尺度忽略蓄热及额外建筑得热，采用 Q_IT,t ≈ P_IT,t。散热是热量指标；设施电功率由电功率分项求和。

定义归一化负载 q_t=P_IT,t/C。使用 PsychroLib SI 公式将相对湿度转换成露点 T_d,t。冷却经验模型采用以下八项特征：

\[
X_t=\bigl[1,q_t,q_t^2,
q_t(T_t-3.969434)_+,
q_t(T_t-20.685997)_+,
q_t(T_{d,t}+8.236311)_+,
q_t(T_{d,t}-7.992787)_+,
(-3.280123-T_t)_+\bigr],
\]

其中 (x)_+=max(x,0)。代码使用 JSON 参数中的完整精度。每个配套分项 j 的功率为：

\[
P_{j,t}=C\sum_{k=1}^{8}\beta_{k,j}X_{k,t}.
\]

系数 β 为 8×4 非负矩阵，见 [cooling-calibration.json](../analysis/compute_load/assets/cooling-calibration.json)，四列依次对应室外冷却、HVAC、泵和其他配套。模型保留固定配套、热负载、温度、露点和冷天运行响应。

最后计算：

\[
P_{cool,t}=P_{outdoor,t}+P_{HVAC,t}+P_{pump,t},\quad
P_{facility,t}=P_{IT,t}+P_{cool,t}+P_{other,t}.
\]

这是一套由 ESIF 同时间功率与天气记录标定的经验响应，尚未识别目标设施的 COP、供水温度、流量和设备启停。相同冷却模块按安装参考功率线性复制到各档规模，规模效率尚未标定。外部共享网络和存储的功率尚未识别。

冷却模型使用 5447 个配对小时：4775 小时拟合、末尾 672 小时留出评估，随后用全部配对小时重新拟合部署系数。时间留出冷却 RMSE=17.754 kW，约为测试均值的 10.276%。这些指标描述该时间留出检验。校准范围为干球 −19.31～38.13℃、露点 −29.42～17.68℃、q=0.2315～1.0。

默认 `extrapolation="flag"` 保留超范围小时并输出标记；`"error"` 拒绝超范围输入。冷却残差上下带来自 ESIF 留出残差的 5%/95% 分位，不能解释为任意网格的覆盖保证。

## 4. 全年累计、文件与恢复运行

每条记录的步长为 1 小时，因此电量为 ΣP_t×1h，单位 MWh。月度与年度电量直接由小时功率求和。年度 PUE 使用全年设施电量 / 全年 IT 电量。CPU 核心代理小时或 GPU 申请卡小时也按相同小时步长累计。程序按目标年份校验 8760/8784 小时，拒绝缺口与重复。

输出目录结构为 `grid_id={网格标识}/{年份}/{cpu或gpu}/{规模}MW.parquet`。每个网格、每年另有 `annual-summary.csv`、`monthly-summary.csv` 和 `manifest.json`。清单记录温湿度值、资源文件、参数资产、代码和小时结果的校验值。

再次运行相同输入时，校验已有小时文件与汇总文件后复用结果。输入、软件或代码发生变化时，要求使用独立的结果目录，防止混合版本。全球入口逐网格处理，输出逐情境落盘。

## 5. 安装与调用

Python 3.10 或更高版本。安装依赖后，在 `analysis/` 目录运行下列命令；在其他工作目录调用时，将该目录加入 `PYTHONPATH`。

```bash
python -m pip install -r compute_load/requirements.txt
```

先从已有完整成果包构建资源库。资源输入按来源许可留在本地或私有研究环境；公共仓库保存代码、标定点和资源库构建方法。

```bash
python build_compute_profile_library.py --reference-dir /path/to/package/outputs --output-dir /path/to/profiles
```

Python API：

```python
import pandas as pd
from compute_load import ComputeLoadModel, ProfileLibrary

weather = pd.read_parquet("grid_weather.parquet")
profiles = ProfileLibrary("/path/to/profiles")
model = ComputeLoadModel()
curve = model.calculate(weather, profiles.for_year("gpu", 2025), 100)
curve.to_parquet("gpu_100MW_2025.parquet", index=False)
```

批量生成某天气文件中的 CPU/GPU 四档规模：

```bash
python -m compute_load --weather grid_weather.parquet --profiles-dir /path/to/profiles --years 2025 --output-dir /path/to/results
```

用户有新的实测资源曲线时，可用 `ResourceProfile("cpu", values, {"source": "来源及测量口径"})` 传入。`values` 必须按天气相同小时顺序排列；CSV 中的资源时间戳应在上游显式按 UTC 连接后再提取。模块不为资源需求生成随机均值。

## 6. 接入现有全球网格项目

服务器项目根目录为 `/data3/dataset/ljt/global-grid-flexibility-20261023`。现有 ERA5 文件包括 `time`、`native_flat_index`、`2m_temperature` 和 `2m_dewpoint_temperature`。网格映射登记 0.5°研究网格及其原生 0.25°天气节点。

全球入口采用映射中 `row_offset=1`、`col_offset=1` 的中心节点作为网格代表气象点，明确记录该选择。开尔文统一转换成摄氏温度，按 UTC 年选择记录，额外的 2026 年右边界记录不进入 2023—2025 年计算。这里使用记录时刻的温湿度代表对应小时环境。

```bash
python analysis/global_grid_compute_load.py --root /data3/dataset/ljt/global-grid-flexibility-20261023 --grid-ids g05_r100_c592 --weather-path data/processed/era5-landuse-trial/g05_r100_c592-hourly.parquet --years 2023 2024 2025
```

`--weather-path`、`--mapping`、`--profiles-dir` 和 `--output-dir` 可显式指定；省略时使用项目既有目录。上例在服务器项目根目录运行。完整合格区域天气到齐后，使用 `--all-grids` 读取现有合格天气目录。该入口仅选择已经完成的 `.parquet` 文件；中心节点缺失、年份不完整或数据质量不符时明确报错。用地筛选和气象下载权限由上游流程维护。

中心节点是当前网格比较的代表位置。拥有设施位置或更合适的网格平均天气时，可直接使用标准温湿度接口；多个点的天气组合应明确空间权重及先后处理顺序，冷却响应存在非线性。

## 新任务数据的全年构造入口

[实际配置与全年负荷说明](../results/scenario-load-methods-20261026-v2/README.md)给出 CPU 计算型、GPU 加速型与混合设备的依据、三项情境维度以及 8760 小时处理结果。GPU 整机已经包含宿主 CPU；混合 IT 配比由对应节点数量和匹配整机功率推导。

在项目根目录运行：

```bash
python analysis/build_task_annual_loads.py --year 2025 --grid-id g05_r100_c592 --weather-path data/processed/era5-landuse-trial/g05_r100_c592-hourly.parquet
python analysis/verify_task_annual_loads.py
```

入口复用 `global_grid_compute_load.grid_weather` 的网格映射和 ERA5 转换，读取已核验 Lisa/Acme/ASI 小时需求。`annual_task_profiles.replay_hourly(source, times, phase_hours=0)` 依次循环完整观测窗口，不改变其平均目标，不添加季节变化。全年资源路径可直接传给 `ComputeLoadModel.calculate`；完整日历、取整、冷却与外推检查沿用现有模块。

已生成 2025 年 16 条、140160 条小时负荷记录与 490 组日相位汇总，另用真实 2024 年天气核验 CPU/GPU 的 8784 小时接口。完整曲线保存在 `data/processed/task-annual-loads-20261026-v2/`；按网格与年份分别保存。现有默认资源库保持独立版本。ASI 一日循环为演示，Acme CPU 为开发诊断，来源归一化及功率映射的适用范围见说明。

## 7. 与供电求解器连接

[prepare_compute_solver_input.py](../analysis/prepare_compute_solver_input.py) 按相同 UTC 小时连接负荷与现有风光容量因子；保留新模块的设施功率和 PUE。供电输入至少包含 `time_utc`、`pv_capacity_factor`、`wind_capacity_factor`，小时必须与负荷完全一致。

```bash
python analysis/prepare_compute_solver_input.py --load-curve /path/to/gpu/100MW.parquet --supply-hourly /path/to/supply_2025.parquet --output /path/to/solver-input.parquet
python analysis/global_grid_flex_solver.py --hourly /path/to/solver-input.parquet --config /path/to/cost-config.json --scenarios C0 CS --output-dir /path/to/supply-results
```

成本配置需显式设置 `"load_model": "weather_coupled_fixed_v1"`，并令 `reference_it_mw` 与负荷的 IT 参考规模一致。`compute_module_usd_year` 对应这一整套参考规模的年化计算设施成本；风光和储能成本保持现有每 MW、每 MWh 单位，并保留价格年份、来源和参数性质。模块不补设未知价格。

这个模式直接把 `facility_mw` 放入逐小时供电平衡，计算设施规模固定为已经给定的节点数量。C0 固定需求且无储能，CS 固定需求且允许储能。沿用本地风光供电和储能首尾循环边界。

现有 CD/CSD 求解器使用恒定任务到达与线性功率关系。新曲线在此模式下拒绝直接进入 CD/CSD：任务重排会改变每小时资源份额，IT 插值及冷却的 q² 项均须重新计算。调度程序得到新资源路径后，可以再次调用 `model.calculate` 计算对应负荷；对它进行联合优化还需要明确任务服务约束与相应的非线性或分段模型。原有线性情景通过 `legacy_affine` 模式保留各自口径。

## 8. 验证与来源

[verify_compute_load_module.py](../analysis/verify_compute_load_module.py) 用已核验的五地点实际天气与资源数据，比较 CPU/GPU、四档规模、三年的 120 条旧曲线，共 1052160 个小时。另检查温湿度接口、无缺口日历、设备取整、错误输入、恢复运行，以及真实 ERA5 168 小时风光与新负荷的供电接口。短周期求解验证接口和守恒，经济结果仍受其明确登记的技术价格情景与循环边界约束。

本机与服务器均通过上述回归及 14 项异常输入检查。既有全球网格 `g05_r100_c592` 的三年原生 ERA5 数据已生成 24 条曲线、210432 条小时记录，并通过来源露点、UTC 连接、Parquet 回读、功率守恒与年度电量检查。记录分别见[本机核验](../data/metadata/compute-load-module-20261025-v1/validation.json)、[服务器核验](../data/metadata/compute-load-module-20261025-v1/server-validation.json)、[原生网格接入核验](../data/metadata/compute-load-module-20261025-v1/native-grid-validation.json)与[原线性求解器核验](../data/metadata/compute-load-module-20261025-v1/legacy-solver-validation.json)。

来源：

- [CPU：SPECpower HPE DL385 Gen10 Plus](https://ftp.spec.org/power_ssj2008/results/res2021q1/power_ssj2008-20210221-01063.html)
- [GPU：Latif 等，HGX H100 实测功率 Table I](https://arxiv.org/html/2412.08602v1)
- [原报告及来源清单](https://github.com/TuzzyT-Laiii/33/blob/main/dc-flexibility-project/results/compute-center-data-audit-20261022/coupled-power-weather-20261025-v4/README.md)
- [PsychroLib：湿空气计算、SI 单位及数组支持](https://github.com/psychrometrics/psychrolib)
- [SciPy linprog：固定需求供电线性规划](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)
- [Apache Arrow Dataset：按节点筛选既有 Parquet 天气](https://arrow.apache.org/docs/python/dataset.html)
