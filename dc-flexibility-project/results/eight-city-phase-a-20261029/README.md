# 八城市第一阶段：天气、风光出力、负荷与可移动潜力

输入版本为 `origin/main@6ffda9b`、典型负荷曲线 v2（PR #21）和冷却模型 `cooling-20261028-v2`。研究天气期为2023-01-01 00:00至2025-12-31 23:00 UTC，共26,304小时；各年分别8,760、8,784、8,760小时。本目录对应第一阶段A，未运行四组供能与柔性配对优化。

## 情境和结果入口

- [情境命名表 Markdown](scenario-names.md) / [CSV](scenario-names.csv)：13个条目，12个可计算情境；A-CLD仍待Alibaba输入。
- [八城市气候、坐标、资源及公开基准](city-coordinates-climate-resource.csv)；[逐年气候与资源](city-climate-resource-annual.csv)。
- [逐城公开基准差异说明](resource-benchmark-comparison.md)；[风电密度、高度与损失分项](wind-model-diagnostics.csv)。
- [8城市×12情境电量与PUE矩阵](load-pattern-summary.md)；[96行完整负荷汇总](scenario-city-load-summary.csv)，含年电量、峰值、PUE、冷却占比和最热周峰值。
- [逐年中央结果288行](annual-load-central.csv)；[中央/低/高参数年度结果864行](annual-load-summary.csv)。
- [物理柔性360组](flexibility-resource-summary.csv)、[12情境电量上界1080组](flexibility-electric-summary.csv)、[城市风光形状相关性](renewable-flexibility-shape-correlations.csv)。
- [求解器逐小时文件索引](processed-hourly-files.csv)、[validation.json](validation.json)、[manifest.json](manifest.json)、[天气溯源](weather-provenance.json)。

情境写为“服务器类型 × 用途”，例如 `D-TRN-10MW`；来源保留在 `demand_source`，旧配置保留在 `legacy_configuration_id`。A/B/C/D/E依次为纯CPU、CPU主导、均衡、GPU主导、纯GPU型（整柜液冷）；HPC/CLD/SAI/TRN/INF依次为科研计算、云通用计算、科研AI、大模型训练/开发、推理服务。12个10 MW情境对应原48个可计算规模单元的同一组12种形状。

E型GB200 NVL72含72颗GPU和36颗Grace CPU。依据[NVIDIA DPS 0.8每GPU最大参考功率1,200 W](https://docs.nvidia.com/datacenter/dps/versions/0.8/guides/concepts/device-specifications/)和[NVIDIA约120 kW整柜参考功率](https://docs.nvidia.com/multi-node-nvlink-systems/partition-guide-v1-2.pdf)，GPU为86.4 kW，设计参考占比72.0%。此比例随实际负载变化，年度用电仍按v2功率模型计算。

## 城市代表点和格点

代表点全部核对[NOAA NCEI ISD station history](https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv)，原始文件SHA见 `source-metadata/source-receipts.json`。使用机场或城市代表气象站，坐标均为WGS84经纬度；不据此声称某个实际数据中心位于该点。

| 城市 | 代表点（NOAA编号） | 纬度 | 经度 | ERA5格点 | 气候与资源背景 |
| --- | --- | ---: | ---: | --- | --- |
| 新加坡 | Paya Lebar（486940） | 1.360 | 103.910 | 1.50,104.00 | 赤道湿热，冷却全年持续；使用柔佛区域气候代理 |
| 菲尼克斯 | Sky Harbor（722780） | 33.428 | -112.004 | 33.50,-112.00 | 炎热干旱，太阳能强，存在温度降额 |
| 阿什本 | 邻近Dulles（724030） | 38.935 | -77.447 | 39.00,-77.50 | 距阿什本约12 km，湿润且冬夏差异明显 |
| 法兰克福 | Frankfurt Main（106370） | 50.026 | 8.543 | 50.00,8.50 | 温带，风光具有明显季节差异 |
| 吕勒奥 | Kallax/Lulea（021860） | 65.544 | 22.122 | 65.50,22.00 | 高纬寒冷，太阳能昼长季节差异大 |
| 乌兰察布 | Jining（534800） | 41.033 | 113.067 | 41.00,113.00 | 高海拔大陆性，低气压影响空气密度 |
| 贵阳 | Longdongbao（578160） | 26.539 | 106.801 | 26.50,106.75 | 高原湿润、多云，光伏资源有限 |
| 利雅得 | King Khaled（404370） | 24.958 | 46.699 | 25.00,46.75 | 炎热干旱，高温影响光伏与制冷 |

新加坡最近格点1.25°N,104°E陆地比例为0.210520，主计算采用北侧1.50°N,104°E格点，陆地比例0.729068，位于马来西亚柔佛。该点距代表站约18 km，是区域天气代理；存在跨境及沿海代表性限制。吕勒奥格点陆地比例0.573253，为沿海混合格点；其余格点均高于0.99。完整陆海检查见 `source-metadata/grid-land-check.json`。

## ERA5原始输入及请求

优先数据集为 `reanalysis-era5-single-levels-timeseries`，CDS控制器逐城提交、每次最多一个本任务活动请求，使用既有全局闸门和锁。保存请求编号后仅复用该编号；不确定提交意图会明确报错，防止重复提交。既有两个风速补取控制器保持运行。

全局闸门占用期间，从[ECMWF官方点时间序列ARCO后端说明](https://confluence.ecmwf.int/pages/viewpage.action?pageId=704944501)对应的官方Zarr源读取同一0.25°格点。七变量使用既有完整冻结原始对象，10 m风使用本次官方源对象；两个元数据版本分别记录SHA，并以精确相同的UTC索引合并。112个七变量压缩原始对象及32个10 m风对象、元数据均保留在服务器，逐对象SHA见天气溯源。菲尼克斯CDS成功请求为 `af8aadcf-a95a-4fe8-9056-012f84f8bde6`，全部九变量逐小时与ARCO结果相同。

原始与派生天气目录为服务器 `/data3/dataset/ljt/eight-city-weather-20261029/`。CDS原始响应ZIP和其中NetCDF均保留；ARCO保留 `frozen-raw/`、`arco-raw/`、逐城NetCDF和Parquet。CDS后续城市按共享闸门可用性继续取得正式回执，已冻结结果与回执分开归档。

| 变量 | 原始字段 | 单位 | 用途 |
| --- | --- | --- | --- |
| 2 m气温、露点 | t2m、d2m | K | 转为°C；PsychroLib计算湿度、湿球 |
| 地表气压 | sp | Pa | 湿空气性质、风电密度修正 |
| 地表向下短波、直射水平辐射 | ssrd、fdir | J m⁻² | 除以3600，转换前一小时平均W m⁻² |
| 10 m风分量 | u10、v10 | m s⁻¹ | 光伏温度模型及诊断 |
| 100 m风分量 | u100、v100 | m s⁻¹ | 轮毂风速与风电曲线 |

温度、风和气压为有效时刻值，辐射为结束于该时间戳的前一小时累积量；小时太阳位置取时间戳前30分钟。日内资源曲线与逐小时输入保留此“结束时刻”约定，实际可再生调度如需小时起始标签，应统一重标全部输入，避免只移动辐射。

## 光伏、风电与公开基准

光伏主计算采用固定倾角 `max(10°,min(|纬度|,60°))`，北半球朝南；单轴跟踪为敏感性（南北轴、±60°、回溯、GCR=0.35）。由直射水平辐射和小时中点太阳位置推算DNI，并由GHI减去直射水平分量取得DHI；太阳接近地平线时进行几何保护，直射超过GHI的小时及处理参数记录在计算清单。采用pvlib Hay–Davies倾斜面模型、反照率0.2、[Faiman温度模型](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.temperature.faiman.html)（u0=25、u1=6.84），10 m风以0.14指数换算到2 m。PVWatts温度系数−0.004 K⁻¹，系统损失14%，逆变器额定效率96%，DC/AC=1.2。

光伏容量因子的分子为交流输出，分母为直流铭牌容量；因此小时上限为1/1.2。未另行估计逐时积雪、沙尘沉积、老化或当地遮挡；统一14%损失为研究参数。固定倾角和跟踪均输出逐小时容量因子。

风电使用[公开IEA 3.4 MW/130 m参考机组功率表](https://github.com/NREL/turbine-models/blob/main/turbine_models/data/Onshore/IEA_Reference_3.4MW_130.csv)及同目录规格文件，IEC IIIa，标称3,370 kW，轮毂110 m。100 m风速以1/7幂律外推到110 m，空气密度为sp/(287.05T)，以密度比立方根修正等效风速，线性插值公开功率表；表外为零，实际轮毂风速超过25 m/s切出。主结果计入统一10%损失，同时保留毛容量因子以辅助比较。

光伏公开基准为[Global Solar Atlas](https://globalsolaratlas.info/support/methodology)的长期PVOUT，除以8,760得到同为DC分母的年容量因子；吕勒奥超出GSA太阳能覆盖，使用[JRC PVGIS 5.3](https://re.jrc.ec.europa.eu/api/v5_3/PVcalc?lat=65.544&lon=22.122&peakpower=1&loss=14&angle=60&aspect=0&outputformat=json)固定倾角、14%损失结果。风电取[Global Wind Atlas](https://globalwindatlas.info/)官网实际使用的GWA4 `cf_iec3`、100 m GeoTIFF图层，在代表站坐标采样原始像元；官网图层配置、接口和SHA均保存。此图层只用作量级参照，未将不同机型、损失及高度视为完全一致。

基准与模型差异同时受2023–2025天气和长期气候期、0.25°再分析与细网格地形/粗糙度、代表站与天气格点、光伏倾角及设备损失、风机功率曲线和轮毂高度影响。尤其城市机场点和高原/沿海粗格点不能代表附近风电场最优选址；容量因子低于GWA的情况保留原结果，未按基准缩放。逐城差值以百分点写入资源汇总。

## 负荷与气候用电影响

使用v2已核验的12份10 MW中央IT曲线，读取时逐一核对原manifest SHA。CPU采用SPECpower方法，GPU采用MIT实测功率/TDP的p50迁移假设；1,574个随机前缀文件的暂定样本边界继承v2，未将跨代、跨用途功率比例视为机型实测。

HPC/SAI/TRN/INF分别重放原轨迹309/237/90/1天窗口，各年从原窗口首小时开始，2024完整保留闰日。2025 IT曲线与v2原曲线逐值核对。各城市使用相同UTC相位，因此城市差异来自天气与冷却；IT轨迹的季节变化来自源窗口顺序重放，不能解释为当地需求季节性，也未假设各国业务按同一当地钟表到达。

10 MW是参考设计目标，整机/整柜数量向下取整，实际安装参考IT容量写入 `installed_reference_it_mw`。A–D采用air，E采用direct_liquid；中央参数为主，低/高参数按既有模型完整计算。该液冷模型包含残余空气支路。冷却模型为工程参数情境，未作八城市设施实测校准。

设施功率=IT+冷却+其他辅助；年均PUE=全年设施电量/IT电量，冷却占比=冷却电量/设施电量。最热周按连续168小时干球温度均值最高选取，报告该窗口内设施峰值。年电量为逐小时MW之和乘1 h；三年汇总中电量取年度算术平均，PUE和冷却占比按三年电量加权，峰值取三年最大。

## 可移动负荷的纯物理边界

每类用途扫描参与份额0/25/50/75/100%、额外推迟0/1/6/12/24/48 h和容量倍数r=1/1.5/2，共4×5×6×3=360组；另按12情境换算1,080组单小时IT电量上界。采用2025完整年度基准形状。

服务量以CPU或GPU分配资源小时作为代理。原始逐作业到达、原执行时长和截止时间未在本阶段识别，将原执行小时定义为参考服务到达小时，并仅允许从该小时向后额外推迟D小时。固定部分为(1−f)a(t)，可调部分到达为fa(t)，总执行不超过r。累计服务满足方案规定的 A(t−D)≤S(t)≤A(t)，年初无积压、年末全部完成；未将工作移出报告期。模型允许可分割、可中断任务，不含检查点、连续运行和最小并行规模约束，因而为技术上界。

`independent_max_removable_*` 是逐个小时分别求取的最大移出量。不同小时的上界不能同时实现，不能相加为全年可转移电量。电量上界保留原有空闲和背景功率，移除的任务按CPU/GPU同步比例代理换算；未计冷却减量、恢复能耗或新增容量待机能耗。GPU用途仅约束主要GPU资源，配套CPU和内存的联合瓶颈尚未识别。

日内/周内峰谷差由全年累计约束下的线性规划分别最小化，单位为基准计算容量的分数。日为UTC连续24小时，周为从1月1日起连续168小时；不足完整周的年末时段仍受服务和容量约束，但不进入周峰谷差目标及平均值。日内与周内为两次独立平坦化，未叠加收益。

INF扫描是“合格离线部分可参与”的假设；在线推理须维持原服务时刻，对应零参与或零额外推迟。HPC/SAI/TRN的实际参与权限也需业务证据，技术上界不能直接用作可签约柔性。额外容量只在本阶段作为物理约束，不计算费用。

相关性按2025各城当地钟表小时的24个平均形状点计算Pearson r，分别列出原资源形状和独立可移出形状与风光形状的相关性。零参与/零延迟等常量形状的相关性留空，未填写零；相关性依赖原轨迹UTC相位，不能替代后续联合调度或因果结论。

## 图件及图注

全部图件为300 dpi PNG和可检索中文PDF；字体Microsoft YaHei。PDF使用PyMuPDF实际渲染，检查文本边界和替代字形，并逐情境查看版面。记录见 `figure-validation.json`、`figure-overview-validation.json`、`figure-render-checks.json`。

| 图件 | 内容与读图说明 |
| --- | --- |
| figure1-climate-resource-overview | 八城年均温度、固定/跟踪光伏CF、净风电CF及GHI；黑色叉号为公开长期基准，与模型各自口径并列 |
| figure2-city-load-comparison | 12情境×8城市的中央PUE及年度设施电量；各图色标独立，10 MW参考IT规模 |
| figure3-renewable-shapes | 各城当地小时和月份的风光平均容量因子，2023–2025 |
| figure4-flexibility-potential | 日内资源峰谷差可降低百分比；须结合CSV中的原始绝对峰谷差阅读 |
| figure5-independent-hour-removal | 各小时独立最大移出资源量的年均值，占基准容量的百分比；不同小时极值不能同时实现 |

`load-patterns/`下每个可算情境各有三张图，共36张：`*-seasonal-days`为8行城市×4季节，按2023–2025当地钟表小时平均，分别绘制IT/冷却/设施功率；`*-typical-week`为2025共同代表周，按八城IT/冷却/设施/温度标准化特征到全部候选周平均形状的均方距离最小选择，保留真实连续168小时；`*-annual-envelope`为2025设施逐日小时最小–最大包络及日均值。代表周起止和选择距离见 `representative-weeks.csv`。

## 第二阶段输入、复现与后台接续

96份中央逐小时Parquet在项目 `data/processed/eight-city-phase-a-20261029/`，不进入Git；本机该目录指向 `D:\dc-eight-city-phase-a-20261029\data\processed\eight-city-phase-a-20261029`。服务器副本在 `/data3/dataset/ljt/eight-city-weather-20261029/processed/`，文件索引记录SHA。每份文件为连续UTC索引，含风光CF、PUE、IT/冷却/设施功率、CPU/GPU分配份额、IT形状、主要资源形状、新情境标识及旧编号。`it_load_shape`以安装参考IT容量归一化，`demand_resource_fraction`为服务代理；第二阶段接入不同求解器时须明确容量分母和服务单位。

逐作业与逐小时MIT衍生数据仅保留在忽略目录及授权服务器；Git内提供汇总、图件、方法和文件校验索引。

本机计算顺序如下；`BASELINE`为已有v2逐小时目录，`WEATHER`为本任务已核验天气目录，`BENCHMARKS`为本任务公开基准缓存。依赖见 `analysis/requirements-eight-city-phase-a.txt`，服务器获取脚本另外使用既有ERA5环境与CDS凭据，不将凭据复制到仓库。

本次计算将本进程的 `OPENBLAS_NUM_THREADS`、`OMP_NUM_THREADS`、`NUMBA_NUM_THREADS`设为1，避免占用其他后台任务的内存提交空间。

```text
python analysis/build_eight_city_phase_a.py --weather WEATHER --benchmarks BENCHMARKS --baseline BASELINE
python analysis/build_eight_city_flexibility.py --baseline BASELINE
python analysis/summarize_eight_city_phase_a.py --benchmarks BENCHMARKS
python analysis/plot_eight_city_phase_a.py --benchmarks BENCHMARKS
python analysis/archive_eight_city_provenance.py --weather WEATHER --benchmarks BENCHMARKS
python analysis/diagnose_eight_city_wind.py --weather WEATHER --benchmarks BENCHMARKS
python analysis/verify_eight_city_figures.py
python analysis/validate_eight_city_phase_a.py --baseline BASELINE
```

CDS控制器在服务器本任务目录执行 `acquire_eight_city_weather.py`，状态为 `status.json`，日志为 `controller.log`，请求保存在各城 `*.request.json`。接续前检查该控制器进程及已有编号；仅当本任务控制器已退出时使用同目录 `launch_eight_city.py`恢复。控制器用 `.controller.lock`防止双启动，复用已有请求编号，继续服从共享闸门。正式CDS回执补齐后可用 `archive_eight_city_provenance.py`核对与冻结ARCO的逐变量差异；不自动替换已经交付的数据版本。

本阶段覆盖天气和技术潜力。真实作业连续性、业务权限、跨地点迁移、风光储容量、费用及四组配对收益由后续阶段处理。A-CLD输入缺口及MIT随机前缀样本暂定边界保留为已知限制。
