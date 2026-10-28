# 合格网格的 ERA5 获取与源数据校验

天气范围来自保护地与坡度联合筛选的最终清单。0.5°网格采用坡度≤5°、保护地之外合格土地总面积≥1 km²，未知条件不进入下载。2026年10月26日的清单包含60,430个合格网格、257,354个必要原生节点，对应18,422个源空间块。

源空间块由ERA5压缩布局决定：每个空间块包含4×4个0.25°节点，可供多个研究网格共用。每个研究网格需要九个原生节点，边界节点参与源块集合计算。请求范围只包含该集合；源压缩块中附带的其他小时和节点保留原样。天气覆盖2023—2025年26,304个分析小时及2026年1月1日00:00的右边界。

## 原始文件版本

ECMWF的源文件按日延长时间轴，包含研究窗口以后的数据。压缩文件SHA变化时，服务器对新旧原文分别使用冻结Zarr布局解码，逐字节比较研究窗口内全部16个节点的float32数组，包括缺失位置。研究窗口发生任何变化都会停止发布。

相同研究数值的不同压缩版本均保留：冻结原文继续使用，新原文按SHA归档到`.source-variants/`，`weather-source-versions/`记录两个原文SHA、字节数、研究窗口数组SHA和缺失数量。中转完成回执记录实际保留的原文字节，发布前继续核对来源、传输SHA、压缩包对象集合、合格空间块和冻结元数据。

## 官方风数据定向补取

`lat-068-lon-228/u100`在ARCO原文中存在真实小时缺失。正式ERA5单日探针已经在同一区域12个必要节点取得有限的`u100`和`v100`。年度定向请求只涉及该区域的这两个变量，原始文件和请求编号保存到`data/raw/era5-eligible-cds-repairs/`；断开连接后使用原请求编号继续获取。

年度原文返回后，程序检查完整小时、精确坐标、单位、float32类型、全部有限值和现有有效时段的绝对差。重合值采用绝对容差`1e-4`，露点单位为K，风分量单位为m/s；每个变量的证明记录重合值数量、最大与均值绝对差、单位、容差及检查结果，超过容差仍然失败。只有全部通过，才能在正式整理时用同一时间、同一节点的官方原文补齐NaN，保留已有有效数值。原ARCO原文不修改，输出检查点记录补取文件SHA、补取单元数、清单SHA和输出SHA。校验失败会记录状态并终止，未完成的块不会生成完成标记。

`complete_era5_wind_repair.py`依次执行获取、校验、生成正式块和恢复既有`era5-eligible`任务。控制进程独立于SSH会话运行；它与原始数据中转分别记录状态。年度CDS请求排队期间，原始数据下载可以继续。

## 官方露点温度定向补取

`lat-084-lon-290/d2m`的10个授权天气节点在2023年1月1日00:00至2024年6月30日23:00（UTC）各缺少13,128小时，共131,280个NaN。2023年的官方原文已取得。2024年7月冻结原文中的7,440个露点值全部有效，作为独立重合月份。

补取使用CDS的`reanalysis-era5-single-levels`产品，仅请求`2m_dewpoint_temperature`及该源块内授权节点的包围区域。当前控制器使用`--monthly --missing-only`，只继续获取2024年1—7月；1—6月补齐NaN，7月与冻结原文逐值比较。既有2023官方原文复用；2025年及2026年右边界直接保留冻结原文有效值。生成的数值表仍覆盖全部26,305小时、精确节点、K单位、float32和全部有限值；原有效值全部保持一致。

`--group dewpoint`使用独立的露点状态文件、进程锁和持久请求编号。控制器依次执行官方原文下载、单位与小时校验、有效重合值绝对容差校验、正式块生成和接续既有整理任务。已有精确一致的风数据证明保持兼容，补取证明的变量集合必须与当前变量匹配。补取表和正式输出中的原有效值继续逐值完全相等。

```bash
ROOT=/data3/dataset/ljt/global-grid-flexibility-20261023
"$ROOT/.venv/bin/python" "$ROOT/analysis/complete_era5_wind_repair.py" --root "$ROOT" --block lat-084-lon-290 --group dewpoint --monthly --missing-only --start-formal
"$ROOT/.venv/bin/python" "$ROOT/analysis/verify_era5_dewpoint_repair.py" --root "$ROOT"
```

露点状态为`status/era5-dewpoint-repair.json`和`status/era5-dewpoint-repair-controller.json`。真实验证脚本核对露点原始缺失、已有官方风原文的填补结果、有效值保留及变量证明隔离；露点官方文件和逐值校验未完成时，结果明确记录为待完成。

月度原文、独立请求编号及SHA回执保存在`dewpoint-monthly/`。2024年1月沿用既有编号；2025年1月请求撤销并归档。后续月份按顺序获取，队列满时保留当前编号；不重复提交相同请求。原年度模式保留用于风数据恢复及历史命令兼容。

控制器在补齐并严格校验后生成正式块，再通过`grid_job_manager --ensure-running`接续整理。管理器已经运行时复用现有进程；每个空间块有独立发布锁，已有完成回执继续复核复用。CDS接口未提供预计等待时长时，不推算队列剩余时间。

```bash
"$ROOT/.venv/bin/python" "$ROOT/analysis/complete_era5_wind_repair.py" --root "$ROOT" --block lat-084-lon-290 --group dewpoint --monthly --missing-only --start-formal
```

按月请求方式参考[ECMWF CDS效率建议](https://confluence.ecmwf.int/spaces/CKB/pages/174856258/Climate+Data+Store+CDS+documentation)。切换策略前先核对并结束原控制器及其补取子进程，确认退出后启动唯一新控制器；服务端请求和已下载原文继续保留。

## 按块隔离与整体完成

`weather-block-quarantine.json`明确记录已核查的露点隔离块、授权节点、冻结清单SHA、缺失数量及原因。正式工作进程继续整理其他块。遇到新的、形状与float32正确且仅含NaN的源数据缺失时，记录`weather-block-defects/<block>.json`后继续；其他数据类型、无穷值、文件损坏、SHA、磁盘及环境错误仍明确终止。

隔离块不生成完成回执、不计入完成块数。最后仍有隔离块时，工作进程返回75，管理器和监控显示`waiting_for_repairs`。只有全部18,422块和257,354节点通过完整校验，才能写入整体完成状态。补取证明通过后，同一块的锁保证控制器和正式工作进程共同复用唯一输出；管理器启动锁及恢复请求处理负责衔接结束阶段的并发启动。

`monitor --check`同时给出真实完成回执数量、其余块进度、隔离数量和进程身份。每小时提醒保持停用，进度按需读取。

按需运行`read_era5_formal_progress.py --root ROOT`，读取它实际生成的`results/buildability-20261025/formal-progress-latest.json`。分别报告`other_blocks_completed/other_blocks_target`、`isolated_pending_blocks`中的变量和补取证明状态、露点控制器的月份状态以及两个替代来源的结果。`formal_completed_blocks`采用实际完成回执及对应Parquet文件，避免将重启时的逐个复核计数误报为文件减少；本次状态统计不重新计算全量SHA。整体完成始终需要全部18,422块。

## 并行替代来源

`try_era5_dewpoint_alternatives.py --source timeseries`使用独立数据集`reanalysis-era5-single-levels-timeseries`，按10个精确节点顺序请求。2023年覆盖用于校验已有官方原文，2024年1—7月用于补齐和重合比较。返回的ZIP由标准`zipfile`读取，原始压缩包及SHA保留。

`--source ncar`使用匿名AWS公共桶`nsf-ncar-era5`的`e5.oper.an.sfc`月文件和`VAR_2D`露点，通过HTTP Range读取10个节点。先比较2023年1月官方原文和2024年7月冻结原文；每个重合月均有7,440个数值。任何差异均记录最大值、均值、分位数和不相等数量，禁止发布。网络失败单独记录，不作为数值一致的证据。

只有两个完整重合月完全一致、全部NaN可补齐、原有效值全部保持、小时/坐标/单位/float32全部通过后，候选来源才竞争同一证明发布锁。首先通过者生成正式块，其他控制器复用已核验结果。

```bash
"$ROOT/.venv/bin/python" "$ROOT/analysis/try_era5_dewpoint_alternatives.py" --root "$ROOT" --source timeseries
"$ROOT/.venv/bin/python" "$ROOT/analysis/try_era5_dewpoint_alternatives.py" --root "$ROOT" --source ncar
"$ROOT/.venv/bin/python" "$ROOT/analysis/verify_era5_block_isolation.py" --root "$ROOT"
"$ROOT/.venv/bin/python" "$ROOT/analysis/read_era5_formal_progress.py" --root "$ROOT"
```

点位请求格式来自[ECMWF示例](https://era-explorer.climate.copernicus.eu/notebooks/content/temperature/temperature_annual_timeseries.html)，NCAR桶及数据布局来自[AWS官方目录](https://registry.opendata.aws/nsf-ncar-era5/)。

## 运行与核查

复用已有服务器Python环境，参考依赖见`analysis/requirements-era5-screened.txt`。需要已有最终选择清单、分批选择清单及冻结源元数据；本次运行不重新执行用地筛选。凭据通过既有CDS配置读取，SSH使用已有别名与主机校验，凭据内容不进入日志或仓库。

```powershell
python analysis/prefetch_eligible_weather_to_server.py --root . --workers 4 --stream
```

```bash
ROOT=/data3/dataset/ljt/global-grid-flexibility-20261023
"$ROOT/.venv/bin/python" "$ROOT/analysis/complete_era5_wind_repair.py" --root "$ROOT" --block lat-068-lon-228 --start-formal
"$ROOT/.venv/bin/python" "$ROOT/analysis/verify_era5_recovery.py" --root "$ROOT"
```

恢复命令只应在确认相应进程未运行后执行。既有筛选程序和任务管理器仍提供输入与进程管理；本次归档包含中转所需的共用来源校验和SSH依赖。原始数据、凭据和本机连接配置均不提交。

风数据控制器按块使用`era5-wind-<block>-repair.json`、`era5-wind-<block>-repair-controller.json`及相应进程锁，优先复用已下载及已受理的年度请求，再顺序获取其余2023、2024、2025年度和2026年右边界。四个块可独立运行，持久请求编号和原文按块保留，重启时复用。遇到明确的队列满拒绝时归档编号及官方原因，每60秒检查官方队列，已有排队请求离开队列后才重新提交；其他拒绝仍然失败。整体整理状态仍为`era5-eligible-acquisition.json`。`verify_era5_recovery.py`使用现有真实原文验证源版本保留、数值改变拒绝、合格空间块集合以及官方单日探针，并单独报告正式年度块是否验证完成。`verify_era5_overlap_repair.py`使用实际7月原文核对容差、错时真实数值的拒绝、NaN补齐、原有效值保留及正式输出。原文下载完成、正式整理完成和研究计算完成分别报告。

## 资料

- [ECMWF ERA5 ARCO产品说明](https://confluence.ecmwf.int/pages/viewpage.action?pageId=699689222)
- [CDS接口与请求状态说明](https://confluence.ecmwf.int/pages/viewpage.action?pageId=656872232)
