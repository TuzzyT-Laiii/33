# 合格网格的 ERA5 获取与源数据校验

天气范围来自保护地与坡度联合筛选的最终清单。0.5°网格采用坡度≤5°、保护地之外合格土地总面积≥1 km²，未知条件不进入下载。2026年10月26日的清单包含60,430个合格网格、257,354个必要原生节点，对应18,422个源空间块。

源空间块由ERA5压缩布局决定：每个空间块包含4×4个0.25°节点，可供多个研究网格共用。每个研究网格需要九个原生节点，边界节点参与源块集合计算。请求范围只包含该集合；源压缩块中附带的其他小时和节点保留原样。天气覆盖2023—2025年26,304个分析小时及2026年1月1日00:00的右边界。

## 原始文件版本

ECMWF的源文件按日延长时间轴，包含研究窗口以后的数据。压缩文件SHA变化时，服务器对新旧原文分别使用冻结Zarr布局解码，逐字节比较研究窗口内全部16个节点的float32数组，包括缺失位置。研究窗口发生任何变化都会停止发布。

相同研究数值的不同压缩版本均保留：冻结原文继续使用，新原文按SHA归档到`.source-variants/`，`weather-source-versions/`记录两个原文SHA、字节数、研究窗口数组SHA和缺失数量。中转完成回执记录实际保留的原文字节，发布前继续核对来源、传输SHA、压缩包对象集合、合格空间块和冻结元数据。

## 官方风数据定向补取

`lat-068-lon-228/u100`在ARCO原文中存在真实小时缺失。正式ERA5单日探针已经在同一区域12个必要节点取得有限的`u100`和`v100`。年度定向请求只涉及该区域的这两个变量，原始文件和请求编号保存到`data/raw/era5-eligible-cds-repairs/`；断开连接后使用原请求编号继续获取。

年度原文返回后，程序检查完整小时、精确坐标、单位、float32类型、全部有限值和现有有效时段的逐值一致性。只有全部通过，才能在正式整理时用同一时间、同一节点的官方原文补齐NaN，保留已有有效数值。原ARCO原文不修改，输出检查点记录补取文件SHA、补取单元数、清单SHA和输出SHA。校验失败会记录状态并终止，未完成的块不会生成完成标记。

`complete_era5_wind_repair.py`依次执行获取、校验、生成正式块和恢复既有`era5-eligible`任务。控制进程独立于SSH会话运行；它与原始数据中转分别记录状态。年度CDS请求排队期间，原始数据下载可以继续。

## 官方露点温度定向补取

`lat-084-lon-290/d2m`的10个授权天气节点在2023年1月1日00:00至2024年6月30日23:00（UTC）各缺少13,128小时，共131,280个NaN。该块的原始文件已经到齐，正式整理的有限值检查会识别这些空值并停止。

补取使用CDS的`reanalysis-era5-single-levels`产品，仅请求`2m_dewpoint_temperature`及该源块内授权节点的包围区域。年度文件覆盖既定2023—2025年及2026年1月1日00:00右边界；提取和校验仅使用最终筛选清单中的10个节点。三年完整值用于与已有有效时段逐值比较，全部检查通过后仅填充NaN，冻结ARCO原文保留。

`--group dewpoint`使用独立的露点状态文件、进程锁和持久请求编号。控制器依次执行官方原文下载、单位与小时校验、有效重合值完全一致性校验、正式块生成和接续既有整理任务。已有风数据证明保持兼容，补取证明的变量集合必须与当前变量匹配。

```bash
ROOT=/data3/dataset/ljt/global-grid-flexibility-20261023
"$ROOT/.venv/bin/python" "$ROOT/analysis/complete_era5_wind_repair.py" --root "$ROOT" --block lat-084-lon-290 --group dewpoint --start-formal
"$ROOT/.venv/bin/python" "$ROOT/analysis/verify_era5_dewpoint_repair.py" --root "$ROOT"
```

露点状态为`status/era5-dewpoint-repair.json`和`status/era5-dewpoint-repair-controller.json`。真实验证脚本核对露点原始缺失、已有官方风原文的填补结果、有效值保留及变量证明隔离；露点官方文件和逐值校验未完成时，结果明确记录为待完成。

露点补取可增加`--monthly`参数，将尚未到齐的2024、2025年拆分为每月请求，按顺序提交和下载，遵循CDS排队数量限制。已经到齐的2023年文件直接复用，既定2026年1月1日00:00右边界仍须获取。保留原年度请求记录；年度先完成时下载并复用年度原文，该年度后续月份停止提交。若年度请求触发队列数量上限，在核对变量、区域、年份和等待状态后撤销已被月度方案替代的年度请求，完整归档原请求与撤销回执。月度原文及其独立请求、SHA回执保存在`dewpoint-monthly/`，正式证明只采用一套无重复的年度或月度数据。

月度文件仍须覆盖原有全部26,305整点；完整拼接、单位、节点、float32、全部有限值以及所有已有有效值的逐值一致性检查继续执行。`verify_era5_dewpoint_repair.py`逐份检查实际到齐月度原文，并报告小时数、SHA和有效重合值数量。控制器保持先校验、再生成正式块、再接续`era5-eligible`的执行顺序。CDS接口未提供预计等待时长时，不推算队列剩余时间。

```bash
"$ROOT/.venv/bin/python" "$ROOT/analysis/complete_era5_wind_repair.py" --root "$ROOT" --block lat-084-lon-290 --group dewpoint --monthly --start-formal
```

按月请求方式参考[ECMWF CDS效率建议](https://confluence.ecmwf.int/spaces/CKB/pages/174856258/Climate+Data+Store+CDS+documentation)。切换策略前先核对并结束原控制器及其补取子进程，确认退出后启动唯一新控制器；服务端请求和已下载原文继续保留。

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

状态文件为`era5-wind-repair.json`、`era5-wind-repair-controller.json`和`era5-eligible-acquisition.json`。`verify_era5_recovery.py`使用现有真实原文验证源版本保留、数值改变拒绝、合格空间块集合以及官方单日探针，并单独报告正式年度块是否验证完成。原文下载完成、正式整理完成和研究计算完成分别报告。

## 资料

- [ECMWF ERA5 ARCO产品说明](https://confluence.ecmwf.int/pages/viewpage.action?pageId=699689222)
- [CDS接口与请求状态说明](https://confluence.ecmwf.int/pages/viewpage.action?pageId=656872232)
