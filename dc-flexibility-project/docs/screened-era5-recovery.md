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
