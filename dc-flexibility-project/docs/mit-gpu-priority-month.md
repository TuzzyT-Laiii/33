# MIT GPU 完整月优先处理与小时功率核验

## 选择的月份

优先窗口为 **2021-06-01 00:00 至 2021-07-01 00:00 UTC**，结束边界不包含在窗口内。月份及小时沿用既有 Slurm 参考时钟与 UTC 划分规则，GPU 时钟转换依据见[流式处理说明](mit-gpu-telemetry-stream.md)。跨月作业下载并汇总完整文件，月度分析按小时截取窗口。

完整 S3 文件清单与经 SHA-256 核验的 `slurm-log.csv`、`tres-mapping.txt` 连接后，得到以下候选统计。分配 GPU 小时按月内作业区间积分；监控文件覆盖部分按匹配节点数/作业节点数加权，假定同一作业在节点间均分 GPU。

| 月份（UTC） | 分配 GPU 小时 | 监控文件覆盖率 | 有 GPU 作业活动的日数 | 相关文件数 |
| --- | ---: | ---: | ---: | ---: |
| 2021-04 | 163,534.8194 | 94.320123% | 30/30 | 7,757 |
| 2021-05 | 198,816.7447 | 97.425456% | 31/31 | 15,072 |
| **2021-06** | **245,946.9453** | **99.998398%** | **30/30** | **12,181** |
| 2021-07 | 243,686.0047 | 99.584872% | 31/31 | 16,835 |
| 2021-08 | 251,045.1564 | 99.297100% | 31/31 | 22,925 |
| 2021-09 | 224,987.9944 | 99.837080% | 30/30 | 26,347 |

六个月每天都有有效作业和 GPU 作业活动；月内启动作业的无效执行区间数、GPU 作业空节点列表数及暂停作业分配时长均为零。6 月覆盖率最高，缺少文件对应约 3.9394 GPU 小时。相关监控文件共 **486,454,147,356 字节**，包含所有与窗口重叠作业的文件，也包含需要单独核验的节点映射记录。

这里的覆盖率说明源文件可获得性。文件内缺测、样本间隙、时钟异常和重复设备观测仍影响实测有效时长；逐日活动和字段检查也无法证明发布的 Slurm 记录没有遗漏。候选表、选择规则及输入 SHA-256 保存在[聚合统计](../results/mit-priority-month-20261029/month-selection.json)。

## 调度与续跑

`database()` 为旧 SQLite 自动增加 `objects.priority INTEGER NOT NULL DEFAULT 0` 及调度索引。已有 `seq`、对象标识、状态、尝试次数、批次目录和 Parquet schema version 2 保持原有含义。窗口以 JSON 保存在 `settings.priority_window`，后续省略参数仍沿用该窗口。

`stream_mit_gpu_telemetry.py run --priority-window START END --priority-timezone UTC` 把重叠作业的全部文件标为优先组，按 `priority DESC, seq` 调度，跳过 `done`。两个组内均保持原固定种子的随机序号。调度屏障等待窗口文件全部成功并持久化后才启动其余文件；窗口出现失败时停止进入剩余组并保留待处理状态。

`progress.json.priority_window` 提供 `completed/total`、待处理数、失败数、总字节数、待处理字节数、`complete`、`eta_seconds` 和 `estimated_completion_utc`。ETA 使用待提交字节数除以本轮完整对象的有效吞吐；刚启动时和大文件集中计算时会明显变化。`active_objects` 提供当前对象的序号及优先级。`run.json.script_path`、`launch.json.script` 记录实际运行代码路径。

实际持续任务使用：

```text
数据：D:\mit-supercloud-gpu-summary-20261028
运行代码：C:\Users\Lantern567\Documents\ChatGPT\ljt的文章\.codex-temp\mit-gpu-telemetry-stream-20261028\dc-flexibility-project\analysis
开发 worktree：C:\Users\Lantern567\Documents\ChatGPT\ljt的文章\.codex-temp\mit-priority-month-20261029
开发分支：codex/mit-priority-month-20261029
```

2026-10-29 10:01 CST，原 PID 76620 正常退出：完成 2,294 个文件、零失败、零下载缓存、零未提交字节，收尾服务器同步成功。停止后的 SQLite 副本通过完整性、批次清单、迁移幂等性和顺序检查。部署至上述运行代码目录后，于 10:06:36 CST 启动 **PID 86244、16 并发**。续跑核验确认旧 2,294 个完成对象的全部状态字段、旧批次修改时间保持一致，当前活动对象全部属于优先组，非窗口对象没有新增完成。

```powershell
$analysis = 'C:\Users\Lantern567\Documents\ChatGPT\ljt的文章\.codex-temp\mit-gpu-telemetry-stream-20261028\dc-flexibility-project\analysis'
$runRoot = 'D:\mit-supercloud-gpu-summary-20261028'
& "$analysis\manage_mit_gpu_telemetry.ps1" status -Root $runRoot
& "$analysis\manage_mit_gpu_telemetry.ps1" stop -Root $runRoot
# 等待原进程退出，progress 显示 stopped 且未提交字节为零，再部署和恢复。
& "$analysis\manage_mit_gpu_telemetry.ps1" resume -Root $runRoot -Concurrency 16 `
    -PriorityStart '2021-06-01' -PriorityEnd '2021-07-01' -PriorityTimezone UTC
```

原随机前缀聚合器仍按原序号选择连续前缀。月度分析单独读取窗口所有已提交对象，部分完成的窗口结果保留选择偏差说明。典型曲线 v2 的分支、worktree 和输出目录保持其各自状态。

## 实测与估算

[原始论文](https://arxiv.org/pdf/2108.02037)第 III 节记录 224 个 GPU 节点、每节点 2 张 V100，GPU 监测随作业开始和结束。月度脚本以 **448 张 GPU** 为库存参考，接受 `--capacity` 显式设置；分配量超过库存时直接报错。[NVIDIA V100 规格](https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf)提供 PCIe 250 W 和 SXM2 300 W 的上限。脚本默认 250 W 作为比例归一化参考，未将其认定为源设备实际设置；同一 TDP 出现在比例分母及估算乘数中，二者代数抵消。

月度算法使用一致的 SQLite 读快照，读取其对应不可变批次，按 `node × gpu_index × hour` 合并时间加权功率矩。仅使用节点匹配、时钟可定位、分配归属明确的窗口内观测。多个作业先合并同一设备的有效时长与功率积分；不能按作业直接累加设备均值。疑似跨作业时间重叠或有效时长超过一小时的设备小时单列并排除。已有小时汇总无法恢复这些记录的精确采样区间并集。

| 输出列 | 含义 |
| --- | --- |
| `allocated_gpu_count` | Slurm 分配 GPU 小时/1 小时，即小时内平均分配卡数 |
| `file_coverage` | 对应源监控文件可获得的分配 GPU 小时比例 |
| `monitoring_coverage` | 实测有效 GPU 小时/分配 GPU 小时 |
| `observed_energy_equivalent_W` | 已覆盖 GPU 观测功率积分/整小时；缺测小时为空 |
| `sum_device_mean_power_W` | 各已观测设备在各自覆盖区间内求均值后相加，随覆盖范围解释 |
| `whole_cluster_measured_W` | 所有库存 GPU 的整个小时均有观测时才填写 |
| `estimated_allocated_W` | 已分配 GPU 数 × 标定实测功率/TDP 比例 × 同一 TDP |
| `estimated_cluster_W` | `estimated_allocated_W + (库存 − 已分配GPU数) × 空载代理功率` |
| `measured_plus_idle_proxy_W` | 达覆盖率阈值的已分配实测积分功率 + 同一未分配空载代理 |

前半月（6 月 1–15 日）标定分配 GPU 的时间加权平均功率/TDP，后半月验证，避免直接用验证小时标定自身。空载代理取前半月内完整作业的 `GPU utilization=0` 功率直方图，按有效时长计算 5 W 箱中心均值，箱宽引入每卡至多 2.5 W 的代表值误差。该代理仍可能含显存和系统活动；也可用有依据的 `--idle-watts` 输入。

主误差、Pearson 相关性及日内/周内形状使用后半月监控覆盖至少 99%、没有疑似重叠的共同小时。另给出相同已观测 GPU 时长范围内的诊断，用于部分数据试跑。形状默认 UTC，可用 `--shape-timezone America/New_York` 输出当地时钟下的日内/周内分组；窗口边界仍由 `--timezone` 单独定义。

源作业文件缺少未分配 GPU 的持续监控。因此，完整月文件处理完成后，仍可能没有任何小时能提供整个物理集群的纯实测功率。`whole_cluster_measured_W` 与相应误差保持空白。`measured_plus_idle_proxy_W` 的空载部分属于模型假设，与估算曲线共用该部分时也无法验证空载假设本身。所有结果均提供实际覆盖率。

## 分析、核验与服务器输出

```powershell
$pythonExe = 'D:\anaconda\python.exe'
# 已有部分批次诊断。
& $pythonExe "$analysis\analyze_mit_gpu_priority_month.py" --root $runRoot --allow-partial --sync
# 窗口完成后运行；未完成会明确报错。
& $pythonExe "$analysis\analyze_mit_gpu_priority_month.py" --root $runRoot --sync
# 也可在独立后台进程等待全部成功后自动分析和同步，每30秒检查一次。
& $pythonExe "$analysis\analyze_mit_gpu_priority_month.py" --root $runRoot --wait-for-completion --sync
```

输出目录为 `D:\mit-supercloud-gpu-summary-20261028\analysis\priority-month\`，服务器对应 `/data3/dataset/ljt/mit-supercloud-gpu-summary-20261028/analysis/priority-month/`。既有同步流程接收此目录，继续保留至少 50 GB 服务器空间，并保持原批次不可变。

- `hourly-cluster-power.csv`：720 个小时的分配卡数、监控覆盖、实测积分、比例估算及误差。
- `device-hours.parquet`：设备小时合并、覆盖和疑似重复诊断。
- `diurnal-shape.csv`、`weekly-shape.csv`：达到阈值的验证小时形状；`*-observed-support-shape.csv` 保存部分观测范围诊断。
- `summary.json`、`verification.json`：聚合统计、方法限制和独立核验。
- `snapshot.json`：私有对象快照；`month-selection.json`、`month-candidates.csv` 保存候选月选择依据。
- `watcher.json`：自动等待进程最近检查时间与窗口状态计数。

`verify_mit_gpu_priority.py` 在新的数据库副本中验证迁移、组内顺序、完成对象排除、配置幂等及全部批次状态。独立真实 S3 试验处理三个既有真实对象，共 957,772 字节，验证优先组提交屏障及再次启动不重复下载。每次分析后自动调用 `verify_mit_gpu_priority_month.py`，使用独立事件积分验证小时分配，复核设备积分守恒、缺测空白、估算公式、误差、相关性及形状计数；核验成功后再同步。核验器也可独立运行。

数据按归档 LICENSE 的 CC BY-NC-ND 4.0 边界保存。仓库仅提交代码、说明、候选月及试跑的聚合统计；小时、设备、作业和完整对象快照保存在私有本机及服务器目录。
