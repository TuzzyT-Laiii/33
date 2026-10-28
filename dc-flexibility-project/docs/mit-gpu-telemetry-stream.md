# MIT Supercloud GPU 监测流式汇总

本流程处理公开 S3 的 `datacenter-challenge/202201/gpu/` 全部对象。原始 CSV 在本机逐个下载、核验、分块计算并删除，只持久保存可合并汇总。代码、说明进入公开仓库；CSV、SQLite、逐文件元数据、逐作业汇总、核验明细和分析结果保存在本机与研究服务器的私有目录。

## 输入与依据

- [AWS 官方登记](https://registry.opendata.aws/dcc/)：数据来源、引用和 CC-BY-NC-ND-4.0 许可。
- [202201 原版 README](https://mit-supercloud-dataset.s3.us-west-2.amazonaws.com/datacenter-challenge/202201/README.md)：Slurm、TRES 与 GPU 监测字段，标称 100 ms 采集周期。
- [官方入门 notebook](https://github.com/MIT-AI-Accelerator/MIT-Supercloud-Dataset/blob/main/notebooks/Intro-to-MIT-Supercloud-Dataset.ipynb)：通过文件名中的作业编号选择 GPU 文件，再按 `gpu_index` 分离设备。
- [官方数据说明](https://github.com/MIT-AI-Accelerator/MIT-Supercloud-Dataset/blob/main/README.md)：TRES `1001=gpu:tesla`、`1002=gpu:volta`。

文件名实测遵循 `<id_job>-<node>.csv`，节点格式为 `r<数字>-n<数字>`。实现使用整段正则匹配，将首段编号作为字符串精确连接 `slurm-log.csv.id_job`，检查节点属于该作业 `nodelist`。节点与 `gpu_index` 联合区分设备。README 没有逐字定义文件名语法；该语法通过完整对象清单与 Slurm 实证检查，保存在 `input-audit.json`。未映射记录仍汇总并单独计数。

原始 Slurm、README、LICENSE、TRES 从已有服务器归档复制至 `reference/`，使用此前经 S3 ETag 校验的 `acquisition.json` 核对字节数及 SHA-256。`audit` 同时记录完整随机序列的 SHA-256、全部文件名的匹配情况，以及 multipart 对象的分段数检查。原归档保持原样。

`202201` 是发布版本，不能将其解释为 2022 年 1 月观测。试点 100 个文件的首尾 Range 核验发现，原始 GPU 时间戳比 Slurm epoch 系统性早 4 小时（夏令时）或 5 小时（冬季）；采用 IANA `America/New_York` 的季节偏移后，首个样本相对 Slurm 开始的最大绝对残差为 1.638 秒，末尾相对 Slurm 结束为 37.058 秒。该时钟解释是本次数据一致性核验的推断，官方 README 没有声明此转换规则。`audit_mit_gpu_clock.py` 可重复核验并保存 `clock-boundary-audit.json`。

汇总统一到 Slurm 参考时钟，再按 UTC 坐标划分小时与月份；文件元数据保留原始时间范围、偏移、转换状态及首尾残差。跨越夏令时切换、无法用一个偏移明确定位的作业保留原始时钟汇总，标记 `dst_transition_requires_review`，从分配窗口主分析中单列。当前时钟推断不作为连接当地气象的依据。

## 统计口径

每个原始样本在时间戳所在的 GPU×小时中计数一次。对 `utilization_gpu_pct`、`utilization_memory_pct`、`power_draw_W`、`temperature_gpu`、`temperature_memory` 分别保存有效样本数、和、平方和、最大值。缺失或不可解析值在相应指标中剔除，并记录异常数量，其他有效指标仍使用。部分原始文件还有 SM/显存/视频频率和功率上限列；实现接受包含必需监测列的这些表头，并在文件元数据中保存原始列名。CSV 结构错误使该对象重试并最终记为失败，避免静默跳过结构损坏的行。

每个 GPU 的样本向前保持到下一样本，时长最多 1 秒；末尾样本不外推。时间戳以整数微秒计算。分配结束及小时边界截断有效区间，跨小时区间拆分时不增加样本数。保存总有效微秒，以及各指标的有效微秒、时间加权和、时间加权平方和。逐 GPU 的重复/逆序时间和超过 1 秒间隙单独计数。实测间隔的毫秒频数保存在文件元数据中，间隔诊断不用于取代真实时长。

合并时按维度累加样本数、时长、和、平方和，对最大值再取最大值：

\[
\bar x=\frac{\sum x}{n},\qquad s^2=\frac{\sum x^2-(\sum x)^2/n}{n-1}.
\]

时间加权均值使用 `time_sum_us / valid_us`。这些充分统计量能够代数合并，计数和微秒时长使用整数；浮点矩存在通常的舍入误差，独立核验使用明确容差。

直方图包括两套：

1. 精确利用率值×月份×作业分配 GPU 总数×GPU 类型：样本数、微秒时长。用于按月、按作业 GPU 数量分组的时间加权利用率分位数。
2. 利用率 `0`、`(0,5]`、……、`(95,100]` ×功率 `[5k,5k+5)` W：样本数、微秒时长。功率箱从零开始，随实际功率扩展；不截断高功率尾部。

功率 P10/P50/P90 给出分位数所在的完整 5 W 箱及箱中心代表值，精度受箱宽限制。主结果按有效时长加权，另以 `sample_` 前缀列输出直接按直方图计数得到的均值或分位数，供检查采样间隔对结果的影响。空载定义为 `utilization_gpu_pct=0`，其中仍可能有显存占用或系统活动。`utilization_memory_pct` 是源监测字段，不能直接解释为已使用显存容量比例。

利用率主分析要求：文件与作业、节点匹配；GPU 请求数、分配数大于零；分配窗口有效；没有无法定位的暂停区间；该节点观测卡数×节点数不超过作业分配卡数；样本在 `[time_start,time_end)` 内。其他记录保留各自原因。节点间分配按同构设备及均分关系核对，数据未提供每张卡的 Slurm 设备分配清单。结果代表有监测覆盖且能够确认分配归属的 GPU 时间。不能给缺失监测的分配小时补零，也不能仅凭监测样本将覆盖率解释为全体分配 GPU 的完整性。聚合同时报告试点文件所对应节点的计划分配 GPU 小时及观测覆盖比例。

功率曲线同时按 TRES GPU 类型输出。混合类型曲线只描述本数据的样本混合，不能直接作为 H100、GB200 或整个机房的功率曲线。

## 运行与目录

Python 依赖见 `analysis/mit-gpu-telemetry-requirements.txt`。实际运行目录为 `D:\mit-supercloud-gpu-summary-20261028`；代码位于从最新 `origin/main` 建立的独立 worktree。C 盘空间不足以承担下载缓存，因此数据运行目录放在 D 盘。

```powershell
$analysis = 'C:\Users\Lantern567\Documents\ChatGPT\ljt的文章\.codex-temp\mit-gpu-telemetry-stream-20261028\dc-flexibility-project\analysis'
$runRoot = 'D:\mit-supercloud-gpu-summary-20261028'
$pythonExe = 'D:\anaconda\python.exe'

New-Item -ItemType Directory -Force "$runRoot\reference" | Out-Null
$sourceRoot = 'pku-test:/data3/dataset/ljt/lisa-gpu-demand-20261028/data/raw/mit-supercloud-202201'
foreach ($name in @('README.md','LICENSE','slurm-log.csv','tres-mapping.txt','acquisition.json')) {
    scp "$sourceRoot/$name" "$runRoot\reference\$name"
}

& $pythonExe "$analysis\stream_mit_gpu_telemetry.py" prepare --root $runRoot
& $pythonExe "$analysis\stream_mit_gpu_telemetry.py" audit --root $runRoot
& $pythonExe "$analysis\audit_mit_gpu_clock.py" --root $runRoot --files 100
& $pythonExe "$analysis\stream_mit_gpu_telemetry.py" benchmark --root $runRoot
& "$analysis\manage_mit_gpu_telemetry.ps1" start -Root $runRoot -Concurrency 16 -Limit 100
```

试点状态显示 `finished`、`completed=100`、`failed=0` 后，运行核验、聚合和同步：

```powershell
& $pythonExe "$analysis\verify_mit_gpu_telemetry.py" --root $runRoot --seqs 0 2 13 14 50 98
& $pythonExe "$analysis\aggregate_mit_gpu_telemetry.py" --root $runRoot --output "$runRoot\aggregates\pilot100"
& $pythonExe "$analysis\stream_mit_gpu_telemetry.py" sync --root $runRoot
```

吞吐测试各档使用相同的 32 个真实对象，每个 Range 读取 2 MiB，共 64 MiB；比较 4、8、16 并发。单次请求连接超时 10 秒、无数据超时 30 秒、持续时间上限约 90 秒，并逐请求记录成功或错误。选择吞吐达到最佳值 90% 且全部成功的最高并发档，上限 16。全量处理的持续有效吞吐和预计剩余时间以 `progress.json` 为准。

2026-10-28 实测结果：4 并发 1.116 MiB/s（1 个请求超时），8 并发 1.536 MiB/s（全部成功），16 并发 2.068 MiB/s（全部成功），选用 16。S3 完整清单包含 109,946 个对象、2,436,905,883,610 字节；其中 109,892 个对象匹配 Slurm 作业及节点，54 个节点不匹配。随机种子为 `20261028`。

试点处理固定随机顺序的前 100 个对象，共 3,032,302,993 字节、61,701,680 行，全部完成且解析异常数为零。六个文件使用独立 CSV 逐行重算，覆盖 1,661,682 行和 6,613 项统计检查；全部通过。全部 100 个文件的样本计数、有效微秒、统计矩与直方图守恒检查通过。实测采样间隔 P10/P50/P90 为 103/106/109 ms。100 个文件均保留汇总，主估计采用 99 个时钟可定位文件，另一个跨夏令时切换的文件单列。试点结果及验证记录保存在私有运行目录。

试点核验完成后，取消 `-Limit` 启动全量任务，SQLite 会自动跳过已提交的试点对象：

```powershell
& "$analysis\manage_mit_gpu_telemetry.ps1" start -Root $runRoot -Concurrency 16
& "$analysis\manage_mit_gpu_telemetry.ps1" status -Root $runRoot
& "$analysis\manage_mit_gpu_telemetry.ps1" stop -Root $runRoot
# 等待 status 显示 stopped，进程退出后续跑。
& "$analysis\manage_mit_gpu_telemetry.ps1" resume -Root $runRoot -Concurrency 16
& $pythonExe "$analysis\aggregate_mit_gpu_telemetry.py" --root $runRoot
& $pythonExe "$analysis\stream_mit_gpu_telemetry.py" sync --root $runRoot
```

启动器使用隐藏窗口 `Start-Process`，进程独立于当前会话。`launch.json` 记录 PID 和标准输出/错误日志路径；`run.json` 记录运行参数；`progress.json` 每约 15 秒原子更新。安全停止写入 `STOP` 文件，等待当前最多 16 个文件完成再提交退出。紧急终止后也可续跑，未提交对象继续下载，完整批次通过清单恢复。操作系统文件锁阻止同一目录启动两个工作进程。

运行目录内容：

| 路径 | 内容 |
| --- | --- |
| `state.sqlite` | 完整对象顺序、key、ETag、字节数、pending/done/failed、尝试次数、失败原因、批次 |
| `cache/*.part` | 正在下载或计算的 CSV，计算及内部守恒检查后立即删除 |
| `batches/<最小seq>-<最大seq>/` | 四类 Parquet 及批次清单；序号范围内允许存在属于其他批次的对象 |
| `input-audit.json` | 输入校验、全清单映射检查、随机顺序指纹 |
| `clock-boundary-audit.json` | 试点首尾时间与 Slurm 的独立时钟核验 |
| `inventory.json`、`benchmark.json` | 完整对象总体与吞吐实测 |
| `pilot-validation.json` | 独立 CSV 重算的逐项验证结果 |
| `aggregates/pilot100/`、`aggregates/current/` | 利用率分布、功率曲线、合并直方图及范围说明 |
| `logs/`、`launch.json`、`run.json`、`progress.json` | 日志、进程、参数和持续进度 |
| `sync.json`、`sync-work/` | 最近成功同步、错误原因与临时增量归档 |

下载前按对象完整长度预留空间，下载缓存硬上限 19,000,000,000 字节，同步暂存预留 1,000,000,000 字节，合计不超过 20 GB；另外要求本机至少 10,000,000,000 字节余量。同步增量归档超过其预留空间时明确记录失败并保留本地汇总。空间不足时停止调度并保留断点。每个文件使用 2 MiB Range 分段，可在共享连接池中预取至多 8 段，按字节顺序核验写入；全局网络请求并发严格限制为选定档位，上限 16。单段超时后从已完成分段继续。下载校验包括每段的 `If-Match`、响应 ETag、Content-Range、Content-Length、实际字节数，以及完整文件的 SHA-256、单段 MD5 或 8 MiB multipart MD5。multipart 分段数在全对象清单上提前核对，内容签名在每个对象完整下载后核对。

每个 Range 最多尝试 5 次，间隔按指数增加，最高 120 秒；连续超过上限的对象标记 `failed`，后续对象继续。校验或解析失败的对象同样最多尝试 5 次。单段响应核验完整后才追加到缓存文件，已完成分段在其他分段重试时保留。全部对象从完整 S3 分页清单生成，使用 `random.Random(20261028).shuffle` 固化顺序。下载按固定序号启动，文件完成即可写入批次并补充空闲位置。

聚合只读取第一个 pending 对象之前的连续完成前缀，筛除尚未进入该前缀的异步完成对象。`progress.json.random_prefix_objects` 记录前缀长度；聚合结果记录实际采用的文件数、前缀内失败数以及仍在等待前缀的完成文件数。因此，统计使用的前缀保持简单随机抽样的顺序性质，减少文件大小对完成时间造成的选择偏差。失败对象单独报告；存在失败时，可用样本仍有非响应偏差，不能宣称完整简单随机样本。`completed_bytes` 是持久化字节数，速率按本轮已完成计算的字节数估计，另列尚未提交的计算结果字节数。

批次达到 25 个文件或已有可提交结果等待约 30 秒时，先写 `.partial` 目录，重新读取 Parquet 检查，再原子改名，最后提交 SQLite。进程异常退出后，启动时检查完整批次清单并恢复 SQLite；未完成的暂存批次重新处理。属于 pending 对象的部分下载文件从完整分段边界续传；其他残留缓存删除。原始 CSV 已删除时，未提交对象仍能从 S3 重新获取。

每 30 分钟或每 2,000 个已提交对象启动独立同步线程。同步只上传新增不可变批次、SQLite 一致快照、进度与聚合结果；失败写入 `sync.json`，本机处理继续。服务器目的地为 `/data3/dataset/ljt/mit-supercloud-gpu-summary-20261028/`，同步前保留至少 50 GB 服务器空间。先在 `.incoming/` 接收解包，再逐批原子改名。服务器不下载 S3 原始数据，不操作 ERA5 进程。

独立核验器重新下载试点中的若干真实对象，使用 Python 标准 `csv` 解析和 `Decimal` 时间戳，逐行重算样本矩、时长、时间加权矩、逐小时最大值及两套直方图，不调用生产汇总函数。整数计数和微秒时长要求一致，浮点统计使用 `rtol=2e-10, atol=1e-5`。原始重算文件核验结束立即删除。

聚合在一个 SQLite 读事务内固定批次及随机前缀快照，仅使用完整批次中属于该前缀的记录。两个 CSV 给出按月份、作业 GPU 数量、GPU 类型分组的利用率分布和功率曲线；`summary.json` 附文件数、原始行数、覆盖边界、映射质量、采样间隔和直方图精度。所有结果保持在私有研究目录，公开 PR 只包含实现与方法说明。
