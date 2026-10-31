# 输入与执行记录

- 用户授权：2026-10-31 17:15 北京时间，处理 ASI 全部 185 天与全部任务类型。
- 基线：PR #30 分支 `codex/inference-data-alternatives-20261031` 的 `d65213bf503d1d4b08eccbb108cad5dc5a8611aa`。
- 独立分支：`codex/asi-185day-workload-types-20261031`；worktree：`D:/asi-185day-workload-types-20261031-worktree`。
- 已检查服务器：启动前无本轮 ASI 全量任务或检查点；共享卷可用约 6.49 TB。保留其他任务的目录、分支与进程。
- 服务器代码目录：`/data3/dataset/ljt/global-grid-flexibility-20261023/analysis/asi-185day-workload-types-20261031/`。
- 服务器运行与检查点目录：`/data3/dataset/ljt/global-grid-flexibility-20261023/results/asi-185day-workload-types-20261031/`。
- 后台处理：17:25:50–19:25:42 北京时间，`setsid nohup`；前期八个工作进程，复用 2,198 小时检查点后切换十六个进程。全部 185 天、4,440 小时及执行跨度表完成，原始文件不落盘。主进程与后台汇总、质量验证进程均已退出；服务器状态为 complete。
- 初始程序与续跑审计提交为 `8ce7238`、`83ea075`；完整统计、质量标识与年度接入均属同一需求。`restart-events.json` 保留切换前状态、信号与版本。
- 首小时真实数据与续跑复用验证通过；独立 pandas/Arrow 资源汇总、pod/workload 精确去重、规模直方图验证通过。运行时长表已完成读取。
- 核心程序按小时原子写入 `checkpoints/hour-XXXXX.json`；`status.json`、`progress.jsonl`、`runs.jsonl`、`run.log` 可用于检查进度。重启前读取状态与 PID，已有进程仍运行时不重复启动。
- 已完成项：全量资源与执行跨度统计；33 个服务器汇总文件字节/SHA 读回；独立样本和端点对照；规模分布守恒；INF 三条源曲线与全年资源重放；共同零资源小时标识原样传播。详见 README、processing-audit、quality-summary 和 validation。
- 遗留证据边界：相对日 5 的 6 时全源资源为零的原因；归属量超过库存的口径；执行跨度表起止/截尾信息；源日历与季节性；硬件性能/功率和 SLA。数据与接口保留这些限制，未自动补造或重解释。
- Git 交付：从 PR #30 派生独立 PR，审阅顺序为先 #30 再本交付；保持待审，不合并。中断恢复时先检查该分支 PR 状态及服务器 complete 状态，避免重新启动全量读取。
