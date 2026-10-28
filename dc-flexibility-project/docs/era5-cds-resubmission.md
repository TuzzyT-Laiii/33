# ERA5 补取请求恢复

`prepare_era5_wind_repair.py` 使用 `era5_cds_request.py` 管理年度和月度持久请求。在读取编号及轮询状态时，HTTP 404 的官方 JSON 正文明确包含 `job not found` 或 `job … dismissed`，请求进入归档和重新提交流程。状态为 `rejected` 或 `failed` 时，仅对 CDS 明确给出的以下队列限制重新提交：

> Maximum number queued requests for this dataset has been reached. Please configure your scripts to request data sequentially

HTTP 401/403、其他 HTTP 404、参数错误及其他失败原因直接报告错误。重新提交沿用原数据集及全部请求参数，服务端返回的数据集、区域、年份和变量仍需与本地请求一致。

作废请求最多重新提交 5 次，依次退避 60、120、240、480、900 秒。次数、下一次提交时间及待提交标志写入原 `.request.json`，进程重启继续沿用。提交前扣减预算，获得新编号后立即写入该文件。达到上限时报告错误，保留失败编号和归档。

队列名额已满单独进入 `waiting_for_queue_slot`，不扣减上述 5 次预算。等待以约 300 秒起步，每次翻倍，加入 ±10% 随机抖动并限制每次最多 3600 秒；累计等待上限为 72 小时。等待次数、累计秒数、当前等待起点、期限和下一次检查时间持久保存，进程重启继续计时。受理后的处理时间不计入队列等待；再次遇到名额限制时沿用已累计的等待时间。达到期限后报告 `queue_wait_exhausted` 并退出。

旧状态自动迁移到 `retry_budget_version=2`：保留全部 `history` 和归档，以 `legacy_retry_count` 保留旧计数，从普通重提预算中扣除已因队列限制发生的重新提交。若当前正在等待队列名额，清除旧的 `retry_exhausted`，并从历史首次队列拒绝时间继续计算期限。

全部新版补取控制器共用 `ROOT/.era5-cds-request-gate/submission.lock`。锁内完整分页读取当前账号的 CDS `accepted/running` 请求，只计入 `reanalysis-era5-single-levels`，检查名额后才提交，持锁保存新编号及令牌。令牌对应的请求未出现在列表时，直接查询其状态，直到官方确认终止或作废才释放名额。已有控制器及账号其他脚本提交的活动请求同样计入名额；已受理请求继续跟踪和下载。

默认并发上限为 **2**。首次初始化时可设置环境变量 `ERA5_CDS_MAX_ACTIVE_REQUESTS`；后续统一读取 `ROOT/.era5-cds-request-gate/config.json` 中的 `max_active_requests` 正整数。需要调整时原子替换此配置文件，全部新版控制器在下一次检查时采用相同上限。初始化时已有请求超过上限，会等待数量降到上限以下；旧进程继续使用其已加载代码，其后续提交仍可能暂时超过此上限。CDS 拒绝详情、认证或队列读取失败会明确报告，认证 401/403 和参数错误直接退出。

每次作废或可重试拒绝均生成 `.request.retired-<旧编号>.json`，记录完整请求、旧编号、CDS 原因、归档时间及重提次数。原请求文件的 `history` 保留完整历史，按块的 `status/era5-wind-<块>-repair.json` 同步记录历史、当前编号、重提次数和下一次提交时间。后续状态更新和进程重启保留历史。

风速年度补取依次处理已有文件或已受理请求、已作废请求、其他年份，保持年度顺序获取。获取后的原文 SHA、时间、节点、单位、类型、完整性以及重合值绝对差 ≤1e-4 的全部校验继续执行。

真实状态验证可在服务器运行 `python analysis/verify_era5_queue_wait.py --root ROOT --output ROOT/.codex-temp/era5-queue-verification.json`。验证读取实际请求及官方 API，只在项目临时目录写入测试状态，不提交新的 CDS 请求；检查旧状态迁移、预算边界、等待期限、完整分页、真实进程锁和科学校验代码的一致性。
