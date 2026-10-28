# ERA5 补取请求恢复

`prepare_era5_wind_repair.py` 使用 `era5_cds_request.py` 管理年度和月度持久请求。在读取编号及轮询状态时，HTTP 404 的官方 JSON 正文明确包含 `job not found` 或 `job … dismissed`，请求进入归档和重新提交流程。状态为 `rejected` 或 `failed` 时，仅对 CDS 明确给出的以下队列限制重新提交：

> Maximum number queued requests for this dataset has been reached. Please configure your scripts to request data sequentially

HTTP 401/403、其他 HTTP 404、参数错误及其他失败原因直接报告错误。重新提交沿用原数据集及全部请求参数，服务端返回的数据集、区域、年份和变量仍需与本地请求一致。

作废请求最多重新提交 5 次，依次退避 60、120、240、480、900 秒。次数、下一次提交时间及待提交标志写入原 `.request.json`，进程重启继续沿用。获得替代编号后计入重提次数并立即保存。达到上限时报告错误，保留失败编号和归档。

队列名额已满单独进入 `waiting_for_queue_slot`，不扣减上述 5 次预算。等待以约 300 秒起步，每次翻倍，加入 ±10% 随机抖动并限制每次最多 3600 秒；累计等待上限为 72 小时。等待次数、累计秒数、当前等待起点、期限和下一次检查时间持久保存，进程重启继续计时。受理后的处理时间不计入队列等待；再次遇到名额限制时沿用已累计的等待时间。达到期限后报告 `queue_wait_exhausted` 并退出。

旧状态自动迁移到 `retry_budget_version=2`：保留全部 `history` 和归档，以 `legacy_retry_count` 保留旧计数，从普通重提预算中扣除已因队列限制发生的重新提交。若当前正在等待队列名额，清除旧的 `retry_exhausted`，并从历史首次队列拒绝时间继续计算期限。

全部新版补取控制器共用 `ROOT/.era5-cds-request-gate/submission.lock`。锁内完整分页读取当前账号的 CDS `accepted/running` 请求，只计入 `reanalysis-era5-single-levels`，检查名额后才提交，持锁保存新编号及令牌。令牌对应的请求未出现在列表时，直接查询其状态，直到官方确认终止或作废才释放名额。已有控制器及账号其他脚本提交的活动请求同样计入名额；已受理请求继续跟踪和下载。

默认并发上限为 **2**。首次初始化时可设置环境变量 `ERA5_CDS_MAX_ACTIVE_REQUESTS`；后续统一读取 `ROOT/.era5-cds-request-gate/config.json` 中的 `max_active_requests` 正整数。需要调整时原子替换此配置文件，全部新版控制器在下一次检查时采用相同上限。初始化时已有请求超过上限，会等待数量降到上限以下；旧进程继续使用其已加载代码，其后续提交仍可能暂时超过此上限。认证 401/403 和参数错误直接退出，临时网络故障进入下述等待流程。

每次作废或可重试拒绝均生成 `.request.retired-<旧编号>.json`，记录完整请求、旧编号、CDS 原因、归档时间及重提次数。原请求文件的 `history` 保留完整历史，按块的 `status/era5-wind-<块>-repair.json` 同步记录历史、当前编号、重提次数和下一次提交时间。后续状态更新和进程重启保留历史。

风速年度补取依次处理已有文件或已受理请求、已作废请求、其他年份，保持年度顺序获取。获取后的原文 SHA、时间、节点、单位、类型、完整性以及重合值绝对差 ≤1e-4 的全部校验继续执行。

状态查询、年度优先级检查、完整分页读取账号队列、名额令牌查询、提交准备、失败详情读取、结果元数据查询与下载统一处理临时网络错误。`requests.ConnectionError`（含 `SSLError`）、`Timeout`、`ChunkedEncodingError`、底层 DNS 解析失败和 SSL EOF，以及 HTTP 429 和全部 5xx 均可等待。每次从约 60 秒开始指数退避，加入 ±10% 抖动，单次最多 1800 秒；同一持久请求累计网络故障等待最多 86400 秒。每次等待前预记本次秒数、下一次时间及事件，重启继续尚未完成的等待；成功后重新从初始退避计数，累计额度保留。等待每段最多 30 秒以响应停止信号，达到累计上限后报告 `network_wait_exhausted`。

网络等待沿用现有请求编号，不归档作废、不提交替代请求、不消耗 5 次作废重提预算。每次 `cds_network_wait` 事件同时写入日志、原 `.request.json` 的 `network_wait_events` 和按块状态；包含操作、错误类别、编号、时间、退避秒数和累计秒数。新字段均有缺省值，原请求参数及全部历史保留。明确的队列满拒绝（包括带官方队列满正文的 429）继续使用 72 小时队列等待；401/403、参数错误、明确作废的 404 及其他失败原因维持各自处理规则。

POST 发出前保存账号现有编号和 `submission_uncertain` 标志。提交响应丢失或返回临时网络错误时，完整分页查询账号队列，并核对新出现请求的全部参数以恢复唯一编号；重启后也先恢复该次提交结果。尚未找到唯一编号时按网络退避继续查询，累计仍受 24 小时限制；多个匹配编号直接报告。此过程中不会再次发送相同 POST。

每次下载尝试重新查询现有请求的结果地址、删除 `.nc.part` 并从头下载，不发送 Range 续传。HTTP 错误显式检查；连接提前结束或未收到 HTTP `Content-Length` 声明的全部字节按网络故障等待。成功下载必须通过官方结果元数据中的文件大小检查，随后执行原有 SHA、NetCDF 完整性及科学校验。

离线核验可运行 `python analysis/verify_era5_network_retry.py --output ROOT/.codex-temp/era5-network-verification.json`。可选 `--baseline` 指向发布前的 `prepare_era5_wind_repair.py`，核对请求构造、原文回执和科学校验代码；可选 `--state-root ROOT` 读取真实旧状态并仅在独立临时目录验证兼容性。脚本模拟异常和等待时钟，并使用真实本机 HTTP 连接及 ECMWF Remote 测试 503、断流与完整重下，不访问 CDS 或创建下载请求。每次运行使用新的输出文件名。

真实状态验证可在服务器运行 `python analysis/verify_era5_queue_wait.py --root ROOT --output ROOT/.codex-temp/era5-queue-verification.json`。验证读取实际请求及官方 API，只在项目临时目录写入测试状态，不提交新的 CDS 请求；检查旧状态迁移、预算边界、等待期限、完整分页、真实进程锁和科学校验代码的一致性。
