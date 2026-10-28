# ERA5 补取请求恢复

`prepare_era5_wind_repair.py` 使用 `era5_cds_request.py` 管理年度和月度持久请求。在读取编号及轮询状态时，HTTP 404 的官方 JSON 正文明确包含 `job not found` 或 `job … dismissed`，请求进入归档和重新提交流程。状态为 `rejected` 或 `failed` 时，仅对 CDS 明确给出的以下队列限制重新提交：

> Maximum number queued requests for this dataset has been reached. Please configure your scripts to request data sequentially

HTTP 401/403、其他 HTTP 404、参数错误及其他失败原因直接报告错误。重新提交沿用原数据集及全部请求参数，服务端返回的数据集、区域、年份和变量仍需与本地请求一致。

每个持久请求最多重新提交 5 次，依次退避 60、120、240、480、900 秒。次数、下一次提交时间及待提交标志写入原 `.request.json`，进程重启继续沿用。提交前扣减预算，获得新编号后立即写入该文件。达到上限时报告错误，保留失败编号和归档。

每次作废或可重试拒绝均生成 `.request.retired-<旧编号>.json`，记录完整请求、旧编号、CDS 原因、归档时间及重提次数。原请求文件的 `history` 保留完整历史，按块的 `status/era5-wind-<块>-repair.json` 同步记录历史、当前编号、重提次数和下一次提交时间。后续状态更新和进程重启保留历史。

风速年度补取依次处理已有文件或已受理请求、已作废请求、其他年份，保持年度顺序获取。获取后的原文 SHA、时间、节点、单位、类型、完整性以及重合值绝对差 ≤1e-4 的全部校验继续执行。
