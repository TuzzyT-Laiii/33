# 冷却模型需求A记录

- 输入：独立worktree基于`origin/main`提交`8b7c6eb`，分支`codex/cooling-model-recalibration-20261028`；PR13天气、映射、两个中央IT文件按原SHA-256核对，全部只读。
- 完成：ESIF系统迁移和1903小时外推诊断；版本化风冷/直接液冷模型及高低敏感性；逐项来源/assumption；响应图；一个网格的标准热负荷全年对照和两个冻结IT抽查；五气候外部量级校验；原生ERA5与全年重放接口；平闰年、守恒、异常输入、旧版精确回归和恢复运行验证。
- 验证记录：[validation.json](../results/cooling-recalibration-20261028-v2/validation.json)；来源：[元数据](../data/metadata/cooling-recalibration-20261028-v2.json)；结果：[报告](../results/cooling-recalibration-20261028-v2/README.md)。
- 待解决：目标设施分项电表、机柜/CDU供回液与流量、部分负载热量占比、实际冷机曲线、气压和极端天气设备降额；全部典型曲线重算按后续需求处理。
- 保留范围：旧冷却资产、原模块默认行为、PR13曲线、Lisa GPU需求文件及主工作区已有改动。
- PR13评论最近成功检查：2026-10-28 10:41:44 UTC，分页读取普通评论、行内评论、审阅和全部讨论；均为空。新PR保持待审，合并需用户明确批准。
