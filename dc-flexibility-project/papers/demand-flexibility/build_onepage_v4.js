// 课题一页大纲 v4：node build_onepage_v4.js <输出.docx>（需要 docx 包；数字来源见文末注释）
const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, BorderStyle, VerticalAlign,
} = require('docx');

const OUT = process.argv[2];
const SONG = { ascii: 'Times New Roman', hAnsi: 'Times New Roman', eastAsia: '宋体' };
const HEI = { ascii: 'Arial', hAnsi: 'Arial', eastAsia: '黑体' };
const NAVY = '1F3864';
const STATUS = { 已有: '2E7D32', 待算: 'C25E00', 部分可算: 'B8860B', 待算待查: 'C25E00' };
const NONE = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };
const LINE = { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF' };

const t = (text, o = {}) => new TextRun({ text, font: o.hei ? HEI : SONG, size: o.size ?? 21, bold: o.bold, color: o.color });
const para = (children, o = {}) => new Paragraph({ alignment: o.align, spacing: { before: o.before ?? 0, after: o.after ?? 40, line: o.line ?? 290 },
  indent: o.indent, keepNext: o.keepNext, children });
const tag = (s) => t('〔' + s + '〕', { bold: true, color: STATUS[s], size: 19 });
const heading = (title, status) => para([t(title, { hei: true, bold: true, size: 23, color: NAVY }), t('  '), tag(status)], { before: 110, after: 30, keepNext: true });
const body = (parts, o = {}) => para(parts.map((x) => (typeof x === 'string' ? t(x, o) : x)), { after: 20 });
const item = (parts) => para([t('•  ', { color: NAVY, bold: true }), ...parts.map((x) => (typeof x === 'string' ? t(x) : x))],
  { indent: { left: 240, hanging: 240 }, after: 10 });

function cell(children, width, o = {}) {
  return new TableCell({ width: { size: width, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    shading: o.fill ? { type: ShadingType.CLEAR, color: 'auto', fill: o.fill } : undefined,
    margins: { top: o.pad ?? 40, bottom: o.pad ?? 40, left: 80, right: 80 },
    borders: o.borders, children });
}

// 核心论点框
const thesis = new Table({
  width: { size: 10200, type: WidthType.DXA }, columnWidths: [10200],
  rows: [new TableRow({ children: [cell([
    para([t('核心论点  ', { hei: true, bold: true, size: 20, color: NAVY }),
      t('任务柔性在全球多数地区都能降低能源系统成本，且无需加卡；但节省金额不到机群年化成本的2%。柔性的经济价值取决于两点：能否借柔性', { size: 20 }),
      t('少配算力设备', { size: 20, bold: true }), t('，以及能否在', { size: 20 }), t('几乎不损失有效产出', { size: 20, bold: true }), t('的前提下实施。', { size: 20 })],
    { line: 280, after: 0 })], 10200, { fill: 'EAF0F8', pad: 80,
      borders: { top: NONE, bottom: NONE, right: NONE, left: { style: BorderStyle.SINGLE, size: 24, color: NAVY } } })] })],
});

// 关键数字
const kpis = [['3–14%', '能源侧收益', '规划阶段，11情境全球中位数'], ['1.87%', '节省占机群年化成本（最高）', '全部网格经济最优下无一超过3%'],
  ['33–44%', '机群峰值余量', '平均任务负荷仅为峰值K的56–67%']];
const kpiTable = new Table({
  width: { size: 10200, type: WidthType.DXA }, columnWidths: [3400, 3400, 3400],
  rows: [new TableRow({ children: kpis.map(([big, label, note], i) => cell([
    para([t(big, { hei: true, bold: true, size: 38, color: NAVY })], { align: AlignmentType.CENTER, after: 0, line: 240 }),
    para([t(label, { hei: true, bold: true, size: 19 })], { align: AlignmentType.CENTER, after: 0 }),
    para([t(note, { size: 17, color: '595959' })], { align: AlignmentType.CENTER, after: 0 }),
  ], 3400, { pad: 50, borders: { top: NONE, bottom: NONE, left: i ? LINE : NONE, right: NONE } })) })],
});

// 主线表
const W = [1000, 2550, 2850, 2350, 1450];
const head = ['部分', '要回答的问题', '指标与方法', '产出图', '现状'];
const rows = [
  ['一', '既有机群调度能省多少、在哪里省', '能源侧收益率；节省占机群年化成本；归因回归', '11情境全球地图、3%阈值图、归因图', '已有'],
  ['二', '柔性能否少配服务器与GPU', '联合优化机群容量与风光储；设备＋能源总节省', '设备节省与总节省全球分布', '待算'],
  ['三', '卡价上涨、有效产出要求下柔性还剩多少空间', '盈亏平衡有效产出损失；盈亏平衡卡价；可用空档上限H', '卡价倍数曲线；H×卡价热图', '部分可算'],
  ['四', '技术进步使上述边界向哪移动', '2030年设备功率、液冷、设备价格、风光储成本情景', '因素方向表；情景对比', '待算待查'],
];
const mainTable = new Table({
  width: { size: 10200, type: WidthType.DXA }, columnWidths: W,
  rows: [
    new TableRow({ tableHeader: true, children: head.map((h, j) => cell([para([t(h, { hei: true, bold: true, size: 18, color: 'FFFFFF' })], { align: AlignmentType.CENTER, after: 0 })],
      W[j], { fill: NAVY, pad: 30, borders: { top: NONE, bottom: NONE, left: NONE, right: NONE } })) }),
    ...rows.map((r, i) => new TableRow({ cantSplit: true, children: r.map((v, j) => cell([para(
      j === 4 ? [t(v, { bold: true, size: 18, color: STATUS[v] })] : [t(v, { size: 18, bold: j === 0, hei: j === 0 })],
      { align: j === 0 || j === 4 ? AlignmentType.CENTER : AlignmentType.LEFT, after: 0, line: 260 })], W[j],
      { fill: i % 2 ? 'F5F7FA' : undefined, pad: 45, borders: { top: NONE, left: NONE, right: NONE, bottom: LINE } })) })),
  ],
});

const children = [
  para([t('数据中心任务柔性的经济边界与规划价值', { hei: true, bold: true, size: 32, color: NAVY })], { align: AlignmentType.CENTER, after: 0 }),
  para([t('课题一页大纲 v4　2026年11月6日', { size: 17, color: '7F7F7F' })], { align: AlignmentType.CENTER, after: 80 }),
  thesis,
  para([], { after: 60 }),
  kpiTable,
  para([], { after: 60 }),
  mainTable,

  heading('一、既有机群：空档充足，但空档本身不值钱', '已有'),
  body(['现有机群按峰值配置，平均有三到四成峰值余量，柔性利用这些空档即可完成负荷转移，所有最优解都不需新增设备。但转移负荷的节省只占能源系统成本的3%–14%（已建成2%–7%），不到机群年化成本的2%（训练约0.83%、推理约0.1%）。',
    t('限制柔性价值的不是空档多少，而是能源侧可省的空间：', { bold: true }),
    '收益主要由风电资源和对市电的依赖决定（训练收益与市电占比相关系数−0.88）。峰值余量指任务分配相对窗口峰值的空余，不同于GPU计算利用率。']),

  heading('二、联合规划：柔性能否少配算力设备', '待算'),
  body([t('命题：', { bold: true }), '保持任务总量、设备性能与完成期限不变，柔性能否削减所需服务器与GPU，使设备节省与能源节省合计达到有意义的规模。',
    t('初步线索：', { bold: true }), '训练任务功率超过峰值90%的时段仅135小时、占任务电量0.26%；若机群因此削减10%，年省设备费约1,330万美元，是能源侧收益（约110万美元）的十倍以上。推理的峰值多为刚性在线负荷，削峰空间有限。需联合优化核实。']),

  heading('三、卡价与有效产出约束：柔性只能“搭便车”', '部分可算'),
  item([t('比例效应：', { bold: true }), '盈亏平衡有效产出损失＝柔性年节省÷机群年化成本，训练约0.8个百分点、推理约0.1个，卡价翻倍则减半；而业界有效产出常在85%–97%之间（Gemini由85%提升至97%）。']),
  item([t('加卡效应：', { bold: true }), '八城市中，为柔性加卡后每推迟1MWh成本约0.58–3.5万美元，而每推迟1MWh仅省73–99美元，相差两到三个数量级。']),
  item([t('利用率效应：', { bold: true }), '卡越贵，运营方越会压缩空档。以柔性可用上限H（峰值、P99、P95、峰值九折和八折）与卡价倍数（0.5–3倍）做二维扫描。']),

  heading('四、技术进步：边界向哪个方向移动', '待算待查'),
  body(['单卡功率上升和风光降价会提高柔性价值，储能降价会削弱柔性价值，检查点与恢复加快会降低拆分类柔性的代价。设置2030年情景重算第一、二部分指标，再结合文献判断综合方向。']),

  para([t('数据与边界  ', { hei: true, bold: true, size: 17, color: '595959' }),
    t('五类真实任务数据（2018–2023年）、11情境、全球4,842个有效网格、ERA5 2025年逐小时气象、12／48小时推迟期限；机群成本按整机资本年化与固定运维核算，设备价格与等效台数为模型假设，3%为研究设定的筛选阈值；窗口折算金额不代表全年实绩，延误、恢复与有效产出损失需另行评价。', { size: 17, color: '595959' })],
    { before: 100, after: 30, line: 260 }),
  para([t('下一步  ', { hei: true, bold: true, size: 17, color: NAVY }),
    t('① 第一部分成本对比图、第三部分比例效应图（不需重算）→ ② 第二部分联合优化与H×卡价扫描（先算八城市）→ ③ 2030情景与文献。', { size: 17 })],
    { after: 0, line: 270 }),
];

const doc = new Document({
  styles: { default: { document: { run: { font: SONG, size: 19 } } } },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 760, bottom: 700, left: 850, right: 850 } } }, children }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log('wrote', OUT); });

// 数字来源：
// 能源侧收益与机群成本比例：results/global-flex-econ-20261103/compute-cost/grid-metrics.parquet（#57），经济最优中位数；最高1.87%为全部网格经济最优最大值。
// 峰值余量＝1−平均任务负荷／K；平均负荷／峰值、超过0.9K的小时与电量：results/flex-class-20261101/native-hourly（与 task-peak-capacity.csv 一致）。
// 训练机群年化成本1.327亿美元（DGX H100 35万美元×1,143台×年化系数0.332）；能源侧收益约110万美元为规划经济最优全球中位数年化值。
// 八城市加卡推迟成本与每MWh节省：简表第七章7.2表（results/flex-card-cost-calendar-20261102）。
// 归因相关系数：results/global-flex-econ-20261103/attribution（#51）。
