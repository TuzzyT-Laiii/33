// 结果部分中文稿 v2（每段一句话，数据由图承载）：node build_results_zh_v2.js <输出.docx>
// 图来自 figures-zh/（Codex 中文图，PR 待开）；缺图时插入占位说明。
const fs = require('fs');
const path = require('path');
const { Document, Packer, Paragraph, TextRun, ImageRun, AlignmentType, HeadingLevel, PageBreak } = require('docx');

const OUT = process.argv[2];
const FIG = process.argv[3] || path.join(__dirname, 'figures-zh');
const SONG = { ascii: 'Times New Roman', hAnsi: 'Times New Roman', eastAsia: '宋体' };
const HEI = { ascii: 'Arial', hAnsi: 'Arial', eastAsia: '黑体' };

function runs(text, o = {}) {
  return text.split(/(\*\*[^*]+\*\*|_[^_]+_)/).filter(Boolean).map((part) => {
    const base = { font: o.font ?? SONG, size: o.size ?? 24, color: o.color };
    if (part.startsWith('**')) return new TextRun({ ...base, text: part.slice(2, -2), bold: true });
    if (part.startsWith('_') && part.endsWith('_') && part.length > 2) return new TextRun({ ...base, text: part.slice(1, -1), italics: true });
    return new TextRun({ ...base, text: part });
  });
}
const P = (text, o = {}) => new Paragraph({ alignment: o.align ?? AlignmentType.JUSTIFIED, spacing: { before: o.before ?? 0, after: o.after ?? 140, line: o.line ?? 400 },
  indent: o.noIndent ? undefined : { firstLine: 480 }, children: runs(text, o) });
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 120, after: 200 }, children: [new TextRun({ text, font: HEI, size: 32, bold: true, color: '000000' })] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, spacing: { before: 280, after: 140, line: 360 }, children: [new TextRun({ text, font: HEI, size: 26, bold: true, color: '000000' })] });
const TODO = (text) => P('〔待算〕' + text, { color: 'C25E00' });

function figure(name, legend) {
  const file = path.join(FIG, name + '.png');
  const out = [];
  if (fs.existsSync(file)) {
    const buf = fs.readFileSync(file);
    const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
    out.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 120 },
      children: [new ImageRun({ type: 'png', data: buf, transformation: { width: 560, height: Math.round(560 * h / w) }, altText: { name, title: name, description: name } })] }));
  } else {
    out.push(P(`［${name} 待插入：中文图由 Codex 绘制中］`, { noIndent: true, align: AlignmentType.CENTER, color: '7F7F7F', before: 160 }));
  }
  out.push(P(legend, { size: 20, line: 320, noIndent: true }));
  return out;
}

const part1 = [
  ['推迟柔性任务使能源系统成本降低3%–14%，且无需新增服务器', [
    '我们在全球4,842个2°网格上，为11个由服务器配置与计算用途组合而成的10 MW数据中心情境，联合优化风光储、市电接入与逐小时任务调度，并在规划阶段与已建成两种边界下比较有无柔性的成本（方法）。',
    '现有机群按峰值配置，平均有27%–52%的峰值余量，全部639,144组配对解都无需新增设备（图1b）。',
    '利用这些余量推迟任务，规划阶段能源系统成本的全球中位数降低3.3%–13.8%，已建成边界降低2.2%–7.3%（图1a）。',
    '收益随可推迟电量份额增加：科研计算、训练和科研AI有24%–58%的IT电量可以推迟，而推理只有约9%。',
  ]],
  ['节省金额只占算力机群成本的很小一部分', [
    '一座10 MW数据中心每年因柔性节省12–114万美元，而其机群年化成本为0.66–1.78亿美元（图1c）。',
    '节省占机群年化成本的比例最高只有1.87%，经济最优下没有一个网格超过3%（图1d）。',
    '因此，限制既有机群柔性价值的不是空档多少，而是能源侧可省的空间。',
    '约1%的收益空间也意味着，推迟或拆分只要使有效产出下降超过约0.8个百分点（训练）或0.1个百分点（推理），收益就会被抵消。',
  ]],
  ['风电资源和对市电的依赖决定柔性在哪里有价值', [
    '大模型训练的收益随风电资源单调上升，从风电出力系数低于0.05网格的9.4%升至高于0.45网格的20.4%（图2a）。',
    '其机制是无柔性系统对市电的依赖程度：训练收益与市电占比的相关系数为−0.88（图2b）。',
    '8个地点属性可解释收益空间差异的65%–84%，风电出力系数是最主要的预测变量（图2c）。',
    '这一归因以市电不分时段同价为前提，在分时电价下依赖市电的地区同样可能获益。',
  ]],
];
const part2 = [
  ['机群规模参与规划时，同样的任务可少配9%–32%的服务器', [
    '新建数据中心若把整机台数与任务调度、供能一同优化，所需的最少台数只取决于作业记录与推迟期限，与地点无关（方法）。',
    '在12/48小时期限下，训练可少配22.6%的服务器，科研AI少配15.8%，科研计算少配18%–21%，推理少配9.2%（图3a）。',
    '推迟期限放宽到48/168小时，训练和科研AI的减配率分别升至27.6%和22.4%，推理则不随期限变化（图3b）。',
    '代价是缩减后的机群接近满载：训练平均推迟17小时、平均占用91%，若保留10%的冗余，设备节省将减少约三分之一（图3c）。',
  ]],
  ['设备节省使柔性的价值提高一到两个数量级', [
    '从原时序规划到联合规划，年净节省为1,366–3,840万美元，相当于机群年化成本的9.4%–33%，而只调度不改机群时仅为0.1%–1.0%（图4a）。',
    '净节省的95%以上来自少买设备，能源部分仅占2%–5%，八个城市之间相差不到2%（图4a）。',
    '设备价格越高，这项节省越大，与只调度时柔性占机群成本的比例随设备变贵而下降的趋势相反（图4b）。',
  ]],
  ['出售空闲算力是同一量级的替代选择', [
    '把空闲GPU按每小时1–2.2美元出租、可售比例为50%时，年净收入为1,038–5,420万美元，与少配服务器处于同一量级（图4c）。',
    '两者持平的售价为：训练每GPU小时2.2–2.8美元，科研AI 1.2–1.5美元，推理1.1–1.4美元（图4c）。',
    '少配服务器与出售空闲算力基本互斥，但两者都比能源侧柔性高一到两个数量级，空闲容量的价值主要在算力一侧。',
    '以上结果未核验逐节点显存与内存装箱，科研计算的最少台数仍为区间，出租情景也未模拟推理市场的需求与价格。',
  ]],
];
const part3 = ['卡价上涨与有效产出约束决定柔性采取哪种形式', [
  '设备价格从0.5倍升至3倍时，只调度的柔性占机群成本的比例下降，而少配服务器和出售空闲算力的收益随之上升。',
  '推迟与拆分造成的有效产出损失须低于柔性收益占机群成本的比例，柔性才划算；我们将给出各情境的盈亏平衡有效产出损失。',
  '当可用空档受运营冗余或高利用率目标限制时，柔性收益与可减配台数如何下降，将通过可用空档上限与卡价的二维扫描给出。',
]];
const part4 = ['技术进步如何改变柔性的价值', [
  '单卡功率上升与液冷普及会提高电费在总成本中的占比，从而提高能源侧柔性的相对价值。',
  '风光与储能降价会改变柔性的价值：风光降价使系统更依赖本地可再生能源，柔性更有用；储能降价则会替代一部分柔性。',
  '检查点与恢复技术的进步会降低拆分类柔性的有效产出代价；我们将以2030年情景重算第一、二部分指标，并给出各因素的作用方向。',
]];

const legends = {
  fig1: '**图1 | 机群余量充足，但相对算力成本价值很小。** **a**，规划阶段、经济最优目标下GPU主导型训练的能源系统成本节省（%），每个2°区块取其代表网格的结果并裁剪到陆地；右侧为各大洲中位数（_n_ = 4,842个网格）。**b**，11个情境的平均任务功率与剩余余量，以窗口任务IT功率峰值_K_的百分比表示。**c**，10 MW数据中心的机群年化成本、无柔性能源系统年化成本与柔性节省中位数（对数坐标）。**d**，柔性节省占机群年化成本比例在网格间的分布，虚线为3%。',
  fig2: '**图2 | 风电资源与市电依赖决定柔性在哪里有价值。** **a**，规划阶段收益中位数随年均风电出力系数分档的变化；阴影为训练的四分位区间。**b**，GPU主导型训练的收益与无柔性系统市电占比的关系（每点一个网格）。**c**，8个地点属性的标准化回归系数，粗体表示去掉后_R_²下降至少0.02。所有面板均为经济最优目标。',
  fig3: '**图3 | 机群规模参与规划时可少配的服务器。** **a**，12/48小时期限下各情境的减配率，科研计算为尚未收紧的区间，旁注为少配的GPU（或服务器）数量。**b**，减配率随推迟期限的变化。**c**，减配的代价：平均推迟时长、缩减后机群平均占用率与保留冗余后的设备节省。',
  fig4: '**图4 | 空闲容量的三种用法。** **a**，从原时序规划到联合规划的年净节省占机群年化成本比例（设备与能源两部分），点为只调度时的柔性占比。**b**，设备价格倍数对联合规划净节省与只调度柔性占比的相反影响。**c**，出售空闲GPU算力的年净收入随售价的变化（阴影为可售比例25%–75%，实线为50%），虚线为少配服务器的设备年化节省，竖线为DeepInfra H100零售价2.2美元（2026年7月，第三方记录）与2.9美元（官方当前价）。',
};

const children = [
  P('草稿v2 · 结果（结论一至四）· 2026年11月8日 · 每段一句话，数据由图承载', { font: HEI, size: 18, color: '7F7F7F', align: AlignmentType.LEFT, noIndent: true }),
  H1('结果'),
  P('**结论一　既有机群的柔性：空档充足，但空档本身不值钱**', { noIndent: true, font: HEI, size: 26, before: 120 }),
];
part1.forEach(([h, ps], i) => {
  children.push(H2(h), ...ps.map((t) => P(t)));
  if (i === 1) children.push(...figure('fig1', legends.fig1));
  if (i === 2) children.push(...figure('fig2', legends.fig2));
});
children.push(P('**结论二　联合规划的柔性：价值在于少配服务器**', { noIndent: true, font: HEI, size: 26, before: 240 }));
part2.forEach(([h, ps], i) => {
  children.push(H2(h), ...ps.map((t) => P(t)));
  if (i === 0) children.push(...figure('fig3', legends.fig3));
  if (i === 2) children.push(...figure('fig4', legends.fig4));
});
children.push(P('**结论三　卡价与有效产出约束（待算）**', { noIndent: true, font: HEI, size: 26, before: 240 }), H2(part3[0]), ...part3[1].map(TODO));
children.push(P('**结论四　技术进步（待算）**', { noIndent: true, font: HEI, size: 26, before: 240 }), H2(part4[0]), ...part4[1].map(TODO));
children.push(new Paragraph({ children: [new PageBreak()] }),
  P('**内部说明（投稿前删除）**', { font: HEI, size: 22, align: AlignmentType.LEFT, noIndent: true }),
  P('本稿按用户要求改为每段一句话、数据由图承载，取代结果第一部分中文草稿v1与第二部分中文草稿v1的正文写法（v1保留作数字明细）。结论一数字来源同第一部分v1内部说明；结论二数字来源同第二部分v1内部说明（PR #65、#67）。科研计算的最少台数仍为13,195–13,620台区间，收紧后需更新。结论三、四为拟论证观点，标〔待算〕。图1–图4为中文版，由Codex绘制（D:\\codex-runs\\paper-figs-zh-20261108）。', { size: 20, color: '595959', align: AlignmentType.LEFT, line: 320, noIndent: true }));

const doc = new Document({
  styles: { default: { document: { run: { font: SONG, size: 24 } } } },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1600, right: 1600 } } }, children }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log('wrote', OUT); });
