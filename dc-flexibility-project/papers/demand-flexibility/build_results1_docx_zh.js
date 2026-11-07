// Results（第一部分）中文草稿：node build_results1_docx_zh.js <输出.docx>（需要 docx 包）
const fs = require('fs');
const path = require('path');
const { Document, Packer, Paragraph, TextRun, ImageRun, AlignmentType, HeadingLevel, PageBreak } = require('docx');

const OUT = process.argv[2];
const FIG = path.join(__dirname, 'figures');
const SONG = { ascii: 'Times New Roman', hAnsi: 'Times New Roman', eastAsia: '宋体' };
const HEI = { ascii: 'Arial', hAnsi: 'Arial', eastAsia: '黑体' };

// **粗体**、_斜体_（用于变量）
function runs(text, o = {}) {
  return text.split(/(\*\*[^*]+\*\*|_[^_]+_)/).filter(Boolean).map((part) => {
    const base = { font: o.font ?? SONG, size: o.size ?? 24, color: o.color };
    if (part.startsWith('**')) return new TextRun({ ...base, text: part.slice(2, -2), bold: true });
    if (part.startsWith('_') && part.endsWith('_') && part.length > 2) return new TextRun({ ...base, text: part.slice(1, -1), italics: true });
    return new TextRun({ ...base, text: part });
  });
}
const P = (text, o = {}) => new Paragraph({ alignment: o.align ?? AlignmentType.JUSTIFIED, spacing: { before: o.before ?? 0, after: o.after ?? 120, line: o.line ?? 400 },
  indent: o.noIndent ? undefined : { firstLine: 480 }, children: runs(text, o) });
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 120, after: 200 }, children: [new TextRun({ text, font: HEI, size: 32, bold: true, color: '000000' })] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, spacing: { before: 240, after: 120, line: 360 }, children: [new TextRun({ text, font: HEI, size: 26, bold: true, color: '000000' })] });
function figure(file, widthPx) {
  const buf = fs.readFileSync(path.join(FIG, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  return new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 120 },
    children: [new ImageRun({ type: 'png', data: buf, transformation: { width: widthPx, height: Math.round(widthPx * h / w) },
      altText: { name: file, title: file, description: file } })] });
}

const s1 = [
  '本研究共设置11个设施情境，由4种服务器配置（纯CPU型、CPU主导型、均衡型和GPU主导型）与5类计算用途（科研计算、云通用计算、科研AI、大模型训练和推理服务）组合而成。每个情境采用对应用途的真实作业记录，按数据源的原生观测窗口计算，不循环补齐为全年。情境被放置在全球4,842个通过可建设土地筛选的2°代表网格上。在每个网格中，一座参考IT功率为10 MW的数据中心以最低成本完成相同的计算服务，风电、光伏、电池储能和市电接入容量与逐小时调度一同优化，气象输入为ERA5 2025年逐小时数据。可推迟任务分为两类：可推迟·整体任务须在到达后12小时内完成，可推迟·可拆分任务可在检查点暂停并须在48小时内完成；刚性任务和用途未知的任务按原时刻执行。我们在两种边界下比较有柔性与无柔性的调度：规划阶段边界允许针对每种情况重新优化能源资产；已建成边界将能源资产固定为无柔性时的最优配置，只有运行调度可以响应（方法）。',
  '现有机群留有充足的空间来承接这些推迟。机群必须满足最繁忙时刻的需求，因此我们以观测窗口内任务IT功率的峰值_K_作为算力容量。11个情境的平均任务负荷只有_K_的48%–73%，即平均有27%–52%的已部署容量处于空闲（图1b）。在120–7,416个观测小时的窗口中，任务负荷超过0.9_K_的时段只有6–383小时。这里的余量不是GPU计算意义上的利用率，而是已分配任务需求相对机群自身峰值的空余，它给出了无需新增硬件即可移动的工作量上限。在全部639,144组配对解中，没有一个柔性最优解需要在_K_之上新增容量，也没有一个柔性解的成本高于对应的无柔性解。',
  '在这一余量之内，推迟任务几乎在所有地点都能降低能源系统的年化成本。经济最优目标下，规划阶段4,842个网格的收益中位数从3.3%（CPU主导型推理）到13.8%（GPU主导型训练）不等；能源资产固定后，收益降至2.2%–7.3%（图1a）。收益大小与可推迟的IT电量份额一致：科研计算有58%的IT电量可推迟，训练为36%–45%，科研AI为24%–40%，而推理以在线服务为主，可推迟的部分只有约9%。若要求每小时用电都来自本地风电、光伏和储能，规划阶段GPU主导型训练的收益中位数升至26.7%，因为推迟任务可以替代原本为最紧缺时段配置的储能。同一要求在已建成边界下几乎消除了全部收益：此时市电购电量本已为零，固定资产只给柔性留下很小的储能吞吐费用可以节省。',
];
const s2 = [
  '与柔性所依赖的算力资本相比，能源侧的节省很小。对一座10 MW的数据中心，按服务器配置不同，既有机群的年化成本为0.66–1.78亿美元，无柔性能源系统的年化成本为520–1,003万美元，而推迟任务带来的年化节省中位数只有12–114万美元（图1c）。以机群年化成本为分母，节省中位数在规划阶段为0.10%–1.03%，在已建成边界为0.07%–0.58%；经济最优下所有网格中的最大值为1.87%（图1d）。在经济最优目标下，无论哪种边界都没有网格超过机群成本的3%；在已建成边界下，任何本地无碳匹配目标都没有网格超过3%。只有当规划阶段还须满足100%逐小时本地无碳供电时，才有部分网格越过3%：纯CPU型科研计算占64%，GPU主导型训练占42%。',
  '以上两项结果共同指出了约束所在。空闲容量很充足，利用它也不需要硬件投入，但在12–48小时内挪动负荷所能创造的价值，受限于能源系统本身能省下多少。约1%机群成本的节省，也意味着柔性对有效计算时间的损失几乎没有容忍空间。大规模训练本身就会因故障和重启损失相当一部分经过时间：谷歌将其最大规模Gemini训练作业的有效产出（计算有用新步骤的时间占作业经过时间的比例）从85%提高到97%（文献G1），机群层面的核算也把只分到部分资源的时间计为调度损失（文献G2）。如果推迟或基于检查点的拆分使有效产出下降超过约0.8个百分点（GPU主导型训练）或0.1个百分点（推理），规划阶段能源侧收益的中位数就会被完全抵消。因此，既有机群中的柔性只有在几乎不损失有效产出时才有经济意义，例如推迟排队作业的开始时间，而不是中断正在运行的作业。',
];
const s3 = [
  '收益的空间格局主要由本地可再生能源资源决定，而不是由冷却需求决定。规划阶段GPU主导型训练的收益中位数，在年均风电出力系数低于0.05的网格为9.4%，并随风电资源增强而单调上升，在出力系数高于0.45的网格达到20.4%（图2a）。各大洲中位数从南美洲的10.1%到大洋洲的19.0%（图1a）；收益最高的是风光资源兼优的地区，包括北非、阿拉伯半岛、中亚和澳大利亚内陆，最低的是亚马孙盆地和刚果盆地等风资源薄弱的地区。',
  '其机制在于无柔性系统对市电的依赖程度。风资源弱的地方，无柔性最优系统约65%的电量按固定电价从市电购买，在时间上挪动负荷几乎省不下钱；风资源强的地方，本地风电、光伏和储能承担了大部分负荷，市电购电占比降至约28%，推迟任务可以减少跟随可再生能源出力所需的储能与弃电。在各网格之间，训练收益与无柔性系统的市电占比高度负相关（_r_ = −0.88；图2b）。对8个地点属性做标准化回归，可以解释除云通用计算外各用途收益空间差异的65%–84%；云通用计算只有120小时窗口，与全年资源属性匹配较差（_R_² = 0.20–0.31；图2c）。规划阶段，年均风电出力系数是训练、科研AI和科研计算最主要的预测变量，其次是光伏出力系数。已建成边界下，市电碳强度的作用增大，因为市电越贵，本地可再生能源装得越多，柔性可利用的对象也越多。推理收益主要取决于光伏资源和市电碳强度，与其可推迟份额小、只能利用光伏日周期的特点一致。',
  '这一归因以市电不分时段同价为前提。在分时电价下，依赖市电的地区同样可以通过挪动负荷获益，地区排序也会改变。因此，图2刻画的是柔性在哪些地方替代了现场能源资产，而不是对所有可能的柔性收益进行排序。',
];
const legend1 = '**图1 | 机群余量充足，但相对算力成本价值很小。** **a**，规划阶段、经济最优目标下GPU主导型训练的能源系统成本节省（相对同一无柔性能源系统，%）。每个2°区块取其代表0.5°网格的结果并裁剪到陆地（Robinson投影）；右侧条形为各大洲中位数（不加权，_n_ = 4,842个网格）。**b**，11个情境的平均任务功率与剩余余量，均以窗口任务IT功率峰值_K_的百分比表示；数字为余量（%）。_K_不含空载IT功率与冷却。**c**，10 MW数据中心的机群年化成本（方块）、无柔性能源系统年化成本中位数（菱形），以及规划阶段（实心圆）与已建成边界（空心圆）的柔性节省中位数，横轴为对数坐标。**d**，各情境、各边界下柔性节省占机群年化成本比例在网格间的分布；箱体为四分位区间，中线为中位数，须线为最小值与最大值；虚线为3%。能源成本和节省按各情境原生窗口折算为等效年值；机群成本为服务器资本按4年、8%折现率年化并加3%年运维（方法）。云通用计算的窗口仅120小时。';
const legend2 = '**图2 | 风电资源与市电依赖决定柔性在哪里有价值。** **a**，规划阶段收益中位数随年均风电出力系数分档的变化，包括科研计算、云通用计算、科研AI、大模型训练和推理服务（GPU用途取GPU主导型配置）；阴影为训练的四分位区间，横轴下方数字为各档网格数。**b**，规划阶段GPU主导型训练的收益与无柔性最优系统中市电占设施用电比例的关系（每点一个网格，_n_ = 4,842；颜色为年均风电出力系数）。**c**，5个代表情境在规划阶段与已建成边界下8个地点属性的标准化普通最小二乘系数，单位为每1个标准差对应的收益百分点；粗体表示去掉该属性后_R_²下降至少0.02，各列下方给出_R_²。所有面板均为经济最优目标。属性由2025年逐小时数据计算；回归只描述关联，未处理空间自相关（方法）。';

const notes = [
  '本稿为Results第一部分中文草稿v1，内容与英文草稿v1一致，按中文学术写法重写。图1、图2暂用英文标注版（PR #63），需要时可另出中文标注版。“方法”尚未撰写，文中为占位。',
  '引用待补：G1 Gemini Team, Google. Gemini: a family of highly capable multimodal models. arXiv:2312.11805（2023），原文“overall goodput for the largest-scale training job increased from 85% to 97%”，已于2026-11-04核对。G2 Machine Learning Fleet Efficiency: Analyzing and Optimizing Large-Scale Google TPU Systems with ML Productivity Goodput. arXiv:2502.06982（2025），作者列表待补。',
  '数字来源：峰值余量与K取自part1-summary.csv（PR #60）；能源侧收益、机群与能源年化成本、机群成本占比取自grid-summary.parquet与flex-econ-copt-20261102.json，并与compute-cost/grid-metrics.parquet（PR #57）交叉核对（经济最优最大1.867%；α = 100%规划阶段超过3%的网格：科研计算64.4%、GPU主导训练41.6%；已建成任何目标均无）；能源系统年化成本下限5.2452百万美元取自PR #63；可推迟份额取自简表第五章；归因数字取自attribution/（PR #51）；639,144组有效配对、无负收益、无新增容量取自grid-summary.parquet。',
];

const doc = new Document({
  styles: { default: { document: { run: { font: SONG, size: 24 } } } },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1700, right: 1700 } } },
    children: [
      P('草稿v1 · 结果第一部分 · 2026年11月7日', { font: HEI, size: 18, color: '7F7F7F', align: AlignmentType.LEFT, noIndent: true }),
      H1('结果'),
      H2('推迟柔性任务使能源系统成本降低3%–14%，且无需新增服务器'),
      ...s1.map((t) => P(t)),
      H2('节省金额只占算力机群成本的很小一部分'),
      ...s2.map((t) => P(t)),
      figure('fig1.png', 560),
      P(legend1, { size: 20, line: 320, noIndent: true }),
      H2('风电资源和对市电的依赖决定柔性在哪里有价值'),
      ...s3.map((t) => P(t)),
      figure('fig2.png', 560),
      P(legend2, { size: 20, line: 320, noIndent: true }),
      new Paragraph({ children: [new PageBreak()] }),
      P('**内部说明（投稿前删除）**', { font: HEI, size: 22, align: AlignmentType.LEFT, noIndent: true }),
      ...notes.map((t) => P(t, { size: 20, color: '595959', align: AlignmentType.LEFT, line: 320, noIndent: true })),
    ],
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log('wrote', OUT); });
