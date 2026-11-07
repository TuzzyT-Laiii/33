// Results（第一部分）英文草稿：node build_results1_docx.js <输出.docx>（需要 docx 包）
const fs = require('fs');
const path = require('path');
const { Document, Packer, Paragraph, TextRun, ImageRun, AlignmentType, HeadingLevel, PageBreak } = require('docx');

const OUT = process.argv[2];
const FIG = path.join(__dirname, 'figures');
const TNR = { ascii: 'Times New Roman', hAnsi: 'Times New Roman', eastAsia: '宋体' };
const ARIAL = { ascii: 'Arial', hAnsi: 'Arial', eastAsia: '黑体' };

// 段落：字符串中 **粗体** 与 _斜体_ 标记
function runs(text, o = {}) {
  return text.split(/(\*\*[^*]+\*\*|_[^_]+_)/).filter(Boolean).map((part) => {
    if (part.startsWith('**')) return new TextRun({ text: part.slice(2, -2), bold: true, font: o.font ?? TNR, size: o.size ?? 22, color: o.color });
    if (part.startsWith('_') && part.endsWith('_') && part.length > 2) return new TextRun({ text: part.slice(1, -1), italics: true, font: o.font ?? TNR, size: o.size ?? 22, color: o.color });
    return new TextRun({ text: part, font: o.font ?? TNR, size: o.size ?? 22, color: o.color });
  });
}
const P = (text, o = {}) => new Paragraph({ alignment: o.align ?? AlignmentType.JUSTIFIED, spacing: { before: o.before ?? 0, after: o.after ?? 160, line: o.line ?? 360 },
  indent: o.indent, children: runs(text, o) });
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 120, after: 200 }, children: [new TextRun({ text, font: ARIAL, size: 30, bold: true })] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, spacing: { before: 240, after: 120, line: 320 }, children: [new TextRun({ text, font: ARIAL, size: 24, bold: true })] });
function figure(file, widthPx) {
  const buf = fs.readFileSync(path.join(FIG, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  return new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 120 },
    children: [new ImageRun({ type: 'png', data: buf, transformation: { width: widthPx, height: Math.round(widthPx * h / w) },
      altText: { name: file, title: file, description: file } })] });
}

const s1 = [
  'We evaluate eleven facility scenarios that combine four server configurations (CPU-only, CPU-heavy, balanced and GPU-heavy) with five workload types: scientific computing, cloud computing, scientific AI, large-language-model (LLM) training and inference. Each scenario is driven by its own observed job trace over its native observation window, without replay to a full year, and is placed at 4,842 representative 2° land cells that pass a buildability screen. At every cell a 10 MW reference facility delivers the same computing service at least cost by co-optimising on-site wind, solar photovoltaics, battery storage and a grid connection, with hourly dispatch driven by ERA5 weather for 2025. Flexible work is either deferrable as a whole, with a 12-hour deadline, or deferrable and splittable at checkpoints, with a 48-hour deadline; rigid and unclassified work runs as it arrives. We compare flexible and inflexible dispatch at two boundaries: a planning boundary, at which the energy assets are re-optimised for each case, and an as-built boundary, at which the assets are fixed at the inflexible optimum and only operation can respond (Methods).',
  'Existing fleets leave ample room for this deferral. Because a fleet must serve its busiest hour, we size compute capacity _K_ to the peak of task IT power over each observation window. Mean task load is only 48–73% of _K_ across the eleven scenarios, so 27–52% of the deployed capacity is idle on average (Fig. 1b). Task load exceeds 90% of _K_ in 6–383 hours of windows that span 120–7,416 observed hours. This slack is not utilisation in the sense of GPU activity; it measures allocated task demand relative to the fleet’s own peak and therefore bounds the work that can be moved without new hardware. Across all 639,144 paired solutions, no flexible optimum adds capacity above _K_, and no flexible solution costs more than its inflexible counterpart.',
  'Within this headroom, deferral lowers the annualised cost of the energy system at almost every location. At the economic optimum, median savings across the 4,842 cells range from 3.3% (CPU-heavy inference) to 13.8% (GPU-heavy LLM training) at the planning boundary and from 2.2% to 7.3% once the energy assets are fixed (Fig. 1a). Savings follow the deferrable share of IT energy: 58% of scientific-computing energy, 36–45% of training energy and 24–40% of scientific-AI energy can be deferred, whereas online serving leaves only about 9% of inference energy deferrable. Requiring every hour of electricity to come from local wind, solar and storage raises the median planning-boundary saving to 26.7% for GPU-heavy training, because deferral then substitutes for storage that would otherwise be sized for the scarcest hours. The same requirement removes almost all savings at the as-built boundary, where grid imports are already zero and the fixed assets leave flexibility only a residual storage-throughput charge.',
];
const s2 = [
  'Energy-side savings remain small when set against the capital on which the flexibility relies. For a 10 MW facility, the annualised cost of the existing server fleet is 66–178 million US dollars, depending on the server configuration, compared with 5.2–10.0 million dollars for the inflexible energy system and 0.12–1.14 million dollars of median annual savings from deferral (Fig. 1c). Expressed as a share of fleet cost, the median saving is 0.10–1.03% at the planning boundary and 0.07–0.58% at the as-built boundary, and the largest value at any cell is 1.87% at the economic optimum (Fig. 1d). No cell exceeds 3% of fleet cost at the economic optimum under either boundary, and none does so at the as-built boundary for any local carbon-free target. The 3% mark is crossed only when the planner must also meet a 100% hourly local carbon-free requirement, in 64% of cells for CPU scientific computing and 42% for GPU-heavy training.',
  'These two results together locate the constraint. Idle capacity is plentiful and using it costs nothing in hardware, yet the value of moving load within a 12–48-hour window is bounded by how much the energy system can save. Savings of about 1% of fleet cost also leave a narrow margin for any loss of productive compute time. Large training jobs already lose a material share of elapsed time to failures and restarts: the goodput of the largest Gemini training job, defined as the time spent computing useful new steps over the elapsed time of the job, was raised from 85% to 97% (ref. G1), and fleet-level accounting treats partially allocated time as a scheduling loss (ref. G2). If deferral or checkpoint-based splitting reduced goodput by more than roughly 0.8 percentage points for GPU-heavy training, or 0.1 percentage points for inference, it would erase the median energy-side benefit at the planning boundary. Flexibility in existing fleets is therefore economically meaningful only when it is close to goodput-neutral, for example when it delays the start of queued jobs rather than interrupting running ones.',
];
const s3 = [
  'The spatial pattern of savings follows local renewable resources rather than cooling demand. For GPU-heavy training at the planning boundary, median savings rise steadily from 9.4% in cells with a mean wind capacity factor below 0.05 to 20.4% in cells above 0.45 (Fig. 2a). Continental medians range from 10.1% in South America to 19.0% in Oceania (Fig. 1a), with the highest savings in regions that combine strong wind and sun, including North Africa, the Arabian Peninsula, Central Asia and inland Australia, and the lowest in weak-wind regions such as the Amazon and Congo basins.',
  'The mechanism is the inflexible system’s reliance on grid power. Where wind is weak, the least-cost inflexible system buys about 65% of its electricity from the grid at a flat tariff, so moving load in time saves little; where wind is strong, local wind, solar and storage supply most of the demand, grid purchases fall to about 28%, and deferral reduces the storage and curtailment needed to follow renewable output. Across cells, training savings are strongly anti-correlated with the inflexible grid share (_r_ = −0.88; Fig. 2b). A standardised regression on eight location attributes explains 65–84% of the spatial variance in savings for every workload except cloud computing, whose 120-hour window matches annual resource attributes poorly (_R_² = 0.20–0.31; Fig. 2c). Mean wind capacity factor is the leading predictor for training, scientific AI and scientific computing at the planning boundary, followed by solar capacity factor. At the as-built boundary the grid carbon intensity gains weight, because a costlier grid supply leads to more local renewable capacity against which flexibility can be used. Inference savings depend mainly on solar resources and grid carbon intensity, consistent with a small deferrable share that can exploit only the daily solar cycle.',
  'This attribution is conditional on a flat grid tariff. Under time-varying prices, grid-reliant locations would also gain from shifting load and the regional ranking would change; Fig. 2 therefore identifies where flexibility substitutes for on-site energy assets rather than ranking all possible flexibility revenues.',
];
const legend1 = [
  '**Fig. 1 | Ample fleet headroom, small value relative to compute cost.** **a**, Energy-system cost savings from task flexibility for GPU-heavy LLM training at the planning boundary and economic optimum, relative to the same energy system without flexibility. Each 2° block takes the value of its representative 0.5° cell and is clipped to land (Robinson projection); bars give continental medians (unweighted, n = 4,842 cells). **b**, Mean task power and remaining headroom as shares of the window peak of task IT power, _K_, for the eleven scenarios; numbers give headroom (%). _K_ excludes idle IT power and cooling. **c**, Annualised fleet cost (squares), median energy-system cost without flexibility (diamonds) and median flexibility savings at the planning (filled circles) and as-built (open circles) boundaries for a 10 MW facility, on a logarithmic axis. **d**, Flexibility savings as a share of annualised fleet cost across cells for each scenario and boundary; boxes span the interquartile range, centre lines mark medians and whiskers the minimum and maximum; the dashed line marks 3%. Energy costs and savings are converted to annual equivalents from each native window; fleet cost is the server capital cost annualised over four years at an 8% discount rate plus 3% annual maintenance (Methods). The cloud-computing window covers only 120 hours.',
];
const legend2 = [
  '**Fig. 2 | Wind resource and grid reliance set where flexibility pays.** **a**, Median planning-boundary savings across bins of annual mean wind capacity factor for scientific computing, cloud computing, scientific AI, LLM training and inference (GPU workloads use the GPU-heavy configuration); the band spans the interquartile range for LLM training and counts below the axis give cells per bin. **b**, Savings for GPU-heavy LLM training against the grid share of facility electricity in the least-cost system without flexibility, at the planning boundary (one point per cell, n = 4,842; colour gives mean wind capacity factor). **c**, Standardised ordinary least-squares coefficients of eight location attributes for the five scenarios at the planning and as-built boundaries, in percentage points of savings per standard deviation; bold values mark attributes whose omission lowers _R_² by at least 0.02, and _R_² is given below each column. All panels use the economic optimum. Attributes are computed from 2025 hourly data; the regressions describe associations and do not adjust for spatial autocorrelation (Methods).',
];

const notes = [
  '内部说明（投稿前删除）',
  '本稿为 Results 第一部分英文草稿，按作者 NC 稿的写法组织：小标题为带数字的结论句，段首给领域主张，再给数量、对照与解释。Methods 尚未撰写，文中“(Methods)”为占位。',
  '引用待补：G1 Gemini Team, Google. Gemini: a family of highly capable multimodal models. arXiv:2312.11805（2023），原文“overall goodput for the largest-scale training job increased from 85% to 97%”，已于 2026-11-04 核对。G2 Machine Learning Fleet Efficiency: Analyzing and Optimizing Large-Scale Google TPU Systems with ML Productivity Goodput. arXiv:2502.06982（2025），作者列表待补。',
  '数字来源：峰值余量与 K 取自 part1-summary.csv（PR #60）；能源侧收益、机群与能源年化成本、机群成本占比取自 grid-summary.parquet 与 flex-econ-copt-20261102.json，并与 compute-cost/grid-metrics.parquet（PR #57）交叉核对（经济最优最大 1.867%；α = 100% 规划阶段超过 3% 的网格：科研计算 64.4%、GPU 主导训练 41.6%；已建成任何目标均无）；可推迟份额取自简表第五章；归因数字取自 attribution/（PR #51）；639,144 组有效配对、无负收益、无新增容量取自 grid-summary.parquet。图为 PR #63。',
];

const doc = new Document({
  styles: { default: { document: { run: { font: TNR, size: 22 } } } },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1418, bottom: 1418, left: 1418, right: 1418 } } },
    children: [
      P('Draft v1 · Results, part 1 · 7 November 2026', { font: ARIAL, size: 18, color: '7F7F7F', align: AlignmentType.LEFT }),
      H1('Results'),
      H2('Deferring flexible work cuts energy-system cost by 3–14% without additional servers'),
      ...s1.map((t) => P(t)),
      H2('The savings are a small fraction of what the computing fleet costs'),
      ...s2.map((t) => P(t)),
      figure('fig1.png', 600),
      ...legend1.map((t) => P(t, { size: 18, line: 276 })),
      H2('Wind resources and reliance on grid power set where flexibility pays'),
      ...s3.map((t) => P(t)),
      figure('fig2.png', 600),
      ...legend2.map((t) => P(t, { size: 18, line: 276 })),
      new Paragraph({ children: [new PageBreak()] }),
      P('**' + notes[0] + '**', { font: ARIAL, size: 20, align: AlignmentType.LEFT }),
      ...notes.slice(1).map((t) => P(t, { size: 18, color: '595959', align: AlignmentType.LEFT, line: 300 })),
    ],
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log('wrote', OUT); });
