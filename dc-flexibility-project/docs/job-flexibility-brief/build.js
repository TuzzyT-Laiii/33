const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType,
  AlignmentType, ShadingType, PageOrientation, VerticalAlign, HeadingLevel, BorderStyle, ImageRun,
} = require('docx');

const OUT = process.argv[2];
const FACTS = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../../results/scenario-typical-day-by-task-20261101-v3/brief-facts.json'), 'utf8'));
const pct = (value) => (value * 100).toFixed(2) + '%';
const otherShare = (label) => FACTS.MIT_OTHER[label].pct_other_gpu_hours.toFixed(2) + '% OTHER';
const mitShare = (label) => FACTS.MIT_OTHER[label].pct_mit_gpu_hours.toFixed(2) + '% MIT';
const FONT = { ascii: 'Times New Roman', hAnsi: 'Times New Roman', eastAsia: '宋体' };
const HEI = { ascii: 'Arial', hAnsi: 'Arial', eastAsia: '黑体' };
const HEAD_FILL = 'D9E2F3';
const SIDE_FILL = 'F2F2F2';

// text may be a string or an array of [text, {bold, color}] parts
function cell(text, width, opts = {}) {
  const parts = Array.isArray(text) ? text : [[text, {}]];
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    shading: opts.fill ? { type: ShadingType.CLEAR, color: 'auto', fill: opts.fill } : undefined,
    margins: { top: 35, bottom: 35, left: 80, right: 80 },
    children: [new Paragraph({
      alignment: opts.left ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: parts.map(([t, o]) => new TextRun({ text: t, size: opts.size || 20, bold: opts.bold || o.bold, color: o.color, font: opts.bold ? HEI : FONT })),
    })],
  });
}

function table(widths, rows, firstColFill) {
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0, cantSplit: true,
      children: r.map((t, j) => cell(t, widths[j], {
        bold: i === 0,
        fill: i === 0 ? HEAD_FILL : (j === 0 && firstColFill ? SIDE_FILL : undefined),
        left: i > 0 && j === r.length - 1 && !firstColFill,
      })),
    })),
  });
}

const h = (t, pb = false) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: pb, spacing: { before: 160, after: 80 }, children: [new TextRun({ text: t, font: HEI, size: 26, bold: true })] });
const p = (t, o = {}) => new Paragraph({ spacing: { before: o.before ?? 60, after: 60 }, children: [new TextRun({ text: t, font: FONT, size: o.size ?? 20, color: o.color })] });

// ---------- 一、情境定义 ----------
const W1 = [2800, 2360, 2360, 2360, 2360, 2360];
const S = (id) => [[id, { bold: true }]];
const NA = [['—', { color: 'A6A6A6' }]];
const scen = [
  ['服务器规格 ＼ 需求', '科研计算', '云通用计算', '科研 AI', '大模型训练/开发', '推理服务'],
  ['A 纯 CPU 型（2∶0）', 'Lisa', 'Alibaba 2018（5 个完整日）', NA, NA, NA],
  ['B CPU 主导型（2∶1）', NA, NA, 'MIT Supercloud', 'Acme', 'ASI'],
  ['C 均衡型（2∶2）', NA, NA, 'MIT Supercloud', 'Acme', 'ASI'],
  ['D GPU 主导型（2∶8）', NA, NA, 'MIT Supercloud', 'Acme', 'ASI'],
];
const scenTable = new Table({
  width: { size: W1.reduce((a, b) => a + b, 0), type: WidthType.DXA },
  columnWidths: W1,
  rows: scen.map((r, i) => new TableRow({
    tableHeader: i === 0, cantSplit: true,
    children: r.map((t, j) => cell(t, W1[j], {
      bold: i === 0,
      fill: i === 0 ? HEAD_FILL : (j === 0 ? SIDE_FILL : undefined),
    })),
  })),
});

// ---------- 二、任务类型 ----------
const Y = [['✓', { bold: true, color: '2E7D32' }]];
const N = [['✗', { bold: true, color: 'C62828' }]];
const Q = [['？', { bold: true, color: '7F7F7F' }]];
const C = (t) => [['◐ ', { bold: true, color: 'E68A00' }], [t, {}]];
const YT = (t) => [['✓ ', { bold: true, color: '2E7D32' }], [t, {}]];
const NT = (t) => [['✗ ', { bold: true, color: 'C62828' }], [t, {}]];
const W2 = [1650, 2350, 1050, 2750, 1250, 3300, 2250];
const D = (t) => [[t, { color: 'A6A6A6' }]];
const tasks = [
  ['任务类型', '主要占比与分母', '时间迁移', '迁移代价参考／现状排队 P50／P90', '中途拆分', '拆分损失／最短分段', '关键约束'],
  ['预训练／SFT', 'Acme 95.5%／0.2%', Y, '预训练13秒／1.5分钟；SFT 11秒／13秒', C('检查点处'), '预训练43–512卡时；≥1.6–9.5小时。SFT 0.8–11卡时；≥0.3–3.2小时', '整组卡同停同启'],
  ['训练（未细分）', 'ASI全业务 ' + pct(FACTS.ASI_all_business_share.training), C('低优先级'), '独立提交时间未公开', C('需验证'), '检查点与恢复成本未测', '全业务背景；INF只取推理'],
  ['离线推理', 'ASI推理子集 ' + pct(FACTS.INF.offline_share_of_inference), C('核实期限'), '独立提交时间未公开；延后成本待核', C('按批次'), '加载模型1–5分钟为假设；每批≥19–95分钟', '断点续跑、输出不重复'],
  ['评估', 'Acme 0.2%', Y, '18秒／4.4分钟；作业时长P50 44秒', NT('无需拆'), D('—'), '保持训练依赖'],
  ['科研／云批处理', 'MIT 22.18%；Lisa CPU 99.98%；Alibaba离线 ' + pct(FACTS.CLD.batch_share), C('整体延后'), 'MIT批处理9秒／3.3小时；MapReduce 1.3／70小时；云任务期限未知', NT('基准不拆'), '无检查点时中断重算：损失＝已运行小时×卡数', '多节点；云任务保持DAG'],
  ['在线推理', 'ASI推理子集 ' + pct(1 - FACTS.INF.offline_share_of_inference), N, D('—'), N, D('—'), '持续在线、时延与可用性'],
  ['交互式／开发／调试', 'MIT 4.19%；ASI全业务开发 ' + pct(FACTS.ASI_all_business_share.dev) + '；Acme 1.4%', N, D('—'), N, D('—'), '即时响应；含Lisa Jupyter'],
  ['MIT DNN训练（含推断）', mitShare('DNN 训练（含推断）') + '；' + otherShare('DNN 训练（含推断）'), C('核实后延后'), '官方标签14.6／47.8小时；推断用途需确认', C('官方待验证'), '官方需验证检查点；推断基准不拆，恢复成本未测', '标签用途；推断需人工确认'],
  ['MIT 长时／多节点', (FACTS.MIT_OTHER['长时单任务'].pct_mit_gpu_hours + FACTS.MIT_OTHER['多节点大作业'].pct_mit_gpu_hours).toFixed(2) + '% MIT', C('整体延后'), '排队P90：长时13.8小时；多节点15.0小时', NT('基准不拆'), '检查点未观测；中断重算', '仅运行形态，核实用途和期限'],
  ['MIT 数组批量', mitShare('数组批量') + '；' + otherShare('数组批量'), C('成员改序'), '44分钟／30.2小时', C('成员之间'), '成员内部恢复未知；开销未测', '核实依赖、并发与期限'],
  ['MIT 低GPU活动占用', mitShare('低 GPU 活动占用') + '；' + otherShare('低 GPU 活动占用'), Q, '低GPU活动不证明业务空闲', Q, '恢复与释放成本未知', '核实后台CPU和释放权限'],
  ['其他与无法判断', mitShare('其他与无法判断') + '；' + otherShare('其他与无法判断') + '；Acme 2.6%', Q, D('—'), NT('基准不拆'), '用途与恢复能力待核实', '含OTHER余项，保留原执行'],
];
const taskTable = new Table({
  width: { size: W2.reduce((a, b) => a + b, 0), type: WidthType.DXA },
  columnWidths: W2,
  rows: tasks.map((r, i) => new TableRow({
    tableHeader: i === 0, cantSplit: true,
    children: r.map((t, j) => cell(t, W2[j], {
      bold: i === 0,
      size: 17,
      fill: i === 0 ? HEAD_FILL : (j === 0 ? SIDE_FILL : undefined),
      left: i > 0 && [1, 3, 5, 6].includes(j),
    })),
  })),
});

// ---------- 三、数据源概况 ----------
const W3 = [1750, 1250, 2150, 2650, 2350, 2850, 1600];
const data = [
  ['数据源', '对应需求', '时间跨度', '记录方式与规模', '作业时长 P50／P90', '任务分类有多细', '主要缺口'],
  ['Lisa（荷兰 SURF 超算）', '科研计算', '2021-12 至 2022-11，约 10 个月', '逐作业 159.7 万条（CPU 分区 139.6 万、GPU 分区 20.1 万）', 'CPU 分区 25 秒／55 分钟；GPU 分区 4.6 分钟／5.0 小时', [['粗', { bold: true }], ['：只有 17 个分区（CPU／GPU、Jupyter、课程），无用途和算法', {}]], 'GPU 卡数'],
  ['Alibaba 2018', '云通用计算', '8天，采用5个完整日', '4,034台机器；接入在线容器和离线实例用量', '离线实例P50 10秒、P90 99秒', [['粗', { bold: true }], ['：在线19.79%、离线51.13%、未对账29.08%（核心小时）', {}]], '未对账量、业务期限'],
  ['MIT Supercloud', '科研 AI', '237天需求；六月功率标定', '39.6万作业，GPU 9.82万；六月12,181文件完成', '原提交类P50 2.2–3.0小时、P90 22–37小时', [['中', { bold: true }], ['：OTHER已分12类，图合并6层；DNN用途证据占OTHER 4.6%（含推断）', {}]], '有效监测覆盖68.33%；用途真值与恢复'],
  ['Acme（上海 AI Lab Kalos 集群）', '大模型训练/开发', '2023-05 至 2023-08，约 3 个月', '逐作业 6.2 万条（GPU 1.99 万、CPU 4.25 万）', '预训练 11 分钟／11.8 小时；评估 44 秒／14 分钟', [['细', { bold: true }], ['：预训练、SFT、评估、调试、其他', {}]], 'RL 标签、检查点记录'],
  ['ASI（Alibaba GPU 2026）', '推理服务', '185天，4,440小时', '逐类型、优先级GPU小时；附执行跨度分布', '离线执行跨度P50 10–30分钟、P90 3–6小时', [['中', { bold: true }], ['：6类业务＋优先级＋模型；INF只取在线和离线推理', {}]], '真实日期、连续运行时间；1个零资源异常小时'],
];
const dataTable = new Table({
  width: { size: W3.reduce((a, b) => a + b, 0), type: WidthType.DXA },
  columnWidths: W3,
  rows: data.map((r, i) => new TableRow({
    tableHeader: i === 0, cantSplit: true,
    children: r.map((t, j) => cell(t, W3[j], {
      bold: i === 0,
      size: 18,
      fill: i === 0 ? HEAD_FILL : (j === 0 ? SIDE_FILL : undefined),
      left: i > 0 && j >= 2,
    })),
  })),
});

const TITLE = '数据中心任务时间柔性简表';
const DATE = '2026-11-01';
const H1 = '一、情境定义：服务器规格 × 需求';
const H2 = '二、任务类型与时间柔性';
const H3 = '三、各数据源概况';
const H4 = '四、各情境典型日负荷曲线';
const FIGURE_DIR = '../../results/scenario-typical-day-by-task-20261101-v3';
const FIGURE = path.resolve(__dirname, FIGURE_DIR, 'overview.png');
const NOTES4 = [
  '选日：完整日先按日电量接近中位数选取，并列时取日内 RMSE 最小者；GPU 需求以 GPU 主导型选日，三种规格共用日期和任务构成。',
  '规模与版本：参考IT 10 MW，功率、PUE和全年对照统一采用v3；其他规模按整数台数比缩放。',
  '类别与标定：MIT六月完成版（partial_data=false），DNN层含官方与推断；长时、数组、多节点仅为运行形态；所选日DNN层为零。',
  '来源边界：ASI 185天推理子集；Alibaba五日在线／离线／未对账分层。二者日历日期与星期未知；图中源日期和参考天气重放日分开记录。',
  '尖峰：Alibaba原始相对06时高值真实存在，单份SPECpower曲线50%→60%的陡升进一步放大功率峰形。',
];
const readmePath = process.argv[3];
const previousReadme = readmePath && fs.existsSync(readmePath) ? fs.readFileSync(readmePath, 'utf8') : '';
const sourceSection = previousReadme.indexOf('## 数字来源与说明');
const retainedSections = sourceSection >= 0 ? previousReadme.slice(sourceSection).trimEnd() : '';
const SOURCE4 = '- 第四章图与逐小时分任务汇总：[`results/scenario-typical-day-by-task-20261101-v3`](../../results/scenario-typical-day-by-task-20261101-v3/README.md)，选日评分见 `*-day-selection.csv`，日期、峰谷与全年电量核对见 `summary.csv`；`mit-other-full-source-shares.csv`、`mit-other-typical-day-shares.csv` 区分全源与所选日的 OTHER 分母。生成脚本为 `analysis/build_scenario_typical_days_v3.py`、`plot_scenario_typical_days_v3.py`；功率、PUE 与年度对照来自 PR #34 的 `results/typical-load-curves-20261101-v3/`，MIT 为六月窗口完成版。';
const SPIKE_SOURCE = '- Alibaba 第6小时尖峰：[原始数据核查与5分钟/10秒汇总](../../results/scenario-typical-day-by-task-20261031/cld-spike-audit-20261101/README.md)，包含原始包 SHA-256、八天复现、机器覆盖、保持上限敏感性、业务对账及非线性功率映射。';
const NOTE1 = '格中为该情境使用的需求数据，— 表示不设该组合。11 个组合各取参考 IT 功率 1／10／100／500 MW 四档规模，共 44 个情境，命名为“服务器规格·需求·规模”，例如 GPU 主导型·大模型训练/开发·100MW。括号内为每台服务器的 CPU∶GPU。';
const LEGEND = '✓ 可以　◐ 有条件　✗ 不行　？ 未知；MIT／OTHER分母分别为全部GPU小时／其中OTHER小时，跨数据集不相加。';
const NOTE2A = '占比为GPU小时；Lisa、Alibaba为核心小时。ASI注明“全业务”或“推理子集”；INF只用推理。现状排队不等于允许延后时长，运行形态不证明用途或调度权限。';
const NOTE2B = 'H＝保存＋释放与调度＋加载恢复＋重算；每次损失卡数×H，5%开销要求每段≥19H。预训练H取5–30分钟，其余H为假设，日志均未测到恢复成本。预训练/SFT卡数沿用512–1024／48–64卡参考。';
const NOTE2C = '候选量级：ASI离线且LP占推理GPU小时 ' + pct(FACTS.INF.offline_lp_share_of_inference) + '；Alibaba离线占核心小时 ' + pct(FACTS.CLD.batch_share) + '。实际可调功率仍需期限、权限及补做容量。';
const NOTE3 = 'Lisa GPU分区仅计宿主CPU核心小时；ASI执行跨度含生命周期及调度影响，不能当作连续运行时长。MIT六月有效监测覆盖68.33%、合格验证3小时，V100→H100与空载映射仍为假设。';

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 21 } } } },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838, orientation: PageOrientation.LANDSCAPE }, margin: { top: 700, bottom: 600, left: 1100, right: 1100 } } },
    children: [
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: TITLE, font: HEI, size: 32, bold: true })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: DATE, size: 20, color: '7F7F7F' })] }),

      h(H1),
      scenTable,
      p(NOTE1, { before: 100, size: 18, color: '595959' }),

      h(H2, true),
      p(LEGEND, { size: 19 }),
      taskTable,
      p(NOTE2A, { before: 100, size: 18, color: '595959' }),
      p(NOTE2B, { size: 18, color: '595959' }),
      p(NOTE2C, { size: 18, color: '595959' }),

      h(H3, true),
      dataTable,
      p(NOTE3, { before: 100, size: 18, color: '595959' }),

      h(H4, true),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        keepNext: true,
        spacing: { before: 40, after: 20 },
        children: [new ImageRun({ type: 'png', data: fs.readFileSync(FIGURE),
          transformation: { width: 944, height: 590 },
          altText: { name: '各情境典型日负荷曲线', title: '各情境典型日负荷曲线', description: '11 个情境按需求分行、服务器规格分列，共用设施功率纵轴。' },
        })],
      }),
      ...NOTES4.map((text) => p(text, { before: 20, size: 16, color: '595959' })),
    ],
  }],
});


// ---------- Markdown 版本（供 GitHub 审阅，与 Word 同源） ----------
const plain = (c) => (Array.isArray(c) ? c.map(([t]) => t).join('') : c).replace(/\|/g, '\|');
const mdTable = (rows) => [rows[0], rows[0].map(() => '---'), ...rows.slice(1)]
  .map((r) => '| ' + r.map((c) => (Array.isArray(c) || typeof c === 'string' ? plain(c) : String(c))).join(' | ') + ' |').join('\n');
const MD_OUT = process.argv[3];
if (MD_OUT) {
  const md = [
    `# ${TITLE}`, '', DATE, '',
    `## ${H1}`, '', mdTable(scen), '', NOTE1, '',
    `## ${H2}`, '', LEGEND, '', mdTable(tasks), '', NOTE2A, '', NOTE2B, '', NOTE2C, '',
    `## ${H3}`, '', mdTable(data), '', NOTE3, '',
    `## ${H4}`, '', `![各情境典型日负荷曲线](${FIGURE_DIR}/overview.png)`, '',
    ...NOTES4.flatMap((text) => [text, '']),
  ].join('\n');
  if (!retainedSections) throw new Error('README 缺少需要保留的数字来源与说明、重新生成章节');
  let tail = retainedSections.includes(SOURCE4) ? retainedSections
    : retainedSections.replace('## 重新生成', `${SOURCE4}\n\n## 重新生成`);
  if (!tail.includes(SPIKE_SOURCE)) tail = tail.replace('## 重新生成', `${SPIKE_SOURCE}\n\n## 重新生成`);
  fs.writeFileSync(MD_OUT, md + '\n' + tail + '\n', 'utf8');
  console.log('wrote', MD_OUT);
}

Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log('wrote', OUT); });
