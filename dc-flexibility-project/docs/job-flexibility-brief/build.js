const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType,
  AlignmentType, ShadingType, PageOrientation, VerticalAlign, HeadingLevel, BorderStyle, ImageRun,
} = require('docx');

const OUT = process.argv[2];
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
  ['任务类型', '主要占比', '时间迁移', '迁移代价参考：现状排队 P50／P90', '中途拆分', '拆分代价：每拆一次损失卡时；每段最短时长', '关键约束'],
  ['预训练', 'Acme 95.5%', Y, '13 秒／1.5 分钟（作业时长 P90 11.8 小时）', C('检查点处'), '512–1024 卡 × H 5–30 分钟 ＝ 43–512 卡时；每段 ≥ 1.6–9.5 小时', '整组卡同停同启'],
  ['微调 SFT', 'Acme 0.2%', Y, '11 秒／13 秒（作业时长 P50 0.9 小时）', C('检查点处'), '48–64 卡 × H 1–10 分钟 ＝ 0.8–11 卡时；每段 ≥ 0.3–3.2 小时', '同预训练，规模小'],
  ['训练（未细分）', 'ASI 22.7%', C('低优先级部分'), '调度 1.9 分钟／5.7 分钟', C('需支持检查点'), '1–16 卡 × H 1–10 分钟 ＝ 0.02–2.7 卡时；每段 ≥ 0.3–3.2 小时', '分不出预训练与微调'],
  ['离线推理', 'ASI 22.4%', Y, '调度 0 秒／4.0 分钟', YT('按批次'), '1–4 卡 × 重新加载模型 1–5 分钟 ＝ 0.02–0.3 卡时；每批 ≥ 19–95 分钟', '断点续跑、输出不重复'],
  ['评估', 'Acme 0.2%', Y, '18 秒／4.4 分钟（作业时长 P50 44 秒）', NT('无需拆'), D('—'), '不能卡住下一轮训练'],
  ['科研批处理（含 MapReduce）', 'MIT 22.2%；Lisa CPU 分区 99.98%、GPU 科研分区 97.9%', YT('整体延后'), 'MIT 批处理 9 秒／3.3 小时；MapReduce 1.3／70 小时；Lisa CPU 3.4 分钟／12 小时', NT('启动后连续'), '没有检查点：中断即从头重算，损失 ＝ 已运行时长 × 卡数', '多节点同时分配'],
  ['在线推理', 'ASI 49.5%', N, D('—'), N, D('—'), '时延与可用性，须持续在线'],
  ['交互式／开发／调试', 'MIT 4.2%；ASI 3.7%；Acme 1.4%；Lisa Jupyter', N, D('—'), N, D('—'), '有人在用，需即时响应'],
  ['用途未知', 'MIT OTHER 73.6%（未用 LLsub 提交，用途未标注）；Acme 2.6%；ASI 1.7%；Lisa 课程 1.9%', Q, D('—'), Q, D('—'), '暂按不迁移、不拆分'],
];
const taskTable = new Table({
  width: { size: W2.reduce((a, b) => a + b, 0), type: WidthType.DXA },
  columnWidths: W2,
  rows: tasks.map((r, i) => new TableRow({
    tableHeader: i === 0, cantSplit: true,
    children: r.map((t, j) => cell(t, W2[j], {
      bold: i === 0,
      size: 18,
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
  ['Alibaba 2018', '云通用计算', '8 天，采用 5 个完整日', '4,034 台机器的 CPU 小时汇总', D('仅小时观测'), [['粗', { bold: true }], ['：CPU 总用量，在线／离线实际占用未分解', {}]], '在线／离线实际占用'],
  ['MIT Supercloud', '科研 AI', '2021-01 至 2021-10，约 9 个月', '逐作业 39.6 万条，其中 GPU 作业 9.8 万；另有逐秒 GPU 监测', '各类 2.2–3.0 小时／22–37 小时', [['粗', { bold: true }], ['：只有 4 种提交方式，OTHER 占 73.6%；官方标签约 3,425 个 DNN 作业（细分进行中）', {}]], '用途、算法'],
  ['Acme（上海 AI Lab Kalos 集群）', '大模型训练/开发', '2023-05 至 2023-08，约 3 个月', '逐作业 6.2 万条（GPU 1.99 万、CPU 4.25 万）', '预训练 11 分钟／11.8 小时；评估 44 秒／14 分钟', [['细', { bold: true }], ['：预训练、SFT、评估、调试、其他', {}]], 'RL 标签、检查点记录'],
  ['ASI（Alibaba GPU 2026）', '推理服务', '原始 185 天，目前只用 1 天（第 109 天）', '按 pod×服务器×小时观测 282 万条，52.5 万个 GPU pod', D('无法计算（只有小时观测）'), [['中', { bold: true }], ['：6 类业务（训练、在线推理、离线推理、开发等）＋高／低优先级＋模型领域', {}]], '完整作业时长；只有 1 天'],
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
const DATE = '2026-10-31';
const H1 = '一、情境定义：服务器规格 × 需求';
const H2 = '二、任务类型与时间柔性';
const H3 = '三、各数据源概况';
const H4 = '四、各情境典型日负荷曲线';
const FIGURE_DIR = '../../results/scenario-typical-day-by-task-20261031';
const FIGURE = path.resolve(__dirname, FIGURE_DIR, 'overview.png');
const NOTES4 = [
  '选日：完整日先按日电量接近中位数选取，并列时取日内 RMSE 最小者；GPU 需求以 GPU 主导型选日，三种规格共用日期和任务构成。',
  '规模：参考 IT 10 MW，沿用整机台数取整；其他规模按台数比例缩放，曲线形状不变。',
  '来源：Lisa、MIT、Acme 按作业与小时重叠秒数累计资源小时，ASI 用第 109 天小时观测；Alibaba 用五个完整日 CPU 汇总，业务未分解。',
  '不确定性：基础与空载 IT 独立列示；v2 功率与冷却均含假设，气象采用既有参考网格重放；ASI 及 Alibaba 的日历日期和星期未知。',
];
const readmePath = process.argv[3];
const previousReadme = readmePath && fs.existsSync(readmePath) ? fs.readFileSync(readmePath, 'utf8') : '';
const sourceSection = previousReadme.indexOf('## 数字来源与说明');
const retainedSections = sourceSection >= 0 ? previousReadme.slice(sourceSection).trimEnd() : '';
const SOURCE4 = '- 第四章图与逐小时分任务汇总：[`results/scenario-typical-day-by-task-20261031`](../../results/scenario-typical-day-by-task-20261031/README.md)，选日评分见 `*-day-selection.csv`，日期、峰谷与全年电量核对见 `summary.csv`；生成与功率分层见 `analysis/build_scenario_typical_days.py` 和 `plot_scenario_typical_days.py`，功率和 PUE 沿用 `results/typical-load-curves-20261029-v2/` 中央口径。';
const NOTE1 = '格中为该情境使用的需求数据，— 表示不设该组合。11 个组合各取参考 IT 功率 1／10／100／500 MW 四档规模，共 44 个情境，命名为“服务器规格·需求·规模”，例如 GPU 主导型·大模型训练/开发·100MW。括号内为每台服务器的 CPU∶GPU。';
const LEGEND = '✓ 可以　　◐ 有条件　　✗ 不行　　？ 未知　　（占比为各数据集自身的 GPU 小时，Lisa 为核心小时，不同数据集之间不能相加）';
const NOTE2A = '迁移代价：同样的工作量换个时段完成，额外耗电很少，主要代价是完成时间推迟。现状排队可作参考：科研集群用户已习惯数小时排队（P90 约 3–70 小时），AI 训练和推理集群几乎不排队（P90 不超过 6 分钟），延后数小时会明显改变服务水平。';
const NOTE2B = '拆分代价：每拆一次损失“卡数 × H”卡时，H ＝ 保存＋释放与重新调度＋加载恢复＋重算。开销控制在 5% 以内要求每段至少 19H。预训练 H 取 5–30 分钟（Google 多级检查点案例：恢复不到 1 分钟至 30 分钟）；其余 H 为假设，数据中均未测到。1 卡时约合 1.3 kWh（DGX H100 整机 10.2 kW ÷ 8 卡），预训练每拆一次约 0.05–0.65 MWh。';
const NOTE2C = '目前最有依据的柔性候选：ASI 低优先级训练＋离线推理，占 GPU 小时 22.7%。这是上限，实际可调比例未知。';
const NOTE3 = '作业时长为单个作业从开始到结束的时长。Lisa 的 GPU 分区没有卡数字段，只能用宿主 CPU 核心小时计量。';

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
          transformation: { width: 975, height: 609 },
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
  const tail = retainedSections.includes(SOURCE4) ? retainedSections
    : retainedSections.replace('## 重新生成', `${SOURCE4}\n\n## 重新生成`);
  fs.writeFileSync(MD_OUT, md + '\n' + tail + '\n', 'utf8');
  console.log('wrote', MD_OUT);
}

Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log('wrote', OUT); });
