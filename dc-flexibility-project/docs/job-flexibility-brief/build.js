const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType,
  AlignmentType, ShadingType, PageOrientation, VerticalAlign, HeadingLevel, BorderStyle, ImageRun,
} = require('docx');

const OUT = process.argv[2];
const RESULT_DIR = '../../results/typical-load-curves-20261101-v4-seren';
const FACTS = JSON.parse(fs.readFileSync(path.resolve(__dirname, RESULT_DIR, 'selected-days/brief-facts.json'), 'utf8'));
const NATIVE = JSON.parse(fs.readFileSync(path.resolve(__dirname, RESULT_DIR, 'brief-summary.json'), 'utf8'));
const pct = (value) => (value * 100).toFixed(2) + '%';
const compact = (value) => Number(value.toFixed(2)).toString();
const seren = (kind) => FACTS.SEREN[kind];
const duration = (seconds) => seconds < 60 ? compact(seconds) + '秒' : seconds < 3600 ? compact(seconds / 60) + '分钟' : compact(seconds / 3600) + '小时';
const pair = (kind, field) => duration(seren(kind)[field + '_p50']) + '／' + duration(seren(kind)[field + '_p90']);
const serenTiming = (kind) => '排队' + pair(kind, 'wait_s') + '；时长' + pair(kind, 'runtime_s');
const serenCards = (kind) => compact(seren(kind).units_p50) + '／' + compact(seren(kind).units_p90) + '卡';
const serenShare = (kind) => 'Seren ' + pct(seren(kind).work_fraction_of_resource_pool);
const splitCost = (kind, low, high) => compact(seren(kind).units_p50 * low / 60) + '–' + compact(seren(kind).units_p90 * high / 60) + '卡时；每段≥' + compact(19 * low / 60) + '–' + compact(19 * high / 60) + '小时';
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
      children: parts.flatMap(([t, o]) => t.split('\n').map((line, index) => new TextRun({ text: line, break: index > 0 ? 1 : undefined, size: opts.size || 20, bold: opts.bold || o.bold, color: o.color, font: opts.bold ? HEI : FONT }))),
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
  ['B CPU 主导型（2∶1）', NA, NA, 'MIT Supercloud', 'Acme Seren', 'ASI'],
  ['C 均衡型（2∶2）', NA, NA, 'MIT Supercloud', 'Acme Seren', 'ASI'],
  ['D GPU 主导型（2∶8）', NA, NA, 'MIT Supercloud', 'Acme Seren', 'ASI'],
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
  ['预训练', serenShare('Pretrain'), Y, serenTiming('Pretrain'), C('检查点处'), splitCost('Pretrain', 5, 30), serenCards('Pretrain') + '；整组同停同启'],
  ['SFT', serenShare('SFT'), Y, serenTiming('SFT'), C('检查点处'), splitCost('SFT', 1, 10), serenCards('SFT') + '；保持反馈依赖'],
  ['MLLM（多模态）', serenShare('MLLM'), C('核实离线用途'), serenTiming('MLLM'), C('验证检查点'), 'H未测；损失N×H；每段≥19H', serenCards('MLLM') + '；用途、期限待核'],
  ['训练（未细分）', 'ASI全业务 ' + pct(FACTS.ASI_all_business_share.training), C('低优先级'), '独立提交时间未公开', C('需验证'), '检查点与恢复成本未测', '全业务背景；INF只取推理'],
  ['离线推理', 'ASI推理子集 ' + pct(FACTS.INF.offline_share_of_inference), C('核实期限'), '独立提交时间未公开；延后成本待核', C('按批次'), '加载模型1–5分钟为假设；每批≥19–95分钟', '断点续跑、输出不重复'],
  ['评估', serenShare('Eval'), Y, serenTiming('Eval'), NT('无需拆'), D('—'), serenCards('Eval') + '；保持训练依赖'],
  ['科研／云批处理', 'MIT 22.18%；Lisa CPU 99.98%；Alibaba离线 ' + pct(FACTS.CLD.batch_share), C('整体延后'), 'MIT批处理9秒／3.3小时；MapReduce 1.3／70小时；云任务期限未知', NT('基准不拆'), '无检查点时中断重算：损失＝已运行小时×卡数', '多节点；云任务保持DAG'],
  ['在线推理', 'ASI推理子集 ' + pct(1 - FACTS.INF.offline_share_of_inference), N, D('—'), N, D('—'), '持续在线、时延与可用性'],
  ['交互式／开发／调试', 'MIT 4.19%；ASI全业务开发 ' + pct(FACTS.ASI_all_business_share.dev) + '；' + serenShare('Debug'), N, 'Seren ' + serenTiming('Debug'), N, D('—'), 'Seren ' + serenCards('Debug') + '；即时响应，含Jupyter'],
  ['MIT DNN训练（含推断）', mitShare('DNN 训练（含推断）') + '；' + otherShare('DNN 训练（含推断）'), C('核实后延后'), '官方标签14.6／47.8小时；推断用途需确认', C('官方待验证'), '官方需验证检查点；推断基准不拆，恢复成本未测', '标签用途；推断需人工确认'],
  ['MIT 长时／多节点', (FACTS.MIT_OTHER['长时单任务'].pct_mit_gpu_hours + FACTS.MIT_OTHER['多节点大作业'].pct_mit_gpu_hours).toFixed(2) + '% MIT', C('整体延后'), '排队P90：长时13.8小时；多节点15.0小时', NT('基准不拆'), '检查点未观测；中断重算', '仅运行形态，核实用途和期限'],
  ['MIT 数组批量', mitShare('数组批量') + '；' + otherShare('数组批量'), C('成员改序'), '44分钟／30.2小时', C('成员之间'), '成员内部恢复未知；开销未测', '核实依赖、并发与期限'],
  ['MIT 低GPU活动占用', mitShare('低 GPU 活动占用') + '；' + otherShare('低 GPU 活动占用'), Q, '低GPU活动不证明业务空闲', Q, '恢复与释放成本未知', '核实后台CPU和释放权限'],
  ['其他与无法判断', mitShare('其他与无法判断') + '；' + otherShare('其他与无法判断') + '；' + serenShare('Other'), Q, D('—'), NT('基准不拆'), '用途与恢复能力待核实', '含OTHER余项，保留原执行'],
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
  ['数据源', '对应需求', '时间跨度', '记录方式与规模', '作业时长／功率证据', '任务分类有多细', '主要缺口'],
  ['Lisa（荷兰 SURF 超算）', '科研计算', '2021-12 至 2022-11，约 10 个月', '逐作业 159.7 万条（CPU 分区 139.6 万、GPU 分区 20.1 万）', 'CPU 分区 25 秒／55 分钟；GPU 分区 4.6 分钟／5.0 小时', [['粗', { bold: true }], ['：只有 17 个分区（CPU／GPU、Jupyter、课程），无用途和算法', {}]], 'GPU 卡数'],
  ['Alibaba 2018', '云通用计算', '8天，采用5个完整日', '4,034台机器；接入在线容器和离线实例用量', '离线实例P50 10秒、P90 99秒', [['粗', { bold: true }], ['：在线19.79%、离线51.13%、未对账29.08%（核心小时）', {}]], '未对账量、业务期限'],
  ['MIT Supercloud', '科研 AI', '237天需求；六月功率标定', '39.6万作业，GPU 9.82万；六月12,181文件完成', '原提交类P50 2.2–3.0小时、P90 22–37小时', [['中', { bold: true }], ['：OTHER已分12类，图合并6层；DNN用途证据占OTHER 4.6%（含推断）', {}]], '有效监测覆盖68.33%；用途真值与恢复'],
  ['Acme Seren', '大模型训练/开发', '作业2023-02至08；正文2057小时、55完整日', '81.83万作业：GPU45.06万、独立CPU36.77万', '预训练' + pair('Pretrain', 'runtime_s') + '；功率报送≥80%、采样≥95%', [['细', { bold: true }], ['：六类含MLLM；集群逐小时反推，无作业映射', {}]], '189小时缺口；在线库存及缺测代表性'],
  ['ASI（Alibaba GPU 2026）', '推理服务', '185天，4,440小时', '逐类型、优先级GPU小时；附执行跨度分布', '离线执行跨度P50 10–30分钟、P90 3–6小时', [['中', { bold: true }], ['：6类业务＋优先级＋模型；INF只取在线和离线推理', {}]], '真实日期、连续运行时间；1个零资源异常小时'],
  ['BurstGPT／Azure 2024', '推理日内形状', '119日／代码和对话各7日', '请求及输入／输出token按小时聚合', '请求日内比2.455／8.963／1.783', '中央用BurstGPT输入／输出各50%；Azure为敏感性', '到达与执行时差；时钟、批量及缓存'],
  ['Seren同期GPU监测', '训练GPU功率', '2023-05-16 10时至08-17 23时UTC；合格2057小时', '2288卡在线库存；报送均功率外推后扣除未分配GPU空闲', 'GPU/TDP均值0.76；P10/P50/P90=0.61/0.77/0.90', '同小时共同功率比例；任务层按分配份额归因', 'A100→H100按TDP迁移；无逐作业映射'],
  ['SPECpower（三配置）', '全部CPU响应', '2023—2024公开结果', '3条整机AC曲线，共33个点', '空闲76–163W；满载318–915W', '归一化平均、单调平滑；保留端点', 'Java负载与目标配置迁移'],
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
const H5 = '五、各情境原生窗口负荷曲线';
const FIGURE_DIR = RESULT_DIR + '/selected-days';
const FIGURE = path.resolve(__dirname, FIGURE_DIR, 'overview.png');
const NATIVE_FIGURE = path.resolve(__dirname, RESULT_DIR, 'native-overview.png');
const NOTES4 = [
  '选日：先取完整日中位电量±2%内的候选日，少于5天放宽至±5%，再取与全部完整日平均24小时曲线RMSE最小者；GPU需求按GPU主导型筛选，三种规格共用日期。',
  '统一模型：两章同用#38 Seren正文2057个合格小时、#35推理请求GPU、#36 CPU及基础/空闲、cooling-20261028-v2。训练仅55个完整UTC日参选，选中2023-05-31；缺口不填充。',
  '无日期来源：先限定完整原生窗口，再取设施均值最接近12次放置中位数的放置用于选日；三种规格共用放置。第五章报告全部放置统计，日内比取全窗口按来源小时平均。',
  '训练GPU/TDP由同期监测反推，源GPU采样时间加权均值0.76，P10–P90为0.61–0.90。A100 400W按TDP比例迁移至H100 PCIe 350W或SXM 700W，吞吐未标定；逐任务功率未测。',
  'Alibaba原始相对06时高值保留，SPEC平滑后50%→60%动态增量约125→41W；ASI与Alibaba真实日期未知，请求与来源小时采用合成相位。',
];
const NOTE5A = '各来源有多长使用多长，不循环或补齐全年。Lisa 309天（2021-12-27至2022-10-31），时区未知，按UTC对应为假设；MIT 237天（2021-02-06至09-30），保留Slurm epoch小时。Seren为2023-05-16 10时至08-17 23时UTC内2057个合格小时（约86天，有缺口），按同期集群GPU监测反推已分配功率。三者按源月日小时对应气象年。Alibaba相对第2–6天共120小时，ASI相对第0–184天共185天；两者日期及时区未知，在12个月初00时UTC放置，年末截断。下表为参考网格2025年、参考IT 10MW。';
const NOTE5B = 'IT均值与峰值取完整原生窗口；设施均值、峰值及日内比取各实际放置窗口统计的中位数，范围为设施均值最小–最大。有日期来源仅一次对应。日内比为全窗口按源小时分组均值的最大/最小，区别于第四章所选日24小时比值。';
const NOTE5C = 'Alibaba仅5天，不报告周形状；ASI的周形状仅为相对七日。ASI七至十二月放置实际为184／153／122／92／61／31天，设施范围同时包含气象和源前缀差异。44情境的最小值、日内/周内比及窗口电量见配套数值表，均不年化。';
const NOTE5D = '训练分配GPU功率/TDP＝[报送均功率×2288−未分配卡数×60W]/[分配卡数×400W]，依赖在线库存与缺测代表性假设；Seren无逐作业设备映射。A100→H100按TDP比例迁移，均值0.76、P10–P90为0.61–0.90；保留8个超过TDP的小时及标记。4192小时补缺方案仅作对照。MIT为六月单月标定迁移H100。';
const NOTE5FIG = '蓝线为原生IT，紫线与阴影为逐源小时可用放置的设施中位数及最小–最大。训练横轴保留真实UTC时间，缺口断开曲线；有日期来源仅一条设施曲线。ASI后段可用放置减少；逐小时中位数包络的峰值与表中窗口峰值的中位数不同。';
const W5 = [4300, 1200, 1400, 1400, 1400, 1400, 1400, 2100];
const nativeTable = new Table({
  width: { size: W5.reduce((a, b) => a + b, 0), type: WidthType.DXA },
  columnWidths: W5,
  borders: Object.fromEntries(['top', 'bottom', 'left', 'right', 'insideHorizontal', 'insideVertical'].map((side) => [side, { style: BorderStyle.SINGLE, size: 4, color: 'D9D9D9' }])),
  rows: NATIVE.rows.map((row, i) => new TableRow({
    tableHeader: i === 0, cantSplit: true,
    children: row.map((text, j) => cell(text, W5[j], {
      bold: i === 0, size: 18, left: i > 0 && j === 0,
      fill: i === 0 ? HEAD_FILL : (i % 2 === 0 ? 'F3F6FA' : undefined),
    })),
  })),
});
const readmePath = process.argv[3];
const previousReadme = readmePath && fs.existsSync(readmePath) ? fs.readFileSync(readmePath, 'utf8') : '';
const sourceSection = previousReadme.indexOf('## 数字来源与说明');
const retainedSections = sourceSection >= 0 ? previousReadme.slice(sourceSection).trimEnd() : '';
const SOURCE4 = '- 第四章图与逐小时分任务汇总：[' + RESULT_DIR.split('/').pop() + '](' + RESULT_DIR + '/README.md)。与第五章同源；55个完整UTC日中选中2023-05-31，按中位电量±2%（少于5天±5%）内最小RMSE选日；三训练规格共日，MLLM单列。其余八情境保持上一版原生窗口结果。selected-days保存评分、小时值和任务分层，chapter4-comparison.csv核对两章取值。Seren任务统计来自PR #31 c16ac75的seren-job-statistics.json。';
const SOURCE5 = '- 第五章44情境、25组地点年份及原生时序：[' + RESULT_DIR.split('/').pop() + '](' + RESULT_DIR + '/README.md)。训练固定#38 bd6c8cd正文main，推理固定#35 520f68d，CPU及基础空闲沿用#36。native-summary.csv、placement-summary.csv和window-summary-all-locations.csv保留汇总；training-10MW-comparison.csv对照PR #37上一版训练数字，来源、窗口、归一化与模型同时变化，差值为描述性比较。manifest.json记录上游指纹及其余情境原字节保留校验。';
const SPIKE_SOURCE = '- Alibaba 第6小时尖峰：[原始数据核查与5分钟/10秒汇总](../../results/scenario-typical-day-by-task-20261031/cld-spike-audit-20261101/README.md)，包含原始包 SHA-256、八天复现、机器覆盖、保持上限敏感性、业务对账及非线性功率映射。';
const NOTE1 = '格中为该情境使用的需求数据，— 表示不设该组合。11 个组合各取参考 IT 功率 1／10／100／500 MW 四档规模，共 44 个情境，命名为“服务器规格·需求·规模”，例如 GPU 主导型·大模型训练/开发·100MW。括号内为每台服务器的 CPU∶GPU。';
const LEGEND = '✓ 可以　◐ 有条件　✗ 不行　？ 未知；MIT／OTHER分母分别为全部GPU小时／其中OTHER小时，跨数据集不相加。';
const NOTE2A = '占比为GPU小时；Lisa、Alibaba为核心小时。Seren占比用完整45.06万GPU作业；排队、时长、卡数为P50／P90，正文2057小时构成另列。ASI注明全业务或推理子集。现状排队、类型标签不证明服务期限或暂停权限。';
const NOTE2B = 'H＝保存＋释放调度＋恢复重算；损失N×H，5%开销要求每段≥19H。预训练H取5–30分钟，SFT取1–10分钟，均为假设；卡数用Seren P50／P90。迁移至H100 SXM的板卡参考约' + FACTS.power.H100_SXM_training_GPU_kwh_per_card_hour.toFixed(3) + ' kWh/卡时（源GPU采样加权，P10–P90约' + FACTS.power.H100_SXM_observed_p10_kwh.toFixed(3) + '–' + FACTS.power.H100_SXM_observed_p90_kwh.toFixed(3) + '）；恢复功率未测，整机及设施按小时模型计算。';
const NOTE2C = '候选量级：ASI离线且LP占推理GPU小时 ' + pct(FACTS.INF.offline_lp_share_of_inference) + '；Alibaba离线占核心小时 ' + pct(FACTS.CLD.batch_share) + '。实际可调功率仍需期限、权限及补做容量。';
const NOTE3 = '作业时长数字为P50／P90。Lisa GPU分区仅计宿主CPU核心小时；ASI执行跨度含生命周期及调度影响。MIT监测覆盖68.33%、合格验证3小时，跨设备与空载迁移为假设。无任务整机敏感性（科研AI另计）：A 120–320、B 195–500、C 270–680、D 1370–4110 W/台；科研AI只变基础及CPU空闲。';

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 21 } } } },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838, orientation: PageOrientation.LANDSCAPE }, margin: { top: 700, bottom: 600, left: 1100, right: 1100 } } },
    children: [
      new Paragraph({ style: 'Title', alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: TITLE, font: HEI, size: 32, bold: true, color: '000000' })] }),
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

      h(H5, true),
      p(NOTE5A, { before: 40, size: 19 }),
      nativeTable,
      ...[NOTE5B, NOTE5C, NOTE5D].map((text) => p(text, { before: 80, size: 18, color: '595959' })),

      h('原生窗口时序总览', true),
      new Paragraph({
        alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 40, after: 20 },
        children: [new ImageRun({ type: 'png', data: fs.readFileSync(NATIVE_FIGURE),
          transformation: { width: 944, height: 590 },
          altText: { name: '原生窗口时序总览', title: '原生窗口时序总览', description: '11个10MW情境各使用完整原生IT窗口，设施曲线显示气象对应或12次放置的中位数与范围。' },
        })],
      }),
      p(NOTE5FIG, { before: 20, size: 17, color: '595959' }),
    ],
  }],
});


// ---------- Markdown 版本（供 GitHub 审阅，与 Word 同源） ----------
const plain = (c) => (Array.isArray(c) ? c.map(([t]) => t).join('') : c).replace(/\|/g, '\|').replace(/\n/g, '<br>');
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
    `## ${H5}`, '', NOTE5A, '', mdTable(NATIVE.rows), '', NOTE5B, '', NOTE5C, '', NOTE5D, '',
    `![原生窗口时序总览](${RESULT_DIR}/native-overview.png)`, '', NOTE5FIG, '',
  ].join('\n');
  if (!retainedSections) throw new Error('README 缺少需要保留的数字来源与说明、重新生成章节');
  const sourceUpdated = retainedSections.split('\n').map((line) => line.startsWith('- 第四章图与逐小时分任务汇总：') ? SOURCE4 : line.startsWith('- 第五章44情境、25组地点年份及原生时序：') ? SOURCE5 : line).join('\n');
  let tail = sourceUpdated.includes(SOURCE4) ? sourceUpdated
    : retainedSections.replace('## 重新生成', `${SOURCE4}\n\n## 重新生成`);
  if (!tail.includes(SOURCE5)) tail = tail.replace('## 重新生成', `${SOURCE5}\n\n## 重新生成`);
  if (!tail.includes(SPIKE_SOURCE)) tail = tail.replace('## 重新生成', `${SPIKE_SOURCE}\n\n## 重新生成`);
  fs.writeFileSync(MD_OUT, md + '\n' + tail + '\n', 'utf8');
  console.log('wrote', MD_OUT);
}

Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log('wrote', OUT); });
