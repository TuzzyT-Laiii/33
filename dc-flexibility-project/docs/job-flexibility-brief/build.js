const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType,
  AlignmentType, ShadingType, PageOrientation, VerticalAlign, HeadingLevel, BorderStyle, ImageRun,
} = require('docx');

const OUT = process.argv[2];
const RESULT_DIR = '../../results/typical-load-curves-20261101-v4-seren';
const FLEX_DIR = '../../results/flex-class-20261101';
const CITY_DIR = '../../results/eight-city-native-flex-20261102';
const CITY = JSON.parse(fs.readFileSync(path.resolve(__dirname, CITY_DIR, 'brief-summary.json'), 'utf8'));
const FLEX = JSON.parse(fs.readFileSync(path.resolve(__dirname, FLEX_DIR, 'brief-summary.json'), 'utf8'));
const FACTS = JSON.parse(fs.readFileSync(path.resolve(__dirname, RESULT_DIR, 'selected-days/brief-facts.json'), 'utf8'));
const NATIVE = JSON.parse(fs.readFileSync(path.resolve(__dirname, RESULT_DIR, 'brief-summary.json'), 'utf8'));
const compact = (value) => Number(value.toFixed(2)).toString();
const seren = (kind) => FACTS.SEREN[kind];
const duration = (seconds) => seconds < 60 ? compact(seconds) + '秒' : seconds < 3600 ? compact(seconds / 60) + '分钟' : compact(seconds / 3600) + '小时';
const pair = (kind, field) => duration(seren(kind)[field + '_p50']) + '／' + duration(seren(kind)[field + '_p90']);
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
const W2 = [3400, 2700, 2900, 3000, 2600];
const tasks = FLEX.classification_rows;
const taskTable = new Table({
  width: { size: W2.reduce((a, b) => a + b, 0), type: WidthType.DXA },
  columnWidths: W2,
  rows: tasks.map((r, i) => new TableRow({
    tableHeader: i === 0, cantSplit: true,
    children: r.map((t, j) => cell(t, W2[j], {
      bold: i === 0,
      size: 20,
      fill: i === 0 ? HEAD_FILL : (j === 0 ? SIDE_FILL : undefined),
      left: false,
    })),
  })),
});

// ---------- 三、数据源概况 ----------
const W3 = [1750, 1250, 2150, 2650, 2350, 2850, 1600];
const data = [
  ['数据源', '对应需求', '时间跨度', '记录方式与规模', '作业时长／功率证据', '任务分类有多细', '主要缺口'],
  ['Lisa（荷兰 SURF 超算）', '科研计算', '2021-12 至 2022-11，约 10 个月', '逐作业 159.7 万条（CPU 分区 139.6 万、GPU 分区 20.1 万）', 'CPU 分区 25 秒／55 分钟；GPU 分区 4.6 分钟／5.0 小时', [['粗', { bold: true }], ['：只有 17 个分区（CPU／GPU、Jupyter、课程），无用途和算法', {}]], 'GPU 卡数'],
  ['Alibaba 2018', '云通用计算', '8天，采用5个完整日', '4,034台机器；接入在线容器和离线实例用量', '离线实例P50 10秒、P90 99秒', [['粗', { bold: true }], ['：在线19.79%、离线51.13%、未对账29.08%（核心小时）', {}]], '未对账量、业务期限'],
  ['MIT Supercloud', '科研 AI', '237天需求；六月功率标定', '39.6万作业，GPU 9.82万；六月12,181文件完成', '原提交类P50 2.2–3.0小时、P90 22–37小时', [['中', { bold: true }], ['：OTHER已分12类；DNN用途证据占OTHER 4.6%（含推断）；图统一四类', {}]], '有效监测覆盖68.33%；用途真值与恢复'],
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
      size: 17,
      fill: i === 0 ? HEAD_FILL : (j === 0 ? SIDE_FILL : undefined),
      left: i > 0 && j >= 2,
    })),
  })),
});

const TITLE = '数据中心任务时间柔性简表';
const DATE = '2026-11-02';
const H1 = '一、情境定义：服务器规格 × 需求';
const H2 = '二、任务时间柔性分类';
const H3 = '三、各数据源概况';
const H4 = '四、各情境典型日负荷曲线';
const H5 = '五、各情境原生窗口负荷曲线';
const H6 = '六、八城市气候下的负荷与柔性';
const FIGURE_DIR = FLEX_DIR + '/selected-days';
const FIGURE = path.resolve(__dirname, FIGURE_DIR, 'overview.png');
const NATIVE_FIGURE = path.resolve(__dirname, FLEX_DIR, 'native-overview.png');
const SHARE_FIGURE = path.resolve(__dirname, FLEX_DIR, 'shares-10MW.png');
const CITY_PUE = path.resolve(__dirname, CITY_DIR, 'city-pue-heatmap.png');
const CITY_DAY = path.resolve(__dirname, CITY_DIR, 'city-typical-day-comparison.png');
const cityTable = (rows, widths) => new Table({
  width: {size: 14600, type: WidthType.DXA}, columnWidths: widths,
  rows: rows.map((row, i) => new TableRow({tableHeader: i === 0, cantSplit: true,
    children: row.map((text, j) => cell(text, widths[j], {bold: i === 0, size: 17,
      fill: i === 0 ? HEAD_FILL : (i % 2 === 0 ? 'F3F6FA' : undefined)})),
  })),
});
const climateTable = cityTable(CITY.climate_rows, [2200, 2500, 2500, 3700, 3700]);
const cityFacilityTable = cityTable(CITY.facility_rows, [1600, 2600, 2600, 2600, 2600, 2600]);
const CITY_NOTE = '主表取五种用途的纯CPU型或GPU主导型代表情境，单位MW；完整8城市×11情境见配套city-summary.csv。设施均值逐年对实际放置取中位数，再平均三年；有日期来源每年一次对应。';
const CITY_PUE_NOTE = 'PUE为窗口设施/IT电量比，逐年对放置取中位数后平均三年，非全年PUE。';
const NOTES4 = [
  '选日：先取完整日中位电量±2%内的候选日，少于5天放宽至±5%，再取与全部完整日平均24小时曲线RMSE最小者；GPU需求按GPU主导型筛选，三种规格共用日期。',
  '统一模型：两章同用#38 Seren正文2057个合格小时、#35推理请求GPU、#36 CPU及基础/空闲、cooling-20261028-v2。训练仅55个完整UTC日参选，选中2023-05-31；缺口不填充。',
  '无日期来源：先限定完整原生窗口，再取设施均值最接近12次放置中位数的放置用于选日；三种规格共用放置。第五章报告全部放置统计，日内比取全窗口按来源小时平均。',
  '训练GPU/TDP由同期监测反推，源GPU采样时间加权均值0.76，P10–P90为0.61–0.90。A100 400W按TDP比例迁移至H100 PCIe 350W或SXM 700W，吞吐未标定；逐任务功率未测。',
  '四类按第二章映射合并，基础与空载IT、冷却等非IT单列，所有图共用六项图例与配色；未知默认按刚性处理。曲线按原有任务资源份额分摊，不代表可调电量。',
];
const NOTE5A = '各来源有多长使用多长，不循环或补齐全年。Lisa 309天（2021-12-27至2022-10-31），时区未知，按UTC对应为假设；MIT 237天（2021-02-06至09-30），保留Slurm epoch小时。Seren为2023-05-16 10时至08-17 23时UTC内2057个合格小时（约86天，有缺口），按同期集群GPU监测反推已分配功率。三者按源月日小时对应气象年。Alibaba相对第2–6天共120小时，ASI相对第0–184天共185天；两者日期及时区未知，在12个月初00时UTC放置，年末截断。下表为参考网格2025年、参考IT 10MW。';
const NOTE5B = 'IT均值与峰值取完整原生窗口；设施均值、峰值及日内比取各实际放置窗口统计的中位数，范围为设施均值最小–最大。有日期来源仅一次对应。日内比为全窗口按源小时分组均值的最大/最小，区别于第四章所选日24小时比值。';
const NOTE5C = 'Alibaba仅5天，不报告周形状；ASI的周形状仅为相对七日。ASI七至十二月放置实际为184／153／122／92／61／31天，设施范围同时包含气象和源前缀差异。44情境的最小值、日内/周内比及窗口电量见配套数值表，均不年化。';
const NOTE5D = '训练分配GPU功率/TDP＝[报送均功率×2288−未分配卡数×60W]/[分配卡数×400W]，依赖在线库存与缺测代表性假设；Seren无逐作业设备映射。A100→H100按TDP比例迁移，均值0.76、P10–P90为0.61–0.90；保留8个超过TDP的小时及标记。4192小时补缺方案仅作对照。MIT为六月单月标定迁移H100。';
const NOTE5FIG = '六层相加为设施功率，四类及配色与第四章相同。训练横轴保留真实UTC时间，缺口断开；无日期来源先限定完整原生窗口，再选设施均值最接近12次实际放置均值中位数的一次放置。图内注释给出全部放置的设施均值范围，包含气象与截断源前缀差异；该代表曲线区别于第五章表中跨放置汇总。';
const SHARE_TITLE = '原生窗口四类任务份额';
const SHARE_NOTE = '下表与条形图均以同一原生窗口的总IT电量为分母；该类电量占比也等于该类平均功率占总IT平均功率的比例。基础与空载IT单列，与四类合计100%；冷却等非IT不计入IT分母。单位：%。';
const SHARE_LIMIT = '份额由任务类型的CPU/GPU资源份额分摊模型IT功率得到；“能推迟多久、拆成多长一段”及存档恢复成本尚无数据支撑，作为情景假设另述，本次不据此计算可调电量。未知默认按刚性处理。';
const WS = [4900, 1900, 2100, 2200, 1500, 2000];
const shareTable = new Table({
  width: { size: 14600, type: WidthType.DXA }, columnWidths: WS,
  rows: FLEX.share_rows.map((row, i) => new TableRow({ tableHeader: i === 0, cantSplit: true,
    children: row.map((text, j) => cell(text, WS[j], {bold: i === 0, size: 18, left: i > 0 && j === 0,
      fill: i === 0 ? HEAD_FILL : (i % 2 === 0 ? 'F3F6FA' : undefined)})),
  })),
});
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
const SOURCE4 = '- 第四章图与逐小时分任务汇总：[' + FLEX_DIR.split('/').pop() + '](' + FLEX_DIR + '/README.md)。按确认的四类映射合并原始任务功率层，保留基础与空载IT和冷却等非IT。11个典型日日期、小时功率及选日规则与Seren组合结果保持；native-hourly及selected-days分别保存原生与典型日六层值。全套图统一六项图例与配色。';
const SOURCE5 = '- 第五章44情境、25组地点年份及原生时序：[' + RESULT_DIR.split('/').pop() + '](' + RESULT_DIR + '/README.md)。训练固定#38 bd6c8cd正文main，推理固定#35 520f68d，CPU及基础空闲沿用#36。native-summary.csv、placement-summary.csv和window-summary-all-locations.csv保留汇总；training-10MW-comparison.csv对照PR #37上一版训练数字，来源、窗口、归一化与模型同时变化，差值为描述性比较。manifest.json记录上游指纹及其余情境原字节保留校验。';
const SOURCE6 = '- 第六章八城市气候与负荷：[' + CITY_DIR.split('/').pop() + '](' + CITY_DIR + '/README.md)。核验8份冻结气象及1320份10MW城市原生小时，复用#35/#36组合与#38 Seren结果；case-summary.csv、year-summary.csv、city-summary.csv分别保存逐窗口、逐年和三年汇总。facility-composition.csv保留六层功率，冷却非IT单列；典型日采用第五章同日逐小时IT份额。verification.json保存SHA-256及24组冷却复算，brief-audit.json保存文档检查。';
const SPIKE_SOURCE = '- Alibaba 第6小时尖峰：[原始数据核查与5分钟/10秒汇总](../../results/scenario-typical-day-by-task-20261031/cld-spike-audit-20261101/README.md)，包含原始包 SHA-256、八天复现、机器覆盖、保持上限敏感性、业务对账及非线性功率映射。';
const NOTE1 = '格中为该情境使用的需求数据，— 表示不设该组合。11 个组合各取参考 IT 功率 1／10／100／500 MW 四档规模，共 44 个情境，命名为“服务器规格·需求·规模”，例如 GPU 主导型·大模型训练/开发·100MW。括号内为每台服务器的 CPU∶GPU。';
const LEGEND = '① 刚性：不能推迟、不能拆分，有人在等或持续在线服务。② 可推迟·整体：可以晚点运行，开始后一次完成。③ 可推迟·可拆分：可以晚点运行，也可在存档点暂停后续跑。未知：数据看不出用途，保守处理。';
const NOTE2A = '表中每格为原始细类按任务用途及推迟、拆分能力合并后的一个代表名称，完整映射见配套数值文件，未知默认按刚性处理。';
const NOTE2B = '此表采用统一的情景分类；用途标签与运行形态本身不提供服务期限、检查点或暂停恢复的实测证据，实际调度仍需核实业务约束。';
const NOTE2C = '原生窗口的分类份额见第五章；推迟时长、分段长度和恢复成本作为情景假设另述，本次不给出可调电量结论。';
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
      p(LEGEND, { size: 18 }),
      taskTable,
      p(NOTE2A, { before: 60, size: 16, color: '595959' }),
      p(NOTE2B + NOTE2C, { size: 16, color: '595959' }),

      h(H3),
      dataTable,
      p(NOTE3, { before: 60, size: 16, color: '595959' }),

      h(H4, true),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        keepNext: true,
        spacing: { before: 40, after: 20 },
        children: [new ImageRun({ type: 'png', data: fs.readFileSync(FIGURE),
          transformation: { width: 944, height: 590 },
          altText: { name: '各情境典型日负荷曲线', title: '各情境典型日负荷曲线', description: '11个情境采用四类任务及基础与空载IT、冷却等非IT的六层堆叠，统一配色。' },
        })],
      }),
      ...NOTES4.map((text) => p(text, { before: 20, size: 16, color: '595959' })),

      h(H5, true),
      p(NOTE5A, { before: 40, size: 19 }),
      nativeTable,
      ...[NOTE5B, NOTE5C, NOTE5D].map((text) => p(text, { before: 80, size: 18, color: '595959' })),

      h(SHARE_TITLE, true),
      p(SHARE_NOTE, { before: 30, size: 17, color: '595959' }),
      shareTable,
      new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 40, after: 10 },
        children: [new ImageRun({type: 'png', data: fs.readFileSync(SHARE_FIGURE), transformation: {width: 944, height: 301},
          altText: {name: SHARE_TITLE, title: SHARE_TITLE, description: '11情境原生窗口四类IT电量份额条形图，基础与空载IT补足100%，冷却不计IT。'}})],
      }),
      p(SHARE_LIMIT, {before: 10, size: 16, color: '595959'}),

      h('原生窗口时序总览', true),
      new Paragraph({
        alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 40, after: 20 },
        children: [new ImageRun({ type: 'png', data: fs.readFileSync(NATIVE_FIGURE),
          transformation: { width: 944, height: 590 },
          altText: { name: '原生窗口时序总览', title: '原生窗口时序总览', description: '11个10MW情境原生窗口按四类任务及基础空载、冷却六层堆叠；代表气象放置及范围另注，真实缺口断开。' },
        })],
      }),
      p(NOTE5FIG, { before: 20, size: 17, color: '595959' }),

      h(H6, true),
      p(CITY.scope + '各城气候指标按三年实际26304小时等权；露点≥20°C表示高水汽含量。', {before: 30, size: 17}),
      climateTable,
      p('10MW代表情境的设施平均功率', {before: 80, size: 20}),
      cityFacilityTable,
      p(CITY_NOTE, {size: 16, color: '595959'}),
      p(CITY.conclusion, {size: 17}),
      p(CITY.comparison, {size: 17}),

      h('八城市 PUE 与典型日构成', true),
      new Paragraph({alignment: AlignmentType.CENTER, keepNext: true, spacing: {before: 20, after: 10},
        children: [new ImageRun({type: 'png', data: fs.readFileSync(CITY_PUE), transformation: {width: 944, height: 277},
          altText: {name: '八城市原生窗口PUE', title: '八城市原生窗口PUE', description: '全部8城市与11情境，冻结2023至2025气象，窗口电量比PUE的逐年放置中位数再平均。'}})],
      }),
      p(CITY_PUE_NOTE, {before: 10, size: 16, color: '595959'}),
      new Paragraph({alignment: AlignmentType.CENTER, keepNext: true, spacing: {before: 20, after: 10},
        children: [new ImageRun({type: 'png', data: fs.readFileSync(CITY_DAY), transformation: {width: 944, height: 248},
          altText: {name: '三城市训练典型日对比', title: '三城市训练典型日对比', description: '吕勒奥、利雅得和新加坡按同一Seren来源日，六层相加为设施功率，冷却非IT单列。'}})],
      }),
      p(CITY.composition + CITY.flexibility, {before: 10, size: 16, color: '595959'}),
      p(CITY.limitations, {size: 16, color: '595959'}),
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
    `### ${SHARE_TITLE}`, '', SHARE_NOTE, '', mdTable(FLEX.share_rows), '', `![四类份额](${FLEX_DIR}/shares-10MW.png)`, '', SHARE_LIMIT, '',
    `![原生窗口时序总览](${FLEX_DIR}/native-overview.png)`, '', NOTE5FIG, '',
    `## ${H6}`, '', CITY.scope, '', mdTable(CITY.climate_rows), '', '### 10MW代表情境设施平均功率（MW）', '',
    mdTable(CITY.facility_rows), '', CITY_NOTE, '', CITY.conclusion, '', CITY.comparison, '',
    `![八城市PUE](${CITY_DIR}/city-pue-heatmap.png)`, '', CITY_PUE_NOTE, '',
    `![三城市典型日](${CITY_DIR}/city-typical-day-comparison.png)`, '', CITY.composition, '', CITY.flexibility, '', CITY.limitations, '',
  ].join('\n');
  if (!retainedSections) throw new Error('README 缺少需要保留的数字来源与说明、重新生成章节');
  const sourceUpdated = retainedSections.split('\n').map((line) => line.startsWith('- 第四章图与逐小时分任务汇总：') ? SOURCE4 : line.startsWith('- 第五章44情境、25组地点年份及原生时序：') ? SOURCE5 : line).join('\n');
  let tail = sourceUpdated.includes(SOURCE4) ? sourceUpdated
    : retainedSections.replace('## 重新生成', `${SOURCE4}\n\n## 重新生成`);
  if (!tail.includes(SOURCE5)) tail = tail.replace('## 重新生成', `${SOURCE5}\n\n## 重新生成`);
  if (!tail.includes(SOURCE6)) tail = tail.replace('## 重新生成', `${SOURCE6}\n\n## 重新生成`);
  if (!tail.includes(SPIKE_SOURCE)) tail = tail.replace('## 重新生成', `${SPIKE_SOURCE}\n\n## 重新生成`);
  fs.writeFileSync(MD_OUT, md + '\n' + tail + '\n', 'utf8');
  console.log('wrote', MD_OUT);
}

Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log('wrote', OUT); });
