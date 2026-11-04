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
const ECON_DIR = '../../results/flex-card-cost-calendar-20261102';
const GLOBAL_DIR = '../../results/global-flex-econ-20261103';
const GLOBAL = JSON.parse(fs.readFileSync(path.resolve(__dirname, GLOBAL_DIR, 'brief/brief-facts.json'), 'utf8'));
const ECON = JSON.parse(fs.readFileSync(path.resolve(__dirname, ECON_DIR, 'brief/brief-facts.json'), 'utf8'));
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
const DATE = '2026-11-04';
const H1 = '一、情境定义：服务器规格 × 需求';
const H2 = '二、任务时间柔性分类';
const H3 = '三、各数据源概况';
const H4 = '四、各情境典型日负荷曲线';
const H5 = '五、各情境原生窗口负荷曲线';
const H6 = '六、八城市气候下的负荷与柔性';
const H7 = '七、柔性调度的经济性（计入加卡，初算）';
const H8 = '八、全球网格的柔性经济性（2° 抽样试算）';
const GLOBAL_PLAN_MAP = path.resolve(__dirname, GLOBAL_DIR, 'figures/planning-economic.png');
const GLOBAL_FIXED_MAP = path.resolve(__dirname, GLOBAL_DIR, 'figures/fixed-economic.png');
const TP = GLOBAL.train_plan_econ;
const NOTE8A = '把第七章模型从八城市铺到全球：全球' + GLOBAL.grids_sampled.toLocaleString('en-US') + '个2°区块各取一个可建设0.5°网格，用该格ERA5 2025年逐小时气象重算风光出力与PUE，11情境×6档匹配目标×规划/已建成两种边界全部求解；'
  + GLOBAL.grids_valid.toLocaleString('en-US') + '格有效，全部线性规划最优，另' + GLOBAL.grids_no_land + '格无陆地气象点未算。成本、电价、碳价与加卡口径同第七章，市电碳因子按所在国家取Ember 2024年均值。下表为有效网格等权中位数。';
const NOTE8B = '已建成边界下100%匹配的收益全部约为0（最大' + GLOBAL.fixed_alpha100_max_pct.toPrecision(2) + '%），表中略去；全部柔性最优解均不需新增算力。八城市列取第七章7.1表经济最优中位数，两者口径相同、样本不同。';
const NOTE8C = '训练（GPU主导型）规划阶段经济最优的全球中位节省为' + TP.median.toFixed(1) + '%，四分位' + TP.p25.toFixed(1) + '–' + TP.p75.toFixed(1) + '%。节省高的是光照好、风电弱的干旱带（北非、中东、澳洲内陆、中亚），可达30%以上；风资源好的高纬与南美南端最低，仅2–4%。推理在各地都在3–4%，地域差别小。';
const GLOBAL_THRESHOLD_MAP = path.resolve(__dirname, GLOBAL_DIR, 'figures/threshold-fixed-economic-3pct.png');
const THRESHOLD = JSON.parse(fs.readFileSync(path.resolve(__dirname, GLOBAL_DIR, 'figures/threshold-fixed-economic-3pct.json'), 'utf8'));
const NOTE8T = '模型未计推迟带来的延误、存档恢复与运营成本，取3%作为柔性“值得做”的下限，给这些成本留出余量。已建成、经济最优下，超过3%的网格占比：'
  + THRESHOLD.panels.map((row) => ({ 'D-TRN': '训练', 'D-SAI': '科研AI', 'A-HPC': '科研计算', 'D-INF': '推理' })[row.scenario] + row.share_pct.toFixed(0) + '%').join('、')
  + '。推理只在光照强的干旱区（撒哈拉、阿拉伯半岛、中亚至中国西北、美国西南、澳洲内陆）超过3%；训练与科研计算几乎全球都超过。分母为各情境' + THRESHOLD.panels[0].grids.toLocaleString('en-US') + '个有效网格。';
const NOTE8MAP = '每个2°区块按其代表网格的结果整体着色（裁到陆地），表示抽样代表值；每幅图单独色标，上限取该图98分位，超出用三角标示。空白为无可建设网格或无陆地气象点的区块。';
const NOTE8LIMIT = '解释边界：2°抽样、成本全球统一、单一气象年与年均碳因子；低于−40°C（534格）与高于50°C（3格）的小时在冷却模型输入端夹到边界值；负荷沿用各来源原生窗口，收益不是完整全年值。完整结果见results/global-flex-econ-20261103。';
const K_FIGURE = path.resolve(__dirname, FLEX_DIR, 'task-peak-capacity.png');
const K_ROWS = (() => {
  const lines = fs.readFileSync(path.resolve(__dirname, FLEX_DIR, 'task-peak-capacity.csv'), 'utf8').replace(/^\uFEFF/, '').trim().split(/\r?\n/);
  const head = lines[0].split(',');
  return lines.slice(1).map((line) => Object.fromEntries(line.split(',').map((value, i) => [head[i], value])));
})();
const K_TITLE = '算力容量 K 的来源：任务 IT 窗口峰值';
const NOTE7K = '本章“不加卡”的算力容量 K 取原生窗口内任务 IT 功率（刚性＋未知＋两类可推迟任务，按原到达时刻）的最大值，不含基础与空载 IT 和冷却等非 IT，因此低于第五章时序总览中的设施峰值。'
  + K_ROWS.map((row) => row.scenario_name + ' K＝' + Number(row.K_mw).toFixed(2) + 'MW，平均为K的' + Number(row.mean_pct_of_K).toFixed(0) + '%，超过95%K仅' + row.hours_above_95pct_K + '小时').join('；') + '。'
  + '多数时段离K尚有30–40%空档，两类可推迟任务可在12/48小时内利用这些空档移位，因而柔性最优解不需加卡。K由单一最高时刻决定，对尖峰敏感：科研计算平时约5.5MW，K来自第101天的6.83MW尖峰；若改用较稳健的分位数定容量，空档与柔性收益会相应减小。';
const ECON_VALUE = path.resolve(__dirname, ECON_DIR, 'brief/value-summary.png');
const ECON_DISPATCH = path.resolve(__dirname, ECON_DIR, 'brief/dispatch-week.png');
const NOTE7A = '参考IT 10MW，杉数COPT求解八城市×11情境原生窗口。两组均允许按基准购置价格情境增加算力。柔性收益＝无柔性模型成本－有柔性模型成本；收益%以无柔性模型成本为分母，仅计能源系统与新增算力，既有机群资本、延误与恢复成本未计入。窗口本地匹配率＝1－窗口市电电量/窗口设施电量；95%约束窗口总量，100%要求各时段市电为0。下表为有效城市配对的中位数，年化金额由原生窗口折算。';
const SUB7A = '7.1 规划阶段（风光储重新优化）';
const NOTE7B1 = '有柔性与无柔性分别优化风光、储能和市电接入容量。可推迟IT电量为第二章情景分类份额；新增算力以等效任务功率计量，表中报告该情境各城市与目标的最大值，小于10⁻⁶MW记为0。';
const SUB7B = '7.2 已建成（风光储固定）';
const reducedPairs = ECON.sample_counts.filter((row) => row.built.some((count) => count !== 8));
const NOTE7B2 = '风光、储能与市电接入固定为无柔性最优配置，只优化运行调度与新增算力。每推迟1MWh的分母为两类任务分别逐小时累计的max(到达功率－执行功率,0)×1h，即移出原时段的任务IT电量。该项与储能单位成本取经济最优目标；“若加卡”假设新增卡每天承接8小时推迟负荷。'
  + (reducedPairs.length ? '有效城市数（经济最优/95%/100%）：' + reducedPairs.map((row) => row.name + ' ' + row.built.join('/')).join('；') + '；其余代表情境均为8/8/8。' : '各代表情境、各目标均有8个有效城市配对。');
const trainPlan = ECON.plan_rows.find((row) => row[0] === 'GPU 主导型·大模型训练/开发');
const trainBuilt = ECON.built_rows.find((row) => row[0] === 'GPU 主导型·大模型训练/开发');
const NOTE7C = 'GPU主导型训练在100%目标下，规划阶段储能中位数为' + trainPlan[4] + 'MWh，模型成本收益为' + trainPlan[2].split('／')[2] + '%，年化节省' + trainPlan[3].split('／')[2] + '万美元。固定已建成容量时，同一目标的模型成本收益约为' + trainBuilt[1].split('／')[2] + '%。该结果取决于本章固定电价、容量固定和成本项目；储能退化、备用与其他服务价值未纳入。';
const NOTE7METHOD = '调度固定刚性、未知和基础空载IT；两类可推迟任务的上限分别设为12和48小时，按源记录的真实时间间隔约束，缺测小时不补写任务或执行量。算力容量以调度前任务功率峰值近似，超过时须付新增服务器年化费用并计全时待机耗电。服务器价情景：纯CPU 2万、配置1/2张H100 PCIe为4.75/7.5万、DGX H100为35万美元；年化系数0.332。风光成本取IRENA 2024、储能取ATB 2025；市电100 USD/MWh、碳价100 USD/t、接入费120 USD/kW·年。';
const NOTE7D = '法兰克福2025-06-24至06-30（UTC）。上排无柔性、下排有柔性，四种情况各自重新优化容量。红虚线为调度前设施负荷，黑线为调度后；面积为光伏、风电、储能放电和市电，高出负荷部分为储能充电或弃电。100%本地无碳下储能为' + ECON.dispatch_battery_mwh_100.map((value) => value.toFixed(0)).join('→') + 'MWh；这是单城示例，与两表的跨城市中位数分别报告。';
const NOTE7E = '解释边界：任务用途与利用率不提供业务调度权限；12/48小时为情景设定。容量尚无逐卡约束；可推迟·整体尚未约束启动后连续执行。服务器价为公开估计与量级假设，推迟和存档恢复的运营成本未知；分时电价与售电未计，Alibaba窗口仅120小时。等效年化收益按观测窗口小时折算，缺测小时未计算，不能视为完整全年收益。三个单位指标的分母分别为推迟电量、储能放电量和假设新增卡承接电量。完整结果见results/flex-card-cost-calendar-20261102。';
const tableOf = (rows, widths) => new Table({
  width: { size: widths.reduce((x, y) => x + y, 0), type: WidthType.DXA }, columnWidths: widths,
  borders: Object.fromEntries(['top', 'bottom', 'left', 'right', 'insideHorizontal', 'insideVertical'].map((side) => [side, { style: BorderStyle.SINGLE, size: 4, color: 'D9D9D9' }])),
  rows: rows.map((row, i) => new TableRow({ tableHeader: i === 0, cantSplit: true,
    children: row.map((text, j) => cell(text, widths[j], { bold: i === 0, size: 18, left: i > 0 && j === 0,
      fill: i === 0 ? HEAD_FILL : (i % 2 === 0 ? 'F3F6FA' : undefined) })),
  })),
});
const globalTable = tableOf(GLOBAL.table_rows, [3400, 3200, 2400, 2600, 2400]);
const regionTable = tableOf(GLOBAL.region_rows, [2600, ...Array(12).fill(1000)]);
const planTable = tableOf(ECON.plan_rows, [3200, 1300, 2800, 2800, 2400, 1800]);
const builtTable = tableOf(ECON.built_rows, [3000, 2500, 2300, 1500, 1300, 2100, 1600]);
const sub = (t) => new Paragraph({ keepNext: true, spacing: { before: 80, after: 30 }, children: [new TextRun({ text: t, font: HEI, size: 20, bold: true, color: '000000' })] });
const economicHeading = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, spacing: { before: 160, after: 80 }, children: [new TextRun({ text: t, font: HEI, size: 26, bold: true, color: '000000' })] });
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
const SOURCE8 = '- 第八章全球网格：[' + GLOBAL_DIR.split('/').pop() + '](' + GLOBAL_DIR + '/README.md)。4,900个2°代表网格×11情境×6档目标×两种边界，958,716个COPT线性规划；brief/brief-facts.json由analysis/global_flex_brief_facts.py从grid-summary.parquet生成，地图为figures/planning-economic与fixed-economic。';
const SOURCE7 = '- 第七章柔性调度经济性：[' + ECON_DIR.split('/').pop() + '](' + ECON_DIR + '/README.md)。COPT规划阶段' + ECON.run_counts.planning + '个算例（runs-cards.csv）与固定供给侧' + ECON.run_counts.built + '个算例（runs-fixed.csv）；训练按实际UTC时间约束12/48小时延迟，brief/保存本章图表、输入校验值及核验记录。';
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

      economicHeading(H7),
      p(NOTE7A, {before: 30, size: 17}),
      sub(SUB7A),
      planTable,
      p(NOTE7B1, {before: 20, size: 16, color: '595959'}),
      sub(SUB7B),
      builtTable,
      p(NOTE7B2, {before: 20, size: 16, color: '595959'}),
      p(NOTE7C, {before: 30, size: 18}),

      economicHeading(K_TITLE),
      new Paragraph({alignment: AlignmentType.CENTER, keepNext: true, spacing: {before: 20, after: 10},
        children: [new ImageRun({type: 'png', data: fs.readFileSync(K_FIGURE), transformation: {width: 760, height: 605},
          altText: {name: K_TITLE, title: K_TITLE, description: '五类代表情境原生窗口任务IT四类堆叠时序及K（窗口峰值），右侧为任务IT持续曲线。'}})],
      }),
      p(NOTE7K, {before: 10, size: 16, color: '595959'}),

      economicHeading('柔性收益与单位成本'),
      new Paragraph({alignment: AlignmentType.CENTER, keepNext: true, spacing: {before: 30, after: 10},
        children: [new ImageRun({type: 'png', data: fs.readFileSync(ECON_VALUE), transformation: {width: 944, height: 261},
          altText: {name: '柔性收益与单位成本', title: '柔性收益与单位成本', description: '左：五种用途代表情境不加卡时柔性收益随本地匹配比例变化；右：峰值内柔性每推迟1MWh的节省、储能每放1MWh成本与加卡柔性每推迟1MWh成本，对数轴。'}})],
      }),
      p(NOTE7METHOD, {before: 60, size: 19}),
      p(NOTE7E, {before: 60, size: 19, color: '595959'}),

      economicHeading('调度前后对比（法兰克福·GPU 主导型·大模型训练/开发）'),
      new Paragraph({alignment: AlignmentType.CENTER, keepNext: true, spacing: {before: 20, after: 10},
        children: [new ImageRun({type: 'png', data: fs.readFileSync(ECON_DISPATCH), transformation: {width: 944, height: 407},
          altText: {name: '调度前后负荷与发电曲线', title: '调度前后负荷与发电曲线', description: '法兰克福训练情境一周，经济最优与100%本地无碳下无柔性与有柔性的负荷、风光发电、储能放电与市电。'}})],
      }),
      p(NOTE7D, {before: 10, size: 17}),

      economicHeading(H8),
      p(NOTE8A, {before: 30, size: 17}),
      globalTable,
      p(NOTE8B, {before: 20, size: 16, color: '595959'}),
      sub('训练（GPU主导型）规划阶段经济最优：各大洲与主要国家中位数'),
      regionTable,
      p(NOTE8C, {before: 40, size: 18}),
      p(NOTE8LIMIT, {before: 30, size: 16, color: '595959'}),

      economicHeading('全球柔性收益地图：规划阶段，经济最优'),
      new Paragraph({alignment: AlignmentType.CENTER, keepNext: true, spacing: {before: 20, after: 10},
        children: [new ImageRun({type: 'png', data: fs.readFileSync(GLOBAL_PLAN_MAP), transformation: {width: 900, height: 486},
          altText: {name: '全球柔性收益地图·规划阶段', title: '全球柔性收益地图·规划阶段', description: '训练、科研AI、科研计算、推理四类代表情境在规划阶段经济最优目标下的柔性节省百分比，2°区块着色。'}})],
      }),
      p(NOTE8MAP, {before: 10, size: 16, color: '595959'}),

      economicHeading('全球柔性收益地图：已建成，经济最优'),
      new Paragraph({alignment: AlignmentType.CENTER, keepNext: true, spacing: {before: 20, after: 10},
        children: [new ImageRun({type: 'png', data: fs.readFileSync(GLOBAL_FIXED_MAP), transformation: {width: 900, height: 486},
          altText: {name: '全球柔性收益地图·已建成', title: '全球柔性收益地图·已建成', description: '四类代表情境在风光储固定为无柔性最优配置时经济最优目标下的柔性节省百分比，2°区块着色。'}})],
      }),
      p(NOTE8MAP, {before: 10, size: 16, color: '595959'}),

      economicHeading('柔性收益是否超过3%：已建成，经济最优'),
      new Paragraph({alignment: AlignmentType.CENTER, keepNext: true, spacing: {before: 20, after: 10},
        children: [new ImageRun({type: 'png', data: fs.readFileSync(GLOBAL_THRESHOLD_MAP), transformation: {width: 900, height: 486},
          altText: {name: '柔性收益超过3%的区域', title: '柔性收益超过3%的区域', description: '已建成边界、经济最优下，训练、科研AI、科研计算、推理四类代表情境柔性收益是否超过3%的二分类地图，2°区块着色。'}})],
      }),
      p(NOTE8T, {before: 10, size: 17}),
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
    `## ${H7}`, '', NOTE7A, '', `### ${SUB7A}`, '', mdTable(ECON.plan_rows), '', NOTE7B1, '', `### ${SUB7B}`, '', mdTable(ECON.built_rows), '', NOTE7B2, '', NOTE7C, '',
    `### ${K_TITLE}`, '', `![${K_TITLE}](${FLEX_DIR}/task-peak-capacity.png)`, '', NOTE7K, '',
    '### 柔性收益与单位成本', '', `![柔性收益与单位成本](${ECON_DIR}/brief/value-summary.png)`, '', NOTE7METHOD, '', NOTE7E, '',
    '### 调度前后对比（法兰克福·GPU 主导型·大模型训练/开发）', '', `![调度前后](${ECON_DIR}/brief/dispatch-week.png)`, '', NOTE7D, '',
    `## ${H8}`, '', NOTE8A, '', mdTable(GLOBAL.table_rows), '', NOTE8B, '', '### 训练（GPU主导型）规划阶段经济最优：各大洲与主要国家中位数', '', mdTable(GLOBAL.region_rows), '', NOTE8C, '', NOTE8LIMIT, '',
    '### 全球柔性收益地图', '', `![规划阶段·经济最优](${GLOBAL_DIR}/figures/planning-economic.png)`, '', `![已建成·经济最优](${GLOBAL_DIR}/figures/fixed-economic.png)`, '', NOTE8MAP, '',
    '### 柔性收益是否超过3%：已建成，经济最优', '', `![超过3%](${GLOBAL_DIR}/figures/threshold-fixed-economic-3pct.png)`, '', NOTE8T, '',
  ].join('\n');
  if (!retainedSections) throw new Error('README 缺少需要保留的数字来源与说明、重新生成章节');
  const sourceUpdated = retainedSections.split('\n').map((line) => line.startsWith('- 第四章图与逐小时分任务汇总：') ? SOURCE4 : line.startsWith('- 第五章44情境、25组地点年份及原生时序：') ? SOURCE5 : line).join('\n');
  let tail = sourceUpdated.includes(SOURCE4) ? sourceUpdated
    : retainedSections.replace('## 重新生成', `${SOURCE4}\n\n## 重新生成`);
  if (!tail.includes(SOURCE5)) tail = tail.replace('## 重新生成', `${SOURCE5}\n\n## 重新生成`);
  if (!tail.includes(SOURCE6)) tail = tail.replace('## 重新生成', `${SOURCE6}\n\n## 重新生成`);
  tail = tail.split(String.fromCharCode(10)).filter((line) => !line.startsWith('- 第七章') || line === SOURCE7).join(String.fromCharCode(10));
  if (!tail.includes(SOURCE7)) tail = tail.replace('## 重新生成', `${SOURCE7}

## 重新生成`);
  if (!tail.includes(SOURCE8)) tail = tail.replace('## 重新生成', `${SOURCE8}

## 重新生成`);
  if (!tail.includes(SPIKE_SOURCE)) tail = tail.replace('## 重新生成', `${SPIKE_SOURCE}\n\n## 重新生成`);
  fs.writeFileSync(MD_OUT, md + '\n' + tail + '\n', 'utf8');
  console.log('wrote', MD_OUT);
}

Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log('wrote', OUT); });
