import itertools
import json
from pathlib import Path
from xml.sax.saxutils import escape

import fitz
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from compute_load.model import HARDWARE, sha256


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/scenario-load-methods-20261026"
TEMP = PROJECT / ".codex-temp/scenario-load-methods-20261026"
STEM = "算力中心_情境设计与负荷计算说明_20261026"
WIDTH = A4[0] - 36 * mm
SIZES = (1, 10, 100, 500)
MIXES = (("CPU 主导", "cpu_dominant", .75), ("GPU 主导", "gpu_dominant", .25), ("均衡", "balanced", .50))
DEMANDS = (
    ("科研超算", "research", "Lisa", "CPU 作业已核验；GPU 需求须补齐监测与卡数口径"),
    ("大模型开发", "llm_development", "Acme Kalos", "CPU/GPU 作业已核验；资源分母与功率映射保留代理性质"),
    ("生产 AI 服务", "production_ai", "ASI", "GPU 一天已核验；独立 CPU 池需求待补，完整时间段待扩展"))


def scenario_rows():
    rows = []
    for size, mix, demand in itertools.product(SIZES, MIXES, DEMANDS):
        name, mix_id, alpha = mix
        business, demand_id, source, readiness = demand
        cpu_budget, gpu_budget = size * alpha, size * (1 - alpha)
        cpu_nodes = int(cpu_budget * 1e6 // HARDWARE["cpu"]["full_w"])
        gpu_nodes = int(gpu_budget * 1e6 // HARDWARE["gpu"]["full_w"])
        rows.append({"scenario_id": f"{size}MW_{mix_id}_{demand_id}", "reference_it_mw": size,
            "center_mode": name, "cpu_capacity_fraction": alpha, "gpu_capacity_fraction": 1-alpha,
            "demand_mode": business, "source": source, "cpu_budget_mw": cpu_budget,
            "gpu_budget_mw": gpu_budget, "cpu_nodes": cpu_nodes, "cpu_cores": cpu_nodes * 128,
            "gpu_nodes": gpu_nodes, "reference_h100_count": gpu_nodes * 8,
            "installed_reference_it_mw": (cpu_nodes*HARDWARE["cpu"]["full_w"] + gpu_nodes*HARDWARE["gpu"]["full_w"])/1e6,
            "exogenous_input": "当地逐小时干球温度及相对湿度或露点",
            "data_readiness": readiness, "calculation_status": "design_only_not_run"})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    scenarios = scenario_rows()
    if len(scenarios) != 36 or len({row["scenario_id"] for row in scenarios}) != 36:
        raise ValueError("情境组合必须包含 36 个唯一编号")
    pd.DataFrame(scenarios).to_csv(OUT / "scenario-matrix.csv", index=False, encoding="utf-8-sig")
    (OUT / "scenario-matrix.json").write_text(json.dumps(scenarios, ensure_ascii=False, indent=2), encoding="utf-8")
    pdfmetrics.registerFont(TTFont("YaHei", "C:/Windows/Fonts/msyh.ttc"))
    pdfmetrics.registerFont(TTFont("YaHeiBold", "C:/Windows/Fonts/msyhbd.ttc"))
    pdfmetrics.registerFontFamily("YaHei", normal="YaHei", bold="YaHeiBold", italic="YaHei", boldItalic="YaHeiBold")
    body = ParagraphStyle("Body", fontName="YaHei", fontSize=10.2, leading=16.2, wordWrap="CJK", spaceAfter=8)
    small = ParagraphStyle("Small", parent=body, fontSize=8.6, leading=13, textColor=colors.HexColor("#444444"))
    heading = ParagraphStyle("Heading", parent=body, fontName="YaHeiBold", fontSize=16, leading=23, spaceAfter=12)
    title = ParagraphStyle("Title", parent=heading, fontSize=20, leading=29)
    sub = ParagraphStyle("Sub", parent=body, fontName="YaHeiBold", fontSize=11.1, leading=17, spaceBefore=5)
    cell = ParagraphStyle("Cell", parent=body, fontSize=8.6, leading=12.5, spaceAfter=0)
    head = ParagraphStyle("Head", parent=cell, fontName="YaHeiBold")
    equation = ParagraphStyle("Equation", parent=body, leftIndent=8, spaceBefore=2, spaceAfter=10)
    story, markdown = [], []

    def p(text, style=body):
        story.append(Paragraph(text, style))
        markdown.append(text.replace("<br/>", "\n\n"))

    def chapter(text, first=False):
        if not first:
            story.append(PageBreak())
        story.append(Paragraph(text, heading))
        markdown.extend(["", "## " + text, ""])

    def subsection(text):
        story.append(Paragraph(text, sub))
        markdown.extend(["", "### " + text, ""])

    def table(rows, widths):
        cells = [[Paragraph(escape(str(value)), head if i == 0 else cell) for value in row] for i, row in enumerate(rows)]
        block = Table(cells, colWidths=[WIDTH*w for w in widths], repeatRows=1, hAlign="LEFT")
        block.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EFF3")),
            ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#8C9AA3")),
            ("LINEBELOW", (0, 1), (-1, -1), .3, colors.HexColor("#D9DFE1")),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        story.extend([block, Spacer(1, 8)])
        markdown.extend(["| " + " | ".join(map(str, rows[0])) + " |", "| " + " | ".join(["---"]*len(rows[0])) + " |"])
        markdown.extend(["| " + " | ".join(map(str, row)) + " |" for row in rows[1:]])
        markdown.append("")

    markdown.append("# 算力中心情境设计与负荷计算说明\n")
    p("算力中心情境设计<br/>与负荷计算说明", title)
    p("规模 × 设备构成 × 需求导向 · 外生气象输入 · 2026 年 10 月 26 日", small)
    chapter("1 情境设计：每个网格、每年 36 组", first=True)
    table([["维度", "比较值", "定义"],
           ["IT 规模 S", "1、10、100、500 MW", "参考高负载 IT 预算，冷却和配套另计"],
           ["数据中心模式", "CPU 主导、GPU 主导、均衡", "按 CPU/GPU 设备池的参考 IT 容量划分"],
           ["需求导向", "科研超算、大模型开发、生产 AI 服务", "规定业务类型、需求序列及其数据来源"]], [.23, .39, .38])
    p("四档规模、三种设备构成和三种需求导向交叉形成 4×3×3=36 组设计情境。每个网格、每个年份复用这套编号，输入当地气象；各组合的数据准备状态另行记录。")
    subsection("设备构成的参考配比")
    table([["模式", "CPU 池占 IT 容量", "GPU 池占 IT 容量"],
           ["CPU 主导", "75%", "25%"], ["GPU 主导", "25%", "75%"], ["均衡", "50%", "50%"]], [.34, .33, .33])
    p("配比是研究比较点。资源占用随原任务序列变化，全年电量占比由模型计算。GPU 池按含宿主 CPU、内存等组件的整机计量；独立 CPU 池只包含另行运行的 CPU 业务。")
    p("例如：100 MW × GPU 主导 × 大模型开发，配置 25 MW CPU 池与 75 MW GPU 池，需求来自 Acme 开发作业，外部环境来自所选网格的逐小时温湿度。当前 36 组为实验设计，尚未完成其联合负荷计算。")

    chapter("2 需求从哪里来，如何形成小时曲线")
    table([["需求导向", "独立 CPU 池", "GPU 整机池", "当前准备状态"],
           ["科研超算", "Lisa CPU 分区作业", "Lisa GPU 分区及节点监测", "CPU 已累计；GPU 须补齐卡数与监测口径"],
           ["大模型开发", "Acme 中 gpu_num=0 的 CPU 作业", "Acme 预训练、微调、评估与调试作业", "两类作业已核验；容量和功率映射须保留代理说明"],
           ["生产 AI 服务", "同类业务的独立 CPU 记录，需补充", "ASI 的训练、在线/离线推理与开发", "GPU 核验一天；GPU 宿主 CPU 不充作独立 CPU 池"]], [.18, .27, .29, .26])
    p("同一需求导向尽量保留同源、同时段的 CPU/GPU 关系。跨来源组合须注明业务含义和相对起点；缺少的业务记录保留为缺口。科研 CPU 加生产 GPU 可作为明确命名的组合业务情境，标为研究构造并分别注明来源。")
    subsection("按实际运行区间累计资源")
    p("对作业 i，资源数量为 n_i，运行区间为 [s_i,e_i)。时间段 h 内的资源小时为：")
    p("W<sub>h</sub> = Σ<sub>i</sub> n<sub>i</sub> × overlap_seconds([s<sub>i</sub>,e<sub>i</sub>),h) / 3600", equation)
    p("W_h 分别以核心小时或 GPU 小时计；除以时间段小时数得到平均占用 D_h。秒级作业可累计为 5 分钟和小时；ASI 保留原生小时粒度。排队不计入执行用量，短作业不向上取整到整小时。")
    p("资源份额 u_h=D_h/K_h，其中 K_h 为同口径的源资源容量。优先采用有依据的库存；仅有观测并发峰值时，标为规模参考。ASI 用归属 GPU 小时形成 D_h，申请数量单独保存。分配或归属数量映射到节点功率时保留代理性质，超出库存的需求须核对原因。")
    p("各来源保留原有曲线形状和任务构成，平均占用自然形成。规模放大采用同形状容量外推；多集群合成时另行保留共时关系，不预设扩大规模后必然更平稳。", small)

    chapter("3 所有情境共用的功率计算步骤")
    subsection("① 将 IT 预算分成两个设备池")
    p("令 CPU 容量份额为 α，CPU/GPU 预算分别为 αS 和 (1−α)S。按参考整机高负载功率取整为节点：")
    p("N<sub>C</sub>=floor(10<sup>6</sup>αS / 546)；N<sub>G</sub>=floor[10<sup>6</sup>(1−α)S / 8430]", equation)
    p("CPU 参考节点为 128 核心，空闲/高负载功率为 120/546 W；GPU 为 8×H100 整机，空闲/联合高负载为 1860/8430 W。卡数为 8N_G，CPU 池核心数为 128N_C。[4,5]")
    subsection("② 将两类资源曲线分别转换成 IT 功率")
    p("按各自实测功率点线性插值得到单节点功率 p_C(u_C,h)、p_G(u_G,h)，单位为 W。合计 IT 电功率为：")
    p("P<sub>IT,h</sub>=[N<sub>C</sub>p<sub>C</sub>(u<sub>C,h</sub>)+N<sub>G</sub>p<sub>G</sub>(u<sub>G,h</sub>)] / 10<sup>6</sup>　(MW)", equation)
    p("现有参考算法将小时平均资源份额映射到整机功率，包含持续通电的空闲功耗。小时内部的短时尖峰会被平均，不能据此恢复分钟峰值；节点功率非线性时，小时平均映射保留近似性质。", small)
    subsection("③ 合并散热需求，计算共享冷却")
    p("设备散热 Q_h≈P_IT,h，单位为 MW 热。取整后的安装参考容量 C=(546N_C+8430N_G)/10<sup>6</sup>，令 q_h=P_IT,h/C。冷却和配套采用同一参考模型：")
    p("P<sub>j,h</sub>=CΣ<sub>k</sub>β<sub>k,j</sub>X<sub>k</sub>(q<sub>h</sub>,T<sub>h</sub>,Td<sub>h</sub>)", equation)
    p("X 包括常数、q、q²、温度及露点分段项和低温响应，β 为 ESIF 的 8×4 标定系数。[6] j 为室外冷却、HVAC、泵和其他配套，前面三个分项相加为 P_cool。设施总功率 P_fac=P_IT+P_cool+P_other；共享冷却按总热负荷计算一次，散热单独保存。")
    p("各情境统一参考冷却配置，便于比较天气差异。目标设施的冷却技术、供水温度及规模效率需要对应标定；GPU 整机宿主功耗已计入，公共网络与存储按其实际边界另列。", small)

    chapter("4 外生输入、全年累计与结果")
    subsection("外生输入：当地逐小时温湿度")
    table([["输入", "单位与要求", "进入计算的位置"],
           ["网格、年份、UTC 小时", "平年 8760、闰年 8784 小时；无重复或缺口", "连接需求重放与当地气象"],
           ["干球温度 T", "℃；开尔文输入显式减 273.15", "冷却、配套响应"],
           ["相对湿度 RH 或露点 Td", "0<RH≤100%；露点以 ℃ 计", "RH 转换露点后进入冷却响应"]], [.29, .42, .29])
    p("同一设备与需求情境放到不同网格时，温湿度进入冷却和配套计算；IT 资源路径按任务输入保持。缺失天气须补齐并登记处理依据。超出模型标定范围的小时保留外推标记。[6]")
    subsection("从观测窗口构造成全年")
    p("不满一年的任务序列按连续周/月重放，登记样本、日历映射、重复次数、相对起点和年末边界；保留跨小时、跨日期的作业。Lisa 时区需核实，ASI 没有实际日历；相对时钟的重放属于研究设定，缺少的季节变化保持未知。")
    p("年度电量 E=Σ_hP_h×1 小时，单位为 MWh；分别累计 IT、冷却、其他与设施总电量。年度 PUE=设施全年电量/IT 全年电量，峰值取最高小时平均功率。另报核心小时、GPU 小时和完成服务量。")
    subsection("每个情境应保存的结果")
    p("输出情境编号、网格、UTC 小时、两类资源份额、节点与卡数、IT 功率、散热、冷却、其他配套、设施总功率及外推标记，并形成月度/年度电量与峰值汇总。每条结果同时记录源文件、参数版本和实际使用的任务/气象时段。")

    chapter("5 柔性权限与控制变量")
    p("柔性权限作为独立的控制设计。基础情境固定为前面的 36 组；同一基础情境下，可以比较允许哪些控制操作。允许的操作范围由控制设置规定，启动时刻、运行状态和功率设定由求解过程选择。")
    table([["控制设置", "求解变量或执行路径", "所需服务条件"],
           ["固定执行", "保留历史启动和执行路径", "作为同一业务与设备配置的参照"],
           ["允许延后启动", "每项作业的启动时刻 s_i", "数据就绪 r_i、完成期限 d_i、运行时长 p_i"],
           ["允许暂停与恢复", "运行状态 x_i,t、暂停与恢复时刻", "检查点、最小连续运行时间、恢复时长及额外能耗"],
           ["允许调整功率或资源", "功率上限、核心数或卡数", "性能关系、最小并行规模、内存与服务质量"]], [.24, .34, .42])
    subsection("控制范围和可行性分别记录")
    p("任务类别用于确定研究中的候选范围。可延后上限、允许暂停的任务、检查点间隔和恢复开销单独登记；缺少实测权限时，明确写为控制设计假设。历史排队时长、训练标签或 LP 优先级不能单独给出新增延后额度。")
    p("固定资源、固定执行速度且连续运行时，作业满足 r_i≤s_i 且 s_i+p_i≤d_i，同时满足 CPU、GPU、内存和并行度约束。允许暂停时仍须完成所需运行量，并计入恢复开销；改变功率或资源时，需用相应性能关系约束完成工作量。")
    subsection("控制后重新计算负荷")
    p("每次求得新的执行路径，重新累计资源占用，并按第 3 节计算 IT、散热和冷却，再形成小时及年度总负荷。不同控制设置保持相同任务、完成要求、设备和气象条件，分别记录服务完成情况及负荷变化。")
    p("当前负荷模块接收给定资源路径；本章定义控制层的输入与约束，尚未完成这些控制变量与联合调度求解器的接入。", small)

    chapter("6 数据来源与计算准备状态")
    table([["来源", "提供的依据", "本工作区已核验范围"],
           ["[1] SURF Lisa v3", "Slurm 作业、硬件表及 30 秒监测", "1596963 条作业；CPU 309 个完整日；完整监测未用于重标定"],
           ["[2] AcmeTrace Kalos", "CPU/GPU 作业时刻、资源数和开发任务类型", "62413 条作业，90 个完整日；3 条 GPU 缺少结束时间"],
           ["[3] Alibaba ASI 2026", "小时任务类型、优先级、归属 GPU 小时与库存", "相对日 109 的 24 小时；来源发布 185 天"],
           ["[4,5] 整机功率", "SPECpower CPU 与 HGX H100 实测锚点", "CPU 11 点；GPU 联合整机曲线"],
           ["[6] 气象与冷却", "用户当地天气/既有 ERA5 网格；ESIF 参考响应", "5447 配对小时；留出 672 小时冷却 RMSE 17.754 kW"]], [.24, .34, .42])
    p("现有八条 24 小时功率示例验证了 CPU/GPU 纯设备池的接口。36 组设计情境需要每个业务模式完整的两类需求、清楚的源容量和年度重放规则；新增混合中心尚未求解。当前模块按固定总规模处理纯池，CPU/GPU 各自容量的计算与共享冷却需按第 3 页方法接入。")
    references = [
        ("[1] Lisa 数据与字段；对应 ICPADS 2024 原始研究。", "https://zenodo.org/records/13685426"),
        ("[2] AcmeTrace 官方数据；对应 NSDI 2024 开发工作负载研究。", "https://github.com/InternLM/AcmeTrace"),
        ("[3] ASI 官方字段；对应 OSDI 2026 生产集群研究。", "https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/docs/schema.md"),
        ("[4] SPECpower：HPE DL385 Gen10 Plus / EPYC 7763。", "https://ftp.spec.org/power_ssj2008/results/res2021q1/power_ssj2008-20210221-01063.html"),
        ("[5] Latif 等，HGX H100 实测整机功率，Table I。", "https://arxiv.org/html/2412.08602v1"),
        ("[6] 通用负荷模块：温湿度接口、ESIF 系数与全球网格接入。", "https://github.com/TuzzyT-Laiii/33/blob/main/dc-flexibility-project/docs/compute-load-module.md")]
    for label, url in references:
        p(f'<a href="{url}" color="#1C5675">{escape(label)}</a>', small)
    p("随文提供 scenario-matrix.csv/json，列明 36 组的容量配比、节点数、参考卡数、来源与准备状态；负荷结果按实际计算另行保存。已有详细分析保留在 20261026-v2 报告中。", small)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("YaHei", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawRightString(A4[0]-18*mm, 11*mm, str(doc.page))
        canvas.restoreState()

    pdf = OUT / f"{STEM}.pdf"
    document = SimpleDocTemplate(str(pdf), pagesize=A4, leftMargin=18*mm, rightMargin=18*mm,
        topMargin=16*mm, bottomMargin=18*mm, title="算力中心情境设计与负荷计算说明", author="数据中心研究工作区")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    (OUT / f"{STEM}.md").write_text("\n\n".join(markdown), encoding="utf-8")
    with fitz.open(pdf) as rendered:
        for i, page in enumerate(rendered):
            page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35)).save(TEMP / f"page-{i+1:02d}.png")
        pages = len(rendered)
        (TEMP / "document-text.txt").write_text("\n\n".join(page.get_text() for page in rendered), encoding="utf-8")
    manifest = {"document": pdf.name, "pages": pages, "scenarios": 36,
        "calculation_status": "design_only_not_run", "exogenous_input": "hourly_local_temperature_and_humidity",
        "flexibility_role": "separate_control_design_not_scenario_axis",
        "source_code_sha256": sha256(Path(__file__)),
        "parameter_sha256": {path.name: sha256(path) for path in sorted((PROJECT / "analysis/compute_load/assets").glob("*.csv"))},
        "cooling_sha256": sha256(PROJECT / "analysis/compute_load/assets/cooling-calibration.json"),
        "files_sha256": {path.name: sha256(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "manifest.json"}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pages": pages, "scenarios": 36, "document_sha256": sha256(pdf)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
