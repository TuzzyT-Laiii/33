import itertools
import json
from pathlib import Path
from xml.sax.saxutils import escape

import fitz
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from compute_load.model import HARDWARE, sha256


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/scenario-load-methods-20261026-v2"
TEMP = PROJECT / ".codex-temp/scenario-load-methods-20261026-v2"
STEM = "算力中心_实际配置与8760小时负荷计算说明_20261026_v2"
WIDTH = A4[0] - 36*mm
SIZES = (1, 10, 100, 500)
MODES = (("CPU 计算型", "cpu_only", 1.), ("GPU 加速型", "gpu_accelerated", 0.),
         ("混合计算型", "heterogeneous", None))
DEMANDS = (("科研超算", "research"), ("大模型开发", "llm_development"), ("生产 AI 服务", "production_ai"))
SOURCES = [
    ("[1] ARCHER2 官方硬件：主计算系统与独立 GPU 开发平台。", "https://www.archer2.ac.uk/about/hardware.html"),
    ("[2] OLCF Frontier：每个计算节点的宿主 CPU 与 MI250X 配置。", "https://docs.olcf.ornl.gov/systems/frontier_user_guide.html"),
    ("[3] NERSC Perlmutter：CPU-only 与 GPU-accelerated 分区清单。", "https://docs.nersc.gov/systems/perlmutter/architecture/"),
    ("[4] Chu 等，Lisa 原始研究，Table I、V；ICPADS 2024。", "https://arxiv.org/html/2409.08949v1"),
    ("[5] Lisa v3 原始作业、硬件与监测数据。", "https://zenodo.org/records/13685426"),
    ("[6] AcmeTrace 官方数据；Hu 等 NSDI 2024 开发工作负载研究。", "https://github.com/InternLM/AcmeTrace"),
    ("[7] Alibaba ASI 2026 官方小时字段和类型定义。", "https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/docs/schema.md"),
    ("[8] SPECpower：HPE DL385 / 双路 EPYC 7763，整机实测。", "https://www.spec.org/power_ssj2008/results/res2021q1/power_ssj2008-20210221-01063.html"),
    ("[9] Latif 等，HGX H100 整机功率，Table I。", "https://arxiv.org/html/2412.08602v1"),
    ("[10] 本项目通用负荷模块：ERA5 接入、ESIF 标定与适用范围。", "https://github.com/TuzzyT-Laiii/33/blob/main/dc-flexibility-project/docs/compute-load-module.md")]


def scenario_rows():
    readiness = {
        ("cpu_only", "research"): ("lisa_cpu", "computed_annual_replay_proxy"),
        ("cpu_only", "llm_development"): ("acme_cpu", "computed_development_cpu_diagnostic"),
        ("cpu_only", "production_ai"): (None, "missing_independent_cpu_demand"),
        ("gpu_accelerated", "research"): (None, "missing_gpu_allocation_and_monitoring"),
        ("gpu_accelerated", "llm_development"): ("acme_gpu", "computed_annual_replay_proxy"),
        ("gpu_accelerated", "production_ai"): ("asi_gpu", "computed_one_day_repetition_demo")}
    rows = []
    for size, mode, demand in itertools.product(SIZES, MODES, DEMANDS):
        name, mode_id, alpha = mode
        business, demand_id = demand
        source, status = readiness.get((mode_id, demand_id), (None, "requires_inventory_matched_node_power_and_joint_demand"))
        nc = int(size*alpha*1e6 // HARDWARE["cpu"]["full_w"]) if alpha is not None else None
        ng = int(size*(1-alpha)*1e6 // HARDWARE["gpu"]["full_w"]) if alpha is not None else None
        rows.append({"scenario_id": f"{size}MW_{mode_id}_{demand_id}", "reference_it_mw": size,
            "center_mode": name, "demand_mode": business, "cpu_pool_capacity_fraction": alpha,
            "gpu_node_pool_capacity_fraction": 1-alpha if alpha is not None else None,
            "cpu_nodes": nc, "cpu_pool_cores": nc*128 if nc is not None else None,
            "gpu_nodes": ng, "reference_h100_count": ng*8 if ng is not None else None,
            "source": source, "status": status,
            "capacity_basis": "reference_compute_node_pools_excluding_shared_network_storage",
            "mixed_fraction_basis": "N_C*p_C_ref/(N_C*p_C_ref+N_G*p_G_ref); matched complete-node power required" if alpha is None else "pure_compute_pool_architecture_boundary",
            "year": 2025 if source else None, "hours": 8760 if source else None,
            "grid_id": "g05_r100_c592" if source else None,
            "curve_file": f"data/processed/task-annual-loads-20261026-v2/g05_r100_c592-2025/{source}-{size}MW.parquet" if source else None})
    return rows


def figures(curve_dir):
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.unicode_minus": False, "legend.fontsize": 8})
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.9), constrained_layout=True)
    for column, (source, label, color) in enumerate((("lisa_cpu", "科研 CPU", "#2166AC"), ("acme_gpu", "开发 GPU", "#C75B12"))):
        curve = pd.read_parquet(curve_dir / f"{source}-100MW.parquet")
        for row, count in ((0, 8760), (1, 168)):
            ax = axes[row, column]
            for field, text, shade, width in (("facility_mw", "设施总功率", color, .6), ("it_mw", "IT 功率", "#777777", .55)):
                ax.plot(np.arange(count), curve[field].iloc[:count], label=text, color=shade, linewidth=width)
            ax.set(xlabel="年内小时" if row == 0 else "第一周小时", ylabel="小时平均功率（MW）", ylim=(0, 120))
            ax.set_title(f"{'abcd'[row*2+column]}  {label} · {'全年' if row == 0 else '第一周'}", loc="left", fontsize=10)
            ax.grid(axis="y", alpha=.18)
            if row == 0:
                ax.legend(frameon=False, loc="lower left")
    fig.savefig(OUT / "annual-hourly-curves.png", dpi=300, facecolor="white")
    fig.savefig(OUT / "annual-hourly-curves.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    annual_manifest = json.loads((OUT / "annualization-manifest.json").read_text(encoding="utf-8"))
    if (annual_manifest["grid_id"], annual_manifest["year"]) != ("g05_r100_c592", 2025):
        raise ValueError("本报告的实例输入须为指定网格的 2025 年结果")
    annual = pd.read_csv(OUT / "annual-load-summary.csv")
    phase = pd.read_csv(OUT / "annual-phase-sensitivity.csv")
    curve_dir = PROJECT / annual_manifest["curves_directory"]
    if len(annual) != 16 or not annual.hours.eq(8760).all():
        raise ValueError("报告需要 16 条已计算的 8760 小时曲线")
    scenarios = scenario_rows()
    pd.DataFrame(scenarios).to_csv(OUT / "scenario-matrix.csv", index=False, encoding="utf-8-sig")
    (OUT / "scenario-matrix.json").write_text(json.dumps(scenarios, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    evidence = {"checked_date": "2026-10-26", "scope": "compute_node_pools", "examples": [
        {"system": "ARCHER2", "cpu_only_main_nodes": 5860, "separate_gpu_development_nodes": 4, "url": SOURCES[0][1]},
        {"system": "Frontier", "cpu_per_gpu_node": 1, "mi250x_packages_per_node": 4, "gcd_per_node": 8, "url": SOURCES[1][1]},
        {"system": "Perlmutter", "cpu_only_nodes": 3072, "gpu_nodes": 1792, "a100_per_gpu_node": 4,
            "cpu_node_fraction": 3072/4864, "gpu_node_fraction": 1792/4864, "it_power_fraction": None, "url": SOURCES[2][1]},
        {"system": "Lisa_paper_snapshot", "cpu_only_nodes": 287, "gpu_nodes": 51, "gpu_job_energy_fraction": .3868,
            "energy_fraction_scope": "paper_Table_V_job_attributed_energy_not_installed_capacity", "url": SOURCES[3][1]}]}
    (OUT / "hardware-evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    figures(curve_dir)
    pdfmetrics.registerFont(TTFont("YaHei", "C:/Windows/Fonts/msyh.ttc"))
    pdfmetrics.registerFont(TTFont("YaHeiBold", "C:/Windows/Fonts/msyhbd.ttc"))
    pdfmetrics.registerFontFamily("YaHei", normal="YaHei", bold="YaHeiBold", italic="YaHei", boldItalic="YaHeiBold")
    body = ParagraphStyle("Body", fontName="YaHei", fontSize=10.1, leading=16, wordWrap="CJK", spaceAfter=8)
    small = ParagraphStyle("Small", parent=body, fontSize=8.5, leading=12.7)
    heading = ParagraphStyle("Heading", parent=body, fontName="YaHeiBold", fontSize=16, leading=23, spaceAfter=12)
    sub = ParagraphStyle("Sub", parent=body, fontName="YaHeiBold", fontSize=11, leading=17, spaceBefore=5)
    cell = ParagraphStyle("Cell", parent=body, fontSize=8.4, leading=12.4, spaceAfter=0)
    head = ParagraphStyle("Head", parent=cell, fontName="YaHeiBold")
    story, markdown = [], ["# 算力中心实际配置与 8760 小时负荷计算说明\n"]

    def p(text, style=body):
        story.append(Paragraph(text, style))
        markdown.append(text.replace("<br/>", "\n\n"))

    def chapter(text, first=False):
        if not first:
            story.append(PageBreak())
        p(text, heading)
        markdown[-1] = "## " + text

    def section(text):
        p(text, sub)
        markdown[-1] = "### " + text

    def table(rows, widths):
        cells = [[Paragraph(escape(str(v)), head if i == 0 else cell) for v in row] for i, row in enumerate(rows)]
        block = Table(cells, colWidths=[WIDTH*w for w in widths], repeatRows=1, hAlign="LEFT")
        block.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E9EFF3")),
            ("LINEBELOW", (0,0), (-1,0), .6, colors.HexColor("#8C9AA3")),
            ("LINEBELOW", (0,1), (-1,-1), .3, colors.HexColor("#D9DFE1")),
            ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5)]))
        story.extend([block, Spacer(1,8)])
        markdown.extend(["| " + " | ".join(map(str, rows[0])) + " |", "| " + " | ".join(["---"]*len(rows[0])) + " |"])
        markdown.extend("| " + " | ".join(map(str, row)) + " |" for row in rows[1:])

    p("算力中心实际配置<br/>与 8760 小时负荷计算", ParagraphStyle("Title", parent=heading, fontSize=20, leading=29))
    p("规模 × 设备模式 × 需求导向 · 逐小时气象 · 2026 年 10 月 26 日", small)
    chapter("1 设备构成采用什么依据", True)
    table([["实际系统", "官方或原始研究的配置", "支持的设备边界"],
        ["ARCHER2 [1]", "主计算系统 5860 个 CPU 节点，每节点双路 64 核心 CPU；另有 4 个 GPU 开发节点", "主计算分区可采用纯 CPU 池；GPU 开发平台单列"],
        ["Frontier [2]", "每计算节点 1 颗宿主 CPU + 4 个 MI250X 封装（8 个 GCD）", "计算节点均为 GPU 加速节点，宿主 CPU 计入整机"],
        ["Perlmutter [3]", "3072 个 CPU-only 节点 + 1792 个 GPU 节点；每 GPU 节点 1 颗 CPU + 4 张 A100", "混合计算由两种节点清单确定"],
        ["Lisa 论文快照 [4]", "287 个 CPU-only 节点 + 51 个 GPU 节点，GPU 硬件异构", "同一中心可同时运行 CPU 与 GPU 分区"]], [.20,.49,.31])
    p("CPU 超算可以没有生产 GPU 计算节点；也可能另有小型开发平台。GPU 加速型包含必要的宿主 CPU、内存和机内配套。两类中心按计算节点类型区分，GPU 节点内的 CPU 不再另建独立 CPU 池。")
    p("Perlmutter 的 CPU/GPU 节点数量占比为 63.16%/36.84%。对应 IT 功率份额须另用匹配硬件的整机功率计算。Lisa 论文中 GPU 作业能量占比为 38.68%，同样不能用作安装容量份额。[3,4]", small)
    p("这些实例支持设备架构的分类；它们尚未组成可用于估计行业平均配比的代表性样本。", small)

    chapter("2 情境设计与规模参数")
    p("基本设计保留 4 档 IT 规模 × 3 类设备模式 × 3 类需求导向，共 36 个设计单元。需求导向为科研超算、大模型开发、生产 AI 服务；每个单元分别登记数据是否足够。")
    table([["设备模式", "独立 CPU 池 / GPU 整机池", "配比依据"],
        ["CPU 计算型", "100% / 0%", "纯 CPU 主计算分区的架构边界 [1]"],
        ["GPU 加速型", "0% / 100%", "全 GPU 加速计算节点的架构边界；包含宿主 CPU [2]"],
        ["混合计算型", "由实际节点清单和整机功率推导", "α=N_C p_C,ref / (N_C p_C,ref+N_G p_G,ref)；未取得匹配功率时留空"]], [.22,.34,.44])
    p("S 表示参考计算节点高负载 IT 预算，冷却另计。公共网络、独立存储和管理设备须按实际清单另外核算。本说明的 1—500 MW 为计算节点规模；参考整机用于跨规模比较，不代表上述超算原设备复刻。", small)
    table([["IT 预算", "纯 CPU：节点 / 核心", "纯 GPU：整机 / H100 参考卡数"], *[
        [f"{s} MW", f"{int(s*1e6//546):,} / {int(s*1e6//546)*128:,}", f"{int(s*1e6//8430):,} / {int(s*1e6//8430)*8:,}"] for s in SIZES]], [.18,.40,.42])
    table([["需求导向", "CPU 计算型", "GPU 加速型", "混合计算型"],
        ["科研超算", "Lisa 全年重放", "卡数及监测待补", "节点功率与双池需求待补"],
        ["大模型开发", "Acme CPU 诊断样本", "Acme GPU 全年重放", "独立 CPU 库存及双池配置待补"],
        ["生产 AI 服务", "独立 CPU 需求待补", "ASI 单日重复演示", "独立 CPU 需求及设备清单待补"]], [.19,.26,.27,.28])
    p("现有数据形成 16 条全年代理/演示曲线；其余设计单元保留具体缺口。‘均衡’仅可用作业务描述；数值配比由设备清单决定。", small)

    chapter("3 原始任务如何形成小时需求")
    section("运行区间积分")
    p("作业 i 的执行区间为 [s_i,e_i)，资源数为 n_i。每小时的资源小时为 W_h=Σ_i n_i×overlap_seconds([s_i,e_i),h)/3600。小时长度为 1 小时，因此 W_h 的数值等于小时平均分配核心数或 GPU 数量。")
    p("短作业保留实际运行秒数，排队单列。先对完整原始执行区间积分，再选取完整日期，保留窗口内跨小时和跨日执行量；缺失结束时间的 Acme 三条 GPU 记录按既有核验规则排除。")
    section("当前实际采用的形状与分母")
    source_rows = [["需求来源", "连续完整日", "资源份额 u_h 的构造", "使用范围"]]
    labels = {"lisa_cpu":"Lisa CPU", "acme_cpu":"Acme CPU", "acme_gpu":"Acme GPU", "asi_gpu":"ASI GPU"}
    for key, label in labels.items():
        meta = annual_manifest["source_profiles"][key]
        denominator = meta["reference_resource_count"]
        basis = f"分配核心数 / {denominator:,.3f}（最高小时）" if key == "lisa_cpu" else (
            f"分配数量 / {denominator:,.0f}（瞬时并发峰值）" if key.startswith("acme") else "归属 GPU 小时 / 同小时 GPU 库存")
        scope = {"lisa_cpu":"科研形状代理", "acme_cpu":"开发 CPU 诊断，低占用", "acme_gpu":"开发 GPU 形状代理", "asi_gpu":"一天演示，异构 GPU"}[key]
        source_rows.append([label, str(meta["days"]), basis, scope])
    table(source_rows, [.18,.16,.42,.24])
    p("这里的分配/归属资源份额是功率映射的输入代理。Lisa 的最高小时归一化、Acme 的观测峰值归一化均不识别完整设施库存；由此得到的 u_h 不等同于芯片实测利用率。ASI 异构卡数也尚未标定成 H100 吞吐等价。")
    p("Acme CPU 来自开发集群的无 GPU 作业，少量长作业造成明显尖峰；该序列保留为来源诊断。不同来源分别生成曲线，保留各自变化和任务结构。", small)

    chapter("4 从观测窗口扩展到 8760 小时")
    section("明确的重放规则")
    p("① 选取连续的完整源日期，检查小时连续、无重复、资源非负。② 保留原顺序、逐小时幅度和尖峰。③ 将完整观测窗口循环映射到目标 UTC 年；末尾按需要截取，保留每小时源索引。④ 使用目标网格全年真实天气逐小时重新计算冷却。")
    p("设源窗口有 L 小时、目标有 H 小时，基准相位 φ=0：<br/>j(h)=(h+φ) mod L；u_year,h=u_source,j(h)，h=0,…,H−1。")
    table([["输入窗口", "L", "2025 年基准重放结果"],
        ["Lisa：309 日", "7416 小时", "完整 1 次 + 前 1344 小时（56 日）"],
        ["Acme：90 日，CPU/GPU 共时", "2160 小时", "完整 4 次 + 前 120 小时（5 日）"],
        ["ASI：相对日 109", "24 小时", "重复 365 次，仅作为方法演示"]], [.27,.21,.52])
    p("2025 年 H=8760；闰年按真实 UTC 日历使用 8784。源午夜对应目标 UTC 午夜是显式相位设定。Lisa 原时区与 ASI 实际日期未识别；这里采用相对时钟，不据此恢复原设施与当地气象的历史同步。")
    section("边界、季节性与敏感性")
    p("循环连接处保存 replay_wrap 标记和首尾资源跳变量，不平滑或填造新需求。完整窗口内的自相关和尖峰被保留，连接处会产生人工周期。样本外季节变化、节假日、需求增长及目标日历的星期效应保持未知。")
    p("实际检查：Lisa 首尾连接的资源份额跳变为 0.473，源窗口内逐小时变化的 95% 分位为 0.126；基准年在第 7416 小时连接处单独标记。该连接变化属于窗口重放边界。", small)
    p("以全部整数日相位 φ=0,24,…,L−24 重复计算 100 MW 年曲线：Lisa 309 组、Acme CPU/GPU 各 90 组、ASI 1 组，共 490 组。报告全年电量与峰值范围；该范围描述需求与气象配对及截尾影响，不作为统计置信区间。", small)
    p("本次重放用于固定路径能量分析。柔性优化须保留原任务、到达、依赖和跨边界执行状态；不能仅复制小时聚合量后据此判定任务可移动。", small)

    chapter("5 IT、散热与当地温湿度的计算")
    section("设备取整与整机功率")
    p("纯 CPU：N_C=floor(S×10<sup>6</sup>/546)，每节点 128 核心；纯 GPU：N_G=floor(S×10<sup>6</sup>/8430)，每节点 8 张 H100。参考 CPU 的空闲/高负载功率为 120/546 W；GPU 整机为 1860/8430 W。[8,9]")
    p("CPU 采用 11 个实测功率锚点，GPU 采用包含宿主的整机锚点插值。P_IT,h=[N_C p_C(u_C,h)+N_G p_G(u_G,h)]/10<sup>6</sup>，单位 MW。实际计算缺失的设备池取零；混合池须先获得对应设备参数。")
    p("规模外扩保持同一来源的归一化需求形状，扩大节点数量；不另设平均忙碌比例。节点取整后实际安装参考容量 C 可能略小于 S。同一硬件与资源路径下，功率和电量随 C 近似线性；规模本身不产生新的业务多样性。")
    section("热负荷与冷却响应")
    p("Q_heat,h≈P_IT,h（MW 热），q_h=P_IT,h/C。外生输入为当地逐小时干球温度 T 与相对湿度 RH 或露点 Td；RH 经 PsychroLib 换算为露点。[10]")
    p("P_j,h=CΣ_kβ_k,j X_k(q_h,T_h,Td_h)。X 为常数、q、q²、两个干球温度暖侧项、两个露点暖侧项及低温项；β 使用现有 ESIF 8×4 系数。室外冷却、HVAC 和泵相加为 P_cool；其他配套为 P_other。")
    p("P_fac,h=P_IT,h+P_cool,h+P_other,h。混合中心按总热量计算共享冷却一次。年度电量 E=Σ_hP_h×1 小时，单位 MWh；年度 PUE=ΣP_fac/ΣP_IT。散热、核心小时及 GPU 小时分别累计。")
    p("参考设备全年资源量：CPU 核心小时=128N_CΣ_hu_C,h；GPU 小时=8N_GΣ_hu_G,h。它们记录目标参考设备的资源占用代理，跨代设备的实际完成工作量另需性能标定。", small)
    p("ESIF 参考响应来自 5447 配对小时，留出 672 小时冷却 RMSE 为 17.754 kW。它提供统一冷却比较条件；具体液冷技术和 500 MW 工程效率尚未由这组数据验证。公共网络和独立存储的 IT 电耗需另外加入，不能由冷却配套项代替。", small)

    chapter("6 已处理的全年结果与适用范围")
    p("已使用全球网格 g05_r100_c592 的中心 ERA5 节点（原生索引 289905）及 2025 年逐小时干球/露点，生成 4 条需求路径 × 4 档规模 =16 条曲线，共 140160 条小时负荷记录。另保存 4 份源小时映射。")
    rows = [["100 MW 参考规模", "IT 电量\nGWh", "冷却电量\nGWh", "设施电量\nGWh", "设施峰值\nMW"]]
    for key in labels:
        row = annual.loc[annual.source.eq(key) & annual.reference_it_capacity_mw.eq(100)].iloc[0]
        rows.append([labels[key], f"{row.annual_it_mwh/1000:.2f}", f"{row.annual_cooling_mwh/1000:.2f}",
            f"{row.annual_facility_mwh/1000:.2f}", f"{row.peak_facility_mw:.2f}"])
    table(rows, [.28,.17,.18,.19,.18])
    p("上述为归一化需求、参考整机及参考冷却耦合后的全年构造结果。设施电量包含另计的其他配套；100 MW 其他配套约 1.17 GWh。各来源分母和业务范围不同，本表不能直接用于比较其真实算力能效。", small)
    rows = [["100 MW 全部日相位", "设施年电量范围（GWh）", "最高小时功率范围（MW）"]]
    for key in labels:
        group = phase.loc[phase.source.eq(key)]
        rows.append([f"{labels[key]}（{len(group)} 组）", f"{group.annual_facility_mwh.min()/1000:.2f}—{group.annual_facility_mwh.max()/1000:.2f}",
            f"{group.peak_facility_mw.min():.2f}—{group.peak_facility_mw.max():.2f}"])
    table(rows, [.31,.36,.33])
    section("逐小时外推标记必须随结果保留")
    p("该网格有 1903/8760 小时天气超出 ESIF 标定范围。IT 热负荷比例超范围小时：Lisa CPU 为 3，Acme CPU 为 8748，Acme GPU 为 353，ASI GPU 为 0。Acme CPU 的年曲线主要受持续空闲功率支配，只作为开发样本诊断。")
    p("ASI 的全年构造重复同一天，无法由此估计生产服务的跨日波动和季节规律。全部曲线均包含跨数据集功率映射的不确定性；此处没有把相位范围解释为完整预测误差。所有 1/10/100/500 MW 统计见 annual-load-summary.csv。", small)

    chapter("7 8760 小时曲线实际呈现的形状")
    story.append(Image(str(OUT / "annual-hourly-curves.png"), width=WIDTH, height=WIDTH*5.9/8))
    markdown.append("![全年与第一周小时曲线](annual-hourly-curves.png)")
    p("图：100 MW 参考计算节点规模。a、b 展示全部 8760 个小时；c、d 放大第一周 168 小时。彩色为设施总功率，灰色为 IT 功率；均保留小时平均值，未进行平滑。科研 CPU 使用 Lisa 309 日窗口，开发 GPU 使用 Acme 90 日窗口。", small)
    p("同一来源在不同规模下沿用相同资源形状，纵向随实际节点容量变化。循环窗口会使 IT 变化重复；设施总负荷同时受到全年温湿度影响。短样本扩展后的重复周期是构造方法带来的特征。")
    p("全天小时平均曲线无法恢复秒级电气尖峰；原始任务的秒级记录仍保留在任务表中。日/月统计由这些小时值累计，年度峰值指最高小时平均功率。")

    chapter("8 柔性权限独立作为控制设计")
    p("规模、设备模式和需求导向定义基础情境；柔性权限独立成章。上述 16 条曲线均为给定执行路径的重放结果，没有从业务标签自动赋予延期或暂停权限。")
    table([["控制设置", "求解变量", "必需服务与性能信息"],
        ["固定执行", "给定运行路径", "保留同一任务集合与完成要求"],
        ["允许延后启动", "启动时刻 s_i", "到达/数据就绪 r_i、截止 d_i、运行时长 p_i"],
        ["允许暂停恢复", "运行状态 x_i,t", "检查点、最小连续运行时间、恢复时长和额外耗能"],
        ["允许调整功率或资源", "节点功率上限、核心或卡数", "性能关系、内存、最小并行度与服务质量"]], [.24,.27,.49])
    p("固定资源、固定速度且连续执行时，满足 r_i≤s_i 且 s_i+p_i≤d_i，同时检查 CPU、GPU、内存和依赖约束。训练、评估、低优先级等标签用于候选筛选，不能单独确定可延期时长。")
    p("全年任务级扩展时，采用含边界作业的连续任务窗口，给每个重复周期生成独立任务编号，并保留跨年开始前已运行任务与年末未完成任务。对照和控制路径使用相同年度服务集合及边界约束，避免利用年末截尾减少完成工作量。")
    p("调度后重新积分资源需求，再计算 IT、散热、冷却和总负荷。当前年度重放代码已实现固定路径；任务级全年复制、检查点与混合节点联合调度尚未实现。")

    chapter("9 数据来源、文件与复现")
    p("设备配置按 2026-10-26 查阅的一手页面记录。Lisa 论文为 338 节点研究快照；公开硬件文件含 343 行，时间与记录边界有差异，未将其直接当作同时在线库存。全年构造使用已核验的小时资源表。", small)
    for label, url in SOURCES:
        p(f'<a href="{url}" color="#1C5675">{escape(label)}</a>', small)
    section("可复现入口与输出")
    p("在项目根目录依次运行：<br/>python analysis/build_task_annual_loads.py --year 2025 --grid-id g05_r100_c592<br/>python analysis/verify_task_annual_loads.py<br/>python analysis/build_scenario_load_methods_note_v2.py", small)
    p("annual-load-summary.csv 保存 16 条年曲线的设备数量、电量、峰值、资源累计与外推小时；annual-phase-sensitivity.csv 保存 490 组日相位结果；annualization-manifest.json 保存源时段、相位规则、输入及输出 SHA-256。scenario-matrix.csv/json 保存 36 个设计单元与实际准备状态。", small)
    p("完整逐小时负荷与源索引保存在项目 data/processed/task-annual-loads-20261026-v2/；代码及汇总进入研究仓库，带来源映射的数据文件保留在本地及服务器研究数据目录。可通过 --weather-path 接入同结构 ERA5 文件；标准逐小时温湿度可直接调用 ComputeLoadModel。", small)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("YaHei", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawRightString(A4[0]-18*mm, 11*mm, str(doc.page))
        canvas.restoreState()

    pdf = OUT / f"{STEM}.pdf"
    SimpleDocTemplate(str(pdf), pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=16*mm,
        bottomMargin=18*mm, title="算力中心实际配置与8760小时负荷计算说明", author="数据中心研究工作区").build(story, onFirstPage=footer, onLaterPages=footer)
    (OUT / f"{STEM}.md").write_text("\n\n".join(markdown), encoding="utf-8")
    with fitz.open(pdf) as doc:
        for i, page in enumerate(doc):
            page.get_pixmap(matrix=fitz.Matrix(1.35,1.35)).save(TEMP / f"page-{i+1:02d}.png")
        pages = len(doc)
        (TEMP / "document-text.txt").write_text("\n\n".join(p.get_text() for p in doc), encoding="utf-8")
    manifest = {"document": pdf.name, "pages": pages, "design_cells": 36, "computed_annual_curves": 16,
        "hours_per_curve": 8760, "total_hourly_records": 140160, "phase_cases": len(phase),
        "source_code_sha256": sha256(Path(__file__)), "mixed_it_fraction_status": "requires_matched_node_power_not_assumed",
        "files_sha256": {p.name: sha256(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "manifest.json"}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"pages": pages, "curves": 16, "sha256": sha256(pdf)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
