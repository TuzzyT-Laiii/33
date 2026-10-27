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

from compute_load.model import sha256


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/server-count-scenarios-20261027"
TEMP = PROJECT / ".codex-temp/server-count-scenarios-20261027"
STEM = "算力中心_单机CPU_GPU典型数量配比与需求来源说明_20261027"
WIDTH = A4[0]-36*mm
REFERENCES = [
    ("[1] Lenovo SR650 V3：双路平台、CPU 选件、GPU 表 84 与插槽/散热规则。", "https://lenovopress.lenovo.com/lp1601-thinksystem-sr650-v3-server"),
    ("[2] NVIDIA DGX H100/H200：固定 CPU/GPU 配置与整机最大功率。", "https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html"),
    ("[3] NVIDIA DGX B200：另一代双 CPU、八 GPU 整机。", "https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html"),
    ("[4] Latif 等：AMD EPYC 9354、8 张 H100 的 HGX 节点功率实测。", "https://arxiv.org/html/2412.08602v1"),
    ("[5] SURF Lisa v3：科研超算作业、硬件与 30 秒监测。", "https://zenodo.org/records/13685426"),
    ("[6] AcmeTrace：大模型开发作业与资源分配。", "https://github.com/InternLM/AcmeTrace"),
    ("[7] Alibaba ASI 2026：生产 AI 小时数据的官方字段。", "https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/docs/schema.md"),
    ("[8] Alibaba 2018：CPU 在线服务与批处理混部，约 4000 台、8 天。", "https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md"),
    ("[9] Microsoft Philly：训练任务、GPU/CPU 分钟监测与服务器 GPU 库存。", "https://github.com/msr-fiddle/philly-traces"),
    ("[10] Google 2019：可与任务记录关联的 57 个电力域功率。", "https://github.com/google/cluster-data/blob/master/PowerData2019.md")]


def configurations():
    rows = []
    for code, label, gpu_count in (("C21", "CPU 数量较多", 1), ("C22", "CPU/GPU 数量相等", 2)):
        rows.append({"configuration_id":code, "label":label, "cpu_packages_per_node":2,
            "gpu_devices_per_node":gpu_count, "cpu_gpu_device_count_ratio":f"2:{gpu_count}",
            "ratio_basis":"physical_CPU_packages_to_GPU_devices_per_complete_server",
            "vendor_platform":"Lenovo ThinkSystem SR650 V3", "server_form_factor":"rackmount", "rack_units_per_node":2,
            "cpu_model":"Intel Xeon Gold 6448Y",
            "cpu_cores_per_package":32, "cpu_cores_per_node":64, "cpu_tdp_w_per_package":225,
            "gpu_model":"NVIDIA H100 PCIe", "gpu_memory_gb_per_device":80, "gpu_tdp_w_per_device":350,
            "evidence_type":"manufacturer_supported_options_research_selection_not_fixed_factory_bundle",
            "source_url":REFERENCES[0][1], "source_sections":"Abstract; Processors; GPU adapters, Table 84; configuration rules",
            "selection_scope":"双路与一/两张 H100 由受支持选件和插槽规则组合选取；未取得销售频率统计",
            "gpu_slots":"2" if gpu_count == 1 else "2,5",
            "chassis_selection":"16x2.5-inch front bays; no middle/rear drive bays; GPU thermal kit and qualified fans/PSUs",
            "two_gpu_max_inlet_c":30 if gpu_count == 2 else None,
            "inlet_scope":"服务器进风要求；不等同于室外干球温度，最终完整 BOM 需满足厂商热规则",
            "max_nameplate_node_w":None, "measured_reference_node_w":None,
            "cpu_it_power_share":None, "gpu_it_power_share":None,
            "node_power_curve_status":"requires_measured_matching_complete_node_CPU_GPU_joint_response"})
    rows.append({"configuration_id":"C28", "label":"GPU 数量较多", "cpu_packages_per_node":2,
        "gpu_devices_per_node":8, "cpu_gpu_device_count_ratio":"2:8",
        "ratio_basis":"physical_CPU_packages_to_GPU_devices_per_complete_server",
        "vendor_platform":"NVIDIA DGX H100", "server_form_factor":"rackmount", "rack_units_per_node":8,
        "cpu_model":"Intel Xeon 8480C",
        "cpu_cores_per_package":56, "cpu_cores_per_node":112, "cpu_tdp_w_per_package":None,
        "gpu_model":"NVIDIA H100 SXM", "gpu_memory_gb_per_device":80, "gpu_tdp_w_per_device":None,
        "evidence_type":"manufacturer_documented_fixed_complete_system", "source_url":REFERENCES[1][1],
        "source_sections":"Component Description, Table 1; Mechanical Specifications, Table 2; Power Specifications, Table 3",
        "selection_scope":"固定八卡整机；DGX B200 官方配置也支持 2:8 架构的跨代实例",
        "gpu_slots":"HGX/NVSwitch integrated baseboard", "chassis_selection":"DGX H100 8U fixed system",
        "two_gpu_max_inlet_c":None, "inlet_scope":"以对应 DGX 官方环境规格为准",
        "max_nameplate_node_w":10200, "measured_reference_node_w":None,
        "cpu_it_power_share":None, "gpu_it_power_share":None,
        "node_power_curve_status":"requires_measured_matching_complete_node_CPU_GPU_joint_response"})
    return rows


def scenario_rows(catalog):
    rows = []
    for size, config, (business, source) in itertools.product((1,10,100,500), catalog,
            (("科研超算","Lisa"),("大模型开发","Acme Kalos"),("生产 AI 服务","ASI"))):
        rows.append({"scenario_id":f"{size}MW_{config['configuration_id']}_{source.split()[0].lower()}",
            "reference_it_mw":size, "configuration_id":config["configuration_id"],
            "cpu_packages_per_node":2, "gpu_devices_per_node":config["gpu_devices_per_node"],
            "cpu_cores_per_node":config["cpu_cores_per_node"], "rack_units_per_node":config["rack_units_per_node"],
            "cpu_gpu_device_count_ratio":config["cpu_gpu_device_count_ratio"], "demand_orientation":business,
            "demand_source":source, "ratio_basis":"physical_device_counts_per_complete_server",
            "node_count":None, "total_cpu_packages":None, "total_cpu_cores":None, "total_gpu_devices":None,
            "measured_reference_node_w":None, "cpu_it_power_share":None, "gpu_it_power_share":None,
            "power_calculation_status":"requires_configuration_matched_whole_node_calibration",
            "demand_readiness": {"Lisa":"CPU 作业已核验；对应 GPU 使用与同机联合占用待核对",
                "Acme Kalos":"任务 CPU/GPU 分配字段已读取；分配与实测利用率、跨硬件性能映射待核对",
                "ASI":"生产类型与库存小时字段已核验一天；完整样本及同机 CPU 占用口径待核对"}[source],
            "exogenous_input":"当地逐小时干球及露点/相对湿度", "flexibility_role":"separate_control_design"})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    catalog = configurations()
    scenarios = scenario_rows(catalog)
    if len(scenarios) != 36 or len({r["scenario_id"] for r in scenarios}) != 36:
        raise ValueError("应包含 36 个唯一的规模×单机数量配置×需求设计单元")
    for filename, rows in (("server-configurations",catalog),("scenario-matrix",scenarios)):
        pd.DataFrame(rows).to_csv(OUT/f"{filename}.csv", index=False, encoding="utf-8-sig")
        (OUT/f"{filename}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pdfmetrics.registerFont(TTFont("YaHei", "C:/Windows/Fonts/msyh.ttc"))
    pdfmetrics.registerFont(TTFont("YaHeiBold", "C:/Windows/Fonts/msyhbd.ttc"))
    pdfmetrics.registerFontFamily("YaHei", normal="YaHei", bold="YaHeiBold", italic="YaHei", boldItalic="YaHeiBold")
    body = ParagraphStyle("Body", fontName="YaHei", fontSize=10.1, leading=16, wordWrap="CJK", spaceAfter=8)
    small = ParagraphStyle("Small", parent=body, fontSize=8.4, leading=12.6)
    heading = ParagraphStyle("Heading", parent=body, fontName="YaHeiBold", fontSize=16, leading=23, spaceAfter=12)
    sub = ParagraphStyle("Sub", parent=body, fontName="YaHeiBold", fontSize=11.1, leading=17, spaceBefore=6)
    cell = ParagraphStyle("Cell", parent=body, fontSize=8.5, leading=12.5, spaceAfter=0)
    head = ParagraphStyle("Head", parent=cell, fontName="YaHeiBold")
    story, markdown = [], ["# 单机 CPU/GPU 典型数量配比与需求来源说明\n"]

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
        cells = [[Paragraph(escape(str(v)), head if i==0 else cell) for v in row] for i,row in enumerate(rows)]
        block = Table(cells, colWidths=[WIDTH*w for w in widths], repeatRows=1, hAlign="LEFT")
        block.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E9EFF3")),
            ("LINEBELOW",(0,0),(-1,0),.6,colors.HexColor("#8C9AA3")),
            ("LINEBELOW",(0,1),(-1,-1),.3,colors.HexColor("#D9DFE1")),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        story.extend([block,Spacer(1,8)])
        markdown.extend(["| "+" | ".join(map(str,rows[0]))+" |", "| "+" | ".join(["---"]*len(rows[0]))+" |"])
        markdown.extend("| "+" | ".join(map(str,row))+" |" for row in rows[1:])

    p("单机 CPU/GPU 典型数量配比<br/>与需求来源说明", ParagraphStyle("Title",parent=heading,fontSize=20,leading=29))
    p("用户确认口径：每台服务器的 CPU 颗数∶GPU 数量 · 2026 年 10 月 27 日",small)
    chapter("1 三种情境采用哪些实际配置", True)
    p("研究基本单元为数据中心内的一台完整机架式服务器。SR650 V3 为双路 2U 通用服务器，DGX H100 为 8U 加速服务器；U 表示机架内设备的高度单位。[1,2]")
    table([["情境", "单机数量", "参考平台", "证据性质"],
        ["C21：CPU 数量较多", "2 CPU∶1 GPU", "双路 SR650 V3＋1 张 H100 PCIe", "从厂商支持选件与插槽规则中选取 [1]"],
        ["C22：两类数量相等", "2 CPU∶2 GPU", "双路 SR650 V3＋2 张 H100 PCIe", "厂商明确列出两卡位置及温度条件 [1]"],
        ["C28：GPU 数量较多", "2 CPU∶8 GPU", "DGX H100：2 颗 Xeon 8480C＋8 张 H100", "官方固定整机配置 [2]"]], [.24,.18,.32,.26])
    p("三种情境覆盖双路通用服务器的少卡选配和八卡专用加速服务器。单卡与双卡为厂商支持项构成的研究配置；本次没有取得装机销量或市场频率，因此‘典型’表示可核查的代表性架构。")
    p("CPU 按物理处理器封装/插槽数量计，GPU 按 PCIe 卡或 SXM 加速器模块数量计。两颗 CPU 各含几十个物理核心：C21/C22 每台共 64 核，C28 每台共 112 核。两颗 CPU 配两张 GPU 的数量为一比一；算力、功率及全年电量分别计算。")
    p("三种单机均配有宿主 CPU。宿主负责该服务器的通用处理和 GPU 任务配套；整机功率已经包含宿主时，其功耗随整机计入。DGX B200 官方同样采用 2 颗 CPU＋8 张 GPU，提供另一代的八卡架构实例。[3]",small)
    p("三种情境同时记录 CPU 型号、GPU 接口与整机结构差异；结果比较描述这些实际架构组合，不能把全部差异归因于 CPU/GPU 数量比例。",small)

    chapter("2 固定的设备参数与适用条件")
    table([["参数", "C21", "C22", "C28"],
        ["机箱形态", "2U 机架式", "2U 机架式", "8U 机架式"],
        ["物理 CPU 颗数/台", "2", "2", "2"],
        ["CPU 参考选件", "Xeon Gold 6448Y", "Xeon Gold 6448Y", "Xeon 8480C"],
        ["CPU 核心数/台", "64（2×32）", "64（2×32）", "112（2×56）"],
        ["GPU 型号/数量", "H100 PCIe×1", "H100 PCIe×2", "H100 SXM×8"],
        ["GPU 显存合计", "80 GB", "160 GB", "640 GB"],
        ["CPU 单颗 TDP", "225 W", "225 W", "本说明未核定"],
        ["GPU 单卡 TDP", "350 W", "350 W", "本说明未核定"],
        ["整机最大铭牌功率", "未核定", "未核定", "10.2 kW [2]"],
        ["匹配整机功率曲线", "待标定", "待标定", "待标定"]], [.27,.24,.24,.25])
    p("SR650 V3 参考选配采用 16×2.5 英寸前置盘位、GPU 插槽 2（单卡）或 2/5（双卡），并遵守 GPU 散热套件、风扇、电源及盘位限制。所列 CPU 和 GPU 为厂商支持选件，完整整机物料组合仍须符合其配置与热规则。[1]",small)
    p("厂商对该双卡 H100 组合规定服务器进风不高于 30℃。这是机房内服务器进风条件；室外天气通过冷却系统影响进风和耗电，不能直接用室外气温判定服务器是否可运行。",small)
    section("数量相等与功率份额分别记录")
    p("C22 的芯片 TDP 合计为 CPU 450 W、GPU 700 W，CPU 在这两部分 TDP 之和中约占 39.1%；它仍是 CPU/GPU 数量相等的情境。TDP 用于描述器件规格，整机还包括内存、机内互联、存储、风扇和电源转换，全年功率需由运行与测量确定。")

    chapter("3 三种单机配置怎样接入负荷计算")
    p("基础设计为 1、10、100、500 MW × C21/C22/C28 × 科研超算/大模型开发/生产 AI 服务，共 36 个设计单元。MW 表示数据中心参考高负载 IT 预算。服务器台数根据完整 IT 清单中分配给计算节点的预算换算，公共网络及独立存储的预算另行计入。")
    section("同台 CPU/GPU 的联合响应")
    p("对配置 k，读取对应硬件的整机响应 p_k(u_CPU,h,u_GPU,h)，保留同一时段 CPU、GPU、内存及任务关系。p_k 已包含整机所有被测组件；CPU 颗数与 GPU 数量用于定义设备和资源上限。")
    p("记计算节点预算为 S_node≤S，台数 N_k=floor(S_node×10<sup>6</sup>/p_k,ref)。小时计算节点功率 P_node,h=N_k p_k(u_CPU,h,u_GPU,h)/10<sup>6</sup>，其中 p 以 W 计、S 与 P 以 MW 计。CPU 颗数=2N_k，物理核心数=c_kN_k（c_k 为 64、64、112），GPU 数量=g_kN_k（g_k 为 1、2、8）。")
    p("完整服务器功率包含 CPU、GPU、内存、主板、机内存储、网卡、风扇和电源转换损耗；这些部分随整机计入 IT。机房冷却设备的耗电在设施侧单独计算。",small)
    p("不同 CPU/GPU 型号需要对应性能与功率标定；PCIe 与 SXM 卡也保留不同硬件标识。增加 GPU 数量会改变互联、内存、风扇和任务性能，整机功率不能仅由卡数比推算。")
    section("散热、温湿度与全年结果")
    p("公共网络和独立存储的 IT 功率按清单另加，得到 P_IT,h。散热需求 Q_h≈P_IT,h；使用当地逐小时温度及露点/相对湿度、冷却技术参数计算 P_cool,h，再形成 P_fac,h=P_IT,h+P_cool,h+P_other,h。")
    p("新三类配置的完整整机功率曲线尚未核定，情境表中台数、总 CPU/GPU 数量、功率份额和年负荷结果保持为空。DGX 10.2 kW 是最大铭牌值；既有 8.43 kW 实测来自另一台 AMD EPYC 9354＋8 张 H100 的 HGX 节点。[2,4]",small)

    chapter("4 Lisa、Acme、ASI 各自起什么作用")
    table([["来源", "需求模式与记录", "本工作区的使用范围"],
        ["Lisa [5]", "科研超算；秒级作业与公开 30 秒监测", "约 160 万作业已核验；CPU 有 309 个完整日，完整功率监测未进入本次重标定"],
        ["Acme [6]", "大模型开发；CPU/GPU 分配、预训练、评估、微调等", "62413 条作业；90 个完整日。GPU 作业也有 CPU 分配字段，仍须核对利用率口径"],
        ["ASI [7]", "生产 AI；逐小时训练、在线/离线推理、开发及优先级", "当前核验相对日 109 的 24 小时；库存和业务类型可用于生产 GPU 需求分析"]], [.18,.39,.43])
    section("ASI 的具体作用")
    p("ASI 补充生产运营中的业务构成与小时变化，提供归属 GPU 小时、服务器 GPU 库存、CPU 容量与请求利用率等字段；也能按高/低优先级区分候选任务。它不直接给出 CPU 物理插槽数量、逐任务允许延期时间或抢占许可。")
    p("因此，ASI 用于选择生产需求路径和柔性候选范围；单机 2∶1、2∶2、2∶8 的设备参数来自厂商配置。任务类型、历史低优先级和小时归属量分别保留其来源含义。")
    p("已有 ASI 单日重复 365 次形成的年曲线属于方法演示，不能据此判断全年生产业务的季节变化。Lisa、Acme、ASI 是本工作区已经详细核验的三套主样本；其他公开数据按其适用问题单独评估。",small)

    chapter("5 其他可用于本研究的公开数据")
    table([["来源", "可补充的问题", "证据与当前处理状态"],
        ["Microsoft Philly [9]", "训练作业、同服务器 CPU/GPU 共同变化；可帮助区分宿主与 GPU 需求", "公开任务运行尝试与每分钟 CPU/GPU 利用率、GPU 库存；本次核对官方字段，未下载计算"],
        ["Alibaba CPU 2018 [8]", "CPU 在线服务与批处理混部，任务依赖与两类业务的资源需求", "约 4000 台、8 天，提供机器/容器与批处理表；本次核对说明，未下载计算"],
        ["Google 2019 PowerData [10]", "核对资源需求如何映射到电力域功率，验证功率模型", "57 个电力域，与对应 Borg 数据有关联；本次核对说明，未提取或拟合"]], [.25,.37,.38])
    p("数据是否适用，取决于研究对象、任务粒度、资源分母、CPU/GPU 共时关系、实测功率和服务权限。任务记录适合研究执行和调度，利用率记录适合研究动态占用，功率记录适合标定用电；三者需要按标识与时间边界连接。")
    p("Philly 的公开数据覆盖 2017 年提交的训练任务，硬件和业务年代与 H100 不同；可用来检验联合需求形状，迁移到新设备仍需性能映射。Google 的电力域功率也需核对其表计边界，不能直接当成单节点 CPU/GPU 功率。",small)
    chapter("6 8760 小时生成规则与已有结果")
    p("源任务先按真实运行区间积分，得到同一时刻的 CPU 核心小时、GPU 小时或利用率；确认同机联合需求后，使用相同的时间映射扩展两条路径。不能把互不对应的独立 CPU 业务曲线直接当成 GPU 服务器的宿主占用。")
    table([["现有源窗口", "2025 年 8760 小时的重放方法", "说明"],
        ["Lisa 309 日", "1 个完整窗口＋前 56 日", "保留连续资源形状，样本外季节性未知"],
        ["Acme 90 日", "4 个完整窗口＋前 5 日", "CPU/GPU 同相位映射，保留业务关系"],
        ["ASI 1 日", "同一天重复 365 次", "方法演示，跨日变化未识别"]], [.23,.42,.35])
    p("规则为 j(h)=(h+φ) mod L；H=8760（平年）或 8784（闰年）。保存每小时源索引、循环边界与起始相位；使用目标年逐小时天气重新计算冷却。年电量 E=Σ_hP_h×1 小时，年度 PUE=ΣP_fac/ΣP_IT。")
    p("已有 20261026-v2 年度报告提供 16 条、140160 条小时负荷记录及 490 组相位检查。它们是既有纯设备池参考模型的计算结果，与本说明的三种单机数量配置分别保存。新三类配置的台数及年曲线须使用本章联合需求和第 3 章对应整机功率标定。")
    p("本次交付完成三类实际配置的参数表和 36 个设计单元；服务器内 CPU 与 GPU 的功率关系保留待测字段。旧年曲线、输入与校验记录保持原始版本，来源和适用范围随其报告使用。",small)

    chapter("7 柔性权限与控制设计")
    p("柔性权限独立于规模、单机数量配置和需求导向。对同一任务集合，可以设置固定执行、允许延后启动、允许暂停恢复或允许调整功率/资源的控制范围。")
    table([["控制设置", "需要的依据", "求解后重新计算"],
        ["延后启动", "数据就绪、完成期限、运行时长和依赖", "CPU/GPU 同时占用及资源约束"],
        ["暂停与恢复", "检查点、最小连续运行、恢复时长和额外能耗", "完整运行量与恢复开销"],
        ["调整功率或资源", "设备与任务的性能关系、最小并行度、内存和服务时延", "完成工作量、功率和冷却"]], [.22,.43,.35])
    p("训练、推理、CPU 标签和任务优先级支持候选分类；可延期时长和暂停权限按实际服务约束或明确的控制设计填写，未知值保留为空。")
    p("在同一服务器上，CPU 与 GPU 可能共同参与一项任务，不能将其独立平移到互不对应的小时。调度改变执行路径后，重新计算联合资源、整机 IT、散热与气象相关冷却。")
    p("本次交付为单机数量情境及参数说明，未求解任务级全年调度或确认新增的业务权限。",small)

    chapter("8 来源、参数文件与复现")
    p("硬件与其他数据源于 2026-10-27 查阅厂商/数据发布者原文。Lenovo 1/2 卡为依据受支持选件及插槽约束选取的研究配置；DGX 8 卡为官方固定配置。原文没有提供这三种配置的市场占有率。",small)
    for label,url in REFERENCES:
        p(f'<a href="{url}" color="#1C5675">{escape(label)}</a>',small)
    section("参数与结果文件")
    p("server-configurations.csv/json 记录机架形态、U 数、每台 CPU/GPU 数量、CPU 核心、型号、显存、部分 TDP、证据类型和完整整机功率状态。scenario-matrix.csv/json 提供 36 个规模×单机配置×需求设计单元，包含单台核心数与 U 数，未知功率、台数和总核心数为空。",small)
    p("在项目根目录运行：<br/>python analysis/build_server_count_scenarios.py<br/>参数核验与生成文件 SHA-256 见 manifest.json、validation.json。既有年度算法与结果见 results/scenario-load-methods-20261026-v2/。",small)

    def footer(canvas,doc):
        canvas.saveState()
        canvas.setFont("YaHei",8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawRightString(A4[0]-18*mm,11*mm,str(doc.page))
        canvas.restoreState()

    pdf = OUT/f"{STEM}.pdf"
    SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=16*mm,
        bottomMargin=18*mm,title="单机CPU/GPU典型数量配比与需求来源说明",author="数据中心研究工作区").build(story,onFirstPage=footer,onLaterPages=footer)
    (OUT/f"{STEM}.md").write_text("\n\n".join(markdown),encoding="utf-8")
    with fitz.open(pdf) as doc:
        for i,page in enumerate(doc):
            page.get_pixmap(matrix=fitz.Matrix(1.35,1.35)).save(TEMP/f"page-{i+1:02d}.png")
        pages = len(doc)
        (TEMP/"document-text.txt").write_text("\n".join(page.get_text() for page in doc),encoding="utf-8")
        links = [link["uri"] for page in doc for link in page.get_links() if link.get("uri")]
        if pages != 8 or len(set(links)) != len(REFERENCES):
            raise ValueError("页数或来源链接数量与报告设计不一致")
        for page in doc:
            for word in page.get_text("words"):
                if not (0<=word[0]<=word[2]<=page.rect.width and 0<=word[1]<=word[3]<=page.rect.height):
                    raise ValueError("PDF 存在越界文字")
    table_readback = pd.read_csv(OUT/"server-configurations.csv")
    matrix_readback = pd.read_csv(OUT/"scenario-matrix.csv")
    if table_readback.gpu_devices_per_node.tolist()!=[1,2,8] or table_readback.cpu_packages_per_node.tolist()!=[2,2,2]:
        raise ValueError("三种单机数量与设计不一致")
    if table_readback.cpu_cores_per_node.tolist()!=[64,64,112]:
        raise ValueError("CPU 核心数量与选定型号不一致")
    if table_readback.rack_units_per_node.tolist()!=[2,2,8] or not table_readback.server_form_factor.eq("rackmount").all():
        raise ValueError("服务器形态与厂商机架规格不一致")
    for config in catalog:
        subset = matrix_readback[matrix_readback.configuration_id.eq(config["configuration_id"])]
        if not subset.cpu_cores_per_node.eq(config["cpu_cores_per_node"]).all() or not subset.rack_units_per_node.eq(config["rack_units_per_node"]).all():
            raise ValueError(f"情境表与服务器参数不一致：{config['configuration_id']}")
    for field in ("measured_reference_node_w","cpu_it_power_share","gpu_it_power_share"):
        if not table_readback[field].isna().all() or not matrix_readback[field].isna().all():
            raise ValueError("未识别的功率参数必须保留为空")
    if len(matrix_readback)!=36 or not matrix_readback.scenario_id.is_unique or not matrix_readback.node_count.isna().all() or not matrix_readback.total_cpu_cores.isna().all():
        raise ValueError("情境表行数、唯一性或台数状态不一致")
    baseline = PROJECT/"results/scenario-load-methods-20261026-v2/算力中心_实际配置与8760小时负荷计算说明_20261026_v2.pdf"
    if sha256(baseline)!="b587cfb06a344d632392b18e11b82416c9c0e3d3c6f2076745858250ebba2eb9":
        raise ValueError("既有年度报告与输入版本不一致")
    validation = {"status":"passed", "checks":["device_counts", "core_counts", "rackmount_form_factors", "scenario_hardware_consistency", "unknown_power_preserved", "36_unique_cells",
        "CSV_JSON_consistency", "source_links", "PDF_bounds", "annual_baseline_preserved"],
        "pages":pages, "configurations":3, "design_cells":36, "source_links":len(set(links)),
        "existing_annual_pdf_sha256":sha256(baseline), "pdf_sha256":sha256(pdf)}
    for stem, expected in (("server-configurations",catalog),("scenario-matrix",scenarios)):
        reread = json.loads((OUT/f"{stem}.json").read_text(encoding="utf-8"))
        if reread!=expected:
            raise ValueError(f"JSON 回读不一致：{stem}")
        from_csv = pd.read_csv(OUT/f"{stem}.csv").astype(object)
        from_csv = from_csv.where(pd.notna(from_csv),None)
        expected_frame = pd.DataFrame(expected).astype(object)
        expected_frame = expected_frame.where(pd.notna(expected_frame),None)
        pd.testing.assert_frame_equal(from_csv,expected_frame,check_dtype=False)
    (OUT/"validation.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
    manifest = {"document":pdf.name,"pages":pages,"checked_date":"2026-10-27", "ratio_basis":"per_server_physical_CPU_package_and_GPU_device_counts",
        "configurations":3,"design_cells":36,"new_hourly_curves_computed":0,
        "source_code_sha256":sha256(Path(__file__)),
        "files_sha256":{p.name:sha256(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!="manifest.json"}}
    (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"pages":pages,"configurations":3,"design_cells":36,"pdf_sha256":sha256(pdf)},ensure_ascii=False))


if __name__ == "__main__":
    main()
