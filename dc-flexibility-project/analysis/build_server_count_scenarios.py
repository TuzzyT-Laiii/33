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
OUT = PROJECT / "results/server-count-scenarios-20261028"
TEMP = PROJECT / ".codex-temp/server-count-scenarios-20261028"
PREVIOUS = PROJECT / "results/server-count-scenarios-20261027"
STEM = "算力中心_CPU_GPU服务器与整柜情境及需求来源说明_20261028"
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
    ("[10] Google 2019：可与任务记录关联的 57 个电力域功率。", "https://github.com/google/cluster-data/blob/master/PowerData2019.md"),
    ("[11] NVIDIA GB200 NVL72：CPU 核心与 GPU 显存规格。", "https://www.nvidia.com/en-us/data-center/gb200-nvl72/"),
    ("[12] NVIDIA DGX GB 硬件指南：托盘组成、冷板液冷及约 120 kW 整柜功耗。", "https://docs.nvidia.com/dgx/dgxgb200-user-guide/hardware.html"),
    ("[13] NVIDIA GB200 参考架构：每托盘两个 GB200 Superchip、四个 B200 GPU。", "https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/dgx-superpod-components.html"),
    ("[14] NVIDIA DGX GB200 数据表：整柜规格，未列外部机械尺寸。", "https://resources.nvidia.com/en-us-dgx-systems/dgx-superpod-gb200-datasheet"),
    ("[15] Alibaba 2018 官方 schema：CPU、机器/容器监测及任务字段。", "https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/schema.txt")]

DEMANDS = {
    "Lisa": ("科研超算", "CPU 作业已核验；对应 GPU 使用与同机联合占用待核对"),
    "Acme Kalos": ("大模型开发", "任务 CPU/GPU 分配字段已读取；分配与实测利用率、跨硬件性能映射待核对"),
    "ASI": ("生产 AI 服务", "生产类型与库存小时字段已核验一天；完整样本及同机 CPU 占用口径待核对"),
    "Alibaba Cluster Trace 2018": ("CPU 云混部", "仅核对官方说明与字段；本次未下载，原始数据下载、校验和字段实值核验待完成"),
}
PAIRINGS = {
    "C20": ("Lisa", "Alibaba Cluster Trace 2018"),
    "C21": ("Lisa", "Acme Kalos", "ASI"),
    "C22": ("Lisa", "Acme Kalos", "ASI"),
    "C28": ("Lisa", "Acme Kalos", "ASI"),
    "R-GB200": ("Acme Kalos", "ASI"),
}


def configurations():
    rows = []
    for code, label, gpu_count in (("C20", "纯 CPU", 0), ("C21", "CPU 数量较多", 1), ("C22", "CPU/GPU 数量相等", 2)):
        rows.append({"configuration_id":code, "label":label, "cpu_packages_per_node":2,
            "gpu_devices_per_node":gpu_count, "cpu_gpu_device_count_ratio":f"2:{gpu_count}",
            "ratio_basis":"physical_CPU_packages_to_GPU_devices_per_complete_server",
            "vendor_platform":"Lenovo ThinkSystem SR650 V3", "server_form_factor":"rackmount", "rack_units_per_node":2,
            "cpu_model":"Intel Xeon Gold 6448Y",
            "cpu_cores_per_package":32, "cpu_cores_per_node":64, "cpu_tdp_w_per_package":225,
            "gpu_model":"NVIDIA H100 PCIe" if gpu_count else None,
            "gpu_memory_gb_per_device":80 if gpu_count else None, "gpu_tdp_w_per_device":350 if gpu_count else None,
            "evidence_type":"manufacturer_supported_options_research_selection_not_fixed_factory_bundle",
            "source_url":REFERENCES[0][1],
            "source_sections":("Standard specifications, Table 3 (2U, one/two sockets, embedded video); Processors, Table 19 (6448Y 32C 225W, max quantity 2); GPU adapters (optional)" if gpu_count == 0
                else "Abstract; Processors; GPU adapters, Table 84; configuration rules"),
            "selection_scope":("双路 6448Y 受支持选件；不安装计算 GPU；管理控制器集成显示不计入计算 GPU" if gpu_count == 0
                else "双路与一/两张 H100 由受支持选件和插槽规则组合选取；未取得销售频率统计"),
            "gpu_slots":{0:None, 1:"2", 2:"2,5"}[gpu_count],
            "chassis_selection":("air-cooled 2U; no compute GPU; matching memory/storage/fan/PSU BOM pending" if gpu_count == 0
                else "16x2.5-inch front bays; no middle/rear drive bays; GPU thermal kit and qualified fans/PSUs"),
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
    for row in rows:
        row.update({"research_unit":"server", "cooling_type":"air",
            "cpu_packages_per_unit":row["cpu_packages_per_node"],
            "gpu_devices_per_unit":row["gpu_devices_per_node"],
            "cpu_cores_per_unit":row["cpu_cores_per_node"],
            "rack_units_per_unit":row["rack_units_per_node"],
            "measured_reference_unit_w":None,
            "cooling_model_status":"existing_air_room_model_requires_applicability_check",
            "source_checked_date":"2026-10-28" if row["configuration_id"] == "C20" else "2026-10-27"})
    rack = dict.fromkeys(rows[0])
    rack.update({"configuration_id":"R-GB200", "label":"基本纯 GPU（GPU 主导）",
        "research_unit":"rack", "cpu_packages_per_unit":36, "gpu_devices_per_unit":72,
        "cpu_gpu_device_count_ratio":"36:72", "cpu_cores_per_unit":2592,
        "ratio_basis":"physical_CPU_packages_to_GPU_devices_per_complete_rack",
        "vendor_platform":"NVIDIA GB200 NVL72 (DGX GB200 reference rack)",
        "server_form_factor":"rack_scale", "cpu_model":"NVIDIA Grace (Arm Neoverse V2)",
        "cpu_cores_per_package":72, "gpu_model":"NVIDIA Blackwell B200 (GB200 Superchip)",
        "gpu_memory_gb_per_device":186, "gpu_memory_tb_per_unit_vendor_rounded":13.4,
        "gpu_memory_basis":"372 GB HBM3E per two-GPU GB200 Superchip / 2 = 186 GB per GPU; vendor rack total rounded to 13.4 TB",
        "compute_trays_per_unit":18, "cpu_packages_per_compute_tray":2, "gpu_devices_per_compute_tray":4,
        "nvlink_switch_trays_per_unit":9, "nvlink_switches_per_tray":2,
        "rack_height_mm":None, "rack_width_mm":None, "rack_depth_mm":None,
        "rack_dimensions_status":"not_listed_in_consulted_NVIDIA_datasheet_or_hardware_guide; pending_official_mechanical_specification",
        "vendor_approx_rack_power_w":120000,
        "vendor_power_scope":"NVIDIA hardware guide approximate rack consumption; workload and measurement boundary unspecified; not a calibrated operating reference",
        "cooling_type":"direct_liquid",
        "cooling_scope":"CPU/GPU cold plates require direct liquid cooling; networking/storage/PSU components retain air cooling",
        "cooling_model_status":"pending_separate_direct_liquid_weather_coupling; no_new_model_implemented",
        "evidence_type":"manufacturer_documented_complete_rack_reference_configuration",
        "source_url":REFERENCES[10][1], "hardware_source_url":REFERENCES[11][1],
        "architecture_source_url":REFERENCES[12][1], "mechanical_review_source_url":REFERENCES[13][1],
        "source_sections":"GB200 NVL72 Specs; Hardware: Rack Configuration, Compute Trays, NVLink Switch Trays, Power Shelves; DGX GB200 datasheet Technical Specifications",
        "selection_scope":"R 表示 rack，整柜为基本单元；36 颗 Grace 仅计计算主机 CPU，不计 DPU/BMC/交换管理处理器",
        "gpu_slots":"18 compute trays x 2 GB200 Superchips x 2 Blackwell GPUs",
        "chassis_selection":"complete NVL72 rack including NVLink switches, internal network/storage, fans and power conversion",
        "inlet_scope":"待核对对应整柜供液、流量和残余风冷要求；室外天气需独立液冷参数",
        "node_power_curve_status":"requires_measured_matching_complete_rack_CPU_GPU_joint_response",
        "source_checked_date":"2026-10-28"})
    rows.append(rack)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    return [{key:row.get(key) for key in fields} for row in rows]


def scenario_rows(catalog):
    rows = []
    combinations = ((size, config, source) for size, config in itertools.product((1,10,100,500), catalog)
        for source in PAIRINGS[config["configuration_id"]])
    for size, config, source in combinations:
        business, readiness = DEMANDS[source]
        if config["configuration_id"] == "C20" and source == "Lisa":
            business, readiness = "科研超算（CPU 作业）", "CPU 分区作业已核验；完整功率监测及向双路 6448Y 的性能映射待核对"
        rows.append({"scenario_id":f"{size}MW_{config['configuration_id']}_{source.split()[0].lower()}",
            "reference_it_mw":size, "configuration_id":config["configuration_id"],
            "research_unit":config["research_unit"], "cooling_type":config["cooling_type"],
            "cpu_packages_per_unit":config["cpu_packages_per_unit"], "gpu_devices_per_unit":config["gpu_devices_per_unit"],
            "cpu_cores_per_unit":config["cpu_cores_per_unit"], "rack_units_per_unit":config["rack_units_per_unit"],
            "cpu_packages_per_node":config["cpu_packages_per_node"], "gpu_devices_per_node":config["gpu_devices_per_node"],
            "cpu_cores_per_node":config["cpu_cores_per_node"], "rack_units_per_node":config["rack_units_per_node"],
            "cpu_gpu_device_count_ratio":config["cpu_gpu_device_count_ratio"], "demand_orientation":business,
            "demand_source":source, "ratio_basis":("physical_device_counts_per_complete_rack" if config["research_unit"] == "rack"
                else "physical_device_counts_per_complete_server"),
            "unit_count":None, "rack_count":None,
            "node_count":None, "total_cpu_packages":None, "total_cpu_cores":None, "total_gpu_devices":None,
            "measured_reference_unit_w":None, "measured_reference_node_w":None,
            "cpu_it_power_share":None, "gpu_it_power_share":None,
            "power_calculation_status":("requires_configuration_matched_whole_rack_calibration" if config["research_unit"] == "rack"
                else "requires_configuration_matched_whole_node_calibration"),
            "cooling_model_status":config["cooling_model_status"],
            "demand_readiness":readiness,
            "exogenous_input":"当地逐小时干球及露点/相对湿度", "flexibility_role":"separate_control_design"})
    return rows


def main():
    previous_hashes = {p.name:sha256(p) for p in sorted(PREVIOUS.iterdir()) if p.is_file()}
    previous_manifest = json.loads((PREVIOUS/"manifest.json").read_text(encoding="utf-8"))
    if any(previous_hashes[name] != digest for name,digest in previous_manifest["files_sha256"].items()):
        raise ValueError("旧版结果与其归档 SHA-256 不一致")
    OUT.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    catalog = configurations()
    scenarios = scenario_rows(catalog)
    if len(scenarios) != 52 or len({r["scenario_id"] for r in scenarios}) != 52:
        raise ValueError("应包含 52 个唯一的规模×配置×允许需求设计单元")
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
    story, markdown = [], ["# CPU/GPU 服务器与整柜情境及需求来源说明\n"]

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

    p("CPU/GPU 服务器与整柜情境<br/>及需求来源说明", ParagraphStyle("Title",parent=heading,fontSize=20,leading=29))
    p("2026 年 10 月 28 日 · 5 种配置 · 52 个设计单元 · 基本单元显式区分服务器与机柜",small)
    chapter("1 五种配置及其研究基本单元", True)
    p("C20/C21/C22/C28 按完整服务器计数；R-GB200 按 GB200 NVL72 整柜计数。R 表示 rack，避免将整柜误读为单台服务器。SR650 V3 为 2U、DGX H100 为 8U；U 表示机架内设备的高度单位。[1,2,11,12]")
    table([["情境", "CPU∶GPU", "参考平台", "证据性质"],
        ["C20：纯 CPU", "2∶0 / 台", "双路 SR650 V3，不装计算 GPU", "厂商支持的双路 6448Y 选配 [1]"],
        ["C21：CPU 数量较多", "2 CPU∶1 GPU", "双路 SR650 V3＋1 张 H100 PCIe", "从厂商支持选件与插槽规则中选取 [1]"],
        ["C22：两类数量相等", "2 CPU∶2 GPU", "双路 SR650 V3＋2 张 H100 PCIe", "厂商明确列出两卡位置及温度条件 [1]"],
        ["C28：GPU 数量较多", "2∶8 / 台", "DGX H100：2 颗 Xeon 8480C＋8 张 H100", "官方固定整机配置 [2]"],
        ["R-GB200：基本纯 GPU", "36∶72 / 柜", "GB200 NVL72，Grace＋Blackwell", "NVIDIA 官方整柜架构 [11-14]"]], [.24,.18,.32,.26])
    p("CPU 按物理处理器封装计，GPU 按计算卡或加速器模块计。管理控制器的集成显示不计入计算 GPU；机柜 DPU、BMC 和交换管理处理器不计入 36 颗 Grace 主机 CPU。数量、算力、功率及全年电量分别核算。",small)
    p("GPU 服务器需要宿主 CPU。‘基本纯 GPU’按 CPU 仅承担主机及 GPU 配套工作、电耗以 GPU 为主的研究形态定义；实际功率份额仍待测量，不能从 36∶72 推算。不同配置的处理器、互联及冷却差异同时保留，比较结果按完整架构解释。",small)

    chapter("2 服务器参数与适用条件")
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
    p("C20 采用同一 SR650 V3 的双路 Xeon Gold 6448Y：每颗 32 核、225 W TDP，每台共 64 核、2U，计算 GPU 数量为零。依据产品指南标准规格和第 19 表的双路受支持选件选取；最终内存、存储、风扇和电源 BOM 待定，整机运行功率留空。[1]",small)
    section("数量相等与功率份额分别记录")
    p("C22 的芯片 TDP 合计为 CPU 450 W、GPU 700 W，CPU 在这两部分 TDP 之和中约占 39.1%；它仍是 CPU/GPU 数量相等的情境。TDP 用于描述器件规格，整机还包括内存、机内互联、存储、风扇和电源转换，全年功率需由运行与测量确定。")

    chapter("3 GB200 NVL72 整柜参数")
    table([["参数", "R-GB200：整柜研究基本单元"],
        ["计算托盘", "18 个 1U compute tray；每个 2 颗 Grace＋4 个 Blackwell B200 GPU [12,13]"],
        ["NVLink 交换托盘", "9 个 1U NVLink switch tray，每个含 2 个 NVSwitch [12]"],
        ["主机 CPU", "36 颗 Grace，每颗 72 个 Arm Neoverse V2 核心，共 2592 核 [11]"],
        ["GPU", "72 个 Blackwell B200，通过一个 NVL72 域互联 [11,13]"],
        ["GPU 显存", "整柜 13.4 TB HBM3E（官方约数）；每个双 GPU Superchip 为 372 GB [11]"],
        ["单 GPU 显存口径", "由 372÷2 得 186 GB；保留 GB200 规格，其他 B200 产品的显存口径另行使用"],
        ["整柜功耗", "硬件指南约 120 kW；未说明任务条件和测量边界，不能作为标定参考 [12]"],
        ["整柜外部尺寸与高度 U 数", "所查 NVIDIA 数据表及硬件指南未列外形尺寸，相关数值留空，待官方机械规格 [12,14]"],
        ["冷却方式", "direct_liquid；CPU/GPU 冷板必须接入液冷，仍有风冷部件 [12,13]"],
        ["运行功率与 CPU/GPU 份额", "匹配整柜联合功率响应待标定，保留为空"]], [.26,.74])
    p("GPU 采用 GB200 Superchip 内的 Blackwell B200，区别于独立八卡 DGX B200 整机。每个计算托盘包含两个 Superchip，每个 Superchip 含一颗 Grace 和两个 B200；18×2=36 颗 CPU、18×4=72 个 GPU。[13]",small)
    p("整柜计量包含托盘、NVLink 交换、柜内网络与存储、风扇及电源转换损耗，避免按托盘再重复相加。柜外公共网络、独立存储及设施侧冷却按各自边界另计。约 120 kW 单独保存为厂商参考描述；CPU/GPU TDP、最大铭牌、实测运行功率和供液参数未核定的字段保持为空。",small)

    chapter("4 配对矩阵与负荷计算边界")
    table([["配置", "允许需求", "每档规模", "四档合计"],
        ["C21/C22/C28", "各配 Lisa、Acme、ASI", "9", "36"],
        ["C20", "Lisa CPU 作业、Alibaba Cluster Trace 2018", "2", "8"],
        ["R-GB200", "Acme、ASI", "2", "8"],
        ["合计", "规模为 1 / 10 / 100 / 500 MW", "13", "52"]], [.22,.46,.16,.16])
    p("MW 表示完整参考 IT 预算。配置按允许需求配对，scenario_id 沿用‘规模MW_配置编号_来源首词小写’，例如 1MW_C20_lisa、1MW_C20_alibaba、1MW_R-GB200_acme；原有 36 个标识保留。各基本单元的数量根据分配预算与对应完整单元参考功率换算。",small)
    section("同台 CPU/GPU 的联合响应")
    p("对配置 k，读取对应硬件的整机响应 p_k(u_CPU,h,u_GPU,h)，保留同一时段 CPU、GPU、内存及任务关系。p_k 已包含整机所有被测组件；CPU 颗数与 GPU 数量用于定义设备和资源上限。")
    p("记基本单元预算为 S_unit≤S，数量 N_k=floor(S_unit×10<sup>6</sup>/p_k,ref)。小时功率 P_unit,h=N_k p_k(u_CPU,h,u_GPU,h)/10<sup>6</sup>。CPU 总颗数=a_kN_k、核心数=c_kN_k、GPU 数量=g_kN_k；a/c/g 读取每个 per_unit 参数。C20 的 g=0；R-GB200 每单元为整柜的 36/2592/72。",small)
    p("完整服务器功率包含 CPU、GPU、内存、主板、机内存储、网卡、风扇和电源转换损耗；这些部分随整机计入 IT。机房冷却设备的耗电在设施侧单独计算。",small)
    p("不同 CPU/GPU 型号需要对应性能与功率标定；PCIe 与 SXM 卡也保留不同硬件标识。增加 GPU 数量会改变互联、内存、风扇和任务性能，整机功率不能仅由卡数比推算。")
    section("散热、温湿度与全年结果")
    p("公共网络和独立存储的 IT 功率按清单另加，得到 P_IT,h。散热需求 Q_h≈P_IT,h；使用当地逐小时温度及露点/相对湿度、冷却技术参数计算 P_cool,h，再形成 P_fac,h=P_IT,h+P_cool,h+P_other,h。")
    p("五类配置均需匹配的完整单元功率曲线，台数/柜数、总 CPU/GPU 数量和功率份额留空。DGX H100 的 10.2 kW 为最大铭牌，GB200 约 120 kW 为官方近似整柜描述；二者均不作为运行参考功率。既有 8.43 kW 实测属于另一台 AMD HGX 节点。[2,4,12]",small)

    chapter("5 需求来源与核验状态")
    table([["来源", "需求模式与记录", "本工作区的使用范围"],
        ["Lisa [5]", "科研超算；秒级作业与公开 30 秒监测", "约 160 万作业已核验；CPU 有 309 个完整日，完整功率监测未进入本次重标定"],
        ["Acme [6]", "大模型开发；CPU/GPU 分配、预训练、评估、微调等", "62413 条作业；90 个完整日。GPU 作业也有 CPU 分配字段，仍须核对利用率口径"],
        ["ASI [7]", "生产 AI；逐小时训练、在线/离线推理、开发及优先级", "当前核验相对日 109 的 24 小时；库存和业务类型可用于生产 GPU 需求分析"],
        ["Alibaba 2018 [8,15]", "CPU 在线服务与批处理混部", "仅核对官方说明与 schema；本次未下载，待下载校验和字段实值核验"]], [.18,.39,.43])
    p("C20 仅配 Lisa 的 CPU 作业与 Alibaba CPU 混部。R-GB200 仅配 Acme、ASI 的 GPU 类需求，仍需核实向 Grace/Blackwell 的性能、显存及同机 CPU 占用映射。配对表示研究设计适用性，demand_readiness 保留实际核验进度。",small)
    section("ASI 的具体作用")
    p("ASI 补充生产运营中的业务构成与小时变化，提供归属 GPU 小时、服务器 GPU 库存、CPU 容量与请求利用率等字段；也能按高/低优先级区分候选任务。它不直接给出 CPU 物理插槽数量、逐任务允许延期时间或抢占许可。")
    p("因此，ASI 用于选择生产需求路径和柔性候选范围；单机 2∶1、2∶2、2∶8 的设备参数来自厂商配置。任务类型、历史低优先级和小时归属量分别保留其来源含义。")
    p("已有 ASI 单日重复 365 次形成的年曲线属于方法演示，不能据此判断全年生产业务的季节变化。Lisa、Acme、ASI 是本工作区已经详细核验的三套主样本；其他公开数据按其适用问题单独评估。",small)

    chapter("6 其他可用于本研究的公开数据")
    table([["来源", "可补充的问题", "证据与当前处理状态"],
        ["Microsoft Philly [9]", "训练作业、同服务器 CPU/GPU 共同变化；可帮助区分宿主与 GPU 需求", "公开任务运行尝试与每分钟 CPU/GPU 利用率、GPU 库存；本次核对官方字段，未下载计算"],
        ["Alibaba CPU 2018 [8]", "CPU 在线服务与批处理混部，任务依赖与两类业务的资源需求", "约 4000 台、8 天，提供机器/容器与批处理表；本次核对说明，未下载计算"],
        ["Google 2019 PowerData [10]", "核对资源需求如何映射到电力域功率，验证功率模型", "57 个电力域，与对应 Borg 数据有关联；本次核对说明，未提取或拟合"]], [.25,.37,.38])
    p("数据是否适用，取决于研究对象、任务粒度、资源分母、CPU/GPU 共时关系、实测功率和服务权限。任务记录适合研究执行和调度，利用率记录适合研究动态占用，功率记录适合标定用电；三者需要按标识与时间边界连接。")
    p("Philly 的公开数据覆盖 2017 年提交的训练任务，硬件和业务年代与 H100 不同；可用来检验联合需求形状，迁移到新设备仍需性能映射。Google 的电力域功率也需核对其表计边界，不能直接当成单节点 CPU/GPU 功率。",small)
    chapter("7 8760 小时生成规则与已有结果")
    p("源任务先按真实运行区间积分，得到同一时刻的 CPU 核心小时、GPU 小时或利用率；确认同机联合需求后，使用相同的时间映射扩展两条路径。不能把互不对应的独立 CPU 业务曲线直接当成 GPU 服务器的宿主占用。")
    table([["现有源窗口", "2025 年 8760 小时的重放方法", "说明"],
        ["Lisa 309 日", "1 个完整窗口＋前 56 日", "保留连续资源形状，样本外季节性未知"],
        ["Acme 90 日", "4 个完整窗口＋前 5 日", "CPU/GPU 同相位映射，保留业务关系"],
        ["ASI 1 日", "同一天重复 365 次", "方法演示，跨日变化未识别"]], [.23,.42,.35])
    p("规则为 j(h)=(h+φ) mod L；H=8760（平年）或 8784（闰年）。保存每小时源索引、循环边界与起始相位；使用目标年逐小时天气重新计算冷却。年电量 E=Σ_hP_h×1 小时，年度 PUE=ΣP_fac/ΣP_IT。")
    p("已有 20261026-v2 年度报告提供 16 条、140160 条小时负荷记录及 490 组相位检查。它们属于既有纯设备池参考模型，与本说明五类配置分别保存。新版年曲线需要联合需求、完整单元功率标定，以及适用的冷却参数。")
    p("本次交付为五类实际配置的参数表和 52 个设计单元，未计算新年曲线。Alibaba 2018 尚未下载，不设置其年度重放长度。旧版 20261027 的全部文件 SHA-256 逐一核对并保持原样。",small)

    chapter("8 冷却方式与气象耦合待办")
    p("scenario-matrix 的每个单元显式保存 cooling_type：C20/C21/C22/C28 为 air（44 个单元），R-GB200 为 direct_liquid（8 个单元）。现有冷却/气象耦合模块按风冷机房设计，GB200 必须另建并验证液冷参数。本次仅登记待办，未实现新冷却模型。")
    p("GB200 的液体经过冷板直接冷却 CPU/GPU，部分网络、存储和电源部件仍需风冷。因此 direct_liquid 表示主要芯片采用直接液冷，残余风冷热负荷仍需纳入设施计量。[12,13]")
    table([["待办", "需要核定的内容"],
        ["供液与换热边界", "相较风冷机房可采用更高供液温度；实际供回液温度、流量及换热效率须有官方规格或测量依据"],
        ["干冷器与冷却塔", "干冷器主要受干球温度影响，蒸发冷却塔受湿球温度影响；按实际散热方案建立不同气象响应"],
        ["冷却电功率", "分别核定 CDU、泵、风机、残余风冷及可能的机械制冷功率，避免与整柜 IT 重复计量"],
        ["校准与验证", "核对整柜工作负载、供液条件、气象覆盖和适用范围；取得数据后验证独立参数"]], [.27,.73])
    p("本次供液温度、流量、热量液冷份额、COP/PUE 和气象响应系数均未赋值。GB200 单元的 cooling_model_status 明确为待建独立液冷耦合；不能直接应用现有风冷机房系数输出其设施负荷。",small)

    chapter("9 柔性权限与控制设计")
    p("柔性权限独立于规模、单机数量配置和需求导向。对同一任务集合，可以设置固定执行、允许延后启动、允许暂停恢复或允许调整功率/资源的控制范围。")
    table([["控制设置", "需要的依据", "求解后重新计算"],
        ["延后启动", "数据就绪、完成期限、运行时长和依赖", "CPU/GPU 同时占用及资源约束"],
        ["暂停与恢复", "检查点、最小连续运行、恢复时长和额外能耗", "完整运行量与恢复开销"],
        ["调整功率或资源", "设备与任务的性能关系、最小并行度、内存和服务时延", "完成工作量、功率和冷却"]], [.22,.43,.35])
    p("训练、推理、CPU 标签和任务优先级支持候选分类；可延期时长和暂停权限按实际服务约束或明确的控制设计填写，未知值保留为空。")
    p("在同一服务器上，CPU 与 GPU 可能共同参与一项任务，不能将其独立平移到互不对应的小时。调度改变执行路径后，重新计算联合资源、整机 IT、散热与气象相关冷却。")
    p("本次交付为服务器与整柜配置情境及参数说明，未求解任务级全年调度或确认新增的业务权限。",small)

    chapter("10 来源、参数文件与复现")
    p("C21/C22/C28 与主样本沿用 2026-10-27 核验；C20、GB200 及 Alibaba 2018 说明/字段于 2026-10-28 核对官方原文。Lenovo 为支持选件构成的研究选配，DGX H100 和 GB200 为官方整机/整柜架构；市场占有率未核定。",small)
    for label,url in REFERENCES:
        p(f'<a href="{url}" color="#1C5675">{escape(label)}</a>',small)
        markdown[-1] = f"{label} [官方或原始来源]({url})"
    section("参数与结果文件")
    p("server-configurations.csv/json 记录 5 种配置，scenario-matrix.csv/json 提供 52 个允许配对单元。research_unit 和 per_unit 字段明确服务器/机柜；兼容旧字段 per_node 仅用于服务器，GB200 对应字段为空。unit_count、rack_count、node_count 和运行参考功率均保持为空。",small)
    p("在项目根目录运行：<br/>python analysis/build_server_count_scenarios.py<br/>参数核验与生成文件 SHA-256 见 manifest.json、validation.json。既有年度算法与结果见 results/scenario-load-methods-20261026-v2/。",small)

    def footer(canvas,doc):
        canvas.saveState()
        canvas.setFont("YaHei",8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawRightString(A4[0]-18*mm,11*mm,str(doc.page))
        canvas.restoreState()

    pdf = OUT/f"{STEM}.pdf"
    SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=16*mm,
        bottomMargin=18*mm,title="CPU/GPU服务器与整柜情境及需求来源说明",author="数据中心研究工作区").build(story,onFirstPage=footer,onLaterPages=footer)
    (OUT/f"{STEM}.md").write_text("\n\n".join(markdown),encoding="utf-8")
    with fitz.open(pdf) as doc:
        for i,page in enumerate(doc):
            page.get_pixmap(matrix=fitz.Matrix(1.35,1.35)).save(TEMP/f"page-{i+1:02d}.png")
        pages = len(doc)
        (TEMP/"document-text.txt").write_text("\n".join(page.get_text() for page in doc),encoding="utf-8")
        links = [link["uri"] for page in doc for link in page.get_links() if link.get("uri")]
        if pages != 10 or len(set(links)) != len(REFERENCES):
            raise ValueError("页数或来源链接数量与报告设计不一致")
        for page in doc:
            for word in page.get_text("words"):
                if not (0<=word[0]<=word[2]<=page.rect.width and 0<=word[1]<=word[3]<=page.rect.height):
                    raise ValueError("PDF 存在越界文字")
    table_readback = pd.read_csv(OUT/"server-configurations.csv")
    matrix_readback = pd.read_csv(OUT/"scenario-matrix.csv")
    if table_readback.gpu_devices_per_unit.tolist()!=[0,1,2,8,72] or table_readback.cpu_packages_per_unit.tolist()!=[2,2,2,2,36]:
        raise ValueError("五种基本单元数量与设计不一致")
    if table_readback.cpu_cores_per_unit.tolist()!=[64,64,64,112,2592]:
        raise ValueError("CPU 核心数量与选定型号不一致")
    servers = table_readback[table_readback.research_unit.eq("server")]
    racks = table_readback[table_readback.research_unit.eq("rack")]
    if servers.rack_units_per_node.tolist()!=[2,2,2,8] or not servers.server_form_factor.eq("rackmount").all():
        raise ValueError("服务器形态与厂商机架规格不一致")
    if len(racks)!=1 or racks.iloc[0].configuration_id!="R-GB200" or racks.iloc[0].vendor_approx_rack_power_w!=120000:
        raise ValueError("GB200 整柜边界或官方近似功率不一致")
    rack = racks.iloc[0]
    if rack.compute_trays_per_unit*rack.cpu_packages_per_compute_tray!=36 or rack.compute_trays_per_unit*rack.gpu_devices_per_compute_tray!=72:
        raise ValueError("GB200 计算托盘与整柜数量不一致")
    if rack.cpu_cores_per_package*rack.cpu_packages_per_unit!=2592 or rack.nvlink_switch_trays_per_unit!=9 or rack.nvlink_switches_per_tray!=2:
        raise ValueError("GB200 CPU 核心或 NVLink 交换托盘组成不一致")
    for field in ("cpu_packages_per_node","gpu_devices_per_node","cpu_cores_per_node","rack_units_per_node",
            "rack_units_per_unit","rack_height_mm","rack_width_mm","rack_depth_mm","max_nameplate_node_w"):
        if not racks[field].isna().all():
            raise ValueError(f"GB200 不适用或未核定字段应为空：{field}")
    for config in catalog:
        subset = matrix_readback[matrix_readback.configuration_id.eq(config["configuration_id"])]
        for field in ("research_unit","cooling_type","cpu_packages_per_unit","gpu_devices_per_unit","cpu_cores_per_unit","rack_units_per_unit",
                "cpu_packages_per_node","gpu_devices_per_node","cpu_cores_per_node","rack_units_per_node","cooling_model_status"):
            valid = subset[field].isna().all() if config[field] is None else subset[field].eq(config[field]).all()
            if not valid:
                raise ValueError(f"情境表与设备参数不一致：{config['configuration_id']} {field}")
        for size in (1,10,100,500):
            demand_subset = subset[subset.reference_it_mw.eq(size)]
            if set(demand_subset.demand_source)!=set(PAIRINGS[config["configuration_id"]]) or len(demand_subset)!=len(PAIRINGS[config["configuration_id"]]):
                raise ValueError(f"配置与需求允许配对不一致：{size}MW {config['configuration_id']}")
    for field in ("measured_reference_unit_w","measured_reference_node_w","cpu_it_power_share","gpu_it_power_share"):
        if not table_readback[field].isna().all() or not matrix_readback[field].isna().all():
            raise ValueError("未识别的功率参数必须保留为空")
    for field in ("unit_count","node_count","rack_count","total_cpu_packages","total_cpu_cores","total_gpu_devices"):
        if not matrix_readback[field].isna().all():
            raise ValueError(f"未标定规模累计字段必须为空：{field}")
    if len(matrix_readback)!=52 or not matrix_readback.scenario_id.is_unique:
        raise ValueError("情境表行数、唯一性或台数状态不一致")
    if matrix_readback.groupby("reference_it_mw").size().to_dict()!={1:13,10:13,100:13,500:13}:
        raise ValueError("每档规模应有 13 个允许组合")
    if matrix_readback.cooling_type.value_counts().to_dict()!={"air":44,"direct_liquid":8}:
        raise ValueError("冷却类型应为 44 个风冷和 8 个直接液冷单元")
    alibaba = matrix_readback[matrix_readback.demand_source.eq("Alibaba Cluster Trace 2018")]
    if len(alibaba)!=4 or not alibaba.demand_readiness.str.contains("本次未下载").all():
        raise ValueError("Alibaba 2018 应如实标记未下载核验")
    for stem, key, current in (("server-configurations","configuration_id",catalog),("scenario-matrix","scenario_id",scenarios)):
        previous_rows = json.loads((PREVIOUS/f"{stem}.json").read_text(encoding="utf-8"))
        by_id = {row[key]:row for row in current}
        for old in previous_rows:
            if any(by_id[old[key]][field]!=value for field,value in old.items()):
                raise ValueError(f"原有配置或 36 个单元的既有字段改变：{old[key]}")
    baseline = PROJECT/"results/scenario-load-methods-20261026-v2/算力中心_实际配置与8760小时负荷计算说明_20261026_v2.pdf"
    if sha256(baseline)!="b587cfb06a344d632392b18e11b82416c9c0e3d3c6f2076745858250ebba2eb9":
        raise ValueError("既有年度报告与输入版本不一致")
    if previous_hashes!={p.name:sha256(p) for p in sorted(PREVIOUS.iterdir()) if p.is_file()}:
        raise ValueError("旧版文件 SHA-256 在生成期间发生变化")
    validation = {"status":"passed", "checks":["device_counts", "core_counts", "rackmount_and_rack_units", "rack_tray_composition",
        "scenario_hardware_consistency", "allowed_demand_pairings", "13_cells_per_scale", "52_unique_cells", "44_air_8_direct_liquid",
        "Alibaba_2018_not_downloaded", "unknown_power_and_counts_preserved", "unknown_mechanical_dimensions_preserved",
        "original_36_cells_and_hardware_preserved", "previous_version_all_file_SHA256_preserved",
        "CSV_JSON_consistency", "source_links", "PDF_bounds", "annual_baseline_preserved"],
        "pages":pages, "configurations":5, "design_cells":52, "source_links":len(set(links)),
        "cooling_type_counts":matrix_readback.cooling_type.value_counts().to_dict(),
        "cells_per_scale":13, "previous_version_files_sha256":previous_hashes,
        "scientific_scope":"design_matrix_only; no_new_power_calibration_hourly_loads_or_liquid_cooling_model",
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
    (OUT/"README.md").write_text(f"""# CPU/GPU 服务器与整柜情境（20261028）

[Markdown 说明]({STEM}.md)与[PDF 说明]({STEM}.pdf)提供五种配置及允许配对的 52 个设计单元。

| 配置 | 基本单元 | CPU∶GPU | 需求配对 | 四档合计 | cooling_type |
| --- | --- | --- | --- | --- | --- |
| C20 纯 CPU | SR650 V3 完整服务器，2U | 2∶0 | Lisa CPU 作业、Alibaba Cluster Trace 2018 | 8 | air |
| C21 | SR650 V3 完整服务器，2U | 2∶1 | Lisa、Acme、ASI | 12 | air |
| C22 | SR650 V3 完整服务器，2U | 2∶2 | Lisa、Acme、ASI | 12 | air |
| C28 | DGX H100 完整服务器，8U | 2∶8 | Lisa、Acme、ASI | 12 | air |
| R-GB200 基本纯 GPU | GB200 NVL72 完整机柜 | 36∶72 | Acme、ASI | 8 | direct_liquid |
| 合计 | 1 / 10 / 100 / 500 MW | 每档 13 个允许组合 | 4×（9＋2＋2） | 52 | 44 air＋8 direct_liquid |

C20 采用双路 Xeon Gold 6448Y，每颗 32 核、225 W TDP，共 64 核；不装计算 GPU。厂商支持选配依据见 [Lenovo 产品指南](https://lenovopress.lenovo.com/lp1601-thinksystem-sr650-v3-server)标准规格和第 19 表。管理控制器集成显示不计入计算 GPU，完整内存/存储/风扇/电源 BOM 与运行功率仍待核定。

R 表示 rack，R-GB200 按整柜计量。每柜有 18 个计算托盘（每个 2 颗 Grace＋4 个 Blackwell B200）及 9 个 NVLink 交换托盘（每个 2 个 NVSwitch），36 颗 Grace 每颗 72 核，共 2592 核。GPU 显存整柜为官方约数 13.4 TB HBM3E；双 GPU Superchip 为 372 GB，单 GPU 的 186 GB 由其除以 2 得到。规格见 [NVIDIA 产品页](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)、[硬件指南](https://docs.nvidia.com/dgx/dgxgb200-user-guide/hardware.html)和[参考架构](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-gb200/latest/dgx-superpod-components.html)。

GPU 服务器需要宿主 CPU。“基本纯 GPU”定义为 CPU 仅承担主机及 GPU 配套工作、电耗以 GPU 为主的研究形态，实际 CPU/GPU 电力份额保留待测。整柜计算 CPU 不包括 DPU/BMC 或交换管理处理器。

硬件指南的整柜功耗约 120 kW 单独记为 `vendor_approx_rack_power_w`；任务条件与测量边界未明确。所查 [NVIDIA 数据表](https://resources.nvidia.com/en-us-dgx-systems/dgx-superpod-gb200-datasheet)及硬件指南未列外部机械尺寸，整柜高/宽/深和高度 U 数留空，待官方机械规格。不得用计算托盘与交换托盘的 U 数之和代替整柜高度。

- [设备参数 CSV](server-configurations.csv)与[JSON](server-configurations.json)：`research_unit=server/rack`、统一的 `per_unit` 数量字段、规格、来源和核验日期。旧 `per_node` 字段仅用于服务器，GB200 对应字段为空。
- [52 个设计单元 CSV](scenario-matrix.csv)与[JSON](scenario-matrix.json)：每个单元具有 `cooling_type`、`demand_readiness` 和功率/冷却模型状态。`scenario_id` 沿用 `规模MW_配置编号_来源首词小写`；原有 36 个标识与既有字段保留。
- [生成核验](validation.json)与[文件 SHA-256](manifest.json)：含旧版全部文件的逐项 SHA-256，旧版 [20261027](../server-count-scenarios-20261027/README.md)保持原样。

Alibaba 2018 只用于 C20 的 CPU 云混部设计。本次只核对[官方说明](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md)与 [schema](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/schema.txt)，未下载原始数据；下载、校验与字段实值核验仍待完成。Lisa、Acme、ASI 的既有核验状态保留，向新硬件的联合占用与性能映射仍需核对。

GB200 的 CPU/GPU 冷板必须接入直接液冷，部分网络、存储及电源部件仍保留风冷。现有冷却/气象模块按风冷机房设计，液冷单元需另建并校准供液温度、CDU/泵/风机和残余风冷热量参数；干冷器与冷却塔分别受干球、湿球条件影响。本次仅登记待办，未实现新模型或赋予供液温度、流量、COP/PUE、气象系数数值。

五类配置的匹配整机/整柜运行功率均未标定。`measured_reference_unit_w`、`measured_reference_node_w`、CPU/GPU 功率份额、`unit_count`/`node_count`/`rack_count` 及累计硬件数量均为空。10.2 kW 为 DGX H100 最大铭牌，约 120 kW 为 GB200 厂商描述，均不作为运行参考功率。未计算新增年曲线；[既有年度工作包](../scenario-load-methods-20261026-v2/README.md)按原版本使用。

在项目根目录运行 `python analysis/build_server_count_scenarios.py` 可生成本版参数、说明、渲染图片及机器核验；图片保存于已忽略的 `.codex-temp/server-count-scenarios-20261028/`。来源核验日期为 2026-10-28，旧配置与样本按 2026-10-27 已核验版本保留。
""",encoding="utf-8")
    manifest = {"document":pdf.name,"pages":pages,"checked_date":"2026-10-28", "ratio_basis":"physical_CPU_package_and_GPU_device_counts_per_explicit_research_unit",
        "configurations":5,"design_cells":52,"new_hourly_curves_computed":0,
        "previous_version":"server-count-scenarios-20261027", "previous_version_files_sha256":previous_hashes,
        "demand_pairings":PAIRINGS, "cooling_type_counts":validation["cooling_type_counts"],
        "new_liquid_cooling_model_implemented":False, "Alibaba_2018_raw_data_downloaded":False,
        "references":[{"label":label,"url":url} for label,url in REFERENCES],
        "source_code_sha256":sha256(Path(__file__)),
        "files_sha256":{p.name:sha256(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!="manifest.json"}}
    (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"pages":pages,"configurations":5,"design_cells":52,"pdf_sha256":sha256(pdf)},ensure_ascii=False))


if __name__ == "__main__":
    main()
