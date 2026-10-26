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
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from compute_load.model import HARDWARE, sha256


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data/processed/task-trace-audit-20261026"
META = PROJECT / "data/metadata/task-trace-audit-20261026"
OUT = PROJECT / "results/task-driven-load-report-20261026"
TEMP = PROJECT / ".codex-temp/task-driven-load-report-20261026"
STEM = "CPU_GPU算力中心_任务需求与柔性约束负荷报告_20261026"
WIDTH = A4[0] - 36 * mm
CPU_COLOR, GPU_COLOR = "#2166AC", "#C75B12"
TYPE_NAMES = {"training": "训练", "online_inference": "在线推理", "offline_inference": "离线推理",
              "dev": "开发", "other": "其他", "unknown": "未知"}


def configure_figures():
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 10,
                         "axes.labelsize": 10, "axes.titlesize": 11, "legend.fontsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "savefig.dpi": 300})


def save_figure(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def curve_ecdf(ax, values, label, color, linestyle="-"):
    values = np.sort(np.asarray(values)[np.asarray(values) > 0])
    x = np.geomspace(1, max(values[-1], 2), 600)
    y = np.searchsorted(values, x, side="right") / len(values) * 100
    ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=1.6)
    ax.set_xscale("log")
    ax.set_ylim(0, 101)
    ax.set_xlabel("作业运行时长（秒，对数坐标）")
    ax.set_ylabel("累计作业比例（%）")
    ax.axvline(3600, color="#777777", linewidth=.8, linestyle=":")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, loc="lower right")


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    audit = json.loads((META / "asi-acme-validation.json").read_text(encoding="utf-8"))
    lisa_meta = json.loads((META / "lisa-summary.json").read_text(encoding="utf-8"))
    for name, expected in audit["outputs_sha256"].items():
        if sha256(DATA / name) != expected:
            raise ValueError(f"报告输入与已核验版本不同：{name}")
    lisa = pd.read_parquet(DATA / "lisa/lisa-tasks.parquet")
    acme = pd.read_parquet(DATA / "acme-tasks.parquet")
    gpu = acme.loc[acme.hardware.eq("gpu") & acme.interval_valid].copy()
    gpu["allocated_gpu_hours"] = gpu.runtime_s * gpu.gpu_num / 3600
    type_table = gpu.groupby("type").agg(valid_jobs=("task_id", "size"),
        allocated_gpu_hours=("allocated_gpu_hours", "sum"), median_runtime_s=("runtime_s", "median"))
    type_table["job_share_pct"] = type_table.valid_jobs / type_table.valid_jobs.sum() * 100
    type_table["gpu_hour_share_pct"] = type_table.allocated_gpu_hours / type_table.allocated_gpu_hours.sum() * 100
    type_table.to_csv(OUT / "acme-gpu-type-summary.csv", encoding="utf-8-sig")
    np.testing.assert_allclose(type_table.allocated_gpu_hours.sum(), audit["acme"]["resource_integrals"]["gpu"]["integrated_resource_hours"])
    asi = pd.read_csv(DATA / "asi-hourly-type-priority.csv")
    asi_hour = pd.read_csv(DATA / "asi-hourly-resource-demand.csv")
    asi_types = asi.groupby("job_type_public").agg(gpu_hours=("used_gpu_hours", "sum"), observations=("pod_observations", "sum"))
    asi_types["gpu_hour_share_pct"] = asi_types.gpu_hours / asi_types.gpu_hours.sum() * 100
    asi_types.to_csv(OUT / "asi-gpu-type-summary.csv", encoding="utf-8-sig")
    lisa_hour = pd.read_parquet(DATA / "lisa/lisa-hourly-core-demand.parquet")
    lisa_five = pd.read_parquet(DATA / "lisa/lisa-5min-core-demand.parquet")
    cpu_hour = lisa_hour.loc[lisa_hour.hardware_group.eq("cpu_partition")].copy()
    cpu_hour["day"] = cpu_hour.source_time.dt.floor("D")
    daily = cpu_hour.groupby("day").allocated_core_hours.agg(["sum", "count"])
    daily = daily.loc[daily["count"].eq(24)]
    daily["mean_allocated_cores"] = daily["sum"] / 24
    daily.to_csv(OUT / "lisa-cpu-daily-core-demand.csv", encoding="utf-8-sig")
    mean_daily, std_daily = daily.mean_allocated_cores.mean(), daily.mean_allocated_cores.std(ddof=1)
    cpu_hour["week"] = cpu_hour.source_time.dt.floor("D") - pd.to_timedelta(cpu_hour.source_time.dt.dayofweek, unit="D")
    weeks = cpu_hour.groupby("week").allocated_core_hours.agg(["sum", "count"])
    weeks = weeks.loc[weeks["count"].eq(168)]
    week = (weeks["sum"] - weeks["sum"].median()).abs().idxmin()
    windows = {}
    for kind in ("cpu", "gpu"):
        for size in (1, 10, 100, 500):
            windows[(kind, size)] = pd.read_parquet(DATA / f"power-interface/{kind}/{size}MW.parquet")
    power_rows = []
    for (kind, size), frame in windows.items():
        power_rows.append({"hardware": kind, "reference_it_mw": size, "hours": len(frame),
                           "mean_it_mw": frame.it_mw.mean(), "cooling_mwh": frame.cooling_mw.sum(),
                           "facility_mwh": frame.facility_mw.sum(), "peak_facility_mw": frame.facility_mw.max(),
                           "energy_weighted_pue": frame.facility_mw.sum() / frame.it_mw.sum()})
    pd.DataFrame(power_rows).to_csv(OUT / "power-24h-summary.csv", index=False, encoding="utf-8-sig")
    stats = {"lisa_cpu_complete_days": len(daily), "lisa_cpu_daily_mean_allocated_cores": float(mean_daily),
             "lisa_cpu_daily_coefficient_of_variation": float(std_daily / mean_daily),
             "lisa_cpu_daily_p10_p50_p90": daily.mean_allocated_cores.quantile([.1, .5, .9]).to_dict(),
             "lisa_display_week_source_start": str(week), "acme_gpu_type_summary": type_table.to_dict("index"),
             "asi_type_summary": asi_types.to_dict("index"), "power_24h_summary": power_rows}
    (OUT / "report-statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return audit, lisa_meta, lisa, acme, gpu, type_table, asi, asi_hour, daily, cpu_hour, lisa_five, week, windows, stats


def figures(lisa, acme, gpu, type_table, asi, asi_hour, daily, cpu_hour, lisa_five, week, windows):
    configure_figures()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    curve_ecdf(axes[0], lisa.loc[lisa.hardware_group.eq("cpu_partition"), "runtime_s"], "Lisa CPU 分区", CPU_COLOR)
    curve_ecdf(axes[0], acme.loc[acme.hardware.eq("cpu") & acme.interval_valid, "runtime_s"], "Acme CPU 作业", "#69A9D0", "--")
    curve_ecdf(axes[1], lisa.loc[lisa.hardware_group.eq("gpu_partition"), "runtime_s"], "Lisa GPU 分区", GPU_COLOR)
    curve_ecdf(axes[1], gpu.runtime_s, "Acme GPU 作业", "#E7A36A", "--")
    axes[0].set_title("a  CPU 作业数量分布")
    axes[1].set_title("b  GPU 作业数量分布")
    save_figure(fig, "01-task-runtime-distributions")
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.1), layout="constrained")
    axes[0].plot(daily.index, daily.mean_allocated_cores, color=CPU_COLOR, linewidth=1)
    axes[0].set_ylabel("日均分配核心数")
    axes[0].set_title("a  Lisa CPU 分区完整日期")
    selected = lisa_five.loc[lisa_five.hardware_group.eq("cpu_partition") & lisa_five.source_time.ge(week) & lisa_five.source_time.lt(week + pd.Timedelta(days=7))]
    h = cpu_hour.loc[cpu_hour.source_time.ge(week) & cpu_hour.source_time.lt(week + pd.Timedelta(days=7))]
    axes[1].plot((selected.source_time - week).dt.total_seconds()/3600, selected.allocated_core_hours * 12,
                 color="#90BADA", linewidth=.65, label="5 分钟平均")
    axes[1].plot((h.source_time - week).dt.total_seconds()/3600, h.allocated_core_hours,
                 color=CPU_COLOR, linewidth=1.25, label="小时平均")
    axes[1].set(xlabel="选定周的相对小时", ylabel="平均分配核心数", title="b  接近完整周资源小时中位数的一周")
    axes[1].legend(frameon=False, loc="lower center", ncol=2)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.set_ylim(bottom=0)
    save_figure(fig, "02-lisa-cpu-resource-profile")
    order = [name for name in ("Pretrain", "Eval", "SFT", "Debug", "Other") if name in type_table.index]
    names = {"Pretrain": "预训练", "Eval": "评估", "SFT": "微调", "Debug": "开发调试", "Other": "其他"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.1), layout="constrained")
    for ax, column, title in zip(axes, ("job_share_pct", "gpu_hour_share_pct"), ("a  有效执行区间的作业条数", "b  分配 GPU 小时")):
        ax.bar([names[name] for name in order], type_table.loc[order, column], color=GPU_COLOR, width=.65)
        ax.set(title=title, ylabel="份额（%）", ylim=(0, 100))
        ax.grid(axis="y", alpha=.2)
    save_figure(fig, "03-acme-gpu-type-composition")
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), layout="constrained")
    pivot = asi.groupby(["hour", "job_type_public"]).used_gpu_hours.sum().unstack(fill_value=0)
    types = [name for name in TYPE_NAMES if name in pivot]
    palette = ["#C75B12", "#2166AC", "#56A6A3", "#A786B9", "#999999", "#D5D5D5"]
    axes[0].stackplot(pivot.index, *[pivot[name] for name in types], labels=[TYPE_NAMES[name] for name in types], colors=palette)
    axes[0].set(xlabel="相对日 109 的小时", ylabel="该小时归属 GPU 小时")
    axes[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.25))
    axes[1].plot(asi_hour.hour, asi_hour.requested_fraction * 100, color="#8E4A17", label="申请量 / 库存")
    axes[1].plot(asi_hour.hour, asi_hour.used_gpu_hour_fraction * 100, color=CPU_COLOR, label="归属 GPU 小时 / 库存")
    axes[1].axhline(100, color="#777777", linestyle=":", linewidth=.8)
    axes[1].set(xlabel="相对日 109 的小时", ylabel="份额（%）", ylim=(0, 130))
    axes[1].legend(frameon=False, loc="lower left", ncol=2)
    for ax in axes:
        ax.set_xlim(0, 23)
        ax.grid(axis="y", alpha=.2)
    save_figure(fig, "04-asi-gpu-hourly-types")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), layout="constrained")
    for ax, kind, color in zip(axes, ("cpu", "gpu"), (CPU_COLOR, GPU_COLOR)):
        anchors = pd.read_csv(PROJECT / f"analysis/compute_load/assets/{kind}-measured-power-anchors.csv")
        if kind == "gpu":
            anchors = anchors.loc[~anchors.measurement.str.contains("no CPU", regex=False)]
        ax.plot(anchors.load_fraction * 100, anchors.node_power_w / 1000, "o-", color=color, markersize=4)
        ax.set(xlabel="基准相对负载或 GPU 利用率（%）", ylabel="整机电功率（kW）", title=f"{'a  CPU' if kind == 'cpu' else 'b  GPU'} 参考节点")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
    save_figure(fig, "05-reference-node-power")
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.5), layout="constrained")
    for ax, kind, title in zip(axes, ("cpu", "gpu"), ("a  CPU 参考路径", "b  GPU 参考路径")):
        frame = windows[(kind, 10)]
        ax.stackplot(np.arange(len(frame)), frame.it_mw, frame.cooling_mw, frame.other_mw,
                     labels=["IT", "冷却", "其他配套"], colors=[CPU_COLOR if kind == "cpu" else GPU_COLOR, "#61A799", "#B7B7B7"])
        ax.set(xlabel="与真实天气配对后的相对小时", ylabel="电功率（MW）", title=title, xlim=(0, 23))
        ax.grid(axis="y", alpha=.2)
    for ax in axes:
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    save_figure(fig, "06-reference-facility-power")


def build_report(audit, lisa_meta, type_table, asi_types, stats):
    pdfmetrics.registerFont(TTFont("YaHei", "C:/Windows/Fonts/msyh.ttc"))
    pdfmetrics.registerFont(TTFont("YaHeiBold", "C:/Windows/Fonts/msyhbd.ttc"))
    pdfmetrics.registerFontFamily("YaHei", normal="YaHei", bold="YaHeiBold", italic="YaHei", boldItalic="YaHeiBold")
    body = ParagraphStyle("Body", fontName="YaHei", fontSize=10.3, leading=16.5, wordWrap="CJK", spaceAfter=8)
    small = ParagraphStyle("Small", parent=body, fontSize=8.5, leading=12.7, textColor=colors.HexColor("#444444"), spaceAfter=7)
    heading = ParagraphStyle("Heading", parent=body, fontName="YaHeiBold", fontSize=15, leading=22, spaceAfter=12)
    sub = ParagraphStyle("Sub", parent=body, fontName="YaHeiBold", fontSize=11.2, leading=17, spaceBefore=6, spaceAfter=7)
    title = ParagraphStyle("Title", parent=heading, fontSize=21, leading=31, spaceAfter=12)
    cell = ParagraphStyle("Cell", parent=body, fontSize=8.4, leading=12.3, spaceAfter=0)
    head = ParagraphStyle("Head", parent=cell, fontName="YaHeiBold")
    formula = ParagraphStyle("Formula", parent=body, leftIndent=10, spaceBefore=3, spaceAfter=10)
    story, markdown = [], []

    def paragraph(text, style=body):
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
        content = [[Paragraph(escape(str(value)), head if i == 0 else cell) for value in row] for i, row in enumerate(rows)]
        result = Table(content, colWidths=[WIDTH * width for width in widths], hAlign="LEFT", repeatRows=1)
        result.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EFF3")),
            ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#8C9AA3")),
            ("LINEBELOW", (0, 1), (-1, -1), .3, colors.HexColor("#D9DFE1")),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]))
        story.extend([result, Spacer(1, 8)])
        markdown.extend(["| " + " | ".join(map(str, rows[0])) + " |", "| " + " | ".join(["---"] * len(rows[0])) + " |"])
        markdown.extend(["| " + " | ".join(map(str, row)) + " |" for row in rows[1:]])
        markdown.append("")

    def figure(name, caption):
        with PILImage.open(OUT / f"{name}.png") as image:
            height = WIDTH * image.height / image.width
        story.append(Image(str(OUT / f"{name}.png"), width=WIDTH, height=height))
        story.append(Paragraph(caption, small))
        markdown.extend([f"![{caption}]({name}.png)", ""])

    paragraph("CPU GPU 算力中心<br/>任务需求与柔性约束负荷报告", title)
    markdown.insert(0, "# CPU GPU 算力中心任务需求与柔性约束负荷报告\n")
    paragraph("Lisa · AcmeTrace · Alibaba ASI | 1、10、100、500 MW IT 参考规模 | 2026 年 10 月 26 日", small)
    chapter("1 研究对象与计算范围", first=True)
    paragraph("本报告以公开生产作业记录构建 CPU/GPU 资源需求，保留每项作业的提交、启动、结束和分配数量，再将执行区间累计为 5 分钟及小时曲线。资源需求通过参考整机功率与气象相关配套模型转换成设施负荷；任务柔性按业务期限、暂停权限及服务质量分别判断。")
    table([["研究内容", "数据与方法", "当前证据范围"],
           ["CPU 超算需求", "Lisa CPU 分区作业与核心数", "已核对全部作业，生成 5 分钟与小时需求"],
           ["GPU 作业需求", "Acme Kalos 作业时间与卡数", "保留预训练、微调、评估等任务类别"],
           ["GPU 运营构成", "ASI 类型、优先级和小时资源", "核对完整一天；来源共覆盖 185 天"],
           ["设施电力需求", "参考节点功率与 ESIF 冷却响应", "四档规模的 8 条 24 小时接口示例"],
           ["任务柔性", "提交、完成期限、暂停和恢复条件", "公开记录缺少的业务权限保持未知"]], [.20, .40, .40])
    subsection("观测资源与参考用电的含义")
    paragraph("作业运行区间、分配核心数、卡数与类型统计来自记录。将这些资源量转换成另一配置的节点利用率，以及将既有冷却响应迁移到参考设施，均属于模型假设。报告中的设施功率用于说明相同计算方法在不同规模下的结果，不作为 Lisa、Acme 或 ASI 原机房的实测用电。")
    paragraph("任务标签已经支持更细的分类，但类别不能替代服务约束。训练任务可能要求连续运行，评估任务可能有很紧的期限，CPU 分区也可能包含交互式作业。小时能源模型与原始作业约束需要同时保留。")

    chapter("2 数据来源与时间粒度")
    table([["输入", "本报告读取范围", "原始粒度", "主要字段"],
           ["SURF Lisa [1]", "1596963 条唯一作业；CPU 分区 1395625 条，GPU 分区 201338 条", "作业到秒；发布的节点监测为 30 秒", "提交、启动、结束、节点、分区、核心数、状态"],
           ["Acme Kalos [2]", "62413 条作业；CPU 42506 条，GPU 19907 条", "时间戳到秒", "类型、分配 CPU/GPU 数、提交与执行时刻"],
           ["Alibaba ASI [3]", "相对日 109 完整 24 小时；2816548 条 GPU pod 小时观测", "1 小时", "类型、优先级、申请量、归属 GPU 小时、库存"]], [.18, .35, .20, .27])
    paragraph("Lisa 作业源时间从 2021 年 12 月 25 日的提交记录延伸至 2022 年 11 月 1 日的结束记录，文件没有附带时区。当前保留源时钟和相对小时，连接全球 UTC 气象前需要确认其时区。30 秒节点监测的粒度已由来源说明确认，完整监测文件尚未用于本报告的功率重标定。[1]")
    paragraph("Acme 本次使用 Kalos 文件，覆盖 2023 年 5 月 17 日至 8 月 16 日，保留来源时区并转换为 UTC。不能把整个 Acme 项目的半年覆盖期直接套用到这一单独文件。GPU 的 Eval 标签保持评估含义，未改成在线推理。[2]")
    paragraph("ASI 用相对日和小时表示时间，不发布实际日历日期。完整来源为 185 天、4440 小时；本报告的一天样本用来检验字段和小时接口，其任务份额不能解释为半年或全年均值。同一 pod 在多个小时出现时，各行代表不同的 pod 小时观测。[3]")
    subsection("时间与资源的质量控制")
    paragraph("Acme 的 3 条 GPU 作业没有结束时间，保留原记录并排除于完整执行区间累计。62410 条完整作业的原 duration 等于提交至结束的总时长，执行负荷统一按启动至结束计算。Lisa 全部作业标识唯一，运行与等待时长非负，原始文件 MD5 与官方发布值一致。")

    chapter("3 从作业记录累计到小时需求")
    paragraph("对作业 i，分配资源数量为 n_i，实际运行区间为 [s_i,e_i)。时间段 h 与该区间的交集决定本段占用。只要保留真实交集，持续几十秒的短作业也可以正确进入小时累计。")
    paragraph("W<sub>h</sub> = Σ<sub>i</sub> n<sub>i</sub> × overlap_seconds([s<sub>i</sub>,e<sub>i</sub>),h) / 3600", formula)
    paragraph("W_h 的单位为核心小时或分配 GPU 小时；除以时间段的小时长度，得到平均分配资源数量。例如 100 个核心运行 30 秒，贡献 0.8333 核心小时。运行时长不向上取整到整小时，也不将排队期间计为执行时间。")
    figure("01-task-runtime-distributions", "图1 作业运行时长的累计分布。左图为 CPU，右图为 GPU；按作业条数统计，零运行时长区间不进入对数坐标。灰色竖线为一小时。Lisa 区分分区，Acme 区分是否申请 GPU，各自标签不建立逐任务对应关系。")
    paragraph("CPU 与 GPU 均存在大量短作业。Lisa CPU 分区 90.62% 的作业少于一小时；Acme GPU 有效执行区间中该比例为 98.18%。小时曲线适合能源平衡，短任务的启动次序、连续执行和完成期限仍应在原始作业层面约束。")

    chapter("4 CPU 超算需求的时间特征")
    paragraph("Lisa CPU 分区的 1395625 条作业累计分配约 26040211 核心小时。作业运行时长中位数为 25 秒，90% 分位为 3280 秒，99% 分位约为 49460 秒。短作业占比较高，与少数长作业持有较多资源的情况可以同时存在。")
    figure("02-lisa-cpu-resource-profile", "图2 Lisa CPU 分区的资源需求。上图为完整日期的日均分配核心数；下图选择完整周资源小时最接近中位数的一周，展示 5 分钟和小时平均。日期与时段均采用源时钟，未解释为 UTC 或全球通用工作时间。")
    paragraph(f"在 {stats['lisa_cpu_complete_days']} 个完整日期中，日均分配核心数的平均值为 {stats['lisa_cpu_daily_mean_allocated_cores']:.1f}，日间变异系数为 {stats['lisa_cpu_daily_coefficient_of_variation']:.3f}。这一指标描述观测期内不同日期需求的离散程度；它不能单独说明负荷允许转移多少。")
    paragraph("分配核心数表示调度器授予的资源，未等于 CPU 实际忙碌核心数或完成的 FLOP。分区包含 normal、shared、shared_jupyter 等标签，这些标签可用于服务审阅分组，不能单独证明作业可延后。CPU 功率标定需要配对的节点利用率和整机功率。")

    chapter("5 GPU 作业类型与资源需求")
    figure("03-acme-gpu-type-composition", "图3 Acme Kalos 有效 GPU 作业的类型构成。左图为作业条数份额，右图为分配 GPU 小时份额，分母均仅包含具有有效执行区间的 GPU 作业。分配 GPU 小时由卡数乘实际运行时长得到，不能直接解释为实测电量或有效训练工作量。")
    type_names = {"Pretrain": "预训练", "Eval": "评估", "SFT": "微调", "Debug": "开发调试", "Other": "其他"}
    table([["GPU 任务类型", "有效作业条数", "分配 GPU 小时", "资源小时份额"],
           *[[type_names[k], f"{int(row.valid_jobs):,}", f"{row.allocated_gpu_hours:,.1f}", f"{row.gpu_hour_share_pct:.2f}%"]
             for k, row in type_table.loc[[k for k in type_names if k in type_table.index]].iterrows()]], [.25, .22, .29, .24])
    paragraph("任务条数与 GPU 小时给出了不同的业务结构。较短的评估作业可以占据多数提交记录，持续时间较长或并行规模较大的预训练作业则可能占据较多资源小时。因此，不能用提交条数比例分配训练、推理或柔性电量。")
    paragraph("本统计保留具有有效执行区间的各类终止状态，包括失败和取消产生的运行占用。它描述已发生的资源消耗，未等于成功完成的服务量。若优化模型要求完成相同服务，应另外定义成功任务、重试关系及质量要求。")

    chapter("5 GPU 作业类型与资源需求（续）")
    figure("04-asi-gpu-hourly-types", "图4 ASI 相对日 109 的 GPU 小时构成及申请与库存关系。上图按任务类型汇总该小时归属 GPU 小时；下图分开显示申请量和归属资源份额。卡数包含异构设备，类型份额是资源份额，不是电量份额或柔性份额。")
    paragraph("这一完整日内，在线推理、离线推理和训练的资源曲线可以分别观察。公开的优先级同时保留 HP、LP 与 Other。LP 表示来源中的低优先级或 spot 业务，尚未给出每项任务的检查点、最大等待或恢复成本。[3]")
    paragraph("样本的申请量/库存为 104.63% 至 116.62%。这个比值不能直接作为物理利用率。GPU 功率接口使用另行保留的归属 GPU 小时份额，并保留其分配资源代理的性质。SM 利用字段存在缺失及大于 100 的值，未被截断或补零。")
    paragraph("ASI 的小时记录能描述运营构成，但无法恢复小时内部的到达顺序、短时突发或暂停过程。秒级 GPU 作业可用 Acme 检查执行约束，小时构成可用 ASI 核对；两个来源需要分别解释，不能将其任务标识或业务权限直接连接。")

    chapter("6 柔性识别与服务约束")
    table([["允许的操作", "必需证据", "可用记录与限制"],
           ["延后启动并连续执行", "数据就绪时刻、最晚完成时刻、所需资源与运行时长", "秒级作业可约束执行区间；业务期限缺失"],
           ["暂停与恢复", "抢占权限、检查点间隔、恢复时间和额外工作量", "取消或失败记录不能证明可无损恢复"],
           ["改变资源或功率", "资源/功率与性能关系，以及服务质量要求", "分配数量与整机功率需要配对标定"],
           ["跨地点调度", "数据位置、传输、时延和接收余量", "当前数据没有完整迁移许可及代价"]], [.23, .39, .38])
    paragraph("三套公开记录均未给出可直接使用的逐任务业务截止时间和暂停权限。当前每项任务的柔性状态保持未知，训练、离线推理、微调和评估仅进入服务约束审阅候选。CPU 的硬件类别、GPU 的业务类别和历史排队不能替代用户允许的操作范围。")
    subsection("可复现的柔性情境")
    paragraph("若研究采用明确的完成期限 d_i、就绪时间 r_i 和固定执行时长 p_i，则连续作业满足 r_i ≤ s_i 且 s_i+p_i ≤ d_i，同时受核心、卡数、内存和并行规模约束。可暂停作业还必须计入检查点与恢复开销。不存在这些权限时，模型只重建原执行路径。")
    paragraph("允许额外延后 1、6 或 24 小时等数值应登记为研究情境参数。采用不晚于历史完成时间的对照，也需要说明这一服务基准属于研究定义，并检查依赖与数据就绪条件。历史已等待多久不能直接转换为可新增的等待额度。")
    subsection("小时能源优化与作业调度的连接")
    paragraph("在相同任务和服务要求下比较固定派发与柔性派发。每次调度后重新累计资源、计算 IT 功率及气象相关冷却，再进入供电平衡。任务连续执行与最小并行规模保留在作业层面，不能把每个小时的工作量任意拆分而仍声称满足原任务约束。")

    chapter("7 设备规模与节点功率")
    paragraph("1、10、100、500 MW 表示参考高负载 IT 预算，不包含外部冷却。CPU 节点采用双路 EPYC 7763 的 128 核心整机参考，最高测试功率 546 W；GPU 节点采用 8×H100 HGX，联合高负载参考为 8430 W。[4,5]")
    rows = [["IT 预算", "CPU 节点 / 核心", "GPU 节点", "H100 参考卡数"]]
    for size in (1, 10, 100, 500):
        cpu_nodes = int(size * 1e6 // HARDWARE["cpu"]["full_w"])
        gpu_nodes = int(size * 1e6 // HARDWARE["gpu"]["full_w"])
        rows.append([f"{size} MW", f"{cpu_nodes:,} / {cpu_nodes*128:,}", f"{gpu_nodes:,}", f"{gpu_nodes*8:,}"])
    table(rows, [.16, .38, .22, .24])
    figure("05-reference-node-power", "图5 参考节点的整机功率点。CPU 来自 SPECpower 测试，GPU 来自 H100 HGX 训练及联合压力观测。曲线按相邻实测点线性插值，分配资源份额到相应利用率仍需验证。[4,5]")
    paragraph("节点数量 N=floor(10⁶S/p_ref)。资源代理份额 u_t 输入节点功率 p(u_t)，计算 P_IT,t=N×p(u_t)/10⁶。宿主 CPU、内存和机内组件已经随整机功率计入；公共网络与存储如另行配置，需要保持同一计费和用电范围。")
    paragraph("500 MW GPU 参考预算对应 59311 个节点、474488 张参考 H100。按论文所列 10.2 kW/节点额定参照约为 604.97 MW，工程配电与冷却还要核对设计温度和冗余。不同代际卡数不能直接作为等吞吐量换算。")

    chapter("8 散热需求与温湿度相关负荷")
    paragraph("小时尺度忽略蓄热和额外建筑得热时，设备散热需求约等于 IT 电功率：Q_IT,t≈P_IT,t。散热需求采用热功率单位，设施电力需求按 IT、冷却和其他配套电功率相加。节点内风扇已随整机计入。")
    paragraph("P_facility,t = P_IT,t + P_cooling,t + P_other,t", formula)
    paragraph("冷却采用 ESIF 同时间 IT、室外温湿度和分项电功率标定的响应。令 C 为安装参考 IT 容量，q_t=P_IT,t/C；室外干球为 T_t、露点为 Td_t。每个分项 j 使用 P_j,t=CΣ_kβ_kjX_k,t，分项为室外冷却、HVAC、泵和其他配套。")
    table([["特征项", "表达式或作用"],
           ["固定配套与热负载", "1、q、q²"],
           ["干球响应", "q×max(T−3.969434,0)、q×max(T−20.685997,0)"],
           ["露点响应", "q×max(Td+8.236311,0)、q×max(Td−7.992787,0)"],
           ["低温固定响应", "max(−3.280123−T,0)"],
           ["湿度输入", "由当地温度与相对湿度计算露点，或直接提供露点"]], [.34, .66])
    paragraph("模型采用 5447 个配对小时，前 4775 小时拟合、最后 672 小时检验；温湿度模型的留出冷却 RMSE 为 17.754 kW，约为测试均值的 10.276%，随后全样本重拟合部署系数。校准干球范围约为 −19.31 至 38.13℃、露点约为 −29.42 至 17.68℃，q 为 0.2315 至 1。[6]")
    paragraph("这些检验只描述该冷却配置的时间留出表现。目标设施的 COP、供水温度、冷却设备启停和规模效率尚未识别。CPU 科研任务、GPU 训练和在线推理也需要对应功率标定；共享参考节点曲线不能证明三类业务具有相同的实际能效。")
    paragraph("外推时保留超标定天气和负载标记。相同任务放在不同网格可以用于比较气候引起的配套用电差异；它不能自动预测该地点的业务到达，也不能证明目标机房具有相同冷却设备。")

    chapter("9 四档规模的小时负荷示例")
    figure("06-reference-facility-power", "图6 10 MW IT 参考预算下的 24 小时设施负荷。CPU 采用 Acme CPU 资源小时最高日期的路径，并以观测并发峰值归一化；GPU 采用 ASI 相对日 109 的归属 GPU 小时/库存。两者与同一段真实天气按相对小时配对，属于接口验证情境。")
    rows = [["类型 / 预算", "平均 IT\nMW", "冷却电量\nMWh", "设施电量\nMWh", "最高设施\nMW"]]
    for row in stats["power_24h_summary"]:
        rows.append([f"{row['hardware'].upper()} / {row['reference_it_mw']} MW", f"{row['mean_it_mw']:.3f}",
                     f"{row['cooling_mwh']:.3f}", f"{row['facility_mwh']:.3f}", f"{row['peak_facility_mw']:.3f}"])
    table(rows, [.28, .18, .18, .19, .17])
    paragraph("CPU 示例的归一化分母为观测并发峰值 15360 核心，GPU 分母为各小时的异构卡库存。天气取网格 g05_r100_c592 的 2025 年 1 月 1 日 UTC 24 小时。两种业务路径不同，不能据此作同等服务的 CPU/GPU 能效比较。Lisa 是 CPU 任务结构的主要资料，当前示例尚未将其无时区时间直接连接气象。", small)

    chapter("10 年度累计 成本与全球网格接入")
    subsection("完整年度的构造和累计")
    paragraph("完整 UTC 日历年要求 8760 或 8784 个无缺口小时。电量 E_year=Σ_tP_tΔt，年度 PUE=全年设施电量/全年 IT 电量。任务层同时累计核心小时、GPU 小时和成功完成的服务量，并分别报告。")
    paragraph("Lisa 作业约覆盖十个月，Acme 本次文件约三个月，ASI 当前读取一天。它们足以验证任务到小时负荷的计算方法，尚未提供完整年度实测需求。全年重放需要明确时钟、工作日与季节映射、跨年作业、未完成任务及年末补做边界；不能把一天直接乘 365 作为实测全年。")
    subsection("规模与成本系数")
    paragraph("设备投资可按同配置节点数量乘单节点报价线性外推，年化费用再采用各资产寿命和折现条件。机房、供配电、制冷、公共网络与存储分别计价；需要额外算力余量时计入新增设备及空闲功耗。未经报价或工程证据支持的价格和规模效率保留为待定参数。")
    paragraph("固定派发与柔性派发完成相同服务，成本范围和候选资产保持一致。储能、风光、购电和任务调整费用按研究情境计入；数据缺少业务期限时，本报告不赋予默认可移峰比例，也不生成柔性净收益。")
    subsection("可复现模块与验证")
    paragraph("任务入口保留原始时间、类型、资源和状态，生成 5 分钟及小时需求；参考功率模块接收当地逐小时温湿度，按四档规模计算 IT、散热、冷却与设施总负荷。全球网格供电模型接收这些曲线时，保持 UTC、单位及服务边界一致。")
    table([["检查", "核验结果"],
           ["Lisa 文件及作业", "官方 MD5 一致；1596963 个唯一标识；运行与等待时长非负"],
           ["资源累计", "原始运行区间总量与 5 分钟、小时累计守恒"],
           ["独立计算", "48 个时间段通过逐作业交集核对；5 项无效输入被拒绝"],
           ["功率接口", "CPU/GPU 四档规模共 8 条各 24 小时；分项电功率守恒"],
           ["服务权限", "缺失截止时间与暂停权限保持未知"]], [.25, .75])

    chapter("参考资料与文件索引")
    references = [
        ("[1] Chu 等，Generic and ML Workloads in an HPC Datacenter，SURF Lisa 数据，Zenodo v3，2024。", "https://zenodo.org/records/13685426"),
        ("[2] Shanghai AI Lab，AcmeTrace 官方数据与字段；Hu 等，Characterization of Large Language Model Development in the Datacenter，NSDI 2024。", "https://github.com/InternLM/AcmeTrace"),
        ("[3] Alibaba Cluster Trace GPU v2026，ASI 公开字段、任务类型、优先级、资源及时间定义。", "https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/docs/schema.md"),
        ("[4] SPECpower_ssj2008，HPE ProLiant DL385 Gen10 Plus，双路 AMD EPYC 7763，2021 年公开结果。", "https://ftp.spec.org/power_ssj2008/results/res2021q1/power_ssj2008-20210221-01063.html"),
        ("[5] Latif 等，Empirical Measurements of AI Training Power Demand on a GPU-Accelerated Node，2024，Table I。", "https://arxiv.org/html/2412.08602v1"),
        ("[6] 研究工作区 CPU/GPU IT 散热与温湿度耦合负荷工作包，20261025-v4；ESIF 配对记录、冷却留出检验与部署系数。", "https://github.com/TuzzyT-Laiii/33/tree/main/dc-flexibility-project/results/compute-center-data-audit-20261022/coupled-power-weather-20261025-v4"),
    ]
    for label, url in references:
        paragraph(f'<a href="{url}" color="#1C5675">{escape(label)}</a>', small)
    subsection("报告数据和程序")
    table([["内容", "文件或目录"],
           ["报告图件与汇总", "results/task-driven-load-report-20261026/"],
           ["任务输入和读取", "analysis/task_trace_inputs.py；fetch_task_trace_samples.py"],
           ["作业核对", "analysis/audit_task_trace_sources.py；audit_lisa_tasks.py"],
           ["负荷模块", "analysis/compute_load/；docs/compute-load-module.md"],
           ["验证与来源", "data/metadata/task-trace-audit-20261026/"],
           ["源记录与派生任务", "data/raw/ 与 data/processed/ 下的 task-trace-audit-20261026/"],
           ["报告生成", "analysis/build_task_driven_load_report.py"]], [.28, .72])
    paragraph("报告与汇总按文件校验值固定输入版本。带任务标识的原始记录和派生表保留在研究数据目录，公开仓库保存报告、图件、生成代码和汇总核验。服务器工作目录为 /data3/dataset/ljt/global-grid-flexibility-20261023。", small)
    paragraph("报告中的任务规模、资源小时、气象及节点功率保留各自的单位和来源。完整年度的任务外推、逐任务服务合同及目标设施功率标定完成后，才能进一步把情境下的可调节潜力解释为特定业务和机房可实施的柔性。", small)

    def page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("YaHei", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawRightString(A4[0] - 18*mm, 11*mm, str(doc.page))
        canvas.restoreState()

    pdf = OUT / f"{STEM}.pdf"
    doc = SimpleDocTemplate(str(pdf), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=16*mm, bottomMargin=18*mm, title="CPU GPU 算力中心任务需求与柔性约束负荷报告",
                            author="数据中心研究工作区")
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    (OUT / f"{STEM}.md").write_text("\n\n".join(markdown), encoding="utf-8")
    with fitz.open(pdf) as document:
        texts = []
        for number, page in enumerate(document):
            page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35)).save(TEMP / f"page-{number+1:02d}.png")
            texts.append(page.get_text())
        (TEMP / "report-text.txt").write_text("\n\n".join(texts), encoding="utf-8")
        pages = len(document)
    record = {"report": pdf.name, "page_count": pages, "report_sha256": sha256(pdf),
              "source_code_sha256": sha256(Path(__file__)), "input_validation_sha256": sha256(META / "asi-acme-validation.json"),
              "lisa_validation_sha256": sha256(META / "lisa-summary.json"),
              "figures": 6, "power_examples": "8 cases × 24 hours", "permission_status": "unknown",
              "software": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
              "files_sha256": {path.name: sha256(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "report-manifest.json"}}
    (OUT / "report-manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main():
    prepared = prepare()
    audit, lisa_meta, lisa, acme, gpu, types, asi, asi_hour, daily, cpu_hour, five, week, windows, stats = prepared
    figures(lisa, acme, gpu, types, asi, asi_hour, daily, cpu_hour, five, week, windows)
    result = build_report(audit, lisa_meta, types, stats["asi_type_summary"], stats)
    print(json.dumps({"report": str(OUT / result["report"]), "pages": result["page_count"],
                      "statistics": stats}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
