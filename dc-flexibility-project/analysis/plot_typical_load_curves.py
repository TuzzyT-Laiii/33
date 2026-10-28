from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIGS = ("C20", "C21", "C22", "C28", "R-GB200")
TITLES = {"C20": "C20 · 双路 CPU", "C21": "C21 · 双路 CPU + 1 张 H100 PCIe",
          "C22": "C22 · 双路 CPU + 2 张 H100 PCIe", "C28": "C28 · DGX H100 八卡整机",
          "R-GB200": "GB200 NVL72 · 整柜 · 仅 IT 功率"}
COLORS = {"Lisa": "#167D8D", "Acme": "#4B60AE", "ASI": "#C27A24", "Alibaba": "#8A579D"}
LABELS = {"Lisa": "Lisa · 309 日资源迹", "Acme": "Acme · 90 日资源迹",
          "ASI": "ASI · 单日重复演示", "Alibaba": "Alibaba · CPU 使用率迹"}
FOOT = "功率参数：公开资料+假设（演示）"


def setup_style():
    font = Path("C:/Windows/Fonts/simhei.ttf")
    if not font.is_file():
        raise FileNotFoundError("需要覆盖中文字符的字体；当前设置为 C:/Windows/Fonts/simhei.ttf")
    font_manager.fontManager.addfont(str(font))
    name = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.family": name, "font.size": 11, "axes.titlesize": 12,
        "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "legend.frameon": False, "savefig.dpi": 300, "xtick.labelsize": 10, "ytick.labelsize": 10})
    return font


def reported(frame, key):
    return frame.it_mw if key == "R-GB200" else frame.facility_mw


def finish(fig, output, stem):
    fig.text(.985, .018, FOOT, ha="right", fontsize=10, color="#444444")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # 检查最终画布边界，避免紧裁切掩盖越界文字。
    outside = []
    for artist in fig.findobj(Text):
        if artist.get_visible() and artist.get_text().strip():
            box = artist.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
                outside.append(artist.get_text())
    fig.savefig(output / f"{stem}.png", dpi=300, facecolor="white")
    fig.savefig(output / f"{stem}.pdf", facecolor="white")
    plt.close(fig)
    return outside


def panel_grid(title, subtitle):
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    fig.subplots_adjust(left=.075, right=.98, bottom=.09, top=.845, wspace=.20, hspace=.48)
    fig.suptitle(title, y=.975, fontsize=18)
    fig.text(.5, .935, subtitle, ha="center", fontsize=11, color="#444444")
    handles = [Line2D([0], [0], color=COLORS[d], lw=2, label=LABELS[d]) for d in ("Lisa", "Acme", "ASI")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .917), ncol=3, fontsize=11)
    for i, ax in enumerate(axes.flat[:5]):
        ax.set_title(f"{chr(97+i)}  {TITLES[CONFIGS[i]]}", loc="left", pad=10)
        ax.grid(axis="y", color="#e5e5e5", lw=.65)
        ax.set_ylabel("功率（MW）")
        ax.set_ylim(0, 11.5)
    return fig, axes


def notes_panel(ax, week_start=None):
    ax.axis("off")
    lines = ["10 MW 为整机参考 IT 预算，设备数量向下取整。", "每个配置采用相同资源份额形状。",
             "跨设备吞吐量与服务量尚未统一。", "Lisa：CPU 需求；GPU 保持空载假设。",
             "Acme / ASI：宿主 CPU 活动含假设。", "GB200：仅 IT；液冷、设施功率和 PUE 待补。"]
    if week_start is not None:
        lines.insert(0, f"共同代表周起点：{week_start:%Y-%m-%d}（UTC）")
    ax.text(.02, .95, "\n\n".join(lines), transform=ax.transAxes, va="top", fontsize=11, linespacing=1.25)


def plot_all(frames, output, week_start):
    font = setup_style()
    outside = {}
    aggregate_rows = []
    fig, axes = panel_grid("典型工作日 · 24 小时负荷", "风冷单元显示设施功率；目标 2025 年周一至周五的 UTC 逐时均值")
    for i, key in enumerate(CONFIGS):
        ax = axes.flat[i]
        for (config, demand), frame in frames.items():
            if config != key:
                continue
            selected = frame.loc[frame.time_utc.dt.dayofweek.lt(5)].copy()
            selected["reported_mw"] = reported(selected, key)
            day = selected.groupby(selected.time_utc.dt.hour).reported_mw.mean()
            ax.plot(day.index, day.values, lw=1.9, color=COLORS[demand], ls="--" if key == "R-GB200" else "-")
            aggregate_rows.extend({"configuration_id": key, "demand": demand, "hour_utc": int(h), "reported_mw": float(v)} for h, v in day.items())
        ax.set_xlim(0, 23)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlabel("UTC 小时")
    notes_panel(axes.flat[5])
    outside["figure1"] = finish(fig, output, "figure1-typical-weekday")
    pd.DataFrame(aggregate_rows).to_csv(output / "typical-weekday.csv", index=False, encoding="utf-8-sig")

    first = next(iter(frames.values())).time_utc.iloc[week_start]
    fig, axes = panel_grid("典型一周 · 168 小时负荷", f"共同代表周 {first:%Y-%m-%d} 至 {(first + pd.Timedelta(hours=167)):%Y-%m-%d} · 保留逐时变化")
    for i, key in enumerate(CONFIGS):
        ax = axes.flat[i]
        for (config, demand), frame in frames.items():
            if config == key:
                values = reported(frame, key).iloc[week_start:week_start+168]
                ax.plot(np.arange(168), values, color=COLORS[demand], lw=1.6, ls="--" if key == "R-GB200" else "-")
        ax.set_xlim(0, 167)
        ax.set_xticks(np.arange(0, 168, 24), ["周一", "周二", "周三", "周四", "周五", "周六", "周日"])
        ax.set_xlabel("UTC 日期；每日起点为 00:00")
    notes_panel(axes.flat[5], first)
    outside["figure2"] = finish(fig, output, "figure2-typical-week")

    daily_rows = []
    fig, axes = panel_grid("全年负荷 · 日均与日内最大/最小包络", "8760 小时汇总到 365 日；阴影为日内范围，季节天气作用于风冷功率")
    for i, key in enumerate(CONFIGS):
        ax = axes.flat[i]
        for (config, demand), frame in frames.items():
            if config != key:
                continue
            selected = frame.copy()
            selected["reported_mw"] = reported(frame, key)
            daily = selected.groupby(selected.time_utc.dt.floor("D")).reported_mw.agg(["mean", "min", "max"])
            ax.fill_between(daily.index, daily["min"], daily["max"], color=COLORS[demand], alpha=.10, linewidth=0)
            ax.plot(daily.index, daily["mean"], color=COLORS[demand], lw=1.0, ls="--" if key == "R-GB200" else "-")
            cooling = selected.groupby(selected.time_utc.dt.floor("D")).cooling_mw.mean()
            if key != "R-GB200":
                ax.plot(cooling.index, cooling, color=COLORS[demand], lw=.7, ls=":", alpha=.95)
            daily_rows.append(daily.reset_index().assign(configuration_id=key, demand=demand, cooling_mean_mw=cooling.to_numpy()))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
        ax.set_xlim(pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC"))
        ax.set_xlabel("2025 年（UTC）；仅 IT 功率" if key == "R-GB200" else "2025 年（UTC）；底部点线为日均冷却功率")
    ax = axes.flat[5]
    weather = next(iter(frames.values()))
    daily_temp = weather.groupby(weather.time_utc.dt.floor("D")).temperature_c.agg(["mean", "min", "max"])
    ax.fill_between(daily_temp.index, daily_temp["min"], daily_temp["max"], color="#999999", alpha=.18)
    ax.plot(daily_temp.index, daily_temp["mean"], color="#555555", lw=1)
    ax.set_title("f  g05_r100_c592 · 2025 年 ERA5 干球温度", loc="left", pad=10)
    ax.set_ylabel("温度（℃）")
    ax.set_xlabel("2025 年（UTC）")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
    ax.set_xlim(pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC"))
    ax.grid(axis="y", color="#e5e5e5", lw=.65)
    outside["figure3"] = finish(fig, output, "figure3-annual-envelope")
    pd.concat(daily_rows, ignore_index=True).to_csv(output / "daily-envelope.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.8))
    fig.subplots_adjust(left=.065, right=.985, top=.78, bottom=.22, wspace=.23)
    fig.suptitle("同一需求在不同配置下的全年平均功率构成", y=.965, fontsize=18)
    fig.text(.5, .90, "参考 IT 预算均为 10 MW；跨配置比较资源份额形状，服务吞吐量尚未统一", ha="center", fontsize=11)
    components = [("it_mw", "IT 功率", "#536B8E"), ("cooling_mw", "冷却功率", "#62A69A"), ("other_mw", "其他设施用电", "#C5C7C9")]
    for i, demand in enumerate(("Lisa", "Acme", "ASI")):
        ax = axes[i]
        selected = [(key, frame) for (key, d), frame in frames.items() if d == demand]
        x = np.arange(len(selected))
        bottom = np.zeros(len(selected))
        for column, label, color in components:
            values = np.array([frame[column].mean() if key != "R-GB200" or column == "it_mw" else 0 for key, frame in selected])
            ax.bar(x, values, bottom=bottom, color=color, width=.65, label=label)
            bottom += values
        for j, (key, frame) in enumerate(selected):
            if key == "R-GB200":
                ax.text(j, bottom[j] + .22, "仅 IT\n液冷待补", ha="center", fontsize=10)
                ax.patches[j].set_hatch("///")
                ax.patches[j].set_edgecolor("#253447")
                ax.patches[j].set_linewidth(.6)
        ax.set_xticks(x, ["GB200" if key == "R-GB200" else key for key, _ in selected])
        ax.set_title(f"{chr(97+i)}  {LABELS[demand]}", loc="left", pad=12)
        ax.set_ylabel("全年平均功率（MW）")
        ax.set_ylim(0, 11.5)
        ax.grid(axis="y", color="#e5e5e5", lw=.65)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .10), ncol=3)
    fig.text(.5, .064, "其他设施用电包括机房辅助设备；GB200 仅显示 IT 部分。", ha="center", fontsize=10)
    outside["figure4"] = finish(fig, output, "figure4-power-components")
    return {"font_file": str(font), "dpi": 300, "outside_text": outside,
            "checks": {"figure_text_within_canvas": all(not labels for labels in outside.values()),
                       "four_png_and_pdf": all((output / f"{stem}.{ext}").is_file() for stem in (
                           "figure1-typical-weekday", "figure2-typical-week", "figure3-annual-envelope", "figure4-power-components") for ext in ("png", "pdf"))}}
