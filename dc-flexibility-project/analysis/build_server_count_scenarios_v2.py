import argparse
import copy
import json
from pathlib import Path
from xml.sax.saxutils import escape

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from compute_load.model import sha256


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / 'results/server-count-scenarios-20261028-v2'
TEMP = PROJECT / '.codex-temp/server-count-scenarios-20261028-v2'
OLD = PROJECT / 'results/server-count-scenarios-20261028'
META = PROJECT / 'data/metadata/research-gpu-demand-20261028'
STEM = '科研GPU需求与服务器配置矩阵_20261028_v2'
RESOURCES = ['gpu_allocation_hours', 'host_cpu_allocation_hours']
REFERENCES = [
    ('SURF Lisa v3 原文及文件校验值', 'https://zenodo.org/records/13685426'),
    ('MIT Supercloud 官方数据说明', 'https://github.com/MIT-AI-Accelerator/MIT-Supercloud-Dataset'),
    ('MIT 数据目录、许可及获取位置', 'https://registry.opendata.aws/dcc/'),
    ('Samsi 等 (2021)，The MIT Supercloud Dataset', 'https://doi.org/10.1109/HPEC49654.2021.9622850'),
]


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def archived_hashes():
    paths = ['server-count-scenarios-20261027', 'server-count-scenarios-20261028', 'typical-load-curves-20261028']
    return {str(p.relative_to(PROJECT)).replace('\\', '/'): sha256(p)
            for name in paths for p in sorted((PROJECT / 'results' / name).rglob('*')) if p.is_file()}


def make_figures(hourly, source_dir):
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 9,
                         'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    monday = hourly.source_time.iloc[0].normalize()
    monday += pd.Timedelta(days=(7 - monday.weekday()) % 7)
    starts = pd.date_range(monday, hourly.source_time.iloc[-1] - pd.Timedelta(hours=167), freq='7D')
    weeks = [hourly[hourly.source_time.ge(start) & hourly.source_time.lt(start + pd.Timedelta(days=7))] for start in starts]
    array = np.stack([week[RESOURCES].to_numpy() for week in weeks])
    scale = hourly[RESOURCES].quantile(.95).to_numpy()
    if (scale <= 0).any() or not np.isfinite(array).all():
        raise ValueError('代表周选择的资源数据无效')
    normalized = array / scale
    scores = ((normalized - normalized.mean(axis=0)) ** 2).mean(axis=(1, 2))
    chosen = int(scores.argmin())
    week = weeks[chosen].copy()
    selection = pd.DataFrame({'source_week_start': starts, 'joint_shape_distance': scores,
                              'selected': np.arange(len(starts)) == chosen})
    selection.to_csv(OUT / 'week-selection.csv', index=False)
    week.to_csv(source_dir / 'research-gpu-example-week.csv', index=False)
    replay = hourly.iloc[np.arange(8760) % len(hourly)].copy().reset_index(drop=True)
    replay['target_time'] = pd.date_range('2025-01-01', periods=8760, freq='h')
    replay['replay_cycle'] = np.arange(8760) // len(hourly)
    replay.to_csv(source_dir / 'research-gpu-annual-replay.csv', index=False)
    daily = replay.groupby(replay.target_time.dt.floor('D'))[RESOURCES].mean()
    daily['target_day'] = np.arange(1, 366)
    daily.to_csv(source_dir / 'research-gpu-annual-daily.csv')
    names = ['GPU 分配数（卡）', '宿主 CPU 分配数\n（Slurm 调度单位）']
    palette = ['#247A9E', '#BA662D']
    for stem, values, x, xlabel in [
        ('research-gpu-week', week, np.arange(168), '源时间：周一至周日（逐小时）'),
        ('research-gpu-annual-daily', daily, np.arange(1, 366), '2025 年示意日序（源窗口按顺序循环）')]:
        fig, axes = plt.subplots(2, 1, figsize=(7.25, 2.95), sharex=True, layout='constrained')
        for index, (ax, column, label, color) in enumerate(zip(axes, RESOURCES, names, palette)):
            ax.plot(x, values[column], color=color, linewidth=.9)
            ax.set_ylabel(label, fontsize=8)
            ax.set_ylim(bottom=0)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', color='#dddddd', linewidth=.5)
            ax.text(.01, .91, chr(97 + index), transform=ax.transAxes, weight='bold')
            if stem.endswith('week'):
                ax.set_xlim(0, 168)
                ax.set_xticks(np.arange(0, 169, 24), ['周一', '周二', '周三', '周四', '周五', '周六', '周日', ''])
            else:
                boundary = len(hourly) / 24
                ax.axvspan(boundary + .5, 365.5, color='#e9e9e9', alpha=.8, zorder=0)
                ax.axvline(boundary + .5, color='#888888', linewidth=.7, linestyle='--')
                empty = daily.gpu_allocation_hours.eq(0)
                ax.scatter(daily.loc[empty, 'target_day'], daily.loc[empty, column], facecolors='none', edgecolors='#666666', s=18, zorder=3)
                ax.set_xlim(1, 365)
        axes[-1].set_xlabel(xlabel)
        title = ('MIT 科研 GPU 作业：' + str(starts[chosen].date()) + ' 起的一周' if stem.endswith('week')
                 else 'MIT 科研 GPU 作业：全年日均分配占用（灰色区间为重复重放）')
        fig.suptitle(title, fontsize=10)
        fig.savefig(OUT / (stem + '.png'), dpi=300)
        fig.savefig(OUT / (stem + '.pdf'))
        plt.close(fig)
    np.testing.assert_array_equal(replay[RESOURCES].to_numpy(), hourly[RESOURCES].to_numpy()[np.arange(8760) % len(hourly)])
    return {'selected_week_start': str(starts[chosen]), 'selected_week_hours': len(week),
            'annual_replay_hours': len(replay), 'daily_rows': len(daily),
            'observed_days': len(hourly) // 24, 'repeated_days': 365 - len(hourly) // 24,
            'zero_allocation_days_observed': int(hourly.groupby(hourly.source_time.dt.floor('D')).gpu_allocation_hours.sum().eq(0).sum()),
            'annual_replay_sha256': sha256(source_dir / 'research-gpu-annual-replay.csv')}


def build_report(catalog, scenarios, audit, figures):
    lisa, mit = audit['lisa'], audit['adopted']
    pdfmetrics.registerFont(TTFont('YaHei', 'C:/Windows/Fonts/msyh.ttc'))
    pdfmetrics.registerFont(TTFont('YaHeiBold', 'C:/Windows/Fonts/msyhbd.ttc'))
    body = ParagraphStyle('Body', fontName='YaHei', fontSize=9.6, leading=15, wordWrap='CJK', spaceAfter=8)
    small = ParagraphStyle('Small', parent=body, fontSize=8.2, leading=12.4)
    heading = ParagraphStyle('Heading', parent=body, fontName='YaHeiBold', fontSize=17, leading=24, spaceAfter=14)
    cell = ParagraphStyle('Cell', parent=small, spaceAfter=0)
    width = A4[0] - 36 * mm
    story, markdown = [], []

    def p(text, style=body):
        story.append(Paragraph(text, style))
        markdown.append(text)

    def chapter(title, first=False):
        if not first:
            story.append(PageBreak())
        p(title, heading)
        markdown[-1] = '## ' + title

    def table(rows, fractions):
        cells = [[Paragraph(escape(str(v)), cell) for v in row] for row in rows]
        block = Table(cells, colWidths=[width * f for f in fractions], repeatRows=1)
        block.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                   ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E9EFF3')),
                                   ('LINEBELOW', (0, 0), (-1, -1), .3, colors.HexColor('#CCD5DB')),
                                   ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
        story.extend([block, Spacer(1, 9)])
        markdown.extend(['| ' + ' | '.join(map(str, rows[0])) + ' |', '| ' + ' | '.join(['---'] * len(rows[0])) + ' |'])
        markdown.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows[1:])

    chapter('科研 GPU 需求与服务器配置矩阵', True)
    p('2026-10-28 v2 · 五种设备配置 · 1 / 10 / 100 / 500 MW IT 预算 · 52 个设计单元', small)
    table([['配置', 'CPU∶GPU / 单元', '允许需求', '每档 / 四档'],
           ['C20', '2∶0 / 台', 'Lisa CPU、Alibaba CPU 2018', '2 / 8'],
           ['C21', '2∶1 / 台', 'MIT 科研 GPU、Acme、ASI', '3 / 12'],
           ['C22', '2∶2 / 台', 'MIT 科研 GPU、Acme、ASI', '3 / 12'],
           ['C28', '2∶8 / 台', 'MIT 科研 GPU、Acme、ASI', '3 / 12'],
           ['R-GB200', '36∶72 / 柜', 'Acme、ASI', '2 / 8'],
           ['合计', '服务器 / 整柜分别计量', '每档 13 个允许组合', '13 / 52']], [.15, .24, .43, .18])
    p('C21/C22/C28 的科研需求采用 MIT Supercloud 202201 中明确请求 GPU 的执行作业；C20 继续采用 Lisa CPU。GPU 科研需求的新标识为“规模MW_配置_mit-supercloud”，例如 10MW_C28_mit-supercloud。其余 40 个设计单元的全部既有字段保持。')
    p('Lisa GPU 核查', heading)
    p(f'只读读取官方 Slurm 压缩包和硬件表，官方 MD5 均通过。{lisa["gpu_partition_rows"]:,} 条 GPU 分区作业从 {lisa["gpu_first_start"]} 至 {lisa["gpu_last_end"]}，中间含 {lisa["full_calendar_days"]} 个完整日时间窗；可核验 GPU 卡小时的天数为 0。')
    p('作业表包含 id、submit_date、start_date、end_date、node、nodetypes、numnodes、numcores、state。没有 GPU 请求数、分配数、利用率或功率字段。GPU 分区的宿主 CPU 核心小时可计算；硬件表 210 张卡的静态库存无法确定共享分区各作业的卡数。', small)
    p('官方另提供 30 秒监测：prom_table_cleaned.zip 为 10,242,740,338 字节，prom_slurm_joined.zip 为 6,895,904,211 字节。本次没有下载这些监测包；现有本地 data/ 和服务器来源目录未发现 Lisa 监测原文。完整监测的 GPU 字段、有效覆盖日与联合占用仍待核验。', small)

    chapter('公开科研 GPU 来源与质量')
    p('采用 MIT Supercloud 202201 的完整 Slurm 调度表，配套官方 TRES 映射、README、LICENSE，以及两份小型 GPU 监测 CSV。原始文件仅下载到 pku-test，本机保留审计及小时聚合输入，原始数据与逐作业处理表未提交仓库。')
    table([['项目', '核验结果与口径'],
           ['原始下载量', f'{mit["acquisition"]["downloaded_bytes"]:,} 字节；slurm-log.csv 单表 103,165,427 字节；全部 S3 ETag 及 SHA-256 已核验/记录'],
           ['规模', f'本文件 {mit["total_job_rows"]:,} 行；明确请求且获配 GPU 的执行作业 {mit["selected_gpu_execution_rows"]:,} 条'],
           ['GPU 数量', 'tres_req 与 tres_alloc 的 1001=gpu:tesla、1002=gpu:volta；小时占用使用实际分配量'],
           ['宿主 CPU', '仅累计上述同批 GPU 作业 tres_alloc 中的 1=cpu；CPU 为 Slurm 调度单位，物理核心/线程口径待核实'],
           ['运行区间', 'time_start 至 time_end，epoch 秒；GPU 请求/分配均需大于零，暂停累计须为零'],
           ['筛选', f'排除 {mit["gpu_allocation_without_request"]:,} 条获配整节点 GPU 但未请求 GPU 的记录；无效区间及暂停记录见审计 JSON'],
           ['源窗口', f'{mit["full_day_begin"]} 至 {mit["full_day_end_exclusive"]}（右端不含），{mit["full_calendar_days"]} 日 / {mit["observed_hours"]} 小时'],
           ['质量', f'{mit["nonzero_gpu_days"]} 日记录到 GPU 分配，另 {figures["zero_allocation_days_observed"]} 日未记录到；空闲与漏记尚无法区分'],
           ['监测样本', '100 ms GPU 利用率/功率等字段见 source-audit.json；两份小样本仅核验 schema，不外推集群活跃利用率'],
           ['许可', '数据 CC BY-NC-ND 4.0，采用下载对象 LICENSE；代码仓库的 MIT 许可不替代数据许可']], [.23, .77])
    p('官方 README 概述约 460,497 条作业、98,177 条 GPU 请求；本次对象实值为 395,914 行、98,176 条 GPU 请求。计数采用已校验下载对象，差异保留为版本质量信息。日志采集完整性和漏报率未独立证明；取消或失败作业在有效执行区间内仍有资源占用。', small)
    p('该样本支持科研 GPU 资源分配需求形状。Tesla/Volta 向 H100 的性能、显存适配、活跃利用率与整机功率响应待核定；分配占用不直接作为计算单元活跃利用率。本次未推定时区、服务截止期限或抢占权限。', small)

    chapter('逐小时占用与年度日均示意')
    for stem, caption in [
        ('research-gpu-week', f'图 1. 同一批科研 GPU 作业的一周分配占用，起点 {figures["selected_week_start"]}。从完整周中，按 GPU 与宿主 CPU 的联合标准化小时形状选择最接近平均周模板的一周。每小时资源小时数除以 1 小时即平均分配数。'),
        ('research-gpu-annual-daily', f'图 2. 将 {mit["full_calendar_days"]} 日源窗口同相位顺序循环到 2025 年 8760 小时，再逐日平均。前 {mit["full_calendar_days"]} 日对应一个源窗口，灰色 {figures["repeated_days"]} 日重复窗口前段。空心点为未记录到 GPU 分配的日期；未补值。该图用于展示重放方法，样本外季节性未知。')]:
        story.append(Image(str(OUT / (stem + '.png')), width=width, height=width * 2.95 / 7.25))
        markdown.append(f'![{stem}]({stem}.png)')
        p(caption, small)
    p('本图仅为新科研需求的资源占用示意，未重算典型 IT/冷却功率曲线。旧典型曲线按其原输入版本归档；其中 C21/C22/C28 的 Lisa CPU 曲线不再作为当前科研 GPU 需求输入。', small)

    chapter('文件、复现与后续边界')
    p('server-configurations.csv/json 从 20261028 版逐字段继承设备参数；scenario-matrix.csv/json 保存 52 个设计单元。每个科研 GPU 单元新增 demand_key、demand_basis、demand_observed_days、demand_observed_hours 及来源审计路径。运行功率和设备数量保持原标定状态，风冷 44 个、直接液冷 8 个。')
    p('data/metadata/research-gpu-demand-20261028/source-audit.json 保存 Lisa 字段、原文校验、MIT 下载来源与体积、请求/分配筛选、覆盖质量及独立积分检查。data/processed/research-gpu-demand-20261028/ 保存逐小时占用、同相位全年示意重放和日均表，按数据目录规则忽略。')
    p('服务器原始目录：' + mit['acquisition']['raw_data_location'], small)
    p('在服务器运行 acquire_research_gpu_source.py，只获取必要调度表、映射和两份小样本；再运行 audit_research_gpu_demand.py，参数分别指定 --lisa-dir、--mit-dir、--output-dir。将审计 JSON 和小时 CSV 恢复到项目 data/processed/research-gpu-demand-20261028/ 后，在项目根目录运行 python analysis/build_server_count_scenarios.py。依赖见 analysis/requirements-research-gpu-demand.txt。', small)
    p('独立审阅：validation.json 核验配对、标识、字段继承、CSV/JSON 一致、原文与旧结果校验和、资源小时守恒、96 次实际区间交集检查以及图表导出。manifest.json 登记输入、代码和交付物 SHA-256。PDF 每页渲染到 .codex-temp/server-count-scenarios-20261028-v2/。', small)
    p('后续需要核实日志漏报、时区和 CPU 调度单位，补足实际利用率及功率监测，并建立向 H100 的工作量与联合功率映射。Alibaba 2018 全表仍待完成下载核验；本次没有使用未完成输入。冷却模型由独立任务维护。', small)
    p('原始与官方来源', heading)
    for label, url in REFERENCES:
        p(f'<a href="{url}" color="#1C5675">{escape(label)}</a>', small)
        markdown[-1] = f'[{label}]({url})'
    p(mit['citation'], small)

    def footer(canvas, doc):
        canvas.setFont('YaHei', 8)
        canvas.drawRightString(A4[0] - 18 * mm, 11 * mm, str(doc.page))

    pdf = OUT / (STEM + '.pdf')
    SimpleDocTemplate(str(pdf), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=16 * mm, bottomMargin=17 * mm, title=STEM).build(story, onFirstPage=footer, onLaterPages=footer)
    (OUT / (STEM + '.md')).write_text('\n\n'.join(markdown), encoding='utf-8')
    with fitz.open(pdf) as document:
        if len(document) != 4:
            raise ValueError(f'报告应为四页，实际为 {len(document)} 页')
        for i, page in enumerate(document):
            page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4)).save(TEMP / f'page-{i + 1:02d}.png')
            for word in page.get_text('words'):
                if word[0] < 0 or word[1] < 0 or word[2] > page.rect.width or word[3] > page.rect.height:
                    raise ValueError('PDF 文字超出页面')
    return pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, default=PROJECT / 'data/processed/research-gpu-demand-20261028')
    args = parser.parse_args()
    for path in (OUT, TEMP, META):
        path.mkdir(parents=True, exist_ok=True)
    before = archived_hashes()
    audit = json.loads((args.source_dir / 'source-audit.json').read_text(encoding='utf-8'))
    mit = audit['adopted']
    hourly_path = args.source_dir / 'research-gpu-observed-hourly.csv'
    if sha256(hourly_path) != mit['observed_hourly_sha256'] or not audit['lisa']['original_bytes_preserved']:
        raise ValueError('来源文件或 Lisa 原文保护核验未通过')
    hourly = pd.read_csv(hourly_path, parse_dates=['source_time'])
    if len(hourly) != mit['observed_hours'] or not np.isfinite(hourly[RESOURCES]).all().all() or hourly[RESOURCES].lt(0).any().any():
        raise ValueError('已审计小时需求无效')
    catalog = json.loads((OLD / 'server-configurations.json').read_text(encoding='utf-8'))
    original = json.loads((OLD / 'scenario-matrix.json').read_text(encoding='utf-8'))
    scenarios = copy.deepcopy(original)
    changed = []
    for row in scenarios:
        if row['configuration_id'] in ('C21', 'C22', 'C28') and row['demand_source'] == 'Lisa':
            changed.append({'old': row['scenario_id'], 'new': row['scenario_id'].removesuffix('_lisa') + '_mit-supercloud'})
            row.update({'scenario_id': changed[-1]['new'], 'demand_source': 'MIT Supercloud GPU',
                        'demand_orientation': '科研超算（GPU 作业）', 'demand_key': 'mit_supercloud_gpu',
                        'demand_basis': 'executed_GPU_allocation_hours_and_same_job_host_CPU_allocation',
                        'demand_observed_days': mit['full_calendar_days'], 'demand_observed_hours': mit['observed_hours'],
                        'demand_source_audit': 'data/metadata/research-gpu-demand-20261028/source-audit.json',
                        'demand_readiness': '明确请求 GPU 的执行作业及同作业宿主 CPU 分配已核验；采集完整性、活跃利用率及 H100 性能/功率映射待核定'})
    if len(changed) != 12 or len(scenarios) != 52 or len({r['scenario_id'] for r in scenarios}) != 52:
        raise ValueError('科研 GPU 替换数量或矩阵标识不一致')
    unchanged = [r for r in original if not (r['demand_source'] == 'Lisa' and r['gpu_devices_per_unit'] > 0)]
    lookup = {r['scenario_id']: r for r in scenarios}
    if len(unchanged) != 40 or any(lookup[r['scenario_id']] != r for r in unchanged):
        raise ValueError('本次范围外的单元发生变化')
    if any(r['demand_source'] == 'Lisa' and r['gpu_devices_per_unit'] > 0 for r in scenarios):
        raise ValueError('GPU 配置仍有 Lisa CPU 需求')
    for name, rows in [('server-configurations', catalog), ('scenario-matrix', scenarios)]:
        save_json(OUT / (name + '.json'), rows)
        pd.DataFrame(rows).to_csv(OUT / (name + '.csv'), index=False, encoding='utf-8-sig')
        readback = pd.read_csv(OUT / (name + '.csv')).astype(object)
        expected = pd.DataFrame(rows).astype(object)
        pd.testing.assert_frame_equal(readback.where(readback.notna(), None), expected.where(expected.notna(), None), check_dtype=False)
        if json.loads((OUT / (name + '.json')).read_text(encoding='utf-8')) != rows:
            raise ValueError('JSON 回读与内存结果不一致')
    figures = make_figures(hourly, args.source_dir)
    save_json(META / 'source-audit.json', audit)
    save_json(META / 'lisa-gpu-audit.json', audit['lisa'])
    pdf = build_report(catalog, scenarios, audit, figures)
    matrix = pd.DataFrame(scenarios)
    if matrix.groupby('reference_it_mw').size().to_dict() != {1: 13, 10: 13, 100: 13, 500: 13}:
        raise ValueError('每档规模的单元数量不符')
    if before != archived_hashes():
        raise ValueError('归档矩阵或典型曲线发生变化')
    validation = {'status': 'passed', 'design_cells': 52, 'configurations': 5, 'cells_per_scale': 13,
                  'checks': ['52_unique_cells', 'no_Lisa_CPU_on_GPU_configurations', '12_research_GPU_replacements',
                             '40_other_cells_unchanged', 'hardware_catalog_unchanged', 'CSV_JSON_readback',
                             '4_scales_13_cells_each', 'old_result_SHA256_preserved', 'Lisa_official_MD5',
                             'Lisa_original_bytes_preserved', 'MIT_S3_ETag', 'resource_hour_conservation',
                             '96_real_independent_interval_checks', 'same_phase_CPU_GPU_replay', '4_page_PDF_text_bounds'],
                  'cooling_type_counts': matrix.cooling_type.value_counts().to_dict(),
                  'source_quality': {k: mit[k] for k in ['selected_gpu_execution_rows', 'full_calendar_days', 'observed_hours', 'nonzero_gpu_days']},
                  'figures': figures, 'new_typical_power_curves_computed': 0,
                  'scientific_scope': 'GPU_requested_execution_allocation_demand; no_active_utilization_or_power_inference'}
    save_json(OUT / 'validation.json', validation)
    readme = f'''# 科研 GPU 需求与服务器配置矩阵（20261028-v2）

[Markdown 说明]({STEM}.md) / [PDF 说明]({STEM}.pdf)。采用 MIT Supercloud 科研 GPU 执行作业，五种配置、四档规模共 **52 个设计单元**。

| 配置 | 科研需求 | 其他需求 | 每档 | 四档 |
| --- | --- | --- | --- | --- |
| C20 | Lisa CPU | Alibaba CPU 2018（待完成下载核验） | 2 | 8 |
| C21 | MIT Supercloud GPU | Acme、ASI | 3 | 12 |
| C22 | MIT Supercloud GPU | Acme、ASI | 3 | 12 |
| C28 | MIT Supercloud GPU | Acme、ASI | 3 | 12 |
| R-GB200 | — | Acme、ASI | 2 | 8 |
| 合计 | 1 / 10 / 100 / 500 MW IT | 44 风冷＋8 直接液冷 | 13 | 52 |

Lisa 有 201,338 条 GPU 分区作业、308 个完整日时间窗；9 个作业字段没有 GPU 请求数、分配数、利用率或功率。节点库存无法补足作业卡数，可用 GPU 占用天数为 0。核查原始文件只读，官方 MD5 与前后 SHA-256 均通过；[证据](../../data/metadata/research-gpu-demand-20261028/lisa-gpu-audit.json)。官方独立监测包尚待获取核验。

采用 [MIT Supercloud 官方数据](https://registry.opendata.aws/dcc/) 202201 版。实际 Slurm 文件 {mit['total_job_rows']:,} 行，筛选 {mit['selected_gpu_execution_rows']:,} 条明确请求且获配 GPU 的执行作业；排除 {mit['gpu_allocation_without_request']:,} 条未请求 GPU 的整节点预留记录。源窗口为 {mit['full_day_begin']} 至 {mit['full_day_end_exclusive']}（右端不含），{mit['full_calendar_days']} 日、{mit['observed_hours']} 小时。{mit['nonzero_gpu_days']} 日有 GPU 分配记录，日志采集完整性与其余日期的空闲/漏报情况尚未证明。

GPU 与宿主 CPU 均来自同批作业的 `tres_alloc`，分别积分为卡小时和 Slurm CPU 资源小时；分配量不代表实测活跃利用率或功率。数据许可 **CC BY-NC-ND 4.0**，原始与逐作业表保留服务器；两份监测小样本只核验字段。所需下载总计 {mit['acquisition']['downloaded_bytes']:,} 字节（约 {mit['acquisition']['downloaded_bytes']/1024**2:.2f} MiB），全套约 TB 级的遥测未下载。来源、许可、字段及计数差异见[完整审计](../../data/metadata/research-gpu-demand-20261028/source-audit.json)。

![一周逐小时占用](research-gpu-week.png)

[一周示意 PDF](research-gpu-week.pdf)。起点 {figures['selected_week_start']}；代表周选择见 [week-selection.csv](week-selection.csv)。

![全年日均占用示意](research-gpu-annual-daily.png)

[全年日均示意 PDF](research-gpu-annual-daily.pdf)。{mit['full_calendar_days']} 日源窗口同相位循环为 365 日，灰色为重复段；空心点标记无 GPU 分配记录的日期。未补值、未添加季节变化。

[设备参数](server-configurations.json)逐字段继承旧版；[矩阵 CSV](scenario-matrix.csv) / [JSON](scenario-matrix.json)中科研 GPU 单元使用新标识 `规模MW_配置_mit-supercloud`。[validation.json](validation.json) / [manifest.json](manifest.json)保存验证与 SHA-256。

旧版 [20261028](../server-count-scenarios-20261028/README.md)、[20261027](../server-count-scenarios-20261027/README.md)及[典型曲线](../typical-load-curves-20261028/README.md)全部保持原字节。典型曲线中的 C21/C22/C28×Lisa CPU 属于原输入版本，当前科研 GPU 需求以后续统一重算结果为准。本次生成资源占用示意，未重算 IT/冷却曲线，未修改冷却模型。

复现：在服务器运行 `analysis/acquire_research_gpu_source.py` 与 `analysis/audit_research_gpu_demand.py`（各参数见 `--help`）；将 `source-audit.json` 与 `research-gpu-observed-hourly.csv` 恢复至 `data/processed/research-gpu-demand-20261028/`，执行 `python analysis/build_server_count_scenarios.py`。依赖为 `analysis/requirements-research-gpu-demand.txt`。逐小时及重放表保存在忽略的数据目录。验证执行 `python analysis/verify_research_gpu_demand.py`。

遗留：Lisa 监测、MIT 日志完整性/时区/CPU 单位、活跃利用率及 H100 性能与功率映射、Alibaba CPU 全表核验。服务截止时间及抢占权限保持未知。
'''
    (OUT / 'README.md').write_text(readme, encoding='utf-8')
    code_names = ['build_server_count_scenarios.py', 'build_server_count_scenarios_v2.py', 'acquire_research_gpu_source.py', 'audit_research_gpu_demand.py', 'task_trace_inputs.py', 'verify_research_gpu_demand.py']
    manifest = {'checked_date': '2026-10-28', 'base_commit': '8b7c6eb', 'design_cells': 52,
                'source_audit_sha256': sha256(META / 'source-audit.json'), 'hourly_input_sha256': sha256(hourly_path),
                'source_raw_data_location': mit['acquisition']['raw_data_location'],
                'scenario_id_replacements': changed, 'previous_result_files_sha256': before,
                'new_typical_power_curves_computed': 0, 'figures': figures,
                'source_code_sha256': {name: sha256(PROJECT / 'analysis' / name) for name in code_names},
                'files_sha256': {p.name: sha256(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'manifest.json'}}
    save_json(OUT / 'manifest.json', manifest)
    print(json.dumps({'design_cells': 52, 'research_gpu_cells': len(changed), 'source_days': mit['full_calendar_days'], 'pdf': str(pdf)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
