import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor


FEASIBLE = {'optimal', 'feasible_with_gap'}


def text(paragraph, content, *, size=11, bold=False, heading=False):
    run = paragraph.add_run(content)
    run.font.name = 'Arial'
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor.from_string('000000')
    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), '黑体' if heading else '宋体')
    return paragraph


def paragraph(doc, content, *, size=11, bold=False, after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.18
    return text(p, content, size=size, bold=bold)


def title(doc, content):
    p = doc.add_paragraph()
    p._p.get_or_add_pPr().get_or_add_pStyle().val = 'Heading1'
    p.paragraph_format.space_after = Pt(9)
    text(p, content, size=15, bold=True, heading=True)
    p.paragraph_format.page_break_before = True
    p.paragraph_format.keep_with_next = True


def table(doc, headers, rows, widths):
    t = doc.add_table(rows=0, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    borders = OxmlElement('w:tblBorders')
    for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        border = OxmlElement('w:' + side)
        for name, value in (('val', 'single'), ('sz', '4'), ('color', 'D9D9D9')):
            border.set(qn('w:' + name), value)
        borders.append(border)
    t._tbl.tblPr.insert_element_before(borders, 'w:tblLayout')
    for col, width in zip(t.columns, widths):
        col.width = Mm(width)
    for i, values in enumerate([headers, *rows]):
        row = t.add_row()
        row_props = row._tr.get_or_add_trPr()
        row_props.append(OxmlElement('w:cantSplit'))
        if i == 0:
            row_props.append(OxmlElement('w:tblHeader'))
        for j, (cell, value) in enumerate(zip(row.cells, values)):
            cell.width = Mm(widths[j])
            if i == 0:
                shade = OxmlElement('w:shd')
                shade.set(qn('w:fill'), 'E7EFF8')
                cell._tc.get_or_add_tcPr().append(shade)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            text(p, str(value), size=9.5, bold=i == 0)
    return t


def status(rows, stage):
    return sum(row[f'{stage}_status'] in FEASIBLE for row in rows)


def span(values, decimals=0):
    low, high = min(values), max(values)
    a, b = f'{low:,.{decimals}f}', f'{high:,.{decimals}f}'
    return a if a == b else f'{a}–{b}'


def update(doc, report):
    economic = [row for row in report['cases'] if row['target'] == 'economic']
    sensitivity = [row for row in report['cases'] if row['target'] == 'alpha100']
    as_of = datetime.fromisoformat(report['generated_at']).astimezone(timezone(timedelta(hours=8)))
    title(doc, '九 新建数据中心的算力与供能联合规划')
    paragraph(doc, f'八城市计算记录  {as_of:%Y年%m月%d日 %H:%M}（北京时间）', size=10)
    paragraph(doc, '八城市与 11 个情境分别使用原生任务窗口、2025 年城市气象及 10 MW 参考任务规模。'
              '每项比较保持任务量、原始业务分类、最迟完成时限、设备型号与成本参数一致。'
              '目标为整机设备与供能的等效年化总成本。')
    table(doc, ('对照', '任务时序', '机群台数', '供能'),
          [('A 原时序规划', '按原时序执行', '满足原时序的最少整机台数', '优化本地风光储和市电接入'),
           ('B 固定机群柔性规划', '按时限重排可调任务', '固定为 A', '按重排后的负荷优化'),
           ('C 算力与供能联合规划', '与 B 使用相同调度规则', '在 A 的上限内选择整数台数', '与任务和台数一并优化')],
          [46, 60, 71, 77])
    paragraph(doc, '刚性及未知任务保留原时段。可拆分批次在 48 小时内执行，CPU、GPU 和任务电量按同一比例移动。'
              '有起止记录的整体作业按相同起止小时分组，统一平移至多 12 小时，组内每个作业保持连续。'
              '越过观测边界或缺少可配对逐作业记录的负荷保留原时序。', after=7)
    paragraph(doc, '服务器台数同时满足 CPU 和 GPU 的逐小时资源需求，GPU 数由整机配置换算。'
              '设备减少后，按实际模型台数重算待机用电，再按城市冷却参数计算设施功率。'
              '台数基于小时资源与参考设备，缺少逐节点装箱、显存和互联验证。')
    paragraph(doc, 'B 到 C 是联合规划的增量收益，分为整机采购额、等效年化设备费用和供能费用变化。'
              'A 到 B 衡量固定机群规模下的调度收益，A 到 C 衡量总收益。'
              '原生窗口费用折算到年仅用于可比性。')

    title(doc, '八城市模型设备容量与配对状态')
    paragraph(doc, f'经济最优的 88 个城市与情境组合中，A、B、C 已有可行解的数量分别为 '
              f'{status(economic, "A")}、{status(economic, "B")}、{status(economic, "C")}。'
              f'100% 本地无碳匹配的 88 个组合中，三组分别为 '
              f'{status(sensitivity, "A")}、{status(sensitivity, "B")}、{status(sensitivity, "C")}。')
    rows = []
    for scenario in dict.fromkeys(row['scenario'] for row in economic):
        cases = [row for row in economic if row['scenario'] == scenario]
        base = [row for row in cases if row['A_status'] in FEASIBLE]
        paired = [row for row in cases if 'joint_annual_saving_usd' in row]
        base_units = span([row['A_servers'] for row in base]) if base else '待计算'
        base_gpus = span([row['A_gpus'] for row in base]) if base else '待计算'
        units = span([row['servers_avoided'] for row in paired]) if paired else '待计算'
        gpus = span([row['gpus_avoided'] for row in paired]) if paired else '待计算'
        saving = span([row['joint_annual_saving_usd'] / 10000 for row in paired], 1) if paired else '待计算'
        rows.append((scenario, base_units, base_gpus, f'{len(paired)} / 8', units, gpus, saving))
    table(doc, ('情境', 'A 服务器（台）', 'A GPU（张）', '配对城市', '少配服务器（台）',
                '少配 GPU（张）', '净节省（万美元/年）'), rows, [26, 36, 32, 31, 42, 39, 48])
    paragraph(doc, '表中节省为已配对城市的可行解范围。达到求解时限的算例另记录最优值界限，覆盖率按经界限确认的净节省统计。'
              '在目前输入中，云通用计算和推理的可整体推迟小时负荷保持原时序。', size=10)


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'report', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError('输出必须采用独立的文档路径')
    report = json.loads(args.report.read_text('utf-8'))
    doc = Document(args.source)
    update(doc, report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(args.output)


if __name__ == '__main__':
    main()
