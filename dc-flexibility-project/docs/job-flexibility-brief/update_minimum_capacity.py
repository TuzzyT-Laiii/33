import argparse
import json
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

from update_joint_capacity import paragraph, table as base_table, title


def table(doc, headers, rows, widths):
    result = base_table(doc, headers, rows, widths)
    borders = result._tbl.tblPr.find(qn('w:tblBorders'))
    for border in borders:
        side = border.tag.rsplit('}', 1)[-1]
        border.set(qn('w:val'), 'single' if side in ('top', 'bottom') else 'nil')
        border.set(qn('w:sz'), '8')
        border.set(qn('w:color'), '000000')
    for cell in result.rows[0].cells:
        properties = cell._tc.get_or_add_tcPr()
        shade = properties.find(qn('w:shd'))
        if shade is not None:
            properties.remove(shade)
        cell_borders = OxmlElement('w:tcBorders')
        bottom = OxmlElement('w:bottom')
        for name, value in (('val', 'single'), ('sz', '6'), ('color', '000000')):
            bottom.set(qn('w:' + name), value)
        cell_borders.append(bottom)
        properties.append(cell_borders)
    return result


def replace_chapter(doc):
    body = doc.element.body
    elements = list(body)
    start = None
    for index, element in enumerate(elements):
        if element.tag == qn('w:p'):
            content = Paragraph(element, doc._body).text
            if content.startswith('九 '):
                start = index
                break
    if start is None:
        raise ValueError('输入简表缺少需要更新的第九章')
    for element in elements[start:]:
        if element.tag != qn('w:sectPr'):
            body.remove(element)


def money_range(row, name):
    median, lower, upper = [row[name + '_' + part] / 1e6 for part in ('median', 'minimum', 'maximum')]
    return f'{median:.2f}〔{lower:.2f}–{upper:.2f}〕'


def capacity_text(row):
    text = f"{row['servers']:,}" if row['lower_servers'] == row['servers'] else f"{row['lower_servers']:,}–{row['servers']:,}"
    return text + ('*' if row['gap_servers'] > 1 else '')


def chapter(doc, report):
    primary = [row for row in report['main'] if row['target'] == 'economic']
    title(doc, '九 完成相同任务所需的最少服务器台数')
    paragraph(doc, '第二部分按与地点无关的任务容量规划和八城市能源复核组织。'
              '三组期限分别为整体作业最多推迟 12／24／48 小时、可拆分工作量最多推迟 48／96／168 小时。'
              '保留原始作业持续时间，整体作业按整小时平移后连续运行。服务器台数同时满足 CPU 和 GPU 小时资源需求。')
    rows = []
    for row in primary:
        rows.append((row['scenario'], f"{row['A_servers']:,}", capacity_text(row),
                     f"{row['servers_avoided']:,}", f"{row['gpus_avoided']:,}",
                     f"{row['reduction_pct']:.1f}", f"{row['purchase_avoided_usd'] / 1e6:.2f}",
                     f"{row['equipment_annual_saving_usd'] / 1e6:.2f}"))
    table(doc, ('情境', 'A 台数', '最少台数区间', '可少配台数', '少配 GPU 数', '减配率 %',
                '少买设备 百万美元', '设备年省 百万美元'), rows, [24, 28, 37, 31, 32, 25, 39, 38])
    paragraph(doc, '容量区间由有效下界和完整可行调度的台数上界构成。星号标记的情境仍未达到上下界相差 1 台以内的证明精度。'
              '采购额和费用换算采用已核验的可行台数。相同小时占用形状和资源数量的作业合并为带数量的整数变量，'
              '每项整数选择还原为同类作业的连续运行。分段上界固定跨段作业，下界省略选定时段之外的可调度作业。')
    paragraph(doc, 'A-CLD 与推理的可推迟部分按小时工作量在期限内转移。'
              'A-CLD 仅有 120 小时，窗口末清空会显著影响期限敏感性。'
              '设备规格、跨硬件需求映射、价格和服务权限沿用研究情境，台数证明适用于本研究的小时资源模型。', size=10)

    for token, heading in (('economic', '八城市经济最优供能下的费用变化'),
                           ('alpha100', '八城市百分之百本地无碳匹配下的费用变化')):
        title(doc, heading)
        paragraph(doc, '各城市将 C 组台数固定为容量表中的可行上界，采用完整任务调度优化供能，并以能源松弛计算费用下界。'
                  '原 MIP 中台数相同且通过核验的可行方案也参与收紧费用上界。B 的设备台数固定为 A。'
                  '表中能源变化与增量净节省均为 B 到 C 的可行方案比较，'
                  '金额按原生观测窗口折算为等效年化。')
        rows = []
        for row in [row for row in report['main'] if row['target'] == token]:
            rows.append((row['scenario'], f"{row['equipment_annual_saving_usd'] / 1e6:.2f}",
                         money_range(row, 'energy_annual_saving_vs_hourly_B_usd'),
                         money_range(row, 'net_annual_saving_vs_hourly_B_usd'),
                         f"{row['net_incremental_pct_of_fleet']:.2f}",
                         money_range(row, 'net_annual_saving_vs_A_usd')))
        table(doc, ('情境', '设备年省 百万美元', '能源年省 中位数〔范围〕',
                    '净年省 中位数〔范围〕', '净省／A机群年化成本 %', 'A到C总年省 中位数〔范围〕'),
              rows, [25, 37, 49, 50, 43, 50])
        selected = [row for row in report['main'] if row['target'] == token]
        widest = max(row['maximum_energy_lp_gap_usd_maximum'] for row in selected) / 1e6
        paragraph(doc, '金额单位为百万美元／年。能源年省为负表示供能费用增加。'
                  '设备年化系数为 0.3319208，净省比例以 A 的设备年化费用为分母。'
                  f'本目标下最大的单城市能源费用区间宽度为 {widest:.2f} 百万美元／年。'
                  '联合任务时序的最优费用位于配套费用区间内，方案比较与原 MIP 的独立误差核验分别记录。'
                  'A-HPC 的费用对应其可行台数上界。', size=10)

    title(doc, '期限敏感性与容量下界核对')
    rows = []
    for scenario in dict.fromkeys(row['scenario'] for row in primary):
        cases = [row for row in report['capacity'] if row['scenario'] == scenario]
        reserve = [row for row in report['reserves'] if row['scenario'] == scenario]
        rows.append((scenario, *(capacity_text(row) for row in cases),
                     f"{cases[0]['C_over_A']:.3f}", f"{cases[0]['claude_power_lower_ratio']:.3f}",
                     f"{reserve[0]['provisioned_servers']:,}", f"{reserve[1]['provisioned_servers']:,}"))
    table(doc, ('情境', '12／48 小时', '24／96 小时', '48／168 小时', 'C／A',
                '功率 LP 比例参考', '另配 5% 余量 台', '另配 10% 余量 台'),
          rows, [25, 35, 35, 35, 25, 37, 32, 30])
    paragraph(doc, '功率 LP 比例以任务功率峰值为分母，C／A 以 CPU／GPU 资源对应的整机台数为分母。'
              '配套对照将可行调度重算为任务功率峰值，逐项检查其与功率 LP 下界的关系。'
              '5% 和 10% 余量列在可行台数上额外配置设备，费用敏感性单列。'
              'HPC 的星号区间仍未收紧；48／168 小时的可行上界沿用较严格期限方案，下界单独计算。'
              '全球能源补充计算尚未达到八城市误差小于 1% 的开展条件。')


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'report', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    doc = Document(args.source)
    replace_chapter(doc)
    chapter(doc, json.loads(args.report.read_text('utf-8')))
    doc.save(args.output)


if __name__ == '__main__':
    main()
