"""在简表第九章末尾追加“空闲容量的三种用法”一节（能源柔性、少配服务器、出售空闲算力）。

用法：python update_idle_resale.py --source <简表.docx> --results <results/idle-gpu-resale-import-20261108> --output <输出.docx>
沿用 update_minimum_capacity.py 的三线表与段落样式；若文末已有本节则先删除再重写。
"""
import argparse
import csv
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

from update_joint_capacity import paragraph, title
from update_minimum_capacity import table

TITLE = '空闲容量的三种用法：能源柔性、少配服务器与出售空闲算力'
NAMES = {'SAI': '科研 AI', 'TRN': '大模型训练', 'INF': '推理服务'}
KINDS = {'B': 'CPU 主导型', 'C': '均衡型', 'D': 'GPU 主导型'}


def read(path):
    with open(path, encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def remove_existing(doc):
    body = doc.element.body
    elements = list(body)
    start = next((i for i, e in enumerate(elements) if e.tag == qn('w:p') and Paragraph(e, doc._body).text.strip() == TITLE), None)
    if start is not None:
        for element in elements[start:]:
            if element.tag != qn('w:sectPr'):
                body.remove(element)


def section(doc, results):
    summary = {row['scenario']: row for row in read(results / 'B-C-D-summary.csv')}
    resale = {(row['scenario'], float(row['price_usd_gpu_h']), float(row['sold_fraction'])): row for row in read(results / 'resale-summary.csv')}
    title(doc, TITLE)
    paragraph(doc, '机群的空闲容量有三种用法：保持机群规模，只在期限内重排任务以降低能源费用（B 相对 A）；把机群台数纳入规划，少配服务器（C 相对 B）；'
              '保持机群规模和原时序，把每小时未被任务占用的 GPU 按固定价格出租给推理服务（情景 D）。'
              '情景 D 扣除售出 GPU 运行所需的电费与碳成本，以可售比例反映空档零碎、需求不足、训练抢占和切换开销。三种用法不能相加。')
    rows = []
    for use in ('SAI', 'TRN', 'INF'):
        for kind in 'BCD':
            scenario = f'{kind}-{use}'
            s = summary[scenario]
            d1 = resale[(scenario, 1.0, 0.5)]
            d22 = resale[(scenario, 2.2, 0.5)]
            energy_cost = float(d1['energy_cost_usd_per_sold_gpu_h_median'])
            sold_hours = float(d1['D_net_annual_usd_median']) / (1.0 - energy_cost)      # 50% 可售比例下的年可售 GPU 小时
            equipment = float(s['equipment_annual_saving_usd_median'])
            breakeven = equipment / sold_hours + energy_cost
            rows.append((f'{KINDS[kind]}·{NAMES[use]}',
                         f"{float(s['B_vs_A_saving_usd_median']) / 1e6:.2f}",
                         f'{equipment / 1e6:.2f}',
                         f"{float(d1['D_net_annual_usd_median']) / 1e6:.2f}",
                         f"{float(d22['D_net_annual_usd_median']) / 1e6:.2f}",
                         f"{float(d1['D_pct_of_A_fleet_median']):.1f}",
                         f'{breakeven:.2f}',
                         f"{float(d1['CD_extra_annual_usd_median']) / 1e6:.2f}"))
    table(doc, ('情境', '能源柔性 B 相对 A', '少配服务器 设备年省', '出售空闲 1 美元/GPU·时', '出售空闲 2.2 美元/GPU·时',
                '出售 1 美元 占机群 %', '持平售价 美元/GPU·时', '少配后再卖 1 美元'), rows, [34, 30, 30, 32, 32, 27, 30, 30])
    paragraph(doc, '金额为百万美元／年，八城市中位数，按原生观测窗口等效年化；出售空闲算力取可售比例 50%。'
              '持平售价为出售收入等于少配服务器设备年化节省时的价格。售出 GPU 的能耗与碳成本为每 GPU 小时 0.02–0.10 美元。'
              '2.2 美元为 DeepInfra H100 独享实例 2026 年 7 月后的价格（第三方价格追踪站记录），其官方当前价为 2.9 美元；运营方实际转售价低于零售价。', size=10)
    paragraph(doc, '能源柔性每年只有 16–95 万美元，比另外两种用法小一到两个数量级。少配服务器与出售空闲算力处于同一量级：'
              '训练的持平售价最高，约 2.2–2.8 美元，科研 AI 与推理约 1.1–1.5 美元，按零售价出售更有利；两者基本互斥，少配服务器后训练剩余空档的出租收入只有 2.6–4.0 百万美元。'
              '情景 D 不模拟推理市场需求与价格形成，未建模显存装箱与模型加载开销。', size=10)


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'results', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    doc = Document(args.source)
    remove_existing(doc)
    section(doc, args.results)
    doc.save(args.output)


if __name__ == '__main__':
    main()
