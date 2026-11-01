import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_scenario_typical_days import setup, save
from plot_scenario_typical_days_v3 import draw, handles


def plot_all(output, rule_caption=None, individual_scenarios=None, training_diurnal=False):
    output = Path(output)
    setup()
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    rows = {row['scenario']: row for row in summary['scenarios']}
    endpoints = pd.read_csv(output / 'idle-endpoints.csv').set_index('scenario')
    upper = float(np.ceil(max(row['peak_mw'] for row in rows.values()) * 1.08))
    checks = {}
    for scenario, row in rows.items():
        if individual_scenarios is not None and scenario not in individual_scenarios:
            continue
        key, use = scenario.split('-')
        idle = endpoints.loc[scenario]
        idle_range = f'{idle.low_unit_w:g}–{idle.high_unit_w:g}'
        fig, ax = plt.subplots(figsize=(10.2, 5.4))
        fig.subplots_adjust(left=.09, right=.98, top=.79, bottom=.37)
        labels = draw(ax, output, row, upper, False)
        ax.text(.99, .96, f"最高/最低＝{row['peak_valley_ratio']:.3f}", transform=ax.transAxes,
            ha='right', va='top', fontsize=10, color='#35414D')
        fig.legend(handles=handles(labels), loc='lower left', bbox_to_anchor=(.085, .165), ncol=4, fontsize=8.5)
        count = summary['selections'][use]['complete_days']
        lines = [f'{count}个完整日：{rule_caption}；参考IT 10 MW。' if rule_caption else f'{count}个完整日：日电量最接近中位数，并列时日内RMSE最小；参考IT 10 MW。']
        if use == 'TRN' and training_diurnal:
            ax.set_xlabel('源 UTC 小时', fontsize=11, labelpad=1)
            lines += ['训练#36：Kalos固定271卡UTC小时形状迁移，GPU小时加权0.80（0.730–0.840）。',
                '反推＝[集群估计功率−未分配卡数×空闲功率]/分配卡数；覆盖不足，直接标定不可用。',
                '敏感性：形状幅度0–1.5、固定集合95%–99.9%；中央水平0.65–0.95为独立范围。']
        elif use == 'TRN':
            lines += ['来源#36：训练GPU/TDP＝0.80；三份SPEC平滑CPU；原作业分配轨迹。',
                f'敏感性：训练比例0.65–0.95；基础/空闲{idle_range} W/台，分别改变。',
                '比例属于工程情境；逐类作业功率未知。源日期与2025年天气重放日分别记录。']
        elif use == 'INF':
            lines += ['来源#35请求形状与GPU映射＋#36平滑CPU；完整重算冷却；源相对日，日历未知。',
                f'敏感性：基础/空闲{idle_range} W/台；在线SXM空闲100–160 W、高负载350–700 W。',
                '曲率γ＝0.7–1.3；PCIe按TDP比例迁移；范围为工程情境，未按请求实测GPU功率。']
        elif use == 'CLD':
            lines += ['来源#36：三份SPEC平滑CPU；在线/离线/未对账资源保持#33。',
                f'基础/空闲敏感性{idle_range} W/台，属于工程情境；真实日历日期和星期未知。',
                '原始相对06时高值保留；CPU在50%→60%区间动态增量约125→41 W，放大效应减弱。']
        else:
            lines += ['来源#36：三份SPEC平滑CPU；' + ('GPU保持MIT六月完成版；DNN含官方与推断。' if use == 'SAI' else 'Lisa作业资源小时保持。'),
                (f'基础/CPU空闲敏感性：无任务整机{idle_range} W/台；GPU空闲保持MIT代理。' if use == 'SAI' else f'基础/空闲敏感性{idle_range} W/台，属于工程情境。'),
                '任务层为超过无任务端点的IT增量；源日期与2025年天气重放日分别记录。']
        for y, line in zip((.12, .085, .05, .015), lines):
            fig.text(.09, y, line, fontsize=8.8, color='#4C535A')
        checks[scenario + '_figure_bounds'] = save(fig, output, scenario)
    fig, axes = plt.subplots(4, 3, figsize=(16, 10))
    fig.subplots_adjust(left=.065, right=.985, top=.885, bottom=.09, hspace=1.15, wspace=.26)
    fig.suptitle('各情境典型日 24 小时负荷曲线', x=.065, y=.985, ha='left', fontsize=20)
    fig.text(.065, .944, '请求日内GPU映射＋用途功率与CPU修正  ·  参考IT 10 MW  ·  同需求共用源日  ·  全部面板共用纵轴', fontsize=11, color='#535B63')
    grid = [['A-HPC', 'A-CLD', None], ['B-SAI', 'C-SAI', 'D-SAI'], ['B-TRN', 'C-TRN', 'D-TRN'], ['B-INF', 'C-INF', 'D-INF']]
    for r, scenarios in enumerate(grid):
        for c, scenario in enumerate(scenarios):
            ax = axes[r, c]
            if scenario is None:
                ax.axis('off')
                ax.legend(handles=handles(['基础与空载 IT', '冷却等非 IT']), loc='upper left', fontsize=10)
                note = ('训练#36：Kalos固定271卡UTC日形状。\nGPU小时加权0.80；比例0.730–0.840。\n幅度0–1.5；固定集合95%–99.9%。\n分配功率反推覆盖不足，无作业映射。\n其余八情境保持v5；全部为10MW。' if training_diurnal else
                    '#35请求GPU＋#36 CPU/训练。\n训练GPU/TDP 0.80（0.65–0.95）。\n基础/空闲中央值沿用，敏感性另列。\nAlibaba原始峰值保留，CPU放大效应减弱。')
                ax.text(.02, .44, note,
                    transform=ax.transAxes, va='top', fontsize=8.8, color='#535B63', linespacing=1.6)
                continue
            labels = draw(ax, output, rows[scenario], upper, True)
            if training_diurnal and scenario.endswith('-TRN'):
                ax.set_xlabel('源 UTC 小时', fontsize=9, labelpad=1)
            ax.text(.99, .94, f"最高/最低 {rows[scenario]['peak_valley_ratio']:.3f}", transform=ax.transAxes,
                ha='right', va='top', fontsize=7.6, color='#35414D')
            if r == 0:
                ax.legend(handles=handles(labels[1:-1]), loc='upper left', bbox_to_anchor=(0, -.32), ncol=4, fontsize=8, handlelength=1, columnspacing=.8)
            elif c == 0:
                ax.legend(handles=handles(labels[1:-1]), loc='upper left', bbox_to_anchor=(0, -.32), ncol=len(labels)-2,
                    fontsize=8 if r == 1 else 9, handlelength=1, columnspacing=.8)
    footer = rule_caption + '；同需求三规格共用日期。' if rule_caption else '任务层为IT增量；DNN含推断，运行形态不代表用途或权限。ASI/Alibaba仅有相对日，请求与源日相位为合成映射。'
    fig.text(.065, .018, footer, fontsize=10, color='#535B63')
    checks['overview_figure_bounds'] = save(fig, output, 'overview')
    return checks
