import json
from pathlib import Path

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_scenario_typical_days import setup, save


COLORS = {'基础与空载 IT': '#D9DEE3', '冷却等非 IT': '#9CABC2', 'CPU 科研分区': '#418A9B',
    'Jupyter': '#B184B5', '课程分区': '#E8B65B', '其他分区': '#7F8C8D',
    '在线服务': '#DB9673', '离线批处理': '#DCA24D', '未对账部分': '#797C7D',
    '在线推理': '#DB9673', '离线推理': '#DCA24D', '批处理': '#418A9B', 'MapReduce': '#DCA24D',
    '交互式': '#B184B5', 'DNN 训练（含推断）': '#4C78A8', '长时单任务': '#59A14F',
    '数组批量': '#F28E2B', '多节点大作业': '#9C6FB0', '低 GPU 活动占用': '#B6992D',
    '其他与无法判断': '#797C7D', '预训练': '#418A9B', 'SFT': '#81BAA5', '评估': '#DCA24D',
    '调试': '#B184B5', '其他': '#788486'}


def handles(labels):
    return [Patch(facecolor=COLORS[label], label=label) for label in labels]


def draw(ax, output, row, upper, small):
    frame = pd.read_csv(output / (row['scenario'] + '-by-task.csv'))
    labels = frame.task_type.drop_duplicates().tolist()
    values = [frame[frame.task_type.eq(label)].sort_values('hour').facility_mw.to_numpy() for label in labels]
    ax.stackplot(np.arange(25), *[np.r_[value, value[-1]] for value in values], step='post',
        colors=[COLORS[label] for label in labels], linewidth=0)
    total = sum(values)
    ax.step(np.arange(25), np.r_[total, total[-1]], where='post', color='#35414D', lw=.8)
    ax.set(xlim=(0,24), ylim=(0,upper), xticks=[0,6,12,18,24])
    ax.set_xlabel('小时', fontsize=9 if small else 11, labelpad=1)
    ax.set_ylabel('设施功率（MW）', fontsize=9 if small else 11, labelpad=2)
    ax.set_title(row['scenario_name'], loc='left', fontsize=10 if small else 15, pad=21 if small else 28)
    ax.text(0, 1.025, row['source_date'] + '  ' + row['weekday'], transform=ax.transAxes,
        fontsize=8.5 if small else 10, color='#535B63')
    ax.grid(axis='y', color='#AAB2BA', lw=.4, alpha=.3)
    ax.tick_params(labelsize=8 if small else 10, pad=2)
    return labels


def plot_all(output):
    output = Path(output)
    setup()
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    rows = {row['scenario']: row for row in summary['scenarios']}
    upper = float(np.ceil(max(row['peak_mw'] for row in rows.values()) * 1.08))
    checks = {}
    for scenario, row in rows.items():
        fig, ax = plt.subplots(figsize=(10.2, 5.4))
        fig.subplots_adjust(left=.09, right=.98, top=.79, bottom=.35)
        labels = draw(ax, output, row, upper, False)
        fig.legend(handles=handles(labels), loc='lower left', bbox_to_anchor=(.085,.135), ncol=4, fontsize=8.5)
        use = scenario.split('-')[1]
        selected = summary['selections'][use]
        fig.text(.09,.09, f"{selected['complete_days']}个完整日：日电量最接近中位数，并列时日内RMSE最小；参考IT 10 MW。", fontsize=9, color='#4C535A')
        text = 'v3；MIT六月完成版，DNN含官方与推断；运行形态不代表用途或可调权限。' if use == 'SAI' else 'v3中央功率与PUE；基础与空载IT、冷却等非IT单列。'
        fig.text(.09,.055, text, fontsize=9, color='#4C535A')
        text = '原始峰值真实存在；单份SPECpower曲线50%→60%的陡升放大功率峰形。' if use == 'CLD' else ('源相对日；日历日期及星期未知。' if use == 'INF' else '源日期与参考网格2025年气象对应日期分别记录，功率含跨设备映射假设。')
        fig.text(.09,.02,text,fontsize=9,color='#4C535A')
        checks[scenario + '_figure_bounds'] = save(fig,output,scenario)
    fig, axes = plt.subplots(4,3,figsize=(16,10))
    fig.subplots_adjust(left=.065,right=.985,top=.885,bottom=.09,hspace=1.15,wspace=.26)
    fig.suptitle('各情境典型日 24 小时负荷曲线',x=.065,y=.985,ha='left',fontsize=20)
    fig.text(.065,.944,'v3  ·  参考IT 10 MW  ·  同一需求共用日期与资源构成  ·  全部面板共用纵轴',fontsize=11,color='#535B63')
    grid = [['A-HPC','A-CLD',None],['B-SAI','C-SAI','D-SAI'],['B-TRN','C-TRN','D-TRN'],['B-INF','C-INF','D-INF']]
    for r, scenarios in enumerate(grid):
        for c, scenario in enumerate(scenarios):
            ax=axes[r,c]
            if scenario is None:
                ax.axis('off')
                ax.legend(handles=handles(['基础与空载 IT','冷却等非 IT']),loc='upper left',fontsize=10)
                ax.text(.02,.44,'MIT六月完成版；DNN训练含推断。\n长时、数组、多节点仅表示运行形态。\n云负荷源相对06时峰值真实存在，\n单份SPECpower的50%→60%陡升放大峰形。',transform=ax.transAxes,va='top',fontsize=8.8,color='#535B63',linespacing=1.6)
                continue
            labels=draw(ax,output,rows[scenario],upper,True)
            if r==0:
                ax.legend(handles=handles(labels[1:-1]),loc='upper left',bbox_to_anchor=(0,-.32),ncol=4,fontsize=8,handlelength=1,columnspacing=.8)
            elif c==0:
                ax.legend(handles=handles(labels[1:-1]),loc='upper left',bbox_to_anchor=(0,-.32),ncol=len(labels)-2,
                    fontsize=8 if r==1 else 9,handlelength=1,columnspacing=.8)
    fig.text(.065,.018,'任务层为资源驱动的IT增量；ASI与Alibaba只有相对日。源日与天气重放日分别记录；功率、PUE及全年对照均采用v3。',fontsize=10,color='#535B63')
    checks['overview_figure_bounds']=save(fig,output,'overview')
    return checks
