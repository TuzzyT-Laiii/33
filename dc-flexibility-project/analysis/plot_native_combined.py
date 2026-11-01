import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd

from native_combined_power import RUN
from plot_scenario_typical_days import setup
from plot_scenario_typical_days_v3 import draw, handles


GRID = [['A-HPC', 'A-CLD', None], ['B-SAI', 'C-SAI', 'D-SAI'], ['B-TRN', 'C-TRN', 'D-TRN'], ['B-INF', 'C-INF', 'D-INF']]


def save(fig, output, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text().strip():
            bounds = text.get_window_extent(renderer)
            if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > fig.bbox.width + 1 or bounds.y1 > fig.bbox.height + 1:
                outside.append(text.get_text())
    if outside:
        raise ValueError({'figure': name, 'outside_text': outside})
    fig.savefig(output / (name + '.png'), dpi=300, facecolor='white')
    fig.savefig(output / (name + '.pdf'), facecolor='white')
    plt.close(fig)


def native(output, summary, series, profiles, weekly=False, averaged=False):
    fig, axes = plt.subplots(4, 3, figsize=(16, 10), sharey=True)
    fig.subplots_adjust(left=.06, right=.985, top=.89, bottom=.08, hspace=.70, wspace=.23)
    label = '周内平均' if weekly else '日内平均' if averaged else '原生窗口小时'
    fig.suptitle(f'各情境{label}负荷曲线', x=.06, y=.98, ha='left', fontsize=20)
    fig.text(.06, .935, '参考IT 10 MW  ·  组合功率模型  ·  各来源保留原生窗口  ·  全部面板共用纵轴', fontsize=11, color='#535B63')
    upper = float(np.ceil(summary.placement_peak_facility_mw_max.max() * 1.04))
    for r, scenarios in enumerate(GRID):
        for c, scenario in enumerate(scenarios):
            ax = axes[r, c]
            if scenario is None:
                ax.axis('off')
                ax.plot([], [], color='#2878B5', label='IT功率')
                ax.plot([], [], color='#A05799', label='设施功率：放置中位数')
                ax.fill_between([], [], [], color='#A05799', alpha=.2, label='设施功率：放置最小–最大')
                ax.legend(loc='upper left', fontsize=10)
                ax.text(.02, .30, '有日期来源按源月日小时对应气象。\n无日期来源在12个月初分别放置。\n年末截断，阴影同时包含天气与源前缀差异。', transform=ax.transAxes, fontsize=9, va='top', linespacing=1.6)
                continue
            sid = scenario + '-10MW'
            row = summary.loc[sid]
            source = pd.read_parquet(output / 'source-hourly' / (sid + '.parquet'))
            ax.set_title(row.scenario_zh, loc='left', fontsize=10)
            if weekly and scenario == 'A-CLD':
                ax.text(.5, .5, '仅5天，不报告周形状', ha='center', transform=ax.transAxes)
                ax.set_axis_off()
                continue
            if averaged or weekly:
                grouping = 'hour_of_source_week' if weekly else 'hour_of_source_day'
                data = profiles.loc[profiles.scenario_id.eq(sid)]
                facility = data.groupby(grouping).facility_mw.agg(['median', 'min', 'max'])
                it = source.groupby(grouping).it_mw.mean()
                x = facility.index.to_numpy()
                ax.plot(it.index, it, color='#2878B5', lw=1)
                ax.plot(x, facility['median'], color='#A05799', lw=1)
                ax.fill_between(x, facility['min'], facility['max'], color='#A05799', alpha=.2, lw=0)
                ax.set_xlabel('源相对七日小时' if weekly and scenario.endswith('INF') else '源星期×小时' if weekly else '源小时', fontsize=9)
                ax.set_xlim(0, 167 if weekly else 23)
                ax.set_xticks([0, 24, 48, 72, 96, 120, 144, 167] if weekly else [0, 6, 12, 18, 23])
            else:
                data = series.loc[series.scenario_id.eq(sid)]
                x = data.source_hour_index / 24
                ax.plot(x, data.facility_mw_median, color='#A05799', lw=.55)
                ax.fill_between(x, data.facility_mw_min, data.facility_mw_max, color='#A05799', alpha=.2, lw=0)
                ax.plot(source.source_hour_index / 24, source.it_mw, color='#2878B5', lw=.55)
                ax.set_xlabel(f'源窗口经过天数（共{int(row.native_days)}天）', fontsize=9)
                ax.set_xlim(0, row.native_days)
                step = int(np.ceil(row.native_days / 4))
                ax.set_xticks(np.unique(np.r_[np.arange(0, row.native_days, step), row.native_days]))
            ax.set_ylim(0, upper)
            ax.set_yticks(np.arange(0, upper + .001, 2))
            ax.tick_params(labelsize=8, pad=2)
            ax.grid(alpha=.15)
            if c == 0:
                ax.set_ylabel('功率（MW）', fontsize=9)
    fig.text(.06, .018, '设施逐小时中位数包络的峰值与“各放置窗口峰值的中位数”不同；完整IT窗口保留原生长度，电量按各实际窗口积分。', fontsize=9, color='#535B63')
    save(fig, output, 'native-week' if weekly else 'native-day' if averaged else 'native-overview')


def selected_days(output):
    typical = output / 'selected-days'
    summary = json.loads((typical / 'summary.json').read_text('utf-8'))
    rows = {row['scenario']: row for row in summary['scenarios']}
    upper = float(np.ceil(max(row['peak_mw'] for row in rows.values()) * 1.08))
    fig, axes = plt.subplots(4, 3, figsize=(16, 10))
    fig.subplots_adjust(left=.065, right=.985, top=.885, bottom=.09, hspace=1.15, wspace=.26)
    fig.suptitle('各情境典型日 24 小时负荷曲线', x=.065, y=.985, ha='left', fontsize=20)
    fig.text(.065, .944, '与原生窗口同一组合模型  ·  参考IT 10 MW  ·  同一用途共用来源日期与放置  ·  全部面板共用纵轴', fontsize=11, color='#535B63')
    for r, scenarios in enumerate(GRID):
        for c, scenario in enumerate(scenarios):
            ax = axes[r, c]
            if scenario is None:
                ax.axis('off')
                ax.legend(handles=handles(['基础与空载 IT', '冷却等非 IT']), loc='upper left', fontsize=10)
                ax.text(.02, .44, '#35请求GPU＋#36 CPU及训练日形状。\n训练比例0.730–0.840，加权均值0.80。\n日内形状为集群迁移假设，无星期修正。\n源日期与对应气象日期分别记录。', transform=ax.transAxes, va='top', fontsize=9, color='#535B63', linespacing=1.6)
                continue
            labels = draw(ax, typical, rows[scenario], upper, True)
            ax.text(.99, .94, f"最高/最低 {rows[scenario]['peak_valley_ratio']:.3f}", transform=ax.transAxes, ha='right', va='top', fontsize=7.6)
            if r == 0 or c == 0:
                ax.legend(handles=handles(labels[1:-1]), loc='upper left', bbox_to_anchor=(0, -.32), ncol=4 if r == 0 else len(labels)-2,
                    fontsize=8 if r <= 1 else 9, handlelength=1, columnspacing=.8)
    fig.text(.065, .018, '完整日中位电量±2%内候选，少于5天放宽至±5%；候选内与全窗口平均24小时设施曲线RMSE最小。', fontsize=10, color='#535B63')
    save(fig, typical, 'overview')
    for scenario, row in rows.items():
        fig, ax = plt.subplots(figsize=(10.2, 5.4))
        fig.subplots_adjust(left=.09, right=.98, top=.80, bottom=.30)
        labels = draw(ax, typical, row, upper, False)
        fig.legend(handles=handles(labels), loc='lower left', bbox_to_anchor=(.085, .10), ncol=4, fontsize=9)
        placement_name = '源月日对应气象' if row['placement'] == 'dated' else f"{int(row['placement'][1:])}月初放置"
        fig.text(.09, .055, f"来源日期{row['source_date']}；{placement_name}；最高/最低{row['peak_valley_ratio']:.3f}；参考IT 10 MW。", fontsize=9)
        fig.text(.09, .018, '模型与第五章原生窗口一致；无日期来源的源相位、气象放置及训练集群形状迁移属于情境假设。', fontsize=9)
        save(fig, typical, scenario)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    setup()
    plt.rcParams.update({'pdf.fonttype': 42})
    output = args.project / 'results' / RUN
    summary = pd.read_csv(output / 'native-summary.csv').query('reference_it_mw == 10').set_index('scenario_id')
    series = pd.read_parquet(output / 'native-timeseries-10MW.parquet')
    native(output, summary, series, None)
    native(output, summary, series, pd.read_csv(output / 'typical-day-placements.csv'), averaged=True)
    native(output, summary, series, pd.read_csv(output / 'typical-week-placements.csv'), weekly=True)
    selected_days(output)
