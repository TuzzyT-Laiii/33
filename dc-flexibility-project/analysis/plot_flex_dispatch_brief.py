"""简表第七章图表：调度前后周曲线、柔性价值汇总图与表格数据。"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


REPS = {'A-HPC': '纯 CPU 型·科研计算', 'A-CLD': '纯 CPU 型·云通用计算', 'D-SAI': 'GPU 主导型·科研 AI',
        'D-TRN': 'GPU 主导型·大模型训练/开发', 'D-INF': 'GPU 主导型·推理服务'}
SHORT = {'A-HPC': '科研计算', 'A-CLD': '云通用计算', 'D-SAI': '科研 AI', 'D-TRN': '训练/开发', 'D-INF': '推理服务'}
ALPHA = ['经济最优', '80%', '90%', '95%', '99%', '100%']
COLOR = {'pv': '#F2B701', 'wind': '#3C9D9B', 'discharge': '#8E6BBF', 'grid': '#A6A6A6',
         'load': '#1F1F1F', 'before': '#D1495B'}
LINE = {'A-HPC': '#2a6f97', 'A-CLD': '#7fb3d5', 'D-SAI': '#e07a1f', 'D-TRN': '#5b8e3e', 'D-INF': '#8E6BBF'}


def style():
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 9


def alpha_label(value):
    return '经济最优' if pd.isna(value) else f'{value:.0%}'


def dispatch_figure(hourly_dir, runs, city, scenario, start, days, out, *, indexed=False):
    begin = pd.Timestamp(start, tz='UTC')
    end = begin + pd.Timedelta(days=days)
    fig, axes = plt.subplots(2, 2, figsize=(13, 5.6), sharex=True, sharey=True)
    titles = {None: '经济最优（可用市电）', 1.0: '100% 本地逐小时无碳（不用市电）'}
    peak = 0
    frames = {}
    for alpha in (None, 1.0):
        for flex in ('none', 'both'):
            if indexed:
                selected = runs[(runs.city == city) & (runs.scenario == scenario)
                                & (runs.flex == flex) & (runs.alpha_label == alpha_label(alpha))]
                if len(selected) != 1:
                    raise ValueError(f'{city}/{scenario}/{alpha}/{flex} 的小时文件索引不唯一')
                record = selected.iloc[0]
                hourly_path = Path(hourly_dir) / record.hourly_path
                if sha256(hourly_path.read_bytes()).hexdigest() != record.hourly_sha256:
                    raise ValueError(f'逐小时文件校验值不符：{hourly_path}')
            else:
                hourly_path = Path(hourly_dir) / f'{city}-{scenario}-opt-a{alpha}-{flex}-main.parquet'
            frame = pd.read_parquet(hourly_path)
            frame['time_utc'] = pd.to_datetime(frame.time_utc, utc=True)
            frame = frame[(frame.time_utc >= begin) & (frame.time_utc < end)].set_index('time_utc')
            executed = frame.fixed_it_mw + frame.whole_exec_mw + frame.split_exec_mw
            pue = frame.facility_mw / executed
            frame['before_mw'] = pue * (frame.fixed_it_mw + frame.whole_arrival_mw + frame.split_arrival_mw)
            # 缺口小时补NaN，曲线断开
            frame = frame.reindex(pd.date_range(begin, end, freq='h', inclusive='left'))
            frames[(alpha, flex)] = frame
            peak = max(peak, float((frame.pv_mw + frame.wind_mw + frame.discharge_mw + frame.grid_mw).max()))
    for col, alpha in enumerate((None, 1.0)):
        for row, flex in enumerate(('none', 'both')):
            ax, frame = axes[row, col], frames[(alpha, flex)]
            x = frame.index
            layers = [frame.pv_mw, frame.wind_mw, frame.discharge_mw, frame.grid_mw]
            ax.stackplot(x, *layers, colors=[COLOR[k] for k in ('pv', 'wind', 'discharge', 'grid')],
                         alpha=.8, linewidth=0, labels=['光伏发电', '风电发电', '储能放电', '市电'])
            ax.plot(x, frame.facility_mw, color=COLOR['load'], lw=1.3, label='设施负荷（调度后）' if flex == 'both' else '设施负荷')
            if flex == 'both':
                ax.plot(x, frame.before_mw, color=COLOR['before'], lw=1.0, ls='--', label='设施负荷（调度前）')
            candidates = runs[(runs.city == city) & (runs.scenario == scenario) & (runs.design == 'opt')
                              & (runs.flex == flex) & (runs.alpha_label == alpha_label(alpha))]
            if not indexed:
                candidates = candidates[candidates.variant == 'main']
            if len(candidates) != 1:
                raise ValueError(f'{city}/{scenario}/{alpha}/{flex} 的调度汇总不唯一')
            row_run = candidates.iloc[0]
            cap = f'光伏 {row_run.pv_mw:.0f} MW · 风电 {row_run.wind_mw:.0f} MW · 储能 {row_run.battery_mwh:.0f} MWh'
            if alpha is None:
                cap += f' · 窗口本地匹配 {row_run.cfe_hourly:.0%}'
            head = ('无柔性' if flex == 'none' else '有柔性（整体12h＋可拆分48h）') + '｜' + titles[alpha]
            ax.set_title(head, fontsize=9.5, loc='left')
            ax.text(.995, .97, cap, transform=ax.transAxes, ha='right', va='top', fontsize=8,
                    bbox=dict(facecolor='white', alpha=.75, lw=0, pad=1.5))
            ax.grid(axis='y', alpha=.3)
            ax.set_ylim(0, peak * 1.12)
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    for ax in axes[:, 0]:
        ax.set_ylabel('功率 MW')
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, frameon=False, fontsize=8.5, bbox_to_anchor=(.5, -.01))
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(out / 'dispatch-week.png', dpi=200)
    fig.savefig(out / 'dispatch-week.pdf')
    plt.close(fig)
    return frames


def value_figure(runs, pairs, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw={'width_ratios': [1, 1.55]})
    x = np.arange(len(ALPHA))
    main = pairs[pairs.flex == 'both']
    for scenario, name in SHORT.items():
        stat = main[main.scenario == scenario].groupby('alpha_label').value_pct.agg(['median', 'min', 'max']).reindex(ALPHA)
        ax1.plot(x, stat['median'], marker='o', ms=3.5, lw=1.6, color=LINE[scenario], label=name)
        ax1.fill_between(x, stat['min'], stat['max'], color=LINE[scenario], alpha=.10, lw=0)
    ax1.set_xticks(x, ALPHA)
    ax1.set_ylabel('柔性降低的系统成本 %')
    ax1.set_xlabel('本地风光逐小时匹配比例')
    ax1.set_title('(a) 柔性价值随无碳目标变化（八城市中位数与范围）', fontsize=9.5, loc='left')
    ax1.grid(axis='y', alpha=.3)
    ax1.legend(fontsize=7.5, frameon=False, ncol=2, loc='upper left')
    ok = runs[(runs.design == 'opt') & (runs.variant == 'main') & (runs.status == 'optimal')]
    parts = {'光伏+风电': ['cost_pv', 'cost_wind'], '储能': ['cost_battery_power', 'cost_battery_energy'],
             '市电接入': ['cost_grid_capacity'], '市电电费+碳价': ['cost_grid_energy', 'cost_carbon']}
    colors = ['#F2B701', '#8E6BBF', '#5f6b7a', '#A6A6A6']
    positions, labels, bottom_rows = [], [], []
    pos = 0
    for alpha in ('经济最优', '100%'):
        for scenario in SHORT:
            for flex in ('none', 'both'):
                part = ok[(ok.scenario == scenario) & (ok.alpha_label == alpha) & (ok.flex == flex)]
                values = [part[cols].sum(axis=1).div(part.baseline_facility_mwh).mean() for cols in parts.values()]
                bottom = 0
                for value, color, name in zip(values, colors, parts):
                    ax2.bar(pos, value, bottom=bottom, color=color, width=.8, label=name if pos == 0 else None)
                    bottom += value
                if flex == 'both':
                    ax2.text(pos, bottom + 4, '有', ha='center', fontsize=6.5, color='#404040')
                else:
                    ax2.text(pos, bottom + 4, '无', ha='center', fontsize=6.5, color='#404040')
                positions.append(pos)
                pos += 1
            labels.append((pos - 1.5, SHORT[scenario]))
            pos += .6
        pos += 1.2
    ax2.set_xticks([p for p, _ in labels], [l for _, l in labels], fontsize=7.5, rotation=0)
    ax2.set_ylabel('设施度电成本 USD/MWh')
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.2)
    mid = (positions[9] + positions[10]) / 2
    ax2.axvline(mid, color='#bfbfbf', lw=.8)
    ax2.text(positions[0], ax2.get_ylim()[1] * .985, '经济最优', fontsize=8.5, va='top')
    ax2.text(positions[10], ax2.get_ylim()[1] * .985, '100% 本地无碳', fontsize=8.5, va='top')
    ax2.set_title('(b) 度电成本构成：无柔性 vs 有柔性（八城市平均）', fontsize=9.5, loc='left')
    ax2.grid(axis='y', alpha=.3)
    ax2.legend(fontsize=7.5, frameon=False, ncol=2, loc='upper left', bbox_to_anchor=(.0, .9))
    fig.tight_layout()
    fig.savefig(out / 'value-summary.png', dpi=200)
    fig.savefig(out / 'value-summary.pdf')
    plt.close(fig)


def brief_rows(runs, pairs, inputs):
    main = pairs[pairs.flex == 'both']
    med = lambda s, d=0: f'{s.median():.{d}f}'
    rows = [['代表情境', '可推迟IT电量%', '经济最优：度电成本\n无→有 USD/MWh', '经济最优：\n柔性价值%',
             '100%无碳：度电成本\n无→有 USD/MWh', '100%无碳：\n柔性价值%', '100%无碳：储能\n无→有 MWh']]
    shares = inputs
    for scenario, name in REPS.items():
        e = main[(main.scenario == scenario) & (main.alpha_label == '经济最优')]
        f = main[(main.scenario == scenario) & (main.alpha_label == '100%')]
        rows.append([name, shares[scenario],
                     f'{e.lcoe_base_usd_mwh.median():.0f}→{e.lcoe_flex_usd_mwh.median():.0f}', med(e.value_pct, 1),
                     f'{f.lcoe_base_usd_mwh.median():.0f}→{f.lcoe_flex_usd_mwh.median():.0f}', med(f.value_pct, 1),
                     f'{f.bat_mwh_base.median():.0f}→{f.bat_mwh_flex.median():.0f}'])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', required=True, help='results/flex-econ-copt-20261102')
    parser.add_argument('--hourly', required=True)
    parser.add_argument('--shares', required=True, help='flex-class-20261101/shares-10MW.csv')
    parser.add_argument('--city', default='frankfurt')
    parser.add_argument('--scenario', default='D-TRN')
    parser.add_argument('--start', default='2025-06-24')
    parser.add_argument('--days', type=int, default=7)
    args = parser.parse_args()
    style()
    result = Path(args.result)
    out = result / 'brief'
    out.mkdir(exist_ok=True)
    runs = pd.read_csv(result / 'runs.csv')
    runs['variant'] = runs.variant.fillna('main')
    runs['alpha_label'] = runs.alpha.map(alpha_label)
    pairs = pd.read_csv(result / 'paired-values.csv')
    frames = dispatch_figure(args.hourly, runs, args.city, args.scenario, args.start, args.days, out)
    value_figure(runs, pairs, out)
    share = pd.read_csv(args.shares, encoding='utf-8-sig').set_index('scenario')
    # 可推迟IT电量%：可推迟·整体+可推迟·可拆分占原生窗口IT电量
    share_map = {s: f"{share.loc[s, '可推迟·整体'] + share.loc[s, '可推迟·可拆分']:.0f}" for s in REPS}
    week = frames[(1.0, 'both')]
    before = frames[(1.0, 'none')]
    facts = {
        'dispatch_case': {'city': args.city, 'scenario': args.scenario, 'start': args.start, 'days': args.days,
                          'missing_hours': int(week.facility_mw.isna().sum())},
        'table_rows': brief_rows(runs, pairs, share_map),
    }
    (out / 'brief-facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps(facts, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
