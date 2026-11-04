import argparse
import json
from pathlib import Path

import pandas as pd

from native_combined_power import RUN


def report(project):
    output = project / 'results' / RUN
    summary = pd.read_csv(output / 'native-summary.csv')
    ten = summary.loc[summary.reference_it_mw.eq(10)].copy()
    order = ['A-HPC', 'A-CLD', 'B-SAI', 'C-SAI', 'D-SAI', 'B-TRN', 'C-TRN', 'D-TRN', 'B-INF', 'C-INF', 'D-INF']
    ten['order'] = ten.scenario_id.str.removesuffix('-10MW').map({v: i for i, v in enumerate(order)})
    ten = ten.sort_values('order')
    columns = [('mean_it_mw', 'IT平均MW'), ('peak_it_mw', 'IT峰值MW'), ('minimum_it_mw', 'IT最小MW'),
        ('it_diurnal_peak_valley', 'IT日内比'), ('it_weekly_peak_valley', 'IT周内比'), ('window_it_mwh', 'IT窗口MWh'),
        ('placement_mean_facility_mw_median', '设施平均MW'), ('placement_peak_facility_mw_median', '设施峰值MW'),
        ('placement_minimum_facility_mw_median', '设施最小MW'), ('placement_facility_diurnal_peak_valley_median', '设施日内比'),
        ('placement_facility_weekly_peak_valley_median', '设施周内比'), ('placement_window_facility_mwh_median', '设施窗口MWh')]
    rows = [['情境（参考IT 10MW）', '原生\n天数', 'IT平均\nMW', 'IT峰值\nMW', '设施平均\nMW', '设施峰值\nMW', '设施日内\n峰谷比', '设施平均范围\nMW']]
    for row in ten.to_dict('records'):
        value_range = f"{row['placement_mean_facility_mw_min']:.3f}–{row['placement_mean_facility_mw_max']:.3f}" if not row['has_real_dates'] else '—'
        rows.append([row['scenario_zh'], str(int(row['native_days'])), f"{row['mean_it_mw']:.3f}", f"{row['peak_it_mw']:.3f}",
            f"{row['placement_mean_facility_mw_median']:.3f}", f"{row['placement_peak_facility_mw_median']:.3f}",
            f"{row['placement_facility_diurnal_peak_valley_median']:.3f}", value_range])
    (output / 'brief-summary.json').write_text(json.dumps({'rows': rows, 'reference_it_mw': 10,
        'model': '#35 GPU请求映射＋#36 CPU响应及Acme训练日形状；cooling-20261028-v2'}, ensure_ascii=False, indent=2), encoding='utf-8')
    summary[['scenario_zh', 'reference_it_mw', 'native_days', *[c for c, _ in columns]]].rename(columns=dict(columns)).to_csv(output / 'all-44-scenarios.csv', index=False, encoding='utf-8-sig')
    comparison = pd.read_csv(output / 'old-model-comparison.csv').query('reference_it_mw == 10').set_index('scenario_id')
    selected = json.loads((output / 'selected-days/summary.json').read_text('utf-8'))
    lines = ['# 组合功率模型下的原生窗口负荷曲线', '',
        '采用11个服务器类型×用途组合、1/10/100/500MW四档，共44个风冷情境。参考IT规模用于设备整数台数，逐小时功率保持模型结果，不按参考规模截断。', '',
        '## 原生窗口和统一模型', '',
        '不循环成全年、不补齐来源。Lisa为309天（2021-12-27至2022-10-31），MIT为237天（2021-02-06至09-30），Acme Kalos为90天（2023-05-18至08-15 UTC）；按源月日小时对应目标气象年。Lisa原文件无时区，直接对应UTC属于假设；MIT保留既有Slurm epoch小时。', '',
        'Alibaba仅5个完整日（相对第2–6天，120小时），ASI为185天（相对第0–184天，4440小时），两者真实日期及时区未知。各自在12个月1日00:00 UTC放置，超过年末截断源尾部。ASI七至十二月实际长度为184/153/122/92/61/31天；放置范围同时包含气象和源前缀变化。Alibaba不报告周形状，ASI周形状仅指相对七日。', '',
        'CPU采用#36三份SPECpower归一化单调响应，基础与空闲采用中央值。推理GPU沿用#35 BurstGPT输入/输出token各50%的日形状和逐地点气象年冻结gain，离线推理维持固定分配功率；原生窗口不重新标定gain，也不要求其短窗口在线电量等于固定每卡模型。训练采用#36固定271张GPU集群日形状shape_central，GPU/TDP为0.730–0.840，原生90天GPU小时加权均值0.80。科研AI采用MIT六月完成版标定，迁移至目标H100。所有设施功率完整重算cooling-20261028-v2。', '',
        '## 10MW汇总', '',
        'IT指标取完整原生窗口；设施指标取参考网格2025年各次实际窗口统计的中位数，有日期来源只有一次对应。设施峰值为窗口峰值的中位数，日内比为各源小时均值的最大/最小再取放置中位数。范围为各次设施均值最小–最大。窗口电量不年化。', '',
        '| ' + ' | '.join(rows[0]).replace('\n', '') + ' |',
        '| ' + ' | '.join(['---'] + ['---:'] * 7) + ' |']
    lines += ['| ' + ' | '.join(row) + ' |' for row in rows[1:]]
    lines += ['', '![原生时序](native-overview.png)', '', '![原生日内平均](native-day.png)', '', '![原生周内平均](native-week.png)', '',
        '时序阴影为每个源小时可用放置的设施功率最小–最大；ASI后段可用放置数逐步减少。逐小时中位数包络的峰值与表中“窗口峰值的中位数”具有不同聚合顺序。', '',
        '## 与旧模型同窗口比较', '',
        '对照#34 c836bba原生结果，所有需求、整数台数、源顺序、天气小时和截断长度均保持；这里的差异来自功率模型。', '',
        '| 情境 | IT平均MW 旧→新 | IT峰值MW 旧→新 | 设施平均MW 旧→新 | 设施峰值MW 旧→新 |', '| --- | ---: | ---: | ---: | ---: |']
    for row in ten.to_dict('records'):
        old = comparison.loc[row['scenario_id']]
        values = [f"{old['old_' + c]:.3f}→{old['new_' + c]:.3f}" for c in ('mean_it_mw', 'peak_it_mw', 'placement_mean_facility_mw_median', 'placement_peak_facility_mw_median')]
        lines.append('| ' + ' | '.join([row['scenario_zh'], *values]) + ' |')
    lines += ['', '## 第四章同源典型日', '',
        '第四章直接读取本目录小时文件，按完整原生窗口选日。无日期来源先限定完整窗口的放置，再取设施均值最接近12次中位数者；同一GPU需求按GPU主导型共用放置与源日。候选日电量为中位数±2%，少于5天放宽至±5%，候选中按与全部完整日平均24小时设施曲线的RMSE最小选日。', '',
        '| 情境 | 源日 | 放置 | 设施峰值MW | 日电量MWh | 当日峰谷比 |', '| --- | --- | --- | ---: | ---: | ---: |']
    for row in selected['scenarios']:
        placement_name = '源月日对应' if row['placement'] == 'dated' else f"{int(row['placement'][1:])}月初"
        lines.append(f"| {row['scenario_name']} | {row['source_date']} | {placement_name} | {row['peak_mw']:.3f} | {row['daily_facility_mwh']:.3f} | {row['peak_valley_ratio']:.3f} |")
    lines += ['', '![第四章典型日](selected-days/overview.png)', '',
        'chapter4-comparison.csv保存前版第四章与本版典型日对照，以及本版第四章与第五章完整IT均值、逐小时取值的零差异核对。前版与本版的典型日变化包含模型、气象映射和选日影响。第四章峰谷比为所选日24小时的最大/最小，第五章为整个原生窗口按小时分组平均的比值。', '',
        '## 数据与验证', '',
        'native-summary.csv覆盖44个情境，包含完整IT的均值/峰值/最小值、日内及周内峰谷比、电量，并保存设施各指标12次放置的中位数与范围。all-44-scenarios.csv为精简数值表；placement-summary.csv为参考网格220个逐放置窗口，window-summary-all-locations.csv覆盖25组地点年份、5500个窗口、15,861,600行小时数据。source-hourly保存44条完整IT源曲线，hourly保存参考网格220条气象匹配曲线；城市小时文件保存在服务器data/processed同名目录。', '',
        'manifest.json记录依赖提交、全部输入/输出SHA-256、冻结的#35标定口径及冷却版本。verification.json独立对照上游小时结果，并检查所有窗口、资源、天气、功率分项、热平衡、统计及放置聚合；原有目录逐字节保持。selected-days保留每日候选、诊断、24小时数值与分任务表，任务层加总还原设施功率。', '',
        '## 解释边界', '',
        'Alibaba五天不足以识别长期或星期规律；ASI与Alibaba无真实日期，月初放置和请求相位均为情境假设。MIT使用单月功率标定迁移H100，监测有效覆盖68.33%，合格验证仅3小时。Acme原生需求与监测仅重叠16.496个采样小时、功率报送覆盖11.45%–35.84%，直接按分配GPU反推功率比例没有合格标定小时；采用的集群日形状混合了分配与运行状态，不能识别预训练或检查点的因果日周期。功率变化和低优先级均不提供延后、暂停或恢复的调度权限。', '',
        '## 复现与审阅', '',
        '后台运行analysis/launch_native_combined.sh，读取已冻结#34原生小时输入；只写新的结果与处理目录。运行verify_native_combined.py全量核验，再运行prepare_native_combined_brief.py、plot_native_combined.py及本脚本。Word由docs/job-flexibility-brief/build.js生成，经Word原生导出PDF并逐页检查。', '',
        '依赖审阅顺序：#27；#30→#32、#33；#34、#35、#36；#31；本PR。各依赖分支保持原状，本PR待审。', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report(args.project)
