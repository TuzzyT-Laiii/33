import argparse
import json
from pathlib import Path

import pandas as pd

from native_load_inputs_v3 import RUN, PREVIOUS


def number(value, digits=3):
    return f'{value:.{digits}f}' if pd.notna(value) else '不提供'


def table(frame, columns):
    headers = [label for _, label in columns]
    rows = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for record in frame.to_dict('records'):
        rows.append('| ' + ' | '.join(str(record[name]) for name, _ in columns) + ' |')
    return '\n'.join(rows)


def report(project):
    output = project / 'results' / RUN
    manifest = json.loads((output / 'manifest.json').read_text('utf-8'))
    sources = json.loads((output / 'source-time-bases.json').read_text('utf-8'))
    mit = json.loads((output / 'mit-calibration.json').read_text('utf-8'))
    flexibility = json.loads((output / 'flexibility-source-statistics.json').read_text('utf-8'))
    summary = pd.read_csv(output / 'native-summary.csv')
    comparison = pd.read_csv(output / 'mean-power-comparison.csv')
    cases = pd.read_csv(output / 'placement-summary.csv')
    selected = summary.loc[summary.reference_it_mw.eq(10)].copy()
    selected['IT'] = selected.apply(lambda row: f'{row.mean_it_mw:.3f} / {row.peak_it_mw:.3f}', axis=1)
    selected['facility'] = selected.apply(lambda row: f'{row.placement_mean_facility_mw_median:.3f} / {row.placement_peak_facility_mw_median:.3f}', axis=1)
    selected['diurnal'] = selected.apply(lambda row: f'{row.it_diurnal_peak_valley:.3f} / {row.placement_facility_diurnal_peak_valley_median:.3f}', axis=1)
    selected['facility_range'] = selected.apply(lambda row: f'{row.placement_mean_facility_mw_min:.3f}–{row.placement_mean_facility_mw_max:.3f}', axis=1)
    selected['days'] = selected.native_days.astype(int)
    main_table = table(selected, [('scenario_id', '情境'), ('days', '原生天数'), ('IT', 'IT 平均/峰值 MW'),
        ('facility', '设施平均/峰值 MW'), ('diurnal', '日内峰谷比 IT/设施'), ('facility_range', '设施平均放置最小–最大 MW')])
    dates = pd.DataFrame([{'use': use, 'window': row['source_start'] + ' 至 ' + row['source_end_inclusive'],
                          'hours': row['source_hours'], 'basis': row['time_basis']} for use, row in sources.items()])
    source_table = table(dates, [('use', '用途'), ('window', '完整窗口，末小时含在内'), ('hours', '小时数'), ('basis', '时间基准')])
    monthly = cases.loc[cases.scenario_id.isin(['A-CLD-10MW', 'D-INF-10MW'])].copy()
    monthly['average'] = monthly.mean_facility_mw.map(lambda value: f'{value:.4f}')
    monthly['peak'] = monthly.peak_facility_mw.map(lambda value: f'{value:.4f}')
    monthly['energy'] = monthly.window_facility_mwh.map(lambda value: f'{value:.2f}')
    monthly_table = table(monthly, [('scenario_id', '情境'), ('placement', '月初放置'), ('hours', '实际小时'),
        ('days', '实际天数'), ('average', '设施平均 MW'), ('peak', '设施峰值 MW'), ('energy', '窗口设施 MWh')])
    compared = comparison.loc[comparison.reference_it_mw.eq(10)].copy()
    for label in ('previous_v3_annual', 'v2_annual'):
        for component in ('it', 'facility'):
            compared[label + '_' + component] = compared[label + '_' + component + '_change_pct'].map(lambda value: f'{value:+.2f}%' if pd.notna(value) else '无 v2 数据')
    comparison_table = table(compared, [('scenario_id', '情境'), ('previous_v3_annual_it', 'IT 较之前 v3 全年均值'),
        ('previous_v3_annual_facility', '设施较之前 v3 全年均值'), ('v2_annual_it', 'IT 较 v2 全年均值'), ('v2_annual_facility', '设施较 v2 全年均值')])
    flex_rows = pd.DataFrame([{'use': use, 'days': row['source_days'], 'share': f'{100 * row["candidate_resource_fraction"]:.3f}%',
        'hours': f'{row["candidate_resource_hours"]:,.2f}', 'screen': row['screen']} for use, row in flexibility.items()])
    flex_table = table(flex_rows, [('use', '用途'), ('days', '窗口天数'), ('share', '候选资源份额'), ('hours', '候选资源小时'), ('screen', '筛选与积分范围')])
    sensitivity = pd.read_csv(output / 'sensitivity-summary.csv')
    central = sensitivity.loc[sensitivity.cooling_case.eq('central')].pivot(index='scenario_id', columns='power_sensitivity', values='mean_it_mw')
    sensitivity_rows = pd.DataFrame([{'scenario_id': index, 'central': f'{row.central:.3f}',
        'range': f'{row.ratio_p10:.3f}–{row.ratio_p90:.3f}', 'idle': f'{row.idle_half:.3f}–{row.idle_double:.3f}'} for index, row in central.iterrows()])
    sensitivity_table = table(sensitivity_rows, [('scenario_id', '情境'), ('central', '中央平均 IT MW'), ('range', 'P10–P90 描述性端点 MW'), ('idle', '空载×0.5–2 MW')])
    readme = f'''# 典型负荷曲线 v3：44 个情境的原生窗口

**按用户 2026-11-01 的要求，本版不用全年循环或重放，各来源有多少完整数据就计算多少。** 每个情境的完整 IT 输入只包含一次原生窗口；设施功率按下述气象对应计算。电量只报告实际窗口积分，不换算成年电量。原 v2、之前 v3 全年工作包以及八城市既有结果目录保持原字节。

本版输出不能直接作为需要 8760/8784 小时输入的八城市柔性价值计算输入。接入该下游前需要另行确定研究窗口及模型接口；本次没有改动、重算或替换下游已有结果。当前成果为工程功率情境与柔性候选筛选，没有求解调度、储能或收益。

已覆盖 **44 个风冷情境**、{manifest['location_years']} 个地点×气象年、{manifest['placement_rows']:,} 个气象对应/放置结果、{manifest['weather_matched_hourly_rows']:,} 条小时功率。44 条完整原生 IT 序列合计 {manifest['native_it_rows']:,} 小时。情境采用[确认命名](../../docs/scenario-naming.md)，设备台数及安装参考 IT 预算保持之前 v3，四档规模为 1/10/100/500MW。

## 原生窗口与气象对应

{source_table}

Lisa 已核验 309 天完整小时序列，跨两个来源年份。Acme 已核验 90 天 GPU 分配序列；[官方 Kalos 示例](https://github.com/InternLM/AcmeTrace#1-job-trace)也明确使用 `+00:00` 时间。Lisa 的[官方数据说明](https://zenodo.org/records/13685426)确认 2021 年 12 月至 2022 年 11 月日期范围，源 Parquet 与现有审计均未给出时区；本版按其源月日小时对应目标 UTC 属于时区映射假设，不声称已核实当地时区。MIT 日期由源 epoch 秒形成，保留既有小时网格；功率监测标定与任务分配的实际当地时区关系仍有不确定性。

有日期来源逐小时按**月–日–小时**对应参考网格 2025、八城市 2023–2025 的同一气象时刻。保留源顺序：Lisa 的源 2021 年 12 月后接 2022 年 1 月，对应同一个目标气象年的 12 月后接 1 月，气象时间列因此不按公历递增。冷却模型逐小时无记忆，在连续气象片段内计算后按源顺序连接；没有重复或增加源需求。目标 2024 年的 2 月 29 日没有对应源小时，直接不使用该日气象；源小时数保持不变。当前三类有日期窗口都不含 2 月 29 日。若来源含该日而目标年没有，程序会对应 2 月 28 日同小时并逐行标识，保留原生小时数。

无真实日期的 Alibaba 和 ASI，分别放在每个目标气象年的 12 个月 1 日 00:00 UTC。**统一采用年末截断政策**：只使用该气象年内可用天气及等长的源前缀，不跨下一年、不循环、不补齐。Alibaba 的 12 次放置都是完整 120 小时。ASI 前六个月均完整 4440 小时，7–12 月分别为 4416、3672、2928、2208、1464、744 小时（184、153、122、92、61、31 天）。每次放置独立保存长度、丢弃尾部小时和电量；截断只涉及设施气象匹配，完整 185 天 IT 序列单独保留。

无日期来源的设施主指标为 **12 次实际放置指标的中位数与最小–最大**，各次等权。由于 ASI 后六次只使用源前缀，范围同时包含天气与样本前缀差异，不能作为已识别的季节效应或同一 185 天设施运行的季节区间。小时序列和图件均保留这个边界。

## 10MW 参考网格结果

{main_table}

IT 数值来自完整原生窗口；设施数值对有日期来源为一次月日小时对应，对无日期来源为 12 次放置指标中位数。表中峰值的中位数与平均值的中位数分别统计，不保证来自同一次放置。完整四档结果见 [all-44-scenarios.md](all-44-scenarios.md) 和 [native-summary.csv](native-summary.csv)；每次实际计算的 IT、制冷、设施平均/峰值/最小值、小时极值比、日内/周内比与窗口 MWh 见 [placement-summary.csv](placement-summary.csv)。

日内比为源小时分组平均功率最大/最小。周内形状有日期来源使用**源日期星期**；ASI 仅使用源第零小时起的相对七日分组，没有真实星期含义。Alibaba 只有五天，不提供周内曲线或周内峰谷比。窗口小时极值比与日内分组峰谷比分别报告，未混用。源季节、节假日、时区和采集完整性限制随数据保留。

## 无日期来源的 12 次放置

{monthly_table}

表中列出 A-CLD 与 D-INF 的 10MW 示例；B-INF/C-INF 和所有规模的 12 次结果均在上述 CSV。各指标的中位数、最小、最大同时提供，不能将中位数曲线当作一次真实设施运行。

## 功率链路与实测迁移

`P_IT = P_base + P_CPU + P_GPU`；`P_facility = P_IT + P_cooling + P_other`。沿用之前 v3 的 CPU 响应、目标设备参数、整数台数、风冷模型和冻结气象。参考网格仍为 g05_r100_c592，气压沿用 101325Pa 假设；八城市使用既有 2023–2025 年温度、露点和气压，文件 SHA-256 均核对，新加坡仍以柔佛陆地占优格点代表区域气候。没有重新下载气象或重新指定库存分母。

SAI 保持 237 天 GPU 与宿主 CPU 分配，分别使用原观测峰值 428.92 GPU 和 7394.9089 Slurm CPU 调度单位归一化。六月窗口完成版 `partial_data=false`，完成 {mit['completed_objects']}/{mit['total_objects']} 个文件；前半月分配 GPU 时间加权功率 {mit['allocated_power_W_per_gpu']:.3f}W/张，对 250W 参考 TDP 比为 {mit['measured_power_to_tdp_ratio']:.4f}，空转代理 {mit['idle_W_per_gpu']:.3f}W/张。迁移公式保持 `n_GPU × [a × TDP_H100 × r_June + (1−a) × TDP_H100 × idle_V100/250]`，不再次乘利用率或六月形状。

六月有效监测覆盖 {100 * mit['monitoring_coverage']:.2f}%，合格验证仅 {mit['qualified_hours']} 小时，物理全集群完整实测小时为 {mit['fully_measured_cluster_hours']}。V100 功率/TDP 和空转/TDP 比迁移至 H100 属于假设，未标定等吞吐或目标设备同步功率，不能把文件处理完成理解为完整物理监测。

{sensitivity_table}

P10/P90 为有效设备小时功率的描述性端点，空载倍数为工程敏感性；均不表示全年或原生窗口功率置信区间。P10 分配端点低于所用空载代理，该敏感性包含跨状态迁移的不确定性。低/中央/高风冷参数与六种 SAI 功率假设的 54 组结果见 [sensitivity-summary.csv](sensitivity-summary.csv)。MIT 仅提交聚合数据，遵循来源 CC BY-NC-ND 4.0 的使用边界，不提交逐设备或逐作业记录。

## 平均功率与历史口径比较

{comparison_table}

对比统一采用平均功率，不比较不同长度的总电量。IT 变化来自完整原生窗口均值与旧循环尾部加权均值的差异；设施变化还包括真实月日天气对应或 12 次放置与年末截断。相对 v2 的 SAI 还包含六月标定统计和空载迁移，INF 还包含业务范围、归一化与来源窗口变化。上述结果属于描述性比较，不能将差异归因于同一真实设施的历史变化或某个单独参数。逐情境精度见 [mean-power-comparison.csv](mean-power-comparison.csv)。

## 原生窗口柔性提示

{flex_table}

资源小时单位：HPC/CLD 为 CPU 核心小时，SAI/TRN/INF 为 GPU 小时。HPC、SAI、TRN 已按执行区间与各自完整窗口的精确交集重新积分，分母与采用的原生需求总资源小时核对。运行时长阈值按原始完整作业执行时长判断；运行时长不等于可推迟时长。CLD 未对账量仍为 {100 * flexibility['CLD']['unreconciled_signed_share']:.2f}%；其业务类型未确认。INF 保留全业务超过库存的质量标识，推理子集自身超过库存为零；未确认零资源小时保留一次并标识。

[flexibility-hints.csv](flexibility-hints.csv)为全部情境提供候选份额；CLD/INF 另提供“同小时移出全部候选、设备仍通电”的原生窗口 IT 功率差与窗口电量差。ASI 全业务有 409 小时归属量超过库存，该异常不能直接归给推理子集。该反事实没有计算补做、容量、期限或暂停恢复成本，不表示节电、保证削峰或收益。所有来源的已验证可调比例下界保持零。

## 图件与输出字段

六组图均为 10MW 参考网格；中文确认命名、300dpi PNG、嵌入字体 PDF。

1. `figure1-native-day.*`：完整原生 IT 典型日及设施/制冷放置中位数、最小–最大。
2. `figure2-native-week.*`：源星期或相对七日形状；A-CLD 面板明确只有五天。
3. `figure3-native-timeseries.*`：各自原生时长的小时 IT 与设施；无日期来源阴影为该源小时仍可用放置的范围，尾部放置数量减少，逐小时数量随数据保存。聚合曲线不对应单次运行。
4. `figure4-native-components.*`：窗口平均功率分项；无日期来源为 12 个实际窗口分项均值的算术平均，保证分项加总，口径与设施中位数分别标注。
5. `figure5-native-mean-comparison.*`：与历史 v2、之前 v3 全年均值的平均功率比较。
6. `figure6-native-sai-sensitivity.*`：237 天分配下的六月迁移与空载敏感性。

`source-hourly/` 的 44 个 Parquet 是各情境恰好一次完整原生 IT 序列，`source_hour_index` 从零连续且唯一；没有气象日期时保留相对时钟。`hourly/` 保存 220 个参考网格气象对应/放置结果。这里 `time_utc`/`weather_time_utc` 均表示所取气象时刻，`source_clock` 表示来源时钟，按 `source_hour_index` 阅读序列；Lisa 跨源年份时气象时刻不按日期排序。

所有八城市×三年的同口径汇总见 [window-summary-all-locations.csv](window-summary-all-locations.csv) 与 [placement-aggregate-all-locations.csv](placement-aggregate-all-locations.csv)。`native-summary.csv` 的无前缀 IT 字段属于完整原生窗口；设施与制冷字段为放置指标中位数，`placement_*_median/min/max` 明确提供各次放置统计。PUE 使用每次实际窗口的 IT 与设施配对积分计算，再分别统计中位数和范围。城市小时文件保留在服务器 `data/processed/{RUN}/`；参考小时与聚合结果发布到 `results/{RUN}/`。SHA-256、来源时间基准、截断信息及计算进度均随工作包保存。

## 复算、核验与审阅

原生计算入口为 `analysis/build_native_load_curves_v3.py`，沿用 `analysis/requirements-load-curves-v3.txt`。参数指定项目、原始处理数据项目、冻结 v2 小时目录、八城市天气目录、MIT 原生小时表和 MIT 私有参考目录。服务器使用 `setsid nohup bash analysis/launch_native_load_curves_v3.sh <run_root> <source_project>` 后台运行；日志、进度、锁和临时文件均在共享卷。

`verify_native_load_curves_v3.py` 核验原始来源哈希、44 个原生序列小时数、月日小时与 12 次放置映射、跨年与闰日、截断、温度/气压、功率守恒、窗口积分及放置统计。图件另做嵌入字体、文本边界、文件可读性和名称检查；遵循用户规范不调用视觉功能。历史目录按输入提交 `{manifest['input_commit']}` 逐文件核对原字节。校验边界和结果见 validation 与 delivery-audit JSON。

本次是 PR #34 当前需求的原生窗口修改，继续使用同一分支。依赖及审阅顺序为 #27 命名；#30→#32 和 #33 输入；最后 #34。保持待审，合并须有用户明确批准。

来源入口：[Alibaba 聚合与业务边界](../alibaba-cpu-cld-20261031/README.md)、[ASI 185 天输入与质量](../asi-185day-workload-types-20261031/README.md)、[MIT 来源与功率标定说明](../../docs/mit-gpu-priority-month.md)。之前的[全年 v3 工作包](../{PREVIOUS}/README.md)按原字节保留，其全年接口不适用于本版原生窗口结论。
'''
    (output / 'README.md').write_text(readme, encoding='utf-8')
    all_rows = summary.copy()
    for field, label in [('mean_it_mw', 'it_average'), ('peak_it_mw', 'it_peak'), ('minimum_it_mw', 'it_minimum'), ('window_it_mwh', 'it_mwh')]:
        all_rows[label] = all_rows[field].map(lambda value: number(value, 4))
    for field, label in [('mean_facility_mw', 'facility_average'), ('peak_facility_mw', 'facility_peak'), ('minimum_facility_mw', 'facility_minimum'), ('window_facility_mwh', 'facility_mwh'), ('facility_diurnal_peak_valley', 'facility_day')]:
        all_rows[label] = all_rows.apply(lambda row: f'{row["placement_" + field + "_median"]:.4f} [{row["placement_" + field + "_min"]:.4f}–{row["placement_" + field + "_max"]:.4f}]', axis=1)
    all_rows['it_day'] = all_rows.it_diurnal_peak_valley.map(lambda value: number(value, 4))
    all_rows['it_week'] = all_rows.it_weekly_peak_valley.map(lambda value: number(value, 4))
    all_rows['facility_week'] = all_rows.placement_facility_weekly_peak_valley_median.map(lambda value: number(value, 4))
    full = '# 全部 44 个情境的原生窗口指标\n\n参考网格 2025 年；IT 使用完整原生窗口。设施格式为中位数 [放置最小–最大]；有日期来源仅一次对应。ASI 年末放置截断，设施窗口长度与原生 IT 全窗口分别记录；各次完整精度见 placement-summary.csv。电量均为实际窗口积分。A-CLD 只有五日，不报告周内形状。\n\n'
    full += table(all_rows, [('scenario_id', '情境'), ('native_days', '原生天数'), ('it_average', 'IT 平均 MW'), ('it_peak', 'IT 峰值 MW'), ('it_minimum', 'IT 最小 MW'),
        ('facility_average', '设施平均 MW'), ('facility_peak', '设施峰值 MW'), ('facility_minimum', '设施最小 MW'),
        ('it_day', 'IT 日内比'), ('facility_day', '设施日内比'), ('it_week', 'IT 周内比'), ('facility_week', '设施周内比中位数'),
        ('it_mwh', '原生 IT MWh'), ('facility_mwh', '实际窗口设施 MWh')]) + '\n'
    (output / 'all-44-scenarios.md').write_text(full, encoding='utf-8')
    print('原生窗口中文报告和44情境指标表已生成', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    report(parser.parse_args().project.resolve())
