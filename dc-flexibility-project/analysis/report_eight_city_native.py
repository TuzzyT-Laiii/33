import argparse
from hashlib import file_digest
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from plot_flex_classes import COLORS, LAYERS, handles
from plot_native_combined import save as save_figure
from plot_scenario_typical_days import setup


RUN = 'eight-city-native-flex-20261102'
ORDER = ['A-HPC', 'A-CLD', 'B-SAI', 'C-SAI', 'D-SAI', 'B-TRN', 'C-TRN', 'D-TRN', 'B-INF', 'C-INF', 'D-INF']
REPRESENTATIVES = ['A-HPC', 'A-CLD', 'D-SAI', 'D-TRN', 'D-INF']


def digest(path):
    with path.open('rb') as handle:
        return file_digest(handle, 'sha256').hexdigest()


def plot(output, climate, city, typical):
    setup()
    plt.rcParams['pdf.fonttype'] = 42
    matrix = city.pivot(index='scenario_id', columns='city_id', values='window_pue').loc[[s+'-10MW' for s in ORDER], climate.city_id]
    labels = city.drop_duplicates('scenario_id').set_index('scenario_id').scenario_zh.reindex(matrix.index)
    fig, ax = plt.subplots(figsize=(16, 4.7))
    fig.subplots_adjust(left=.32, right=.91, top=.85, bottom=.12)
    cmap = LinearSegmentedColormap.from_list('cooling_pue', ['#F3F6FA', '#9CABC2', '#35414D'])
    image = ax.imshow(matrix, aspect='auto', cmap=cmap)
    ax.set_xticks(range(8), climate.city_zh, fontsize=11)
    ax.set_yticks(range(11), labels, fontsize=10.5)
    for i in range(11):
        for j in range(8):
            normalized = (matrix.iloc[i, j]-matrix.min().min())/(matrix.max().max()-matrix.min().min())
            ax.text(j, i, f'{matrix.iloc[i,j]:.3f}', ha='center', va='center', fontsize=10.5,
                color='white' if normalized > .66 else '#263441')
    colorbar = fig.colorbar(image, ax=ax, fraction=.035, pad=.025)
    colorbar.set_label('窗口电量比 PUE', fontsize=11)
    colorbar.ax.tick_params(labelsize=10)
    fig.text(.32, .95, '八城市原生窗口 PUE', fontsize=18)
    fig.text(.32, .89, '逐年放置中位数的三年均值；非全年PUE  ·  深色表示更高的非IT负担', fontsize=10, color='#535B63')
    save_figure(fig, output, 'city-pue-heatmap')
    compared = ['lulea', 'riyadh', 'singapore']
    names = climate.set_index('city_id').city_zh
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    fig.subplots_adjust(left=.055, right=.985, top=.70, bottom=.20, wspace=.23)
    upper = float(np.ceil(typical.loc[typical.city_id.isin(compared), 'facility_mw'].max()*1.04))
    for ax, location in zip(axes, compared):
        frame = typical.loc[typical.city_id.eq(location)].sort_values('hour_utc')
        values = [frame[label].to_numpy() for label in LAYERS]
        ax.stackplot(np.arange(25), *[np.r_[value, value[-1]] for value in values], step='post', colors=[COLORS[label] for label in LAYERS], linewidth=0)
        ax.step(np.arange(25), np.r_[frame.facility_mw, frame.facility_mw.iloc[-1]], where='post', color='#35414D', lw=.7)
        ax.set(xlim=(0,24), ylim=(0,upper), xticks=[0,6,12,18,24])
        ax.set_title(names.loc[location], fontsize=13, loc='left')
        ax.set_xlabel('来源UTC小时', fontsize=11)
        ax.tick_params(labelsize=10)
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('设施功率（MW）', fontsize=11)
    fig.text(.055, .985, 'GPU 主导型·大模型训练/开发·10MW：同一来源日的气候比较', fontsize=16, va='top')
    fig.legend(handles=handles(), loc='upper right', bbox_to_anchor=(.99,.915), ncol=3, fontsize=10)
    fig.text(.055, .82, 'Seren来源日2023-05-31；分别对应2023–2025同月日小时，取逐小时三年平均', fontsize=10, color='#535B63')
    fig.text(.055, .07, '四类及基础与空载IT按第五章同日逐小时份额归因；冷却等非IT单列，六层相加为设施功率。', fontsize=11, color='#535B63')
    save_figure(fig, output, 'city-typical-day-comparison')


def report(project):
    output = project / 'results' / RUN
    climate = pd.read_csv(output / 'climate-summary.csv', float_precision='round_trip').sort_values('mean_temperature_c')
    city = pd.read_csv(output / 'city-summary.csv', float_precision='round_trip')
    daily = pd.read_csv(output / 'typical-day-city-year.csv', float_precision='round_trip')
    numeric = ['it_mw', 'facility_mw', 'temperature_c', 'dew_point_c', *LAYERS]
    typical = daily.groupby(['city_id', 'city_zh', 'scenario_zh', 'source_date', 'hour_utc'], sort=False)[numeric].mean().reset_index()
    np.testing.assert_allclose(typical[LAYERS].sum(axis=1), typical.facility_mw, atol=1e-10)
    typical.to_csv(output / 'typical-day-city-three-year.csv', index=False, encoding='utf-8-sig')
    composition = []
    for row in city.to_dict('records'):
        for label in LAYERS:
            value = row['mean_non_it_mw'] if label == '冷却等非 IT' else row[label]
            composition.append({'city_id': row['city_id'], 'city_zh': row['city_zh'], 'scenario_id': row['scenario_id'],
                'scenario_zh': row['scenario_zh'], 'layer': label, 'mean_facility_component_mw': value,
                'share_of_facility_pct': 100*value/row['mean_facility_mw'], 'attribution': '冷却非IT单列；任务及基础按第五章IT份额归因'})
    pd.DataFrame(composition).to_csv(output / 'facility-composition.csv', index=False, encoding='utf-8-sig')
    cold, hot = climate.city_id.iloc[0], climate.city_id.iloc[-1]
    effects = []
    for sid, group in city.groupby('scenario_id', sort=False):
        low, high = group.loc[group.mean_facility_mw.idxmin()], group.loc[group.mean_facility_mw.idxmax()]
        lookup = group.set_index('city_id')
        effects.append({'scenario_id': sid, 'scenario_zh': low.scenario_zh, 'minimum_city': low.city_zh, 'maximum_city': high.city_zh,
            'minimum_facility_mw': low.mean_facility_mw, 'maximum_facility_mw': high.mean_facility_mw,
            'facility_range_mw': high.mean_facility_mw-low.mean_facility_mw,
            'facility_range_pct': 100*(high.mean_facility_mw/low.mean_facility_mw-1),
            'coldest_city': lookup.loc[cold, 'city_zh'], 'warmest_city': lookup.loc[hot, 'city_zh'],
            'coldest_facility_mw': lookup.loc[cold, 'mean_facility_mw'], 'warmest_facility_mw': lookup.loc[hot, 'mean_facility_mw'],
            'cold_to_warm_difference_mw': lookup.loc[hot, 'mean_facility_mw']-lookup.loc[cold, 'mean_facility_mw'],
            'minimum_pue': float(group.window_pue.min()), 'maximum_pue': float(group.window_pue.max()),
            'minimum_non_it_mw': float(group.mean_non_it_mw.min()), 'maximum_non_it_mw': float(group.mean_non_it_mw.max())})
    effect = pd.DataFrame(effects)
    effect.to_csv(output / 'climate-effects.csv', index=False, encoding='utf-8-sig')
    climate_rows = [['城市（由冷到热）', '平均气温\n°C', '平均露点\n°C', '气温≥30°C\n小时%', '露点≥20°C\n小时%']]
    for row in climate.to_dict('records'):
        climate_rows.append([row['city_zh'], f'{row["mean_temperature_c"]:.1f}', f'{row["mean_dew_point_c"]:.1f}', f'{row["hot_30c_pct"]:.1f}', f'{row["humid_dew20c_pct"]:.1f}'])
    lookup = city.set_index(['city_id', 'scenario_id'])
    scenario_names = city.drop_duplicates('scenario_id').set_index('scenario_id').scenario_zh
    facility_rows = [['城市', *[scenario_names.loc[s+'-10MW'] for s in REPRESENTATIVES]]]
    for row in climate.to_dict('records'):
        facility_rows.append([row['city_zh'], *[f'{lookup.loc[(row["city_id"],s+"-10MW"), "mean_facility_mw"]:.3f}' for s in REPRESENTATIVES]])
    largest = effect.loc[effect.facility_range_mw.idxmax()]
    relative = effect.loc[effect.facility_range_pct.idxmax()]
    training = effect.set_index('scenario_id').loc['D-TRN-10MW']
    conclusion = f'城市间设施平均功率的绝对差异以{largest.scenario_zh}最大，为{largest.facility_range_mw:.3f}MW（{largest.minimum_city}至{largest.maximum_city}）；相对差异以{relative.scenario_zh}最大，为{relative.facility_range_pct:.1f}%。'
    comparison = f'同一GPU主导型训练情境，{training.coldest_city}至年平均最热的{training.warmest_city}设施均值由{training.coldest_facility_mw:.3f}增至{training.warmest_facility_mw:.3f}MW，差{training.cold_to_warm_difference_mw:.3f}MW；城市间冷却等非IT为{training.minimum_non_it_mw:.3f}–{training.maximum_non_it_mw:.3f}MW。'
    brief = {'climate_rows': climate_rows, 'facility_rows': facility_rows, 'conclusion': conclusion, 'comparison': comparison,
        'coldest_city': str(climate.city_zh.iloc[0]), 'warmest_city': str(climate.city_zh.iloc[-1]),
        'scope': '同第五章原生窗口和组合模型，使用冻结2023–2025逐小时气温、露点、气压；冷却为cooling-20261028-v2风冷。',
        'aggregation': '主表每年对12次实际放置的设施均值取中位数，再等权平均三年；有日期来源每年一次对应。范围保存三年全部放置极值。PUE为各窗口设施/IT电量比的逐年放置中位数再平均，非全年PUE。',
        'composition': '四类及基础空载IT按第五章原生窗口份额归因城市IT，冷却等非IT单列；三城典型日采用第五章同日逐小时IT份额。',
        'limitations': '仅三年气象、各来源不同原生窗口；无日期来源的放置和年末截断影响比较。新加坡采用1.50°N、104.00°E柔佛陆地占优格点（陆地比例0.729），为区域代理。风冷采用水侧经济器和水冷冷机，室外适用−40至50°C；设备、湿度及压降参数为工程情景，未作目标设施实测拟合。',
        'flexibility': 'IT推迟可能同时改变冷却功率，方向和幅度取决于执行时刻的温湿度与设备负载；此处未改变调度，不能据此给出柔性可调电量。'}
    (output / 'brief-summary.json').write_text(json.dumps(brief, ensure_ascii=False, indent=2), encoding='utf-8')
    plot(output, climate, city, typical)
    lines = ['# 八城市原生窗口负荷、冷却与四类任务构成', '', brief['scope'], '', '## 气候概况', '',
        '每城冻结气象覆盖2023-01-01至2025-12-31 UTC共26304小时；平均气温、露点和阈值小时比例均按实际小时等权。露点≥20°C表示高水汽含量，不等同于相对湿度阈值。气压进入湿空气性质及风机功率计算。', '',
        '| ' + ' | '.join(x.replace('\n','') for x in climate_rows[0])+' |', '| '+' | '.join(['---']*5)+' |']
    lines += ['| '+' | '.join(row)+' |' for row in climate_rows[1:]]
    lines += ['', '## 10MW设施平均功率（MW）', '', brief['aggregation'], '',
        '正文取纯CPU型科研计算、纯CPU型云通用计算及GPU主导型科研AI/训练/推理五个代表组合；全部8城市×11情境88行及264个城市年份情境、1320个实际窗口见city-summary.csv、year-summary.csv、case-summary.csv。', '',
        '| '+' | '.join(facility_rows[0])+' |', '| '+' | '.join(['---']*6)+' |']
    lines += ['| '+' | '.join(row)+' |' for row in facility_rows[1:]]
    lines += ['', '![PUE热力图](city-pue-heatmap.png)', '', conclusion, '', comparison, '',
        '## 四类与冷却构成', '', brief['composition'], '',
        '设施分解为四类IT增量、基础与空载IT、冷却等非IT六层。固定IT份额是核算归因，不声称各城市具有相同的实际业务构成；冷却未分配给可推迟类。每年偶数个放置的设施中位数取中间两次，同批IT和非IT指标配对平均，确保六层守恒，再等权汇总三年；PUE独立按窗口比值取中位数。', '',
        'facility-composition.csv提供88组合×6层的平均MW和设施占比；typical-day-city-year.csv保留24个城市年份的同一训练来源日，typical-day-city-three-year.csv为三年逐小时均值。图取吕勒奥、利雅得和新加坡，源日固定2023-05-31，无重新选日；城市三年图不代表特定年份的实测设施曲线。', '',
        '![典型日气候对比](city-typical-day-comparison.png)', '', brief['flexibility'], '',
        '## 原生窗口、复用与验证', '',
        'Lisa309天、Alibaba120小时、MIT237天、Seren2057个合格小时、ASI185天；不循环、不补全年。日期来源按源月日小时对应各城市目标年；无日期来源在12个月初放置、年末截断，范围含天气及源前缀变化。Seren189内部缺口小时、54连续片段保持。', '',
        '非训练1248个10MW城市窗口复用组合模型原生结果（#34窗口、#35推理、#36 CPU），训练72个窗口复用#38 Seren正文main；总1320份小时文件逐份核对原manifest SHA-256、温度/露点/气压对应、原生长度、日期或放置及截断、功率电量汇总与IT/冷却/其他分项守恒。8份冻结天气逐份匹配#34/#38相同输入SHA-256，三年UTC索引完整。24个GPU主导型Seren城市年份窗口另用当前同参数冷却模型重算核对。', '',
        'verification.json记录上游小时和气象指纹、冷却复算误差、输入溯源及方法。manifest.json保存本机文档输入和旧目录指纹；brief-audit.json记录独立汇总及Word/PDF检查。新章节主表选5代表，PUE热力图包含全部11情境。', '',
        '## 局限', '', brief['limitations'], '',
        '## 复现与审阅', '',
        'collect_eight_city_native.py --prepare准备冻结索引；launch_eight_city_native.sh在共享卷以setsid nohup后台核验并输出进度；report_eight_city_native.py汇总、制图及生成Word输入；build.js生成简表，Word原生导出PDF逐页检查。计算仅读取旧目录，所有新输出独立保存。MIT流式目录、STOP及看门狗保持原状。', '',
        '同一PR #37继续审阅，依赖#27；#30→#32、#33；#34、#35、#36→#38；#31；#37。不合并。', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    verification = json.loads((output / 'verification.json').read_text('utf-8'))
    for path, value in verification['input_provenance']['source_sha256'].items():
        assert digest(project / path) == value, path
    old_files = {path.relative_to(project).as_posix(): digest(path)
        for folder in ('flex-class-20261101', 'typical-load-curves-20261101-v4-seren')
        for path in (project / 'results' / folder).rglob('*') if path.is_file()}
    manifest = {'base_commit': 'f1588457559b2e2e5f5c9d2216c1a586e0d1654a',
        'source_sha256': verification['input_provenance']['source_sha256'],
        'previous_result_files_sha256': old_files,
        'server_collect_code_sha256': verification['code_sha256'],
        'code_sha256': {name: digest(project / 'analysis' / name) for name in ('collect_eight_city_native.py', 'report_eight_city_native.py')},
        'fifth_palette_sha256': digest(project / 'analysis/plot_flex_classes.py'),
        'composition_palette': COLORS, 'representative_scenarios': REPRESENTATIVES,
        'aggregation': verification['aggregation'], 'verification_sha256': digest(output / 'verification.json')}
    assert manifest['server_collect_code_sha256'] == manifest['code_sha256']['collect_eight_city_native.py']
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'rows': facility_rows, 'conclusion': conclusion, 'comparison': comparison}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    report(parser.parse_args().project)
