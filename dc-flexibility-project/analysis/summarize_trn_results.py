import argparse
import json
from pathlib import Path

import pandas as pd


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(value) for value in row) + ' |' for row in rows]) + '\n'


def pair(row, name, decimals=3):
    return f'{row["v3_" + name]:.{decimals}f} → {row["new_" + name]:.{decimals}f}'


def main(project):
    output = project / 'results/trn-power-model-20261101'
    comparison = pd.read_csv(output / 'v3-comparison.csv')
    selected = comparison.loc[comparison.reference_it_mw.eq(10)].sort_values('scenario')
    rows = [[row.scenario, pair(row, 'mean_it_mw'), pair(row, 'peak_it_mw'), pair(row, 'it_diurnal_peak_valley'), pair(row, 'task_share_of_it_pct', 2)]
            for _, row in selected.iterrows()]
    text = '参考网格、2025 年、10MW 参考 IT 规模。箭头左侧为 v3，右侧为修正模型。年均和峰值单位均为 MW。\n\n'
    text += table(['情境', '年均 IT', '峰值 IT', 'IT 日内峰谷比', '任务增量占 IT（%）'], rows)
    text += '\n'
    text += table(['情境', '年均设施', '峰值设施', '设施日内峰谷比', '年化 PUE'],
                  [[row.scenario, pair(row, 'mean_facility_mw'), pair(row, 'peak_facility_mw'), pair(row, 'facility_diurnal_peak_valley'), pair(row, 'annual_pue')]
                   for _, row in selected.iterrows()])
    text += '\n日内峰谷比为完整生成年度按 UTC 小时平均后，24 个均值的最大值/最小值；峰值为全年小时最大值。任务增量为 IT 减去所有服务器开机且无任务时的 IT 功率，份额按年度能量计算。\n'
    (output / 'comparison.md').write_text(text, encoding='utf-8')

    sensitivity = pd.read_csv(output / 'sensitivity-summary.csv')
    labels = {'central': '中央值', 'gpu_low': 'GPU 比例 0.65', 'gpu_high': 'GPU 比例 0.95', 'gpu_only': '仅修正 GPU、保留 v3 CPU', 'idle_low': '空闲低值', 'idle_high': '空闲高值'}
    rows = [[row.scenario, labels[row.sensitivity], f'{row.mean_it_mw:.3f}', f'{row.mean_facility_mw:.3f}', f'{row.task_share_of_it_pct:.2f}']
            for _, row in sensitivity.loc[sensitivity.use.eq('TRN')].iterrows()]
    (output / 'sensitivity.md').write_text('10MW 参考规模；空闲敏感性保持 GPU 比例 0.80，GPU 敏感性保持中央空闲参数。\n\n' +
        table(['情境', '敏感性', '年均 IT（MW）', '年均设施（MW）', '任务增量占 IT（%）'], rows), encoding='utf-8')

    after = pd.read_csv(output / 'annual-summary-all-locations.csv')
    before = pd.read_csv(output / 'v3-summary-all-locations.csv')
    keys = ['location_id', 'year', 'scenario_id']
    metrics = ['mean_it_mw', 'peak_it_mw', 'mean_facility_mw', 'peak_facility_mw', 'it_diurnal_peak_valley', 'facility_diurnal_peak_valley', 'task_share_of_it_pct', 'annual_pue']
    merged = before[keys + metrics].merge(after[keys + metrics], on=keys, suffixes=('_v3', '_new'), validate='one_to_one')
    if len(merged) != 1100:
        raise ValueError('城市年度新旧比较数量不守恒')
    merged.to_csv(output / 'v3-comparison-all-locations.csv', index=False)

    acme = output / 'acme'
    telemetry_run = acme / 'telemetry-run.json'
    if telemetry_run.exists() and json.loads(telemetry_run.read_text('utf-8'))['complete']:
        data = pd.read_csv(acme / 'gpu-telemetry-summary.csv')
        rows = []
        for _, row in data.iterrows():
            day = pd.read_csv(acme / f'{row.cluster}-{row.metric}-diurnal.csv')
            rows.append([row.cluster, row.metric, row.unit, f'{row["mean"]:.2f}', f'{row.p10:.2f}', f'{row.p50:.2f}', f'{row.p90:.2f}', f'{100 * row.valid_fraction:.2f}', f'{day["mean"].max() / day["mean"].min():.3f}'])
        text = '全部有效 GPU 监测样本，包含已分配与未分配设备。分位数为直方图箱中心近似：功率箱宽 1W、利用率箱宽 1 个百分点、显存箱宽 64MiB。覆盖列比例包含设备增减造成的结构性空白，不代表物理集群监测完整率。\n\n'
        text += table(['集群', '指标', '单位', '均值', 'P10', 'P50', 'P90', '有效单元格（%）', '日内均值峰谷比'], rows)
        text += '\n原始时间窗及间断、零值比例、超 400W 比例、SHA-256 见各 summary.json。GPU 设备列在整个窗口累积，不能解释为同一时点的 GPU 保有量。\n'
        (output / 'acme-statistics.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    main(parser.parse_args().project.resolve())
