import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def describe(values):
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError('统计样本为空或含非有限值')
    return dict(count=len(array), minimum=float(array.min()), p10=float(np.quantile(array, .1)),
                median=float(np.median(array)), p90=float(np.quantile(array, .9)), maximum=float(array.max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--gpu', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.input
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    paths = dict(capacity=root / 'capacity/global-capacity.parquet',
                 price=root / 'price/electricity-price.parquet',
                 physical=root / 'physical/summary.csv', gpu=args.gpu)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'所需完整计算结果缺失: {path}')
    physical = pd.read_csv(paths['physical'])
    if len(physical) != 44 or physical.events.sum() != 170284 or physical.max_residual.max() > 1e-6:
        raise ValueError('持续调节结果未通过覆盖或精度核验')
    shutil.copy2(paths['physical'], output / 'physical-summary.csv')
    capacity = pd.read_parquet(paths['capacity'])
    if (len(capacity) != 53262 or capacity.duplicated(['grid_id', 'scenario']).any() or
            not (capacity.groupby('scenario').size() == 4842).all() or
            not np.isfinite(capacity.select_dtypes(include='number').to_numpy()).all() or
            capacity.max_residual.max() > 1e-5):
        raise ValueError('全球减配结果配对覆盖不完整')
    if (capacity.net_saving_lower_usd_mwh > capacity.net_saving_upper_usd_mwh + 1e-6).any():
        raise ValueError('全球净收益界限颠倒')
    capacity['interval_width_usd_mwh'] = capacity.net_saving_upper_usd_mwh - capacity.net_saving_lower_usd_mwh
    capacity['equipment_share_of_feasible_saving'] = capacity.equipment_saving_feasible_usd_year / (
        capacity.E0_mwh_year * capacity.net_saving_lower_usd_mwh)
    cap_summary = capacity.groupby('scenario').agg(
        grids=('grid_id', 'size'), feasible_median_usd_mwh=('net_saving_lower_usd_mwh', 'median'),
        upper_median_usd_mwh=('net_saving_upper_usd_mwh', 'median'),
        feasible_minimum_usd_mwh=('net_saving_lower_usd_mwh', 'min'),
        interval_median_usd_mwh=('interval_width_usd_mwh', 'median'),
        interval_max_usd_mwh=('interval_width_usd_mwh', 'max'),
        equipment_share_median=('equipment_share_of_feasible_saving', 'median'),
        idle_saving_median_mwh_year=('idle_energy_reduction_mwh_year', 'median'),
        servers_A=('servers_A', 'first'), servers_C_feasible=('servers_C_feasible', 'first'),
        servers_C_lower=('servers_C_lower', 'first'), max_residual=('max_residual', 'max'))
    cap_summary.to_csv(output / 'global-capacity-summary.csv')
    price = pd.read_parquet(paths['price'])
    keys = ['grid_id', 'scenario', 'multiplier']
    if len(price) != 450306 or price.duplicated(keys + ['mode']).any() or price.max_residual.max() > 1e-5:
        raise ValueError('全球电价结果覆盖或精度不满足要求')
    if not np.isfinite(price.select_dtypes(include='number').to_numpy()).all():
        raise ValueError('全球电价结果存在非有限数值')
    if (set(price.multiplier) != {.5, 1., 1.5} or
            not set(price['mode']).issubset({'baseline', 'coordination', 'sell_all'}) or
            (price.scenario.str.startswith('A-') & price['mode'].eq('sell_all')).any()):
        raise ValueError('全球电价包含未定义的情景或倍数')
    coverage = price.groupby(['grid_id', 'scenario']).size().rename('rows').reset_index()
    coverage = capacity[['grid_id', 'scenario']].merge(coverage, how='outer',
                                                      validate='one_to_one', indicator=True)
    expected_rows = np.where(coverage.scenario.str.startswith('A-'), 6, 9)
    if not coverage['_merge'].eq('both').all() or not np.array_equal(coverage.rows, expected_rows):
        raise ValueError('全球电价与减配的网格、机群或情景覆盖不一致')
    for _, group in price.groupby(['grid_id', 'scenario']):
        if group.E0_mwh.max() - group.E0_mwh.min() > 1e-8:
            raise ValueError('电价情景基准用电量不一致')
    paired = price.pivot(index=keys, columns='mode', values='total_cost_usd')
    meta = price.groupby(keys)[['E0_mwh', 'sold_gpu_hours', 'hours']].max()
    paired = paired.join(meta)
    paired['coordination_usd_mwh'] = (paired.baseline - paired.coordination) / paired.E0_mwh
    paired['baseline_usd_mwh'] = paired.baseline / paired.E0_mwh
    paired['coordination_share_of_supply_pct'] = 100 * (paired.baseline - paired.coordination) / paired.baseline
    paired['extra_supply_usd_mwh'] = (paired.sell_all - paired.baseline) / paired.E0_mwh
    paired['sell_all_net_usd_mwh'] = (2.9 * paired.sold_gpu_hours - paired.sell_all + paired.baseline) / paired.E0_mwh
    paired = paired.reset_index()
    original = paired[paired.multiplier == 1.].set_index(['grid_id', 'scenario'])
    paired = paired.join(original[['coordination_usd_mwh', 'sell_all_net_usd_mwh']],
                         on=['grid_id', 'scenario'], rsuffix='_reference')
    for field in ['coordination', 'sell_all_net']:
        paired[field + '_change_pct'] = 100 * (paired[field + '_usd_mwh'] / paired[field + '_usd_mwh_reference'] - 1)
    paired.to_parquet(output / 'electricity-price-paired.parquet', index=False)
    price_summary = paired.groupby(['scenario', 'multiplier']).agg(
        grids=('grid_id', 'size'), coordination_median_usd_mwh=('coordination_usd_mwh', 'median'),
        baseline_median_usd_mwh=('baseline_usd_mwh', 'median'),
        coordination_share_of_supply_median_pct=('coordination_share_of_supply_pct', 'median'),
        coordination_change_median_pct=('coordination_change_pct', 'median'),
        sell_all_median_usd_mwh=('sell_all_net_usd_mwh', 'median'),
        sell_all_change_median_pct=('sell_all_net_change_pct', 'median'),
        sell_all_change_p10_pct=('sell_all_net_change_pct', lambda x: x.quantile(.1)),
        sell_all_change_p90_pct=('sell_all_net_change_pct', lambda x: x.quantile(.9)))
    price_summary.to_csv(output / 'electricity-price-summary.csv')
    comparison = capacity.merge(paired[paired.multiplier == 1.], on=['grid_id', 'scenario'],
                                suffixes=['_capacity', '_price'], validate='one_to_one')
    if len(comparison) != len(capacity):
        raise ValueError('全球策略比较缺少配对结果')
    np.testing.assert_allclose(comparison.E0_mwh_year,
                               comparison.E0_mwh * 8760 / comparison.hours_price, rtol=1e-10, atol=1e-6)
    comparison['downsizing_over_coordination'] = comparison.net_saving_lower_usd_mwh / comparison.coordination_usd_mwh
    equipment_annual = (comparison.equipment_saving_feasible_usd_year * comparison.servers_A /
                        (comparison.servers_A - comparison.servers_C_feasible))
    comparison['coordination_share_of_total_pct'] = (100 * comparison.coordination_usd_mwh *
        comparison.E0_mwh_year / (equipment_annual + comparison.baseline * 8760 / comparison.hours_price))
    comparison['selling_over_downsizing'] = comparison.sell_all_net_usd_mwh / comparison.net_saving_lower_usd_mwh
    comparison['selling_over_joint_upper'] = comparison.sell_all_net_usd_mwh / comparison.net_saving_upper_usd_mwh
    comparison['selling_over_coordination'] = comparison.sell_all_net_usd_mwh / comparison.coordination_usd_mwh
    comparison['sale_break_even_coordination'] = (comparison.coordination_usd_mwh + comparison.extra_supply_usd_mwh) * comparison.E0_mwh / comparison.sold_gpu_hours
    comparison['sale_break_even_downsizing'] = (comparison.net_saving_lower_usd_mwh + comparison.extra_supply_usd_mwh) * comparison.E0_mwh / comparison.sold_gpu_hours
    comparison.groupby('scenario').agg(
        downsizing_over_coordination_median=('downsizing_over_coordination', 'median'),
        coordination_share_of_total_median_pct=('coordination_share_of_total_pct', 'median'),
        coordination_share_of_total_maximum_pct=('coordination_share_of_total_pct', 'max'),
        selling_over_downsizing_median=('selling_over_downsizing', 'median'),
        selling_over_joint_upper_median=('selling_over_joint_upper', 'median'),
        selling_over_joint_upper_minimum=('selling_over_joint_upper', 'min'),
        selling_over_coordination_median=('selling_over_coordination', 'median'),
        sale_break_even_coordination_median=('sale_break_even_coordination', 'median'),
        sale_break_even_downsizing_median=('sale_break_even_downsizing', 'median')).to_csv(output / 'strategy-comparison.csv')
    gpu = pd.read_csv(paths['gpu'])
    gpu_audit_path = paths['gpu'].parent / 'audit.json'
    gpu_audit = json.loads(gpu_audit_path.read_text())
    if (gpu_audit['whole_groups'] != 36 or gpu_audit['total_rows'] != 162 or
            gpu_audit['max_independent_residual'] > 1e-6 or
            gpu_audit['result_sha256'] != sha256(paths['gpu'].read_bytes()).hexdigest()):
        raise ValueError('完整作业服务方案尚未通过独立核验')
    gpu_keys = ['scenario', 'phase_hours', 'offline_share', 'throughput_multiplier']
    if len(gpu) != 162 or gpu.duplicated(gpu_keys).any() or gpu.max_residual.max() > 1e-6:
        raise ValueError('独立推理服务结果覆盖或精度不满足要求')
    if not np.isfinite(gpu.select_dtypes(include='number').to_numpy()).all():
        raise ValueError('独立推理服务结果存在非有限数值')
    if (gpu.feasible_share_of_idle > gpu.upper_share_of_idle + 1e-6).any():
        raise ValueError('独立推理服务容量上下界颠倒')
    shutil.copy2(paths['gpu'], output / 'gpu-service-capacity.csv')
    counts = gpu.groupby('scenario').GPUs.first()
    distribution = []
    for scenario, group in comparison[~comparison.scenario.str.startswith('A-')].groupby('scenario'):
        annual = group.E0_mwh * 8760 / group.hours_price
        sell_year = group.sell_all_net_usd_mwh * annual
        coord_year = group.coordination_usd_mwh * annual
        sell_change = 100 * (sell_year / sell_year.median() - 1)
        coord_change = 100 * (coord_year / coord_year.median() - 1)
        distribution.append(dict(scenario=scenario, GPUs=int(counts[scenario]),
                                 idle_share=float((group.sold_gpu_hours / counts[scenario] / group.hours_price).median()),
                                 GPUs_per_mean_facility_MW=float((counts[scenario] * group.hours_price / group.E0_mwh).median()),
                                 sale_annual_p10_pct=float(sell_change.quantile(.1)),
                                 sale_annual_p90_pct=float(sell_change.quantile(.9)),
                                 coordination_annual_p10_pct=float(coord_change.quantile(.1)),
                                 coordination_annual_p90_pct=float(coord_change.quantile(.9))))
    pd.DataFrame(distribution).to_csv(output / 'sale-distribution.csv', index=False)
    clock_review_path = root / 'forecast-clock-review.json'
    clock_review = json.loads(clock_review_path.read_text())
    if [clock_review[key] for key in ['continuous_groups', 'training_groups', 'total_groups']] != [128, 48, 176]:
        raise ValueError('滚动预测真实时间核验覆盖不完整')
    if (clock_review['terminal_rule'] != 'backlog_by_class_no_greater_than_earliest_deadline_execution' or
            clock_review['perfect_deadline_feasible_pairs'] != 176 or
            clock_review['sequential_control_replays'] != 352 or
            clock_review['sequential_cost_max_difference_usd'] > .02):
        raise ValueError('滚动策略的完全预知与顺序调度对照核验未通过')
    sequential_audit_path = root / 'forecast-seq-verified/audit.json'
    sequential_audit = json.loads(sequential_audit_path.read_text())
    if (sha256(sequential_audit_path.read_bytes()).hexdigest() != clock_review['sequential_audit_sha256'] or
            sequential_audit['total_controls'] != 352 or
            sequential_audit['fresh_replays'] != clock_review['sequential_fresh_replays'] or
            sequential_audit['max_difference_usd'] > .02 or sequential_audit['max_residual'] > 2e-6):
        raise ValueError('顺序调度对照审计记录不一致')
    forecast_paths = sorted((root / 'forecast').glob('*.json'))
    if {path.name: sha256(path.read_bytes()).hexdigest() for path in forecast_paths} != clock_review['case_hashes']:
        raise ValueError('滚动预测结果与真实时间核验记录不一致')
    forecasts = [json.loads(path.read_text()) for path in forecast_paths]
    if len(forecasts) != 176 or len({(r['city'], r['scenario'], r['intensity_of_max']) for r in forecasts}) != 176:
        raise ValueError('滚动预测结果覆盖不完整')
    cases = []
    for group in forecasts:
        if {(row['mode'], row['forecast']) for row in group['cases']} != {
                ('seq', 'previous_day'), ('joint', 'previous_day'), ('seq', 'perfect'), ('joint', 'perfect')}:
            raise ValueError('滚动预测缺少配对控制情景')
        arrivals = np.asarray(group['flex_arrival_mwh'])
        perfect = [row for row in group['cases'] if row['forecast'] == 'perfect']
        group['perfect_service_complete'] = all(sum(row['unserved_mwh']) < 1e-5 for row in perfect)
        group['perfect_all_deadlines_met'] = group['perfect_service_complete'] and all(
            sum(row['deadline_missed_mwh']) < 1e-5 for row in perfect)
        if not group['perfect_service_complete']:
            group['perfect_rolling_coordination_usd_mwh'] = None
        for row in group['cases']:
            if row['max_residual'] > 2e-6 or row['conservation_residual_mwh'] > 1e-5:
                raise ValueError('滚动预测约束或任务守恒核验未通过')
            unserved = np.asarray(row['unserved_mwh'])
            missed = np.asarray(row['deadline_missed_mwh'])
            executed = np.asarray(row['executed_mwh'])
            np.testing.assert_allclose(executed + unserved, arrivals, rtol=0, atol=1e-5)
            if not np.isfinite([*unserved, *missed, *executed, row['cost_usd']]).all():
                raise ValueError('滚动预测结果存在非有限数值')
            if (missed > arrivals + 1e-5).any() or min(unserved.min(), missed.min(), executed.min()) < -1e-6:
                raise ValueError('滚动预测逾期或执行量超出到达工作量')
            cases.append(dict(city=group['city'], scenario=group['scenario'], intensity=group['intensity_of_max'],
                              mode=row['mode'], forecast=row['forecast'], E0_mwh=group['E0_mwh'],
                              cost_usd_mwh=row['cost_usd'] / group['E0_mwh'],
                              unserved_mwh=float(unserved.sum()), missed_mwh=float(missed.sum()),
                              missed_fraction=float(missed.sum() / arrivals.sum()),
                              unserved_fraction=float(unserved.sum() / arrivals.sum()),
                              extra_grid_capacity_mw=row['extra_grid_capacity_mw'],
                              demand_wape=row['forecast_wape'][0], solar_wape=row['forecast_wape'][1],
                              wind_wape=row['forecast_wape'][2],
                              service_complete=bool(unserved.sum() < 1e-5),
                              deadlines_met=bool(unserved.sum() < 1e-5 and missed.sum() < 1e-5)))
    cases = pd.DataFrame(cases)
    cases.to_csv(output / 'forecast-cases.csv', index=False)
    forecast = pd.DataFrame([{k: v for k, v in group.items() if k != 'cases'} for group in forecasts])
    forecast.to_csv(output / 'forecast-summary.csv', index=False)
    group_summaries = []
    for level, group in forecast.groupby('intensity_of_max'):
        safe = group[group.all_deadlines_met]
        complete = group[group.service_complete]
        actual_joint = cases[(cases.intensity == level) & (cases.forecast == 'previous_day') & (cases['mode'] == 'joint')]
        safe_keys = set(zip(safe.city, safe.scenario))
        oracle_joint = cases[(cases.intensity == level) & (cases.forecast == 'perfect') & (cases['mode'] == 'joint')]
        oracle_safe = set(zip(oracle_joint[oracle_joint.deadlines_met].city,
                              oracle_joint[oracle_joint.deadlines_met].scenario))
        comparable = group[[key in safe_keys & oracle_safe for key in zip(group.city, group.scenario)]]
        group_summaries.append(dict(intensity=level, total=len(group), complete_pairs=len(complete),
                                    deadline_feasible_pairs=len(safe),
                                    perfect_deadline_feasible_pairs=int(group.perfect_all_deadlines_met.sum()),
                                    positive_deadline_feasible_pairs=int((safe.deadline_feasible_coordination_usd_mwh > 0).sum()),
                                    deadline_feasible_saving=(describe(safe.deadline_feasible_coordination_usd_mwh)
                                                             if len(safe) else None),
                                    deadline_missed_fraction=describe(actual_joint.missed_fraction),
                                    unfinished_fraction=describe(actual_joint.unserved_fraction),
                                    additional_grid_mw=describe(actual_joint.extra_grid_capacity_mw),
                                    demand_wape=describe(actual_joint.demand_wape),
                                    solar_wape=describe(actual_joint.solar_wape),
                                    wind_wape=describe(actual_joint.wind_wape),
                                    matched_forecast_extra_cost=(describe(comparable.forecast_extra_joint_cost_usd_mwh)
                                                                if len(comparable) else None)))
    paths.update(gpu_audit=gpu_audit_path, forecast_clock_review=clock_review_path,
                 sequential_forecast_audit=sequential_audit_path)
    manifest = dict(input_hashes={name: sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
                    physical_events=int(physical.events.sum()), capacity_groups=len(capacity),
                    price_rows=len(price), gpu_rows=len(gpu), forecast_groups=len(forecast),
                    gpu_proven_rows=int(gpu.capacity_proven.sum()), forecast=group_summaries,
                    capacity_positive_lower_groups=int((capacity.net_saving_lower_usd_mwh > 0).sum()),
                    selling_above_joint_upper_groups=int((comparison.selling_over_joint_upper > 1).sum()),
                    residuals=dict(physical=float(physical.max_residual.max()), capacity=float(capacity.max_residual.max()),
                                   price=float(price.max_residual.max()), gpu=float(gpu.max_residual.max())))
    produced = ['physical-summary.csv', 'global-capacity-summary.csv', 'electricity-price-paired.parquet',
                'electricity-price-summary.csv', 'strategy-comparison.csv', 'gpu-service-capacity.csv',
                'sale-distribution.csv', 'forecast-cases.csv', 'forecast-summary.csv']
    manifest['output_hashes'] = {name: sha256((output / name).read_bytes()).hexdigest() for name in produced}
    (output / 'validation.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()
