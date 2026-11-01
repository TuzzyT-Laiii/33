import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compute_load.model import sha256
from inf_diurnal_power import CASES
from load_curve_scenarios_v3 import identity


def verify(project, storage):
    output = project / 'results/inf-diurnal-20261101'
    manifest = json.loads((output / 'manifest.json').read_text('utf-8'))
    for relative, expected in manifest['aggregate_inputs_sha256'].items():
        if sha256(project / relative) != expected:
            raise ValueError('聚合输入SHA不符：' + relative)
    v3 = project / 'results/typical-load-curves-20261101-v3'
    mapping = pd.read_csv(v3 / 'INF-annual-replay-map.csv').iloc[:4440]
    params = pd.read_csv(v3 / 'power-parameters.csv').query("use == 'INF'").set_index('server_type')
    summaries = pd.read_csv(output / 'annual-summary-all-locations.csv')
    if manifest['hourly_files'] not in (12, 300) or len(summaries) != 2 * manifest['hourly_files']:
        raise ValueError('年度汇总覆盖不完整')
    calibrations = pd.read_csv(output / 'power-calibration.csv')
    verified = []
    for relative, expected in manifest['hourly_sha256'].items():
        path = (project if '/hourly/' in relative else storage) / relative
        if sha256(path) != expected:
            raise ValueError('小时输出SHA不符：' + relative)
        new = pd.read_parquet(path)
        location, year, sid = new.location_id.iloc[0], int(new.year.iloc[0]), new.scenario_id.iloc[0]
        prefix = 'results/typical-load-curves-20261101-v3/hourly' if location == 'reference-grid' else f'data/processed/typical-load-curves-20261101-v3/{location}'
        before_path = storage / prefix / f'{sid}-{year}.parquet'
        if sha256(before_path) != manifest['inputs_sha256'][str(before_path.relative_to(storage))]:
            raise ValueError('v3输入SHA不符')
        old = pd.read_parquet(before_path)
        time = pd.DatetimeIndex(new.time_utc)
        expected_index = pd.date_range(f'{year}-01-01', f'{year + 1}-01-01', freq='h', inclusive='left', tz='UTC')
        if not time.equals(expected_index):
            raise ValueError('全年小时索引不连续')
        key, scale = new.server_type.iloc[0], int(new.reference_it_mw.iloc[0])
        if not new.scenario_zh.eq(identity(key, 'INF', scale)['scenario_zh']).all():
            raise ValueError('情境中文全称不符')
        for column in ('base_it_mw', 'cpu_it_mw', 'u_cpu_proxy', 'allocated_gpu_fraction', 'unit_count',
                       'installed_reference_it_mw', 'source_hour_index', 'source_unresolved',
                       'all_business_inventory_excess', 'temperature_c', 'relative_humidity_pct',
                       'input_dew_point_c', 'pressure_pa', 'pressure_assumed'):
            np.testing.assert_array_equal(new[column], old[column])
        online = mapping.online_fraction.to_numpy()[new.source_hour_index]
        offline = mapping.offline_fraction.to_numpy()[new.source_hour_index]
        np.testing.assert_allclose(new.online_allocated_gpu_fraction, online)
        np.testing.assert_allclose(new.offline_allocated_gpu_fraction, offline)
        p = params.loc[key]
        cards = float(p.gpu_devices_per_unit * old.unit_count.iloc[0])
        mean_w = float(p.target_gpu_tdp_W * p.allocated_gpu_ratio)
        cal = calibrations.loc[calibrations.scenario_id.eq(sid) & calibrations.location_id.eq(location) & calibrations.year.eq(year)].iloc[0]
        idle_ratio, high_ratio, exponent = CASES['central']
        idle, high = idle_ratio * p.target_gpu_tdp_W, high_ratio * p.target_gpu_tdp_W
        per_card = idle + (high - idle) * np.minimum(cal.gain * new.request_intensity.to_numpy() ** exponent, 1)
        np.testing.assert_allclose(new.online_gpu_w_each, per_card, atol=1e-12)
        if per_card.min() < idle or per_card.max() > high:
            raise ValueError('每卡功率越过端点')
        np.testing.assert_allclose(np.dot(online, per_card), online.sum() * mean_w, rtol=1e-12)
        np.testing.assert_allclose(new.offline_gpu_it_mw, cards * offline * mean_w / 1e6)
        np.testing.assert_allclose(new.unallocated_gpu_it_mw, cards * (1 - online - offline) * p.idle_gpu_W / 1e6)
        np.testing.assert_allclose(new.gpu_it_mw, cards / 1e6 * (online * per_card + offline * mean_w + (1 - online - offline) * p.idle_gpu_W))
        np.testing.assert_allclose(new.it_mw, new.base_it_mw + new.cpu_it_mw + new.gpu_it_mw)
        np.testing.assert_allclose(new.facility_mw, new.it_mw + new.cooling_mw + new.other_mw)
        np.testing.assert_allclose(new.it_mw.sum(), old.it_mw.sum(), rtol=1e-12)
        if new.heat_balance_residual_mw.abs().max() > 1e-8:
            raise ValueError('热平衡残差超过容差')
        for version, data in [('v3', old), ('diurnal', new)]:
            record = summaries.loc[summaries.scenario_id.eq(sid) & summaries.location_id.eq(location)
                                   & summaries.year.eq(year) & summaries.version.eq(version)].iloc[0]
            for component in ('it', 'facility'):
                values = data[component + '_mw']
                for metric, expected_value in [('annual', values.sum()), ('mean', values.mean()), ('peak', values.max())]:
                    column = f'{metric}_{component}_' + ('mwh' if metric == 'annual' else 'mw')
                    np.testing.assert_allclose(record[column], expected_value, rtol=1e-12)
                for period, groups in [('diurnal', time.hour), ('weekly', time.dayofweek * 24 + time.hour)]:
                    curve = values.groupby(groups).mean()
                    np.testing.assert_allclose(record[f'{component}_{period}_peak_valley'], curve.max() / curve.min(), rtol=1e-12)
        verified.append(dict(scenario_id=sid, location=location, year=year, hours=len(new), sha256=expected))
    shapes = pd.read_csv(output / 'request-shape-24h.csv')
    for _, group in shapes.groupby(['source', 'zero_policy', 'metric']):
        np.testing.assert_allclose(group.normalized.mean(), 1., rtol=1e-12)
    request = pd.read_csv(output / 'request-analysis-hourly.csv')
    burst = request.loc[request.source.eq('burst')]
    if len(burst) != 2856 or int(burst.requests.sum()) != 5252477 or int(burst.zero_observation.sum()) != 149:
        raise ValueError('Burst119日窗口核对失败')
    if int(burst.input_tokens.sum()) != 1853340790 or int(burst.output_tokens.sum()) != 309164730:
        raise ValueError('Burst token守恒失败')
    sensitivity = pd.read_csv(output / 'sensitivity-summary.csv')
    if len(sensitivity) != 66 or sensitivity.groupby('server_type').size().ne(22).any():
        raise ValueError('敏感性覆盖不完整')
    for key, rows in sensitivity.groupby('server_type'):
        central = rows.loc[rows.sensitivity.eq('central')].iloc[0]
        matched = rows.loc[~rows.sensitivity.str.startswith('mean-online-')]
        np.testing.assert_allclose(matched.annual_it_mwh, central.annual_it_mwh, rtol=1e-12)
        if matched.it_diurnal_peak_valley.le(1).any():
            raise ValueError('请求形状没有产生预期日内变化')
    report = dict(checked_utc=datetime.now(timezone.utc).isoformat(), status='passed', files=len(verified),
        hourly_rows=sum(x['hours'] for x in verified), location_years=len(summaries[['location_id', 'year']].drop_duplicates()),
        real_input_only=True, checks=['input/output SHA-256', 'full-year and leap-year indices', 'unchanged allocation/CPU/base/weather/units',
        'independent bounded GPU formula', 'online and total IT annual energy conservation', 'unchanged offline energy',
        'power/heat balance', 'annual integration and diurnal/weekly summary recomputation', 'Burst counts and normalization'],
        verified=verified)
    (output / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'verified'}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--storage', type=Path, required=True)
    args = parser.parse_args()
    verify(args.project.resolve(), args.storage.resolve())
