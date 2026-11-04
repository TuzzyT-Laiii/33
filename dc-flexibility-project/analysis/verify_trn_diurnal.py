import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_trn_diurnal import metrics, RUN
from build_typical_load_curves_v3 import checks
from compute_load.model import sha256
from load_curve_scenarios_v3 import catalogue, TDP
from trn_power_models import power


def verify(args):
    project, inputs = args.project.resolve(), args.input_project.resolve()
    output = project / 'results' / RUN
    audit = json.loads((output / 'native-calculation-audit.json').read_text('utf-8'))
    transfer = json.loads((output / 'transfer-audit.json').read_text('utf-8'))
    monitoring = json.loads((output / 'kalos-hourly-audit.json').read_text('utf-8'))
    for name, digest in transfer['input_sha256'].items():
        if sha256(output / name) != digest:
            raise ValueError('形状统计输入哈希变化：' + name)
    original = json.loads((project / 'results/trn-power-model-20261101/acme/telemetry-run.json').read_text('utf-8'))
    hourly = pd.read_csv(output / 'kalos-hourly.csv')
    for metric in ('POWER_USAGE', 'GPU_UTIL'):
        if monitoring['monitoring_sha256'][metric + '.csv'] != original['source_files'][f'hf/data/utilization/kalos/{metric}.csv']:
            raise ValueError('监测原件与既有获取审计不一致')
        previous = json.loads((project / f'results/trn-power-model-20261101/acme/kalos-{metric}-summary.json').read_text('utf-8'))
        prefix, value = ('power', 'mean_gpu_power_w') if metric == 'POWER_USAGE' else ('util', 'mean_gpu_util_pct')
        counts = hourly[prefix + '_valid_device_samples']
        if int(counts.sum()) != previous['valid_cells']:
            raise ValueError('监测小时聚合计数不守恒')
        np.testing.assert_allclose(np.average(hourly[value], weights=counts), previous['mean'], rtol=1e-12)
    zero = pd.read_csv(output / 'kalos-zero-util-power-histogram.csv')
    if int(zero.samples.sum()) != monitoring['zero_util_samples']:
        raise ValueError('零利用率样本直方图不守恒')
    np.testing.assert_allclose(np.average(zero.power_w_bin_lower + .5, weights=zero.samples), monitoring['zero_util_power_mean_w'], atol=.5, rtol=0)
    if transfer['inverse_eligible_hours_at_idle60'] != 0 or not transfer['scheme_b'].startswith('not_identifiable'):
        raise ValueError('当前原件不支持直接比例方案')
    if (output / 'transfer-profile.csv').exists() and sha256(output / 'transfer-profile.csv') != audit['transfer_profile_sha256']:
        raise ValueError('迁移形状文件发生变化')
    configs = catalogue(project)
    summary = pd.read_csv(output / 'window-summary-all-locations.csv').set_index(['scenario_id', 'location_id', 'weather_year', 'case'])
    checked, rows = 0, 0
    immutable = ['source_clock', 'source_hour_index', 'allocated_gpu_fraction', 'u_cpu_proxy', 'unit_count',
        'installed_reference_it_mw', 'temperature_c', 'time_utc', 'hour_of_source_day', 'hour_of_source_week']
    for relative, digest in audit['inputs_sha256'].items():
        path = inputs / relative
        if sha256(path) != digest:
            raise ValueError('原生输入哈希变化')
        source = pd.read_parquet(path)
        flat = pd.read_parquet(output / 'hourly' / str(source.location_id.iloc[0]) / 'constant_080' / path.name)
        for case in audit['variants']:
            candidate = output / 'hourly' / str(source.location_id.iloc[0]) / case / path.name
            if sha256(candidate) != audit['output_hourly_sha256'][str(candidate.relative_to(output))]:
                raise ValueError('小时输出哈希变化')
            current = pd.read_parquet(candidate)
            if len(current) != 2160 or not np.array_equal(current.source_hour_index, np.arange(2160)):
                raise ValueError('原生窗口被重复或截断')
            pd.testing.assert_frame_equal(source[immutable], current[immutable], check_exact=True)
            pd.testing.assert_frame_equal(current[['base_it_mw', 'cpu_it_mw']], flat[['base_it_mw', 'cpu_it_mw']], check_exact=True)
            checks(current)
            a, r = current.allocated_gpu_fraction, current.allocated_gpu_power_ratio
            key = current.server_type.iloc[0]
            if not r.between(0, 1).all():
                raise ValueError('目标GPU功率比越界')
            expected = configs[key]['gpu_devices_per_unit'] * current.unit_count.iloc[0] / 1e6 * (a * r * TDP[key] + (1 - a) * (70 if key == 'D' else 50))
            np.testing.assert_allclose(expected, current.gpu_it_mw, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(np.average(r, weights=a), .8, atol=1e-12)
            np.testing.assert_allclose(current.it_mw.mean(), flat.it_mw.mean(), rtol=1e-12)
            record = metrics(current, case)
            actual = summary.loc[(record['scenario_id'], record['location_id'], record['weather_year'], case)]
            for name in ('mean_it_mw', 'peak_it_mw', 'minimum_it_mw', 'mean_facility_mw', 'peak_facility_mw',
                    'minimum_facility_mw', 'it_diurnal_peak_valley', 'it_weekly_peak_valley',
                    'facility_diurnal_peak_valley', 'facility_weekly_peak_valley'):
                np.testing.assert_allclose(actual[name], record[name], rtol=1e-12)
            checked += 1
            rows += len(current)
    # 用本次修改前的真实小时输出检查新增可选参数没有改变既有静态调用。
    ratios = json.loads((project / 'results/typical-load-curves-20261029-v2/mit-power-ratio-summary.json').read_text('utf-8'))['transfer_distribution']
    mit = json.loads((project / 'results/typical-load-curves-20261101-v3/mit-calibration.json').read_text('utf-8'))
    cpu = json.loads((project / 'results/trn-power-model-20261101/cpu-shape.json').read_text('utf-8'))
    legacy_count = 0
    for path in (project / 'results/trn-power-model-20261101/hourly').glob('*.parquet'):
        frame = pd.read_parquet(path)
        key, use = frame.server_type.iloc[0], frame.use.iloc[0]
        predicted = power(configs[key], key, use, frame.u_cpu_proxy.to_numpy(), frame.allocated_gpu_fraction.to_numpy(), ratios, mit, cpu)
        for component in ('base', 'cpu', 'gpu'):
            np.testing.assert_allclose(predicted[component + '_w'] * frame.unit_count.iloc[0] / 1e6, frame[component + '_it_mw'], rtol=1e-12, atol=1e-12)
        legacy_count += 1
    if checked != len(audit['output_hourly_sha256']) or rows != audit['hourly_rows'] or legacy_count != 44:
        raise ValueError('验证覆盖范围不完整')
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'hourly_files': checked, 'hourly_rows': rows,
        'legacy_static_scenarios': legacy_count, 'legacy_static_predictions_unchanged': True,
        'source_hashes_verified': True, 'native_2160h_once': True, 'frozen_demand_weather_units_cpu': True,
        'independent_gpu_formula': True, 'it_facility_heat_conservation': True,
        'constant_080_gpu_hour_mean_preserved': True, 'summary_extremes_daily_weekly_recomputed': True,
        'raw_monitoring_cell_count_and_mean_conservation': True, 'direct_scheme_not_imputed': True}
    (output / 'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--input-project', type=Path, required=True)
    verify(parser.parse_args())
