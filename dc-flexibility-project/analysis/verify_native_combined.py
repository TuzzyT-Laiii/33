import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compute_load.model import sha256
from native_combined_power import NATIVE, RUN


def verify(project, inputs, reference_only=False):
    output = project / 'results' / RUN
    previous = json.loads((inputs / 'results' / NATIVE / 'manifest.json').read_text('utf-8'))
    if reference_only:
        paths = [name.replace(NATIVE, RUN) for name in previous['hourly_sha256'] if '/hourly/' in name]
        manifest = {'hourly_sha256': {name: sha256(project / name) for name in paths}}
        cases = None
    else:
        manifest = json.loads((output / 'manifest.json').read_text('utf-8'))
        cases = pd.read_csv(output / 'window-summary-all-locations.csv').set_index(['scenario_id', 'location_id', 'weather_year', 'placement'])
    cpu_cache, gpu_cache = {}, {}
    checked, rows, maximum_balance = 0, 0, 0.
    unchanged = ['source_clock', 'source_hour_index', 'allocated_gpu_fraction', 'u_cpu_proxy', 'unit_count', 'time_utc',
        'temperature_c', 'hour_of_source_day', 'hour_of_source_week', 'source_unresolved', 'source_quality_flag']
    for relative, digest in manifest['hourly_sha256'].items():
        current_path = project / relative
        if sha256(current_path) != digest:
            raise ValueError('组合结果指纹不符：' + relative)
        old_relative = relative.replace(RUN, NATIVE)
        old_path = inputs / old_relative
        if sha256(old_path) != previous['hourly_sha256'][old_relative]:
            raise ValueError('原结果字节变化：' + old_relative)
        frame, old = pd.read_parquet(current_path), pd.read_parquet(old_path)
        pd.testing.assert_frame_equal(frame[unchanged], old[unchanged], check_exact=True)
        scenario, location, year, placement = [frame[name].iloc[0] for name in ('scenario_id', 'location_id', 'weather_year', 'placement')]
        use, key, count = frame.use.iloc[0], frame.server_type.iloc[0], int(frame.unit_count.iloc[0])
        position = frame.source_hour_index.to_numpy(int)
        sid = str(frame.scenario.iloc[0])
        if use != 'TRN' and sid not in cpu_cache:
            upstream = pd.read_parquet(inputs / 'results/trn-power-model-20261101/hourly' / f'{sid}-10MW-2025.parquet')
            cpu_cache[sid] = upstream.drop_duplicates('source_hour_index').set_index('source_hour_index')
        if use != 'TRN':
            upstream = cpu_cache[sid].loc[position]
            for name in ('u_cpu_proxy', 'allocated_gpu_fraction'):
                np.testing.assert_allclose(frame[name], upstream[name], atol=1e-12, rtol=1e-12)
            for name in ('base_it_mw', 'cpu_it_mw'):
                np.testing.assert_allclose(frame[name], upstream[name].to_numpy() * count / upstream.unit_count.to_numpy(), atol=1e-11, rtol=1e-12)
        if use == 'TRN':
            upstream_path = inputs / 'results/trn-power-model-20261101/diurnal-acme/hourly' / location / 'shape_central' / current_path.name
            training = pd.read_parquet(upstream_path)
            for name in ('base_it_mw', 'cpu_it_mw', 'it_mw', 'gpu_it_mw', 'facility_mw', 'cooling_mw', 'allocated_gpu_power_ratio'):
                np.testing.assert_allclose(frame[name], training[name], atol=1e-10, rtol=1e-12)
            np.testing.assert_allclose(np.average(frame.allocated_gpu_power_ratio, weights=frame.allocated_gpu_fraction), .8, rtol=1e-12)
        elif use == 'INF':
            cache_key = (sid, location, int(year))
            if cache_key not in gpu_cache:
                folder = inputs / ('results/inf-diurnal-20261101/hourly' if location == 'reference-grid' else f'data/processed/inf-diurnal-20261101/{location}')
                upstream = pd.read_parquet(folder / f'{sid}-10MW-{int(year)}.parquet')
                gpu_cache[cache_key] = upstream.drop_duplicates('source_hour_index').set_index('source_hour_index')
            upstream = gpu_cache[cache_key].loc[position]
            np.testing.assert_allclose(frame.gpu_it_mw, upstream.gpu_it_mw.to_numpy() * count / upstream.unit_count.to_numpy(), atol=1e-10, rtol=1e-12)
            np.testing.assert_allclose(frame.request_intensity, upstream.request_intensity.to_numpy(), atol=1e-12)
        else:
            np.testing.assert_array_equal(frame.gpu_it_mw, old.gpu_it_mw)
        np.testing.assert_allclose(frame.it_mw, frame.base_it_mw + frame.cpu_it_mw + frame.gpu_it_mw, atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(frame.facility_mw, frame.it_mw + frame.cooling_mw + frame.other_mw, atol=1e-10, rtol=1e-12)
        maximum_balance = max(maximum_balance, float(frame.heat_balance_residual_mw.abs().max()))
        if reference_only:
            checked += 1
            rows += len(frame)
            continue
        row = cases.loc[(scenario, location, year, placement)]
        if int(row.hours) != len(frame) or not np.array_equal(position, np.arange(len(frame))):
            raise ValueError('窗口重复、截断或小时数不符')
        for component in ('it', 'facility'):
            values = frame[component + '_mw'].to_numpy()
            for prefix, expected in [('mean', values.mean()), ('peak', values.max()), ('minimum', values.min()), ('window', values.sum())]:
                unit = 'mwh' if prefix == 'window' else 'mw'
                np.testing.assert_allclose(row[f'{prefix}_{component}_{unit}'], expected, rtol=1e-12)
            for label, grouping in [('diurnal', 'hour_of_source_day'), ('weekly', 'hour_of_source_week')]:
                if use == 'CLD' and label == 'weekly':
                    if not pd.isna(row[f'{component}_weekly_peak_valley']):
                        raise ValueError('Alibaba仅五天，不允许周形状')
                    continue
                groups = frame[grouping].to_numpy(int)
                means = np.bincount(groups, weights=values) / np.bincount(groups)
                np.testing.assert_allclose(row[f'{component}_{label}_peak_valley'], means.max() / means.min(), rtol=1e-12)
        checked += 1
        rows += len(frame)
        if checked % 200 == 0:
            print(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'verified_files': checked, 'total': len(manifest['hourly_sha256'])}), flush=True)
    if reference_only:
        result = dict(utc=datetime.now(timezone.utc).isoformat(), passed=True, files=checked, hourly_rows=rows, scope='reference-grid upstream hourly cross-check')
        (output / 'verification-reference.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(result, flush=True)
        return
    summary = pd.read_csv(output / 'native-summary.csv')
    for record in summary.to_dict('records'):
        expanded = cases.reset_index()
        matches = expanded.loc[expanded.scenario_id.eq(record['scenario_id']) & expanded.location_id.eq('reference-grid')]
        for metric in ('mean_facility_mw', 'peak_facility_mw', 'minimum_facility_mw', 'window_facility_mwh', 'facility_diurnal_peak_valley'):
            for suffix, expected in [('median', matches[metric].median()), ('min', matches[metric].min()), ('max', matches[metric].max())]:
                np.testing.assert_allclose(record[f'placement_{metric}_{suffix}'], expected, rtol=1e-12)
    if checked != 5500 or rows != 15861600 or len(summary) != 44:
        raise ValueError('覆盖不完整')
    result = dict(utc=datetime.now(timezone.utc).isoformat(), passed=True, files=checked, hourly_rows=rows, scenarios=44,
        upstream_checks='非训练CPU/基础对照#36全年小时；INF GPU对照#35逐小时；TRN各分项及整条IT/设施对照#36原生90天shape_central；其他GPU对照#34',
        unchanged_native_inputs=True, window_statistics=True, placement_statistics=True, cooling_heat_balance_maximum_residual_mw=maximum_balance)
    (output / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--reference-only', action='store_true')
    args = parser.parse_args()
    verify(args.project, args.inputs, args.reference_only)
