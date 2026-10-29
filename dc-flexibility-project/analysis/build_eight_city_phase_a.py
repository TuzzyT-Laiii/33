import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from compute_load.cooling import CoolingModel
from compute_load.model import sha256
from eight_city_physics import read_profiles, replay, resource_capacity_factors
from eight_city_scenarios import scenario_fields, write_table


RUN = 'eight-city-phase-a-20261029'


def build(args):
    project, raw, benchmarks = args.project.resolve(), args.weather.resolve(), args.benchmarks.resolve()
    output, processed = project / 'results' / RUN, project / 'data/processed' / RUN
    output.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    write_table(output)
    cities = json.loads((project / 'analysis/eight_city_locations.json').read_text(encoding='utf-8'))
    profiles, inputs = read_profiles(project, args.baseline)
    curve = pd.read_csv(benchmarks / 'IEA_Reference_3.4MW_130.csv')
    summaries, climates, missing, source_records, checks, supply_tables = [], [], [], [], {}, {}
    for city in cities:
        name = city['city_id']
        cds_path, arco_path = raw / (name + '.parquet'), raw / (name + '-arco.parquet')
        path = raw / (name + '-frozen-base.nc') if args.thermal_only else (cds_path if cds_path.exists() else arco_path)
        if not path.exists():
            missing.append(name)
            continue
        if args.thermal_only:
            with path.open('rb') as stream, xr.open_dataset(stream) as dataset:
                data = dataset.to_dataframe().reset_index().rename(columns={'time': 'time_utc'})
            data = data[['time_utc','t2m','d2m','sp']]
        else:
            data = pd.read_parquet(path)
        data.time_utc = pd.to_datetime(data.time_utc, utc=True)
        times = pd.date_range('2023-01-01', '2026-01-01', freq='h', inclusive='left', tz='UTC')
        if not pd.DatetimeIndex(data.time_utc).equals(times):
            raise ValueError('城市天气必须完整覆盖26304小时：' + name)
        if not np.isfinite(data.drop(columns='time_utc').to_numpy()).all():
            raise ValueError('天气包含非有限数值：' + name)
        weather = pd.DataFrame({'time_utc': times, 'temperature_c': data.t2m.to_numpy(dtype=float) - 273.15,
            'dew_point_c': data.d2m.to_numpy(dtype=float) - 273.15, 'pressure_pa': data.sp.to_numpy(dtype=float)})
        if args.thermal_only:
            supply, resource_parameters = pd.DataFrame({'time_utc': times}), {'resources': 'pending'}
        else:
            supply, resource_parameters = resource_capacity_factors(pd.concat([weather, data.drop(columns='time_utc')], axis=1), city, curve)
            supply.to_parquet(processed / (name + '-supply.parquet'), index=False)
            supply_tables[name] = supply
        source_records.append({'city_id': name, 'path': str(path), 'sha256': sha256(path),
            'method': 'CDS' if path == cds_path else 'ECMWF_official_ARCO', 'resource_parameters': resource_parameters})
        city_frames = {}
        for year in (2023, 2024, 2025):
            selected = weather.loc[weather.time_utc.dt.year.eq(year)].reset_index(drop=True)
            year_supply = supply.loc[supply.time_utc.dt.year.eq(year)].reset_index(drop=True)
            count = len(selected)
            hot_end = int(selected.temperature_c.rolling(168).mean().idxmax())
            hot_start = hot_end - 167
            climate = {'city_id': name, 'city_zh': city['city_zh'], 'year': year, 'hours': count,
                'mean_temperature_c': float(selected.temperature_c.mean()),
                'minimum_temperature_c': float(selected.temperature_c.min()), 'maximum_temperature_c': float(selected.temperature_c.max()),
                'mean_dewpoint_c': float(selected.dew_point_c.mean()), 'mean_pressure_pa': float(selected.pressure_pa.mean()),
                'hottest_week_start_utc': str(selected.time_utc.iloc[hot_start]),
                'hottest_week_end_utc': str(selected.time_utc.iloc[hot_end])}
            if not args.thermal_only:
                climate.update({col: float(year_supply[col].mean()) for col in (
                    'pv_capacity_factor','pv_tracking_capacity_factor','wind_capacity_factor','wind_gross_capacity_factor')})
                climate.update(mean_wind100_m_s=float(year_supply.wind100_m_s.mean()),
                    ghi_kwh_m2=float(year_supply.ghi_w_m2.sum()/1000))
            climates.append(climate)
            for (config, demand), profile in profiles.items():
                cpu, gpu, it = replay(profile, count)
                for cooling_case in ('low', 'central', 'high'):
                    model = CoolingModel(cooling_type='direct_liquid' if config == 'R-GB200' else 'air', parameter_set=cooling_case)
                    frame = model.calculate(selected, it, profile['capacity_mw'])
                    energy, energy_it = float(frame.facility_mw.sum()), float(frame.it_mw.sum())
                    row = {**scenario_fields(config, demand), 'city_id': name, 'city_zh': city['city_zh'],
                        'year': year, 'hours': count, 'reference_it_mw': 10, 'installed_reference_it_mw': profile['capacity_mw'],
                        'cooling_case': cooling_case, 'cooling_type': model.cooling_type,
                        'annual_it_mwh': energy_it, 'annual_cooling_mwh': float(frame.cooling_mw.sum()),
                        'annual_other_mwh': float(frame.other_mw.sum()), 'annual_facility_mwh': energy,
                        'peak_facility_mw': float(frame.facility_mw.max()), 'peak_it_mw': float(frame.it_mw.max()),
                        'annual_pue': energy / energy_it, 'cooling_share': float(frame.cooling_mw.sum()) / energy,
                        'hottest_week_peak_mw': float(frame.facility_mw.iloc[hot_start:hot_end+1].max()),
                        'hottest_week_start_utc': climate['hottest_week_start_utc']}
                    summaries.append(row)
                    label = f'{name}-{year}-{config}-{demand}-{cooling_case}'
                    checks[label] = bool(np.isfinite(frame[['it_mw','cooling_mw','facility_mw','pue']]).all().all()
                        and np.allclose(frame.facility_mw, frame.it_mw+frame.cooling_mw+frame.other_mw, atol=1e-9)
                        and frame.heat_balance_residual_mw.abs().max() < 1e-8)
                    if cooling_case == 'central':
                        columns = ['time_utc','temperature_c','dew_point_c','relative_humidity_pct','pressure_pa',
                            'wet_bulb_c','it_mw','cooling_mw','other_mw','facility_mw','pue','installed_reference_it_mw']
                        solver = frame[columns].copy()
                        if not args.thermal_only:
                            solver['pv_capacity_factor'] = year_supply.pv_capacity_factor.to_numpy()
                            solver['wind_capacity_factor'] = year_supply.wind_capacity_factor.to_numpy()
                            solver['pv_tracking_capacity_factor'] = year_supply.pv_tracking_capacity_factor.to_numpy()
                        solver['u_cpu_proxy'], solver['allocated_gpu_fraction'] = cpu, gpu
                        solver['it_load_shape'] = it / profile['capacity_mw']
                        solver['demand_resource_fraction'] = cpu if demand == 'Lisa' else gpu
                        for field, value in scenario_fields(config, demand).items():
                            solver[field] = value
                        solver['city_id'] = name
                        solver['power_method'] = profile['method']
                        city_frames.setdefault((config, demand), []).append(solver)
        for (config, demand), years in city_frames.items():
            combined = pd.concat(years, ignore_index=True)
            combined = combined.set_index('time_utc', drop=False)
            combined.index.name = 'utc_index'
            if not combined.index.equals(times):
                raise ValueError('求解器三年连续索引错误')
            suffix = '-thermal-only' if args.thermal_only else ''
            scenario = scenario_fields(config, demand)['scenario_scale_id']
            combined.to_parquet(processed / f'{name}-{scenario}{suffix}.parquet', index=True, compression='zstd')
        print(json.dumps({'city_complete': name, 'utc': datetime.now(timezone.utc).isoformat()}), flush=True)
    if summaries:
        summary = pd.DataFrame(summaries)
        summary.to_csv(output / 'annual-load-summary.csv', index=False, encoding='utf-8-sig')
        central = summary.loc[summary.cooling_case.eq('central')]
        central.to_csv(output / 'annual-load-central.csv', index=False, encoding='utf-8-sig')
        city_summary = central.groupby(['city_id','city_zh']).agg(
            annual_facility_mwh_mean=('annual_facility_mwh','mean'), annual_facility_mwh_min=('annual_facility_mwh','min'),
            annual_facility_mwh_max=('annual_facility_mwh','max'), annual_it_mwh_sum=('annual_it_mwh','sum'),
            annual_facility_mwh_sum=('annual_facility_mwh','sum'), cooling_share_mean=('cooling_share','mean'),
            peak_facility_mw_max=('peak_facility_mw','max')).reset_index()
        city_summary['energy_weighted_pue'] = city_summary.annual_facility_mwh_sum / city_summary.annual_it_mwh_sum
        city_summary.to_csv(output / 'city-load-comparison.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(climates).to_csv(output / 'city-climate-resource-annual.csv', index=False, encoding='utf-8-sig')
    validation = {'created_utc': datetime.now(timezone.utc).isoformat(), 'checks': checks,
        'computed_cities': len(cities)-len(missing), 'missing_cities': missing,
        'annual_rows': len(summaries), 'all_computed_checks_passed': all(checks.values()),
        'phase_b_optimization_run': False,
        'status': 'thermal_completed_resources_pending' if args.thermal_only else ('complete' if not missing else 'partial_weather_pending')}
    (output / 'weather-load-validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = {'base_commit': '6ffda9b', 'weather_sources': source_records, 'load_sources': inputs,
        'processed_directory': str(processed), 'pending_weather_cities': missing,
        'created_utc': datetime.now(timezone.utc).isoformat()}
    (output / 'calculation-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    if not all(checks.values()):
        raise ValueError('城市功率或热平衡检查失败')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--weather', type=Path, required=True)
    parser.add_argument('--benchmarks', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--thermal-only', action='store_true')
    build(parser.parse_args())
