import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from eight_city_onsite_solver import annual_costs, sha256, value
from run_eight_city_flex_value import DEFAULT_CONFIG, PROJECT, write_json


def main():
    parser = argparse.ArgumentParser(description='只读ATB工作簿与Ember原始CSV，验证配置来源与单位')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--atb', type=Path, required=True)
    parser.add_argument('--ember', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=PROJECT / 'data/metadata/eight-city-flex-value-20261029/cost-source-validation.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    workbook = openpyxl.load_workbook(args.atb, data_only=True, read_only=True)
    sheet = workbook['Utility-Scale Battery - R&D']
    assert sheet['G18'].value == sheet['G24'].value == 2024
    np.testing.assert_allclose(sheet['G20'].value, value(config, 'battery_energy_capex'), rtol=0, atol=1e-9)
    np.testing.assert_allclose(sheet['G26'].value, value(config, 'battery_power_capex'), rtol=0, atol=1e-9)
    np.testing.assert_allclose(value(config, 'battery_price_year_factor'), 313.689 / 304.702, rtol=0, atol=1e-12)
    assert sha256(args.atb) == config['atb_source_sha256']
    assert sha256(args.ember) == config['emission_factor_metadata']['sha256']
    countries = {record['country'] for record in config['cities'].values()}
    columns = ['Area', 'Year', 'Electricity source', 'Emissions intensity (gCO2e/kWh)']
    selected_chunks = []
    for chunk in pd.read_csv(args.ember, usecols=columns, chunksize=10000):
        selected_chunks.append(chunk.loc[chunk.Area.isin(countries) & chunk['Electricity source'].eq('Total generation')])
    frame = pd.concat(selected_chunks, ignore_index=True)
    records = []
    for city, record in config['cities'].items():
        selected = frame.loc[frame.Area.eq(record['country']) & frame['Electricity source'].eq('Total generation')]
        selected = selected.dropna(subset=['Emissions intensity (gCO2e/kWh)']).sort_values('Year')
        newest = selected.iloc[-1]
        assert newest.Year == config['emission_factor_metadata']['data_year']
        np.testing.assert_allclose(newest['Emissions intensity (gCO2e/kWh)'] / 1000, record['emission_factor'], atol=1e-12)
        records.append({'city': city, 'country': record['country'], 'year': int(newest.Year),
                        'emission_factor_tco2e_mwh': record['emission_factor']})
    costs = annual_costs(config)
    report = {'status': 'passed', 'utc': datetime.now(timezone.utc).isoformat(),
        'config_sha256': sha256(args.config), 'atb_sha256': sha256(args.atb), 'ember_sha256': sha256(args.ember),
        'atb_sheet': 'Utility-Scale Battery - R&D', 'atb_energy_cell': 'G20', 'atb_power_cell': 'G26',
        'battery_energy_2023usd_kwh': sheet['G20'].value, 'battery_power_2023usd_kw': sheet['G26'].value,
        'battery_energy_2024usd_kwh': sheet['G20'].value * value(config, 'battery_price_year_factor'),
        'battery_power_2024usd_kw': sheet['G26'].value * value(config, 'battery_price_year_factor'),
        'annualized_costs': dict(zip(['compute_reference_usd_year','pv_usd_mw_year','wind_usd_mw_year',
                                    'battery_power_usd_mw_year','battery_energy_usd_mwh_year'], costs.tolist())),
        'emissions': records, 'emission_boundary': config['emission_factor_metadata']['status']}
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
