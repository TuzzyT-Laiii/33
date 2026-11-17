import json

import fitz
import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image

from paper_figure_common import (OUT, REPRESENTATIVES, ROOT, RUN, SCENARIOS, SOURCE,
                                 digest, load_economics)


def main():
    head, valid, factor = load_economics()
    baseline = pd.read_csv(RUN / 'part1/part1-summary.csv').set_index('scenario').loc[SCENARIOS]
    for column in ['hours', 'K_mw', 'mean_over_K_pct', 'hours_above_90pct_K']:
        np.testing.assert_allclose(head.loc[SCENARIOS, column], baseline[column], atol=.000051, rtol=0)
    c = pd.read_csv(SOURCE / 'fig1c.csv').set_index(['scenario', 'boundary'])
    for boundary in ['planning', 'fixed']:
        for source_column, old_column, scale in [
                ('saving_annual_usd', f'saving_{boundary}_kusd', 1000),
                ('energy_savings_pct', f'energy_pct_{boundary}', 1),
                ('fleet_savings_pct', f'fleet_pct_{boundary}', 1)]:
            values = np.array([c.loc[(s, boundary), source_column] for s in SCENARIOS]) / scale
            np.testing.assert_allclose(values, baseline[old_column], atol=.000051, rtol=0)
    raw_distribution = pd.read_csv(SOURCE / 'fig1d.csv')
    np.testing.assert_allclose(raw_distribution.fleet_savings_pct,
                               100 * raw_distribution.saving_annual_usd / raw_distribution.fleet_annual_usd, rtol=1e-9)
    assert len(raw_distribution) == 4842 * 11 * 2
    med = valid.groupby(['scenario', 'boundary'])[['savings_pct', 'fleet_savings_pct', 'energy_annual_usd',
                                                  'saving_annual_usd', 'fleet_annual_usd']].median()
    train = valid.loc[valid.scenario.eq('D-TRN') & valid.boundary.eq('planning')]
    table = pd.read_parquet(RUN / 'attribution/attribution-table.parquet')
    paired = table.merge(valid[['grid_id', 'scenario', 'boundary', 'savings_pct']],
                          on=['grid_id', 'scenario', 'boundary'], validate='one_to_one', suffixes=('_attribution', '_summary'))
    assert len(paired) == len(valid)
    np.testing.assert_allclose(paired.savings_pct_attribution, paired.savings_pct_summary, atol=1e-10)
    regressions = pd.read_csv(SOURCE / 'fig2c.csv')
    fits = pd.read_csv(RUN / 'attribution/fit.csv').set_index(['scenario', 'boundary'])
    for (scenario, boundary), coefficients in regressions.groupby(['scenario', 'boundary']):
        data = table.loc[table.scenario.eq(scenario) & table.boundary.eq(boundary)]
        columns = coefficients.variable.tolist()
        x = (data[columns] - data[columns].mean()) / data[columns].std()
        design = np.column_stack([np.ones(len(x)), x])
        y = data.savings_pct.to_numpy()
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        np.testing.assert_allclose(beta[1:], coefficients.beta_pp_per_sd, atol=1e-8)
        r2 = 1 - (y - design @ beta).var() / y.var()
        np.testing.assert_allclose(r2, fits.loc[(scenario, boundary), 'r2_exogenous'], atol=1e-10)
        for i, row in enumerate(coefficients.itertuples()):
            reduced = np.delete(design, i + 1, axis=1)
            reduced_beta, *_ = np.linalg.lstsq(reduced, y, rcond=None)
            loss = r2 - (1 - (y - reduced @ reduced_beta).var() / y.var())
            np.testing.assert_allclose(loss, row.delta_r2, atol=1e-10)
            assert row.important_delta_r2_ge_0_02 == (loss >= .02)
    bins = pd.read_csv(SOURCE / 'fig2a.csv')
    original_bins = pd.read_csv(RUN / 'attribution/wind-bins.csv')
    for scenario in REPRESENTATIVES:
        before = original_bins.loc[original_bins.scenario.eq(scenario) & original_bins.boundary.eq('planning')]
        after = bins.loc[bins.scenario.eq(scenario)].sort_values('wind_bin_index')
        np.testing.assert_allclose(after['median'], before.savings_median, atol=1e-8)
        np.testing.assert_allclose(after.q25, before.savings_p25, atol=1e-8)
        np.testing.assert_allclose(after.q75, before.savings_p75, atol=1e-8)
        np.testing.assert_array_equal(after.n, before.n)
        assert after.n.sum() == 4842
    scatter = pd.read_csv(SOURCE / 'fig2b.csv')
    r = scatter.grid_share.corr(scatter.savings_pct)
    assert round(r, 2) == -.88
    assert tuple(np.rint([head.headroom_pct.min(), head.headroom_pct.max()])) == (27, 52)
    assert (head.hours_above_90pct_K.min(), head.hours_above_90pct_K.max()) == (6, 383)
    assert round(valid.fleet_savings_pct.max(), 2) == 1.87

    cells = gpd.read_parquet(OUT / 'map-assets/map-block-land-cells.parquet')
    audit = pd.read_csv(SOURCE / 'fig1a_geometry_audit.csv')
    assert len(audit) == 4842 and audit.map_drawn.sum() == 4832
    assert set(audit.loc[~audit.map_drawn, 'grid_id']) <= set(cells.loc[cells.geometry.is_empty, 'grid_id'])
    map_hashes = {'map-block-land-cells.parquet': '51f76d598f7245bda1f9fdf5e18048006f356d73db7abafb51ae9f7660ae7775',
                  'countries-10m.zip': 'ce1ac7036499a0edd641fbc093cd209a98f96a49d2eca8480aaacad35138a7f6'}
    for name, sha in map_hashes.items():
        assert digest(OUT / 'map-assets' / name) == sha

    pdfs = {}
    for name, height in [('fig1', 160), ('fig2', 154)]:
        with fitz.open(OUT / f'{name}.pdf') as doc:
            assert len(doc) == 1
            page = doc[0]
            np.testing.assert_allclose([page.rect.width, page.rect.height], np.array([180, height]) / 25.4 * 72, atol=.01)
            spans = [s for block in page.get_text('dict')['blocks'] if 'lines' in block
                     for line in block['lines'] for s in line['spans']]
            outside = [s['text'] for s in spans if not (s['bbox'][0] >= 0 and s['bbox'][1] >= 0
                       and s['bbox'][2] <= page.rect.width and s['bbox'][3] <= page.rect.height)]
            assert not outside, f'{name}: text outside page: {outside}'
            assert min(s['size'] for s in spans) >= 5.99
            assert not any('\u4e00' <= char <= '\u9fff' for s in spans for char in s['text'])
            fonts = page.get_fonts(full=True)
            assert all(font[2] != 'Type3' and doc.extract_font(font[0])[3] for font in fonts)
            images = page.get_images(full=True)
            if name == 'fig1':
                assert len(images) >= 1
                assert all(image[2] > 1000 and image[3] > 1000 for image in images)
            else:
                assert len(images) == 0
            pdfs[name] = {'width_mm': 180, 'height_mm': height, 'pages': len(doc),
                          'minimum_font_pt': min(s['size'] for s in spans), 'embedded_fonts': sorted({f[3] for f in fonts}),
                          'raster_images': len(images), 'all_text_inside_page': True}
        with Image.open(OUT / f'{name}.png') as png:
            assert abs(png.info['dpi'][0] - 600) < .01
            assert png.size == (int(180 / 25.4 * 600), int(height / 25.4 * 600))
            pdfs[name]['png_pixels'] = list(png.size)
            pdfs[name]['png_dpi'] = png.info['dpi'][0]
    not_cloud = fits.loc[fits.index.get_level_values('scenario').isin([s for s in REPRESENTATIVES if s != 'A-CLD'])]
    plan = med.xs('planning', level='boundary')
    fixed = med.xs('fixed', level='boundary')
    metrics = {
        'valid_grids': 4842, 'scenario_boundary_pairs': 22, 'economic_optimum_grid_rows': len(valid),
        'headroom_pct_range': [head.headroom_pct.min(), head.headroom_pct.max()],
        'hours_above_90pct_K_range': [int(head.hours_above_90pct_K.min()), int(head.hours_above_90pct_K.max())],
        'planning_median_energy_savings_pct_range': [plan.savings_pct.min(), plan.savings_pct.max()],
        'as_built_median_energy_savings_pct_range': [fixed.savings_pct.min(), fixed.savings_pct.max()],
        'D_TRN_planning_savings_pct_q25_median_q75': train.savings_pct.quantile([.25, .5, .75]).tolist(),
        'fleet_annual_cost_million_usd_range': [med.fleet_annual_usd.min() / 1e6, med.fleet_annual_usd.max() / 1e6],
        'baseline_energy_annual_cost_million_usd_range': [med.energy_annual_usd.min() / 1e6, med.energy_annual_usd.max() / 1e6],
        'annual_savings_million_usd_range': [med.saving_annual_usd.min() / 1e6, med.saving_annual_usd.max() / 1e6],
        'planning_median_fleet_savings_pct_range': [plan.fleet_savings_pct.min(), plan.fleet_savings_pct.max()],
        'as_built_median_fleet_savings_pct_range': [fixed.fleet_savings_pct.min(), fixed.fleet_savings_pct.max()],
        'maximum_grid_fleet_savings_pct': valid.fleet_savings_pct.max(),
        'D_TRN_low_wind_savings_median_pct': scatter.loc[scatter.cf_wind < .05, 'savings_pct'].median(),
        'D_TRN_high_wind_savings_median_pct': scatter.loc[scatter.cf_wind > .45, 'savings_pct'].median(),
        'D_TRN_grid_share_pearson_r': r,
        'representative_non_cloud_r2_range': [not_cloud.r2_exogenous.min(), not_cloud.r2_exogenous.max()],
        'cloud_r2_range': [fits.loc['A-CLD', 'r2_exogenous'].min(), fits.loc['A-CLD', 'r2_exogenous'].max()],
        'annualization_factor': factor, 'mapped_blocks': 4832, 'valid_grids_without_land_geometry': 10,
    }
    inputs = [RUN / 'grid-summary.parquet', RUN / 'part1/part1-summary.csv',
              ROOT / 'analysis/flex-econ-copt-20261102.json']
    inputs += [RUN / f'attribution/{name}' for name in ['attribution-table.parquet', 'regression.csv', 'fit.csv', 'wind-bins.csv']]
    inputs += [ROOT / f'results/flex-class-20261101/native-hourly/{s}.parquet' for s in SCENARIOS]
    inputs += list((OUT / 'map-assets').glob('*'))
    hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in inputs if p.is_file()}
    report = {'checks_passed': True, 'metrics': metrics, 'formats': pdfs, 'input_sha256': hashes,
              'scope': 'Existing results only; no optimization or source-data changes',
              'checks': ['Native peak-load reconstruction and PR60 summary agreement',
                         'Economic-optimum statuses, costs and zero added task capacity',
                         'Attribution target agreement and all 10 displayed OLS fits / predictor omission losses',
                         'All wind-bin medians, quartiles and counts against PR51',
                         'Map asset hashes / coverage, PDF text bounds / embedded fonts / images, PNG resolution']}
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', 'utf-8')
    print(json.dumps(metrics, indent=2))
    print(json.dumps(pdfs, indent=2))


if __name__ == '__main__':
    main()
