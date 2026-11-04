import argparse
import gzip
from pathlib import Path
import json
import os
import shutil

os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

import duckdb
import numpy as np
import pandas as pd

from global_flex_inputs import digest, settings, utc_now, write_json


def provenance(run, source, folder):
    target = folder / 'provenance'
    target.mkdir(exist_ok=True)
    names = ['inputs/selected-grids.parquet', 'inputs/grid-points.parquet', 'inputs/pilot-grids.json',
             'inputs/city-grid-points.csv', 'inputs/template-index.csv', 'inputs/selection-manifest.json',
             'inputs/un-m49-country-regions.csv', 'inputs/country-boundary-cache.json',
             'inputs/extraction-all.json', 'inputs/wind10-all.json', 'inputs/cf-build-all.json',
             'sources/un-m49-receipt.json', 'sources/ember-complete-headers.txt']
    scope = json.loads((source / 'run-manifest.json').read_text('utf-8'))['scope']
    names.append(f'inputs/grid-input-index-{scope}.csv')
    names.extend(str(path.relative_to(run)) for path in (run / 'validation').glob('*.json'))
    names.extend(str(path.relative_to(run)) for path in (run / 'validation').glob('*differences.csv'))
    names.extend(str(path.relative_to(run)) for path in (run / 'validation').glob('*comparison*.csv'))
    names.extend(str(path.relative_to(run)) for path in (run / 'validation').glob('*-audit.csv'))
    for name in names:
        shutil.copy2(run / name, target / Path(name).name)
    for name in ['run-manifest.json', 'progress.json', 'execution-attempts.jsonl', 'hourly-grid-selection.json']:
        if scope == 'pilot' and not (source / name).exists():
            continue
        shutil.copy2(source / name, target / name)
    receipts = []
    for path in sorted((run / 'inputs/raw-blocks').glob('*.json')):
        receipts.append({'kind': 'weather_7_variables', **json.loads(path.read_text('utf-8'))})
    for path in sorted((run / 'inputs/wind10-source').rglob('*.receipt.json')):
        row = json.loads(path.read_text('utf-8'))
        object_name = str(path.relative_to(run / 'inputs/wind10-source')).removesuffix('.receipt.json')
        if object_name == row['key']:
            receipts.append({'kind': 'wind10', **row})
    receipt_path = target / 'weather-source-receipts.jsonl.gz'
    with receipt_path.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as compressed:
            for row in receipts:
                compressed.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8'))
    write_json(target / 'source-downloads.json', {
        'ember': {'url': 'https://files.ember-energy.org/public-downloads/generation/outputs/release_generation_yearly_global.csv',
                  'local_path': str(run / 'sources/ember-yearly-complete.csv'),
                  'sha256': digest(run / 'sources/ember-yearly-complete.csv'),
                  'data_year': 2024, 'download_date': '2026-11-03', 'last_modified': '2026-10-30 15:45:34 GMT'},
        'countries': {'url': 'https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip',
                      'local_path': str(run / 'sources/countries-10m.zip'),
                      'sha256': digest(run / 'sources/countries-10m.zip')},
        'weather_receipts': {'records': len(receipts), 'file': receipt_path.name, 'sha256': digest(receipt_path)}})


def completed_shards(folder, partial):
    paths = sorted(folder.glob('*.jsonl'))
    if not partial:
        return [str(path) for path in paths]
    selected = []
    for path in paths:
        payload = path.read_bytes()
        if not payload.endswith(b'\n'):
            continue
        rows = [json.loads(line) for line in payload.splitlines()]
        if len(rows) == 24 and len({row['case_key'] for row in rows}) == 24:
            selected.append(str(path))
    return selected


def load_shards(connection, files, table):
    columns = set()
    for start in range(0, len(files), 32):
        batch = files[start:start + 32]
        connection.execute("""CREATE OR REPLACE TEMP TABLE json_batch AS SELECT * FROM read_json(?,
            format='newline_delimited', union_by_name=true, sample_size=-1,
            maximum_sample_files=32, maximum_object_size=1048576)""", [batch])
        description = connection.execute('DESCRIBE json_batch').fetchall()
        if start == 0:
            connection.execute(f'CREATE OR REPLACE TABLE {table} AS SELECT * FROM json_batch')
            columns = {row[0] for row in description}
        else:
            for name, kind, *_ in description:
                if name not in columns:
                    identifier = '"' + name.replace('"', '""') + '"'
                    connection.execute(f'ALTER TABLE {table} ADD COLUMN {identifier} {kind}')
                    columns.add(name)
            connection.execute(f'INSERT INTO {table} BY NAME SELECT * FROM json_batch')
        connection.execute('DROP TABLE json_batch')
        if start % 3200 == 0 or start + 32 >= len(files):
            print(f'读取{table}分片 {min(start + 32, len(files))}/{len(files)}', flush=True)


def distributions(pairs, folder):
    for unit in ('global', 'continent', 'country'):
        keys = ([] if unit == 'global' else [unit]) + ['scenario', 'alpha', 'boundary']
        groups = pairs.groupby(keys, dropna=False, observed=True)
        rows = []
        for identity, data in groups:
            values = data.savings_pct.dropna()
            rows.append(dict(zip(keys, identity), grids=len(data), valid_pairs=len(values),
                             median_pct=float(values.median()) if len(values) else None,
                             p25_pct=float(values.quantile(.25)) if len(values) else None,
                             p75_pct=float(values.quantile(.75)) if len(values) else None))
        pd.DataFrame(rows).to_csv(folder / (unit + '-distributions.csv'), index=False, encoding='utf-8-sig')


def capacity_statistics(connection, folder):
    table = connection.sql('''SELECT boundary, scenario, alpha, count(*) AS optimal_flex_cases,
        sum(CASE WHEN extra_task_mw>1e-6 THEN 1 ELSE 0 END) AS added_capacity_cases,
        max(extra_task_mw/peak_task_mw) AS maximum_added_capacity_ratio_of_K,
        quantile_cont(extra_task_mw/peak_task_mw, .5) AS median_added_capacity_ratio_of_K,
        sum(CASE WHEN extra_task_mw>peak_task_mw+1e-6 THEN 1 ELSE 0 END) AS added_capacity_above_K_cases,
        sum(CASE WHEN extra_task_mw>1e-6 AND
              extra_task_mw >= (CASE WHEN boundary='fixed' THEN 2 ELSE 1 END)*peak_task_mw-1e-6
            THEN 1 ELSE 0 END) AS added_capacity_limit_hits
        FROM runs WHERE flex='both' AND status='optimal'
        GROUP BY boundary, scenario, alpha ORDER BY boundary, scenario, alpha NULLS FIRST''').df()
    table.to_csv(folder / 'additional-compute-capacity.csv', index=False, encoding='utf-8-sig')
    return table


def city_comparison(pairs, project, run, folder, carbon_year):
    gridmap = pd.read_csv(run / 'inputs/city-grid-points.csv')[['city', 'grid_id', 'lat', 'lon']].rename(
        columns={'lat': 'city_weather_lat', 'lon': 'city_weather_lon'})
    cards = pd.read_csv(project / 'results/flex-card-cost-calendar-20261102/runs-cards.csv')
    fixed = pd.read_csv(project / 'results/flex-card-cost-calendar-20261102/runs-fixed.csv')
    baseline = cards.loc[cards.flex.eq('none') & cards.card_price.astype(str).eq('forced0')].copy()
    planning = cards.loc[cards.flex.eq('both') & pd.to_numeric(cards.card_price, errors='coerce').eq(1.)].copy()
    built = fixed.loc[fixed.flex.eq('both') & pd.to_numeric(fixed.card_price, errors='coerce').eq(1.)].copy()
    reference = []
    for boundary, data in [('planning', planning), ('fixed', built)]:
        current = baseline[['city', 'scenario', 'alpha', 'status', 'total_cost_usd']].merge(
            data[['city', 'scenario', 'alpha', 'status', 'total_cost_usd']], on=['city', 'scenario', 'alpha'], validate='one_to_one', suffixes=('_base', '_flex'))
        valid = current.status_base.eq('optimal') & current.status_flex.eq('optimal')
        current['city_savings_pct'] = np.where(valid, 100 * (current.total_cost_usd_base - current.total_cost_usd_flex) / current.total_cost_usd_base, np.nan)
        current['boundary'] = boundary
        reference.append(current)
    reference = pd.concat(reference, ignore_index=True).merge(gridmap, on='city', validate='many_to_one')
    city_parameters = json.loads((project / 'analysis/eight-city-flex-value-20261029.json').read_text('utf-8'))
    reference['city_country'] = reference.city.map({city: item['country'] for city, item in city_parameters['cities'].items()})
    reference['city_carbon_t_mwh'] = reference.city.map({city: item['emission_factor'] for city, item in city_parameters['cities'].items()})
    reference['city_carbon_year'] = city_parameters['emission_factor_metadata']['data_year']
    reference['grid_carbon_year'] = carbon_year
    result = reference.merge(pairs, on=['grid_id', 'scenario', 'alpha', 'boundary'], how='left', validate='one_to_one', suffixes=('_city', '_grid'))
    result['grid_minus_city_percentage_points'] = result.savings_pct - result.city_savings_pct
    result.to_csv(folder / 'eight-city-grid-comparison.csv', index=False, encoding='utf-8-sig')
    return result


def map_cells(run, pairs):
    """每个代表网格着色其所在 2° 区块的陆地部分（区块中心 block_lat/block_lon，边长 2°）。"""
    import geopandas as gpd
    import shapely

    source = run / 'sources/countries-10m.zip'
    countries = gpd.read_file('zip://' + str(source))
    cache = run / 'inputs/map-block-land-cells.parquet'
    if cache.exists():
        return gpd.read_parquet(cache), countries
    blocks = pairs[['grid_id', 'block_lat', 'block_lon']].drop_duplicates('grid_id')
    index = countries.sindex
    geometries = []
    for row in blocks.itertuples():
        cell = shapely.box(row.block_lon - 1, row.block_lat - 1, row.block_lon + 1, row.block_lat + 1)
        hits = index.query(cell, predicate='intersects')
        parts = [cell.intersection(countries.geometry.iloc[int(hit)]) for hit in hits]
        geometries.append(shapely.union_all(parts) if parts else shapely.GeometryCollection())
    result = gpd.GeoDataFrame({'grid_id': blocks.grid_id.to_numpy()}, geometry=geometries, crs='EPSG:4326')
    result.to_parquet(cache, index=False, compression='zstd')
    return result, countries


def maps(pairs, run, folder, partial):
    os.environ.setdefault('MPLCONFIGDIR', str(run / 'matplotlib-cache'))
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from fontTools.ttLib import TTFont
    import cartopy.crs as ccrs
    import frykit.plot as fplt

    font_family = 'WenQuanYi Zen Hei'
    font_path = Path(font_manager.findfont(font_family, fallback_to_default=False))
    with TTFont(font_path, fontNumber=0) as font:
        if 'glyf' not in font:
            raise ValueError('PDF Type42输出需要TrueType轮廓字体：' + str(font_path))
    mpl.rcParams.update({'font.family': font_family, 'font.size': 9, 'axes.unicode_minus': False,
                         'pdf.fonttype': 42, 'savefig.dpi': 300})
    scenarios = [('D-TRN', 'GPU 主导型·大模型训练/开发'), ('D-SAI', 'GPU 主导型·科研 AI'),
                 ('A-HPC', '纯 CPU 型·科研计算'), ('D-INF', 'GPU 主导型·推理服务')]
    display = pairs.loc[pairs.scenario.isin([key for key, _ in scenarios]) & (pairs.alpha.isna() | pairs.alpha.eq(1.))]
    values = display.savings_pct.dropna()
    if values.empty:
        raise ValueError('指定地图情境没有最优配对')
    cmap = mpl.colormaps['viridis']
    cells, countries = map_cells(run, pairs)
    geometry_audit = cells[['grid_id']].copy()
    geometry_audit['empty_land_intersection'] = cells.geometry.is_empty
    geometry_audit.drop(columns='geometry', errors='ignore').to_csv(folder / 'map-geometry-audit.csv', index=False)
    borders = countries.geometry.simplify(.08, preserve_topology=True).to_list()
    captions = []
    for boundary, boundary_zh in [('planning', '规划边界'), ('fixed', '已建成边界')]:
        for alpha, alpha_label, token in [(None, '经济最优', 'economic'), (1., 'α = 100%', 'alpha100')]:
            fig, axes = plt.subplots(2, 2, figsize=(12.6, 6.8), subplot_kw={'projection': ccrs.Robinson()})
            counts = []
            panel = display.loc[display.boundary.eq(boundary) & (display.alpha.isna() if alpha is None else display.alpha.eq(alpha))].savings_pct.dropna().clip(lower=0)
            # 每幅图按本图 98 分位取色标上限，超出部分用三角延伸标示
            top = float(panel.quantile(.98)) if len(panel) else 1.
            step = 5. if top > 10 else (1. if top > 1 else 10 ** np.floor(np.log10(max(top, 1e-6))))
            vmax = max(step, float(np.ceil(top / step) * step))
            norm = mpl.colors.Normalize(0, vmax)
            for number, (ax, (scenario, label)) in enumerate(zip(axes.flat, scenarios)):
                current = display.loc[display.boundary.eq(boundary) & display.scenario.eq(scenario) & (display.alpha.isna() if alpha is None else display.alpha.eq(alpha))]
                current = cells.merge(current[['grid_id', 'savings_pct']], on='grid_id', how='inner', validate='one_to_one')
                current = current.loc[current.savings_pct.notna() & ~current.geometry.is_empty]
                ax.set_global()
                fplt.add_geometries(ax, borders, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='#a8a8a8', linewidth=.22, zorder=1)
                if len(current):
                    fplt.add_geometries(ax, current.geometry.to_list(), crs=ccrs.PlateCarree(),
                                       facecolor=cmap(norm(current.savings_pct.clip(lower=0).to_numpy())), edgecolor='none', linewidth=0, zorder=2,
                                       rasterized=True)  # 约 4,800 个区块多边形，PDF 中栅格化以控制体积
                ax.set_title(f'({chr(97 + number)}) {scenario}  {label}', fontsize=9, pad=4)
                counts.append({'scenario': scenario, 'colored_grids': len(current)})
            title = ('试跑样本：' if partial else '') + f'{boundary_zh}，{alpha_label}'
            fig.suptitle(title, fontsize=12, y=.99)
            fig.subplots_adjust(left=.015, right=.985, top=.91, bottom=.14, hspace=.12, wspace=.025)
            bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(),
                               orientation='horizontal', fraction=.035, pad=.035, shrink=.65, aspect=45,
                               extend='max' if float(panel.max()) > vmax else 'neither')
            bar.set_label('柔性节省 / 无柔性能源系统成本（%）')
            fig.text(.5, .022, '每个有合格网格的 2° 区块选取一个代表 0.5° 网格，按其结果为整个区块的陆地着色。空白为无合格网格或无有效配对的区块。',
                     ha='center', va='bottom', fontsize=8)
            stem = f'{boundary}-{token}'
            fig.savefig(folder / (stem + '.png'), dpi=300, facecolor='white')
            fig.savefig(folder / (stem + '.pdf'), facecolor='white')
            plt.close(fig)
            captions.append({'file': stem, 'title': title, 'counts': counts, 'color_min_pct': 0, 'color_max_pct': vmax})
    write_json(folder / 'map-manifest.json', {'utc': utc_now(), 'partial': partial, 'projection': 'Robinson, central_longitude=0',
        'renderer_sha256': digest(Path(__file__)), 'data_sha256': digest(folder.parent / 'grid-summary.parquet'),
        'font_family': font_family, 'font_path': str(font_path), 'font_sha256': digest(font_path), 'pdf_fonttype': 42,
        'coordinates': 'WGS84', 'boundary_source': 'Natural Earth 1:10m admin0', 'land_clip': '代表网格所在2度区块与国界陆地多边形交集',
        'color_scale': '每幅图0至本图98分位（取整），超出用三角延伸', 'colormap': 'viridis',
        'selected_grids': len(cells), 'empty_land_intersections': int(cells.geometry.is_empty.sum()), 'panels': captions})
    (folder / '图件说明.md').write_text(
        '四幅全球地图分别展示规划和已建成边界下经济最优及 α=100% 目标的结果，每幅依次为 D-TRN、D-SAI、A-HPC、D-INF。'
        '颜色表示同参数无柔性基线成本的节省百分比，同一幅图的四个子图共用色标，色标上限取该图 98 分位。规划阶段允许风光储和市电接入容量重新优化；已建成阶段固定为无柔性最优资产。'
        '每个 2° 区块按其代表 0.5° 网格的结果整体着色（裁到陆地），表示该区块的抽样代表值，不代表区块内全部土地可建设或结果均一。统计单位为等权抽样网格，未按土地面积、人口或算力规模加权。'
        '无有效最优配对的组合保留在统计表中，地图留空。\n', encoding='utf-8')


def report(args):
    own, paths = settings(args.config)
    run, project = paths['run_root'], args.config.parent.parent
    source = run / args.input
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    progress = json.loads((source / 'progress.json').read_text('utf-8'))
    if not args.partial and progress['state'] != 'complete':
        raise ValueError('求解尚未完成，只允许使用--partial生成调试统计')
    files = completed_shards(source / 'shards', args.partial)
    if not files:
        raise ValueError('没有完整的算例分片')
    cache = run / 'report-cache'
    cache.mkdir(exist_ok=True)
    connection = duckdb.connect(str(cache / (folder.name + '.duckdb')),
                                config={'memory_limit': '4GB', 'threads': 2, 'temp_directory': str(cache),
                                        'preserve_insertion_order': False})
    load_shards(connection, files, 'runs')
    lineage = []
    if args.country_audit:
        audit = json.loads((run / 'validation/country-factor-match-audit.json').read_text('utf-8'))
        replacement_progress = json.loads((args.country_audit / 'progress.json').read_text('utf-8'))
        if not audit['passed'] or replacement_progress['state'] != 'complete':
            raise ValueError('国家因子重算尚未通过核验或完成')
        replacement_files = completed_shards(args.country_audit / 'shards', False)
        load_shards(connection, replacement_files, 'country_runs')
        replacement_count = connection.sql('SELECT count(*) FROM country_runs').fetchall()[0][0]
        replacement_grids = {row[0] for row in connection.sql('SELECT DISTINCT grid_id FROM country_runs').fetchall()}
        if replacement_count != audit['required_summary_rows'] or replacement_grids != set(audit['grid_ids']):
            raise ValueError('国家因子重算网格或算例数与核验清单不同')
        connection.execute('DROP TABLE IF EXISTS base_runs')
        connection.execute('ALTER TABLE runs RENAME TO base_runs')
        connection.execute('''CREATE TEMP TABLE runs AS
            SELECT * FROM base_runs WHERE grid_id NOT IN (SELECT grid_id FROM country_runs)
            UNION ALL BY NAME SELECT * FROM country_runs''')
        lineage = [{'kind': 'base', 'path': str(source), 'manifest': json.loads((source / 'run-manifest.json').read_text('utf-8'))},
                   {'kind': 'country_factor', 'path': str(args.country_audit), 'grid_ids': audit['grid_ids'],
                    'manifest': json.loads((args.country_audit / 'run-manifest.json').read_text('utf-8')),
                    'progress': replacement_progress}]
        country_provenance = folder / 'provenance/country-factor'
        country_provenance.mkdir(parents=True, exist_ok=True)
        for name in ('run-manifest.json', 'progress.json', 'execution-attempts.jsonl', 'active-source-version.json'):
            shutil.copy2(args.country_audit / name, country_provenance / name)
        shutil.copy2(args.config, country_provenance / args.config.name)
    selected = pd.read_parquet(run / 'inputs/selected-grids.parquet')
    assert selected.grid_id.is_unique
    connection.register('selected_metadata', selected)
    mismatched_carbon = connection.sql('''SELECT count(*) FROM runs r LEFT JOIN selected_metadata m USING(grid_id)
        WHERE m.grid_id IS NULL OR r.carbon_source IS DISTINCT FROM m.carbon_source
        OR r.carbon_t_mwh IS DISTINCT FROM m.carbon_t_mwh''').fetchall()[0][0]
    if mismatched_carbon:
        raise ValueError('结果与最终国家碳因子元数据不一致：' + str(mismatched_carbon))
    duplicates = connection.sql('SELECT grid_id, scenario, alpha, boundary, flex, count(*) AS n FROM runs GROUP BY ALL HAVING n > 1').fetchall()
    if duplicates:
        raise ValueError('算例组合重复：' + repr(duplicates[:10]))
    count = connection.sql('SELECT count(*) FROM runs').fetchall()[0][0]
    if not args.partial and count != progress['total']:
        raise ValueError('汇总行数与完成进度不一致')
    full_path = folder / 'runs-full.parquet'
    connection.sql('SELECT * FROM runs ORDER BY grid_id, scenario, alpha NULLS FIRST, boundary, flex').write_parquet(str(full_path), compression='zstd')
    connection.execute('''CREATE TEMP TABLE pairs AS
        SELECT b.grid_id, b.block_id, b.lat, b.lon, m.block_lat, m.block_lon, b.iso3, b.country, b.continent,
               b.carbon_source, b.carbon_gco2_kwh, m.carbon_matched_iso3, m.carbon_match_method,
               m.carbon_reference_area, m.carbon_year, b.land_points, b.scenario, b.alpha, b.boundary,
               b.country_assignment_method, b.nearest_boundary_distance_km, b.country_assignment_distance_km, b.m49_region,
               b.raw_min_temperature_c, b.raw_max_temperature_c, b.low_temperature_clipped_hours,
               b.high_temperature_clipped_hours, b.temperature_clipped_hours, b.dewpoint_clipped_hours,
               b.status AS baseline_status, f.status AS flex_status,
               b.total_cost_usd AS baseline_cost_usd, f.total_cost_usd AS flex_cost_usd,
               CASE WHEN b.status='optimal' AND f.status='optimal' AND b.total_cost_usd>0
                    THEN 100*(b.total_cost_usd-f.total_cost_usd)/b.total_cost_usd END AS savings_pct,
               CASE WHEN b.status='optimal' AND f.status='optimal'
                    THEN b.total_cost_usd-f.total_cost_usd END AS savings_usd,
               b.hours, b.baseline_it_mwh, b.baseline_facility_mwh,
               b.pv_mw AS baseline_pv_mw, b.wind_mw AS baseline_wind_mw,
               b.battery_mw AS baseline_battery_mw, b.battery_mwh AS baseline_battery_mwh,
               b.grid_cap_mw AS baseline_grid_cap_mw,
               f.pv_mw AS flex_pv_mw, f.wind_mw AS flex_wind_mw,
               f.battery_mw AS flex_battery_mw, f.battery_mwh AS flex_battery_mwh,
               f.grid_cap_mw AS flex_grid_cap_mw, f.extra_task_mw,
               f.moved_whole_mwh, f.moved_split_mwh, f.grid_mwh AS flex_grid_mwh,
               f.facility_mwh AS flex_facility_mwh
        FROM runs b LEFT JOIN runs f
          ON b.grid_id=f.grid_id AND b.scenario=f.scenario AND b.alpha IS NOT DISTINCT FROM f.alpha
         AND b.boundary=f.boundary AND f.flex='both'
        LEFT JOIN selected_metadata m ON b.grid_id=m.grid_id
        WHERE b.flex='none'
        ORDER BY b.grid_id, b.scenario, b.alpha NULLS FIRST, b.boundary''')
    pairs = connection.sql('SELECT * FROM pairs').df()
    pair_path = folder / 'grid-summary.parquet'
    compact_columns = ['grid_id', 'block_id', 'lat', 'lon', 'block_lat', 'block_lon', 'iso3', 'country', 'continent',
                       'country_assignment_method', 'nearest_boundary_distance_km', 'country_assignment_distance_km',
                       'carbon_source', 'carbon_gco2_kwh', 'carbon_matched_iso3', 'carbon_match_method',
                       'carbon_reference_area', 'carbon_year', 'm49_region',
                       'scenario', 'alpha', 'boundary', 'baseline_status', 'flex_status', 'baseline_cost_usd', 'flex_cost_usd',
                       'savings_pct', 'extra_task_mw', 'low_temperature_clipped_hours', 'high_temperature_clipped_hours',
                       'temperature_clipped_hours', 'raw_min_temperature_c', 'raw_max_temperature_c', 'land_points']
    pairs[compact_columns].to_parquet(pair_path, index=False, compression='zstd', compression_level=9)
    climate = connection.sql('''SELECT DISTINCT grid_id, scenario, raw_min_temperature_c, raw_max_temperature_c,
        low_temperature_clipped_hours, high_temperature_clipped_hours, temperature_clipped_hours, dewpoint_clipped_hours
        FROM runs ORDER BY grid_id, scenario''').df()
    climate.to_csv(folder / 'cooling-input-clipping.csv.gz', index=False, compression='gzip')
    statuses = connection.sql('SELECT boundary, flex, status, lp_performed, reused_baseline, count(*) AS cases FROM runs GROUP BY ALL ORDER BY ALL').df()
    statuses.to_csv(folder / 'status-counts.csv', index=False, encoding='utf-8-sig')
    valid = pairs.loc[pairs.savings_pct.notna()]
    if (valid.savings_pct < -.001).any():
        raise AssertionError('存在柔性成本高于同参数无柔性基线的有效配对，须核验')
    asset_errors = connection.sql("SELECT count(*) FROM runs WHERE boundary='fixed' AND status='optimal' AND (fixed_asset_equal IS NOT TRUE OR fixed_asset_max_abs_difference>1e-6)").fetchall()[0][0]
    if asset_errors:
        raise AssertionError('固定资产一致性检查失败')
    audits = connection.sql("SELECT count(*) FROM runs WHERE status='optimal' AND dispatch_audit_passed IS NOT TRUE").fetchall()[0][0]
    if audits:
        raise AssertionError('最优算例缺少独立调度验收')
    distributions(pairs, folder)
    capacity_statistics(connection, folder)
    city_comparison(pairs, project, run, folder, own['carbon']['year'])
    figures = folder / 'figures'
    figures.mkdir(exist_ok=True)
    partial = bool(args.partial or progress['scope'] != 'all')
    metadata = {'utc': utc_now(), 'partial': partial, 'scope': progress['scope'], 'source': str(source),
                'reporter_sha256': digest(Path(__file__)),
                'scientific_fingerprint': json.loads((source / 'run-manifest.json').read_text('utf-8'))['scientific_fingerprint'],
                'summary_rows': count, 'pairs': len(pairs),
                'valid_pairs': len(valid), 'fixed_asset_errors': asset_errors, 'missing_dispatch_audits': audits,
                'status_counts': statuses.to_dict('records'), 'progress': progress, 'execution_lineage': lineage,
                'final_actual_lps': connection.sql('SELECT count(*) FROM runs WHERE lp_performed').fetchall()[0][0],
                'final_optimal_lps': connection.sql("SELECT count(*) FROM runs WHERE lp_performed AND status='optimal'").fetchall()[0][0],
                'final_reused_baselines': connection.sql('SELECT count(*) FROM runs WHERE reused_baseline').fetchall()[0][0],
                'final_input_unavailable_rows': connection.sql("SELECT count(*) FROM runs WHERE starts_with(status, 'input_')").fetchall()[0][0],
                'final_carbon_metadata_mismatches': mismatched_carbon,
                'artifacts': {path.name: {'sha256': digest(path), 'bytes': path.stat().st_size} for path in (full_path, pair_path)}}
    write_json(folder / 'summary-manifest.json', metadata)
    provenance(run, source, folder)
    if args.maps:
        maps(pairs, run, figures, partial)
    connection.close()
    print(json.dumps({key: metadata[key] for key in ('partial', 'summary_rows', 'pairs', 'valid_pairs')}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('global-flex-econ-20261103.json'))
    parser.add_argument('--input', default='full')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--maps', action='store_true')
    parser.add_argument('--maps-only', action='store_true')
    parser.add_argument('--country-audit', type=Path)
    args = parser.parse_args()
    if args.maps_only:
        own, paths = settings(args.config)
        manifest = json.loads((args.output / 'summary-manifest.json').read_text('utf-8'))
        table = args.output / 'grid-summary.parquet'
        if digest(table) != manifest['artifacts']['grid-summary.parquet']['sha256']:
            raise ValueError('地图输入与已验证汇总表的SHA不一致')
        maps(pd.read_parquet(table), paths['run_root'], args.output / 'figures', manifest['partial'])
    else:
        report(args)


if __name__ == '__main__':
    main()
