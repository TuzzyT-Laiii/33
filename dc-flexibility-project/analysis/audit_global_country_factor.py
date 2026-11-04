import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from global_flex_inputs import digest, select, utc_now, write_json


def functions(path, names):
    tree = ast.parse(path.read_text('utf-8'))
    return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names}


def prepare(args):
    base, run = args.base_root, args.run_root
    if base == run:
        raise ValueError('国家因子核验必须使用独立运行目录')
    project = Path(__file__).parent.parent
    config = project / 'analysis/global-flex-econ-20261103.json'
    own = json.loads(config.read_text('utf-8'))
    own['paths']['run_root'] = str(run)
    write_json(config, own)
    for folder in ('sources', 'inputs', 'validation', 'logs'):
        (run / folder).mkdir(parents=True, exist_ok=True)
    for name in ('grids', 'templates', 'raw-blocks', 'wind10-source'):
        target = run / 'inputs' / name
        if target.exists():
            assert target.resolve() == (base / 'inputs' / name).resolve()
        else:
            target.symlink_to(base / 'inputs' / name, target_is_directory=True)
    for name in ('grid-input-index-all.csv', 'template-index.csv', 'country-boundary-cache.parquet',
                 'country-boundary-cache.json', 'extraction-all.json', 'wind10-all.json', 'cf-build-all.json',
                 'map-sampled-land-cells.parquet'):
        shutil.copy2(base / 'inputs' / name, run / 'inputs' / name)
    for name in ('ember-yearly-complete.csv', 'ember-complete-headers.txt', 'countries-10m.zip', 'un-m49.html'):
        shutil.copy2(base / 'sources' / name, run / 'sources' / name)
    for path in (base / 'validation').iterdir():
        if path.is_file():
            shutil.copy2(path, run / 'validation' / path.name)
    select(config)
    old = pd.read_parquet(base / 'inputs/selected-grids.parquet').set_index('grid_id').sort_index()
    new = pd.read_parquet(run / 'inputs/selected-grids.parquet').set_index('grid_id').sort_index()
    pd.testing.assert_index_equal(old.index, new.index)
    changed = old.carbon_t_mwh.ne(new.carbon_t_mwh)
    grids = new.index[changed].tolist()
    if len(grids) != 1 or new.loc[grids, 'country'].tolist() != ['Kosovo']:
        raise AssertionError('本次国家名称匹配影响范围超出已核验的一格')
    np.testing.assert_allclose(new.loc[grids, 'carbon_gco2_kwh'], [924.9], atol=0, rtol=0)
    assert new.loc[grids, 'carbon_source'].eq('country').all()
    allowed = {'carbon_gco2_kwh', 'carbon_t_mwh', 'carbon_source', 'carbon_reference_area', 'ember_region',
               'carbon_regional_country_count', 'carbon_regional_generation_twh'}
    pd.testing.assert_frame_equal(old.drop(columns=list(allowed)), new[old.columns].drop(columns=list(allowed)))
    pd.testing.assert_frame_equal(old.loc[~changed], new.loc[~changed, old.columns])
    pd.testing.assert_frame_equal(pd.read_parquet(base / 'inputs/grid-points.parquet'),
                                  pd.read_parquet(run / 'inputs/grid-points.parquet'))
    pilot = json.loads((run / 'inputs/pilot-grids.json').read_text('utf-8'))
    assert not set(grids).intersection(pilot['city_grids'] + pilot['random_grids'])
    names = ['facility_frame', 'weather_frame']
    assert functions(base / 'project/analysis/global_flex_inputs.py', names) == functions(project / 'analysis/global_flex_inputs.py', names)
    names = ['build_frame', 'run_group', 'audit_dispatch']
    assert functions(base / 'project/analysis/global_flex_econ_run.py', names) == functions(project / 'analysis/global_flex_econ_run.py', names)
    core = ['flex_econ_copt.py', 'eight_city_physics.py', 'flex-econ-copt-20261102.json',
            'eight-city-flex-value-20261029.json', 'compute_load/model.py', 'compute_load/cooling.py',
            'compute_load/assets/cooling-20261028-v2.json']
    for name in core:
        assert digest(base / 'project/analysis' / name) == digest(project / 'analysis' / name), name
    write_json(run / 'inputs/carbon-country-grid-ids.json', {'grid_ids': grids})
    columns = ['country', 'iso3', 'carbon_gco2_kwh', 'carbon_source']
    write_json(run / 'validation/country-factor-match-audit.json', {
        'utc': utc_now(), 'passed': True, 'base_root': str(base), 'run_root': str(run), 'grid_ids': grids,
        'before': json.loads(old.loc[grids, columns].reset_index().to_json(orient='records')),
        'after': json.loads(new.loc[grids, columns + ['carbon_matched_iso3', 'carbon_match_method']].reset_index().to_json(orient='records')),
        'unaffected_grids_identical': len(new) - len(grids), 'pilot_unaffected': True,
        'core_physics_cost_sha256_identical': core, 'dispatch_and_cooling_ast_identical': True,
        'input_index_sha256': digest(run / 'inputs/grid-input-index-all.csv'),
        'base_selected_sha256': digest(base / 'inputs/selected-grids.parquet'),
        'selected_sha256': digest(run / 'inputs/selected-grids.parquet'),
        'carbon_source_counts': new.carbon_source.value_counts().to_dict(),
        'required_actual_lps': len(grids) * len(own['scenarios']) * len(own['alpha']) * 3,
        'required_summary_rows': len(grids) * len(own['scenarios']) * len(own['alpha']) * 4})
    print('国家因子与未受影响网格核验通过', grids, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'results'])
    parser.add_argument('--base-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args)
    else:
        results(args)


def results(args):
    run, base = args.run_root, args.base_root
    audit = json.loads((run / 'validation/country-factor-match-audit.json').read_text('utf-8'))
    progress = json.loads((run / 'copt/progress.json').read_text('utf-8'))
    assert progress['state'] == 'complete' and progress['completed'] == audit['required_summary_rows']
    own = json.loads((run / 'project/analysis/global-flex-econ-20261103.json').read_text('utf-8'))
    rows, baseline = [], []
    for grid in audit['grid_ids']:
        for scenario in own['scenarios']:
            new_path = run / 'copt/shards' / (grid + '-' + scenario + '.jsonl')
            old_path = base / 'full/shards' / new_path.name
            current = [json.loads(line) for line in new_path.read_text('utf-8').splitlines()]
            assert len(current) == 24 and len({row['case_key'] for row in current}) == 24
            rows.extend(current)
            baseline.extend(json.loads(line) for line in old_path.read_text('utf-8').splitlines())
    new, old = pd.DataFrame(rows), pd.DataFrame(baseline)
    assert new.carbon_source.eq('country').all() and new.carbon_gco2_kwh.eq(924.9).all()
    optimal = new.loc[new.status.eq('optimal')]
    assert optimal.dispatch_audit_passed.eq(True).all()
    fixed = optimal.loc[optimal.boundary.eq('fixed')]
    assert fixed.fixed_asset_equal.eq(True).all() and fixed.fixed_asset_max_abs_difference.le(1e-6).all()
    paired = new.merge(old, on=['grid_id', 'scenario', 'case_key'], validate='one_to_one', suffixes=('_new', '_old'))
    zero_grid = paired.loc[paired.alpha_new.eq(1.) & paired.status_new.eq('optimal') & paired.status_old.eq('optimal')]
    difference = (zero_grid.total_cost_usd_new - zero_grid.total_cost_usd_old).abs()
    if not difference.le(.01).all():
        raise AssertionError('零市电约束下仅改变电网碳因子仍改变目标值，须核查')
    value = {'utc': utc_now(), 'passed': True, 'rows': len(new), 'actual_lps': int(new.lp_performed.sum()),
             'reused_baselines': int(new.reused_baseline.sum()), 'status_counts': dict(Counter(new.status)),
             'dispatch_and_fixed_asset_checks_passed': True, 'zero_grid_comparisons': len(zero_grid),
             'zero_grid_maximum_cost_difference_usd': float(difference.max()),
             'maximum_cost_difference_other_alpha_usd': float((paired.total_cost_usd_new - paired.total_cost_usd_old).abs().max()),
             'auditor_sha256': digest(Path(__file__))}
    write_json(run / 'validation/country-factor-recompute-audit.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
