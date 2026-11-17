"""全球网格柔性收益的归因：逐网格资源与成本特征 → 柔性收益%（经济最优目标）的标准化线性回归。

在服务器运行：
  venv/bin/python project/analysis/global_flex_attribution.py --run-root <运行目录> --output <输出目录>
读取 <运行目录>/inputs/grids/*.npz（2025 年逐小时风光出力系数与气温）、report-full/grid-summary.parquet 与 report-full/runs-full.parquet，
输出 grid-features.parquet、attribution-table.parquet、regression.csv、wind-bins.csv、summary.json。
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXOGENOUS = {'cf_wind': '风电年均出力系数', 'cf_pv': '光伏年均出力系数', 'carbon': '市电碳因子', 'pv_wind_corr': '风光逐小时相关',
             'wind_daily_cv': '风电逐日波动', 'pv_season': '光伏季节差', 'temp_mean': '年均气温', 'abs_lat': '纬度绝对值'}
MECHANISM = {'grid_share': '无柔性方案市电占比', 'pv_gen_share': '无柔性方案光伏发电占比', 'battery_h': '无柔性方案储能时长',
             'curtail_share': '无柔性方案弃电率'}


def features(run):
    rows = []
    for path in sorted((run / 'inputs/grids').glob('*.npz')):
        data = np.load(path)
        pv, wind, temp = data['cf_pv'], data['cf_wind'], data['temperature_c']
        daily_wind = wind.reshape(365, 24).mean(1)
        monthly_pv = pd.Series(pv, index=pd.date_range('2025-01-01', periods=8760, freq='h')).resample('MS').mean()
        rows.append({'grid_id': path.stem, 'cf_pv': pv.mean(), 'cf_wind': wind.mean(),
                     'pv_wind_corr': np.corrcoef(pv, wind)[0, 1] if wind.std() > 0 else 0.,
                     'wind_daily_cv': daily_wind.std() / max(daily_wind.mean(), 1e-9),
                     'pv_season': monthly_pv.max() / max(monthly_pv.min(), 1e-3), 'temp_mean': temp.mean()})
    return pd.DataFrame(rows)


def ols(frame, columns, target='savings_pct'):
    """标准化自变量的 OLS；返回每 1 个标准差的系数（百分点）、HC1 稳健 t、单变量相关与去掉该变量后的 R² 降幅。"""
    z = (frame[columns] - frame[columns].mean()) / frame[columns].std()
    design = np.column_stack([np.ones(len(frame)), z.to_numpy()])
    y = frame[target].to_numpy()
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    n, k = design.shape
    inverse = np.linalg.inv(design.T @ design)
    cov = inverse @ (design.T * resid ** 2) @ design @ inverse * n / (n - k)
    r2 = 1 - resid.var() / y.var()
    drop = []
    for i in range(len(columns)):
        reduced = np.delete(design, i + 1, 1)
        b2, *_ = np.linalg.lstsq(reduced, y, rcond=None)
        drop.append(r2 - (1 - (y - reduced @ b2).var() / y.var()))
    return pd.DataFrame({'variable': columns, 'beta_pp_per_sd': beta[1:], 't_hc1': beta[1:] / np.sqrt(np.diag(cov))[1:],
                         'univariate_r': [np.corrcoef(frame[c], y)[0, 1] for c in columns], 'delta_r2': drop}), r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run_root, args.output
    out.mkdir(parents=True, exist_ok=True)
    feat = features(run)
    feat.to_parquet(out / 'grid-features.parquet', index=False)

    summary = pd.read_parquet(run / 'report-full/grid-summary.parquet')
    runs = pd.read_parquet(run / 'report-full/runs-full.parquet', columns=['grid_id', 'scenario', 'alpha', 'boundary', 'flex', 'status',
                           'pv_mw', 'wind_mw', 'battery_mwh', 'facility_mwh', 'grid_mwh', 'hours', 'generation_mwh', 'curtail_mwh'])
    base = runs.loc[runs.boundary.eq('planning') & runs.flex.eq('none') & runs.alpha.isna() & runs.status.eq('optimal')]
    base = base.merge(feat[['grid_id', 'cf_pv', 'cf_wind']], on='grid_id')
    pv_gen, wind_gen = base.pv_mw * base.cf_pv, base.wind_mw * base.cf_wind
    base = base.assign(pv_gen_share=pv_gen / (pv_gen + wind_gen).where(pv_gen + wind_gen > 0),
                       battery_h=base.battery_mwh / (base.facility_mwh / base.hours),
                       grid_share=base.grid_mwh / base.facility_mwh,
                       curtail_share=base.curtail_mwh / base.generation_mwh.where(base.generation_mwh > 0))
    base[['pv_gen_share', 'curtail_share']] = base[['pv_gen_share', 'curtail_share']].fillna(0.)
    valid = summary.loc[summary.baseline_status.eq('optimal') & summary.flex_status.eq('optimal') & summary.alpha.isna()]
    table = (valid.merge(feat, on='grid_id')
             .merge(base[['grid_id', 'scenario', *MECHANISM]], on=['grid_id', 'scenario'])
             .assign(abs_lat=lambda d: d.lat.abs(), carbon=lambda d: d.carbon_gco2_kwh))
    keep = ['grid_id', 'lat', 'lon', 'country', 'continent', 'scenario', 'boundary', 'savings_pct', *EXOGENOUS, *MECHANISM]
    table[keep].to_parquet(out / 'attribution-table.parquet', index=False)

    results, fits = [], []
    for (scenario, boundary), frame in table.groupby(['scenario', 'boundary']):
        exo, r2_exo = ols(frame, list(EXOGENOUS))
        full, r2_full = ols(frame, list(EXOGENOUS) + list(MECHANISM))
        for model, coef in (('exogenous', exo), ('with_mechanism', full)):
            results.append(coef.assign(scenario=scenario, boundary=boundary, model=model, n=len(frame)))
        fits.append({'scenario': scenario, 'boundary': boundary, 'n': len(frame), 'r2_exogenous': r2_exo, 'r2_with_mechanism': r2_full,
                     'median_savings_pct': frame.savings_pct.median()})
    regression = pd.concat(results, ignore_index=True)
    regression['label'] = regression.variable.map({**EXOGENOUS, **MECHANISM})
    regression.to_csv(out / 'regression.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(fits).to_csv(out / 'fit.csv', index=False, encoding='utf-8-sig')

    edges = [0, .05, .1, .15, .2, .25, .3, .35, .45, 1.]
    bins = (table.assign(wind_bin=pd.cut(table.cf_wind, edges, include_lowest=True))
            .groupby(['scenario', 'boundary', 'wind_bin'], observed=True)
            .agg(n=('savings_pct', 'size'), savings_median=('savings_pct', 'median'), savings_p25=('savings_pct', lambda s: s.quantile(.25)),
                 savings_p75=('savings_pct', lambda s: s.quantile(.75)), grid_share_median=('grid_share', 'median'),
                 cf_wind_median=('cf_wind', 'median')).reset_index())
    bins['wind_bin'] = bins.wind_bin.astype(str)
    bins.to_csv(out / 'wind-bins.csv', index=False, encoding='utf-8-sig')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (out / 'summary.json').write_text(json.dumps({
        'target': 'savings_pct（经济最优目标，规划/已建成两种边界，有效最优配对）',
        'method': 'OLS，自变量标准化（系数为每 1 个标准差对应的百分点），HC1 稳健标准误；未处理空间自相关，t 值偏乐观，重要性以去掉该变量后的 R² 降幅为主',
        'feature_year': '2025 年全年逐小时（与各情境原生窗口对应的气象小时不完全相同）',
        'grids': int(table.grid_id.nunique()), 'exogenous': EXOGENOUS, 'mechanism': MECHANISM,
        'inputs': {'grid-summary.parquet': digest(run / 'report-full/grid-summary.parquet')},
    }, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(pd.DataFrame(fits).round(3).to_string())


if __name__ == '__main__':
    main()
