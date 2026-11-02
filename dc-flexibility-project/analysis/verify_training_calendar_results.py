import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ['city', 'scenario', 'target', 'flex', 'card_price']
ASSETS = ['pv_mw', 'wind_mw', 'battery_mw', 'battery_mwh', 'grid_cap_mw']


def load(path):
    frame = pd.read_csv(path, dtype={'card_price': str})
    frame['target'] = frame.alpha.fillna(-1)
    if frame.duplicated(KEYS).any():
        raise ValueError('结果键重复：' + str(path))
    return frame


def tolerance(a, b):
    return .02 + 2e-6 * np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports, failures = {}, []
    for name, expected in [('runs-cards.csv', 864), ('runs-fixed.csv', 432)]:
        prior, current = load(args.prior / name), load(args.current / name)
        current = current[current.scenario.str.endswith('-TRN')]
        if len(current) != expected or not current.status.eq('optimal').all():
            raise ValueError(f'{name}应有{expected}个训练最优结果')
        if not np.isfinite(current[['total_cost_usd', *ASSETS]].to_numpy(float)).all():
            raise ValueError('成本或装机字段含非有限值：' + name)
        prior_training = prior.loc[prior.scenario.str.endswith('-TRN')]
        if set(map(tuple, current[KEYS].to_numpy())) != set(map(tuple, prior_training[KEYS].to_numpy())):
            raise ValueError('新旧训练组合键集合不相同：' + name)
        old = prior[prior.scenario.str.endswith('-TRN') & prior.status.eq('optimal')].set_index(KEYS)
        new = current.set_index(KEYS)
        comparable = new.index.intersection(old.index)
        delta = new.loc[comparable].total_cost_usd - old.loc[comparable].total_cost_usd
        tol = tolerance(new.loc[comparable].total_cost_usd, old.loc[comparable].total_cost_usd)
        baseline = comparable.get_level_values('flex') == 'none'
        asset_delta = (new.loc[comparable, ASSETS] - old.loc[comparable, ASSETS]).abs().max(axis=1)
        same_assets = asset_delta.lt(1e-5) if name == 'runs-fixed.csv' else pd.Series(True, index=comparable)
        changed_baseline = baseline & delta.abs().gt(tol)
        lower_after_tighter_deadline = (~baseline) & same_assets & delta.lt(-tol)
        pair_keys = ['city', 'scenario', 'target']
        baseline_rows = current[current.flex.eq('none')].set_index(pair_keys)
        flex_rows = current[current.flex.eq('both')].copy()
        flex_rows = flex_rows.join(baseline_rows.total_cost_usd.rename('baseline_cost_usd'), on=pair_keys)
        if flex_rows.baseline_cost_usd.isna().any():
            raise ValueError('柔性结果缺少同参数无柔性配对：' + name)
        worse_than_baseline = flex_rows.total_cost_usd.gt(flex_rows.baseline_cost_usd + tolerance(flex_rows.total_cost_usd, flex_rows.baseline_cost_usd))
        for label, mask in [('baseline_cost_changed', changed_baseline), ('cost_lower_after_tighter_deadline', lower_after_tighter_deadline)]:
            if mask.any():
                failures.append({'table': name, 'check': label, 'cases': [list(index) for index in comparable[mask]]})
        if worse_than_baseline.any():
            failures.append({'table': name, 'check': 'flex_cost_above_no_flex', 'cases': flex_rows.loc[worse_than_baseline, KEYS].to_dict('records')})
        reports[name] = {'training_rows': len(current), 'optimal_rows': int(current.status.eq('optimal').sum()),
            'comparable_prior_optimal_rows': len(comparable), 'same_assets_rows': int(same_assets.sum()),
            'baseline_cost_max_abs_difference_usd': float(delta.loc[baseline].abs().max()),
            'minimum_flex_cost_change_usd': float(delta.loc[~baseline & same_assets].min()),
            'new_source_sha256': sha256((args.current / name).read_bytes()).hexdigest(),
            'prior_source_sha256': sha256((args.prior / name).read_bytes()).hexdigest()}
    result = {'passed': not failures, 'cost_tolerance': '0.02 USD + 2e-6 * max(abs(compared costs), 1 USD)',
        'checks': reports, 'failures': failures}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    if failures:
        raise RuntimeError('成本一致性检查未通过，详见输出')


if __name__ == '__main__':
    main()
