import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd


KEYS = ['city', 'scenario', 'alpha', 'flex', 'card_price']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit = json.loads((args.current / 'independent-dispatch-audit.json').read_text('utf-8'))
    if not audit['passed']:
        raise ValueError('逐小时独立验收未通过')
    if audit['case_count'] != 1296 or audit['index_sha256'] != sha256((args.current / 'hourly-index.csv').read_bytes()).hexdigest():
        raise ValueError('逐小时审查范围或索引指纹不符')
    economics = json.loads((args.current / 'economic-consistency-audit.json').read_text('utf-8'))
    if not economics['passed']:
        raise ValueError('成本一致性验收未通过')
    records = []
    for name, expected in [('runs-cards.csv', 864), ('runs-fixed.csv', 432)]:
        prior_path, new_path = args.prior / name, args.current / name
        if economics['checks'][name]['new_source_sha256'] != sha256(new_path.read_bytes()).hexdigest():
            raise ValueError('成本审查对应的训练汇总指纹不符：' + name)
        if economics['checks'][name]['prior_source_sha256'] != sha256(prior_path.read_bytes()).hexdigest():
            raise ValueError('成本审查对应的原汇总指纹不符：' + name)
        old = pd.read_csv(prior_path, dtype={'card_price': str}, float_precision='round_trip')
        new = pd.read_csv(new_path, dtype={'card_price': str}, float_precision='round_trip')
        if len(new) != expected or not new.scenario.str.endswith('-TRN').all() or not new.status.eq('optimal').all():
            raise ValueError('训练重算范围或状态不符：' + name)
        retained = old.loc[~old.scenario.str.endswith('-TRN')]
        merged = pd.concat([retained, new], ignore_index=True).sort_values(KEYS, na_position='first')
        if merged.duplicated(KEYS).any() or len(merged) != len(old):
            raise ValueError('合并结果数量或唯一键不符：' + name)
        target = args.output / name
        merged.to_csv(target, index=False, encoding='utf-8-sig')
        readback = pd.read_csv(target, dtype={'card_price': str}, float_precision='round_trip')
        unchanged = readback.loc[~readback.scenario.str.endswith('-TRN'), old.columns].sort_values(KEYS, na_position='first').reset_index(drop=True)
        pd.testing.assert_frame_equal(retained.sort_values(KEYS, na_position='first').reset_index(drop=True), unchanged, check_dtype=False, check_exact=True)
        records.append({'file': name, 'total_rows': len(merged), 'recomputed_training_rows': len(new),
            'unchanged_other_rows': len(retained), 'unchanged_other_values_verified_exact': True,
            'prior_sha256': sha256(prior_path.read_bytes()).hexdigest(), 'training_sha256': sha256(new_path.read_bytes()).hexdigest(),
            'output_sha256': sha256(target.read_bytes()).hexdigest()})
    (args.output / 'delivery-manifest.json').write_text(json.dumps({'files': records,
        'deadline_basis': 'Native elapsed source_hour_index; deadlines before a missing-data interval are satisfied in observed hours before the gap',
        'calculation_scope': 'Original retained native hours only; no new task, weather, generation or execution values added for gaps',
        'economic_consistency': economics, 'hourly_audit_file': 'independent-dispatch-audit.json'}, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'economic-consistency-audit.json').write_bytes((args.current / 'economic-consistency-audit.json').read_bytes())
    (args.output / 'independent-dispatch-audit.json').write_bytes((args.current / 'independent-dispatch-audit.json').read_bytes())
    (args.output / 'hourly-index.csv').write_bytes((args.current / 'hourly-index.csv').read_bytes())
    print(json.dumps(records, ensure_ascii=False))


if __name__ == '__main__':
    main()
