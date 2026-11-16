import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    original, verified_input = root / 'forecast', root / 'forecast-terminal-verified'
    for folder, count in [(original, 176), (verified_input, 176)]:
        if not (folder / 'summary.csv').is_file() or len(list(folder.glob('*.json'))) != count:
            raise ValueError(f'滚动回放尚未完成: {folder}')
    clock_path = root / 'forecast-clock-validation.json'
    clock = json.loads(clock_path.read_text())
    if (len(clock['clock_checks']) != 33 or
            max(item['reference_difference'] for item in clock['clock_checks']) > 1e-7):
        raise ValueError('真实时间期限核验未通过')
    audit_folder = root / 'forecast-seq-verified'
    audit_path = audit_folder / 'audit.json'
    audit = json.loads(audit_path.read_text())
    if (audit['total_controls'] != 352 or audit['max_difference_usd'] > .02 or
            audit['max_residual'] > 2e-6):
        raise ValueError('顺序调度的独立回放核验未通过')
    audit_files = sorted(path for path in audit_folder.glob('*.json') if path.name != 'audit.json')
    if {path.name: sha256(path.read_bytes()).hexdigest() for path in audit_files} != audit['case_hashes']:
        raise ValueError('顺序调度审计文件的校验值不一致')
    archive = root / 'forecast-controller-reference'
    archive.mkdir(exist_ok=True)
    verified, cost_differences, applied = [], [], set()
    terminal_rule = 'backlog_by_class_no_greater_than_earliest_deadline_execution'
    for path in sorted(verified_input.glob('*.json')):
        destination = original / path.name
        reference_path = archive / path.name if (archive / path.name).exists() else destination
        previous, row = json.loads(reference_path.read_text()), json.loads(path.read_text())
        if row['terminal_rule'] != terminal_rule:
            raise ValueError('滚动计算缺少窗口末积压约束')
        for key in ['city', 'scenario', 'intensity_of_max', 'hours', 'E0_mwh', 'external_online_mwh']:
            if previous[key] != row[key]:
                raise ValueError(f'回放基础输入不一致 {path.name} {key}')
        np.testing.assert_array_equal(previous['flex_arrival_mwh'], row['flex_arrival_mwh'])
        arrivals = np.asarray(row['flex_arrival_mwh'])
        if len(row['cases']) != 4:
            raise ValueError('预测配对控制情景不完整')
        for index, case in enumerate(row['cases']):
            if case['mode'] == 'seq':
                reference = next(item for item in previous['cases']
                                 if (item['mode'], item['forecast']) == (case['mode'], case['forecast']))
                if abs(case['cost_usd'] - reference['cost_usd']) > .02:
                    record_path = audit_folder / f'{path.stem}-{case["forecast"]}.json'
                    record = json.loads(record_path.read_text())
                    if (record['reference_sha256'] != sha256(reference_path.read_bytes()).hexdigest() or
                            record['terminal_result_sha256'] != sha256(path.read_bytes()).hexdigest() or
                            record['name'] != path.name or record['forecast'] != case['forecast'] or
                            record['case']['mode'] != 'seq' or record['case']['forecast'] != case['forecast']):
                        raise ValueError('顺序调度独立回放与输入记录不一致')
                    case = record['case']
                    row['cases'][index] = case
                    applied.add(record_path.name)
                difference = abs(case['cost_usd'] - reference['cost_usd'])
                if difference > .02:
                    raise ValueError('顺序调度的供能费用与既有结果不一致')
                cost_differences.append(difference)
            np.testing.assert_allclose(np.asarray(case['executed_mwh']) + case['unserved_mwh'], arrivals,
                                       rtol=0, atol=1e-5)
            if (case['max_residual'] > 2e-6 or case['conservation_residual_mwh'] > 1e-5 or
                    (np.asarray(case['deadline_missed_mwh']) > arrivals + 1e-5).any()):
                raise ValueError('预测回放约束核验失败')
            if case['forecast'] == 'perfect' and (sum(case['unserved_mwh']) >= 1e-5 or sum(case['deadline_missed_mwh']) >= 1e-5):
                raise ValueError(f'完全预知对照存在未交付工作: {path.name} {case["mode"]}')
        actual = {case['mode']: case for case in row['cases'] if case['forecast'] == 'previous_day'}
        oracle = {case['mode']: case for case in row['cases'] if case['forecast'] == 'perfect'}
        complete = all(sum(case['unserved_mwh']) < 1e-5 for case in actual.values())
        deadlines_met = complete and all(sum(case['deadline_missed_mwh']) < 1e-5 for case in actual.values())
        saving = (actual['seq']['cost_usd'] - actual['joint']['cost_usd']) / row['E0_mwh']
        row.update(service_complete=complete, all_deadlines_met=deadlines_met,
                   realized_coordination_usd_mwh=saving if complete else None,
                   deadline_feasible_coordination_usd_mwh=saving if deadlines_met else None,
                   perfect_rolling_coordination_usd_mwh=(oracle['seq']['cost_usd'] - oracle['joint']['cost_usd']) / row['E0_mwh'],
                   forecast_extra_joint_cost_usd_mwh=(actual['joint']['cost_usd'] - oracle['joint']['cost_usd']) / row['E0_mwh'])
        verified.append((path, destination, row))
    if applied != set(audit['case_hashes']) or len(applied) != audit['fresh_replays']:
        raise ValueError('顺序调度独立回放覆盖不完整')
    keys = {(row['city'], row['scenario'], row['intensity_of_max']) for _, _, row in verified}
    if len(keys) != 176:
        raise ValueError('滚动回放的配对键重复')
    for path, destination, row in verified:
        if not (archive / destination.name).exists():
            shutil.copy2(destination, archive / destination.name)
        destination.write_text(json.dumps(row, allow_nan=False))
    rows = [json.loads(path.read_text()) for path in sorted(original.glob('*.json'))]
    pd.DataFrame([{k: v for k, v in row.items() if k != 'cases'} for row in rows]).to_csv(original / 'summary.csv', index=False)
    report = dict(continuous_groups=128, training_groups=48, total_groups=176,
                  terminal_rule=terminal_rule, perfect_deadline_feasible_pairs=176,
                  sequential_control_replays=len(cost_differences),
                  sequential_cost_max_difference_usd=max(cost_differences),
                  sequential_fresh_replays=len(applied),
                  sequential_audit_sha256=sha256(audit_path.read_bytes()).hexdigest(),
                  clock_validation_sha256=sha256(clock_path.read_bytes()).hexdigest(),
                  case_hashes={path.name: sha256(path.read_bytes()).hexdigest() for path in sorted(original.glob('*.json'))})
    (root / 'forecast-clock-review.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'case_hashes'}))


if __name__ == '__main__':
    main()
