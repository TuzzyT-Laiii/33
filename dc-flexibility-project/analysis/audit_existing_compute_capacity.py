import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


ORDER = ['A-HPC', 'A-CLD', 'B-SAI', 'C-SAI', 'D-SAI', 'B-TRN', 'C-TRN', 'D-TRN', 'B-INF', 'C-INF', 'D-INF']
FIELDS = {'HPC': ('u_cpu_proxy', 'CPU核心分配量按源窗口峰值归一化'),
          'CLD': ('u_cpu_proxy', '整机CPU小时占用率'),
          'SAI': ('allocated_gpu_fraction', '已分配GPU小时按源窗口小时峰值归一化'),
          'TRN': ('allocated_gpu_fraction', '已分配GPU小时除以完整源作业瞬时并发峰值2032'),
          'INF': ('allocated_gpu_fraction', '推理子集GPU资源小时按既有需求曲线基准归一化')}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    project, fingerprints = args.project, {}
    reps = pd.read_csv(project / 'results/flex-class-20261101/representative-placements.csv').set_index('scenario')
    configs = {r['configuration_id']: r for r in json.loads((project / 'results/server-count-scenarios-20261028-v2/server-configurations.json').read_text('utf-8'))}
    config_ids = {'A': 'C20', 'B': 'C21', 'C': 'C22', 'D': 'C28'}
    native = []
    for scenario in ORDER:
        key, use = scenario.split('-')
        path = project / 'results/typical-load-curves-20261101-v4-seren/hourly' / (scenario + '-10MW-2025-' + reps.loc[scenario, 'placement'] + '.parquet')
        fingerprints[path.relative_to(project).as_posix()] = digest(path)
        frame = pd.read_parquet(path)
        field, meaning = FIELDS[use]
        excluded = frame.source_unresolved.astype(bool) if 'source_unresolved' in frame else pd.Series(False, index=frame.index)
        values = frame.loc[~excluded, field]
        if not np.isfinite(values).all() or (values < -1e-10).any():
            raise ValueError('分配份额含无效数值：' + scenario)
        peak = float(values.max())
        ratio = values / peak
        units = int(frame.unit_count.iloc[0])
        config = configs[config_ids[key]]
        elapsed_hours = pd.to_numeric(frame.source_hour_index, errors='raise')
        if not np.isfinite(elapsed_hours).all() or not elapsed_hours.diff().iloc[1:].gt(0).all():
            raise ValueError('源小时轴含无效值或未严格递增：' + scenario)
        native.append({'scenario': scenario, 'scenario_name': frame.scenario_zh.iloc[0], 'source_field': field,
            'field_meaning': meaning, 'hours': len(frame), 'valid_hours': len(values), 'unknown_hours': int(excluded.sum()),
            'calendar_gap_count': int(elapsed_hours.diff().gt(1).sum()),
            'elapsed_hour_axis': 'source_hour_index',
            'mean_fraction': float(values.mean()), 'peak_fraction': peak, 'minimum_fraction': float(values.min()),
            'mean_at_window_peak_pct': float(ratio.mean() * 100), 'mean_headroom_at_window_peak_pct': float((1 - ratio.mean()) * 100),
            'headroom_p05_at_window_peak_pct': float(np.quantile(1 - ratio, .05) * 100),
            'hours_above_95pct_of_peak': int(ratio.gt(.95).sum()), 'unit_count': units,
            'scenario_gpu_devices': units * config['gpu_devices_per_unit'],
            'peak_equivalent_gpu_devices': units * config['gpu_devices_per_unit'] * peak if key != 'A' else None,
            'observed_physical_free_gpu_count': None,
            'physical_dispatchable_capacity_status': 'unknown; hourly normalized resource proxy lacks device topology and verified permissions'})
    pd.DataFrame(native).to_json(args.output / 'native-capacity.json', orient='records', indent=2, force_ascii=False)
    tables = {}
    for name in ('runs-cards.csv', 'runs-fixed.csv'):
        path = args.results / name
        fingerprints[path.resolve().relative_to(project.resolve()).as_posix()] = digest(path)
        frame = pd.read_csv(path, dtype={'card_price': str})
        frame['target'] = frame.alpha.fillna(-1)
        keys = ['city', 'scenario', 'target']
        if name == 'runs-cards.csv':
            baseline = frame.loc[frame.flex.eq('none') & frame.status.eq('optimal')].set_index(keys)
            cases = frame.loc[frame.flex.eq('both') & frame.card_price.eq('forced0') & frame.status.eq('optimal')].set_index(keys)
        else:
            baseline = frame.loc[frame.card_price.eq('noflex') & frame.status.eq('optimal')].set_index(keys)
            cases = frame.loc[frame.card_price.eq('1.0') & frame.status.eq('optimal')].set_index(keys)
        if not baseline.index.is_unique or not cases.index.is_unique:
            raise ValueError('配对结果含重复键：' + name)
        paired = cases[['hours', 'extra_task_mw', 'moved_whole_mwh', 'moved_split_mwh', 'max_residual', 'peak_task_mw']].join(
            baseline.total_cost_usd.rename('base_cost_usd'), how='inner').join(cases.total_cost_usd.rename('flex_cost_usd'))
        paired['moved_mwh'] = paired.moved_whole_mwh + paired.moved_split_mwh
        paired['moved_mwh_per_observed_hour'] = paired.moved_mwh / paired.hours
        paired['saving_usd'] = paired.base_cost_usd - paired.flex_cost_usd
        paired['saving_pct_of_energy_and_extra_hardware'] = 100 * paired.saving_usd / paired.base_cost_usd
        rows = paired.reset_index()
        rows.to_json(args.output / ('planning-pairs.json' if name == 'runs-cards.csv' else 'fixed-supply-pairs.json'), orient='records', indent=2)
        tables[name] = {'total_rows': len(frame), 'status_counts': frame.status.value_counts().to_dict(),
            'valid_pairs': len(rows), 'pairs_with_positive_shift_and_no_addition': int((rows.moved_mwh.gt(1e-6) & rows.extra_task_mw.abs().le(1e-6)).sum()),
            'max_additional_task_power_mw': float(rows.extra_task_mw.max()), 'max_saved_residual': float(rows.max_residual.max())}
    (args.output / 'capacity-audit.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(),
        'scope': '11 scenarios, original windows; normalized allocation and aggregate-power scheduling kept distinct',
        'source_sha256': fingerprints, 'paired_results': tables,
        'hardware_resource_feasibility_verified': False,
        'unverified_constraints': ['individual GPU assignment', 'GPU memory and network topology', 'whole-job non-preemption',
            'CPU/GPU joint resource schedule', 'actual service deadlines and pause/resume permissions', 'checkpoint overhead'],
        'code_sha256': digest(Path(__file__))}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(pd.DataFrame(native)[['scenario', 'mean_at_window_peak_pct', 'mean_headroom_at_window_peak_pct', 'peak_fraction', 'unknown_hours']].to_string(index=False))
    print(json.dumps(tables, ensure_ascii=False))


if __name__ == '__main__':
    main()
