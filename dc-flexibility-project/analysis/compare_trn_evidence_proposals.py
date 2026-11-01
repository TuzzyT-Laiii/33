import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_trn_diurnal import metrics, recalculate
from compute_load.model import sha256
from load_curve_scenarios_v3 import catalogue


def main(args):
    project = args.project.resolve()
    output = project / 'results/trn-power-model-20261101/job-power-evidence'
    native = project / 'results/typical-load-curves-20261101-v3-native'
    reference = pd.read_parquet(native / 'source-hourly/D-TRN-10MW.parquet')
    weights = reference.groupby('hour_of_source_day').allocated_gpu_fraction.sum().reindex(range(24)).to_numpy()
    seren = pd.read_csv(output / 'seren/seren-diurnal.csv').query('case == "coverage_qualified_all_ratios"').sort_values('hour_utc')
    current = pd.read_csv(project / 'results/trn-power-model-20261101/diurnal-acme/transfer-profile.csv')
    relative = seren.gpu_hour_weighted_mean_ratio.to_numpy()
    relative /= np.average(relative, weights=weights)
    profiles = pd.DataFrame({'hour_utc': np.arange(24), 'current_080_Kalos': current.shape_central,
        'recommended_080_constant': .8, 'sensitivity_080_Seren': .8 * relative,
        'sensitivity_065_constant': .65, 'sensitivity_095_constant': .95, 'small_batch_040_constant': .4})
    profiles.to_csv(output / 'proposal-profiles.csv', index=False)
    configs = catalogue(project)
    ratios = json.loads((project / 'results/typical-load-curves-20261029-v2/mit-power-ratio-summary.json').read_text('utf-8'))['transfer_distribution']
    mit = json.loads((project / 'results/typical-load-curves-20261101-v3/mit-calibration.json').read_text('utf-8'))
    cpu = json.loads((project / 'results/trn-power-model-20261101/cpu-shape.json').read_text('utf-8'))
    manifest = json.loads((native / 'manifest.json').read_text('utf-8'))
    records, days, sources = [], [], {}
    for path in sorted((native / 'hourly').glob('*-TRN-*.parquet')):
        relative_path = str(path.relative_to(project)).replace('\\', '/')
        if sha256(path) != manifest['hourly_sha256'][relative_path]:
            raise ValueError('原生输入校验不符')
        source = pd.read_parquet(path)
        if len(source) != 2160:
            raise ValueError('比较须保留原生90天窗口')
        sources[relative_path] = sha256(path)
        for case in profiles.columns[1:]:
            candidate = recalculate(source, configs[source.server_type.iloc[0]], ratios, mit, cpu,
                profiles[case].to_numpy()[source.hour_of_source_day.to_numpy(int)])
            records.append(metrics(candidate, case))
            if source.reference_it_mw.iloc[0] == 10:
                day = candidate.groupby('hour_of_source_day')[['it_mw', 'facility_mw']].mean().reset_index()
                days.append(day.assign(scenario_id=source.scenario_id.iloc[0], case=case))
    pd.DataFrame(records).to_csv(output / 'proposal-comparison.csv', index=False)
    pd.concat(days).to_csv(output / 'proposal-typical-day.csv', index=False)
    (output / 'proposal-audit.json').write_text(json.dumps({'input_native_sha256': sources,
        'source_version': 'PR36@1594fb77148f3c9307d00230c25e864f6a286625',
        'cases': profiles.columns[1:].tolist(), 'reference_scenarios': 12, 'native_hours_each': 2160,
        'baseline_results_replaced': False, 'new_hourly_body_outputs_written': False,
        'scope': 'Report-only comparison; reference-grid weather 2025; CPU, idle, cooling and demand frozen',
        'methods': 'Current Kalos versus constant 0.80 and Seren shape; low/high levels and small-batch endpoint are engineering sensitivities.'}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(pd.DataFrame(records).query('reference_it_mw == 10')[['scenario_id', 'case', 'mean_it_mw', 'peak_it_mw', 'it_diurnal_peak_valley']].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    main(parser.parse_args())
