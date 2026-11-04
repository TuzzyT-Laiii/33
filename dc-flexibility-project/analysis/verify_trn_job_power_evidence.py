import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from compute_load.model import sha256


def main(args):
    project = args.project.resolve()
    output = project / 'results/trn-power-model-20261101/job-power-evidence'
    seren = pd.read_csv(output / 'seren/seren-hourly.csv')
    for metric, count, column in [('POWER_USAGE', 'power_device_samples', 'mean_gpu_power_w'),
                                  ('GPU_UTIL', 'util_device_samples', 'mean_gpu_util_pct')]:
        old = json.loads((project / f'results/trn-power-model-20261101/acme/seren-{metric}-summary.json').read_text('utf-8'))
        assert int(seren[count].sum()) == old['valid_cells']
        np.testing.assert_allclose(np.average(seren[column], weights=seren[count]), old['mean'], rtol=1e-12)
    type_columns = [c for c in seren if c.startswith('allocated_') and c != 'allocated_gpu_count']
    np.testing.assert_allclose(seren[type_columns].sum(axis=1), seren.allocated_gpu_count, atol=1e-7)
    for idle in (50., 60., 70.):
        alloc = seren.sample_matched_allocated_gpus
        expected = (seren.overlap_estimated_power_w - (2288 - alloc) * idle) / alloc / 400
        np.testing.assert_allclose(expected, seren[f'inverse_ratio_idle_{idle:.3f}'], rtol=1e-12)
    input_audit = json.loads((output / 'mit/mit-input-audit.json').read_text('utf-8'))
    immutable_checked = 0
    for batch, digest in input_audit['batch_file_table_sha256'].items():
        assert sha256(args.mit_root / 'batches' / batch / 'files.parquet') == digest
        immutable_checked += 1
    assert sha256(args.mit_root / 'reference/labelled_jobids.csv') == input_audit['source_label_sha256']
    inventory = pd.read_parquet(args.private / 'mit-file-snapshot.parquet')
    assert not inventory.key.duplicated().any()
    missing = pd.read_parquet(args.private / 'missing-labelled-files.parquet')
    assert len(missing) == input_audit['missing_labelled_files']
    assert int(missing.bytes.sum()) == input_audit['missing_labelled_bytes'] > 30_000_000_000
    jobs = pd.read_parquet(args.private / 'mit-job-power.parquet')
    assert jobs.id_job.nunique() == len(jobs)
    summary = pd.read_csv(output / 'mit/mit-cohort-summary.csv').set_index('cohort')
    for cohort, mask in [('all_observed_V100', jobs.id_job.notna()), ('official_DNN', jobs.model.notna()),
                         ('NLP_finetuning', jobs.family.eq('NLP_finetuning'))]:
        subset = jobs.loc[mask]
        assert len(subset) == summary.loc[cohort, 'jobs']
        np.testing.assert_allclose(np.average(subset.power_mean_w, weights=subset.gpu_hours) / 250,
            summary.loc[cohort, 'gpu_time_weighted_mean_ratio'], rtol=1e-12)
    h100 = pd.read_csv(output / 'h100-experiments/h100-llm-session-summary.csv')
    progress = pd.read_csv(output / 'h100-experiments/h100-llm-session-progress.csv')
    assert len(h100) == 9 and h100.source_sha256.nunique() == 9
    for experiment, part in progress.groupby('experiment'):
        row = h100.set_index('experiment').loc[experiment]
        np.testing.assert_allclose(np.average(part.mean_power_tdp_ratio, weights=part.duration_s), row.mean_gpu_tdp_ratio, rtol=1e-12)
        np.testing.assert_allclose(part.duration_s.sum(), row.recorded_seconds, rtol=1e-12)
    acquisition = json.loads((output / 'literature-acquisition.json').read_text('utf-8'))
    for record in acquisition:
        assert record['status'] == 'acquired', record['file']
        assert sha256(args.literature / record['file']) == record['sha256']
    proposal = pd.read_csv(output / 'proposal-comparison.csv')
    old = pd.read_csv(project / 'results/trn-power-model-20261101/diurnal-acme/window-summary.csv').query('case == "shape_central"').set_index('scenario_id')
    current = proposal.query('case == "current_080_Kalos"').set_index('scenario_id')
    assert proposal.scenario_id.nunique() == 12 and len(proposal) == 72
    for column in ('mean_it_mw', 'peak_it_mw', 'mean_facility_mw', 'peak_facility_mw', 'it_diurnal_peak_valley', 'it_weekly_peak_valley'):
        np.testing.assert_allclose(current[column].sort_index(), old[column].sort_index(), rtol=1e-12)
    for scenario, rows in proposal.groupby('scenario_id'):
        rows = rows.set_index('case')
        central = rows.loc[['current_080_Kalos', 'recommended_080_constant', 'sensitivity_080_Seren']]
        np.testing.assert_allclose(central.mean_it_mw, central.mean_it_mw.iloc[0], rtol=1e-12)
        assert rows.loc['small_batch_040_constant', 'mean_it_mw'] < rows.loc['sensitivity_065_constant', 'mean_it_mw'] < rows.loc['recommended_080_constant', 'mean_it_mw'] < rows.loc['sensitivity_095_constant', 'mean_it_mw']
    subprocess.run(['git', 'diff', '--exit-code', '1594fb7', '--', 'analysis/trn_power_models.py',
        'results/trn-power-model-20261101/diurnal-acme'], cwd=project, check=True, capture_output=True)
    register = pd.read_csv(output / 'public-source-register.csv')
    assert len(register) == 12 and not register.isna().any().any()
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'seren_sample_count_and_mean_conservation': True,
        'seren_allocation_type_conservation_and_inverse_formula': True, 'mit_immutable_batch_metadata_files': immutable_checked,
        'mit_unique_job_and_file_joins': True, 'mit_aggregate_means_independently_recomputed': True,
        'missing_mit_download_over_threshold': True, 'h100_independent_sessions': 9,
        'h100_progress_energy_conservation': True, 'literature_hashed_sources': len(acquisition),
        'proposal_scenarios': 12, 'proposal_cases': 6, 'current_Kalos_values_match_previous_delivery': True,
        'existing_body_results_and_power_model_unchanged': True, 'public_source_register_rows': 12,
        'software': {'numpy': np.__version__, 'pandas': pd.__version__},
        'scope': 'Real source aggregates and immutable batch tables; no synthetic inputs; no MIT source or STOP writes'}
    (output / 'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--mit-root', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--literature', type=Path, required=True)
    main(parser.parse_args())
