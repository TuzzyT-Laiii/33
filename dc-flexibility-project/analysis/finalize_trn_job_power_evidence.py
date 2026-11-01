import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from compute_load.model import sha256


def main(args):
    project = args.project.resolve()
    output = project / 'results/trn-power-model-20261101/job-power-evidence'
    raw = project / 'data/raw/trn-job-power-evidence-20261101/literature'
    nodes = {}
    for kind in ('ac', 'dc'):
        entries = json.loads((raw / f'mlperf-{kind}-64nodes-3.json').read_text('utf-8'))
        nodes[kind] = {'node_log_files': sum(x['name'].startswith('node_') and x['name'].endswith('.txt') for x in entries),
            'text_bytes_one_run': sum(x['size'] for x in entries if x['name'].endswith('.txt')),
            'directory_metadata_sha256': sha256(raw / f'mlperf-{kind}-64nodes-3.json')}
    extra = {'mlperf_inspected_run': 'result_5848-240517221859825376616_2', 'node_power_scope': 'whole-node AC/DC; not GPU-only',
        'sample_cadence_seconds': 2.005, 'run_total_energy_text_kwh': {'ac': 415.22, 'dc': 408.44},
        'network_power_estimate_w': 59570, 'directory_inventory': nodes,
        'sample_acquisition': json.loads((raw / 'mlperf-sample-acquisition.json').read_text('utf-8')),
        'raw_64node_data_downloaded': False, 'sample_node_logs_downloaded': True}
    (output / 'mlperf-artifact-audit.json').write_text(json.dumps(extra, ensure_ascii=False, indent=2), encoding='utf-8')
    files = pd.read_parquet(project / 'data/processed/trn-job-power-evidence-20261101/mit/mit-file-snapshot.parquet')
    files = files.loc[files.eligibility.eq('eligible') & files.gpu_type.eq('gpu:volta')]
    windows = {}
    for name, subset in [('all_observed_V100', files), ('official_DNN', files.loc[files.model.notna()])]:
        windows[name] = {'first_file_sample_utc': pd.to_datetime(subset.time_min_us.min(), unit='us', utc=True).isoformat(),
            'last_file_sample_utc': pd.to_datetime(subset.time_max_us.max(), unit='us', utc=True).isoformat(),
            'interpretation': 'Corrected file timestamp bounds, not continuous coverage; allocation-external samples excluded from statistics.'}
    (output / 'mit/mit-observation-windows.json').write_text(json.dumps(windows, ensure_ascii=False, indent=2), encoding='utf-8')
    code_names = ['audit_seren_trn_power.py', 'summarize_seren_trn_power.py', 'audit_mit_training_power.py',
        'summarize_mit_training_variation.py', 'acquire_trn_job_literature.py', 'acquire_llm_training_dataset.py',
        'analyze_h100_llm_experiments.py', 'compare_trn_evidence_proposals.py', 'verify_trn_job_power_evidence.py',
        'plot_trn_job_power_evidence.py', 'finalize_trn_job_power_evidence.py', 'requirements-trn-job-power-evidence.txt']
    validation = json.loads((output / 'validation.json').read_text('utf-8'))
    qa = json.loads((output / 'figure-qa.json').read_text('utf-8'))
    if not validation['existing_body_results_and_power_model_unchanged'] or len(qa) != 2:
        raise ValueError('数据或图件检查未完成')
    status = {'utc': datetime.now(timezone.utc).isoformat(), 'input_commit': '1594fb77148f3c9307d00230c25e864f6a286625',
        'same_pr': 36, 'branch': 'codex/trn-power-model-20261101',
        'completed_steps': ['Seren overlap coverage and cluster inversion', 'MIT official DNN matched telemetry and cost audit',
            'mandatory public source verification', 'additional matched H100 experiment analysis', 'report-only proposal comparison', 'validation and Chinese figures'],
        'body_results_replaced': False, 'MIT_raw_download_bytes_this_task': 0,
        'MIT_source_writes': False, 'MIT_process_or_STOP_actions': False,
        'new_H100_archive_bytes': 209685111, 'new_H100_archive_location': 'local ignored data/raw/trn-job-power-evidence-20261101/llm-experiments',
        'scientific_limits': ['Seren population and online-state assumptions', 'no Seren per-type power mapping',
            'MIT CNN-dominated and partial official-label coverage', 'H100 experiments are short windows with training stage unspecified',
            'H100 dataset license metadata differs from archive license', 'fixed natural-day TRN cycle not identified'],
        'code_sha256': {name: sha256(project / 'analysis' / name) for name in code_names},
        'baseline_result_sha256': {path.relative_to(project).as_posix(): sha256(path)
            for path in sorted((project / 'results/trn-power-model-20261101/diurnal-acme').rglob('*')) if path.is_file()}}
    (output / 'task-status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    artifacts = {path.relative_to(output).as_posix(): sha256(path) for path in sorted(output.rglob('*'))
        if path.is_file() and path.name not in ('artifact-manifest.json', 'publication.json')}
    (output / 'artifact-manifest.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(),
        'artifact_sha256': artifacts, 'raw_monitoring_or_job_device_rows_published': False}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Summary artifacts', len(artifacts), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    main(parser.parse_args())
