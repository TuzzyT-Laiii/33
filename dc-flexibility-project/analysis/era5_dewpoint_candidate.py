import fcntl
import json
from pathlib import Path

import numpy as np
import pandas as pd

from acquire_buildability_protection import sha, utc_now, write_json
from era5_source_validation import overlap_check_passed


def differences(reference, values):
    valid = np.isfinite(reference) & np.isfinite(values)
    delta = np.abs(values[valid].astype(np.float64) - reference[valid].astype(np.float64))
    return {'count': int(values.size), 'finite_pair_count': int(valid.sum()),
            'candidate_nan_count': int(np.isnan(values).sum()), 'candidate_inf_count': int(np.isinf(values).sum()),
            'unequal_count': int(np.count_nonzero(delta)) + int((~valid).sum()),
            'max_abs_difference': float(delta.max()) if delta.size else None,
            'mean_abs_difference': float(delta.mean()) if delta.size else None,
            'quantiles_0_50_95_99_100': np.quantile(delta, [0, .5, .95, .99, 1]).tolist() if delta.size else [],
            'exact_equal': bool(delta.size and valid.all() and np.array_equal(reference, values))}


def publish_candidate(root, table, report):
    label = report['block']
    path = root / 'data/metadata/buildability-20261025/weather-cds-repairs' / (label + '.json')
    with (root / '.era5-dewpoint-proof.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            prior = json.loads(path.read_text())
            if prior['status'] == 'verified':
                if (prior['selection_manifest_sha256'] != report['selection_manifest_sha256']
                        or prior['source_metadata_sha256'] != report['source_metadata_sha256']
                        or prior['native_flat_indices'] != report['native_flat_indices']
                        or sha(Path(prior['output_path'])) != prior['output_sha256']):
                    raise ValueError('并行来源的已核验补取证明与当前输入不同')
                return prior
        if not all(overlap_check_passed(item) for item in report['checks'].values()):
            raise ValueError('补取来源未通过有效重合值绝对容差检查')
        for item in report['raw_sources']:
            if sha(Path(item['path'])) != item['sha256']:
                raise ValueError('候选补取原文SHA不一致')
        output = root / 'data/interim/era5-eligible-cds-repairs' / (label + '.parquet')
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.parquet.part')
        table.to_parquet(temporary, index=False, compression='zstd')
        pd.testing.assert_frame_equal(table, pd.read_parquet(temporary))
        temporary.replace(output)
        report.update(status='verified', verified_utc=utc_now(), output_path=str(output),
                      output_sha256=sha(output), output_bytes=output.stat().st_size)
        write_json(path, report)
        return report
