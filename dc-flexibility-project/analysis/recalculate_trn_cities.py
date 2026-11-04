import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from build_trn_power_model import BASE, RUN, calculate, summary
from compute_load.model import sha256
from load_curve_scenarios_v3 import catalogue, identity
from trn_power_models import idle_power


def main(args):
    project, source = args.project.resolve(), args.source.resolve()
    output = project / 'results' / RUN
    if args.finalize_only:
        record = json.loads((output / 'manifest.json').read_text('utf-8'))
        record['hourly_sha256'] = {path.name: sha256(path) for path in sorted((output / 'hourly').glob('*.parquet'))}
        record['canonical_hourly_serialization'] = 'server city recalculation; scientific arrays checked independently against reference summaries'
        record['finalization_environment'] = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__}
        record['all_locations_manifest_sha256'] = sha256(output / 'all-locations-manifest.json')
        record['scripts']['recalculate_trn_cities.py'] = sha256(Path(__file__))
        (output / 'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        return
    baseline = source / 'results' / BASE
    manifest = json.loads((baseline / 'manifest.json').read_text('utf-8'))
    ratios = json.loads((project / 'results/typical-load-curves-20261029-v2/mit-power-ratio-summary.json').read_text('utf-8'))['transfer_distribution']
    mit = json.loads((baseline / 'mit-calibration.json').read_text('utf-8'))
    shape = json.loads((output / 'cpu-shape.json').read_text('utf-8'))
    configs = catalogue(project)
    records, before_records, hashes = [], [], {}
    total_hours = 0
    for name, digest in manifest['hourly_sha256'].items():
        path = source / name
        if sha256(path) != digest:
            raise ValueError('v3 城市小时输入校验失败：' + name)
        old = pd.read_parquet(path)
        key, use = old.server_type.iloc[0], old.use.iloc[0]
        scale = int(old.reference_it_mw.iloc[0])
        config = configs[key]
        current = calculate(old, config, key, use, ratios, mit, shape)
        fields = {**identity(key, use, scale), 'location_id': old.location_id.iloc[0], 'year': int(pd.DatetimeIndex(old.time_utc)[0].year)}
        old['idle_it_mw'] = idle_power(config, key, use, mit) * old.unit_count.iloc[0] / 1e6
        old['task_it_mw'] = old.it_mw - old.idle_it_mw
        records.append(summary(current, fields))
        before_records.append(summary(old, fields))
        if name.startswith('results/'):
            target = output / 'hourly' / path.name
        else:
            target = project / name.replace(BASE, RUN)
        target.parent.mkdir(parents=True, exist_ok=True)
        current.to_parquet(target, index=False)
        hashes[str(target.relative_to(project))] = sha256(target)
        total_hours += len(current)
        if len(records) % 44 == 0:
            print(f'{len(records)}/1100 scenarios; {total_hours} hours', flush=True)
    if len(records) != 1100 or total_hours != 9644448:
        raise ValueError('地点、年份或小时数不守恒')
    pd.DataFrame(records).to_csv(output / 'annual-summary-all-locations.csv', index=False)
    pd.DataFrame(before_records).to_csv(output / 'v3-summary-all-locations.csv', index=False)
    save = {'utc': datetime.now(timezone.utc).isoformat(), 'hours': total_hours, 'scenarios': len(records),
            'source_manifest_sha256': sha256(baseline / 'manifest.json'), 'hourly_sha256': hashes,
            'scope': 'reference grid 2025 and eight cities 2023-2025, frozen v3 hourly inputs'}
    (output / 'all-locations-manifest.json').write_text(json.dumps(save, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--finalize-only', action='store_true')
    main(parser.parse_args())
