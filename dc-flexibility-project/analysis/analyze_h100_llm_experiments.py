import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW


def main(args):
    rows, profiles, aliases = [], [], []
    seen = {}
    with zipfile.ZipFile(args.archive) as archive:
        names = [name for name in archive.namelist() if '/Text Generation LLMs/H100/' in name and name.endswith('.csv')]
        readme_name = next(name for name in archive.namelist() if name.endswith('/README.md'))
        readme = archive.read(readme_name).decode('utf-8-sig')
        license_name = next(name for name in archive.namelist() if name.endswith('/LICENSE'))
        license_text = archive.read(license_name).decode('utf-8-sig')
        print('ARCHIVE_LICENSE', license_text.splitlines()[:3], flush=True)
        for name in names:
            with archive.open(name) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest in seen:
                aliases.append({'archive_path': name, 'identical_to': seen[digest], 'sha256': digest})
                continue
            seen[digest] = name
            with archive.open(name) as stream:
                frame = pd.read_csv(stream)
            time = pd.to_datetime(frame.timestamp, errors='raise')
            seconds = (time - time.iloc[0]).dt.total_seconds().to_numpy()
            dt = np.r_[np.diff(seconds), 0.]
            if np.any(dt < 0) or seconds[-1] <= 0:
                raise ValueError('H100实验时间未递增或窗口为空')
            power = frame[[f'gpu{i}_power_W' for i in range(8)]].to_numpy(float)
            published = frame[[f'gpu{i}_Power_TDP' for i in range(8)]].to_numpy(float)
            util = frame[[f'gpu{i}_utilization_percent' for i in range(8)]].to_numpy(float)
            if not np.isfinite(power).all() or np.any(power < 0) or not np.isfinite(util).all():
                raise ValueError('H100实验缺少有效GPU功率或利用率')
            percentage_difference = np.abs(power / 700 * 100 - published)
            weights = np.repeat(dt, 8)
            r = power.ravel() / 700
            quantiles = DescrStatsW(r, weights=weights).quantile([.1, .5, .9], return_pandas=False)
            mean = float(np.average(power.mean(axis=1), weights=dt))
            variance = np.average((power - mean) ** 2, weights=np.broadcast_to(dt[:, None], power.shape))
            name_base = Path(name).name
            model_b = int(name_base.split('_Lamma')[1].split('B_')[0])
            rows.append({'experiment': name_base, 'model_billion_parameters': model_b, 'gpu_model': 'H100 SXM 80GB',
                'gpus': 8, 'tdp_w': 700, 'samples': len(frame), 'start_local_timezone_unknown': str(time.iloc[0]),
                'end_local_timezone_unknown': str(time.iloc[-1]), 'recorded_seconds': float(seconds[-1]),
                'median_poll_interval_s': float(np.median(dt[:-1])), 'gaps_over1s': int((dt > 1).sum()),
                'mean_power_w_per_gpu': mean, 'mean_gpu_tdp_ratio': mean / 700,
                'p10_device_time_ratio': float(quantiles[0]), 'median_device_time_ratio': float(quantiles[1]),
                'p90_device_time_ratio': float(quantiles[2]), 'within_session_cv': float(np.sqrt(variance) / mean),
                'mean_gpu_util_pct': float(np.average(util.mean(axis=1), weights=dt)),
                'above_tdp_device_time_fraction': float(np.average(power > 700, weights=np.broadcast_to(dt[:, None], power.shape))),
                'published_percentage_mismatch_fraction': float(np.mean(percentage_difference > 1e-6)),
                'maximum_percentage_difference_points': float(percentage_difference.max()),
                'source_sha256': digest})
            progress = np.minimum(19, np.floor(seconds / seconds[-1] * 20).astype(int))
            for index in range(20):
                mask = (progress == index) & (dt > 0)
                if mask.any():
                    profiles.append({'experiment': name_base, 'progress_bin': index, 'duration_s': float(dt[mask].sum()),
                        'mean_power_tdp_ratio': float(np.average(power[mask].mean(axis=1), weights=dt[mask]) / 700)})
    args.output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.output / 'h100-llm-session-summary.csv', index=False)
    pd.DataFrame(profiles).to_csv(args.output / 'h100-llm-session-progress.csv', index=False)
    summary = {'h100_llm_csv_paths': len(names), 'unique_sessions': len(rows), 'duplicate_copies': len(aliases),
        'gpus_per_session': 8, 'reference_tdp_w': 700,
        'percentage_field_status': 'GPU Power_TDP and power_W are not identical at every sample; use power_W / 700 and report mismatch without rewriting either field',
        'recorded_total_seconds': float(table.recorded_seconds.sum()),
        'gpu_time_weighted_mean_ratio': float(np.average(table.mean_gpu_tdp_ratio, weights=table.recorded_seconds)),
        'equal_session_mean_ratio': float(table.mean_gpu_tdp_ratio.mean()), 'minimum_session_mean_ratio': float(table.mean_gpu_tdp_ratio.min()),
        'maximum_session_mean_ratio': float(table.mean_gpu_tdp_ratio.max()),
        'scope': 'Recorded LLaMA 1B/3B/8B training experiment windows on one 8-H100-SXM node; not whole-lifecycle production pretraining',
        'training_stage': 'Pretraining versus SFT is not declared in the supplied CSV or README; LLaMA-Factory framework is identified in the paper',
        'diurnal_scope': 'Independent short experiment windows with no verified timezone; no 24-hour or weekly shape inferred',
        'no_gap_holding_beyond_last_sample': True, 'duplicate_files': aliases,
        'readme_sha256': hashlib.sha256(readme.encode('utf-8')).hexdigest(),
        'archive_license_first_line': license_text.splitlines()[0]}
    (args.output / 'h100-llm-experiment-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'duplicate_files'}, ensure_ascii=False), flush=True)
    print(table[['experiment', 'recorded_seconds', 'mean_gpu_tdp_ratio', 'mean_gpu_util_pct']].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
