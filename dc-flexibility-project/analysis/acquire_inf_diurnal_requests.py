import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd

from summarize_inference_requests import ASSETS, digest, download, summarize_file


AZURE_COMMIT = '215becdacba1ce682c7368642ce97d5a332de7a6'


def main(args):
    args.raw.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    records, hourly = [], []
    for name, url, size, expected in ASSETS:
        if name not in ('burst-1', 'burst-2', 'azure-code', 'azure-conv'):
            continue
        path = args.raw / url.rsplit('/', 1)[-1]
        transfer = download(url, path, size, expected, 30, 2)
        frame, detail = summarize_file(name, path)
        frame['dataset'] = {'azure-code': 'azure-2024-code', 'azure-conv': 'azure-2024-conv'}.get(name, name)
        hourly.append(frame)
        records.append(dict(dataset=name, url=url, bytes=size, sha256=expected, **transfer, **detail))
        (args.output / 'request-input-audit.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
        print(name, detail['rows'], flush=True)
    for service in ('code', 'conv'):
        name = f'azure-2023-{service}'
        filename = f'AzureLLMInferenceTrace_{service}.csv'
        url = f'https://raw.githubusercontent.com/Azure/AzurePublicDataset/{AZURE_COMMIT}/data/{filename}'
        path = args.raw / filename
        if not path.exists():
            if shutil.disk_usage(args.raw).free < 32 * 1024**3:
                raise ValueError('共享卷剩余空间低于32GiB')
            subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error', '--max-filesize',
                            str(5 * 1024**3), '--output', str(path.with_suffix('.part')), url], check=True)
            path.with_suffix('.part').replace(path)
        frame, detail = summarize_file(name, path)
        frame['dataset'] = name
        hourly.append(frame)
        records.append(dict(dataset=name, url=url, upstream_commit=AZURE_COMMIT,
                            bytes=path.stat().st_size, sha256=digest(path), **detail))
        print(name, detail['rows'], flush=True)
    combined = pd.concat(hourly, ignore_index=True)
    for record, frame in zip(records, hourly):
        if int(frame.requests.sum()) != record['rows']:
            raise ValueError('小时请求数量不守恒')
    combined.to_csv(args.output / 'request-hourly-observed.csv', index=False)
    audit = {'created_utc': datetime.now(timezone.utc).isoformat(), 'inputs': records,
             'chunksize': 250000, 'license': 'CC BY 4.0; BurstGPT KDD2025, Azure Splitwise ISCA2024 / DynamoLLM HPCA2025',
             'raw_directory': str(args.raw), 'hourly_sha256': digest(args.output / 'request-hourly-observed.csv')}
    (args.output / 'request-input-audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
