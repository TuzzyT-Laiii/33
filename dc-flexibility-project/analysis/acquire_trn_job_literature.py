import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from bs4 import BeautifulSoup
import fitz
import requests

from compute_load.model import sha256


SOURCES = {
    'pm100-metadata.json': 'https://zenodo.org/api/records/10127767',
    'pm100.html': 'https://zenodo.org/records/10127767',
    'pm100-readme.md': 'https://raw.githubusercontent.com/francescoantici/PM100-data/main/README.md',
    'pm100-fields.md': 'https://raw.githubusercontent.com/francescoantici/PM100-data/main/documentation/job_features.md',
    'pm100-paper.pdf': 'https://cris.unibo.it/retrieve/6fbbd46f-a7e7-4469-b1ab-8adbe6de6d98/3624062.3624263.pdf',
    'm100.xml': 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10193329/fullTextXML',
    'pai-readme.md': 'https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/README.md',
    'pai-license.txt': 'https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/LICENSE',
    'philly-readme.md': 'https://raw.githubusercontent.com/msr-fiddle/philly-traces/master/README.md',
    'philly-license.txt': 'https://raw.githubusercontent.com/msr-fiddle/philly-traces/master/LICENSE',
    'perlmutter-pearc25.pdf': 'https://www.bu.edu/peaclab/files/2025/07/GPU_Utilization_Analysis_PEARC25.pdf',
    'perlmutter-longitudinal.pdf': 'https://arxiv.org/pdf/2502.18680',
    'nersc-power-capping.html': 'https://docs.nersc.gov/jobs/power-capping/',
    'mlperf-training.html': 'https://mlcommons.org/benchmarks/training/',
    'mlperf-v40-root.json': 'https://api.github.com/repos/mlcommons/training_results_v4.0/contents',
    'mlperf-v40-smc.json': 'https://api.github.com/repos/mlcommons/training_results_v4.0/contents/Supermicro/results',
    'mlperf-ac-results.json': 'https://api.github.com/repos/mlcommons/training_results_v4.0/contents/smc-ac-power/results',
    'mlperf-dc-results.json': 'https://api.github.com/repos/mlcommons/training_results_v4.0/contents/smc-dc-power/results',
    'mlperf-ac-64nodes-3.json': 'https://api.github.com/repos/mlcommons/training_results_v4.0/contents/smc-ac-power/results/64xSMC-H100-SXM-80GB/gpt3/power/result_5848-240517221859825376616_2',
    'mlperf-dc-64nodes-3.json': 'https://api.github.com/repos/mlcommons/training_results_v4.0/contents/smc-dc-power/results/64xSMC-H100-SXM-80GB/gpt3/power/result_5848-240517221859825376616_2',
    'mlperf-ac-node_1.txt': 'https://raw.githubusercontent.com/mlcommons/training_results_v4.0/main/smc-ac-power/results/64xSMC-H100-SXM-80GB/gpt3/power/result_5848-240517221859825376616_2/node_1.txt',
    'mlperf-ac-total_power.txt': 'https://raw.githubusercontent.com/mlcommons/training_results_v4.0/main/smc-ac-power/results/64xSMC-H100-SXM-80GB/gpt3/power/result_5848-240517221859825376616_2/total_power.txt',
    'mlperf-ac-sw_0.txt': 'https://raw.githubusercontent.com/mlcommons/training_results_v4.0/main/smc-ac-power/results/64xSMC-H100-SXM-80GB/gpt3/power/result_5848-240517221859825376616_2/sw_0.txt',
    'mlperf-dc-node_1.txt': 'https://raw.githubusercontent.com/mlcommons/training_results_v4.0/main/smc-dc-power/results/64xSMC-H100-SXM-80GB/gpt3/power/result_5848-240517221859825376616_2/node_1.txt',
    'mlperf-dc-total_power.txt': 'https://raw.githubusercontent.com/mlcommons/training_results_v4.0/main/smc-dc-power/results/64xSMC-H100-SXM-80GB/gpt3/power/result_5848-240517221859825376616_2/total_power.txt',
    'mlperf-dc-sw_0.txt': 'https://raw.githubusercontent.com/mlcommons/training_results_v4.0/main/smc-dc-power/results/64xSMC-H100-SXM-80GB/gpt3/power/result_5848-240517221859825376616_2/sw_0.txt',
    'mlperf-license.md': 'https://raw.githubusercontent.com/mlcommons/training_results_v4.0/main/LICENSE.md',
    'mlperf-power-rules.adoc': 'https://raw.githubusercontent.com/mlcommons/training_policies/master/training_rules.adoc',
    'smc-gpt3.html': 'https://smc.co/mlperfv40/results/gpt3',
    'olmo.html': 'https://arxiv.org/html/2503.05804v1',
    'llama3.html': 'https://arxiv.org/html/2407.21783v3',
    'bloom.pdf': 'https://arxiv.org/pdf/2211.02001',
    'llm-highres.xml': 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13538459/fullTextXML',
    'llm-highres-figshare.json': 'https://api.figshare.com/v2/articles/31654879',
    'mit-readme.md': 'https://mit-supercloud-dataset.s3.us-west-2.amazonaws.com/datacenter-challenge/202201/README.md',
    'mit-license.txt': 'https://mit-supercloud-dataset.s3.us-west-2.amazonaws.com/datacenter-challenge/202201/LICENSE',
}


def main(args):
    raw, output = args.raw.resolve(), args.output.resolve()
    raw.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
    token = subprocess.run(['gh', 'auth', 'token'], capture_output=True, text=True, check=True).stdout.strip()
    def acquire(item):
        name, url = item
        path = raw / name
        record = {'file': name, 'url': url, 'utc': datetime.now(timezone.utc).isoformat()}
        if not path.exists():
            session = requests.Session()
            session.trust_env = False
            session.headers['User-Agent'] = 'Mozilla/5.0'
            headers = {'Authorization': 'Bearer ' + token} if url.startswith('https://api.github.com/') else {}
            try:
                response = session.get(url, headers=headers, timeout=(15, 40))
            except requests.RequestException as error:
                return {**record, 'status': 'access_failure', 'error': str(error)}
            if response.status_code != 200:
                return {**record, 'status': 'http_error', 'http_status': response.status_code}
            path.write_bytes(response.content)
        record.update(status='acquired', bytes=path.stat().st_size, sha256=sha256(path))
        if name.endswith('.html'):
            soup = BeautifulSoup(path.read_text('utf-8'), 'lxml')
            for element in soup(['script', 'style', 'nav']):
                element.decompose()
            (raw / (name + '.txt')).write_text(soup.get_text('\n', strip=True), encoding='utf-8')
        elif name.endswith('.pdf'):
            with fitz.open(path) as document:
                (raw / (name + '.txt')).write_text('\n'.join(page.get_text() for page in document), encoding='utf-8')
        return record
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(acquire, SOURCES.items()))
    (output / 'literature-acquisition.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    samples = []
    for row in records:
        if row['file'].startswith(('mlperf-ac-', 'mlperf-dc-')) and row['file'].endswith('.txt') and row['status'] == 'acquired':
            kind = row['file'].split('-')[1]
            samples.append({'kind': kind, 'file': row['file'].removeprefix(f'mlperf-{kind}-'),
                'url': row['url'], 'bytes': row['bytes'], 'sha256': row['sha256']})
    (raw / 'mlperf-sample-acquisition.json').write_text(json.dumps(samples, indent=2), encoding='utf-8')
    for row in records:
        print(row['file'], row['status'], row.get('bytes', row.get('http_status', row.get('error'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
