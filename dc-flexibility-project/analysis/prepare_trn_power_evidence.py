import argparse
import json
from pathlib import Path

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from acquire_trn_power_evidence import download, save_json
import requests


SOURCES = {
    'spec-01377': 'https://www.spec.org/power_ssj2008/results/res2024q1/power_ssj2008-20240227-01377.html',
    'spec-01376': 'https://www.spec.org/power_ssj2008/results/res2024q1/power_ssj2008-20240227-01376.html',
    'spec-01282': 'https://www.spec.org/power_ssj2008/results/res2023q3/power_ssj2008-20230619-01282.html',
    'olmo-2025': 'https://arxiv.org/html/2503.05804v1',
    'h100-node-2024': 'https://arxiv.org/html/2412.08602v2',
    'llama3': 'https://arxiv.org/html/2407.21783v3',
    'mlperf-smc-gpt3': 'https://smc.co/mlperfv40/results/gpt3',
    'dgx-h100': 'https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html',
    'microsoft-2024': 'https://www.microsoft.com/en-us/research/uploads/prod/2024/03/GPU_Power_ASPLOS_24.pdf',
    'acme-nsdi24': 'https://www.usenix.org/system/files/nsdi24-hu.pdf',
    'doe-2024': 'https://www.energy.gov/sites/default/files/2024-07/best-practice-guide-data-center-design_0.pdf',
}


def spec_points(path):
    soup = BeautifulSoup(path.read_bytes(), 'xml')
    records = []
    table = next(table for table in soup.find_all('table') if
                 'Average Active Power (W)' in table.get_text() and 'Actual Load' in table.get_text())
    for tr in table.find_all('tr'):
        cells = [cell.get_text(' ', strip=True) for cell in tr.find_all(['td', 'th'], recursive=False)]
        if cells and cells[0] in [f'{i * 10}%' for i in range(1, 11)]:
            if len(cells) != 5:
                continue
            records.append({'load': int(cells[0][:-1]) / 100, 'whole_system_w': float(cells[3])})
        elif cells and cells[0] == 'Active Idle' and len(cells) == 4:
            records.append({'load': 0., 'whole_system_w': float(cells[2])})
    frame = pd.DataFrame(records).drop_duplicates().sort_values('load')
    np.testing.assert_array_equal(frame.load, np.arange(11) / 10)
    return frame


def bernstein(u, coefficients):
    return 3 * coefficients[0] * u * (1 - u) ** 2 + 3 * coefficients[1] * u ** 2 * (1 - u) + u ** 3


def main(args):
    raw, output = args.raw.resolve(), args.output.resolve()
    raw.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.proxies.update({'https': args.proxy, 'http': args.proxy})
    manifest, records, shapes = [], [], []
    for key, url in SOURCES.items():
        path = raw / (key + ('.pdf' if url.endswith('.pdf') else '.html'))
        item = download(session, url, path)
        manifest.append({'source_id': key, **item})
        save_json(output / 'literature-acquisition.json', manifest)
        if key.startswith('spec-'):
            frame = spec_points(path)
            frame['normalized_dynamic'] = (frame.whole_system_w - frame.whole_system_w.iloc[0]) / (frame.whole_system_w.iloc[-1] - frame.whole_system_w.iloc[0])
            frame['source_id'], frame['source_url'] = key, url
            frame['cpu_packages'] = 1 if key == 'spec-01377' else 2
            frame['cpu_model'] = 'Xeon Platinum 8490H' if key == 'spec-01282' else 'Xeon Gold 6448Y'
            records.append(frame)
            shapes.append(frame.normalized_dynamic.to_numpy())
        print(key, item['bytes'], flush=True)
    pd.concat(records).to_csv(output / 'specpower-evidence.csv', index=False)
    u = np.arange(11) / 10
    mean = np.mean(shapes, axis=0)
    fit = minimize(lambda c: np.sum((bernstein(u, c) - mean) ** 2), [.3, .6],
                   bounds=[(0, 1), (0, 1)], constraints=[{'type': 'ineq', 'fun': lambda c: c[1] - c[0]}],
                   method='SLSQP', options={'ftol': 1e-14, 'maxiter': 1000})
    if not fit.success:
        raise ValueError(f'CPU 平滑拟合失败：{fit.message}')
    save_json(output / 'cpu-shape.json', {'method': 'equal-weight mean of three normalized SPEC curves; monotone cubic Bernstein least squares, endpoints 0 and 1',
              'source_ids': [key for key in SOURCES if key.startswith('spec-')], 'coefficients': fit.x.tolist(),
              'rmse_to_mean': float(np.sqrt(fit.fun / len(u))), 'load': u.tolist(), 'mean_shape': mean.tolist(),
              'fitted_shape': bernstein(u, fit.x).tolist(), 'scope': 'whole-system response transferred to dynamic CPU proxy; not package metering; not independent hardware repeats'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--proxy', default='http://127.0.0.1:7897')
    main(parser.parse_args())
