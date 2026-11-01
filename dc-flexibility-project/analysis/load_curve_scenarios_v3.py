import json
from pathlib import Path

import numpy as np

from typical_power_models import component_power, reference_power
from typical_power_models_v2 import power_v2


RUN = 'typical-load-curves-20261101-v3'
SCALES = (1, 10, 100, 500)
# 历史编号仅用于输入对照，v3 文件名与科学情境字段采用确认命名。
LEGACY = {'A': 'C20', 'B': 'C21', 'C': 'C22', 'D': 'C28'}
TYPES = {'A': '纯 CPU 型', 'B': 'CPU 主导型', 'C': '均衡型', 'D': 'GPU 主导型'}
USES = {'HPC': '科研计算', 'CLD': '云通用计算', 'SAI': '科研 AI', 'TRN': '大模型训练/开发', 'INF': '推理服务'}
DEMAND = {'HPC': 'Lisa', 'CLD': 'Alibaba', 'SAI': 'MIT', 'TRN': 'Acme', 'INF': 'ASI'}
DAYS = {'HPC': 309, 'CLD': 5, 'SAI': 237, 'TRN': 90, 'INF': 185}
TDP = {'A': 0., 'B': 350., 'C': 350., 'D': 700.}


def catalogue(project):
    path = Path(project) / 'results/server-count-scenarios-20261028-v2/server-configurations.json'
    records = {row['configuration_id']: row for row in json.loads(path.read_text('utf-8'))}
    return {key: records[value] for key, value in LEGACY.items()}


def combinations():
    return [('A', 'HPC'), ('A', 'CLD')] + [(key, use) for key in ('B', 'C', 'D') for use in ('SAI', 'TRN', 'INF')]


def identity(key, use, scale):
    return {'scenario_id': f'{key}-{use}-{scale}MW', 'scenario': f'{key}-{use}',
            'scenario_zh': f'{TYPES[key]}·{USES[use]}·{scale}MW', 'server_type': key,
            'use': use, 'reference_it_mw': scale, 'source_days': DAYS[use], 'cooling_type': 'air'}


def power(config, key, use, cpu, gpu, old_ratios, mit, method='central'):
    if use != 'SAI':
        return power_v2(config, cpu, gpu, 'cpu_spec' if key == 'A' else 'ratio_p50', old_ratios)
    old = component_power(config, cpu, gpu)
    fraction = np.asarray(gpu, dtype=float)
    if not np.isfinite(fraction).all() or ((fraction < 0) | (fraction > 1)).any():
        raise ValueError('科研 AI GPU 分配份额超出有效范围')
    ratio = mit['measured_power_to_tdp_ratio']
    if method in ('ratio_p10', 'ratio_p90'):
        ratio = mit['calibration_device_hour_power_to_tdp'][method.removeprefix('ratio_')]
    elif method not in ('central', 'legacy_idle', 'idle_half', 'idle_double'):
        raise ValueError('未知科研 AI 功率敏感性')
    idle = TDP[key] * mit['idle_W_per_gpu'] / mit['tdp_W']
    if method == 'legacy_idle':
        idle = 70. if key == 'D' else 50.
    elif method == 'idle_half':
        idle *= .5
    elif method == 'idle_double':
        idle *= 2
    gpu_w = config['gpu_devices_per_unit'] * (fraction * TDP[key] * ratio + (1 - fraction) * idle)
    return {**old, 'gpu_w': gpu_w, 'unit_w': old['base_w'] + old['cpu_w'] + gpu_w}


def size(config, scale):
    units = int(scale * 1e6 // reference_power(config))
    return units, units * reference_power(config) / 1e6
