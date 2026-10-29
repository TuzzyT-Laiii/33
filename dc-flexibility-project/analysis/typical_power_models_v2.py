import numpy as np
import pandas as pd

from typical_power_models import PARAMETERS, component_power, reference_power


GPU_TDP_W = {'C20': 0., 'C21': 350., 'C22': 350., 'C28': 700., 'R-GB200': 1200.}
GPU_IDLE_W = {'C20': 0., 'C21': 50., 'C22': 50., 'C28': 70., 'R-GB200': 200.}
GB200_POWER_SOURCE = 'https://docs.nvidia.com/datacenter/dps/versions/0.8/guides/concepts/device-specifications/'
GPU_METHODS = ('v1_utilization', 'ratio_p10', 'ratio_p50', 'ratio_p90')


def power_v2(configuration, cpu_fraction, allocated_gpu_fraction, method, ratios):
    key = configuration['configuration_id']
    old = component_power(configuration, cpu_fraction, allocated_gpu_fraction, 'central')
    if key == 'C20' or method == 'v1_utilization':
        return old
    if method not in GPU_METHODS:
        raise ValueError('未知 GPU 功率方法: ' + method)
    fraction = np.asarray(allocated_gpu_fraction, dtype=float)
    if not np.isfinite(fraction).all() or ((fraction < 0) | (fraction > 1)).any():
        raise ValueError('GPU 分配份额必须在0至1之间')
    ratio = ratios[method.removeprefix('ratio_')]
    allocated_w = GPU_TDP_W[key] * ratio
    gpu = configuration['gpu_devices_per_unit'] * (fraction * allocated_w + (1 - fraction) * GPU_IDLE_W[key])
    return {**old, 'gpu_w': gpu, 'unit_w': old['base_w'] + old['cpu_w'] + gpu}


def parameters_v2(configurations, ratios):
    records = []
    for config in configurations:
        key = config['configuration_id']
        base, ci, cf, _, _ = PARAMETERS[key]['central']
        for method in (GPU_METHODS if config['gpu_devices_per_unit'] else ('cpu_spec',)):
            records.append({'configuration_id': key, 'method': method,
                'base_w': base, 'cpu_idle_w_each': ci, 'cpu_full_w_each': cf,
                'cpu_packages_per_unit': config['cpu_packages_per_unit'],
                'gpu_devices_per_unit': config['gpu_devices_per_unit'],
                'gpu_tdp_w_each': GPU_TDP_W[key], 'unallocated_gpu_idle_w_each': GPU_IDLE_W[key],
                'allocated_gpu_power_ratio': ratios[method.removeprefix('ratio_')] if method.startswith('ratio_') else None,
                'sizing_reference_w': reference_power(config),
                'units_at_10mw': int(10e6 // reference_power(config)),
                'gpu_idle_source': GB200_POWER_SOURCE if key == 'R-GB200' else 'PR13 typical_power_models.py central idle assumption',
                'gpu_idle_evidence': 'NVIDIA DPS device-template minLoadWatts; operating scenario assumption' if key == 'R-GB200' else 'assumption; no matched target idle measurement',
                'gpu_tdp_source': GB200_POWER_SOURCE if key == 'R-GB200' else config['source_url'],
                'transfer_status': 'assumption; V100 PCIe monitored GPU-hour power/TDP transferred across generations and workloads',
                'cpu_status': 'PR13 central SPECpower normalized whole-system response; package endpoints remain assumptions'})
    return pd.DataFrame(records)
