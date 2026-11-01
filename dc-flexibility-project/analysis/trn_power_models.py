import numpy as np

from load_curve_scenarios_v3 import power as power_v3, TDP
from typical_power_models import PARAMETERS, validate_utilization


TRN_RATIOS = {'low': .65, 'central': .80, 'high': .95}


def cpu_response(cpu, shape):
    u = validate_utilization(cpu)
    c1, c2 = shape['coefficients']
    if not 0 <= c1 <= c2 <= 1:
        raise ValueError('CPU Bernstein 系数必须单调且在0至1之间')
    return 3 * c1 * u * (1 - u) ** 2 + 3 * c2 * u ** 2 * (1 - u) + u ** 3


def power(config, key, use, cpu, gpu, old_ratios, mit, shape, *, training_case='central',
          cpu_method='smooth', idle_case='central', training_ratio=None):
    if training_case not in TRN_RATIOS or cpu_method not in ('smooth', 'v3') or idle_case not in PARAMETERS[config['configuration_id']]:
        raise ValueError('未知功率敏感性参数')
    cpu, gpu = validate_utilization(cpu), validate_utilization(gpu)
    old = power_v3(config, key, use, cpu, gpu, old_ratios, mit)
    base, ci, cf, _, _ = PARAMETERS[config['configuration_id']]['central']
    cpu_w = old['cpu_w'] if cpu_method == 'v3' else config['cpu_packages_per_unit'] * (ci + (cf - ci) * cpu_response(cpu, shape))
    gpu_w = old['gpu_w']
    if use == 'TRN':
        ratio = TRN_RATIOS[training_case] if training_ratio is None else np.asarray(training_ratio, dtype=float)
        if np.ndim(ratio) > 1 or (np.ndim(ratio) == 1 and ratio.shape != gpu.shape):
            raise ValueError('训练功率比例须为标量或与需求等长的序列')
        if not np.isfinite(ratio).all() or np.any((ratio < 0) | (ratio > 1)):
            raise ValueError('训练功率比例必须在0至1之间')
        idle = 70. if key == 'D' else 50.
        gpu_w = config['gpu_devices_per_unit'] * (gpu * TDP[key] * ratio + (1 - gpu) * idle)
    elif training_ratio is not None:
        raise ValueError('逐小时训练功率比例仅适用于 TRN')
    # 空载敏感性只改变无任务端点，保持动态 CPU 幅度与分配 GPU 功率口径。
    alt_base, alt_ci, _, alt_gi, _ = PARAMETERS[config['configuration_id']][idle_case]
    if idle_case != 'central':
        base = alt_base
        cpu_w = cpu_w + config['cpu_packages_per_unit'] * (alt_ci - ci)
        if use != 'SAI':
            gpu_w = gpu_w + config['gpu_devices_per_unit'] * (1 - gpu) * (alt_gi - (70. if key == 'D' else 50. if key != 'A' else 0.))
    base_w = np.broadcast_to(float(base), cpu.shape)
    return {'base_w': base_w, 'cpu_w': cpu_w, 'gpu_w': gpu_w, 'unit_w': base_w + cpu_w + gpu_w}


def idle_power(config, key, use, mit, idle_case='central'):
    base, ci, _, gi, _ = PARAMETERS[config['configuration_id']][idle_case]
    if use == 'SAI':
        gi = TDP[key] * mit['idle_W_per_gpu'] / mit['tdp_W']
    return base + config['cpu_packages_per_unit'] * ci + config['gpu_devices_per_unit'] * gi
