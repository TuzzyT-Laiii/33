import numpy as np


# 低、中、高指波动敏感性；各组同时满足年度在线GPU电量约束。
CASES = {'low': (160 / 700, .50, .7), 'central': (120 / 700, .75, 1.),
         'high': (100 / 700, 1., 1.3)}


def online_power(intensity, allocated, tdp, mean_w, case='central'):
    idle_ratio, high_ratio, exponent = CASES[case]
    idle, high = tdp * idle_ratio, tdp * high_ratio
    workload = np.asarray(intensity, dtype=float)
    allocated = np.asarray(allocated, dtype=float)
    if workload.shape != allocated.shape or not np.isfinite(workload).all() or (workload < 0).any():
        raise ValueError('负载强度必须是与分配份额等长的非负有限序列')
    if not np.isfinite(allocated).all() or ((allocated < 0) | (allocated > 1)).any() or allocated.sum() <= 0:
        raise ValueError('在线分配份额无效')
    if not idle < mean_w < high:
        raise ValueError('年度平均每卡功率必须位于空载和高负载端点之间')
    demand = workload ** exponent
    target = (mean_w - idle) / (high - idle)
    max_response = np.average(demand > 0, weights=allocated)
    if max_response <= target:
        raise ValueError('正观测负载小时不足以满足年度能量约束')
    lower, upper = 0., 1.
    while np.average(np.minimum(upper * demand, 1), weights=allocated) < target:
        upper *= 2
    for _ in range(90):
        middle = (lower + upper) / 2
        if np.average(np.minimum(middle * demand, 1), weights=allocated) < target:
            lower = middle
        else:
            upper = middle
    gain = (lower + upper) / 2
    response = np.minimum(gain * demand, 1.)
    watts = idle + (high - idle) * response
    np.testing.assert_allclose(np.dot(allocated, watts), mean_w * allocated.sum(), rtol=1e-12)
    return watts, dict(idle_w=idle, high_w=high, exponent=exponent, gain=gain,
        target_mean_w=mean_w, allocation_weighted_mean_w=float(np.average(watts, weights=allocated)),
        realized_min_w=float(watts.min()), realized_max_w=float(watts.max()),
        saturated_hours=int((response >= 1 - 1e-12).sum()))
