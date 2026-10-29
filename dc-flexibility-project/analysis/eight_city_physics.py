from pathlib import Path
import json

import numpy as np
import pandas as pd
import pvlib
from scipy import sparse
from scipy.optimize import linprog

from compute_load.model import sha256
from typical_power_models import reference_power
from typical_power_models_v2 import power_v2


DEMAND_DAYS = {'Lisa': 309, 'MIT': 237, 'Acme': 90, 'ASI': 1}
SHARES = (0., .25, .5, .75, 1.)
DELAYS = (0, 1, 6, 12, 24, 48)
MULTIPLES = (1., 1.5, 2.)


def read_profiles(project, baseline):
    manifest = json.loads((project / 'results/typical-load-curves-20261029-v2/manifest.json').read_text(encoding='utf-8'))
    configs = json.loads((project / 'results/server-count-scenarios-20261028-v2/server-configurations.json').read_text(encoding='utf-8'))
    ratios = json.loads((project / 'results/typical-load-curves-20261029-v2/mit-power-ratio-summary.json').read_text(encoding='utf-8'))['transfer_distribution']
    profiles, inputs = {}, []
    for config in configs:
        key = config['configuration_id']
        demands = ('Lisa',) if key == 'C20' else (('Acme', 'ASI') if key == 'R-GB200' else ('MIT', 'Acme', 'ASI'))
        for demand in demands:
            method = 'cpu_spec' if key == 'C20' else 'ratio_p50'
            path = baseline / f'{key}-{demand}-{method}-central.parquet'
            digest = sha256(path)
            expected = manifest['processed_sha256'][path.name]
            if digest != expected:
                raise ValueError('典型负荷曲线SHA与v2清单不同：' + path.name)
            frame = pd.read_parquet(path)
            count = DEMAND_DAYS[demand] * 24
            profiles[(key, demand)] = {'config': config, 'method': method, 'ratios': ratios,
                'cpu': frame.u_cpu_proxy.to_numpy()[:count], 'gpu': frame.allocated_gpu_fraction.to_numpy()[:count],
                'units': int(frame.unit_count.iloc[0]), 'capacity_mw': float(frame.installed_reference_it_mw.iloc[0])}
            inputs.append({'path': str(path), 'sha256': digest, 'source_days': DEMAND_DAYS[demand]})
    if len(profiles) != 12:
        raise ValueError('必须有12个可计算10MW组合')
    return profiles, inputs


def replay(profile, hours):
    position = np.arange(hours) % len(profile['cpu'])
    cpu, gpu = profile['cpu'][position], profile['gpu'][position]
    power = power_v2(profile['config'], cpu, gpu, profile['method'], profile['ratios'])
    it = power['unit_w'] * profile['units'] / 1e6
    return cpu, gpu, it


def resource_capacity_factors(weather, city, curve):
    time = pd.DatetimeIndex(weather.time_utc)
    midpoint = time - pd.Timedelta(minutes=30)
    lat, lon = city.get('weather_latitude', city['latitude']), city.get('weather_longitude', city['longitude'])
    # 累积辐射对应结束于时间戳的前一小时；太阳位置取该小时中点。
    solar = pvlib.solarposition.get_solarposition(midpoint, lat, lon,
        temperature=weather.temperature_c.to_numpy(), pressure=weather.pressure_pa.to_numpy())
    zenith = solar.apparent_zenith.to_numpy()
    ghi = weather.ssrd.to_numpy(dtype=float) / 3600
    direct = weather.fdir.to_numpy(dtype=float) / 3600
    if min(ghi.min(), direct.min()) < -1:
        raise ValueError('辐射存在超过浮点舍入范围的负值')
    ghi, direct = np.maximum(ghi, 0), np.maximum(direct, 0)
    excess = direct > ghi
    direct = np.minimum(direct, ghi)
    cosz = np.maximum(np.cos(np.deg2rad(zenith)), 0)
    dni = np.divide(direct, cosz, out=np.zeros(len(direct)), where=cosz > .065)
    extra = pvlib.irradiance.get_extra_radiation(midpoint).to_numpy()
    dni = np.minimum(dni, extra)
    dhi = np.maximum(ghi - dni * cosz, 0)
    wind10 = np.hypot(weather.u10.to_numpy(), weather.v10.to_numpy())
    wind100 = np.hypot(weather.u100.to_numpy(), weather.v100.to_numpy())
    wind2 = wind10 * (2 / 10) ** .14
    tracking = pvlib.tracking.singleaxis(zenith, solar.azimuth.to_numpy(), axis_tilt=0,
        axis_azimuth=0, max_angle=60, backtrack=True, gcr=.35)
    tilt = max(10., min(abs(lat), 60.))
    factors = {}
    for name, slope, azimuth in (
        ('pv_capacity_factor', tilt, 180. if lat >= 0 else 0.),
        ('pv_tracking_capacity_factor', np.nan_to_num(tracking['surface_tilt']),
         np.nan_to_num(tracking['surface_azimuth'], nan=180.))):
        poa = pvlib.irradiance.get_total_irradiance(slope, azimuth, zenith, solar.azimuth.to_numpy(),
            dni, ghi, dhi, dni_extra=extra, albedo=.2, model='haydavies')['poa_global']
        poa = np.maximum(np.asarray(poa), 0)
        cell = pvlib.temperature.faiman(poa, weather.temperature_c.to_numpy(), wind2, u0=25., u1=6.84)
        dc = pvlib.pvsystem.pvwatts_dc(poa, cell, pdc0=1., gamma_pdc=-.004)
        # 容量分母为DC铭牌；DC/AC=1.2，额外系统损失14%，逆变器额定效率96%。
        ac = pvlib.inverter.pvwatts(np.maximum(dc, 0) * .86, pdc0=(1 / 1.2) / .96, eta_inv_nom=.96)
        factors[name] = np.clip(ac, 0, 1 / 1.2)
    # IEA 3.4MW参考机组轮毂110m；10–100m逐小时切变仅作诊断，主计算固定1/7指数。
    hub_wind = wind100 * (110 / 100) ** (1 / 7)
    density = weather.pressure_pa.to_numpy() / (287.05 * (weather.temperature_c.to_numpy() + 273.15))
    effective = hub_wind * (density / 1.225) ** (1 / 3)
    # 档案标称3370kW，表格额定段微小数值误差按铭牌上限处理。
    power = np.interp(effective, curve['Wind Speed [m/s]'], curve['Power [kW]'], left=0, right=0)
    power = np.where(hub_wind > 25, 0, power)
    factors['wind_capacity_factor'] = np.clip(power / 3370, 0, 1) * .90
    factors['wind_gross_capacity_factor'] = factors['wind_capacity_factor'] / .90
    for column, value in factors.items():
        if not np.isfinite(value).all() or ((value < 0) | (value > 1.00001)).any():
            raise ValueError('容量因子范围异常：' + column)
    return pd.DataFrame({'time_utc': time, **factors, 'wind10_m_s': wind10, 'wind100_m_s': wind100,
        'ghi_w_m2': ghi, 'dni_w_m2': dni, 'dhi_w_m2': dhi}), {
        'solar_evaluation': 'hour_ending_UTC; midpoint_geometry', 'direct_above_global_hours': int(excess.sum()),
        'fixed_tilt_deg': tilt, 'hub_height_m': 110., 'wind_shear_exponent': 1 / 7,
        'pv_capacity_basis': 'AC_generation_per_DC_nameplate', 'dc_ac_ratio': 1.2, 'wind_loss_fraction': .10}


def cumulative_bounds(arrivals, delay):
    upper = np.cumsum(arrivals)
    lower = upper.copy() if delay == 0 else np.r_[np.zeros(min(delay, len(arrivals))), upper[:-delay]]
    # 年末全部完成；禁止把最后一个等待窗口的任务移到下一年。
    lower[-1] = upper[-1]
    return lower, upper


def independent_hour_envelope(load, share, delay, multiple):
    fixed, arrivals = (1 - share) * load, share * load
    capacity = multiple - fixed
    lower, upper = cumulative_bounds(arrivals, delay)
    forward = np.zeros(len(load) + 1)
    for t in range(len(load)):
        forward[t + 1] = min(upper[t], forward[t] + capacity[t])
    backward = lower.copy()
    for t in range(len(load) - 2, -1, -1):
        backward[t] = max(backward[t], backward[t + 1] - capacity[t + 1])
    minimum = np.maximum(0, backward - forward[:-1])
    if (minimum > arrivals + 1e-8).any():
        raise ValueError('原计划应当始终满足服务约束')
    return np.maximum(arrivals - minimum, 0)


def flatten_periods(load, share, delay, multiple, period_hours):
    n = len(load)
    groups = np.arange(n) // period_hours
    k = int(groups[-1]) + 1
    fixed = (1 - share) * load
    capacity = multiple - fixed
    arrivals = share * load
    lower, upper = cumulative_bounds(arrivals, delay)
    if share == 0 or delay == 0:
        return load.copy(), {'max_cumulative_residual': 0., 'max_capacity_residual': 0.}
    diff = sparse.diags([np.ones(n), -np.ones(n-1)], [0, -1], shape=(n, n), format='csr')
    membership = sparse.csr_matrix((np.ones(n), (np.arange(n), groups)), shape=(n, k))
    zero = sparse.csr_matrix((n, k))
    matrix = sparse.vstack([
        sparse.hstack([diff, zero, zero]),
        sparse.hstack([-diff, zero, zero]),
        sparse.hstack([diff, -membership, zero]),
        sparse.hstack([-diff, zero, membership])], format='csr')
    bounds = np.vstack([np.column_stack([lower, upper]), np.tile([0., multiple], (2*k, 1))])
    # 年末不足一个完整周的时段只约束期限与容量，不进入完整周的平均峰谷差目标。
    weights = (np.bincount(groups, minlength=k) == period_hours).astype(float)
    weights /= weights.sum()
    objective = np.r_[np.zeros(n), weights, -weights]
    result = linprog(objective, A_ub=matrix, b_ub=np.r_[capacity, np.zeros(n), -fixed, fixed],
        bounds=bounds, method='highs-ds', options={'presolve': False,
            'dual_feasibility_tolerance': 1e-7, 'primal_feasibility_tolerance': 1e-7})
    if not result.success:
        raise RuntimeError('纯物理平坦化求解失败：' + result.message)
    service = diff @ result.x[:n]
    scheduled = fixed + service
    cumulative = np.cumsum(service)
    residual = max(float(np.max(lower-cumulative)), float(np.max(cumulative-upper)), 0.)
    cap_residual = max(float(np.max(scheduled-multiple)), float(np.max(-service)), 0.)
    if max(residual, cap_residual, abs(cumulative[-1]-upper[-1])) > 1e-6:
        raise ValueError('纯物理调度违反服务守恒、期限或容量约束')
    return scheduled, {'max_cumulative_residual': residual, 'max_capacity_residual': cap_residual}


def mean_range(values, period):
    full = values[:len(values) // period * period].reshape(-1, period)
    return float(np.ptp(full, axis=1).mean())
