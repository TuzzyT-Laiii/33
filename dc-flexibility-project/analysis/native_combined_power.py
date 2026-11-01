import json

import numpy as np
import pandas as pd

from inf_request_shapes import shape as request_shape
from load_curve_scenarios_v3 import catalogue
from trn_power_models import power


RUN = 'typical-load-curves-20261101-v3-native-combined'
NATIVE = 'typical-load-curves-20261101-v3-native'


class CombinedPower:
    def __init__(self, project):
        self.configs = catalogue(project)
        self.ratios = json.loads((project / 'results/typical-load-curves-20261029-v2/mit-power-ratio-summary.json').read_text('utf-8'))['transfer_distribution']
        self.mit = json.loads((project / 'results/typical-load-curves-20261101-v3/mit-calibration.json').read_text('utf-8'))
        self.cpu = json.loads((project / 'results/trn-power-model-20261101/cpu-shape.json').read_text('utf-8'))
        self.training = pd.read_csv(project / 'results/trn-power-model-20261101/diurnal-acme/transfer-profile.csv').set_index('hour_utc').shape_central
        self.parameters = pd.read_csv(project / 'results/typical-load-curves-20261101-v3/power-parameters.csv').query('use == "INF"').set_index('server_type')
        self.calibration = pd.read_csv(project / 'results/inf-diurnal-20261101/power-calibration.csv').query('reference_it_mw == 10').set_index(['server_type', 'location_id', 'year'])
        self.request_root = project / 'results/inf-diurnal-20261101'
        if self.mit['partial_data'] or not self.calibration.index.is_unique:
            raise ValueError('必须采用MIT六月完成版和唯一的推理标定参数')

    def unit(self, source, key, use, location='reference-grid', year=2025):
        cpu, allocated = source.u_cpu_proxy.to_numpy(), source.allocated_gpu_fraction.to_numpy()
        extra = {}
        if use == 'TRN':
            extra['training_ratio'] = self.training.reindex(source.hour_of_source_day).to_numpy()
        result = power(self.configs[key], key, use, cpu, allocated, self.ratios, self.mit, self.cpu, **extra)
        if use == 'TRN':
            result['allocated_gpu_power_ratio'] = extra['training_ratio']
        if use == 'INF':
            calibration = self.calibration.loc[(key, location, int(year))]
            clock = pd.DatetimeIndex(pd.Timestamp('2025-01-01', tz='UTC') + pd.to_timedelta(source.source_hour_index.to_numpy(), unit='h'))
            intensity = request_shape(self.request_root, clock)
            online_w = calibration.idle_w + (calibration.high_w - calibration.idle_w) * np.minimum(calibration.gain * intensity ** calibration.exponent, 1.)
            parameters = self.parameters.loc[key]
            mean_w = parameters.target_gpu_tdp_W * parameters.allocated_gpu_ratio
            cards = self.configs[key]['gpu_devices_per_unit']
            online, offline = source.online_gpu_fraction.to_numpy(), source.offline_gpu_fraction.to_numpy()
            np.testing.assert_allclose(online + offline, allocated, rtol=1e-12, atol=1e-12)
            result.update(request_intensity=intensity, online_gpu_w_each=online_w,
                online_gpu_w=cards * online * online_w, offline_gpu_w=cards * offline * mean_w,
                unallocated_gpu_w=cards * (1 - allocated) * parameters.idle_gpu_W)
            result['gpu_w'] = result['online_gpu_w'] + result['offline_gpu_w'] + result['unallocated_gpu_w']
            result['unit_w'] = result['base_w'] + result['cpu_w'] + result['gpu_w']
        return result
