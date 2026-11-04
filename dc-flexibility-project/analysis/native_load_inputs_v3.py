import calendar
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alibaba_cpu_profile import load_alibaba_source
from annual_task_profiles import load_sources
from asi_inference_profiles import load_asi_sources
from audit_research_gpu_demand import tres_count
from compute_load.cooling import CoolingModel
from compute_load.model import sha256
from load_curve_scenarios_v3 import DAYS
from task_trace_inputs import interval_resource_hours


RUN = 'typical-load-curves-20261101-v3-native'
PREVIOUS = 'typical-load-curves-20261101-v3'
POWER_COLUMNS = ('it', 'cooling', 'other', 'facility', 'base_it', 'cpu_it', 'gpu_it')


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def native_sources(project, source_project, mit_hourly):
    old = load_sources(source_project, include_asi=False)
    lisa, acme = old['lisa_cpu'], old['acme_gpu']
    audit = json.loads((project / 'data/metadata/research-gpu-demand-20261028/source-audit.json').read_text('utf-8'))
    if sha256(mit_hourly) != audit['adopted']['observed_hourly_sha256']:
        raise ValueError('MIT 原生分配输入与来源审计不符')
    mit = pd.read_csv(mit_hourly, parse_dates=['source_time'])
    cld = load_alibaba_source(project)
    inf = load_asi_sources(project)
    result = {}
    for use, source in [('HPC', lisa), ('TRN', acme), ('CLD', cld), ('INF', inf['asi_gpu'])]:
        values = source['profile'].fraction
        result[use] = pd.DataFrame({'source_hour_index': np.arange(len(values)), 'source_clock': source['clock'],
            'u_cpu_proxy': values if use in ('HPC', 'CLD') else .25 * values,
            'allocated_gpu_fraction': np.zeros(len(values)) if use in ('HPC', 'CLD') else values})
    result['SAI'] = pd.DataFrame({'source_hour_index': np.arange(len(mit)), 'source_clock': mit.source_time.astype(str),
        'u_cpu_proxy': mit.host_cpu_allocation_hours / mit.host_cpu_allocation_hours.max(),
        'allocated_gpu_fraction': mit.gpu_allocation_hours / mit.gpu_allocation_hours.max()})
    metadata = {}
    for use, frame in result.items():
        if len(frame) != DAYS[use] * 24 or not np.array_equal(frame.source_hour_index, np.arange(len(frame))):
            raise ValueError('原生窗口小时不符：' + use)
        dated = use in ('HPC', 'SAI', 'TRN')
        if dated:
            clock = pd.DatetimeIndex(pd.to_datetime(frame.source_clock, utc=True))
            if not clock.to_series().diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
                raise ValueError('原生日期不连续：' + use)
            frame['hour_of_source_day'] = clock.hour
            frame['hour_of_source_week'] = clock.dayofweek * 24 + clock.hour
        else:
            frame['hour_of_source_day'] = frame.source_hour_index % 24
            frame['hour_of_source_week'] = frame.source_hour_index % 168
        frame['source_quality_flag'] = 'allocation_proxy' if dated else 'relative_clock'
        frame['all_business_inventory_excess'] = False
        frame['source_unresolved'] = False
        metadata[use] = {'source_days': DAYS[use], 'source_hours': len(frame), 'has_real_dates': dated,
            'source_start': str(frame.source_clock.iloc[0]), 'source_end_inclusive': str(frame.source_clock.iloc[-1]),
            'time_basis': {'HPC': '原文件无时区的日期；源月日小时直接对应目标 UTC，时区对应为假设',
                'SAI': '由源 Slurm epoch 秒形成的日期；保留既有 epoch 小时，不额外换算当地时区',
                'TRN': 'Kalos 原始提交/执行时间带 +00:00，按 UTC',
                'CLD': '相对日 2–6、相对小时 48–167；真实日期和时区未知',
                'INF': '相对日 0–184；真实日期和时区未知'}[use],
            'weekly_basis': '源日期星期' if dated else ('不提供：仅五天' if use == 'CLD' else '从源第0小时起的相对七日分组，无真实星期含义')}
    result['INF']['source_quality_flag'] = inf['asi_gpu']['quality_flags']
    result['INF']['source_unresolved'] = result['INF'].source_quality_flag.eq('all_resources_zero_with_pod_inventory')
    anomalies = pd.read_csv(project / 'results/asi-185day-workload-types-20261031/inventory-attribution-anomalies.csv')
    result['INF']['all_business_inventory_excess'] = result['INF'].source_hour_index.isin(anomalies.hour_index)
    result['INF']['online_gpu_fraction'] = inf['asi_online_gpu']['profile'].fraction
    result['INF']['offline_gpu_fraction'] = inf['asi_offline_gpu']['profile'].fraction
    component = pd.read_csv(project / 'results/alibaba-cpu-cld-20261031/complete-component-window.csv')
    np.testing.assert_array_equal(component.relative_hour, np.arange(48, 168))
    for name in ('online', 'batch', 'residual'):
        result['CLD'][name + '_share'] = component[name + '_share_of_machine'].to_numpy()
    metadata['HPC']['input_sha256'] = lisa['metadata']['input_sha256']
    metadata['TRN']['input_sha256'] = acme['metadata']['input_sha256']
    metadata['SAI']['input_sha256'] = sha256(mit_hourly)
    metadata['CLD']['input_sha256'] = cld['metadata']['input_sha256']
    metadata['INF']['input_sha256'] = inf['asi_gpu']['metadata']['input_sha256']
    return result, metadata


def weather_placements(source, weather, year, dated):
    if dated:
        clock = pd.DatetimeIndex(pd.to_datetime(source.source_clock, utc=True))
        leap = (clock.month == 2) & (clock.day == 29) & (not calendar.isleap(year))
        target = pd.DatetimeIndex([value.replace(year=year, day=28 if flag else value.day)
                                  for value, flag in zip(clock, leap)])
        cases = [('dated', target, np.arange(len(source)), leap)]
    else:
        cases = []
        for month in range(1, 13):
            begin = pd.Timestamp(year=year, month=month, day=1, tz='UTC')
            available = int((pd.Timestamp(year=year + 1, month=1, day=1, tz='UTC') - begin).total_seconds() / 3600)
            hours = min(len(source), available)
            cases.append((f'm{month:02d}', pd.date_range(begin, periods=hours, freq='h'), np.arange(hours), np.zeros(hours, bool)))
    indexed = weather.set_index('time_utc')
    for label, target, position, leap in cases:
        selected = indexed.loc[target].reset_index(names='time_utc')
        if not pd.DatetimeIndex(selected.time_utc).equals(target):
            raise ValueError('气象逐小时对应失败')
        yield label, selected, source.iloc[position].reset_index(drop=True), leap


def calculate(weather, it_mw, capacity, parameter_set='central'):
    # 模型逐小时无记忆；跨源年份和跳过目标闰日时分段调用，保留原生顺序。
    split = np.r_[0, np.flatnonzero(pd.DatetimeIndex(weather.time_utc).to_series().diff().iloc[1:].ne(pd.Timedelta(hours=1)).to_numpy()) + 1, len(weather)]
    frames = []
    for start, end in zip(split[:-1], split[1:]):
        if end - start < 2:
            raise ValueError('气象连续片段不足两个小时')
        frames.append(CoolingModel(cooling_type='air', parameter_set=parameter_set).calculate(
            weather.iloc[start:end], np.asarray(it_mw)[start:end], capacity, full_year=False))
    return pd.concat(frames, ignore_index=True)


def ratio(values):
    low = float(np.min(values))
    return float(np.max(values) / low) if low > 0 else None


def metrics(frame, fields):
    result = {**fields, 'hours': len(frame), 'days': len(frame) / 24,
        'weekly_shape_available': frame.use.iloc[0] != 'CLD',
        'source_quality_flagged_hours': int(frame.source_unresolved.sum())}
    for column in POWER_COLUMNS:
        values = frame[column + '_mw']
        for name, reducer in [('mean', np.mean), ('peak', np.max), ('minimum', np.min)]:
            result[f'{name}_{column}_mw'] = float(reducer(values))
        result['window_' + column + '_mwh'] = float(values.sum())
        if column in ('it', 'facility'):
            result[column + '_window_peak_valley'] = ratio(values)
            result[column + '_diurnal_peak_valley'] = ratio(values.groupby(frame.hour_of_source_day).mean())
            result[column + '_weekly_peak_valley'] = ratio(values.groupby(frame.hour_of_source_week).mean()) if result['weekly_shape_available'] else None
    result['window_pue'] = float(frame.facility_mw.sum() / frame.it_mw.sum()) if 'facility_mw' in frame else None
    return result


def clip_resource(starts, ends, units, source, selected):
    clock = pd.DatetimeIndex(pd.to_datetime(source.source_clock, utc=True))
    edges = pd.date_range(clock[0], periods=len(clock) + 1, freq='h')
    start = pd.DatetimeIndex(pd.to_datetime(starts, utc=True)).asi8
    end = pd.DatetimeIndex(pd.to_datetime(ends, utc=True)).asi8
    return interval_resource_hours(start[selected], end[selected], np.asarray(units)[selected], edges.asi8)


def flexibility_inputs(project, source_project, mit_reference, sources):
    stats, candidates, inputs = {}, {}, {}
    lisa_path = source_project / 'data/processed/task-trace-audit-20261026/lisa/lisa-tasks.parquet'
    lisa = pd.read_parquet(lisa_path, columns=['start_date', 'end_date', 'numcores', 'nodetypes'])
    valid = ~lisa.nodetypes.str.startswith('gpu')
    screen = valid & ~lisa.nodetypes.str.contains('jupyter', case=False) & (lisa.end_date - lisa.start_date).dt.total_seconds().ge(21600)
    all_work = clip_resource(lisa.start_date, lisa.end_date, lisa.numcores, sources['HPC'], valid.to_numpy())
    candidates['HPC'] = clip_resource(lisa.start_date, lisa.end_date, lisa.numcores, sources['HPC'], screen.to_numpy())
    stats['HPC'] = {'resource_hours': float(all_work.sum()), 'candidate_resource_hours': float(candidates['HPC'].sum()), 'screen': '非Jupyter且源运行时长至少6小时；积分裁剪到309天窗口'}
    inputs['HPC'] = sha256(lisa_path)
    acme_path = source_project / 'data/processed/task-trace-audit-20261026/acme-tasks.parquet'
    acme = pd.read_parquet(acme_path)
    valid = acme.interval_valid & acme.gpu_num.gt(0)
    screen = valid & acme.type.eq('Pretrain') & acme.runtime_s.ge(3600)
    all_work = clip_resource(acme.start_time, acme.end_time, acme.gpu_num, sources['TRN'], valid.to_numpy())
    candidates['TRN'] = clip_resource(acme.start_time, acme.end_time, acme.gpu_num, sources['TRN'], screen.to_numpy())
    stats['TRN'] = {'resource_hours': float(all_work.sum()), 'candidate_resource_hours': float(candidates['TRN'].sum()), 'screen': '预训练且源运行时长至少1小时；积分裁剪到90天窗口'}
    inputs['TRN'] = sha256(acme_path)
    path = mit_reference / 'slurm-log.csv'
    mit = pd.read_csv(path, usecols=['time_start', 'time_end', 'time_suspended', 'tres_req', 'tres_alloc', 'job_type'], dtype={'tres_req': str, 'tres_alloc': str})
    requested = sum(tres_count(mit.tres_req, key).fillna(0) for key in (1001, 1002))
    allocated = sum(tres_count(mit.tres_alloc, key).fillna(0) for key in (1001, 1002))
    valid = requested.gt(0) & allocated.gt(0) & mit.time_start.gt(0) & mit.time_end.ge(mit.time_start) & mit.time_suspended.eq(0)
    starts, ends = pd.to_datetime(mit.time_start, unit='s', utc=True), pd.to_datetime(mit.time_end, unit='s', utc=True)
    screen = valid & mit.job_type.isin(['LLSUB:BATCH', 'LLMAPREDUCE:MAP']) & (mit.time_end - mit.time_start).ge(21600)
    all_work = clip_resource(starts, ends, allocated, sources['SAI'], valid.to_numpy())
    candidates['SAI'] = clip_resource(starts, ends, allocated, sources['SAI'], screen.to_numpy())
    stats['SAI'] = {'resource_hours': float(all_work.sum()), 'candidate_resource_hours': float(candidates['SAI'].sum()), 'screen': '明确批处理/Map且源运行时长至少6小时；积分裁剪到237天窗口'}
    inputs['SAI'] = sha256(path)
    for use, denominator in [('HPC', 5411.345), ('TRN', 1536.), ('SAI', 428.92)]:
        fraction = sources[use].u_cpu_proxy if use == 'HPC' else sources[use].allocated_gpu_fraction
        np.testing.assert_allclose(stats[use]['resource_hours'], float(fraction.sum() * denominator), rtol=1e-9, atol=1e-4)
        if np.any(candidates[use] > fraction.to_numpy() * denominator + 1e-5):
            raise ValueError('候选资源超过原生窗口总资源：' + use)
    cld = pd.read_csv(project / 'results/alibaba-cpu-cld-20261031/complete-component-window.csv')
    stats['CLD'] = {'resource_hours': float(cld.occupied_core_hours.sum()), 'candidate_resource_hours': float(cld.batch_mean_core_hours.sum()),
        'online_resource_share': float(cld.online_request_based_core_hours.sum() / cld.occupied_core_hours.sum()),
        'unreconciled_signed_share': float(cld.residual_core_hours.sum() / cld.occupied_core_hours.sum()), 'screen': '120小时窗口的离线实例CPU执行量'}
    typed_path = project / 'results/asi-185day-workload-types-20261031/hourly-type-priority-model.parquet'
    typed = pd.read_parquet(typed_path)
    inference = typed.job_type_public.isin(['online_inference', 'offline_inference'])
    screen = typed.job_type_public.eq('offline_inference') & typed.priority_class.eq('LP')
    candidates['INF'] = typed.loc[screen].groupby('hour_index').used_gpu_hours.sum().reindex(range(4440), fill_value=0).to_numpy()
    stats['INF'] = {'resource_hours': float(typed.loc[inference, 'used_gpu_hours'].sum()), 'candidate_resource_hours': float(candidates['INF'].sum()),
        'offline_share': float(typed.loc[typed.job_type_public.eq('offline_inference'), 'used_gpu_hours'].sum() / typed.loc[inference, 'used_gpu_hours'].sum()), 'screen': '185天推理子集内离线且LP'}
    inputs['CLD'] = sha256(project / 'results/alibaba-cpu-cld-20261031/complete-component-window.csv')
    inputs['INF'] = sha256(typed_path)
    for use, row in stats.items():
        row.update(use=use, source_days=DAYS[use], source_hours=len(sources[use]), candidate_resource_fraction=row['candidate_resource_hours'] / row['resource_hours'],
                   verified_dispatchable_fraction=0., input_sha256=inputs[use])
    return stats, candidates
