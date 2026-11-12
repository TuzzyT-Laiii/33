import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ('requests', 'input_tokens', 'output_tokens')


def ratio(values):
    return float(values.max() / values.min()) if values.min() > 0 else None


def get_series(observed, audit):
    sources = {'burst': observed.loc[observed.dataset.isin(['burst-1', 'burst-2'])]}
    sources.update({name: rows for name, rows in observed.groupby('dataset') if name.startswith('azure')})
    for name, rows in sources.items():
        values = rows.groupby('hour_index')[list(METRICS)].sum().sort_index()
        first, last = int(values.index.min()), int(values.index.max())
        if name == 'burst':
            # 上游核准窗口为相对日1至119；边界日不作为完整日。
            index = np.arange(24, 120 * 24)
        elif name.startswith('azure-2024'):
            record = next(r for r in audit['inputs'] if r['dataset'] == name.replace('azure-2024-', 'azure-'))
            # 官方七日文件从午夜开始；边界距午夜不超过一分钟时保留边界日。
            start_day = first // 24 + int(record['min_seconds'] % 86400 >= 60)
            stop_day = last // 24 + int(record['max_seconds'] % 86400 >= 86340)
            index = np.arange(start_day * 24, stop_day * 24)
        else:
            index = np.arange(first, last + 1)
        if not len(index):
            raise ValueError(f'无有效小时：{name}')
        values = values.reindex(index, fill_value=0).rename_axis('hour_index').reset_index()
        values['day'] = values.hour_index // 24
        values['hour'] = values.hour_index % 24
        values['phase_day'] = values.day % 7 if name == 'burst' else (values.day + 3) % 7
        values['source'] = name
        values['zero_observation'] = values.requests.eq(0)
        yield name, values, first, last


def summarize(output):
    observed = pd.read_csv(output / 'request-hourly-observed.csv')
    audit = json.loads((output / 'request-input-audit.json').read_text('utf-8'))
    hourly, diurnal, weekly, weekday, stats, phase_stats = [], [], [], [], [], []
    for source, values, first, last in get_series(observed, audit):
        hourly.append(values)
        variants = [('retained_zero', values)]
        if source == 'burst':
            valid = values.groupby('day').zero_observation.sum().eq(0)
            variants.append(('complete_positive_days', values.loc[values.day.isin(valid.index[valid])]))
        for policy, data in variants:
            day_shape = data.groupby('hour')[list(METRICS)].mean()
            week_shape = data.groupby(['phase_day', 'hour'])[list(METRICS)].mean()
            # 每个完整相对周独立归一化，检查增长对周形状的影响。
            detrended = data.copy()
            detrended['week_block'] = (detrended.day - data.day.min()) // 7
            for metric in METRICS:
                detrended[metric] /= detrended.groupby('week_block')[metric].transform('mean')
            deweek = detrended.groupby(['phase_day', 'hour'])[list(METRICS)].mean()
            for metric in METRICS:
                mean = float(data[metric].mean())
                d = day_shape[metric]
                complete24 = len(d) == 24
                stats.append({'source': source, 'zero_policy': policy, 'metric': metric,
                    'hours': len(data), 'days': data.day.nunique(), 'zero_observed_hours': int(data.zero_observation.sum()),
                    'first_observed_hour': first, 'last_observed_hour': last,
                    'mean': mean, 'total': int(data[metric].sum()), 'diurnal_peak_valley': ratio(d) if complete24 else None,
                    'observed_support_peak_valley': ratio(d), 'peak_hour': int(d.idxmax()), 'valley_hour': int(d.idxmin()),
                    'full_24h_support': complete24,
                    'weekly_168_peak_valley': ratio(week_shape[metric]) if len(week_shape) == 168 else None,
                    'weekly_detrended_peak_valley': ratio(deweek[metric]) if len(deweek) == 168 else None})
                for hour, value in d.items():
                    diurnal.append(dict(source=source, zero_policy=policy, metric=metric,
                                        hour=hour, mean_value=value, normalized=value / mean))
                for (day, hour), value in week_shape[metric].items():
                    weekly.append(dict(source=source, zero_policy=policy, metric=metric, phase_day=day, hour=hour,
                                       normalized=value / mean, detrended_normalized=deweek.loc[(day, hour), metric]))
            phases = range(7) if source == 'burst' else [0]
            for phase in phases:
                # Burst 的 phase 表示 day0 的假定星期（0=周一）；Azure 使用时间戳星期。
                dow = (data.day + phase) % 7 if source == 'burst' else data.phase_day
                if dow.nunique() < 7:
                    continue
                for metric in METRICS:
                    work = data.loc[dow < 5, metric]
                    end = data.loc[dow >= 5, metric]
                    phase_stats.append(dict(source=source, zero_policy=policy, assumed_day0_weekday=phase,
                        metric=metric, weekend_to_weekday=float(end.mean() / work.mean()),
                        weekday_hours=len(work), weekend_hours=len(end)))
                for label, selected in [('weekday', data.loc[dow < 5]), ('weekend', data.loc[dow >= 5])]:
                    curves = selected.groupby('hour')[list(METRICS)].mean()
                    for metric in METRICS:
                        for hour, value in curves[metric].items():
                            weekday.append(dict(source=source, zero_policy=policy, assumed_day0_weekday=phase,
                                day_type=label, metric=metric, hour=hour, normalized=value / data[metric].mean()))
    for name, rows in [('request-analysis-hourly.csv', pd.concat(hourly, ignore_index=True)),
                       ('request-shape-24h.csv', pd.DataFrame(diurnal)),
                       ('request-shape-168h.csv', pd.DataFrame(weekly)),
                       ('request-weekday-weekend.csv', pd.DataFrame(weekday)),
                       ('request-shape-statistics.csv', pd.DataFrame(stats)),
                       ('request-weekend-phase-sensitivity.csv', pd.DataFrame(phase_stats))]:
        rows.to_csv(output / name, index=False, encoding='utf-8-sig')


def shape(output, times, source='burst', mode='daily', alpha=.5, phase=0, policy='retained_zero'):
    path = output / ('request-shape-168h.csv' if mode == 'weekly' else 'request-shape-24h.csv')
    table = pd.read_csv(path)
    rows = table.loc[table.source.eq(source) & table.zero_policy.eq(policy)]
    if mode == 'weekly':
        # 合成日历：day0 假设为 phase 指定的星期；不声称观测真实星期。
        selected_day = (times.dayofweek - phase) % 7 if source == 'burst' else times.dayofweek
        coordinates = pd.MultiIndex.from_arrays([selected_day, times.hour])
        normalized = rows.set_index(['metric', 'phase_day', 'hour']).normalized
        vectors = {m: normalized.loc[m].reindex(coordinates).to_numpy() for m in METRICS}
    else:
        normalized = rows.set_index(['metric', 'hour']).normalized
        vectors = {m: normalized.loc[m].reindex(times.hour).to_numpy() for m in METRICS}
    if any(not np.isfinite(v).all() for v in vectors.values()):
        raise ValueError(f'请求形状没有完整支持：{source}/{mode}')
    return vectors['requests'] if mode == 'requests' else alpha * vectors['input_tokens'] + (1 - alpha) * vectors['output_tokens']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    summarize(parser.parse_args().output)
