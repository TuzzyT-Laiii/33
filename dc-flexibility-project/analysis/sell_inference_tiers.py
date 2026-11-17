"""第二节卖推理三层（八城市）：理想上限、真实形状上限、上限的 25/50/75%。

沿用 sell_idle.py（第二节出售空闲算力）的机群、任务与能源成本口径：机群 = A，原任务按原时序，
额外能源成本 = 售出 GPU 小时 × 每 GPU 动态任务功率 × 当小时 PUE × (市电 100 美元/MWh + 碳价 100 美元/t × 当地碳因子)，不做电力调度。
- ideal：每小时闲置 GPU（A 的 GPU 数 − 任务占用）全部售出，即旧表 s = 100%。
- real_max：外部推理需求按 ASI 推理曲线（在线 + 离线）逐时分布，平均量 = 可承接上限 × 平均闲置 GPU
  （上限见 residual_flex_max_intensity.py，在算力侧期限与容量约束下求得，与地点无关）。
- real_25/50/75：real_max 的 25/50/75%。
时间对齐与第三节相同：训练窗口整周平移 14 天，科研 AI 只保留与 ASI 重叠的小时；三层在同一窗口上计算后按窗口小时年化。
售价为 GPU·时等效价，token 换算另做。两种计价：flat（全时同价）与 ds_tou（DeepSeek API 现行峰谷：
周一至周五 UTC 01:00–04:00、06:00–10:00 为高峰，不含中国法定节假日；其余时段（含周末、节假日全天）半价；
价格档视为窗口内全时段平均价（高峰 = 2 × 低谷，按窗口小时平均后等于价格档），只改变收入在时段上的分布。
主价格 2.9 美元/GPU·时 = DeepInfra H100 80GB 自定义部署价（2026-11-15 核对 https://deepinfra.com/pricing，不分时段）。2026-11-15 核对 https://api-docs.deepseek.com/quick_start/pricing）。
"""
import glob
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

P = Path('/home/ljt/.codex-temp/joint-compute-energy-20261107')
D = Path('/data3/dataset/ljt/residual-flex-global-20261114')
OUT = D / 'sell-inference-tiers'
SCENARIOS = [f'{k}-{u}' for u in ('SAI', 'TRN', 'INF') for k in 'BCD']
PRICES = [0.5, 1.0, 1.5, 2.2, 2.9]
LEVELS = {'real_25': 0.25, 'real_50': 0.5, 'real_75': 0.75, 'real_max': 1.0}
GRID_PRICE, CARBON_PRICE = 100.0, 100.0
SHIFT_DAYS = {'TRN': 14}
# 2025 年中国法定节假日（国务院办公厅 2024-11 通知）；调休上班日按周末处理（DeepSeek 规则只写周一至周五）
CN_HOLIDAYS_2025 = pd.to_datetime(['2025-01-01'] + [f'2025-01-{d}' for d in range(28, 32)] + [f'2025-02-0{d}' for d in range(1, 5)]
                                  + ['2025-04-04', '2025-04-05', '2025-04-06'] + [f'2025-05-0{d}' for d in range(1, 6)]
                                  + ['2025-05-31', '2025-06-01', '2025-06-02'] + [f'2025-10-0{d}' for d in range(1, 9)]).date


def ds_tou_multiplier(time_utc):
    t = pd.to_datetime(time_utc, utc=True)
    beijing_date = (t + pd.Timedelta(hours=8)).dt.date
    peak = t.dt.hour.isin([1, 2, 3, 6, 7, 8, 9]) & (t.dt.dayofweek < 5) & ~pd.Series(beijing_date).isin(CN_HOLIDAYS_2025).to_numpy()
    return np.where(peak, 1.0, 0.5)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from residual_flex_pilot import asi_inference  # noqa: E402


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    asi = asi_inference(D / 'inputs/asi-D-INF-native-hourly.parquet').set_index('time_utc')
    max_i = pd.read_csv(D / 'max-intensity.csv').set_index('scenario').max_intensity
    rows = []
    for scenario in SCENARIOS:
        res = pd.read_parquet(P / f'inputs/resources/{scenario}.parquet')[['source_hour_index', 'gpu_hours']]
        for a_file in sorted(glob.glob(str(P / f'results/A/*-{scenario}-economic.json'))):
            a = json.load(open(a_file))
            city = a['city']
            frame = pd.read_parquet(P / f'inputs/cities/{city}-{scenario}.parquet').merge(res, on='source_hour_index', how='inner', validate='one_to_one')
            full_hours = len(frame)
            t = pd.to_datetime(frame.time_utc, utc=True) - pd.Timedelta(days=SHIFT_DAYS.get(scenario.split('-')[1], 0))
            covered = t.isin(asi.index).to_numpy()
            frame = frame.loc[covered].reset_index(drop=True)
            shape = asi.reindex(t[covered])
            shape = (shape.on + shape.off).to_numpy()
            hours = len(frame)
            scale = 8760 / hours
            gpus = a['gpus']
            used = frame.gpu_hours.to_numpy()
            task_mw = (frame.rigid + frame.unknown + frame.whole + frame.split).to_numpy()
            mw_per_gpu = task_mw.sum() / used.sum()
            ef = a['emissions_t'] / a['grid_mwh'] if a['grid_mwh'] > 0 else 0.0
            cost_per_gpu_h = mw_per_gpu * frame.pue.to_numpy() * (GRID_PRICE + CARBON_PRICE * ef)
            idle = np.clip(gpus - used, 0, None)
            tou = ds_tou_multiplier(frame.time_utc)
            tou = tou / tou.mean()  # 平均价不变
            row = {'scenario': scenario, 'city': city, 'peak_hour_share': float((tou == 1).mean()), 'hours_window': hours, 'hours_dropped_outside_asi': full_hours - hours,
                   'gpus_A': gpus, 'max_used_over_gpus': used.max() / gpus, 'mean_idle_share': idle.mean() / gpus,
                   'max_intensity': max_i[scenario], 'mw_per_gpu': mw_per_gpu, 'grid_ef_t_per_mwh': ef,
                   'fleet_annual_usd': a['equipment_annual_usd']}
            sold = {'ideal': idle}
            for name, level in LEVELS.items():
                sold[name] = level * max_i[scenario] * idle.mean() * shape / shape.mean()
            for name, s in sold.items():
                row[f'{name}_gpu_hours_year'] = s.sum() * scale
                row[f'{name}_share_of_idle'] = s.sum() / idle.sum()
                energy = (s * cost_per_gpu_h).sum() * scale
                row[f'{name}_energy_cost_usd'] = energy
                row[f'{name}_tou_factor'] = (s * tou).sum() / s.sum()
                for p in PRICES:
                    row[f'{name}_net_p{p}'] = s.sum() * scale * p - energy
                    row[f'{name}_net_tou_p{p}'] = (s * tou).sum() * scale * p - energy
            rows.append(row)
            print(scenario, city, f"max={max_i[scenario]:.3f}", f"ideal(p2.2)={row['ideal_net_p2.2'] / 1e4:.0f}万",
                  f"real_max(p2.2)={row['real_max_net_p2.2'] / 1e4:.0f}万", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'sell-inference-tiers-city.csv', index=False, encoding='utf-8-sig')
    df.drop(columns=['city']).groupby('scenario').median(numeric_only=True).to_csv(OUT / 'sell-inference-tiers-median.csv', encoding='utf-8-sig')


if __name__ == '__main__':
    main()
