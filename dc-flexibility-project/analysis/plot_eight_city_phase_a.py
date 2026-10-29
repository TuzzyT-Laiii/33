import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
import numpy as np
import pandas as pd
from PIL import Image

from eight_city_scenarios import PURPOSES


RUN = 'eight-city-phase-a-20261029'
CITY_ORDER = ['singapore','phoenix','ashburn','frankfurt','lulea','ulanqab','guiyang','riyadh']
COMPONENTS = [('it_mw','IT','#315b7d'),('cooling_mw','冷却','#d88a32'),('facility_mw','设施','#39794e')]
SEASONS = {0: '冬季 DJF', 1: '春季 MAM', 2: '夏季 JJA', 3: '秋季 SON'}


def setup():
    available = {f.name for f in fontManager.ttflist}
    font = next((f for f in ('Microsoft YaHei','SimHei','Noto Sans CJK SC','SimSun') if f in available), None)
    if font is None:
        raise RuntimeError('图件需要实际可用的中文字体')
    plt.rcParams.update({'font.family': font, 'font.size': 9, 'axes.unicode_minus': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelsize': 10, 'legend.frameon': False, 'savefig.dpi': 300})
    return font


def save(fig, path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    with Image.open(path.with_suffix('.png')) as preview:
        width, height = preview.size
        dpi = preview.info.get('dpi', (0,0))
    records.append({'figure': str(path), 'width_px': width, 'height_px': height, 'dpi': list(dpi),
        'png_bytes': path.with_suffix('.png').stat().st_size, 'pdf_bytes': path.with_suffix('.pdf').stat().st_size})
    plt.close(fig)


def heatmap(ax, values, rows, columns, label, fmt='.2f', cmap='viridis', vmin=None, vmax=None):
    image = ax.imshow(values, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(rows)), rows)
    for y, row in enumerate(values):
        for x, value in enumerate(row):
            color = 'white' if image.norm(value) < .45 else '#151515'
            ax.text(x,y,format(value,fmt),ha='center',va='center',fontsize=7.5,color=color)
    ax.figure.colorbar(image,ax=ax,label=label,shrink=.8)
    return image


def load_curve(processed, city, scenario):
    full = processed / f'{city}-{scenario}.parquet'
    thermal = processed / f'{city}-{scenario}-thermal-only.parquet'
    path = full if full.exists() else thermal
    frame = pd.read_parquet(path).reset_index(drop=True)
    frame.time_utc = pd.to_datetime(frame.time_utc,utc=True)
    return frame


def load_patterns(output, processed, cities, scenarios, records):
    folder = output / 'load-patterns'
    week_records, aggregate = [], []
    for scenario in scenarios:
        frames = {c:load_curve(processed,c,scenario) for c in CITY_ORDER}
        fig, axes = plt.subplots(8,4,figsize=(15.8,20),sharex=True,sharey=True,layout='constrained')
        ymax = max(f.facility_mw.max() for f in frames.values()) * 1.06
        for row, city in enumerate(CITY_ORDER):
            frame = frames[city]
            local = frame.time_utc.dt.tz_convert(cities[city]['timezone'])
            season = (local.dt.month % 12 // 3).to_numpy()
            for column in range(4):
                selected = frame.loc[season == column].copy()
                selected['local_hour'] = local.loc[season == column].dt.hour.to_numpy()
                means = selected.groupby('local_hour')[[v[0] for v in COMPONENTS]].mean()
                ax = axes[row,column]
                for field,label,color in COMPONENTS:
                    ax.plot(means.index,means[field],color=color,lw=1.4,label=label)
                ax.set_ylim(0,ymax)
                ax.set_xlim(0,23)
                ax.set_xticks([0,6,12,18,23])
                ax.grid(axis='y',alpha=.18)
                if row == 0:
                    ax.set_title(SEASONS[column],fontsize=11)
                if column == 0:
                    ax.set_ylabel(cities[city]['city_zh']+'\n功率（MW）')
                if row == 7:
                    ax.set_xlabel('当地钟表小时')
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside upper center',ncols=3,bbox_to_anchor=(.5,1.023))
        fig.suptitle(scenario+'｜四季典型日（2023–2025，中央参数）',fontsize=15,y=1.047)
        save(fig,folder / (scenario+'-seasonal-days'),records)
        # 每个情境在八城市使用共同代表周，兼顾各城冷却、设施与室外气温形状。
        year_frames = {c:f.loc[f.time_utc.dt.year.eq(2025)].reset_index(drop=True) for c,f in frames.items()}
        times = pd.DatetimeIndex(year_frames[CITY_ORDER[0]].time_utc)
        starts = [i for i,t in enumerate(times) if t.dayofweek == 0 and t.hour == 0 and i+168 <= len(times)]
        features = np.column_stack([year_frames[c][v].to_numpy() for c in CITY_ORDER
            for v in ('it_mw','cooling_mw','facility_mw','temperature_c')])
        weeks = np.stack([features[i:i+168] for i in starts])
        scale = np.maximum(features.std(axis=0),1e-8)
        score = np.mean(((weeks-weeks.mean(axis=0))/scale)**2,axis=(1,2))
        start = starts[int(np.argmin(score))]
        fig,axes = plt.subplots(4,2,figsize=(14.5,11),sharex=True,sharey=True,layout='constrained')
        for ax,city in zip(axes.flat,CITY_ORDER):
            frame = year_frames[city].iloc[start:start+168]
            for field,label,color in COMPONENTS:
                ax.plot(np.arange(168),frame[field],color=color,lw=1.1,label=label)
            ax.set_title(cities[city]['city_zh'])
            ax.set_ylim(0,ymax)
            ax.set_ylabel('功率（MW）')
            ax.set_xticks(np.arange(0,168,24),['周一','周二','周三','周四','周五','周六','周日'])
            ax.grid(axis='y',alpha=.18)
        fig.legend(handles,labels,loc='outside upper center',ncols=3,bbox_to_anchor=(.5,1.025))
        fig.suptitle(f'{scenario}｜代表周 {times[start]:%Y-%m-%d} 至 {times[start+167]:%m-%d}（UTC）',fontsize=14,y=1.058)
        save(fig,folder / (scenario+'-typical-week'),records)
        week_records.append({'scenario_scale_id':scenario,'week_start_utc':str(times[start]),
            'week_end_utc':str(times[start+167]),'standardized_distance':float(score.min()),
            'selection':'common_eight_city_week; IT_cooling_facility_temperature_standardized_MSE'})
        fig,axes = plt.subplots(4,2,figsize=(14.5,11),sharex=True,sharey=True,layout='constrained')
        for ax,city in zip(axes.flat,CITY_ORDER):
            frame = year_frames[city].copy()
            frame['day'] = frame.time_utc.dt.floor('D')
            daily = frame.groupby('day').facility_mw.agg(['mean','min','max'])
            ax.fill_between(daily.index,daily['min'],daily['max'],color='#9fc8b0',alpha=.55,label='设施小时最小–最大')
            ax.plot(daily.index,daily['mean'],color='#39794e',lw=1.2,label='设施日均')
            ax.set_title(cities[city]['city_zh'])
            ax.set_ylim(0,ymax)
            ax.set_ylabel('设施功率（MW）')
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m月'))
            ax.grid(axis='y',alpha=.18)
        h,l=axes[0,0].get_legend_handles_labels()
        fig.legend(h,l,loc='outside upper center',ncols=2,bbox_to_anchor=(.5,1.025))
        fig.suptitle(scenario+'｜全年包络（2025，UTC）',fontsize=14,y=1.058)
        save(fig,folder / (scenario+'-annual-envelope'),records)
        print(json.dumps({'load_patterns_complete':scenario}),flush=True)
    pd.DataFrame(week_records).to_csv(output/'representative-weeks.csv',index=False,encoding='utf-8-sig')


def overview(output, processed, cities, scenarios, benchmark, records):
    annual=pd.read_csv(output/'annual-load-central.csv')
    names=[cities[c]['city_zh'] for c in CITY_ORDER]
    fig,axes=plt.subplots(1,2,figsize=(16,7),layout='constrained')
    for ax,field,label,fmt in ((axes[0],'annual_pue','年均 PUE','.2f'),
        (axes[1],'annual_facility_mwh','年设施电量（GWh）','.1f')):
        table=annual.groupby(['scenario_scale_id','city_id'])[field].mean().unstack().reindex(index=scenarios,columns=CITY_ORDER)
        values=table.to_numpy()/(1000 if field.endswith('mwh') else 1)
        heatmap(ax,values,scenarios,names,label,fmt)
        ax.tick_params(axis='x',rotation=30)
    fig.suptitle('八城市 × 十二情境（10 MW，2023–2025年均，中央冷却）',fontsize=14)
    save(fig,output/'figure2-city-load-comparison',records)
    climate=pd.read_csv(output/'city-climate-resource-annual.csv')
    if 'pv_capacity_factor' not in climate:
        return
    means=climate.groupby('city_id').mean(numeric_only=True).reindex(CITY_ORDER)
    bench=benchmark.set_index('city_id').reindex(CITY_ORDER)
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    x=np.arange(8)
    axes[0,0].bar(x,means.mean_temperature_c,color='#b66b47')
    axes[0,0].set_ylabel('2023–2025 年均气温（°C）')
    axes[0,1].bar(x-.14,means.pv_capacity_factor*100,width=.28,label='固定倾角')
    axes[0,1].bar(x+.14,means.pv_tracking_capacity_factor*100,width=.28,label='单轴跟踪',color='#d49c3a')
    axes[0,1].scatter(x,bench.benchmark_pv_cf_dc_basis*100,marker='x',color='black',s=42,label='公开基准')
    axes[0,1].set_ylabel('光伏年容量因子（%，DC分母）')
    axes[0,1].legend(fontsize=8)
    axes[1,0].bar(x,means.wind_capacity_factor*100,color='#448278',label='IEA参考机型，含10%损失')
    axes[1,0].scatter(x,bench.gwa_iec3_capacity_factor*100,marker='x',color='black',s=42,label='GWA IEC III 基准')
    axes[1,0].set_ylabel('风电年容量因子（%）')
    axes[1,0].legend(fontsize=8)
    axes[1,1].bar(x,means.ghi_kwh_m2,color='#d49c3a')
    axes[1,1].set_ylabel('年水平总辐射（kWh/m²）')
    for ax in axes.flat:
        ax.set_xticks(x,names,rotation=30)
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    save(fig,output/'figure1-climate-resource-overview',records)
    fig,axes=plt.subplots(2,2,figsize=(13.5,9),layout='constrained')
    colors=plt.get_cmap('tab10').colors
    for i,city in enumerate(CITY_ORDER):
        supply=pd.read_parquet(processed/(city+'-supply.parquet'))
        local=pd.to_datetime(supply.time_utc,utc=True).dt.tz_convert(cities[city]['timezone'])
        for row,field in enumerate(('pv_capacity_factor','wind_capacity_factor')):
            daily=supply.groupby(local.dt.hour)[field].mean()
            monthly=supply.groupby(local.dt.month)[field].mean()
            axes[row,0].plot(daily.index,daily*100,color=colors[i],label=cities[city]['city_zh'],lw=1.5)
            axes[row,1].plot(monthly.index,monthly*100,color=colors[i],lw=1.5)
    for row,label in enumerate(('光伏','风电')):
        for col in range(2):
            axes[row,col].set_ylabel(label+'容量因子（%）')
            axes[row,col].grid(alpha=.18)
            axes[row,col].set_ylim(bottom=0)
    for ax in axes[:,0]:
        ax.set_xlabel('当地钟表小时')
        ax.set_xticks([0,6,12,18,23])
    for ax in axes[:,1]:
        ax.set_xlabel('月份')
        ax.set_xticks(range(1,13))
    h,l=axes[0,0].get_legend_handles_labels()
    fig.legend(h,l,loc='outside upper center',ncols=8,bbox_to_anchor=(.5,1.035),fontsize=9)
    save(fig,output/'figure3-renewable-shapes',records)


def flexibility_plot(output,records):
    path=output/'flexibility-resource-summary.csv'
    if not path.exists():
        return
    frame=pd.read_csv(path)
    fig,axes=plt.subplots(4,3,figsize=(14,13),layout='constrained')
    for row,purpose in enumerate(('HPC','SAI','TRN','INF')):
        for col,multiple in enumerate((1.,1.5,2.)):
            selected=frame.loc[frame.purpose.eq(purpose)&frame.capacity_multiple.eq(multiple)].copy()
            selected['range_reduction_pct']=selected.day_range_reduction/selected.baseline_day_range*100
            table=selected.pivot(index='share',columns='max_extra_delay_h',values='range_reduction_pct').sort_index()
            ax=axes[row,col]
            heatmap(ax,table.to_numpy(),[f'{v*100:.0f}%' for v in table.index],table.columns,
                '日内峰谷差降低（%）','.0f',vmin=0,vmax=100)
            ax.set_title(f'{purpose}｜r={multiple:g}')
            ax.set_xlabel('最大额外推迟（h）')
            if col==0:
                ax.set_ylabel('参与份额')
    fig.suptitle('可分割任务的物理平坦化上界（2025，按资源小时核算）',fontsize=14)
    save(fig,output/'figure4-flexibility-potential',records)
    fig,axes=plt.subplots(4,3,figsize=(14,13),layout='constrained')
    for row,purpose in enumerate(('HPC','SAI','TRN','INF')):
        for col,multiple in enumerate((1.,1.5,2.)):
            selected=frame.loc[frame.purpose.eq(purpose)&frame.capacity_multiple.eq(multiple)].copy()
            selected['removable_pct']=selected.mean_independent_removable_resource_h*100
            table=selected.pivot(index='share',columns='max_extra_delay_h',values='removable_pct').sort_index()
            ax=axes[row,col]
            heatmap(ax,table.to_numpy(),[f'{v*100:.0f}%' for v in table.index],table.columns,
                '独立移出上界的年均值（基准容量的%）','.0f',vmin=0,vmax=100)
            ax.set_title(f'{purpose}｜r={multiple:g}')
            ax.set_xlabel('最大额外推迟（h）')
            if col==0:
                ax.set_ylabel('参与份额')
    fig.suptitle('逐小时独立移出上界的年均值（2025；各小时极值不能同时实现）',fontsize=14)
    save(fig,output/'figure5-independent-hour-removal',records)


def main(args):
    project=args.project.resolve()
    output,processed=project/'results'/RUN,project/'data/processed'/RUN
    font=setup()
    cities={c['city_id']:c for c in json.loads((project/'analysis/eight_city_locations.json').read_text(encoding='utf-8'))}
    scenarios=pd.read_csv(output/'scenario-names.csv').query("status == '可计算'").scenario_scale_id.tolist()
    benchmark=pd.read_csv(args.benchmarks/'city-benchmarks.csv')
    records=[]
    if not args.overview_only:
        load_patterns(output,processed,cities,scenarios,records)
    overview(output,processed,cities,scenarios,benchmark,records)
    flexibility_plot(output,records)
    report={'font':font,'created_utc':datetime.now(timezone.utc).isoformat(),'figures':records,
        'visual_review':'pending','all_png_at_least_300dpi':all(min(r['dpi'])>=299.9 for r in records)}
    (output/('figure-overview-validation.json' if args.overview_only else 'figure-validation.json')).write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--benchmarks',type=Path,required=True)
    parser.add_argument('--overview-only',action='store_true')
    main(parser.parse_args())
