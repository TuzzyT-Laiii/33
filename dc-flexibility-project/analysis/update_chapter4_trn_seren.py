import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa

from build_scenario_typical_days import save_json
from build_scenario_typical_days_v4 import split_layers
from compute_load.model import sha256
from load_curve_scenarios_v3 import catalogue
from plot_scenario_typical_days_v4 import plot_all
from reselect_scenario_typical_days import daily_metrics, select_day, audit_day, RULE, RULE_CAPTION


RUN = 'scenario-typical-day-by-task-20261101-v7'
PREVIOUS = 'scenario-typical-day-by-task-20261101-v6'
UPSTREAM = 'trn-seren-20261101'
COMMIT = 'bd6c8cdcb99a08b994136372881cee4a98d55d46'
TRN = ['B-TRN','C-TRN','D-TRN']
LABELS = {'Pretrain':'预训练','SFT':'SFT','MLLM':'MLLM（多模态大模型）','Eval':'评估','Debug':'调试','Other':'其他'}
CASES = ['main','idle50','idle70','idle_zero_proxy','clip_0_1','seren_constant_080']


def complete_days(frame):
    clock = pd.DatetimeIndex(pd.to_datetime(frame.source_clock,utc=True))
    assert clock.is_unique and clock.is_monotonic_increasing
    dates = clock.floor('d')
    available = pd.DataFrame({'source_day':dates,'hour':clock.hour}).groupby('source_day').agg(
        hours=('hour','size'),unique_hours=('hour','nunique'),first_hour=('hour','min'),last_hour=('hour','max'))
    available['complete'] = available.hours.eq(24) & available.unique_hours.eq(24) & available.first_hour.eq(0) & available.last_hour.eq(23)
    chosen = available.index[available.complete]
    return frame.loc[dates.isin(chosen)].reset_index(drop=True),chosen,available.reset_index()


def report(output,summary,comparisons,stats):
    selected = summary['selections']['TRN']
    lines = ['# 分任务典型日：Acme Seren同期功率', '', '![各情境总览](overview.png)', '',
        '参考IT 10MW；训练采用PR #38正文方案main，其他八情境的日期、单图和逐小时数据按v6原字节复制。', '', RULE, '',
        '## 正文窗口与选日', '',
        'Seren正文为UTC 2023-05-16 10:00至08-17 23:00之间的2057个合格观测小时。内部189小时缺口保持；不填零、不插值、不延续、不压缩时间，不采用补功率的完整4192小时对照。', '',
        f"仅55个具有00–23时全部观测的完整UTC日参与中位电量、全部日平均曲线及RMSE计算。GPU主导型参考选中{selected['source_date']}（{selected['weekday']}），±2%候选{selected['candidate_count_at_2pct']}天，实际±{selected['applied_window_pct']:g}%候选{selected['candidate_count']}天；三种规格共用日期和任务构成。", '',
        'TRN-day-availability.csv保存每个源日的观测数量和完整标记；TRN-day-selection.csv保存55个完整日的电量、偏差、RMSE、候选及选中标记。评分使用设施功率，RMSE参照全部完整日的逐小时平均设施曲线。', '',
        '## 功率与任务分层', '',
        f'- PR #38 `{COMMIT}`，采用同期逐小时Seren实测条件反推比例，目标CPU、基础空载、设备数及冷却沿用#36，按源月日小时对应参考网格2025天气。只读下载汇总小时文件，逐份匹配上游calculation-audit.json的SHA-256。',
        '- 每15秒按报送GPU均功率×2288外推全群，再扣除未分配GPU×60W空闲并除以分配GPU×400W，聚合为同小时比例r(t)。依赖2288卡在线及缺测代表性假设，缺少作业—设备映射；不能将各类型功率解释为逐作业直接测量。',
        '- A100-SXM 400W的比例迁移到B/C H100 PCIe 350W、D H100 SXM 700W；未保持同等吞吐、训练进度或运行时长。CPU代理为0.25×GPU分配份额；GPU需求按完整表瞬时峰值2032卡归一化。',
        '- 源GPU采样时间加权均值0.762316、P10/中位/P90=0.613946/0.768276/0.896305；观测范围0.309131–1.062429，8小时超过1，原值及标记保留。正文没有额外日形状或星期修正。',
        '- 敏感性：源空闲50/70W、零利用率代理67.300W、覆盖75%/85%、比例截断0–1；其中覆盖阈值会改变可用小时集合。目标GPU空闲仍50/70W，不随源反推假设一起改变。所选日完整覆盖的五个敏感性/对照方案另存sensitivity-hourly.csv及sensitivity-summary.csv，中央方案选日固定。',
        '- 分任务GPU小时直接取作业开始/结束与小时重叠的累计值，加入MLLM（多模态大模型）；同小时CPU/GPU动态增量按各类别GPU份额分摊。基础空载IT与冷却等非IT单列，每小时分别与IT及设施总功率守恒。', '',
        '## 典型日新旧对照', '',
        '旧列为v6 Kalos所选典型日，新列为Seren正文所选典型日；来源、窗口、归一化、任务构成均变化，差值为描述性比较。日内最高/最低比取所选日24小时，区别于上游按源小时分组的期间均值比。', '',
        '| 大模型训练/开发规格 | 日期，旧→新 | 设施峰值MW | 设施谷值MW | 最高/最低比 | 日电量MWh |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in comparisons:
        lines.append(f"| {row['scenario_name']} | {row['old_source_date']} → {row['new_source_date']} | {row['old_peak_mw']:.3f} → {row['new_peak_mw']:.3f} | {row['old_minimum_mw']:.3f} → {row['new_minimum_mw']:.3f} | {row['old_peak_valley_ratio']:.3f} → {row['new_peak_valley_ratio']:.3f} | {row['old_daily_facility_mwh']:.2f} → {row['new_daily_facility_mwh']:.2f} |")
    lines += ['', '## 简表统计口径', '',
        'seren-job-statistics.json/csv为重新只读扫描原始818327行作业表得到的六类GPU作业汇总。GPU作业450590条，独立CPU367737条。GPU小时、排队、运行时长和卡数均按原简表框架：有效执行时间end−start且排队start−submit非负；分位数按作业数线性插值；卡数取P50/P90。原始duration与执行时长一致，负排队为零。', '',
        '简表第二章占比沿用完整GPU作业表分母5626519.861 GPU小时；正文2057小时的构成另列，避免混用两个窗口。H与恢复成本为研究假设，运行形态或业务标签不直接赋予调度权限。', '',
        '| 类别 | 完整表GPU小时% | 正文GPU小时% | 典型日GPU小时% | 排队P50/P90秒 | 时长P50/P90秒 | 卡数P50/P90 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    composition = next(row for row in summary['scenarios'] if row['scenario']=='D-TRN')['composition']
    for row in stats['statistics']:
        lines.append(f"| {LABELS[row['class']]} | {100*row['work_fraction_of_resource_pool']:.2f} | {100*row['qualified_gpu_share']:.2f} | {100*composition[LABELS[row['class']]]:.2f} | {row['wait_s_p50']:g}/{row['wait_s_p90']:g} | {row['runtime_s_p50']:g}/{row['runtime_s_p90']:g} | {row['units_p50']:g}/{row['units_p90']:g} |")
    lines += ['', '## 复现与核验', '',
        '```powershell', "$env:OPENBLAS_NUM_THREADS='1'", "$env:OMP_NUM_THREADS='1'", 'python dc-flexibility-project/analysis/update_chapter4_trn_seren.py', '```', '',
        '本机data/processed/'+RUN+'/upstream/<方案>/保存服务器results/trn-seren-20261101/hourly/reference-grid的18份10MW汇总文件。seren-job-statistics.json由summarize_seren_jobs_for_brief.py在服务器只读执行，输入原始作业表SHA-256与#38一致；逐作业结果不输出。', '',
        'period-summary.csv区分TRN正文2057个实际小时与其他八情境历史全年范围，按各自每24保留小时均值检查电量数量级；TRN不外推全年。selected-day-diagnostics.csv保存三个新日提示，旧版结果全部保留。verify_chapter4_trn_seren.py核对来源、选日、分层、统计及Word/PDF。', '',
        '简表一至四章、README数字来源和重新生成说明同步更新。其他来源及敏感性见[v6](../'+PREVIOUS+'/README.md)和[Seren正文报告](../'+UPSTREAM+'/README.md)。服务器原件、MIT进程及STOP未修改。', '']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')


def build(project):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    previous,output,local = project/'results'/PREVIOUS,project/'results'/RUN,project/'data/processed'/RUN
    output.mkdir(parents=True,exist_ok=True)
    inputs = []
    def track(path):
        inputs.append({'file':str(path.resolve()),'sha256':sha256(path)})
        return path
    def read_json(path):
        return json.loads(track(path).read_text('utf-8'))
    summary = read_json(previous/'summary.json')
    before = {row['scenario']:row.copy() for row in summary['scenarios']}
    upstream = project/'results'/UPSTREAM
    audit = read_json(upstream/'calculation-audit.json')
    source_audit = read_json(upstream/'source-audit.json')
    stats = read_json(local/'seren-job-statistics.json')
    assert stats['job_sha256'] == source_audit['job_sha256']
    assert stats['script_sha256'] == sha256(project/'analysis/summarize_seren_jobs_for_brief.py')
    assert stats['source_hourly_sha256'] == sha256(upstream/'source-hourly.csv') == audit['source_hourly_sha256']
    np.testing.assert_allclose(stats['full_gpu_hours'],source_audit['full_gpu_hours'],rtol=1e-12)
    shutil.copyfile(local/'seren-job-statistics.json',output/'seren-job-statistics.json')
    pd.DataFrame(stats['statistics']).to_csv(output/'seren-job-statistics.csv',index=False,encoding='utf-8-sig')
    configs = catalogue(project)
    ratios = read_json(project/'results/typical-load-curves-20261029-v2/mit-power-ratio-summary.json')['transfer_distribution']
    mit = read_json(project/'results/typical-load-curves-20261101-v3/mit-calibration.json')
    cpu_shape = read_json(project/'results/trn-power-model-20261101/cpu-shape.json')
    curves,variants,hashes,checks = {},{}, {},{}
    for key in ('B','C','D'):
        variants[key] = {}
        for case in CASES:
            path = track(local/'upstream'/case/f'{key}-TRN-10MW-2025.parquet')
            assert sha256(path) == audit['hourly_sha256'][f'hourly/reference-grid/{case}/{path.name}']
            hashes[str(path.resolve())] = sha256(path)
            variants[key][case] = pd.read_parquet(path)
        curves[key] = variants[key]['main']
        assert len(curves[key]) == 2057 and not curves[key].power_imputed.any()
        checks[key+'_Seren_hourly_upstream_fingerprints_no_imputation'] = True
    complete,dates,available = complete_days(curves['D'])
    day,scores,selection = select_day(daily_metrics(complete,len(dates)))
    scores.insert(0,'source_day',dates.strftime('%Y-%m-%d'))
    scores.to_csv(output/'TRN-day-selection.csv',index=False,encoding='utf-8-sig')
    available.to_csv(output/'TRN-day-availability.csv',index=False,encoding='utf-8-sig')
    date = dates[day]
    source_date,weekday = date.strftime('%Y-%m-%d'),'星期'+'一二三四五六日'[date.dayofweek]
    selection.update(day_index=day,source_date=source_date,weekday=weekday,qualified_hours=2057,
        reference_scenario='D-TRN',reference_scenario_name=before['D-TRN']['scenario_name'],
        model_weather_date=source_date.replace('2023','2025'))
    summary['selections']['TRN'] = selection
    summary.pop('TRN_constant_080_selection',None)
    preserved = {}
    for row in summary['scenarios']:
        if row['scenario'] in TRN:
            continue
        for suffix in ('-hourly.csv','-by-task.csv','.png','.svg'):
            name = row['scenario']+suffix
            shutil.copyfile(previous/name,output/name)
            preserved[name] = sha256(output/name)
    for name in ('HPC-day-selection.csv','CLD-day-selection.csv','SAI-day-selection.csv','INF-day-selection.csv','idle-endpoints.csv'):
        shutil.copyfile(previous/name,output/name)
        preserved[name] = sha256(output/name)
    comparison,diagnostics,sensitivity,sensitivity_hours = [],[],[],[]
    for row in summary['scenarios']:
        scenario = row['scenario']
        if scenario not in TRN:
            continue
        key,full = scenario[0],curves[scenario[0]]
        clock = pd.DatetimeIndex(pd.to_datetime(full.source_clock,utc=True))
        frame = full.loc[clock.floor('d')==date].reset_index(drop=True)
        assert pd.DatetimeIndex(pd.to_datetime(frame.source_clock,utc=True)).hour.tolist()==list(range(24))
        resource = pd.DataFrame({prefix+label:frame['gpu_hours_'+kind]/2032*factor
            for kind,label in LABELS.items() for prefix,factor in [('gpu:',1),('cpu:',.25)]})
        amounts = resource.filter(regex='^gpu:').sum()
        composition = {name[4:]:float(value/amounts.sum()) for name,value in amounts.items()}
        layers = split_layers(frame,resource,configs[key],key,'TRN',ratios,mit,cpu_shape)
        fields = {'scenario':scenario,'scenario_name':row['scenario_name'],'source_date':source_date,'weekday':weekday}
        pd.DataFrame({**fields,'hour':range(24),'model_time_utc':frame.time_utc,'reference_it_mw':10.,
            'installed_reference_it_mw':frame.installed_reference_it_mw,'allocated_gpu_power_ratio':frame.allocated_gpu_power_ratio,
            'it_mw':frame.it_mw,'facility_mw':frame.facility_mw,'pue':frame.pue,
            'power_ratio_above_tdp':frame.power_ratio_above_tdp,'source_quality_flag':'Seren_qualified_observation'}).to_csv(output/(scenario+'-hourly.csv'),index=False,encoding='utf-8-sig')
        pd.concat([pd.DataFrame({**fields,'hour':range(24),'task_type':label,'it_mw':np.zeros(24) if label=='冷却等非 IT' else values,
            'facility_mw':values,'component_scope':'non_IT' if label=='冷却等非 IT' else 'IT'}) for label,values in layers.items()]).to_csv(output/(scenario+'-by-task.csv'),index=False,encoding='utf-8-sig')
        complete_key,key_dates,_ = complete_days(full)
        assert key_dates.equals(dates)
        diagnosis = audit_day(complete_key,daily_metrics(complete_key,len(dates)),day)
        diagnostics.append({**fields,**diagnosis,'above_tdp_hours':int(frame.power_ratio_above_tdp.sum())})
        row.update(**fields,complete_days=len(dates),peak_mw=float(frame.facility_mw.max()),minimum_mw=float(frame.facility_mw.min()),
            peak_valley_ratio=float(frame.facility_mw.max()/frame.facility_mw.min()),daily_it_mwh=float(frame.it_mw.sum()),
            daily_facility_mwh=float(frame.facility_mw.sum()),period_facility_mwh=float(full.facility_mw.sum()),period_hours=len(full),
            daily_to_period_mean_ratio=float(frame.facility_mw.sum()/full.facility_mw.mean()/24),composition=composition,
            model_weather_date=selection['model_weather_date'],task_share_of_daily_it=float((frame.it_mw-frame.idle_it_mw).sum()/frame.it_mw.sum()),unresolved_selected_hours=0)
        comparison.append({'scenario':scenario,'scenario_name':row['scenario_name'],'old_source_date':before[scenario]['source_date'],
            'new_source_date':source_date,**{version+'_'+name:record[name] for version,record in [('old',before[scenario]),('new',row)]
                for name in ('peak_mw','minimum_mw','peak_valley_ratio','daily_facility_mwh')}})
        for case,variant in variants[key].items():
            current = variant.loc[pd.DatetimeIndex(pd.to_datetime(variant.source_clock,utc=True)).floor('d')==date]
            assert len(current)==24
            sensitivity.append({**fields,'case':case,'hours':len(variant),'mean_it_mw':float(variant.it_mw.mean()),
                'mean_facility_mw':float(variant.facility_mw.mean()),'daily_facility_mwh':float(current.facility_mw.sum()),
                'peak_mw':float(current.facility_mw.max()),'minimum_mw':float(current.facility_mw.min()),
                'peak_valley_ratio':float(current.facility_mw.max()/current.facility_mw.min())})
            sensitivity_hours.append(pd.DataFrame({**fields,'case':case,'hour':range(24),'it_mw':current.it_mw.to_numpy(),'facility_mw':current.facility_mw.to_numpy()}))
        checks[scenario+'_24h_task_layer_balance_and_energy_scale'] = bool(.5<row['daily_to_period_mean_ratio']<2)
    facts = read_json(previous/'brief-facts.json')
    facts['SEREN'] = {row['class']:row for row in stats['statistics']}
    facts['power'] = {'TRN_source':'Acme Seren PR38 main','TRN_commit':COMMIT,
        'TRN_GPU_sampling_weighted_TDP_ratio':source_audit['central_power_statistics']['gpu_hour_weighted_mean_ratio'],
        'TRN_observed_ratio_p10':source_audit['central_power_statistics']['p10'],'TRN_observed_ratio_p90':source_audit['central_power_statistics']['p90'],
        'H100_SXM_training_GPU_kwh_per_card_hour':.7*source_audit['central_power_statistics']['gpu_hour_weighted_mean_ratio'],
        'H100_SXM_observed_p10_kwh':.7*source_audit['central_power_statistics']['p10'],
        'H100_SXM_observed_p90_kwh':.7*source_audit['central_power_statistics']['p90']}
    save_json(output/'brief-facts.json',facts)
    summary.update(data_version='Seren_PR38_main_plus_eight_v6_scenarios',training_commit=COMMIT,
        training_window='2057 qualified source UTC hour bins; only complete days enter selection',selection_rule=RULE)
    save_json(output/'summary.json',summary)
    pd.DataFrame([{**row,'composition':json.dumps(row['composition'],ensure_ascii=False)} for row in summary['scenarios']]).to_csv(output/'summary.csv',index=False,encoding='utf-8-sig')
    for name,rows in [('comparison',comparison),('selected-day-diagnostics',diagnostics),('sensitivity-summary',sensitivity)]:
        pd.DataFrame(rows).to_csv(output/(name+'.csv'),index=False,encoding='utf-8-sig')
    pd.concat(sensitivity_hours).to_csv(output/'sensitivity-hourly.csv',index=False,encoding='utf-8-sig')
    periods = pd.read_csv(previous/'period-summary.csv')
    for row in summary['scenarios']:
        if row['scenario'] not in TRN:
            continue
        mask = periods.scenario.eq(row['scenario'])
        periods.loc[mask,['period','hours','period_facility_mwh','mean_daily_facility_mwh','daily_to_period_mean_ratio']] = [
            'Seren_qualified_2057_hours',2057,row['period_facility_mwh'],row['period_facility_mwh']/2057*24,row['daily_to_period_mean_ratio']]
    periods.to_csv(output/'period-summary.csv',index=False,encoding='utf-8-sig')
    report(output,summary,comparison,stats)
    checks.update(plot_all(output,rule_caption=RULE_CAPTION,individual_scenarios=TRN,training_seren=True))
    assert all(checks.values())
    save_json(output/'validation.json',{'status':'passed','checks':checks})
    save_json(output/'manifest.json',{'utc':datetime.now(timezone.utc).isoformat(),'PR38_commit':COMMIT,'input_files':inputs,
        'upstream_hourly_sha256':hashes,'preserved_v6_files_sha256':preserved,
        'code_sha256':{name:sha256(project/'analysis'/name) for name in (Path(__file__).name,'summarize_seren_jobs_for_brief.py',
            'plot_scenario_typical_days_v3.py','plot_scenario_typical_days_v4.py','reselect_scenario_typical_days.py')}})
    print(json.dumps({'selection':selection,'comparison':comparison,'diagnostics':diagnostics,'checks':len(checks)},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    build(parser.parse_args().project)
