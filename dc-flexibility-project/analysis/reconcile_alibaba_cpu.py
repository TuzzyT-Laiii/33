import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from acquire_alibaba_cpu import now, sha256
from audit_alibaba_cpu import END, HOURS, intervals_by_hour, read_table
from stream_alibaba_components import CHECKSUMS


def build(directory):
    parts = directory / "components"
    states = {}
    for name, filename in (("container_usage","online-native-status.json"),("batch_instance","batch_instance-status.json")):
        state=json.loads((parts / filename).read_text(encoding="utf-8"))
        if state["status"] != "complete" or state["sha256"] != CHECKSUMS[name]:
            raise ValueError(f"大表尚未完成官方校验：{name}")
        states[name]=state
    hourly=pd.read_csv(directory / "alibaba-hourly-cpu-demand.csv")
    machine=pd.read_parquet(directory / "machine-hourly.parquet")
    online=pd.read_parquet(parts / "container_usage-machine-hourly.parquet")
    batch=pd.read_parquet(parts / "batch_instance-machine-hourly.parquet")
    joined=machine.merge(online,on=["machine_id","relative_hour"],validate="one_to_one").merge(batch,on=["machine_id","relative_hour"],validate="one_to_one")
    if len(joined) != len(machine):
        raise ValueError("逐机器小时连接丢失数据")
    joined=joined.rename(columns={"cpu_core_hours":"batch_mean_core_hours","cpu_alternative_core_hours":"batch_max_field_integral_core_hours",
                                  "covered_hours":"batch_instance_hours","request_based_core_hours":"online_request_based_core_hours",
                                  "limit_based_core_hours":"online_limit_based_core_hours","one_core_based_core_hours":"online_one_core_based_core_hours"})
    joined["residual_core_hours"]=joined.occupied_core_hours-joined.online_request_based_core_hours-joined.batch_mean_core_hours
    joined.to_parquet(directory / "machine-component-reconciliation.parquet",index=False)
    quantities=["online_request_based_core_hours","online_limit_based_core_hours","online_one_core_based_core_hours",
                "batch_mean_core_hours","batch_max_field_integral_core_hours","batch_instance_hours","sample_container_hours","residual_core_hours"]
    sums=joined.groupby("relative_hour")[quantities].sum().reset_index()
    result=hourly.merge(sums,on="relative_hour",validate="one_to_one")
    events=read_table(directory.parents[1] / "data/raw/alibaba-cpu-2018","container_meta").sort_values(["container_id","timestamp"])
    events["end"]=events.groupby("container_id").timestamp.shift(-1).fillna(END)
    started=events.status.eq("started")
    expected_active=intervals_by_hour(events.loc[started,"timestamp"],events.loc[started,"end"],np.ones(started.sum()))
    if len(result) != HOURS:
        raise ValueError("对账小时数量与源窗口不符")
    result["online_started_container_hours"]=expected_active
    result["online_sample_support_ratio"]=result.sample_container_hours/result.online_started_container_hours.where(result.online_started_container_hours.gt(0))
    for part,column in (("online","online_request_based_core_hours"),("batch","batch_mean_core_hours"),("residual","residual_core_hours")):
        result[f"{part}_share_of_machine"]=result[column]/result.occupied_core_hours.where(result.occupied_core_hours.gt(0))
    result["online_share_of_measured_components"]=result.online_request_based_core_hours/(result.online_request_based_core_hours+result.batch_mean_core_hours)
    result["batch_share_of_measured_components"]=1-result.online_share_of_measured_components
    result["residual_limit_basis_core_hours"]=result.occupied_core_hours-result.online_limit_based_core_hours-result.batch_mean_core_hours
    result["residual_one_core_basis_core_hours"]=result.occupied_core_hours-result.online_one_core_based_core_hours-result.batch_mean_core_hours
    result.to_csv(directory / "component-reconciliation.csv",index=False)
    selected=result.loc[result.complete_day]
    selected.to_csv(directory / "complete-component-window.csv",index=False)
    selected.groupby("hour_of_day")[quantities+["occupied_core_hours"]].mean().to_csv(directory / "component-intraday.csv")
    result.groupby("relative_day")[quantities+["occupied_core_hours"]].sum().to_csv(directory / "component-relative-days.csv")
    histogram=pd.read_csv(parts / "batch-instance-duration-histogram.csv")
    cumulative=histogram.instances.cumsum().to_numpy()
    duration_quantiles={f"p{int(q*100):02d}":int(histogram.duration_seconds.iloc[np.searchsorted(cumulative,q*cumulative[-1])]) for q in (.05,.25,.5,.75,.9,.95,.99)}
    edges=[0,1,10,60,300,1800,3600,21600,86400,np.inf]
    histogram["duration_bin"]=pd.cut(histogram.duration_seconds,bins=edges,right=True)
    binned=histogram.groupby("duration_bin",observed=False).instances.sum().reset_index()
    binned["instance_fraction"]=binned.instances/binned.instances.sum()
    binned.to_csv(directory / "batch-duration-bins.csv",index=False)
    totals=selected[["occupied_core_hours",*quantities]].sum()
    fractions={part:float(totals[column]/totals.occupied_core_hours) for part,column in (
        ("online","online_request_based_core_hours"),("batch_candidate","batch_mean_core_hours"),("unreconciled","residual_core_hours"))}
    sensitivity=[]
    for basis in ("request","limit","one_core"):
        column=f"online_{basis}_based_core_hours"
        residual=selected.occupied_core_hours-selected[column]-selected.batch_mean_core_hours
        sensitivity.append({"basis":basis,"online_core_hours":selected[column].sum(),"residual_core_hours":residual.sum(),
                            "negative_residual_cluster_hours":int(residual.lt(0).sum()),"minimum_cluster_residual_core_hours":residual.min()})
    pd.DataFrame(sensitivity).to_csv(directory / "online-unit-sensitivity.csv",index=False)
    audit={"built_at_utc":now(),"scope":"六表汇总；完整窗口和整机年度曲线使用 machine_usage，业务拆分独立保留差额",
           "large_table_states":states,"common_complete_hours":len(selected),"common_relative_days":sorted(selected.relative_day.unique().tolist()),
           "common_window_core_hours":{key:float(value) for key,value in totals.items()},"shares_of_machine":fractions,
           "negative_residual_cluster_hours_complete_window":int(selected.residual_core_hours.lt(0).sum()),
           "negative_residual_machine_hours_complete_window":int(joined.loc[joined.relative_hour.isin(selected.relative_hour),"residual_core_hours"].lt(0).sum()),
           "online_sample_support_ratio_complete_window":float(selected.sample_container_hours.sum()/selected.online_started_container_hours.sum()),
           "online_sample_support_ratio_hour_min":float(selected.online_sample_support_ratio.min()),
           "online_sample_support_ratio_hour_max":float(selected.online_sample_support_ratio.max()),
           "batch_duration_quantiles_seconds":duration_quantiles,"batch_duration_histogram_instances":int(cumulative[-1]),
           "online_percent_denominator":"2018 维护者 issue #62 明确为容器请求 CPU 数量的百分比；正式结果为 cpu_util_percent×cpu_request/10000 核。limit 和单核百分数作为敏感性诊断。",
           "online_percent_authority":"https://github.com/alibaba/clusterdata/issues/62#issuecomment-476964315",
           "online_sampling":"不移动时间戳；每条有效样本按源 10 秒时间格累计。缺测时段不插值，原始小时平均占用有采样支持边界。",
           "batch_sampling":"cpu_avg/100 在原始 start—end 区间恒定积分；小时内部形状未知。cpu_max 字段恒定积分仅为诊断量。",
           "flexibility":"在线刚性、离线可推迟是用户指定情境假设。离线执行量仅为柔性候选；未得到提交时间、业务期限、抢占/检查点权限，不能直接作为全部可移负荷。",
           "residual":"保留有符号差额；可能含系统开销、日志缺口、计量窗口/CPU单位差异、实例均值分摊误差，无法全部归为系统开销或在线服务。"}
    (directory / "component-audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    source_path=directory / "source-audit.json"
    source_audit=json.loads(source_path.read_text(encoding="utf-8"))
    source_audit.pop("online_actual_cpu_share",None)
    source_audit.pop("batch_actual_cpu_share",None)
    source_audit["component_audit"]="component-audit.json"
    source_audit["share_limit"]="请求/计划包络仅为配置诊断；在线实际监测与实例平均 CPU 积分、单位解释及有符号差额见 component-audit.json；可推迟权限需独立确认。"
    source_audit["code_sha256"]["reconcile_alibaba_cpu.py"]=sha256(__file__)
    source_path.write_text(json.dumps(source_audit,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"shares":fractions,"duration_quantiles_seconds":duration_quantiles},ensure_ascii=False))
    return audit


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="核对 Alibaba 在线、离线、整机 CPU 汇总并保留有符号差额")
    parser.add_argument("--directory",type=Path,required=True)
    build(parser.parse_args().directory)
