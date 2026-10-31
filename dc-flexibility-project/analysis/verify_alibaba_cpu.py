import argparse
import json
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from acquire_alibaba_cpu import EXPECTED, now, sha256
from alibaba_cpu_profile import load_alibaba_source
from annual_task_profiles import replay_hourly
from compute_load.model import ComputeLoadModel


def verify(project, require_components=False, weather_path=None):
    directory=project / "results/alibaba-cpu-cld-20261031"
    metadata=project / "data/metadata/alibaba-cpu-cld-20261031"
    audit=json.loads((directory / "source-audit.json").read_text(encoding="utf-8"))
    download=json.loads((metadata / "download-audit.json").read_text(encoding="utf-8"))
    for name,expected in EXPECTED.items():
        assert download["files"][name]["sha256"] == expected
        assert download["files"][name]["official_sha256_matches"]
    for name,expected in audit["outputs_sha256"].items():
        assert sha256(directory / name) == expected, name
    hourly=pd.read_csv(directory / "alibaba-hourly-cpu-demand.csv")
    complete=pd.read_csv(directory / "complete-window.csv")
    daily=pd.read_csv(directory / "daily-coverage.csv")
    inventory=pd.read_csv(directory / "machine-inventory.csv")
    hist=pd.read_csv(directory / "cpu-sample-histogram.csv",index_col=0)
    assert inventory.cpu_num.sum() == audit["inventory_cores"]
    assert len(inventory) == audit["machines"]
    assert hourly.relative_hour.tolist() == list(range(192))
    assert complete.relative_day.unique().tolist() == audit["complete_days"]
    assert daily.loc[daily.complete_day,"relative_day"].tolist() == audit["complete_days"]
    assert complete.coverage_fraction.min() >= audit["minimum_hour_core_coverage"]
    assert int(hist.to_numpy().sum()) == audit["valid_cpu_unique_samples"]
    np.testing.assert_allclose(complete.resource_fraction,complete.occupied_core_hours/complete.observed_core_hours,atol=1e-12)
    sums=pd.read_parquet(directory / "machine-hourly.parquet").groupby("relative_hour")[["occupied_core_hours","observed_core_hours"]].sum()
    np.testing.assert_allclose(sums.occupied_core_hours,hourly.occupied_core_hours,atol=1e-6)
    np.testing.assert_allclose(sums.observed_core_hours,hourly.observed_core_hours,atol=1e-6)
    source=load_alibaba_source(project)
    annual={}
    for year,length in ((2025,8760),(2024,8784)):
        times=pd.date_range(f"{year}-01-01",f"{year+1}-01-01",freq="h",inclusive="left",tz="UTC")
        profile,mapping=replay_hourly(source,times)
        expected=complete.resource_fraction.to_numpy()[np.arange(length)%len(complete)]
        np.testing.assert_array_equal(profile.fraction,expected)
        saved=pd.read_csv(directory / f"A-CLD-{year}-replay-map.csv")
        np.testing.assert_allclose(saved.resource_fraction,expected,atol=1e-14)
        annual[year]={"hours":len(mapping),"source_hours":len(complete),"cycles":length//len(complete),"tail_hours":length%len(complete)}
    components=None
    if require_components:
        components=json.loads((directory / "component-audit.json").read_text(encoding="utf-8"))
        result=pd.read_csv(directory / "component-reconciliation.csv")
        np.testing.assert_allclose(result.occupied_core_hours,
            result.online_request_based_core_hours+result.batch_mean_core_hours+result.residual_core_hours,atol=1e-6)
        detailed=pd.read_parquet(directory / "machine-component-reconciliation.parquet")
        summary=detailed.groupby("relative_hour")[["online_request_based_core_hours","batch_mean_core_hours","residual_core_hours"]].sum()
        for column in summary:
            np.testing.assert_allclose(summary[column],result[column],atol=1e-6)
        assert np.isclose(sum(components["shares_of_machine"].values()),1)
        duration=pd.read_csv(directory / "components/batch-instance-duration-histogram.csv")
        assert duration.instances.sum() == components["batch_duration_histogram_instances"]
        assert (duration.duration_seconds > 0).all()
        for state in components["large_table_states"].values():
            assert state["status"] == "complete" and state["official_sha256_matches"]
            assert state["compressed_bytes_read"] == state["bytes_expected"]
    power=None
    if weather_path:
        weather=pd.read_csv(weather_path)
        profile,_=replay_hourly(source,pd.to_datetime(weather.time_utc,utc=True))
        model=ComputeLoadModel()
        cases=[]
        for size in (1,10,100,500):
            curve=model.calculate(weather,profile,size)
            assert len(curve) == 8760
            assert np.isfinite(curve[["it_mw","cooling_mw","facility_mw"]]).all().all()
            assert curve.it_mw.gt(0).all()
            cases.append({"reference_it_mw":size,"hours":len(curve),"annual_it_mwh":float(curve.it_mw.sum())})
        power={"purpose":"使用仓库既有功率接口及真实气象验证资源输入可调用；不作为目标 Lenovo 服务器的新功率标定",
               "weather_sha256":sha256(weather_path),"cases":cases}
    stats={"complete_hours":len(complete),"mean_cpu_pct":float(complete.resource_fraction.mean()*100),
        "minimum_cpu_pct":float(complete.resource_fraction.min()*100),"maximum_cpu_pct":float(complete.resource_fraction.max()*100),
        "peak_to_trough":float(complete.resource_fraction.max()/complete.resource_fraction.min()),
        "occupied_core_hours":float(complete.occupied_core_hours.sum()),"minimum_hour_core_coverage":float(complete.coverage_fraction.min()),
        "intraday":pd.read_csv(directory / "A-CLD-intraday.csv").to_dict("records"),
        "shape_comparison":pd.read_csv(directory / "shape-comparison.csv").replace({np.nan:None}).to_dict("records")}
    (directory / "report-statistics.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    validation={"verified_at_utc":now(),"status":"passed","scope":"四表哈希、机器小时加总、完整日、样本分布、平/闰年重放、业务拆分对账及实测气象接口",
                "annual":annual,"components_verified":require_components,"power_interface":power,"summary_statistics":stats,
                "environment":{"python":sys.version,"packages":{package:version(package) for package in ("numpy","pandas","pyarrow","requests","isal","numba","psychrolib")},
                               "interfaces_sha256":{name:sha256(project / name) for name in ("analysis/annual_task_profiles.py","analysis/compute_load/model.py")}}}
    (directory / "validation.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"status":"passed","annual":annual,"components_verified":require_components},ensure_ascii=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="从提交的小时汇总独立复算 A-CLD 关键数字和接口")
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-components",action="store_true")
    parser.add_argument("--weather",type=Path)
    args=parser.parse_args()
    verify(args.project,args.require_components,args.weather)
