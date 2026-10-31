import argparse
import io
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
import pandas as pd

from acquire_alibaba_cpu import now
from audit_alibaba_cpu import END, HOURS, read_table
from stream_alibaba_components import Forward
from stream_alibaba_online import run


def verify(raw,archive,directory,output):
    if output.exists():
        raise FileExistsError("真实前缀验证需要新的独立目录")
    output.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(directory / "machine-inventory.csv",output.parent / "machine-inventory.csv")
    run(raw,output,stop_after_chunks=1,archive_path=archive)
    run(raw,output,stop_after_chunks=1,archive_path=archive)
    with np.load(output / "online-exact-checkpoint.npz") as saved:
        metadata=json.loads(str(saved["metadata"]))
        assert metadata["processed"] == 4000000
        actual={key:saved[key].copy() for key in ("request_based_core_hours","limit_based_core_hours","one_core_based_core_hours","sample_container_hours")}
    with tarfile.open(archive,"r:gz") as stream:
        member=stream.next()
        with stream.extractfile(member) as csv:
            frame=pd.read_csv(io.BufferedReader(Forward(csv)),header=None,usecols=[0,1,2,3],nrows=4000000)
    frame.columns=["container_id","machine_id","timestamp","cpu"]
    frame=frame.loc[frame.container_id.notna()&frame.machine_id.notna()].copy()
    frame.timestamp=frame.timestamp.astype(np.int64)
    meta=read_table(raw,"container_meta").rename(columns={"machine_id":"meta_machine"})
    joined=pd.merge_asof(frame.sort_values("timestamp",kind="stable"),meta.sort_values("timestamp",kind="stable"),on="timestamp",by="container_id",direction="backward")
    good=joined.timestamp.ge(0)&joined.timestamp.lt(END)&joined.cpu.between(0,100)&joined.cpu_request.ge(0)&joined.cpu_limit.ge(0)&joined.machine_id.eq(joined.meta_machine)
    joined=joined.loc[good].copy()
    joined=joined.drop_duplicates(["container_id","timestamp"],keep="first")
    joined["relative_hour"]=(joined.timestamp//3600).astype(int)
    joined["request_based_core_hours"]=joined.cpu*joined.cpu_request/10000*10/3600
    joined["limit_based_core_hours"]=joined.cpu*joined.cpu_limit/10000*10/3600
    joined["one_core_based_core_hours"]=joined.cpu/100*10/3600
    joined["sample_container_hours"]=10/3600
    expected=joined.groupby(["machine_id","relative_hour"])[list(actual)].sum()
    inventory=pd.read_csv(directory / "machine-inventory.csv")
    index=pd.MultiIndex.from_product([inventory.machine_id,range(HOURS)],names=["machine_id","relative_hour"])
    expected=expected.reindex(index,fill_value=0)
    for key in actual:
        np.testing.assert_allclose(actual[key].ravel(),expected[key].to_numpy(),rtol=1e-11,atol=1e-9)
    report={"verified_at_utc":now(),"input":"真实 container_usage 压缩包前 4000000 行",
            "stop_then_resume":True,"resume_from_rows":2000000,"verified_total_rows":4000000,
            "independent_method":"pandas merge_asof 连接事件与 groupby 求和，对比生产脚本 searchsorted 连接与 NumPy 累计",
            "columns_verified":list(actual),"status":"passed"}
    (directory / "online-resume-validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="以真实原文前缀验证在线累计及停止后恢复，避免依赖 mock")
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    verify(args.raw,args.archive,args.directory,args.output)
