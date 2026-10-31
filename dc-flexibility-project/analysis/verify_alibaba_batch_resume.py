import argparse
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
import pandas as pd

from acquire_alibaba_cpu import now
from audit_alibaba_cpu import HOURS
from stream_alibaba_components import run


def verify(raw,archive,directory,output):
    if output.exists():
        raise FileExistsError("真实前缀验证需要独立新目录")
    output.parent.mkdir(parents=True,exist_ok=True)
    inventory=pd.read_csv(directory / "machine-inventory.csv")
    shutil.copyfile(directory / "machine-inventory.csv",output.parent / "machine-inventory.csv")
    run(raw,output,"batch_instance",chunk_rows=50000,stop_after_chunks=1,archive_path=archive)
    run(raw,output,"batch_instance",chunk_rows=50000,stop_after_chunks=1,archive_path=archive)
    with np.load(output / "batch_instance-checkpoint.npz") as saved:
        metadata=json.loads(str(saved["metadata"]))
        assert metadata["processed_rows"] == 100000
        actual={key:saved[key].copy() for key in ("cpu_core_hours","cpu_alternative_core_hours","covered_hours")}
    with tarfile.open(archive,"r:gz") as stream:
        with stream.extractfile(stream.next()) as csv:
            frame=pd.read_csv(csv,header=None,usecols=[5,6,7,10,11],nrows=100000)
    frame.columns=["start","end","machine_id","cpu_avg","cpu_max"]
    mapping=dict(zip(inventory.machine_id,range(len(inventory))))
    m=frame.machine_id.map(mapping).fillna(-1).to_numpy(int)
    start,end=frame.start.to_numpy(float),frame.end.to_numpy(float)
    avg,maximum=frame.cpu_avg.to_numpy(float),frame.cpu_max.to_numpy(float)
    valid=np.isfinite(start)&np.isfinite(end)&(start >= 0)&(end > start)&np.isfinite(avg)&(avg >= 0)&(m >= 0)
    expected={key:np.zeros_like(value) for key,value in actual.items()}
    for hour in range(HOURS):
        overlap=np.maximum(0,np.minimum(end,(hour+1)*3600)-np.maximum(start,hour*3600))/3600
        expected["cpu_core_hours"][:,hour]=np.bincount(m[valid],weights=overlap[valid]*avg[valid]/100,minlength=len(inventory))
        maxvalid=valid&np.isfinite(maximum)&(maximum >= 0)
        expected["cpu_alternative_core_hours"][:,hour]=np.bincount(m[maxvalid],weights=overlap[maxvalid]*maximum[maxvalid]/100,minlength=len(inventory))
        expected["covered_hours"][:,hour]=np.bincount(m[valid],weights=overlap[valid],minlength=len(inventory))
    for key in actual:
        np.testing.assert_allclose(actual[key],expected[key],rtol=1e-11,atol=1e-9)
    report={"verified_at_utc":now(),"input":"真实 batch_instance 压缩包前 100000 行","stop_then_resume":True,
            "resume_from_rows":50000,"verified_total_rows":100000,
            "independent_method":"逐小时直接求 min/max 区间重叠，与生产脚本边界分摊及差分累计比较",
            "columns_verified":list(actual),"status":"passed"}
    (directory / "batch-resume-validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="用真实实例前缀独立核对跨小时积分及恢复累计")
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    verify(args.raw,args.archive,args.directory,args.output)
