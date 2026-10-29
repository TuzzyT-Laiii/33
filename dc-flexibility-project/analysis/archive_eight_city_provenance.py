import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from compute_load.model import sha256


def archive(args):
    output = args.project/'results/eight-city-phase-a-20261029'
    metadata = output/'source-metadata'
    metadata.mkdir(exist_ok=True)
    cities=json.loads((args.project/'analysis/eight_city_locations.json').read_text(encoding='utf-8'))
    records=[]
    for city in cities:
        name=city['city_id']
        path=args.weather/(name+'-arco.parquet')
        validation_path=args.weather/(name+'-arco.validation.json')
        validation=json.loads(validation_path.read_text(encoding='utf-8'))
        if sha256(path)!=validation['parquet_sha256']:
            raise ValueError('天气Parquet SHA不一致：'+name)
        record={'city_id':name,'arco_parquet_sha256':sha256(path),'validation':validation}
        for suffix in ('-arco.source.json','-frozen.source.json','.request.json','.response.json','.validation.json'):
            source=args.weather/(name+suffix)
            if source.exists():
                shutil.copyfile(source,metadata/source.name)
        request=args.weather/(name+'.request.json')
        record['cds_request_id']=json.loads(request.read_text())['request_id'] if request.exists() else None
        cds=args.weather/(name+'.parquet')
        if cds.exists():
            first,second=pd.read_parquet(cds),pd.read_parquet(path)
            if not first.time_utc.equals(second.time_utc):
                raise ValueError('CDS/ARCO时间戳不同')
            errors=(first.drop(columns='time_utc')-second.drop(columns='time_utc')).abs().max()
            record['cds_vs_arco_max_absolute_difference']=errors.to_dict()
            record['cds_vs_arco_all_nine_variables_identical']=bool(np.max(errors.to_numpy())==0)
        records.append(record)
    for source in (args.benchmarks/'source-receipts.json',args.benchmarks/'city-benchmarks.csv',
        args.weather/'grid-land-check.json'):
        shutil.copyfile(source,metadata/source.name)
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'cities':records,
        'raw_server_directory':'/data3/dataset/ljt/eight-city-weather-20261029',
        'arco_primary_variables':{'frozen':['t2m','d2m','sp','ssrd','fdir','u100','v100'],'current':['u10','v10']},
        'cds_request_policy':'sequential; shared_global_gate; maximum_one_own_active_request; reuse_saved_ids',
        'arco_reason':'official_backing_store_used_while_shared_CDS_submission_gate_occupied'}
    (output/'weather-provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--weather',type=Path,required=True)
    parser.add_argument('--benchmarks',type=Path,required=True)
    archive(parser.parse_args())
