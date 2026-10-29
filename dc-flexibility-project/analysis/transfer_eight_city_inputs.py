import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

import pandas as pd


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def run(args):
    index=pd.read_csv(args.index)
    records=[]
    for row in index.itertuples():
        if Path(row.filename).name!=row.filename:
            raise ValueError('文件索引只接受当前目录文件名')
        path=args.directory/row.filename
        if digest(path)!=row.sha256 or path.stat().st_size!=row.bytes:
            raise ValueError('传输文件SHA或字节数不同：'+row.filename)
        records.append({'filename':row.filename,'sha256':row.sha256,'bytes':row.bytes})
    if len(records)!=96:
        raise ValueError('需要96份完整城市×情境输入')
    if args.mode=='pack':
        with tarfile.open(args.archive,'w') as archive:
            for row in records:
                archive.add(args.directory/row['filename'],arcname=row['filename'],recursive=False)
        print(json.dumps({'archive':str(args.archive),'bytes':args.archive.stat().st_size,'files':len(records)}))
    else:
        result={'created_utc':datetime.now(timezone.utc).isoformat(),'directory':str(args.directory),
            'index_sha256':digest(args.index),'files':len(records),'total_bytes':sum(r['bytes'] for r in records),
            'all_sha256_and_bytes_matched':True,'file_records':records}
        args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'files':len(records),'all_sha256_and_bytes_matched':True}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('pack','verify'))
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--report',type=Path)
    run(parser.parse_args())
