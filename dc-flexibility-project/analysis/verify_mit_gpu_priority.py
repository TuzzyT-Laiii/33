import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from mit_gpu_common import atomic_json, database, identity, load_jobs, utc
from mit_gpu_priority import configure, overlaps, priority_progress, window_bounds


def fingerprint(db, query):
    digest = hashlib.sha256()
    for row in db.execute(query):
        digest.update((json.dumps(tuple(row)) + '\n').encode())
    return digest.hexdigest()


def verify(root, output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'state.sqlite'
    if path.exists():
        raise FileExistsError('验证使用新的空目录: ' + str(output))
    source = sqlite3.connect((root / 'state.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
    target = sqlite3.connect(path)
    source.backup(target)
    source.close()
    integrity = target.execute('PRAGMA integrity_check').fetchone()[0]
    if integrity != 'ok':
        raise ValueError('源数据库完整性检查失败: ' + integrity)
    sequence_query = 'SELECT seq,key,etag,bytes FROM objects ORDER BY seq'
    state_query = 'SELECT seq,status,attempts,reason,batch,finished FROM objects ORDER BY seq'
    before = {'sequence': fingerprint(target, sequence_query), 'state': fingerprint(target, state_query)}
    target.close()
    db = database(output)
    jobs = load_jobs(root)
    begin, finish = window_bounds('2021-06-01', '2021-07-01')
    expected = [row['seq'] for row in db.execute('SELECT * FROM objects ORDER BY seq')
                if overlaps(identity(row['key'], jobs)[2], begin, finish)]
    configure(db, jobs, '2021-06-01', '2021-07-01')
    actual = [r[0] for r in db.execute('SELECT seq FROM objects WHERE priority=1 ORDER BY seq')]
    assert expected == actual
    assert before['sequence'] == fingerprint(db, sequence_query)
    assert before['state'] == fingerprint(db, state_query)
    ordered = list(db.execute("SELECT seq,priority FROM objects WHERE status='pending' ORDER BY priority DESC,seq"))
    assert [(r['priority'], -r['seq']) for r in ordered] == sorted([(r['priority'], -r['seq']) for r in ordered], reverse=True)
    assert not any(db.execute("SELECT 1 FROM objects WHERE status='done' AND seq=?", (r['seq'],)).fetchone() for r in ordered)
    configure(db, jobs, '2021-06-01', '2021-07-01')
    assert before['state'] == fingerprint(db, state_query)
    for batch, in db.execute('SELECT DISTINCT batch FROM objects WHERE batch IS NOT NULL'):
        manifest = json.loads((root / 'batches' / batch / 'manifest.json').read_text(encoding='utf-8'))
        for row in manifest['objects']:
            current = db.execute('SELECT * FROM objects WHERE seq=?', (row['seq'],)).fetchone()
            assert (current['status'], current['batch'], current['etag'], current['bytes']) == (row['status'], batch, row['etag'], row['bytes'])
    report = {'verified_at': utc(), 'integrity_check': integrity, 'before': before,
        'priority_window': priority_progress(db), 'checks': ['旧库迁移', '原序号/对象指纹保持', '完成状态/尝试次数/批次保持',
        '窗口匹配', '组内原随机顺序', '已完成对象排除', '重复配置幂等', '完整批次清单与SQLite一致']}
    atomic_json(output / 'verification.json', report)
    db.close()
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(args.root, args.output)
