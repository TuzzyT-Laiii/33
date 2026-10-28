import json

def quarantined_blocks(root, selection_sha, metadata_sha, blocks):
    path = root / 'data/metadata/buildability-20261025/weather-block-quarantine.json'
    record = json.loads(path.read_text()) if path.exists() else {'blocks': {}}
    if path.exists() and (record['selection_manifest_sha256'] != selection_sha
                         or record['source_metadata_sha256'] != metadata_sha):
        raise ValueError('隔离清单与冻结筛选或天气源版本不一致')
    for defect_path in sorted((root / 'data/metadata/buildability-20261025/weather-block-defects').glob('*.json')):
        defect = json.loads(defect_path.read_text())
        if (defect['selection_manifest_sha256'] != selection_sha or defect['source_metadata_sha256'] != metadata_sha
                or defect['status'] != 'awaiting_source_repair' or defect['reason'] != 'source_nan_values_verified'
                or defect['original_nan_count'] <= 0 or defect['counts_as_complete'] is not False
                or defect['label'] != defect_path.stem or defect_path.stem in record['blocks']):
            raise ValueError('动态隔离记录的来源、缺失证据或名称无效')
        record['blocks'][defect_path.stem] = defect
    planned = dict(blocks)
    pending = {}
    for label, item in record['blocks'].items():
        if (label not in planned or item['status'] not in ('awaiting_dewpoint_repair', 'awaiting_source_repair')
                or not item['variables'] or not set(item['variables']) <= {'d2m', 't2m', 'u100', 'v100', 'sp', 'ssrd', 'fdir'}
                or item['native_flat_indices'] != planned[label].native_flat_index.astype(int).tolist()):
            raise ValueError('隔离清单的状态、变量或授权节点不一致')
        proof_path = root / 'data/metadata/buildability-20261025/weather-cds-repairs' / (label + '.json')
        if proof_path.exists():
            proof = json.loads(proof_path.read_text())
            if proof['status'] == 'verified':
                if (proof['selection_manifest_sha256'] != selection_sha
                        or proof['source_metadata_sha256'] != metadata_sha
                        or proof['native_flat_indices'] != item['native_flat_indices']
                        or not set(item['variables']) <= set(proof.get('variables', ('u100', 'v100')))):
                    raise ValueError('隔离块补取证明与授权版本不同')
                continue
        pending[label] = item
    return pending
