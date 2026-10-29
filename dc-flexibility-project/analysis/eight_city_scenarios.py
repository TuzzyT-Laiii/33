import json
from pathlib import Path

import pandas as pd


SERVER_TYPES = {
    'C20': ('A', '纯 CPU 型', '2:0', 'Lenovo SR650 V3'),
    'C21': ('B', 'CPU 主导型', '2:1', 'SR650 V3 + H100 PCIe'),
    'C22': ('C', '均衡型', '2:2', 'SR650 V3 + 2×H100 PCIe'),
    'C28': ('D', 'GPU 主导型', '2:8', 'DGX H100'),
    'R-GB200': ('E', '纯 GPU 型（整柜液冷）', '36:72', 'GB200 NVL72'),
}
PURPOSES = {
    'Lisa': ('HPC', '科研计算'), 'Alibaba': ('CLD', '云通用计算'),
    'MIT': ('SAI', '科研 AI'), 'Acme': ('TRN', '大模型训练/开发'), 'ASI': ('INF', '推理服务'),
}


def scenario_fields(configuration, demand):
    server, server_name, ratio, reference = SERVER_TYPES[configuration]
    purpose, purpose_name = PURPOSES[demand]
    scenario = server + '-' + purpose
    return {'scenario_id': scenario, 'scenario_scale_id': scenario + '-10MW',
        'scenario_name_zh': server_name + '·' + purpose_name, 'server_type': server,
        'server_type_zh': server_name, 'purpose': purpose, 'purpose_zh': purpose_name,
        'legacy_configuration_id': configuration, 'demand_source': demand,
        'cpu_gpu_count_ratio': ratio, 'reference_platform': reference}


def write_table(output):
    rows = []
    for configuration in SERVER_TYPES:
        demands = ('Lisa','Alibaba') if configuration == 'C20' else (
            ('Acme','ASI') if configuration == 'R-GB200' else ('MIT','Acme','ASI'))
        for demand in demands:
            rows.append({**scenario_fields(configuration, demand), 'reference_it_mw': 10,
                'status': '待数据' if demand == 'Alibaba' else '可计算',
                'cooling_type': 'direct_liquid' if configuration == 'R-GB200' else 'air'})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'scenario-names.csv', index=False, encoding='utf-8-sig')
    lines = ['# 服务器类型 × 用途情境命名', '',
        '情境名称由服务器类型和用途组成，规模后缀为 `-10MW`。数据来源保留在单独字段中。', '',
        '| 情境 | 服务器类型 | 用途 | CPU:GPU | 旧配置 | 需求来源 | 状态 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append(f'| {row["scenario_scale_id"]} | {row["server_type_zh"]} | {row["purpose_zh"]} | {row["cpu_gpu_count_ratio"]} | {row["legacy_configuration_id"]} | {row["demand_source"]} | {row["status"]} |')
    lines += ['', 'E 型的“纯 GPU 型（整柜液冷）”按用户指定表示 GPU 主导的机柜形态。GB200 NVL72 含 72 颗 GPU 和 36 颗 Grace CPU。',
        'NVIDIA DPS 0.8 设备模板给出每颗 Blackwell GPU 的 maxLoadWatts=1200；NVIDIA GB200 NVL 分区指南给出约120 kW整柜功率。',
        '据此核算 GPU 参考最大功率为72×1.2=86.4 kW，占120 kW整柜参考功率的72.0%；其余28.0%包含CPU、内存、网络、交换和供电等。',
        '72.0%是不同官方设计参数组合后的参考功率占比，随负载变化；年度功率分项继续按v2模型计算。Grace保留主机计算功能，液冷机柜仍包含空气冷却部件。', '',
        '- [NVIDIA DPS 0.8 设备参数](https://docs.nvidia.com/datacenter/dps/versions/0.8/guides/concepts/device-specifications/)',
        '- [NVIDIA GB200 NVL 分区指南](https://docs.nvidia.com/multi-node-nvlink-systems/partition-guide-v1-2.pdf)',
        '- [NVIDIA GB200 系统架构](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/system.html)', '']
    (output / 'scenario-names.md').write_text('\n'.join(lines), encoding='utf-8')
    return frame
