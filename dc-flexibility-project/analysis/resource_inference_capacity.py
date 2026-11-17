import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

CONTEXT = {}
BENCHMARK = {
    'source': 'https://nvidia.github.io/TensorRT-LLM/0.18.2/performance/perf-overview.html',
    'measured_release': 'TensorRT-LLM 0.17', 'model': 'Llama 3.1 8B',
    'gpu': 'H100 80GB HBM3', 'precision': 'FP8', 'tensor_parallel': 1,
    'input_tokens_per_request': 1000, 'output_tokens_per_request': 1000,
    'output_tokens_per_second': 15244.78,
    'load_condition': 'maximum throughput, infinite request rate, in-flight batching',
    'throughput_multipliers': [.25, .5, 1.],
    'scope': '标准请求服务能力情景；PCIe 型和真实响应延迟另受硬件与服务条件限制',
}


def prepare(args):
    sys.path[:0] = [str(args.capacity / 'analysis'), str(args.joint / 'analysis'),
                    str(args.project / 'analysis'), str(args.residual_code)]
    from joint_capacity_copt import resource_layers
    from capacity_plan import apply_whole_plan
    source = args.joint / 'inputs'
    manifest = json.loads((source / 'resources/resource-manifest.json').read_text())
    reference = SimpleNamespace(reference=source / 'reference')
    scenarios = {}
    for scenario, fleet in manifest.items():
        if not fleet['gpus_per_unit']:
            continue
        hours, rigid, whole, split = resource_layers(reference, scenario, fleet)
        with np.load(args.capacity / 'results/capacity' / f'{scenario}-w12-s48.dispatch.npz') as saved:
            certified = saved['resource_dispatch']
        fixed_whole = np.zeros_like(whole)
        if fleet['use'] != 'INF':
            frame = pd.read_parquet(source / 'cities' / f'frankfurt-{scenario}.parquet')
            np.testing.assert_array_equal(frame.source_hour_index, hours)
            jobs = pd.read_parquet(args.capacity / 'inputs/exact-jobs' / f'{scenario}.parquet')
            plan = pd.read_parquet(args.capacity / 'results/capacity' / f'{scenario}-w12-s48.whole-plan.parquet')
            fixed_whole = apply_whole_plan(np.column_stack((whole, frame.whole)), hours, jobs, plan)[:, :2]
        count = fleet['baseline_units']
        if certified.shape != rigid.shape or certified.max() > count + 1e-4:
            raise ValueError('容量调度资源数组无效')
        inf = pd.read_parquet(source / 'resources' / f'{scenario[0]}-INF.parquet')
        cpu_per_gpu = float(inf.cpu_core_hours.sum() / inf.gpu_hours.sum())
        host_ratio = cpu_per_gpu * fleet['gpus_per_unit'] / fleet['cores_per_unit']
        scenarios[scenario] = dict(hours=hours, rigid=rigid / count, whole=whole / count,
                                   split=split / count, certified=certified / count,
                                   fixed_whole=fixed_whole / count, use=fleet['use'],
                                   gpus=count * fleet['gpus_per_unit'], host_ratio=host_ratio)
    request = pd.read_csv(args.requests)
    request = request[request.series == 'burst-1+2'].sort_values('hour_index')
    np.testing.assert_array_equal(request.hour_index.to_numpy(), np.arange(24, 2880))
    CONTEXT.update(scenarios=scenarios, requests=request.requests.to_numpy(float))


def capacity(state, demand, offline_share, bound):
    from flex_econ_copt import backlog_limit
    from residual_flex_pilot import LP, solve_copt
    hours = state['hours']
    n = len(hours)
    idx = np.arange(n)
    block_count = 5 if bound == 'relaxed_upper' else 1
    lp = LP(n, 1)
    lp.count = 1 + block_count * 2 * n
    lower, upper = np.zeros(lp.count), np.full(lp.count, np.inf)
    objective = np.zeros(lp.count)
    objective[0] = -1.
    online, offline = demand * (1 - offline_share), demand * offline_share
    execution = 1 + idx
    backlog = 1 + n + idx
    # 外部请求量由同一个强度变量决定；离线服务须在真实经过时间 24 小时内完成。
    lp.block('eq', [(idx, execution, 1.), (idx, backlog, 1.),
                    (idx[1:], backlog[:-1], -1.), (idx, 0, -offline)], np.zeros(n))
    lp.block('ub', [(idx, backlog, 1.),
                    (idx, 0, -backlog_limit(offline, 24, hours))], np.zeros(n))
    resources = [[], []]
    fixed = state['certified'] if bound == 'feasible_lower' else state['rigid']
    if bound == 'relaxed_upper':
        for resource in range(2):
            for k, (arrival, deadline) in enumerate([(state['whole'][:, resource], 12),
                                                     (state['split'][:, resource], 48)]):
                block = 1 + 2 * resource + k
                x = 1 + block * 2 * n + idx
                b = x + n
                upper[b] = backlog_limit(arrival, deadline, hours)
                lp.block('eq', [(idx, x, 1.), (idx, b, 1.), (idx[1:], b[:-1], -1.)], arrival)
                resources[resource].append((idx, x, 1.))
    for resource, coefficient in enumerate([state['host_ratio'], 1.]):
        lp.block('ub', resources[resource] + [(idx, execution, coefficient),
                                             (idx, 0, coefficient * online)], 1. - fixed[:, resource])
    aeq, beq, aub, bub = lp.matrices()
    values, status = solve_copt(objective, aeq, beq, aub, bub, lower, upper)
    if values is None:
        raise RuntimeError(f'GPU 服务容量求解失败 {bound} {status}')
    residual = max(float(np.abs(aeq @ values - beq).max()),
                   float(np.maximum(aub @ values - bub, 0).max()),
                   float(np.maximum(lower - values, 0).max()),
                   float(np.maximum(values - upper, 0).max()))
    if residual > 1e-6:
        raise RuntimeError(f'GPU 服务容量残差超限 {residual}')
    return float(values[0]), residual


def coupled_capacity(state, demand, offline_share):
    from flex_econ_copt import backlog_limit
    from residual_flex_pilot import LP, solve_copt
    hours = state['hours']
    n = len(hours)
    idx = np.arange(n)
    fixed = state['rigid'] + state['fixed_whole']
    profiles = [(state['split'], 48)]
    if state['use'] == 'INF':
        profiles.append((state['whole'], 12))
    option_blocks, option_slots, option_values = [], [], []
    blocks = 0
    for profile, deadline in profiles:
        origins = np.flatnonzero(np.any(profile > 1e-12, axis=1))
        candidates = hours[origins, None] + np.arange(deadline + 1)
        targets = np.searchsorted(hours, candidates)
        valid = (targets < n) & (hours[targets.clip(max=n - 1)] == candidates)
        origin, delay = np.nonzero(valid)
        option_blocks.append(blocks + origin)
        option_slots.append(targets[origin, delay])
        option_values.append(profile[origins[origin]])
        blocks += len(origins)
    block = np.concatenate(option_blocks)
    slot = np.concatenate(option_slots)
    resource = np.concatenate(option_values)
    columns = 1 + 2 * n + np.arange(len(block))
    lp = LP(n, 1)
    lp.count = 1 + 2 * n + len(block)
    lower, upper = np.zeros(lp.count), np.full(lp.count, np.inf)
    upper[columns] = 1.
    objective = np.zeros(lp.count)
    objective[0] = -1.
    execution, backlog = 1 + idx, 1 + n + idx
    online, offline = demand * (1 - offline_share), demand * offline_share
    lp.block('eq', [(idx, execution, 1.), (idx, backlog, 1.),
                    (idx[1:], backlog[:-1], -1.), (idx, 0, -offline)], np.zeros(n))
    lp.block('ub', [(idx, backlog, 1.), (idx, 0, -backlog_limit(offline, 24, hours))], np.zeros(n))
    lp.block('eq', [(block, columns, 1.)], np.ones(blocks))
    for k, coefficient in enumerate([state['host_ratio'], 1.]):
        lp.block('ub', [(slot, columns, resource[:, k]), (idx, execution, coefficient),
                        (idx, 0, coefficient * online)], 1. - fixed[:, k])
    aeq, beq, aub, bub = lp.matrices()
    values, status = solve_copt(objective, aeq, beq, aub, bub, lower, upper)
    if values is None:
        raise RuntimeError(f'CPU/GPU 耦合容量求解失败 {status}')
    residual = max(float(np.abs(aeq @ values - beq).max()),
                   float(np.maximum(aub @ values - bub, 0).max()),
                   float(np.maximum(lower - values, 0).max()),
                   float(np.maximum(values - upper, 0).max()))
    if residual > 1e-6:
        raise RuntimeError(f'CPU/GPU 耦合容量残差超限 {residual}')
    return float(values[0]), residual


def one(task):
    scenario, phase, offline = task
    state = CONTEXT['scenarios'][scenario]
    position = state['hours'] - state['hours'][0] - phase
    mask = (position >= 0) & (position < len(CONTEXT['requests']))
    shape = np.zeros(len(position))
    shape[mask] = CONTEXT['requests'][position[mask].astype(int)]
    original = state['rigid'] + state['whole'] + state['split']
    idle_fraction_hours = float((1. - original[mask, 1]).sum())
    if idle_fraction_hours <= 0 or shape.sum() <= 0:
        raise ValueError('独立需求或 GPU 余量为空')
    demand = shape / shape.sum() * idle_fraction_hours
    low, low_res = capacity(state, demand, offline, 'feasible_lower')
    coupled, coupled_res = coupled_capacity(state, demand, offline)
    if coupled < low - 1e-6:
        raise RuntimeError('耦合调度未包含既有可行方案')
    low, low_res = max(low, coupled), max(low_res, coupled_res)
    high, high_res = capacity(state, demand, offline, 'relaxed_upper')
    if low > high + 1e-6:
        raise RuntimeError('可行服务量超过松弛上界')
    rate = BENCHMARK['output_tokens_per_second'] / BENCHMARK['output_tokens_per_request'] * 3600
    rows = []
    for multiplier in BENCHMARK['throughput_multipliers']:
        rows.append(dict(scenario=scenario, phase_hours=phase, offline_share=offline,
                         active_hours=int(mask.sum()), GPUs=state['gpus'],
                         throughput_multiplier=multiplier, standard_requests_per_GPU_hour=rate * multiplier,
                         idle_gpu_hours=idle_fraction_hours * state['gpus'],
                         feasible_share_of_idle=low, upper_share_of_idle=high,
                         capacity_proven=bool(high - low < 1e-6),
                         feasible_standard_requests=low * idle_fraction_hours * state['gpus'] * rate * multiplier,
                         upper_standard_requests=high * idle_fraction_hours * state['gpus'] * rate * multiplier,
                         feasible_trace_multiplier=low * idle_fraction_hours * state['gpus'] * rate * multiplier / shape.sum(),
                         max_residual=max(low_res, high_res)))
    print(scenario, phase, offline, low, high, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    for name in ('project', 'joint', 'capacity', 'residual-code', 'requests', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare(args)
    tasks = [(scenario, phase, share) for scenario in CONTEXT['scenarios']
             for phase in (0, 24, 168) for share in (0., .28)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [row for group in pool.map(one, tasks) for row in group]
    pd.DataFrame(rows).to_csv(args.output / 'gpu-service-capacity.csv', index=False)
    (args.output / 'benchmark.json').write_text(json.dumps(BENCHMARK, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
