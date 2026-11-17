import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--requests', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    gpu_hours = 226.75 * 8 * 24
    computed_revenue = (342e9 * .14 + (608e9 - 342e9) * .55 + 168e9 * 2.19) / 1e6
    request = pd.read_csv(args.requests)
    request = request[request.series == 'burst-1+2']
    result = {
        'source': 'https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md',
        'observation': '2025-02-27 12:00 to 2025-02-28 12:00 UTC+8',
        'hardware': '8 H800 GPUs per node; V3/R1 production inference; FP8/BF16',
        'average_nodes': 226.75, 'peak_nodes': 278, 'observed_gpu_hours': gpu_hours,
        'input_tokens': 608e9, 'output_tokens': 168e9, 'cache_hit_input_tokens': 342e9,
        'input_output_ratio': 608 / 168, 'input_cache_hit_fraction': 342 / 608,
        'average_over_peak_occupied_nodes': 226.75 / 278,
        'prefill_tokens_per_second_per_node': 73700., 'decode_tokens_per_second_per_node': 14800.,
        'published_hypothetical_revenue_usd': 562027.,
        'revenue_from_rounded_token_totals_usd': computed_revenue,
        'rounding_difference_fraction': (computed_revenue - 562027.) / 562027.,
        'hypothetical_gross_usd_per_occupied_H800_hour': 562027 / gpu_hours,
        'hypothetical_gross_usd_per_peak_installed_H800_hour': 562027 / (278 * 8 * 24),
        'actual_revenue': None,
        'scope': '全部 token 均按当时 R1 价格收费的理论毛收入；原报告明确实际收入更低，不能视作 H100 通用租价或净利润。',
        'burst_requests': int(request.requests.sum()),
        'burst_input_tokens': int(request.input_tokens.sum()),
        'burst_output_tokens': int(request.output_tokens.sum()),
        'burst_input_output_ratio': float(request.input_tokens.sum() / request.output_tokens.sum()),
        'benchmark_model': 'Llama 3.1 8B FP8, TensorRT-LLM 0.17, H100 80GB HBM3, TP1',
        'benchmark_input_output_tokens': [1000, 1000],
        'benchmark_output_tokens_per_second': 15244.78,
        'benchmark_standard_requests_per_GPU_hour': 15244.78 / 1000 * 3600,
        'benchmark_source': 'https://nvidia.github.io/TensorRT-LLM/0.18.2/performance/perf-overview.html',
        'benchmark_scope': '公开最大吞吐基准；请求时序采用 BurstGPT，服务量按标准长度请求换算。吞吐量 25%、50%、100% 为情景，未测量特定 PCIe 设备的响应延迟。',
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
