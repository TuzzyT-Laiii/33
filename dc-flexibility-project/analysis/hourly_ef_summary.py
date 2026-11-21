"""逐时碳因子敏感性汇总：减排（kg/MWh）三种口径——年均因子、同一调度按逐时核算、逐时因子重新优化；按电网区与用途。"""
import sys

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
USE = {'CLD': '云计算', 'HPC': 'HPC', 'INF': '推理', 'SAI': '科研 AI', 'TRN': '训练'}
r = pd.read_json(sys.argv[1], lines=True)
print('状态', r.status.value_counts().to_dict(), '网格', r.grid_id.nunique())
r = r[r.status.eq('optimal')].copy()
for k in ('const', 'post', 'opt'):
    r[f'dEm_{k}'] = 1000 * (r[f'em_seq_{k}'] - r[f'em_joint_{k}']) / r.E0_window_mwh
r['post_ratio'] = r.dEm_post / r.dEm_const.where(r.dEm_const > 0.5)
r['opt_ratio'] = r.dEm_opt / r.dEm_const.where(r.dEm_const > 0.5)
r['dV'] = r.V_hourly - r.V_const
r['use'] = r.scenario.str.split('-').str[1].map(USE)
assert r.check_seq_em.abs().max() < 1e-6
cols = ['ci_cv', 'dEm_const', 'dEm_post', 'dEm_opt', 'post_ratio', 'opt_ratio', 'V_const', 'dV']
print('\n按电网区（逐对中位数）：')
print(r.groupby('zone')[cols].median().round(3).to_string())
print('\n按用途（全部网格）：')
print(r.groupby('use')[['dEm_const', 'dEm_post', 'dEm_opt', 'post_ratio', 'opt_ratio']].median().round(3).to_string())
print('\n日内波动大的区（DE、CISO、AZPS、ERCO）按用途：')
hi = r[r.zone.isin(['DE', 'US-CISO', 'US-AZPS', 'US-ERCO'])]
print(hi.groupby('use')[['dEm_const', 'dEm_post', 'dEm_opt', 'post_ratio', 'opt_ratio']].median().round(3).to_string())
print('\n合并：', r[['dEm_const', 'dEm_post', 'dEm_opt', 'post_ratio', 'opt_ratio', 'dV']].median().round(3).to_dict())
print('post_ratio P10/P90', r.post_ratio.quantile([.1, .9]).round(3).tolist(), 'opt_ratio P10/P90', r.opt_ratio.quantile([.1, .9]).round(3).tolist())
print('opt 减排高于年均口径的比例 %.1f%%' % (100 * (r.dEm_opt > r.dEm_const).mean()))
r.to_parquet(sys.argv[2], index=False)
