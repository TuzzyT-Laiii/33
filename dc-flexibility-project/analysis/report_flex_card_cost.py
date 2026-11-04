"""柔性成本汇总：峰值内柔性（不加卡）的收益、加卡柔性的单位成本与储能单位成本对比。"""
import argparse
import json
from pathlib import Path

import pandas as pd


NAMES = {
    'A-HPC': '纯 CPU 型·科研计算', 'A-CLD': '纯 CPU 型·云通用计算',
    'B-SAI': 'CPU 主导型·科研 AI', 'C-SAI': '均衡型·科研 AI', 'D-SAI': 'GPU 主导型·科研 AI',
    'B-TRN': 'CPU 主导型·大模型训练/开发', 'C-TRN': '均衡型·大模型训练/开发', 'D-TRN': 'GPU 主导型·大模型训练/开发',
    'B-INF': 'CPU 主导型·推理服务', 'C-INF': '均衡型·推理服务', 'D-INF': 'GPU 主导型·推理服务'}
PUE, ENERGY_PRICE = 1.38, 133.   # 新增卡待机耗电按典型PUE与市电+碳价估价
HOURS = {'8h/天': 2920, '4h/天': 1460}


def label(value):
    return '经济最优' if pd.isna(value) else f'{value:.0%}'


def mr(series, digits=0):
    return f'{series.median():.{digits}f}（{series.min():.{digits}f}–{series.max():.{digits}f}）'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cards', required=True, help='runs-cards.csv')
    parser.add_argument('--fixed', required=True, help='runs-fixed.csv')
    parser.add_argument('--old', required=True, help='首版paired-values.csv')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    keys = ['city', 'scenario', 'a']

    # 1 供给侧重新优化、机群按峰值（边界A）
    cards = pd.read_csv(args.cards)
    cards['a'] = cards.alpha.map(label)
    none = cards[cards.flex == 'none'].set_index(keys)
    both = cards[(cards.flex == 'both') & (cards.card_price.astype(str) == '1.0')].set_index(keys)
    a = pd.DataFrame({'pct': 100 * (none.total_cost_usd - both.total_cost_usd) / none.total_cost_usd,
                      'save_kusd_year': (none.total_cost_usd - both.total_cost_usd) * 8760 / none.hours / 1e4,
                      'extra_cards_mw': both.extra_task_mw}).reset_index()
    old = pd.read_csv(args.old)
    old = old[old.flex == 'both'].rename(columns={'alpha_label': 'a'}).set_index(keys).value_pct
    a = a.join(old.rename('pct_nameplate'), on=keys)
    a.to_csv(out / 'boundary-a-values.csv', index=False, encoding='utf-8-sig')

    # 2 供给侧固定为无柔性最优
    fixed = pd.read_csv(args.fixed)
    fixed['a'] = fixed.alpha.map(label)
    fixed = fixed[fixed.status == 'optimal']
    n = fixed[fixed.card_price == 'noflex'].set_index(keys)
    f = fixed[fixed.card_price == '1.0'].set_index(keys)
    d = pd.DataFrame({'moved_mwh': f.moved_whole_mwh + f.moved_split_mwh, 'saving_usd': n.total_cost_usd - f.total_cost_usd,
                      'storage_usd': n.cost_battery_power + n.cost_battery_energy, 'discharge_mwh': n.discharge_mwh,
                      'hours': n.hours, 'total_usd': n.total_cost_usd, 'extra_cards_mw': f.extra_task_mw}).dropna()
    d['pct'] = 100 * d.saving_usd / d.total_usd
    d['save_kusd_year'] = d.saving_usd * 8760 / d.hours / 1e4
    d['moved_mwh_year'] = d.moved_mwh * 8760 / d.hours
    d['benefit_usd_per_mwh'] = d.saving_usd / d.moved_mwh
    d['storage_usd_per_mwh'] = d.storage_usd / d.discharge_mwh.where(d.discharge_mwh > 1)
    d = d.reset_index()
    d.to_csv(out / 'fixed-supply-values.csv', index=False, encoding='utf-8-sig')

    # 3 加卡柔性单位成本（解析）
    spec = cards[(cards.flex == 'both') & (cards.card_price.astype(str) == '1.0')].groupby('scenario').agg(
        card=('card_usd_per_mw_year', 'first'), idle=('idle_per_mw', 'first'), peak=('peak_task_mw', 'first'),
        servers_per_mw=('servers_per_mw', 'first'), fleet=('baseline_fleet_usd_year', 'first'))
    spec['idle_usd'] = spec.idle * 8760 * PUE * ENERGY_PRICE
    for name, hours in HOURS.items():
        spec['usd_per_mwh_' + name] = (spec.card + spec.idle_usd) / hours
    spec.to_csv(out / 'card-unit-cost.csv', encoding='utf-8-sig')

    rows = [['情境', '峰值内柔性\n每年推迟MWh', '能源节省\n万美元/年', '每推迟1MWh收益\nUSD（经济最优）',
             '每推迟1MWh收益\nUSD（95%）', '储能每放1MWh\nUSD（经济最优/100%）', '加卡柔性每推迟1MWh\nUSD（每天用8h）']]
    for scenario, name in NAMES.items():
        e = d[(d.scenario == scenario) & (d.a == '经济最优')]
        h = d[(d.scenario == scenario) & (d.a == '95%')]
        full = d[(d.scenario == scenario) & (d.a == '100%')]
        rows.append([name, f'{e.moved_mwh_year.median():,.0f}', mr(e.save_kusd_year), mr(e.benefit_usd_per_mwh),
                     mr(h.benefit_usd_per_mwh), f'{e.storage_usd_per_mwh.median():.0f} / {full.storage_usd_per_mwh.median():.0f}',
                     f'{spec.loc[scenario, "usd_per_mwh_8h/天"]:,.0f}'])
    pd.DataFrame(rows[1:], columns=rows[0]).to_csv(out / 'table-flex-cost.csv', index=False, encoding='utf-8-sig')
    rows_a = [['情境', '经济最优 柔性收益%\n（首版铭牌余量）', '100%无碳 柔性收益%\n（首版）', '经济最优\n万美元/年', '100%无碳\n万美元/年', '基线机群年化\n万美元/年']]
    for scenario, name in NAMES.items():
        e = a[(a.scenario == scenario) & (a.a == '经济最优')]
        full = a[(a.scenario == scenario) & (a.a == '100%')]
        rows_a.append([name, f'{e.pct.median():.1f}（{e.pct_nameplate.median():.1f}）', f'{full.pct.median():.1f}（{full.pct_nameplate.median():.1f}）',
                       f'{e.save_kusd_year.median():.0f}', f'{full.save_kusd_year.median():.0f}', f'{spec.loc[scenario, "fleet"] / 1e4:,.0f}'])
    pd.DataFrame(rows_a[1:], columns=rows_a[0]).to_csv(out / 'table-boundary-a.csv', index=False, encoding='utf-8-sig')
    facts = {'flex_cost_rows': rows, 'boundary_a_rows': rows_a,
             'cards_bought_market_price_max_mw': float(max(a.extra_cards_mw.max(), d.extra_cards_mw.max())),
             'fixed_status': fixed.status.value_counts().to_dict()}
    (out / 'brief-facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=1), 'utf-8')
    for table in (rows_a, rows):
        print(pd.DataFrame(table[1:], columns=[c.replace('\n', '') for c in table[0]]).to_string(index=False))


if __name__ == '__main__':
    main()
