import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
import numpy as np
import pandas as pd
from pypdf import PdfReader


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def target_name(value):
    return '经济最优' if pd.isna(value) else f'{value:.0%}'


def prefix_content(document):
    content = []
    for element in document.element.body:
        text = ''.join(run.text or '' for run in element.iter(qn('w:t')))
        if '七、柔性调度的经济性' in text:
            return content
        images = []
        for image in element.iter(qn('a:blip')):
            relationship = image.get(qn('r:embed'))
            images.append(sha256(document.part.related_parts[relationship].blob).hexdigest())
        content.append((element.tag, text, images))
    raise ValueError('文档缺少第七章')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--previous-docx', type=Path, required=True)
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--word-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--visual-confirmed', action='store_true')
    args = parser.parse_args()
    project = args.project.resolve()
    result = args.result.resolve()
    docx_path = project / 'docs/job-flexibility-brief/数据中心任务时间柔性简表.docx'
    facts = json.loads((result / 'brief/brief-facts.json').read_text('utf-8'))
    require(all(row['deadline_clock'] == 'actual_utc_next_observed_slot' for row in facts['dispatch_sources']), '训练图件的时间约束版本不符')
    dispatch_audit = json.loads((result / 'independent-dispatch-audit.json').read_text('utf-8'))
    economic_audit = json.loads((result / 'economic-consistency-audit.json').read_text('utf-8'))
    require(dispatch_audit['passed'] and economic_audit['passed'], '训练结果尚未通过独立验收')
    require(dispatch_audit['index_sha256'] == digest(result / 'hourly-index.csv'), '小时索引与已验收版本不一致')
    source_index = pd.read_csv(result / 'hourly-index.csv')
    dispatch_sources = []
    for source in facts['dispatch_sources']:
        matched = source_index[source_index.hourly_sha256.eq(source['hourly_sha256'])]
        require(len(matched) == 1, '图件小时来源与已验收索引不一致')
        source_inputs = matched[['input_path', 'input_sha256']].iloc[0].to_dict()
        dispatch_sources.append({**source, **source_inputs})
    cards = pd.read_csv(result / 'runs-cards.csv')
    fixed = pd.read_csv(result / 'runs-fixed.csv')
    shares = pd.read_csv(project / 'results/flex-class-20261101/shares-10MW.csv').set_index('scenario')
    unit = pd.read_csv(result / 'card-unit-cost.csv', index_col=0)
    keys = ['city', 'scenario', 'target']
    for frame in (cards, fixed):
        frame['target'] = frame.alpha.map(target_name)
    cards_ok = cards[cards.status.eq('optimal')]
    fixed_ok = fixed[fixed.status.eq('optimal')]
    pn = cards_ok[cards_ok.flex.eq('none')].set_index(keys)
    pf = cards_ok[cards_ok.flex.eq('both') & cards_ok.card_price.astype(str).eq('1.0')].set_index(keys)
    bn = fixed_ok[fixed_ok.card_price.astype(str).eq('noflex')].set_index(keys)
    bf = fixed_ok[fixed_ok.card_price.astype(str).eq('1.0')].set_index(keys)
    require(all(frame.index.is_unique for frame in (pn, pf, bn, bf)), '求解结果存在重复配对键')
    targets = ['经济最优', '95%', '100%']
    audited = []
    for index, counts in enumerate(facts['sample_counts'], start=1):
        scenario = counts['scenario']
        name = counts['name']
        plan_values = []
        built_values = []
        for target in targets:
            pair_metrics = []
            for mode, (baseline, flexible) in enumerate(((pn, pf), (bn, bf))):
                left = baseline.xs((scenario, target), level=('scenario', 'target'))
                right = flexible.xs((scenario, target), level=('scenario', 'target'))
                cities = left.index.intersection(right.index)
                require(len(cities) > 0, f'{scenario}/{target} 缺少有效城市配对')
                left, right = left.loc[cities], right.loc[cities]
                np.testing.assert_array_equal(left.hours, right.hours)
                savings = left.total_cost_usd - right.total_cost_usd
                metrics = {
                    'n': len(cities),
                    'cities': list(cities),
                    'pct': float((100 * savings / left.total_cost_usd).median()),
                    'annual_10k_usd': float((savings * 8760 / left.hours / 1e4).median()),
                    'battery_baseline': float(left.battery_mwh.median()),
                    'battery_flexible': float(right.battery_mwh.median()),
                    'extra_max_mw': float(right.extra_task_mw.max()),
                }
                if mode == 1:
                    metrics['saving_per_mwh'] = float((savings / (right.moved_whole_mwh + right.moved_split_mwh)).median())
                    metrics['storage_per_mwh'] = float(((left.cost_battery_power + left.cost_battery_energy) / left.discharge_mwh.where(left.discharge_mwh > 1)).median())
                pair_metrics.append(metrics)
            plan_values.append(pair_metrics[0])
            built_values.append(pair_metrics[1])
        require(counts['planning'] == [item['n'] for item in plan_values], f'{scenario} 规划配对数不符')
        require(counts['built'] == [item['n'] for item in built_values], f'{scenario} 已建成配对数不符')
        plan_row = facts['plan_rows'][index]
        built_row = facts['built_rows'][index]
        require(plan_row[0] == name == built_row[0], f'{scenario} 行名不符')
        require(plan_row[1] == f"{shares.loc[scenario, '可推迟·整体'] + shares.loc[scenario, '可推迟·可拆分']:.0f}", f'{scenario} 份额不符')
        for values, row, pct_col, saving_col in ((plan_values, plan_row, 2, 3), (built_values, built_row, 1, 2)):
            require(row[pct_col] == '／'.join(f"{item['pct']:.1f}" for item in values), f'{scenario} 百分比与原始求解输出不符')
            require(row[saving_col] == '／'.join(f"{item['annual_10k_usd']:.0f}" for item in values), f'{scenario} 年化金额与原始求解输出不符')
            extra = max(item['extra_max_mw'] for item in values)
            extra_text = '0' if abs(extra) < 1e-6 else f'{extra:.3f}'
            require(row[-1] == extra_text, f'{scenario} 新增算力与原始求解输出不符')
        require(plan_row[4] == f"{plan_values[2]['battery_baseline']:.0f}→{plan_values[2]['battery_flexible']:.0f}", f'{scenario} 储能容量不符')
        require(built_row[3] == f"{built_values[0]['saving_per_mwh']:.0f}", f'{scenario} 单位柔性收益不符')
        require(built_row[4] == f"{built_values[0]['storage_per_mwh']:.0f}", f'{scenario} 单位储能成本不符')
        require(built_row[5] == f"{unit.loc[scenario, 'usd_per_mwh_8h/天']:,.0f}", f'{scenario} 单位加卡成本不符')
        audited.append({'scenario': scenario, 'targets': targets, 'planning': plan_values, 'built': built_values})
    document = Document(docx_path)
    require(len(document.tables) == 9, 'Word 表格总数应为9')
    for table, expected in zip(document.tables[-2:], (facts['plan_rows'], facts['built_rows'])):
        require([[cell.text for cell in row.cells] for row in table.rows] == expected, 'Word 两组表格与核验数据不一致')
    previous = Document(args.previous_docx)
    require(prefix_content(document) == prefix_content(previous), '第七章之前的文字、表格或图片发生变化')
    layout = json.loads(args.word_audit.read_text('utf-8-sig'))
    require(layout['docx_sha256'].lower() == digest(docx_path), 'Word 排版记录对应了其他输入文件')
    require(layout['pdf_sha256'].lower() == digest(args.pdf), 'PDF 与排版记录不一致')
    require(all(table['start_page'] == table['end_page'] for table in layout['tables']), 'Word 存在跨页表格')
    pages = [page.extract_text() for page in PdfReader(args.pdf).pages]
    require(len(pages) == layout['pages'], 'PDF 页数与Word记录不符')
    require('7.1 规划阶段' in pages[8] and '7.2 已建成' in pages[8], '第9页应同时包含两组标题')
    require('\ufffd' not in ''.join(pages), 'PDF 出现替代字符')
    require(args.visual_confirmed, '须完成全部页面视觉检查后再确认')
    inputs = ['runs-cards.csv', 'runs-fixed.csv', 'fixed-supply-values.csv', 'card-unit-cost.csv',
              'hourly-index.csv', 'independent-dispatch-audit.json', 'economic-consistency-audit.json',
              'brief/brief-facts.json', 'brief/value-summary.png', 'brief/dispatch-week.png']
    report = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'passed': True,
        'scope': '两组表格的原始求解输出、数字舍入、城市配对、Word一致性及分页；不重复验证求解模型',
        'pages': len(pages),
        'table_pages': [table['start_page'] for table in layout['tables']],
        'first_six_chapters_text_tables_images_unchanged': True,
        'all_pages_visually_reviewed': True,
        'docx_sha256': digest(docx_path),
        'pdf_sha256': digest(args.pdf),
        'pdf_exporter': layout['exporter'],
        'previous_input': {'filename': args.previous_docx.name, 'sha256': digest(args.previous_docx),
                           'modified_at_utc': datetime.fromtimestamp(args.previous_docx.stat().st_mtime, timezone.utc).isoformat()},
        'input_sha256': {name: digest(result / name) for name in inputs},
        'classification_shares_sha256': digest(project / 'results/flex-class-20261101/shares-10MW.csv'),
        'dispatch_sources': dispatch_sources,
        'audited_rows': audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps({'passed': True, 'pages': len(pages), 'rows': len(audited), 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
