import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
import numpy as np
import pandas as pd


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def table_signature(table):
    return tuple(tuple(cell.text for cell in row.cells) for row in table.rows)


def prefix(doc):
    records = []
    for element in doc.element.body:
        text = ''.join(element.itertext())
        words = [item.text for item in element.iter(qn('w:t'))]
        if '七、柔性调度的经济性（计入加卡，初算）' in ''.join(words):
            break
        pictures = [sha256(doc.part.related_parts[item.get(qn('r:embed'))].blob).hexdigest()
                    for item in element.iter(qn('a:blip'))]
        records.append((element.tag, words, pictures))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-docx', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = root / 'results/global-flex-econ-20261103'
    folder = result / 'compute-cost'
    base = pd.read_parquet(result / 'grid-summary.parquet')
    base = base.loc[base.baseline_status.eq('optimal') & base.flex_status.eq('optimal')]
    actual = pd.read_parquet(folder / 'grid-metrics.parquet')
    keys = ['grid_id', 'scenario', 'alpha', 'boundary']
    pairs = base.merge(actual, on=keys, validate='one_to_one', how='outer', indicator=True)
    assert pairs['_merge'].eq('both').all() and len(pairs) == 639144
    hours = pd.read_csv(result / 'provenance/template-index.csv').set_index('scenario').hours
    city = pd.read_csv(root / 'results/flex-card-cost-calendar-20261102/runs-cards.csv')
    fleet = city.groupby('scenario').baseline_fleet_usd_year.first()
    expected = (pairs.baseline_cost_usd - pairs.flex_cost_usd) / pairs.scenario.map(hours) * 8760 / pairs.scenario.map(fleet) * 100
    np.testing.assert_allclose(pairs.compute_cost_pct, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pairs.energy_savings_pct, pairs.savings_pct, rtol=1e-12, atol=1e-12)
    assert (pairs.above_threshold == (expected > 3)).all()
    summaries = pd.read_csv(folder / 'scenario-summary.csv')
    for row in summaries.itertuples():
        alpha = None if row.target == 'economic' else int(row.target[5:]) / 100
        selected = actual.loc[actual.scenario.eq(row.scenario) & actual.boundary.eq(row.boundary)]
        selected = selected.loc[selected.alpha.isna() if alpha is None else selected.alpha.eq(alpha)]
        assert row.grids == len(selected) == 4842
        assert row.above == selected.above_threshold.sum()
        np.testing.assert_allclose(row.share_pct, selected.above_threshold.mean() * 100, atol=1e-12)
        np.testing.assert_allclose(row.compute_cost_median_pct, selected.compute_cost_pct.median(), atol=1e-12)
    doc_path = root / 'docs/job-flexibility-brief/数据中心任务时间柔性简表.docx'
    doc, original = Document(doc_path), Document(args.baseline_docx)
    assert prefix(doc) == prefix(original), '前六章文字、表格或图片改变'
    current_tables = Counter(table_signature(table) for table in doc.tables)
    for signature, count in Counter(table_signature(table) for table in original.tables).items():
        assert current_tables[signature] >= count, '既有数据表发生变化'
    assert len(doc.element.xpath('//m:oMath')) == 4, '缺少可编辑公式'
    embedded = {sha256(part.blob).hexdigest() for part in doc.part.related_parts.values() if part.content_type.startswith('image/')}
    figures = sorted(folder.glob('*.png'))
    assert len(figures) == 4 and all(digest(path) in embedded for path in figures)
    text = '\n'.join(p.text for p in doc.paragraphs)
    assert '该图仅包括能源侧调度节省，未重新优化既有算力容量' in text
    assert '取3%作为柔性“值得做”的下限' not in text
    report = {'passed': True, 'verified_pairs': len(pairs), 'verified_summary_groups': len(summaries),
              'native_equations': 4, 'new_figures_embedded': len(figures), 'first_six_chapters_unchanged': True,
              'all_existing_tables_preserved': True, 'docx_sha256': digest(doc_path),
              'baseline_docx_sha256': digest(args.baseline_docx), 'verifier_sha256': digest(Path(__file__))}
    (folder / 'document-verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    manifest = json.loads((folder / 'artifact-manifest.json').read_text('utf-8'))
    manifest['files'] = {path.name: digest(path) for path in folder.iterdir() if path.is_file() and path.name != 'artifact-manifest.json'}
    (folder / 'artifact-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
