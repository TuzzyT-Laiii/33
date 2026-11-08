import json
from pathlib import Path
import subprocess

import fitz
import numpy as np
import pandas as pd
from PIL import Image

from paper_figure_zh_common import (CAPACITY, ENGLISH, OUT, RESALE, ROOT, SOURCE,
                                    TEMP, digest)


def main():
    report = {'figures': {}, 'inputs': {}, 'numbers': {}}
    for n, panels in [(1, 'abcd'), (2, 'abc'), (3, 'abcd'), (4, 'abc')]:
        name = f'fig{n}'
        for panel in panels:
            data = pd.read_csv(SOURCE / f'{name}{panel}.csv')
            assert len(data) > 0
        doc = fitz.open(OUT / f'{name}.pdf')
        assert len(doc) == 1
        page = doc[0]
        assert abs(page.rect.width / 72 * 25.4 - 180) < .01
        images = page.get_images(full=True)
        assert bool(images) == (n == 1), (name, len(images))
        fonts = sorted({f[3] for f in page.get_fonts(full=True)})
        assert any('Arial' in f for f in fonts)
        assert any('YaHei' in f or 'Noto' in f for f in fonts)
        text = page.get_text()
        assert '\ufffd' not in text
        with Image.open(OUT / f'{name}.png') as png:
            assert abs(png.width - 180 / 25.4 * 600) < 1.1
            assert abs(png.info['dpi'][0] - 600) < .1
            png_size = list(png.size)
        meta = json.loads((SOURCE / f'{name}_metadata.json').read_text(encoding='utf-8'))
        assert meta['minimum_text_pt'] >= 6
        boxes = json.loads((TEMP / f'{name}-text-boxes.json').read_text(encoding='utf-8'))
        letters = [b for b in boxes if b['text'] in list(panels)]
        assert sorted(b['text'] for b in letters) == list(panels)
        assert all(b['size_pt'] == 8 for b in letters)
        outside = []
        for b in boxes:
            x, y, w, h = b['box']
            if x < -.002 or y < -.002 or x + w > 1.002 or y + h > 1.002:
                outside.append(b)
        assert not outside, (name, outside)
        # 从 PDF 再次渲染，以核查最终提交的文件。
        page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72)).save(TEMP / f'{name}-pdf-review.png')
        report['figures'][name] = {'width_mm': page.rect.width / 72 * 25.4,
                                   'png_pixels': png_size, 'dpi': 600,
                                   'fonts': fonts, 'raster_images': len(images),
                                   'minimum_text_pt': meta['minimum_text_pt'],
                                   'panel_letters': panels, 'text_inside_canvas': True,
                                   'pdf_sha256': digest(OUT / f'{name}.pdf'),
                                   'png_sha256': digest(OUT / f'{name}.png')}
        doc.close()

    unchanged = ['fig1a', 'fig1a_continents', 'fig1b', 'fig1c', 'fig1d', 'fig1d_summary',
                 'fig2a', 'fig2a_gridvalues', 'fig2b', 'fig2c']
    for name in unchanged:
        original = pd.read_csv(ENGLISH / f'source-data/{name}.csv')
        translated = pd.read_csv(SOURCE / f'{name}.csv')
        assert len(original) == len(translated)
        for column in original.select_dtypes(include='number'):
            np.testing.assert_allclose(original[column], translated[column], rtol=1e-12, atol=1e-10, equal_nan=True)
    report['fig1_fig2_numeric_identity'] = True

    head = pd.read_csv(SOURCE / 'fig1b.csv')
    costs = pd.read_csv(SOURCE / 'fig1c.csv')
    dist = pd.read_csv(SOURCE / 'fig1d.csv')
    numbers = report['numbers']
    numbers['峰值余量范围_pct'] = [float(head.headroom_pct.min()), float(head.headroom_pct.max())]
    for boundary in ['planning', 'fixed']:
        v = costs.loc[costs.boundary.eq(boundary), 'energy_savings_pct']
        numbers[f'能源侧节省中位数范围_{boundary}_pct'] = [float(v.min()), float(v.max())]
    numbers['节省占机群成本最高_pct'] = float(dist.fleet_savings_pct.max())
    bins = pd.read_csv(SOURCE / 'fig2a.csv')
    train = bins.loc[bins.scenario.eq('D-TRN')].sort_values('wind_bin_index')
    numbers['训练首末风电档节省_pct'] = [float(train['median'].iloc[0]), float(train['median'].iloc[-1])]
    scatter = pd.read_csv(SOURCE / 'fig2b.csv')
    numbers['训练市电依赖相关系数'] = float(scatter.grid_share.corr(scatter.savings_pct))
    a = pd.read_csv(SOURCE / 'fig3a.csv').set_index('scenario')
    numbers['主期限减配率_pct'] = a.reduction_pct.to_dict()
    numbers['科研计算减配率区间_pct'] = [float(a.loc['A-HPC', 'reduction_pct']), float(a.loc['A-HPC', 'reduction_upper_pct'])]
    sensitivity = pd.read_csv(SOURCE / 'fig3b.csv')
    numbers['48_168小时减配率_pct'] = sensitivity.loc[sensitivity.whole_h.eq(48)].set_index('scenario').reduction_lower_pct.to_dict()
    c = pd.read_csv(SOURCE / 'fig3c.csv').set_index('scenario')
    numbers['训练可推迟电量加权平均推迟_h'] = float(c.loc['D-TRN', 'flex_energy_mean_delay_h'])
    numbers['训练缩减后资源平均占用率_pct'] = float(c.loc['D-TRN', 'occupancy_pct'])
    d = pd.read_csv(SOURCE / 'fig3d.csv')
    numbers['训练冗余设备节省_百万美元'] = (d.loc[d.scenario.eq('D-TRN')].set_index('reserve_fraction').annual_equipment_saving_usd / 1e6).to_dict()
    a4 = pd.read_csv(SOURCE / 'fig4a.csv')
    numbers['联合规划净节省占机群成本范围_pct'] = [float(a4.net_total_pct_of_fleet.min()), float(a4.net_total_pct_of_fleet.max())]
    numbers['联合规划能源节省范围_百万美元'] = [float(a4.energy_annual_saving_vs_A_usd_median.min() / 1e6), float(a4.energy_annual_saving_vs_A_usd_median.max() / 1e6)]
    b4 = pd.read_csv(SOURCE / 'fig4b.csv')
    numbers['训练价格1_2倍净节省_百万美元'] = (b4.loc[b4.scenario.eq('D-TRN') & b4.price_multiplier.isin([1, 2])].sort_values('price_multiplier').net_annual_saving_vs_A_usd / 1e6).tolist()
    resale = pd.read_csv(RESALE / 'resale-price-share-city.csv')
    assert len(resale) == 1080 and resale.groupby(['scenario', 'price_usd_gpu_h', 'sold_fraction']).city.nunique().eq(8).all()
    for price in [1, 2.2]:
        v = resale.loc[resale.sold_fraction.eq(.5) & resale.price_usd_gpu_h.eq(price)].groupby('scenario').D_net_annual_usd.median() / 1e6
        numbers[f'出租比例50_价格{price}_净收入范围_百万美元'] = [float(v.min()), float(v.max())]
    numbers['持平售价_美元每GPU小时'] = pd.read_csv(SOURCE / 'fig4c_break_even.csv').set_index('scenario').break_even_usd_gpu_h_s50.to_dict()
    assert round(numbers['节省占机群成本最高_pct'], 2) == 1.87
    assert round(numbers['训练市电依赖相关系数'], 2) == -.88
    np.testing.assert_allclose(numbers['训练首末风电档节省_pct'], [9.4, 20.4], atol=.05)
    np.testing.assert_allclose(numbers['训练价格1_2倍净节省_百万美元'], [31.49, 61.58], atol=.005)
    np.testing.assert_allclose(list(numbers['训练冗余设备节省_百万美元'].values()), [30.09, 24.86, 19.75], atol=.005)
    np.testing.assert_allclose(numbers['出租比例50_价格1_净收入范围_百万美元'], [10.38, 24.01], atol=.005)
    np.testing.assert_allclose(numbers['出租比例50_价格2.2_净收入范围_百万美元'], [23.28, 54.20], atol=.005)

    dependencies = {
        '60': 'origin/codex/part1-headroom-cost-figure-20261106',
        '63': 'origin/codex/paper-results1-figures-20261107',
        '65': 'origin/feat/minimum-server-capacity-20261108',
        '67': 'origin/feat/idle-gpu-resale-import-20261108',
    }
    for pr, ref in dependencies.items():
        sha = subprocess.run(['git', 'rev-parse', ref], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        report['inputs'][f'PR{pr}'] = {'ref': ref, 'commit': sha}
    paths = [*sorted((ENGLISH / 'source-data').glob('*.csv')),
             *sorted((ENGLISH / 'map-assets').glob('*')),
             *[CAPACITY / name for name in ['part2-main-table.csv', 'capacity-sensitivity.csv',
                'waiting-and-utilization.csv', 'equipment-reserve-sensitivity.csv', 'equipment-price-sensitivity.csv']],
             RESALE / 'resale-price-share-city.csv',
             ROOT / 'results/global-flex-econ-20261103/part1/part1-summary.csv']
    report['input_sha256'] = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    (OUT / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(numbers, ensure_ascii=False, indent=2))
    print('14 个主面板源数据、数值恒等关系、字体、字号、尺寸、600 dpi、PDF 图层和画布边界检查通过。')


if __name__ == '__main__':
    main()
