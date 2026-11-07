import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/global-flex-econ-20261103'
OUT = ROOT / 'papers/demand-flexibility/figures'
SOURCE = OUT / 'source-data'
TEMP = ROOT / '.codex-temp/paper-results1'
SCENARIOS = ['A-HPC', 'A-CLD', 'B-SAI', 'C-SAI', 'D-SAI', 'B-TRN', 'C-TRN', 'D-TRN', 'B-INF', 'C-INF', 'D-INF']
REPRESENTATIVES = ['A-HPC', 'A-CLD', 'D-SAI', 'D-TRN', 'D-INF']
USES = {'HPC': 'Scientific computing (CPU)', 'CLD': 'Cloud computing (CPU)',
        'SAI': 'Scientific AI', 'TRN': 'LLM training', 'INF': 'Inference'}
SPECS = {'A': 'CPU-only', 'B': 'CPU-heavy', 'C': 'Balanced', 'D': 'GPU-heavy'}
COLORS = {'HPC': '#7B61A8', 'CLD': '#C18B22', 'SAI': '#0072B2', 'TRN': '#009E73', 'INF': '#D55E00'}
TASKS = ['刚性', '未知', '可推迟·整体', '可推迟·可拆分']


def scenario_label(scenario):
    spec, use = scenario.split('-')
    return USES[use] if spec == 'A' else f'{USES[use]} · {SPECS[spec]}'


def color(scenario):
    return COLORS[scenario.split('-')[1]]


def shade(scenario):
    fraction = {'A': 0, 'B': .47, 'C': .23, 'D': 0}[scenario[0]]
    return tuple(np.array(mpl.colors.to_rgb(color(scenario))) * (1 - fraction) + fraction)


def setup():
    for directory in (OUT, SOURCE, TEMP):
        directory.mkdir(parents=True, exist_ok=True)
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    family = next(name for name in ['Arial', 'Helvetica', 'DejaVu Sans'] if name in fonts)
    mpl.rcParams.update({
        'font.family': family, 'font.size': 6.5, 'axes.labelsize': 6.5,
        'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
        'legend.frameon': False,
        'axes.linewidth': .5, 'lines.linewidth': 1, 'patch.linewidth': .5,
        'xtick.major.width': .5, 'ytick.major.width': .5,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 600,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })
    return family


def panel(fig, letter, x, y):
    fig.text(x, y, letter, fontsize=8, fontweight='bold', va='top')


def csv(frame, name):
    frame.to_csv(SOURCE / f'{name}.csv', index=False, float_format='%.15g')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_economics():
    rows = []
    for scenario in SCENARIOS:
        frame = pd.read_parquet(ROOT / f'results/flex-class-20261101/native-hourly/{scenario}.parquet')
        task = frame[TASKS].sum(axis=1)
        k = task.max()
        rows.append({'scenario': scenario, 'label': scenario_label(scenario), 'hours': len(frame),
                     'K_mw': k, 'mean_over_K_pct': task.mean() / k * 100,
                     'headroom_pct': (1 - task.mean() / k) * 100,
                     'hours_above_90pct_K': int((task > .9 * k).sum())})
    head = pd.DataFrame(rows).set_index('scenario')
    cards = json.loads((ROOT / 'analysis/flex-econ-copt-20261102.json').read_text('utf-8'))['cards']
    factor = .08 * 1.08 ** 4 / (1.08 ** 4 - 1) + .03
    fleet = {s: cards['server_price_usd'][s[0]]['value'] * cards['fleet'][s]['unit_count']
             * cards['fleet'][s]['u_max'] * factor for s in SCENARIOS}
    grid = pd.read_parquet(RUN / 'grid-summary.parquet')
    valid = grid.loc[grid.alpha.isna() & grid.baseline_status.eq('optimal') & grid.flex_status.eq('optimal')].copy()
    assert not valid.duplicated(['grid_id', 'scenario', 'boundary']).any()
    assert (valid.groupby(['scenario', 'boundary']).size() == 4842).all()
    assert valid[['baseline_cost_usd', 'flex_cost_usd', 'savings_pct']].notna().all().all()
    np.testing.assert_allclose(valid.savings_pct, 100 * (1 - valid.flex_cost_usd / valid.baseline_cost_usd), atol=1e-8)
    valid['window_hours'] = valid.scenario.map(head.hours)
    valid['saving_annual_usd'] = (valid.baseline_cost_usd - valid.flex_cost_usd) * 8760 / valid.window_hours
    valid['energy_annual_usd'] = valid.baseline_cost_usd * 8760 / valid.window_hours
    valid['fleet_annual_usd'] = valid.scenario.map(fleet)
    valid['fleet_savings_pct'] = 100 * valid.saving_annual_usd / valid.fleet_annual_usd
    assert valid.extra_task_mw.abs().max() < 1e-6
    return head, valid, factor


def export(fig, name):
    # 不裁切画布，保证导出的实际宽度始终为 180 mm。
    fig.savefig(OUT / f'{name}.pdf', dpi=600)
    fig.savefig(OUT / f'{name}.png', dpi=600)
    fig.savefig(TEMP / f'{name}-preview.png', dpi=190)
    plt.close(fig)
