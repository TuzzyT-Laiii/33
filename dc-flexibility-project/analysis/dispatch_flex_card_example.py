import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, 'project/analysis')
import flex_econ_copt as f

p, own = f.load_config(Path('project'))
out = Path('out/hourly-cards')
out.mkdir(exist_ok=True)
rows = []
city, scenario = 'frankfurt', 'D-TRN'
frame = pd.read_parquet(f'out/inputs/{city}-{scenario}.parquet')
frame.attrs['scenario'] = scenario
for alpha in (None, 1.0):
    for flex, (w, s) in (('none', (0, 0)), ('both', (12, 48))):
        spec = f.card_spec(p, own, frame, scenario, 1.0)
        summary, hourly = f.solve(frame, p, city, alpha=alpha, design='opt', whole_h=w, split_h=s, keep_hourly=True, cards=spec)
        hourly.to_parquet(out / f'{city}-{scenario}-opt-a{alpha}-{flex}-main.parquet', index=False)
        summary.update(flex=flex, variant='main')
        rows.append(summary)
pd.DataFrame(rows).to_csv(out / 'dispatch-runs.csv', index=False)
print(pd.DataFrame(rows)[['alpha', 'flex', 'status', 'total_cost_usd', 'extra_task_mw', 'pv_mw', 'wind_mw', 'battery_mwh', 'cfe_hourly']])
