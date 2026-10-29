import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd


def diagnose(args):
    project=args.project
    output=project/'results/eight-city-phase-a-20261029'
    curve=pd.read_csv(args.benchmarks/'IEA_Reference_3.4MW_130.csv')
    benchmark=pd.read_csv(args.benchmarks/'city-benchmarks.csv').set_index('city_id')
    rows=[]
    for city in json.loads((project/'analysis/eight_city_locations.json').read_text(encoding='utf-8')):
        name=city['city_id']
        frame=pd.read_parquet(args.weather/(name+'-arco.parquet'))
        wind=np.hypot(frame.u100.to_numpy(dtype=float),frame.v100.to_numpy(dtype=float))
        hub=wind*(110/100)**(1/7)
        density=frame.sp.to_numpy(dtype=float)/(287.05*frame.t2m.to_numpy(dtype=float))
        def factor(speed,actual):
            power=np.interp(speed,curve['Wind Speed [m/s]'],curve['Power [kW]'],left=0,right=0)
            return float(np.where(actual>25,0,np.clip(power/3370,0,1)).mean())
        gross=factor(hub*(density/1.225)**(1/3),hub)
        uncorrected=factor(hub,hub)
        hundred=factor(wind*(density/1.225)**(1/3),wind)
        rows.append({'city_id':name,'city_zh':city['city_zh'],'mean_air_density_kg_m3':float(density.mean()),
            'hub110_density_gross_cf':gross,'hub110_standard_density_gross_cf':uncorrected,
            'hub100_density_gross_cf':hundred,'density_effect_percentage_points':(gross-uncorrected)*100,
            'height110_vs100_effect_percentage_points':(gross-hundred)*100,
            'assumed10pct_loss_percentage_points':-gross*10,
            'gwa_iec3_cf':float(benchmark.loc[name,'gwa_iec3_capacity_factor']),
            'gross_minus_gwa_percentage_points':(gross-benchmark.loc[name,'gwa_iec3_capacity_factor'])*100,
            'net_minus_gwa_percentage_points':(.9*gross-benchmark.loc[name,'gwa_iec3_capacity_factor'])*100})
    pd.DataFrame(rows).to_csv(output/'wind-model-diagnostics.csv',index=False,encoding='utf-8-sig')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--weather',type=Path,required=True)
    parser.add_argument('--benchmarks',type=Path,required=True)
    diagnose(parser.parse_args())
