import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from compute_load import CoolingModel
from compute_load.cooling import COOLING_TYPES, LEGACY_VERSION, PARAMETER_SETS, VERSION, WEATHER_COLUMNS
from compute_load.model import sha256
from global_grid_compute_load import grid_weather


def read_frame(path):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("输入文件需为CSV或Parquet")


def filesystem_path(path):
    path = Path(path).resolve()
    value = str(path)
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return Path(value)


def replay(it_curve, model, output, *, weather=None, weather_source=None, capacity_mw=None):
    output = filesystem_path(output)
    curve = read_frame(it_curve)
    curve["time_utc"] = pd.to_datetime(curve.time_utc, utc=True)
    if curve.empty or curve.time_utc.isna().any():
        raise ValueError("IT曲线时间轴为空或包含缺失值")
    if curve.time_utc.duplicated().any():
        raise ValueError("单个IT曲线不能包含重复时间")
    curve = curve.sort_values("time_utc").reset_index(drop=True)
    if weather is None:
        weather = curve
    else:
        weather = weather.copy()
        weather["time_utc"] = pd.to_datetime(weather.time_utc, utc=True)
        weather = weather.sort_values("time_utc").reset_index(drop=True)
        if not np.array_equal(curve.time_utc.to_numpy(), weather.time_utc.to_numpy()):
            raise ValueError("天气和IT曲线必须包含完全相同的UTC小时")
        if "grid_id" in curve and "grid_id" in weather and not curve.grid_id.astype(str).equals(weather.grid_id.astype(str)):
            raise ValueError("天气和IT曲线网格标识不一致")
    capacities = curve.installed_reference_it_mw.unique() if "installed_reference_it_mw" in curve else []
    if capacity_mw is None:
        if len(capacities) != 1:
            raise ValueError("需要唯一installed_reference_it_mw或显式--capacity-mw")
        capacity_mw = float(capacities[0])
    grid = str(weather.grid_id.iloc[0]) if "grid_id" in weather else "unspecified"
    if "grid_id" in weather and weather.grid_id.nunique() != 1:
        raise ValueError("一个IT曲线文件只接受一个网格")
    summaries = []
    for year, group in curve.groupby(curve.time_utc.dt.year, sort=True):
        selected_weather = weather.loc[group.index].reset_index(drop=True)
        weather_columns = [column for column in WEATHER_COLUMNS if column in selected_weather]
        weather_digest = hashlib.sha256(pd.util.hash_pandas_object(selected_weather[weather_columns], index=False).to_numpy().tobytes()).hexdigest()
        request = {"model": model.metadata, "it_curve_sha256": sha256(it_curve),
                   "weather_source": weather_source, "capacity_mw": capacity_mw,
                   "weather_columns": weather_columns, "weather_values_sha256": weather_digest,
                   "replay_code_sha256": sha256(Path(__file__)), "year": int(year)}
        directory = (Path(output) / model.version / model.cooling_type / model.parameter_set
                     / ("grid_id=" + quote(grid, safe="")) / str(year) / it_curve.stem)
        manifest = directory / "manifest.json"
        if manifest.exists():
            previous = json.loads(manifest.read_text(encoding="utf-8"))
            if previous["request"] != request:
                raise ValueError(f"已有输出与本次输入或版本不同，请使用独立目录：{manifest}")
            if sha256(directory / "hourly.parquet") != previous["hourly_sha256"]:
                raise ValueError(f"已有小时结果校验失败：{directory}")
            summaries.append(previous["summary"])
            continue
        result = model.calculate(selected_weather, group.it_mw.to_numpy(), capacity_mw)
        directory.mkdir(parents=True, exist_ok=True)
        result.to_parquet(directory / "hourly.parquet", index=False, compression="zstd")
        summary = {"grid_id": grid, "year": int(year), "hours": len(result),
                   "it_mwh": float(result.it_mw.sum()), "cooling_mwh": float(result.cooling_mw.sum()),
                   "facility_mwh": float(result.facility_mw.sum())}
        summary["annual_pue"] = summary["facility_mwh"] / summary["it_mwh"] if summary["it_mwh"] > 0 else None
        manifest.write_text(json.dumps({"request": request, "summary": summary,
            "hourly_sha256": sha256(directory / "hourly.parquet")}, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        summaries.append(summary)
    return summaries


def main():
    parser = argparse.ArgumentParser(description="用版本明确的冷却模型重放已有IT曲线；支持全球网格原生ERA5天气")
    parser.add_argument("--it-curve", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cooling-type", choices=COOLING_TYPES, required=True)
    parser.add_argument("--cooling-version", choices=(VERSION, LEGACY_VERSION), default=VERSION)
    parser.add_argument("--parameter-set", choices=PARAMETER_SETS, default="central")
    parser.add_argument("--capacity-mw", type=float)
    weather = parser.add_mutually_exclusive_group()
    weather.add_argument("--weather", type=Path)
    weather.add_argument("--native-weather", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--grid-id")
    args = parser.parse_args()
    model = CoolingModel(cooling_type=args.cooling_type, version=args.cooling_version, parameter_set=args.parameter_set)
    external, source = None, None
    if args.weather:
        external = read_frame(args.weather)
        source = {"weather_sha256": sha256(args.weather)}
    elif args.native_weather:
        if args.mapping is None or args.grid_id is None:
            parser.error("--native-weather需要--mapping和--grid-id")
        mapping = pd.read_parquet(args.mapping)
        center = mapping.loc[mapping.grid_id.eq(args.grid_id) & mapping.row_offset.eq(1) & mapping.col_offset.eq(1)]
        if len(center) != 1:
            raise ValueError("网格需要唯一中心气象节点")
        times = pd.to_datetime(read_frame(args.it_curve).time_utc, utc=True)
        external = grid_weather(ds.dataset(args.native_weather, format="parquet"), center.iloc[0], sorted(times.dt.year.unique()))
        source = {"weather_sha256": sha256(args.native_weather), "mapping_sha256": sha256(args.mapping),
                  "grid_id": args.grid_id, "native_flat_index": int(center.iloc[0].native_flat_index),
                  "adapter_sha256": sha256(Path(__file__).with_name("global_grid_compute_load.py"))}
    print(json.dumps(replay(args.it_curve, model, args.output_dir, weather=external,
                            weather_source=source, capacity_mw=args.capacity_mw), ensure_ascii=False))


if __name__ == "__main__":
    main()
