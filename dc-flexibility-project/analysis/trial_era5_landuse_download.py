import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests
import xarray as xr
import yaml
import zarr


SOURCE_URL = "https://arco.datastores.ecmwf.int/cadl-arco-geo-002/arco/reanalysis_era5_single_levels/sfc/geoChunked.zarr"
VARIABLES = {
    "2m_temperature": ("t2m", "K", 1),
    "2m_dewpoint_temperature": ("d2m", "K", 1),
    "100m_u_component_of_wind": ("u100", "m s**-1", 1),
    "100m_v_component_of_wind": ("v100", "m s**-1", 1),
    "surface_pressure": ("sp", "Pa", 1),
    "mean_surface_downward_short_wave_radiation_flux": ("ssrd", "J m**-2", 3600),
    "mean_surface_direct_short_wave_radiation_flux": ("fdir", "J m**-2", 3600),
}


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class CachedHTTPStore(zarr.storage.DirectoryStore):
    def __init__(self, directory, session):
        super().__init__(str(directory))
        self.session = session
        self.events = []

    def getitems(self, keys, *, contexts=None):
        return {key: self[key] for key in keys}

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError("Zarr对象键超出已选择的数据源")
        started = time.monotonic()
        response = self.session.get(SOURCE_URL + "/" + key, timeout=(15, 60))
        if response.status_code == 404:
            raise KeyError(key)
        response.raise_for_status()
        payload = response.content
        super().__setitem__(key, payload)
        self.events.append({"key": key, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                            "http_status": response.status_code, "seconds": round(time.monotonic() - started, 3)})
        print(json.dumps({"event": "object_received", "key": key, "bytes": len(payload)}), flush=True)
        return payload


def run(args):
    if not zarr.__version__.startswith("2."):
        raise RuntimeError("本次试验使用Zarr 2的DirectoryStore，要求zarr<3")
    grid_path = args.selection_dir / "landuse-grid-screen.parquet"
    mapping_path = args.selection_dir / "grid-weather-mapping.parquet"
    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    for name, path in (("grids", grid_path), ("mapping", mapping_path)):
        if file_digest(path) != selection["outputs"][name]["sha256"]:
            raise ValueError(f"试验输入与候选选择清单SHA不一致：{path}")
    grids = pd.read_parquet(grid_path)
    chosen = grids.loc[grids.grid_id.eq(args.grid_id)]
    if len(chosen) != 1 or not chosen.selected_for_weather.all():
        raise ValueError("试验网格必须通过有明确候选干土地类的初筛")
    mapping = pd.read_parquet(mapping_path)
    points = mapping.loc[mapping.grid_id.eq(args.grid_id)].sort_values("native_flat_index")
    if len(points) != 9 or points.native_flat_index.duplicated().any():
        raise ValueError("试验片区未包含完整3乘3个原生ERA5节点")
    key = os.environ.get("CDSAPI_KEY")
    if not key:
        config = yaml.safe_load((Path.home() / ".cdsapirc").read_text(encoding="utf-8"))
        key = config["key"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        session.headers.update({"Authorization": f"Bearer {key}", "User-Agent": "dc-flexibility-research/1.0"})
        store = CachedHTTPStore(args.cache_dir, session)
        metadata_payload = store[".zmetadata"]
        metadata = json.loads(metadata_payload)["metadata"]
        source_names = [value[0] for value in VARIABLES.values()]
        for name in source_names:
            spec = metadata[name + "/.zarray"]
            if spec["shape"][1:] != [721, 1440] or spec["chunks"] != [67584, 4, 4] or spec["dtype"] != "<f4":
                raise ValueError(f"空间分块源结构与本次核验的条件不一致：{name}")
        source = xr.open_zarr(store, consolidated=True, chunks=None)
        with source:
            times = pd.date_range(args.start, args.end, inclusive="left", freq="h").to_numpy(dtype="datetime64[ns]")
            if len(times) == 0 or source.indexes["time"].get_indexer(times).min() < 0:
                raise ValueError("空间分块源未覆盖全部试验小时")
            lat = (90 - points.lat_index.to_numpy() * 0.25).astype(np.float64)
            lon = (points.lon_index.to_numpy() * 0.25).astype(np.float64)
            source_lon = (lon + 180) % 360 - 180
            source_lat_indices = source.indexes["latitude"].get_indexer(lat)
            source_lon_indices = source.indexes["longitude"].get_indexer(source_lon)
            if np.any(source_lat_indices < 0) or np.any(source_lon_indices < 0):
                raise ValueError("候选片区原生坐标未在空间分块源中精确找到")
            np.testing.assert_array_equal(source.latitude.values[source_lat_indices], lat)
            np.testing.assert_array_equal(source.longitude.values[source_lon_indices], source_lon)
            block_pairs = set(zip(source_lat_indices // 4, source_lon_indices // 4))
            all_points = mapping.drop_duplicates("native_flat_index")
            all_lat_indices = source.indexes["latitude"].get_indexer(90 - all_points.lat_index.to_numpy() * 0.25)
            all_lon_indices = source.indexes["longitude"].get_indexer((all_points.lon_index.to_numpy() * 0.25 + 180) % 360 - 180)
            if np.any(all_lat_indices < 0) or np.any(all_lon_indices < 0):
                raise ValueError("候选节点集合与空间分块源坐标无法精确对应")
            all_blocks = set(zip(all_lat_indices // 4, all_lon_indices // 4))
            output = pd.DataFrame({
                "time": np.repeat(times, len(points)),
                "native_flat_index": np.tile(points.native_flat_index.to_numpy(), len(times)),
                "latitude": np.tile(lat, len(times)), "longitude_0_360": np.tile(lon, len(times)),
            })
            variables_report = {}
            for target, (name, unit, divisor) in VARIABLES.items():
                if source[name].dims != ("time", "latitude", "longitude") or source[name].attrs.get("units") != unit:
                    raise ValueError(f"空间分块源变量单位或维度不一致：{name}")
                array = source[name].sel(time=times).isel(
                    latitude=xr.DataArray(source_lat_indices, dims="weather_point"),
                    longitude=xr.DataArray(source_lon_indices, dims="weather_point"),
                ).load()
                values = array.values
                if values.shape != (len(times), len(points)) or values.dtype != np.dtype("float32") or not np.isfinite(values).all():
                    raise ValueError(f"试验变量存在缺失或结构异常：{name}")
                converted = values / np.float32(divisor)
                output[target] = converted.reshape(-1)
                variables_report[target] = {"source_variable": name, "source_units": unit,
                    "output_units": "W m**-2" if divisor == 3600 else unit, "divisor": divisor,
                    "raw_selected_values_sha256": digest(values), "output_values_sha256": digest(converted),
                    "minimum": float(converted.min()), "maximum": float(converted.max())}
            # 只允许实际选中空间块的原文请求；检查没有下载全球天气块。
            data_events = [event for event in store.events if event["key"].split("/")[0] in source_names and not event["key"].split("/")[-1].startswith(".")]
            for event in data_events:
                coordinates = event["key"].split("/")[1].split(".")
                if len(coordinates) != 3 or (int(coordinates[1]), int(coordinates[2])) not in block_pairs:
                    raise ValueError("下载对象超出候选片区所需的空间分块")
            target_path = args.output_dir / f"{args.grid_id}-hourly.parquet"
            output.to_parquet(target_path, index=False, compression="zstd")
            pd.testing.assert_frame_equal(pd.read_parquet(target_path), output)
            cached_data = []
            for name in source_names:
                for path in (args.cache_dir / name).glob("*"):
                    if not path.name.startswith("."):
                        cached_data.append({"key": name + "/" + path.name, "bytes": path.stat().st_size, "sha256": file_digest(path)})
            report = {
                "status": "passed", "verified_utc": datetime.now(timezone.utc).isoformat(),
                "source_url": SOURCE_URL, "source_metadata_sha256": hashlib.sha256(metadata_payload).hexdigest(),
                "grid_id": args.grid_id, "grid_landuse_status": str(chosen.landuse_screen_status.iloc[0]),
                "selection_manifest_sha256": file_digest(args.selection_manifest),
                "hours": len(times), "start_utc": str(times[0]), "last_inclusive_utc": str(times[-1]),
                "native_points": len(points), "spatial_blocks": len(block_pairs),
                "all_candidate_spatial_blocks": len(all_blocks), "spatial_chunk_shape": [67584, 4, 4],
                "new_http_data_objects": len(data_events), "new_http_data_bytes": sum(event["bytes"] for event in data_events),
                "cached_data_objects": len(cached_data), "cached_data_bytes": sum(event["bytes"] for event in cached_data),
                "source_objects": cached_data, "http_events_this_run": store.events, "variables": variables_report,
                "output": {"path": str(target_path), "sha256": file_digest(target_path), "bytes": target_path.stat().st_size},
                "qa": {"all_hours_present": True, "all_values_finite": True, "native_coordinates_exact": True,
                       "only_selected_spatial_chunks_requested": True, "parquet_exact_roundtrip": True},
                "radiation_conversion": "ERA5逐小时辐射累计量除以3600秒，时间戳表示此前一小时的平均通量；保留原始分块、原值SHA和转换关系",
                "references": ["https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=analysis_ready_data",
                    "https://raw.githubusercontent.com/ecmwf-training/dss-notebooks/main/datasets/reanalysis-era5-single-levels/arco-access.ipynb",
                    "https://confluence.ecmwf.int/pages/viewpage.action?pageId=185075432"],
                "scope": "通过用地初筛的单个片区下载试验；尚未下载全部候选片区，也未完成最终建设适宜性核查",
            }
            write_json(args.output_dir / "download-validation.json", report)
            print(json.dumps({key: value for key, value in report.items() if key not in ("variables", "source_objects", "http_events_this_run", "references")}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="按候选用地试验ECMWF空间分块ERA5下载，保留原文缓存和验证记录。")
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--grid-id", default="g05_r100_c592")
    parser.add_argument("--start", default="2023-01-01T00:00:00")
    parser.add_argument("--end", default="2026-01-01T01:00:00")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
