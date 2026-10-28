import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
from PIL import Image

from compute_load.model import sha256


STEMS = ("figure1-typical-weekday", "figure2-typical-week", "figure3-annual-envelope", "figure4-power-components")


def verify(project, visual_review_confirmed=False):
    output = project / "results/typical-load-curves-20261028"
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    processed = project / manifest["processed_directory"]
    checks = {}
    expected_clock = pd.date_range("2025-01-01", "2026-01-01", freq="h", inclusive="left", tz="UTC")
    for name, checksum in manifest["processed_sha256"].items():
        path = processed / name
        checks[f"data_sha256:{name}"] = sha256(path) == checksum
        frame = pd.read_parquet(path)
        checks[f"clock:{name}"] = pd.DatetimeIndex(frame.time_utc).equals(expected_clock)
    central = pd.read_csv(output / "annual-summary.csv")
    summary = pd.read_csv(output / "sensitivity-summary.csv")
    baseline_count = 13 if manifest["alibaba_status"] == "included" else 12
    checks["baseline_and_sensitivity_count"] = len(central) == baseline_count and len(summary) == 3 * baseline_count
    checks["baseline_unique"] = not central.curve_id.duplicated().any()
    source_days = central.groupby("demand").source_days.first().to_dict()
    checks["source_days"] = all(source_days[key] == days for key, days in {"Acme": 90, "ASI": 1, "Lisa": 309}.items())
    parameters = pd.read_csv(output / "power-parameters.csv")
    for row in parameters.itertuples():
        key = f"{row.configuration_id}-{row.case}"
        checks[f"parameter_endpoints:{key}"] = bool(np.isclose(row.unit_idle_w,
            row.p_base_w + row.cpu_packages * row.cpu_idle_w_each + row.gpus * row.gpu_idle_w_each)
            and np.isclose(row.unit_full_w, row.p_base_w + row.cpu_packages * row.cpu_full_w_each + row.gpus * row.gpu_full_w_each))
        checks[f"integer_budget:{key}"] = row.units_at_10mw == int(10e6 // row.sizing_reference_w)
    for row in summary.itertuples():
        path = processed / f"{row.curve_id}-{row.case}.parquet"
        frame = pd.read_parquet(path)
        key = f"{row.curve_id}-{row.case}"
        checks[f"IT_energy:{key}"] = bool(np.isclose(row.annual_it_mwh, frame.it_mw.sum(), atol=1e-8))
        values = frame.it_mw if row.configuration_id == "R-GB200" else frame.facility_mw
        checks[f"reported_energy_peak_valley:{key}"] = bool(
            np.isclose(row.reported_annual_mwh, values.sum()) and np.isclose(row.peak_mw, values.max())
            and np.isclose(row.peak_valley_ratio, values.max() / values.min())
            and np.isclose(row.average_load_factor, values.mean() / values.max()))
        if row.configuration_id == "R-GB200":
            checks[f"liquid_missing:{key}"] = bool(frame[["cooling_mw", "facility_mw", "pue"]].isna().all().all()
                and pd.isna(row.annual_facility_mwh) and pd.isna(row.annual_cooling_mwh) and pd.isna(row.annual_pue))
        else:
            checks[f"facility_energy_balance:{key}"] = bool(np.isclose(row.annual_facility_mwh,
                row.annual_it_mwh + row.annual_cooling_mwh + row.annual_other_mwh))
            checks[f"PUE:{key}"] = bool(np.isclose(row.annual_pue, frame.facility_mw.sum() / frame.it_mw.sum()))
    weekdays = pd.read_csv(output / "typical-weekday.csv")
    daily = pd.read_csv(output / "daily-envelope.csv")
    for row in central.itertuples():
        frame = pd.read_parquet(processed / f"{row.curve_id}-central.parquet")
        values = frame.it_mw if row.configuration_id == "R-GB200" else frame.facility_mw
        mask = frame.time_utc.dt.dayofweek.lt(5)
        expected = values.loc[mask].groupby(frame.time_utc.loc[mask].dt.hour).mean().to_numpy()
        actual = weekdays.loc[weekdays.configuration_id.eq(row.configuration_id) & weekdays.demand.eq(row.demand)].sort_values("hour_utc")
        checks[f"weekday_aggregate:{row.curve_id}"] = len(actual) == 24 and bool(np.allclose(actual.reported_mw, expected))
        actual_daily = daily.loc[daily.configuration_id.eq(row.configuration_id) & daily.demand.eq(row.demand)]
        checks[f"365_days:{row.curve_id}"] = len(actual_daily) == 365 and bool(np.isclose(actual_daily["mean"].sum() * 24, values.sum()))
    render_dir = project / ".codex-temp/typical-load-pdf-review"
    render_dir.mkdir(parents=True, exist_ok=True)
    pdfs = []
    for stem in STEMS:
        with Image.open(output / f"{stem}.png") as image:
            dpi = image.info.get("dpi", (0, 0))
            checks[f"PNG_300dpi:{stem}"] = min(dpi) >= 299
        with fitz.open(output / f"{stem}.pdf") as document:
            checks[f"PDF_single_page:{stem}"] = len(document) == 1
            page = document[0]
            extracted = "".join(page.get_text().split())
            checks[f"PDF_Chinese_footer:{stem}"] = "功率参数：公开资料+假设（演示）" in extracted
            fonts = page.get_fonts(full=True)
            checks[f"PDF_embedded_font:{stem}"] = bool(fonts) and all(bool(document.extract_font(font[0])[3]) for font in fonts)
            rendered = render_dir / f"{stem}.png"
            page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(rendered)
            pdfs.append({"pdf": f"{stem}.pdf", "rendered": str(rendered.relative_to(project)), "fonts": [font[3] for font in fonts]})
    if not all(checks.values()):
        raise AssertionError([name for name, passed in checks.items() if not passed])
    validation_path = output / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).isoformat()
    validation["delivery_checked_at_utc"] = stamp
    validation["delivery_checks"] = checks
    validation["pdf_rendering"] = pdfs
    validation["status"] = "passed" if visual_review_confirmed else "numerical_and_export_checks_passed; visual_review_pending"
    validation["visual_review"] = {"reviewer": "Codex", "confirmed": visual_review_confirmed,
        "checked_at_utc": stamp if visual_review_confirmed else None,
        "scope": "四张300dpi PNG与四张PDF渲染；中文、布局、图例、轴标和缺失值标记"}
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["files_sha256"] = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file() and path != manifest_path}
    metadata = project / "data/metadata/typical-load-curves-20261028.json"
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    manifest["metadata_sha256"] = sha256(metadata)
    manifest["code_sha256"]["verify_typical_load_curves.py"] = sha256(Path(__file__))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": validation["status"], "delivery_checks": len(checks), "render_dir": str(render_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从实际小时结果复核汇总、导出PDF并核验交付清单")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--visual-review-confirmed", action="store_true")
    args = parser.parse_args()
    verify(args.project.resolve(), args.visual_review_confirmed)
