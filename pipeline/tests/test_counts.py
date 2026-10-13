"""Phase 2 fix 1: detection-count harmonisation, on synthetic fires with a known answer.

Every day the same number of fires burn in Florida. MODIS sees 60% of them with 1
detection each; VIIRS (from 2012) sees all of them with 4 detections each (finer pixels).
So raw detections jump about 6.7x at 2012, active cells about 1.67x, and after
harmonisation the jump must be gone (step about 1.0, like the real MODIS-only change).
"""
import numpy as np
import pandas as pd
import yaml

from shift.aggregate import to_celldays, write_celldays
from shift.config import DEFAULT_SETTINGS, load_settings
from shift.counts import run_counts


def _settings(tmp_path):
    cfg = yaml.safe_load(DEFAULT_SETTINGS.read_text())
    cfg["dataset"] = "global"
    cfg["periods"].update({"calibration_start": "2012-01-20", "calibration_end": "2013-12-31",
                           "holdout_start": "2014-01-01", "holdout_end": "2015-12-31"})
    cfg["regions"]["test"] = {"florida": cfg["regions"]["test"]["florida"]}
    p = tmp_path / "settings.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return load_settings(p, root=tmp_path)


def _write_fires(s):
    rng = np.random.default_rng(1)
    rows = []
    for day in pd.date_range("2009-01-01", "2015-12-31"):
        for f in range(10):                                  # 10 fires a day, each in its own cell
            lat, lon = 26.02 + 0.1 * f, -81.02 - 0.1 * (day.dayofyear % 20)   # cell centres, away from edges
            if rng.random() < 0.6:                           # MODIS sees 60% of fires
                rows.append((day, lat, lon, "Aqua" if f % 2 else "Terra", "MODIS"))
            if day >= pd.Timestamp("2012-01-20"):            # VIIRS sees every fire, 4 pixels each
                rows += [(day, lat + 0.001 * i, lon, "S-NPP", "VIIRS") for i in range(4)]
    det = pd.DataFrame(rows, columns=["acq_date", "lat", "lon", "satellite", "sensor_family"])
    det["frp_mw"], det["daynight"], det["confidence_class"], det["type_known"] = 10.0, "D", "nominal", True
    for _, d in det.groupby("sensor_family"):
        write_celldays(to_celldays(d, 0.05), s.celldays_dir)


def test_counts_removes_the_2012_jump(tmp_path):
    s = _settings(tmp_path)
    _write_fires(s)
    rep = run_counts(s)

    k = rep["k_per_region"]["florida"]
    assert 1.5 < k < 1.85                       # VIIRS active cells per MODIS active cell, about 1/0.6

    step = {r["region"]: r for r in rep["step_2012"]}["florida"]
    assert step["step_raw"] > 5                 # raw detections: about 6.7x jump
    assert 0.9 < step["step_harmonised"] < 1.1  # harmonised: jump gone
    assert abs(step["step_harmonised"] - step["real_change_modis"]) < 0.1

    hold = {r["region"]: r for r in rep["holdout"]}["florida"]
    assert abs(hold["total_error_pct"]) < 5     # 2014-2015 never used in the fit
    assert hold["raw_ratio_detections"] > 5 and 1.5 < hold["raw_ratio_active_cells"] < 1.85

    out = s.processed_dir / "counts"
    assert {"daily.parquet", "factors.csv", "validation_holdout.csv", "step_2012.csv"} <= {p.name for p in out.iterdir()}
    daily = pd.read_parquet(out / "daily.parquet")
    assert {"global", "florida"} <= set(daily["region"]) and {"MODIS", "S-NPP"} <= set(daily["group"])


def test_continuity_ties_noaa20_to_snpp_and_fills_the_gap(tmp_path):
    """NOAA-20 sees 90% of the fires S-NPP sees (k about 1.11). S-NPP is silent for one
    month in 2025: NOAA-20 x k must stand in for it."""
    from shift.counts import run_continuity
    s = _settings(tmp_path)
    s.raw["continuity"].update({"fit_years": [2019, 2023], "test_years": [2024, 2025]})
    rng = np.random.default_rng(2)
    gap = (pd.Timestamp("2025-05-01"), pd.Timestamp("2025-05-31"))
    rows = []
    for day in pd.date_range("2019-01-01", "2025-12-31"):
        for f in range(10):
            lat, lon = 26.02 + 0.1 * f, -81.02 - 0.1 * (day.dayofyear % 20)
            if not gap[0] <= day <= gap[1]:
                rows.append((day, lat, lon, "S-NPP", "VIIRS"))
            if rng.random() < 0.9:
                rows.append((day, lat, lon, "NOAA-20", "VIIRS"))
    det = pd.DataFrame(rows, columns=["acq_date", "lat", "lon", "satellite", "sensor_family"])
    det["frp_mw"], det["daynight"], det["confidence_class"], det["type_known"] = 10.0, "D", "nominal", True
    write_celldays(to_celldays(det, 0.05), s.celldays_dir)
    # coverage table as qc would write it
    from shift.qc import coverage, daily_totals
    cov = coverage(daily_totals(s), 31, 0.25)
    cp = s.path(s["coverage"]["path"]); cp.parent.mkdir(parents=True, exist_ok=True)
    cov.to_parquet(cp, index=False)

    run_counts(s)
    rep = run_continuity(s)
    k = rep["k_cells_per_region"]["florida"]
    assert 1.07 < k < 1.15
    test = {r["region"]: r for r in rep["test"]}["florida"]
    assert abs(test["total_error_pct"]) < 3
    frp = {r["region"]: r for r in rep["frp"]}["florida"]
    assert 1.07 < frp["k_frp"] < 1.15 and abs(frp["test_total_frp_error_pct"]) < 3
    assert rep["gapfill"]["days"] == 31 and rep["gapfill"]["first"] == "2025-05-01"
