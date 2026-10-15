"""Fix 3b: the selection stage, on synthetic fires with a known answer.

Fires vary from month to month (so some months are unusual). Aqua sees a steady 50% of
fires; Terra sees 50% until 2014 and then fewer every year (orbital drift). Half the
places have small fires, half large. The rule fixed before the run must not pick Terra, every
test must produce output, and the backward test must show Terra's drift as an error.
"""
import numpy as np
import pandas as pd
import yaml

from shift.aggregate import to_celldays, write_celldays
from shift.config import DEFAULT_SETTINGS, load_settings
from shift.counts import run_counts
from shift.selection import classify, pick_winner, run_selection


def _settings(tmp_path):
    cfg = yaml.safe_load(DEFAULT_SETTINGS.read_text())
    cfg["dataset"] = "global"
    cfg["periods"].update({"calibration_start": "2012-01-20", "calibration_end": "2014-12-31",
                           "holdout_start": "2015-01-01", "holdout_end": "2017-12-31"})
    cfg["regions"]["test"] = {"florida": cfg["regions"]["test"]["florida"]}
    cfg["selection"].update({
        "candidates": {"MODIS": [["MODIS", 2000, 2100]], "Aqua": [["Aqua", 2000, 2100]],
                       "Terra": [["Terra", 2000, 2100]], "MODIS_then_Aqua": [["MODIS", 2000, 2014], ["Aqua", 2015, 2100]]},
        "overlap_years": [2012, 2017], "backward_fit_years": [2015, 2017], "backward_test_years": [2012, 2014],
        "forward_fit_years": [2012, 2014], "forward_test_years": [2015, 2017],
        "modis_first_year": 2009, "terra_stable_until": 2014, "bootstrap_samples": 200})
    p = tmp_path / "settings.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return load_settings(p, root=tmp_path)


def _write_fires(s):
    rng = np.random.default_rng(5)
    rows = []
    level = {}
    for day in pd.date_range("2009-01-01", "2017-12-31"):
        key = (day.year, day.month)
        if key not in level:
            level[key] = int(rng.integers(10, 31))          # fires per day this month
        terra_p = 0.5 if day.year <= 2014 else 0.5 - 0.12 * (day.year - 2014)
        for f in range(level[key]):
            lat = 25.02 + 0.1 * (f % 30)
            lon = -81.02 - 0.1 * (day.dayofyear % 25)
            frp = 2.0 if f % 2 else 60.0                       # small and large fire places
            if rng.random() < 0.5:
                rows.append((day, lat, lon, "Aqua", "MODIS", frp))
            if rng.random() < terra_p:
                rows.append((day, lat, lon, "Terra", "MODIS", frp))
            if day >= pd.Timestamp("2012-01-20"):
                rows.append((day, lat, lon, "S-NPP", "VIIRS", frp))
    det = pd.DataFrame(rows, columns=["acq_date", "lat", "lon", "satellite", "sensor_family", "frp_mw"])
    det["daynight"], det["confidence_class"], det["type_known"] = "D", "nominal", True
    for _, d in det.groupby("sensor_family"):
        write_celldays(to_celldays(d, 0.05), s.celldays_dir)


def test_classify_bands():
    mu, sd = pd.Series([np.log1p(100.0)] * 4), pd.Series([0.5] * 4)
    vals = pd.Series([100 * np.exp(-0.6) - 1, 100.0, 100 * np.exp(0.75) - 1, 100 * np.exp(1.2) - 1])
    assert list(classify(vals, mu, sd)) == ["low", "normal", "unusual", "critical"]


def test_rule_is_applied_as_written():
    summ = pd.DataFrame({"candidate": ["A", "B", "C"], "score": [80.0, 79.0, 90.0],
                         "global_backward_error_pct": [2.0, -3.0, 25.0],
                         "global_drift_pct_per_year": [0.9, 0.1, 0.0]})
    win, why = pick_winner(summ, {"max_global_backward_error_pct": 10, "tie_margin_points": 2})
    assert win == "B"                 # C is out (25% error); A and B tie within 2 points; B drifts less
    assert "lowest global drift" in why


def test_selection_end_to_end(tmp_path):
    s = _settings(tmp_path)
    _write_fires(s)
    run_counts(s)
    rep = run_selection(s)
    assert rep["winner"] in {"Aqua", "MODIS_then_Aqua"}
    summ = {r["candidate"]: r for r in rep["summary"]}
    assert abs(summ["Terra"]["global_backward_error_pct"]) > 20          # drift caught backwards
    assert abs(summ["Aqua"]["global_backward_error_pct"]) < 5
    assert abs(summ["Terra"]["global_drift_pct_per_year"]) > abs(summ["Aqua"]["global_drift_pct_per_year"])
    era = {r["region"]: r for r in rep["modis_era"]}["florida"]
    assert abs(era["aqua_terra_drift_stable_pct_per_year"]) < 3
    assert era["aqua_terra_drift_late_pct_per_year"] > 10
    assert 1.4 < era["step_2002_merged_vs_terra"] < 1.9                    # merged sees ~75% vs 50%
    out = s.processed_dir / "counts"
    names = {p.name for p in out.iterdir()}
    assert {"selection_summary.csv", "selection_scores.csv", "selection_bootstrap_k.csv",
            "selection_fire_size.csv", "selection_modis_era.csv"} <= names
    size = pd.read_csv(out / "selection_fire_size.csv")
    assert {"small", "large"} <= set(size["size"])
    boot = pd.read_csv(out / "selection_bootstrap_k.csv")
    assert (boot["k_lo"] <= boot["k"]).all() and (boot["k"] <= boot["k_hi"]).all()
