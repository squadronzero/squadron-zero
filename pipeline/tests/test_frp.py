"""Fix 4: the intercept problem, on synthetic fires with a known answer.

Every fire has a true power. VIIRS sees every fire. MODIS (Aqua here) sees 90% of large
fires but only 30% of small ones, and reads power with noise. A line fitted on paired
cell-days (v1 style) cannot bring back the small fires MODIS missed, and its intercept adds
power per MODIS cell; the ratio of all fire power per region and month can. The rule must
pick ratio_totals and the v1 Florida line must be far off.
"""
import numpy as np
import pandas as pd
import yaml

from shift.aggregate import to_celldays, write_celldays
from shift.config import DEFAULT_SETTINGS, load_settings
from shift.counts import run_counts
from shift.frp import fit_line, run_frp


def _settings(tmp_path):
    cfg = yaml.safe_load(DEFAULT_SETTINGS.read_text())
    cfg["dataset"] = "global"
    cfg["periods"].update({"calibration_start": "2012-01-20", "calibration_end": "2014-12-31",
                           "holdout_start": "2015-01-01", "holdout_end": "2017-12-31"})
    cfg["regions"]["test"] = {"florida": cfg["regions"]["test"]["florida"]}
    cfg["counts"]["source_segments"] = [["MODIS", 2000, 2014], ["Aqua", 2015, 2100]]
    cfg["frp"].update({"overlap_years": [2012, 2017], "backward_fit_years": [2015, 2017],
                       "backward_test_years": [2012, 2014], "forward_fit_years": [2012, 2014],
                       "forward_test_years": [2015, 2017]})
    p = tmp_path / "settings.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return load_settings(p, root=tmp_path)


def test_line_fit_from_sums():
    x = np.array([1.0, 2.0, 3.0, 4.0]); y = 2 * x + 1
    st = pd.DataFrame([{"region": "r", "group": "g", "year": 2020, "month": 1, "n": 4,
                        "sx": x.sum(), "sy": y.sum(), "sxx": (x * x).sum(), "sxy": (x * y).sum()}])
    ols = fit_line(st, "g", [2020], "ols_cellday", (0.3, 24.7)).loc["r"]
    assert abs(ols["a"] - 2) < 1e-9 and abs(ols["b"] - 1) < 1e-9
    assert fit_line(st, "g", [2020], "ratio_paired", (0.3, 24.7)).loc["r", "a"] == y.sum() / x.sum()


def test_frp_end_to_end(tmp_path):
    s = _settings(tmp_path)
    rng = np.random.default_rng(7)
    rows, level = [], {}
    for day in pd.date_range("2010-01-01", "2017-12-31"):
        key = (day.year, day.month)
        if key not in level:
            level[key] = int(rng.integers(10, 31))
        for f in range(level[key]):
            lat, lon = 25.02 + 0.1 * (f % 30), -81.02 - 0.1 * (day.dayofyear % 25)
            power = float(rng.lognormal(2.3, 1.0))                     # median about 10 MW
            if rng.random() < (0.9 if power > 15 else 0.3):
                rows.append((day, lat, lon, "Aqua", "MODIS", power * 1.4 * rng.lognormal(0, 0.4)))
            if day >= pd.Timestamp("2012-01-20"):
                rows.append((day, lat, lon, "S-NPP", "VIIRS", power))
    det = pd.DataFrame(rows, columns=["acq_date", "lat", "lon", "satellite", "sensor_family", "frp_mw"])
    det["daynight"], det["confidence_class"], det["type_known"] = "D", "nominal", True
    for _, d in det.groupby("sensor_family"):
        write_celldays(to_celldays(d, 0.05), s.celldays_dir)

    run_counts(s)
    rep = run_frp(s)
    summ = {r["candidate"]: r for r in rep["summary"]}
    assert rep["winner"] == "ratio_totals"
    assert abs(summ["ratio_totals"]["global_backward_error_pct"]) < 5
    assert abs(summ["v1_equation"]["global_backward_error_pct"]) > 10
    assert rep["intercept_share_pct"]["v1_equation"]["florida"] > 50       # mostly intercept, not fire
    assert summ["ratio_paired"]["global_backward_error_pct"] < -10          # misses the small fires
    assert rep["intercept_share_pct"]["ols_cellday"]["florida"] > 5
    names = {p.name for p in (s.processed_dir / "counts").iterdir()}
    assert {"frp_summary.csv", "frp_scores.csv", "frp_pair_stats.csv", "frp_lines.csv"} <= names
