"""Fix 9: the harmonised record, end to end on a synthetic world with a known answer.

The same fires burn every year 2010-2014 in two tiles. S-NPP sees all of them; Terra and Aqua
each see part, so raw MODIS counts are much lower. After harmonisation the MODIS years must sit
on the S-NPP level (no 2012 jump), a whole-world S-NPP outage must be filled from NOAA-20, a
regional data loss must be left out of the factor fits, and every row must say where its value
came from.
"""
import numpy as np
import pandas as pd
import yaml

from shift.aggregate import to_celldays, write_celldays
from shift.config import DEFAULT_SETTINGS, load_settings
from shift.factors import data_loss_keys, run_factors
from shift.harmonise import SOURCES, run_harmonise
from shift.nodata import run_nodata

P = {"S-NPP": ("VIIRS", 1.0), "NOAA-20": ("VIIRS", 0.95), "Aqua": ("MODIS", 0.45), "Terra": ("MODIS", 0.35)}


def _settings(tmp_path):
    cfg = yaml.safe_load(DEFAULT_SETTINGS.read_text())
    cfg["dataset"] = "global"
    cfg["counts"]["source_segments"] = [["MODIS", 2000, 2019], ["Aqua", 2020, 2100]]
    cfg["factors"].update({"min_ref_cells_per_month": 20, "overlap_years": [2012, 2014],
                           "backward_fit_years": [2013, 2014], "backward_test_years": [2012, 2012],
                           "bootstrap_samples": 20})
    cfg["nodata"].update({
        "regional": {**cfg["nodata"]["regional"], "min_partner_cells": 5, "min_expected_cells": 8},
        "links": [{"target": "S-NPP", "substitute": "NOAA-20", "years": [2012, 2014]},
                  {"target": "S-NPP", "substitute": "D19", "years": [2012, 2014]},
                  {"target": "MODIS", "substitute": "Aqua", "years": [2010, 2011]},
                  {"target": "MODIS", "substitute": "Terra", "years": [2010, 2011]}],
        "record": [{"from": "2010-01-01", "to": "2012-01-19", "target": "MODIS", "fill": ["Aqua", "Terra"]},
                   {"from": "2012-01-20", "to": "2100-12-31", "target": "S-NPP", "fill": ["NOAA-20", "D19"]}],
        "replicates": 1})
    cfg["harmonise"]["gate"].update({"before_years": [2010, 2011], "after_years": [2013, 2014]})
    p = tmp_path / "settings.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return load_settings(p, root=tmp_path)


def _world(s):
    rng = np.random.default_rng(9)
    days = pd.date_range("2010-01-01", "2014-12-31")
    snpp_off = set(pd.date_range("2013-06-01", "2013-06-20"))
    fl_loss = set(pd.date_range("2013-09-10", "2013-09-12"))           # S-NPP lost over Florida only
    rows = []
    for day in days:
        for tile, (blat, blon) in {"fl": (25.02, -81.48), "af": (8.02, 20.02)}.items():
            n = int(min(70, max(2, 25 * rng.lognormal(0, 0.6))))
            for f in range(n):
                lat, lon = blat + 0.05 * (f % 14), blon + 0.05 * (f // 14)
                power = float(rng.lognormal(2.3, 0.8))
                for sat, (fam, p) in P.items():
                    if sat in ("S-NPP", "NOAA-20") and day < pd.Timestamp("2012-01-20"):
                        continue
                    if sat == "S-NPP" and (day in snpp_off or (tile == "fl" and day in fl_loss)):
                        continue
                    if rng.random() < p:
                        rows.append((day, lat, lon, sat, fam, power * (1.5 if fam == "MODIS" else 1.0)))
    det = pd.DataFrame(rows, columns=["acq_date", "lat", "lon", "satellite", "sensor_family", "frp_mw"])
    det["daynight"], det["confidence_class"], det["type_known"] = "D", "nominal", True
    for _, d in det.groupby("sensor_family"):
        write_celldays(to_celldays(d, 0.05), s.celldays_dir)
    cov = []
    for sat in P:
        for day in days:
            if sat in ("S-NPP", "NOAA-20") and day < pd.Timestamp("2012-01-20"):
                continue
            cov.append({"acq_date": day, "satellite": sat, "status": "none" if (sat == "S-NPP" and day in snpp_off) else "ok"})
    cp = s.path(s["coverage"]["path"])
    cp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cov).to_parquet(cp, index=False)


def test_harmonise_end_to_end(tmp_path):
    s = _settings(tmp_path)
    _world(s)
    run_nodata(s)                                   # zone-day status, fill factors, data-loss list
    keys = data_loss_keys(s)
    assert keys is not None and len(keys) >= 3     # the Florida loss is found and will be left out
    fac = run_factors(s)
    assert fac["data_loss_zone_days_left_out"] >= 3
    rep = run_harmonise(s)
    assert "note" not in rep, rep

    out = s.processed_dir / "l3"
    l3 = pd.concat([pd.read_parquet(p) for p in out.glob("year=*/month=*/part.parquet")], ignore_index=True)
    assert set(l3["src"].unique()) <= set(SOURCES)
    assert {0, 1, 3} <= set(l3["src"].unique())                       # S-NPP, NOAA-20 fill, MODIS x k
    jun = l3[(l3["date"] >= "2013-06-01") & (l3["date"] <= "2013-06-20")]
    assert len(jun) and set(jun["src"].unique()) == {1}                # the outage is filled from NOAA-20

    yt = pd.read_csv(out / "yearly_totals.csv")
    g = yt[yt["region"] == "global"].set_index("year")
    ratio = g.loc[2011, "cells_h"] / g.loc[2013, "cells_h"]
    assert 0.85 < ratio < 1.15                                         # MODIS years on the S-NPP level
    raw_jump = g.loc[2013, "S-NPP_cells"] / g.loc[2011, "MODIS_cells"]
    assert raw_jump > 1.3                                              # ... while the raw record jumps

    gate = rep["gate_2012_step"]
    assert gate["global"] is not None and abs(gate["global"]["cells_gap_pct"]) < 15
    unc = pd.read_parquet(out / "l3_uncertainty.parquet")
    assert (unc["range_pct"] >= 0).all() and {"cells", "frp"} <= set(unc["variable"])
    assert rep["anomaly_rule"].get("adopted") in ("plain", "era-fair", None)
