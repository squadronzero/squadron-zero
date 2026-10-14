"""Worldwide check on synthetic fires in two far-apart tiles.

Every tile: Aqua sees a steady 50% of fires, Terra 50% until 2015 then fewer each year.
Fire power: VIIRS reads true power, MODIS 1.4x, and MODIS misses most small fires, so the
v1 line (fitted elsewhere) is wrong and the ratio of all fire power is right.
"""
import numpy as np
import pandas as pd
import yaml

from shift.aggregate import to_celldays, write_celldays
from shift.config import DEFAULT_SETTINGS, load_settings
from shift.tiles import run_worldwide, tile_name


def _settings(tmp_path):
    cfg = yaml.safe_load(DEFAULT_SETTINGS.read_text())
    cfg["dataset"] = "global"
    cfg["regions"]["test"] = {"florida": cfg["regions"]["test"]["florida"]}
    cfg["selection"]["candidates"] = {"MODIS": [["MODIS", 2000, 2100]], "Aqua": [["Aqua", 2000, 2100]],
                                      "Terra": [["Terra", 2000, 2100]],
                                      "MODIS_then_Aqua": [["MODIS", 2000, 2015], ["Aqua", 2016, 2100]]}
    cfg["counts"]["source_segments"] = [["MODIS", 2000, 2015], ["Aqua", 2016, 2100]]
    cfg["worldwide"].update({"min_ref_cells_per_month": 5, "overlap_years": [2012, 2017],
                             "backward_fit_years": [2015, 2017], "backward_test_years": [2012, 2014]})
    p = tmp_path / "settings.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return load_settings(p, root=tmp_path)


def test_tile_names():
    assert tile_name(20, -90, 10) == "N20W090" and tile_name(-10, 0, 10) == "S10E000"


def test_worldwide_end_to_end(tmp_path):
    s = _settings(tmp_path)
    rng = np.random.default_rng(11)
    rows, level = [], {}
    for day in pd.date_range("2011-01-01", "2017-12-31"):
        for base_lat, base_lon in ((25.02, -81.02), (5.02, 20.02)):
            key = (base_lat, day.year, day.month)
            if key not in level:
                level[key] = int(rng.integers(8, 20))
            terra_p = 0.5 if day.year <= 2015 else 0.5 - 0.2 * (day.year - 2015)
            for f in range(level[key]):
                lat, lon = base_lat + 0.1 * (f % 20), base_lon + 0.1 * (day.dayofyear % 25)
                power = float(rng.lognormal(2.3, 1.0))
                seen = 0.9 if power > 15 else 0.3
                if rng.random() < 0.5 * seen * 2:
                    rows.append((day, lat, lon, "Aqua", "MODIS", power * 1.4))
                if rng.random() < terra_p * seen * 2:
                    rows.append((day, lat, lon, "Terra", "MODIS", power * 1.4))
                if day >= pd.Timestamp("2012-01-20"):
                    rows.append((day, lat, lon, "S-NPP", "VIIRS", power))
    det = pd.DataFrame(rows, columns=["acq_date", "lat", "lon", "satellite", "sensor_family", "frp_mw"])
    det["daynight"], det["confidence_class"], det["type_known"] = "D", "nominal", True
    for _, d in det.groupby("sensor_family"):
        write_celldays(to_celldays(d, 0.05), s.celldays_dir)

    rep = run_worldwide(s)
    assert rep["tiles_tested"] == 3          # Florida fires straddle two tiles
    assert rep["d21"]["verdict"] == "CONFIRMED"
    d21 = {r["candidate"]: r for r in rep["d21"]["summary"]}
    assert d21["ratio_totals"]["fire_weighted_score"] > d21["v1_equation"]["fire_weighted_score"]
    d19 = {r["candidate"]: r for r in rep["d19"]["summary"]}
    assert rep["d19"]["chosen"] == "MODIS_then_Aqua"
    assert d19["Terra"]["fire_weighted_score"] < d19["MODIS_then_Aqua"]["fire_weighted_score"]
    assert rep["d19"]["verdict"] in {"CONFIRMED", "CHECK"}
    out = s.processed_dir / "counts"
    t = pd.read_csv(out / "worldwide_d19_tiles.csv")
    assert {"N20W090", "N00E020"} <= set(t["region"]) and {"lat", "lon", "score"} <= set(t.columns)
