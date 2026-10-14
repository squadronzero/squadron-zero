"""D21 factors for every place, with the gap fixes, on synthetic fires.

Tile A (Florida area): plain fires, VIIRS sees all, MODIS (Aqua) half.
Tile B (Sahara gas field): the same fires plus 20 gas-flare cells that both sensors report
every day until 2014 and never after (as if NASA started labelling them). A ratio fitted on
2015-2017 then predicts 2012-2014 badly there unless the flare cells are masked in every year.
Tile C: a thin tile with a few fires, which must still get factors through the fallback chain.
"""
import numpy as np
import pandas as pd
import yaml

from shift.aggregate import to_celldays, write_celldays
from shift.config import DEFAULT_SETTINGS, load_settings
from shift.factors import run_factors
from shift.grid import cell_index


def _settings(tmp_path):
    cfg = yaml.safe_load(DEFAULT_SETTINGS.read_text())
    cfg["dataset"] = "global"
    cfg["regions"]["test"] = {"florida": cfg["regions"]["test"]["florida"]}
    cfg["counts"]["source_segments"] = [["MODIS", 2000, 2015], ["Aqua", 2016, 2100]]
    cfg["factors"].update({"min_ref_cells_per_month": 20, "overlap_years": [2012, 2017],
                           "backward_fit_years": [2015, 2017], "backward_test_years": [2012, 2014],
                           "bootstrap_samples": 30, "watch_tiles": ["N20E000"]})
    p = tmp_path / "settings.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return load_settings(p, root=tmp_path)


def test_factors_end_to_end(tmp_path):
    s = _settings(tmp_path)
    rng = np.random.default_rng(21)
    rows, level = [], {}
    flares = [(25.02 + 0.1 * i, 5.02) for i in range(20)]
    for day in pd.date_range("2011-01-01", "2017-12-31"):
        for tile, (blat, blon, n) in {"A": (25.02, -81.02, None), "B": (25.52, 5.52, None), "C": (-35.02, 150.02, 2)}.items():
            if tile == "C" and day.day != 1:                                   # thin: fires on one day a month
                continue
            key = (tile, day.year, day.month)
            if key not in level:
                level[key] = n if n else int(rng.integers(10, 25))
            for f in range(level[key]):
                lat, lon = blat + 0.1 * (f % 20), blon + 0.1 * (day.dayofyear % 25)
                power = float(rng.lognormal(2.3, 1.0))
                if rng.random() < 0.5:
                    rows.append((day, lat, lon, "Aqua", "MODIS", power * 1.4))
                if day >= pd.Timestamp("2012-01-20"):
                    rows.append((day, lat, lon, "S-NPP", "VIIRS", power))
        if day.year <= 2014:
            for lat, lon in flares:
                rows.append((day, lat, lon, "Aqua", "MODIS", 30.0))
                if day >= pd.Timestamp("2012-01-20"):
                    rows.append((day, lat, lon, "S-NPP", "VIIRS", 30.0))
    det = pd.DataFrame(rows, columns=["acq_date", "lat", "lon", "satellite", "sensor_family", "frp_mw"])
    det["daynight"], det["confidence_class"], det["type_known"] = "D", "nominal", True
    for _, d in det.groupby("sensor_family"):
        write_celldays(to_celldays(d, 0.05), s.celldays_dir)
    mp = s.path(s["static_mask"]["path"])
    mp.parent.mkdir(parents=True, exist_ok=True)
    ids = cell_index(np.array([a for a, _ in flares]), np.array([b for _, b in flares]), 0.05)
    pd.DataFrame({"cell_id": np.asarray(ids, dtype="int64")}).to_parquet(mp, index=False)

    rep = run_factors(s)
    steps = {r["step"]: r for r in rep["steps"]}
    assert rep["chosen"]["mask"] is True                                      # the flare fix is kept
    assert steps["+ non-fire mask in every year"]["kept"]
    assert rep["tiles_tested"] >= 2 and rep["tiles_with_fire"] >= 4         # Florida fires straddle two tiles
    fac = pd.read_parquet(s.processed_dir / "counts" / "factors_world.parquet")
    assert {"cells", "frp"} <= set(fac["variable"])
    thin = fac[fac["tile"] == "S40E150"]
    assert len(thin) and thin["k"].notna().all() and not thin["tested_tile"].any()
    a = fac[(fac["tile"] == "N20W090") & (fac["variable"] == "cells")]
    assert 1.6 < a["k"].median() < 2.4                                        # MODIS sees half the fires
    assert (a["k_lo"] <= a["k"] + 1e-9).all() and (a["k"] <= a["k_hi"] + 1e-9).all()
    assert set(fac["fit_years"]) == {"2012-2015", "2016-2017"}                # each piece on all its years
