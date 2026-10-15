"""Fix 8: per-place work on a synthetic world.

Tile N20W090 (Florida): plain fires. Tile N20E080 (Bangladesh box): plain fires, the spotlight.
Tile N20E000 (Sahara gas field): fires plus 20 flare cells that both sensors report every day
until 2014 and never after. NASA labelled 10 of them in two years (D22 removes those) and the
other 10 in ONE year only, so D22 leaves them in and the backward test (predict 2012-2014 from
later years) overshoots there. The stronger mask in flare tiles must remove them.
Tile S40E150: thin, a few fires a month.
"""
import numpy as np
import pandas as pd
import pytest
import yaml

from shift.aggregate import to_celldays, write_celldays
from shift.config import DEFAULT_SETTINGS, load_settings
from shift.counts import run_counts
from shift.grid import cell_index
from shift.places import decide_source, mask_tiers, run_places


def _settings(tmp_path):
    cfg = yaml.safe_load(DEFAULT_SETTINGS.read_text())
    cfg["dataset"] = "global"
    cfg["regions"]["test"] = {"florida": cfg["regions"]["test"]["florida"],
                              "bangladesh": cfg["regions"]["test"]["bangladesh"]}
    cfg["counts"]["source_segments"] = [["MODIS", 2000, 2015], ["Aqua", 2016, 2100]]
    cfg["periods"].update({"calibration_end": "2015-12-31", "holdout_start": "2016-01-01", "holdout_end": "2017-12-31"})
    cfg["factors"].update({"min_ref_cells_per_month": 20, "overlap_years": [2012, 2017],
                           "backward_fit_years": [2015, 2017], "backward_test_years": [2012, 2014],
                           "watch_tiles": ["N20E000"]})
    cfg["places"].update({"selection_years": [2015, 2017], "flare_share": 0.05})
    p = tmp_path / "settings.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return load_settings(p, root=tmp_path)


def _world(s):
    rng = np.random.default_rng(8)
    flares = [(25.02 + 0.1 * i, 5.02) for i in range(20)]
    rows, level = [], {}
    places = {"fl": (25.02, -81.02, None), "sa": (25.52, 5.52, None), "bd": (23.02, 89.02, None), "th": (-35.02, 150.02, 2)}
    for day in pd.date_range("2011-01-01", "2017-12-31"):
        for tile, (blat, blon, n) in places.items():
            if tile == "th" and day.day != 1:
                continue
            key = (tile, day.year, day.month)
            if key not in level:
                season = 1.0 + (2.5 if tile == "bd" and day.month in (2, 3, 4) else 0.0)
                level[key] = n if n else int(rng.integers(8, 20) * season)
            for f in range(level[key]):
                lat, lon = blat + 0.1 * (f % 20), blon + 0.1 * (day.dayofyear % 25)
                power = float(rng.lognormal(2.3, 1.0))
                for sat, fam, p, mult in (("Aqua", "MODIS", 0.45, 1.4), ("Terra", "MODIS", 0.35, 1.4)):
                    if rng.random() < p:
                        rows.append((day, lat, lon, sat, fam, power * mult))
                if day >= pd.Timestamp("2012-01-20"):
                    rows.append((day, lat, lon, "S-NPP", "VIIRS", power))
        if day.year <= 2014:
            for lat, lon in flares:
                for sat, fam in (("Aqua", "MODIS"), ("Terra", "MODIS")):
                    rows.append((day, lat, lon, sat, fam, 40.0))
                if day >= pd.Timestamp("2012-01-20"):
                    rows.append((day, lat, lon, "S-NPP", "VIIRS", 40.0))
    det = pd.DataFrame(rows, columns=["acq_date", "lat", "lon", "satellite", "sensor_family", "frp_mw"])
    det["daynight"], det["confidence_class"], det["type_known"] = "D", "nominal", True
    for _, d in det.groupby("sensor_family"):
        write_celldays(to_celldays(d, 0.05), s.celldays_dir)
    ids = np.asarray(cell_index(np.array([a for a, _ in flares]), np.array([b for _, b in flares]), 0.05), "int64")
    src = s.path(s["static_mask"]["sources_dir"])
    src.mkdir(parents=True, exist_ok=True)
    two, one = ids[:10], ids[10:]
    pd.DataFrame({"cell_id": np.concatenate([two, one]), "type": 2, "n_detections": 100}).to_parquet(src / "MODIS_2013.parquet", index=False)
    pd.DataFrame({"cell_id": two, "type": 2, "n_detections": 100}).to_parquet(src / "MODIS_2014.parquet", index=False)
    pd.DataFrame({"cell_id": two}).to_parquet(s.path(s["static_mask"]["path"]), index=False)     # the D22 mask
    conf = s.processed_dir / "counts"
    conf.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"tile": ["N20W090", "N20E000"], "gap_trend_pct_points_per_year": [0.1, -0.9]}).set_index("tile").to_csv(
        conf / "confidence_tile_asymmetry.csv")
    return ids


def test_mask_tiers(tmp_path):
    s = _settings(tmp_path)
    ids = _world(s)
    t = mask_tiers(s)
    assert set(t.loc[ids[:10]]) == {"m2"} and set(t.loc[ids[10:]]) == {"m1"}
    assert (t == "ring").sum() > 0 and not t.index.duplicated().any()


def test_places_end_to_end(tmp_path):
    s = _settings(tmp_path)
    _world(s)
    run_counts(s)                                   # the spotlight reads counts/daily.parquet
    rep = run_places(s)
    out = s.processed_dir / "places"

    assert "N20E000" in rep["flare_tiles"] and "N20W090" not in rep["flare_tiles"]
    steps = pd.read_csv(out / "mask_steps.csv")
    assert len(steps) == 3 and bool(steps["kept"].iloc[1])                   # single-year flares removed
    assert rep["mask_chosen"].startswith("D22+m1")
    w = pd.read_csv(out / "mask_watch_tiles.csv", index_col=0)
    assert abs(w.loc["N20E000", "chosen_backward_error_pct"]) < abs(w.loc["N20E000", "D22_backward_error_pct"])

    src = pd.read_csv(out / "source_per_tile_test.csv")
    assert len(src) == 3 and rep["source_choice"] in ("D19 everywhere", "per tile")

    lt = pd.read_csv(out / "borrowed_factor_test.csv")
    assert len(lt) == 4 and (lt["half_width_95_log"] >= 0).all()

    pc = pd.read_csv(out / "place_choices.csv", index_col=0)
    assert pc.loc["N20E000", "flare_tile"] and pc.loc["N20E000", "confidence_drift_pct"] == pytest.approx(8.1)
    assert set(pc["modis_source"].dropna()) <= {"D19", "merged"}

    sp = rep["spotlight"]
    assert sp and sp["region"] == "bangladesh" and "N20E080" in sp["tiles"]
    cal = pd.read_csv(out / "bangladesh_calendar.csv")
    assert {"year", "month", "value", "source", "class"} <= set(cal.columns)
    assert set(cal["class"]) <= {"low", "normal", "unusual", "critical"}
    assert (cal[cal["year"] >= 2012]["source"] == "S-NPP").all()


def test_source_rule():
    d19 = {"backward_agreement": 75.0, "tiles_within_limit_pct": 70.0}
    rule = {"source_gain_points": 2}
    assert decide_source(d19, {"backward_agreement": 78.0, "tiles_within_limit_pct": 70.0}, rule)[0] == "per tile"
    assert decide_source(d19, {"backward_agreement": 76.5, "tiles_within_limit_pct": 75.0}, rule)[0] == "D19 everywhere"
    assert decide_source(d19, {"backward_agreement": 79.0, "tiles_within_limit_pct": 65.0}, rule)[0] == "D19 everywhere"
