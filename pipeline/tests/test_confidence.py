"""Fix 5: the confidence asymmetry.

1. confscan keeps ONLY low-confidence vegetation-fire detections, in their own folder, and
   leaves the main record alone.
2. confidence: a synthetic world where S-NPP starts labelling far more fires "low" from 2015
   (MODIS never does). Dropping low then makes S-NPP look smaller after 2015, so factors fitted
   on 2015-2017 predict 2012-2014 badly. Adding the low detections back removes the problem,
   and the pre-registered rule must pick a variant that keeps low-confidence S-NPP.
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from shift.aggregate import read_celldays, to_celldays, write_celldays
from shift.config import DEFAULT_SETTINGS, load_settings
from shift.confidence import low_dir, run_confidence, run_confscan

MODIS_HEADER = "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight,type"
VIIRS_HEADER = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight,type"


def _zip(path: Path, folder: str, rows: list[str], header: str) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{folder}/x.csv", "\n".join([header, *rows]) + "\n")


def test_confscan_keeps_only_low(tmp_path):
    server = tmp_path / "server"
    server.mkdir()
    _zip(server / "modis_2020_all_countries.zip", "modis/2020", [
        "10.0,20.0,330,1,1,2020-07-01,1000,Aqua,MODIS,70,6.03,290,25.0,D,0",     # nominal: not here
        "10.0,20.0,330,1,1,2020-07-01,1000,Aqua,MODIS,20,6.03,290,5.0,D,0",      # low: kept
        "19.4,-155.29,400,1,1,2020-07-01,0830,Aqua,MODIS,10,6.03,300,9.0,N,1",   # low but volcano: dropped
    ], MODIS_HEADER)
    _zip(server / "viirs-snpp_2020_all_countries.zip", "viirs-snpp/2020", [
        "27.125,-81.435,330,0.4,0.4,2020-03-05,1820,N,VIIRS,l,2,295,1.5,D,0",    # low: kept
        "27.126,-81.436,330,0.4,0.4,2020-03-05,1820,N,VIIRS,l,2,295,2.5,D,0",    # low, same cell: one cell-day
        "27.125,-81.435,331,0.4,0.4,2020-03-05,1820,N,VIIRS,h,2,296,4.1,D,0",    # high: not here
    ], VIIRS_HEADER)
    cfg = yaml.safe_load(DEFAULT_SETTINGS.read_text())
    cfg["dataset"] = "global"
    cfg["fetch"]["base_url"] = server.as_uri()
    cfg["fetch"]["last_year"] = 2020
    for p in ("MODIS", "VIIRS"):
        cfg["fetch"]["products"][p]["first_year"] = 2020
    cfg["fetch"]["tmp_dir"] = str(tmp_path)
    sp = tmp_path / "settings.yaml"
    sp.write_text(yaml.safe_dump(cfg))
    s = load_settings(sp, root=tmp_path)

    rep = run_confscan(s, families=["MODIS", "VIIRS"])
    assert rep["years"]["MODIS_2020"]["low_kept"] == 1
    assert rep["years"]["VIIRS_2020"]["low_kept"] == 2
    low = pd.read_parquet(low_dir(s))
    assert len(low) == 2 and set(low["satellite"].astype(str)) == {"Aqua", "S-NPP"}
    assert low.loc[low["satellite"].astype(str) == "S-NPP", "frp_sum_mw"].iloc[0] == 4.0
    assert read_celldays(s).empty                          # main record untouched
    again = run_confscan(s, families=["MODIS", "VIIRS"])   # resumable
    assert set(again["skipped"]) == {"MODIS_2020", "VIIRS_2020"}


def test_confidence_picks_low_when_labelling_changes(tmp_path):
    cfg = yaml.safe_load(DEFAULT_SETTINGS.read_text())
    cfg["dataset"] = "global"
    cfg["counts"]["source_segments"] = [["Aqua", 2000, 2100]]
    cfg["factors"].update({"min_ref_cells_per_month": 10, "overlap_years": [2012, 2017],
                           "backward_fit_years": [2015, 2017], "backward_test_years": [2012, 2014]})
    sp = tmp_path / "settings.yaml"
    sp.write_text(yaml.safe_dump(cfg))
    s = load_settings(sp, root=tmp_path)

    rng = np.random.default_rng(5)
    main, low = [], []
    for month in pd.date_range("2012-01-01", "2017-12-01", freq="MS"):
        n = int(rng.integers(20, 60))                       # fires this month, varies so categories differ
        p_low = 0.05 if month.year <= 2014 else 0.45        # S-NPP labelling change in 2015
        for day in (1, 11, 21):
            for f in range(n):
                lat, lon = 25.02 + 0.05 * (f % 40), -81.02 + 0.05 * (day // 10)
                power = float(rng.lognormal(2.0, 0.8))
                date = month + pd.Timedelta(days=day - 1)
                if f % 2 == 0:
                    main.append((date, lat, lon, "Aqua", "MODIS", power * 1.3))
                (low if rng.random() < p_low else main).append((date, lat, lon, "S-NPP", "VIIRS", power))
    for rows, root in ((main, s.celldays_dir), (low, low_dir(s))):
        det = pd.DataFrame(rows, columns=["acq_date", "lat", "lon", "satellite", "sensor_family", "frp_mw"])
        det["daynight"], det["confidence_class"], det["type_known"] = "D", "nominal", True
        for _, d in det.groupby("sensor_family"):
            write_celldays(to_celldays(d, 0.05), root)

    rep = run_confidence(s)
    v = {r["variant"]: r for r in rep["variants"]}
    assert rep["winner"] in {"all", "viirs_all"}
    assert abs(v["current"]["global_backward_error_frp_pct"]) > 15          # the hidden step shows
    assert abs(v["viirs_all"]["global_backward_error_frp_pct"]) < 10
    assert rep["low_share_trend"]["S-NPP"]["cells_pct_points_per_year"] > 3  # S-NPP low share rises
    out = s.processed_dir / "counts"
    assert (out / "confidence_variants.csv").exists() and (out / "confidence_low_share_yearly.csv").exists()
