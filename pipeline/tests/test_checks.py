"""Tests for the static-source mask, the qc stage and the cross-check."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from shift.aggregate import read_celldays, to_celldays, write_celldays
from shift.config import DEFAULT_SETTINGS, load_settings
from shift.grid import cell_index
from shift.ingest import run_ingest
from shift.qc import run_qc

NRT_HEADER = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight"


def make_settings(tmp_path, dataset="global", **over):
    cfg = yaml.safe_load(DEFAULT_SETTINGS.read_text())
    cfg["dataset"] = dataset
    cfg["regions"]["clip"] = {}
    for k, v in over.items():
        cfg[k].update(v)
    p = tmp_path / "settings.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return load_settings(p, root=tmp_path)


def test_static_mask_drops_unlabelled_detections_in_known_source_cells(tmp_path):
    s = make_settings(tmp_path)
    flare = (29.0, 47.5)       # a gas-flare cell
    path = s.path(s["static_mask"]["path"])
    path.parent.mkdir(parents=True)
    pd.DataFrame({"cell_id": [int(cell_index(pd.Series([flare[0]]), pd.Series([flare[1]]), 0.05)[0])]}).to_parquet(path)
    raw = s.raw_dir
    raw.mkdir(parents=True)
    (raw / "fire_nrt_SV-C2_x.csv").write_text("\n".join([
        NRT_HEADER,
        f"{flare[0]+0.01},{flare[1]+0.01},330,0.4,0.4,2026-08-01,1000,N,VIIRS,n,2.0NRT,295,9.0,N",   # in flare cell
        "-10.0,30.0,330,0.4,0.4,2026-08-01,1000,N,VIIRS,n,2.0NRT,295,5.0,D",                        # savanna fire
    ]) + "\n")
    t = run_ingest(s)["totals"]
    assert t["static_source_masked"] == 1 and t["rows_final"] == 1
    # without a mask file nothing is masked
    path.unlink()
    assert run_ingest(s)["totals"].get("static_source_masked", 0) == 0


def _celldays(rows):
    det = pd.DataFrame(rows, columns=["acq_date", "lat", "lon", "satellite", "sensor_family", "frp_mw"])
    det["acq_date"] = pd.to_datetime(det["acq_date"])
    det["daynight"], det["confidence_class"], det["type_known"] = "D", "nominal", True
    return pd.concat([to_celldays(d, 0.05) for _, d in det.groupby("sensor_family")])


def test_qc_finds_a_silent_stretch(tmp_path):
    s = make_settings(tmp_path, qc={"window_days": 11, "step_checks": []})
    days = pd.date_range("2026-03-01", "2026-05-31")
    rows = []
    for d in days:
        silent = pd.Timestamp("2026-04-28") <= d <= pd.Timestamp("2026-05-03")
        for i in range(0 if silent else 20):
            rows.append((d, -10 + i * 0.1, 30.0, "S-NPP", "VIIRS", 5.0))
        rows.append((d, 0.0, 20.0, "Aqua", "MODIS", 10.0))
    write_celldays(_celldays(rows), s.celldays_dir)
    rep = run_qc(s)
    gap = rep["longest_gaps"][0]
    assert gap["satellite"] == "S-NPP" and gap["start"] == "2026-04-28" and gap["end"] == "2026-05-03"
    assert gap["days_none"] == 6
    assert rep["days_by_status"]["Aqua"].get("none", 0) == 0
    assert (s.processed_dir / "qc" / "daily_coverage.parquet").exists()
    # with a mask covering the Aqua cell, every Aqua detection counts as "in mask"
    from shift.grid import cell_index
    mp = s.path(s["static_mask"]["path"]); mp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cell_id": [int(cell_index(pd.Series([0.0]), pd.Series([20.0]), 0.05)[0])]}).to_parquet(mp)
    rep2 = run_qc(s)
    assert rep2["real_fires_in_mask_pct"]["MODIS"] == 100.0 and rep2["real_fires_in_mask_pct"]["VIIRS"] == 0.0


def test_crosscheck_identical_routes_pass(tmp_path):
    from shift.crosscheck import run_crosscheck
    g = make_settings(tmp_path, checks={"crosscheck": {"reference_dataset": "florida",
                       "region": "config/regions/florida.geojson", "inset_deg": 0.05, "years": [2015, 2015]}})
    import shutil
    (tmp_path / "config" / "regions").mkdir(parents=True)
    shutil.copy(Path(DEFAULT_SETTINGS).parent / "regions" / "florida.geojson", tmp_path / "config" / "regions")
    rows = [("2015-03-01", 28.5, -81.5, "Aqua", "MODIS", 30.0), ("2015-03-01", 28.5, -81.5, "S-NPP", "VIIRS", 9.0),
            ("2015-03-02", 33.7, -84.4, "Aqua", "MODIS", 50.0)]   # Atlanta: outside Florida, ignored
    cd = _celldays(rows)
    write_celldays(cd, g.celldays_dir)
    f = load_settings(tmp_path / "settings.yaml", root=tmp_path, dataset="florida")
    write_celldays(cd[cd["cell_lat"] < 31], f.celldays_dir)
    rep = run_crosscheck(g)
    assert rep["verdict"] == "PASS"
    assert rep["totals"]["in_both"] == 2 and rep["totals"]["same_count_pct"] == 100


def test_bad_days_only_counts_shared_period(tmp_path):
    from shift.coverage import bad_days
    s = make_settings(tmp_path)
    cov = pd.concat([
        pd.DataFrame({"acq_date": pd.date_range("2010-01-01", "2010-01-10"), "satellite": "Aqua", "status": "ok"}),
        pd.DataFrame({"acq_date": pd.date_range("2010-01-05", "2010-01-10"), "satellite": "S-NPP",
                      "status": ["ok", "none", "ok", "low", "ok", "ok"]}),
    ])
    p = s.path(s["coverage"]["path"])
    p.parent.mkdir(parents=True)
    cov.to_parquet(p)
    bad = bad_days(s, ["Aqua", "S-NPP"])
    assert [d.day for d in bad] == [6, 8]       # days 1-4 are before S-NPP existed: not counted
    (tmp_path / "x").mkdir()
    assert bad_days(make_settings(tmp_path / "x"), ["Aqua"]) is None   # no table yet
