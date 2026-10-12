"""Tests for stage 1 (ingest) and stage 2 (aggregate) on small synthetic FIRMS files.

Run from the pipeline/ folder:  python -m pytest -q
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from shift.adapters import firms_csv
from shift.aggregate import read_celldays, run_aggregate, to_celldays
from shift.config import DEFAULT_SETTINGS, load_settings
from shift.grid import cell_centre, cell_index
from shift.ingest import apply_filters, read_detections, run_ingest

MODIS_HEADER = "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight,type"
VIIRS_HEADER = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight,type"
VIIRS_NRT_HEADER = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight"


@pytest.fixture()
def project(tmp_path: Path):
    raw = tmp_path / "data" / "raw" / "test"
    raw.mkdir(parents=True)
    (raw / "fire_archive_M-C61_test.csv").write_text("\n".join([
        MODIS_HEADER,
        "27.1234,-81.4321,320.1,1.0,1.0,2012-03-05,1530,Terra,MODIS,85,6.1,295.0,40.5,D,0",   # kept, high
        "27.1201,-81.4399,318.0,1.0,1.0,2012-03-05,1531,Terra,MODIS,55,6.1,294.0,12.0,D,0",   # kept, same cell
        "27.1200,-81.4300,310.0,1.0,1.0,2012-03-05,1532,Terra,MODIS,20,6.1,290.0,5.0,D,0",    # low confidence
        "19.4000,-155.2900,400.0,1.0,1.0,2012-03-05,0830,Aqua,MODIS,95,6.1,300.0,900.0,N,1",  # volcano
        "60.0000,100.0000,330.0,1.0,1.0,2012-03-06,0405,Aqua,MODIS,70,6.1,280.0,25.0,N,0",    # kept, Aqua night
    ]) + "\n")
    (raw / "fire_archive_SV-C2_test.csv").write_text("\n".join([
        VIIRS_HEADER,
        "27.1250,-81.4350,330.0,0.4,0.4,2012-03-05,1820,N,VIIRS,n,2,295.0,3.2,D,0",           # kept
        "27.1251,-81.4351,331.0,0.4,0.4,2012-03-05,1820,N,VIIRS,h,2,296.0,4.1,D,0",           # kept, duplicated in NRT
        "-10.0000,30.0000,330.0,0.4,0.4,2012-03-05,1100,N,VIIRS,l,2,295.0,1.0,D,0",           # low confidence
        "5.0000,5.0000,330.0,0.4,0.4,2012-03-05,1100,N,VIIRS,n,2,295.0,8.0,D,2",              # static source
    ]) + "\n")
    (raw / "fire_nrt_SV-C2_test.csv").write_text("\n".join([
        VIIRS_NRT_HEADER,
        "27.1251,-81.4351,331.0,0.4,0.4,2012-03-05,1820,N,VIIRS,h,2.0NRT,296.0,4.1,D",        # duplicate of archive row
        "-15.5000,28.2000,340.0,0.4,0.4,2012-03-07,105,N,VIIRS,n,2.0NRT,290.0,6.0,N",         # NRT only, time 0105
    ]) + "\n")

    cfg = yaml.safe_load(DEFAULT_SETTINGS.read_text())
    cfg["processing"]["chunk_rows"] = 2  # force several chunks per file
    cfg["dataset"] = "test"
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(yaml.safe_dump(cfg))
    return load_settings(settings_path, root=tmp_path)


def test_family_detection():
    assert firms_csv.detect_family(MODIS_HEADER.split(",")) == "MODIS"
    assert firms_csv.detect_family(VIIRS_HEADER.split(",")) == "VIIRS"
    with pytest.raises(ValueError):
        firms_csv.detect_family(["latitude", "longitude"])


def test_zip_input_matches_csv_input(tmp_path, project):
    """A FIRMS zip dropped into raw/ gives exactly the same result as its CSVs."""
    import zipfile
    from_csv = run_ingest(project)["totals"]
    raw = project.raw_dir
    with zipfile.ZipFile(raw / "DL_FIRE_test.zip", "w") as zf:
        for f in sorted(raw.glob("*.csv")):
            zf.write(f, f.name)
            f.unlink()
        zf.writestr("Readme.txt", "not a csv")
    report = run_ingest(project)
    assert report["totals"] == from_csv
    assert set(report["files"]) == {"fire_archive_M-C61_test.csv", "fire_archive_SV-C2_test.csv",
                                     "fire_nrt_SV-C2_test.csv"}
    assert sorted(p.name for p in raw.iterdir()) == ["DL_FIRE_test.zip"]  # nothing unpacked into raw/


def test_region_clip(tmp_path, project):
    """A dataset with a boundary drops detections outside it; others are untouched."""
    import json
    box = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
           "geometry": {"type": "Polygon", "coordinates": [[[-82, 27], [-81, 27], [-81, 28], [-82, 28], [-82, 27]]]}}]}
    (tmp_path / "box.geojson").write_text(json.dumps(box))
    project.raw.setdefault("regions", {})["clip"] = {"test": "box.geojson"}
    report = run_ingest(project)
    t = report["totals"]
    assert t["outside_region"] > 0
    det = read_detections(project)
    assert det["lat"].between(27, 28).all() and det["lon"].between(-82, -81).all()
    # Florida boundary shipped with the repo loads and contains Orlando, not Atlanta.
    from shift import region
    fl = load_settings(DEFAULT_SETTINGS, dataset="florida")
    assert list(region.inside(fl, [-81.38, -84.39], [28.54, 33.75])) == [True, False]
    assert region.inside(load_settings(DEFAULT_SETTINGS, dataset="global"), [0], [0]) is None


def test_viirs_file_with_modis_style_columns(tmp_path, project):
    """Real Florida VIIRS download (Oct 2026): `brightness`/`bright_t31` columns,
    satellite and instrument both "SNPP". Must still be read as VIIRS."""
    f = tmp_path / "fire_archive_SV-C2_florida.csv"
    f.write_text("\n".join([
        MODIS_HEADER,
        "27.01686,-80.13492,298.55,0.36,0.58,2012-01-20,0622,SNPP,SNPP,n,2,283.53,1.29,N,0",
        "27.01654,-80.13119,307.4,0.36,0.58,2012-01-20,0622,SNPP,SNPP,h,2,282.91,0.9,N,0",
    ]) + "\n")
    df = pd.concat(firms_csv.read(f, project, 10))
    assert set(df["sensor_family"]) == {"VIIRS"}
    assert set(df["satellite"]) == {"S-NPP"}
    assert list(df["confidence_class"]) == ["nominal", "high"]


def test_grid_roundtrip_and_edges():
    r = 0.05
    lat = np.array([27.1234, -89.999, 90.0, 0.0])
    lon = np.array([-81.4321, -180.0, 180.0, 0.0])
    cid = cell_index(lat, lon, r)
    clat, clon = cell_centre(cid, r)
    assert np.all(np.abs(clat - lat) <= r / 2 + 1e-9)
    # 180 deg longitude clips into the last column; centre is within one cell.
    assert np.all(np.abs(clon - lon) <= r + 1e-9)
    assert cid.min() >= 0 and cid.max() < (180 / r) * (360 / r)


def test_ingest_filters_and_dedup(project):
    report = run_ingest(project)
    t = report["totals"]
    assert t["rows_read"] == 11
    assert t["confidence_filtered"] == 2
    assert t["non_vegetation_type"] == 2
    assert t["duplicates_removed"] == 1
    assert t["rows_final"] == 6

    det = read_detections(project)
    assert len(det) == 6
    assert set(det["satellite"]) == {"Terra", "Aqua", "S-NPP"}
    # The duplicated VIIRS row survives once, from the archive file.
    dup = det[(det["lat"] == 27.1251) & (det["lon"] == -81.4351)]
    assert len(dup) == 1 and dup["source_product"].iloc[0] == "VIIRS_SNPP_archive"
    # NRT row with time "105" parses as 01:05 UTC and has unknown type.
    nrt = det[det["source_product"] == "VIIRS_SNPP_NRT"].iloc[0]
    assert nrt["acq_datetime_utc"].hour == 1 and nrt["acq_datetime_utc"].minute == 5
    assert not nrt["type_known"] and nrt["fire_type"] == -1
    # MODIS confidence classes
    modis = det[det["sensor_family"] == "MODIS"].set_index("frp_mw")["confidence_class"]
    assert modis[40.5] == "high" and modis[12.0] == "nominal"


def test_aggregate_celldays(project):
    run_ingest(project)
    report = run_aggregate(project)
    cd = read_celldays(project)
    assert report["detections_in"] == 6
    # Two Terra detections fall in one cell on 5 Mar 2012.
    terra = cd[(cd["satellite"] == "Terra")]
    assert len(terra) == 1
    row = terra.iloc[0]
    assert row["n_detections"] == 2
    assert row["frp_sum_mw"] == pytest.approx(52.5)
    assert row["frp_max_mw"] == pytest.approx(40.5)
    assert row["n_high_conf"] == 1
    # Both VIIRS archive detections share one cell with the Terra pair.
    snpp = cd[(cd["satellite"] == "S-NPP") & (cd["acq_date"] == pd.Timestamp("2012-03-05"))]
    assert len(snpp) == 1 and snpp.iloc[0]["cell_id"] == row["cell_id"]
    assert snpp.iloc[0]["n_detections"] == 2
    assert int(cd["n_detections"].sum()) == 6


def test_unknown_type_can_be_dropped(project):
    project.raw["filters"]["keep_unknown_type"] = False
    df = pd.concat(firms_csv.read(project.raw_dir / "fire_nrt_SV-C2_test.csv", project, 10))
    kept, counts = apply_filters(df, project)
    assert len(kept) == 0 and counts["unknown_type_dropped"] == 2


def test_empty_aggregation_keeps_schema():
    empty = to_celldays(pd.DataFrame(), 0.05)
    assert len(empty) == 0 and "cell_id" in empty.columns


def test_full_pipeline_runs_end_to_end(project):
    from shift.calibrate import run_calibrate
    run_ingest(project)
    run_aggregate(project)
    report = run_calibrate(project)
    # The tiny sample has one MODIS/VIIRS pair, too few to fit, so the stage skips cleanly.
    assert report["pairs_total"] == 1 and "skipped" in report
