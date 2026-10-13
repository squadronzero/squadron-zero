"""Tests for the fetch stage using fake yearly world zips served from local files (file:// URLs)."""
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from shift.aggregate import read_celldays, run_aggregate
from shift.config import DEFAULT_SETTINGS, load_settings
from shift.fetch import run_fetch
from shift.ingest import run_ingest

MODIS_HEADER = "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight,type"
VIIRS_HEADER = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight,type"


def make_zip(path: Path, folder: str, files: dict[str, list[str]], header: str) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, rows in files.items():
            zf.writestr(f"{folder}/{name}", "\n".join([header, *rows]) + "\n")


@pytest.fixture()
def project(tmp_path: Path):
    server = tmp_path / "server"
    server.mkdir()
    # MODIS 2020: two country files. The same 0.05 deg cell (27.10-27.15, -81.45 to -81.40)
    # appears in both on the same day, as happens for cells on a border.
    make_zip(server / "modis_2020_all_countries.zip", "modis/2020", {
        "modis_2020_CountryA.csv": [
            "27.1234,-81.4321,320,1,1,2020-03-05,1530,Terra,MODIS,85,6.03,295,40.5,D,0",
            "27.1201,-81.4399,318,1,1,2020-03-05,1531,Terra,MODIS,55,6.03,294,12.0,D,0",
            "19.4000,-155.2900,400,1,1,2020-03-05,0830,Aqua,MODIS,95,6.03,300,900.0,N,1",   # volcano, dropped
        ],
        "modis_2020_CountryB.csv": [
            "27.1240,-81.4310,321,1,1,2020-03-05,1530,Terra,MODIS,90,6.03,296,60.0,D,0",   # same cell as A
            "10.0000,20.0000,330,1,1,2020-07-01,1000,Aqua,MODIS,70,6.03,290,25.0,D,0",
            "10.0000,20.0000,330,1,1,2020-07-01,1000,Aqua,MODIS,20,6.03,290,5.0,D,0",      # low confidence
        ],
    }, MODIS_HEADER)
    # VIIRS 2020 only exists under the second prefix, to test the fallback.
    make_zip(server / "viirs_snpp_2020_all_countries.zip", "viirs-snpp/2020", {
        "viirs-snpp_2020_CountryA.csv": [
            "27.1250,-81.4350,330,0.4,0.4,2020-03-05,1820,N,VIIRS,n,2,295,3.2,D,0",
            "27.1251,-81.4351,331,0.4,0.4,2020-03-05,1820,N,VIIRS,h,2,296,4.1,D,0",
        ],
    }, VIIRS_HEADER)

    cfg = yaml.safe_load(DEFAULT_SETTINGS.read_text())
    cfg["dataset"] = "global"
    cfg["fetch"]["base_url"] = server.as_uri()
    cfg["fetch"]["last_year"] = 2021          # 2021 does not exist on the "server"
    cfg["fetch"]["products"]["MODIS"]["first_year"] = 2020
    cfg["fetch"]["products"]["VIIRS"]["first_year"] = 2020
    cfg["fetch"]["tmp_dir"] = str(tmp_path / "scratch")
    (tmp_path / "scratch").mkdir()
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(yaml.safe_dump(cfg))
    return load_settings(settings_path, root=tmp_path)


def test_fetch_builds_celldays_and_merges_borders(project):
    report = run_fetch(project)
    assert set(report["years"]) == {"MODIS_2020", "VIIRS_2020"}
    assert set(report["missing"]) == {"MODIS_2021", "VIIRS_2021"}
    assert report["years"]["VIIRS_2020"]["url"].endswith("viirs_snpp_2020_all_countries.zip")

    m = report["years"]["MODIS_2020"]
    assert m["files"] == 2 and m["rows_read"] == 6 and m["rows_kept"] == 4
    assert m["non_vegetation_type"] == 1 and m["confidence_filtered"] == 1

    cd = read_celldays(project)
    terra = cd[(cd["satellite"] == "Terra") & (cd["acq_date"] == pd.Timestamp("2020-03-05"))]
    assert len(terra) == 1                       # border cell merged into one row
    row = terra.iloc[0]
    assert row["n_detections"] == 3
    assert row["frp_sum_mw"] == pytest.approx(112.5)
    assert row["frp_max_mw"] == pytest.approx(60.0)
    assert row["frp_mean_mw"] == pytest.approx(37.5)
    assert row["n_high_conf"] == 2
    assert int(cd["n_detections"].sum()) == 6    # 4 MODIS + 2 VIIRS

    # Raw downloads are not kept anywhere.
    assert not any(project.root.joinpath("scratch").iterdir())
    assert not project.raw_dir.exists() or not any(project.raw_dir.rglob("*.csv"))


def test_fetch_resumes_and_skips_done_years(project):
    run_fetch(project, years=[2020], families=["MODIS"])
    again = run_fetch(project)
    assert "MODIS_2020" in again["skipped"]
    assert "VIIRS_2020" in again["years"]
    marker = json.loads((project.processed_dir / "_fetch_done" / "MODIS_2020.json").read_text())
    assert marker["partitions"] == ["sensor_family=MODIS/year=2020/month=3", "sensor_family=MODIS/year=2020/month=7"]


def test_aggregate_keeps_fetched_years_and_blocks_overlap(project):
    run_fetch(project, families=["MODIS"])
    before = len(read_celldays(project))

    # A recent-months form download (after last_year) is added beside the yearly files.
    project.raw_dir.mkdir(parents=True, exist_ok=True)
    (project.raw_dir / "fire_nrt_M-C61_recent.csv").write_text("\n".join([
        "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight",
        "5.0,5.0,320,1,1,2026-02-01,1200,Terra,MODIS,85,6.1NRT,295,30.0,D",
    ]) + "\n")
    run_ingest(project)
    run_aggregate(project)
    cd = read_celldays(project)
    assert len(cd) == before + 1                 # fetched partitions untouched, 2026 added

    run_aggregate(project)                       # rerun replaces only its own partitions
    assert len(read_celldays(project)) == before + 1

    # A form download that overlaps a fetched year must be refused (double counting).
    (project.raw_dir / "fire_nrt_M-C61_recent.csv").write_text("\n".join([
        "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight",
        "5.0,5.0,320,1,1,2020-03-02,1200,Terra,MODIS,85,6.1NRT,295,30.0,D",
    ]) + "\n")
    run_ingest(project)
    with pytest.raises(RuntimeError, match="double counting"):
        run_aggregate(project)


def test_dataset_folders_are_separate(tmp_path):
    g = load_settings(root=tmp_path, dataset="global")
    f = load_settings(root=tmp_path, dataset="florida")
    assert g.raw_dir == tmp_path / "data" / "raw" / "global"
    assert f.processed_dir == tmp_path / "data" / "processed" / "florida"


def test_scan_versions_static_sources_and_fingerprints(project):
    from shift.scan import run_scan
    run_fetch(project)
    rep = run_scan(project)
    m = rep["years"]["MODIS_2020"]
    assert m["versions"] == {"6.03": 6}
    assert m["same_file_as_fetch"] is True
    assert len(m["sha256"]) == 64
    assert m["static_detections"] == {"volcano": 1}          # the Kilauea row, type 1
    fmarker = json.loads((project.processed_dir / "_fetch_done" / "MODIS_2020.json").read_text())
    assert fmarker["sha256"] == m["sha256"]                    # fetch marker carries the fingerprint
    assert rep["version_summary"]["MODIS"]["changes_at"] == []
    # one year only, so with min_years=2 nothing qualifies; with 1 the volcano cell does
    assert rep["static_mask"]["cells"] == 0
    project.raw["static_mask"]["min_years"] = 1
    from shift.scan import build_static_mask
    assert build_static_mask(project)["cells"] == 1
    # rerun skips finished years
    assert set(run_scan(project)["skipped"]) == {"MODIS_2020", "VIIRS_2020"}
