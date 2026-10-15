"""Phase 2 package (15 Oct): version test, region test, daily skill and day basis, hybrid fills,
the checks stage, outside-data reducers and provenance, on synthetic worlds with known answers."""
import json

import numpy as np
import pandas as pd
import yaml

from shift.aggregate import to_celldays, write_celldays
from shift.checks import runs_from, run_checks, zscores
from shift.config import DEFAULT_SETTINGS, load_settings
from shift.daily import run_daily, wmedian
from shift.external import gfed_grid, pick_unit, region_totals, to_fine
from shift.factors import Cube, run_factors
from shift.harmonise import run_harmonise, scale_month
from shift.nodata import run_nodata
from shift.provenance import settings_hash, stamp
from shift.resolution import run_resolution, version_test
from shift.solar import local_day_shift

AREAS = {"area_a": (25.02, -81.48, 0.60, 0.50), "area_b": (28.02, -86.48, 0.35, 0.25)}


def _settings(tmp_path):
    cfg = yaml.safe_load(DEFAULT_SETTINGS.read_text())
    cfg["dataset"] = "global"
    cfg["regions"]["test"] = {"area_a": {"bbox": [-82.5, 24.0, -80.0, 26.5], "regime": "a"},
                              "area_b": {"bbox": [-87.5, 27.5, -85.0, 29.9], "regime": "b"}}
    cfg["counts"]["source_segments"] = [["MODIS", 2000, 2013], ["Aqua", 2014, 2100]]
    cfg["factors"].update({"min_ref_cells_per_month": 20, "overlap_years": [2012, 2014],
                           "backward_fit_years": [2013, 2014], "backward_test_years": [2012, 2012],
                           "bootstrap_samples": 10})
    cfg["resolution"].update({"min_ref_cells_per_month": 10})
    cfg["harmonise"]["gate"].update({"before_years": [2010, 2011], "after_years": [2013, 2014]})
    cfg["nodata"].update({
        "regional": {**cfg["nodata"]["regional"], "min_partner_cells": 5, "min_expected_cells": 8},
        "links": [{"target": "S-NPP", "substitute": "NOAA-20", "years": [2013, 2014]},
                  {"target": "MODIS", "substitute": "Aqua", "years": [2010, 2011]}],
        "record": [{"from": "2009-01-01", "to": "2012-01-19", "target": "MODIS", "fill": ["Aqua"]},
                   {"from": "2012-01-20", "to": "2100-12-31", "target": "S-NPP", "fill": ["NOAA-20"]}],
        "replicates": 1})
    cfg["daily"].update({"test_years": [2012, 2012], "fit_years": [2013, 2013]})
    pc = cfg["phase2_checks"]
    pc.update({"baseline_years": [2009, 2014],
               "final_test": {"months": ["2015-01", "2015-09"], "error_limit_pct": 10, "allow_rerun": False},
               "coverage": {**pc["coverage"], "calibrate_years": [2013, 2013], "verify_years": [2012, 2012]},
               "events": {"burst_a": {"bbox": [-82.5, 24.0, -80.0, 26.5], "months": ["2010-08"]}},
               "gfed": {"before_years": [2010, 2011], "after_years": [2013, 2014]}})
    p = tmp_path / "settings.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return load_settings(p, root=tmp_path)


def _world(s):
    rng = np.random.default_rng(5)
    days = pd.date_range("2009-01-01", "2015-12-31")
    burst = (pd.Timestamp("2010-08-01"), pd.Timestamp("2010-08-31"))
    rows = []
    for day in days:
        for name, (blat, blon, pa, pt) in AREAS.items():
            n = int(min(60, max(3, 18 * rng.lognormal(0, 0.4))))
            if name == "area_a" and burst[0] <= day <= burst[1]:
                n = 140                                               # a known event: fire far above normal
            for f in range(n):
                lat, lon = blat + 0.05 * (f % 12), blon + 0.05 * (f // 12)
                power = float(rng.lognormal(2.3, 0.8))
                dn = "N" if f % 7 == 0 else "D"
                if day >= pd.Timestamp("2012-01-20"):
                    rows.append((day, lat, lon, "S-NPP", "VIIRS", power, dn))
                if day >= pd.Timestamp("2013-01-01") and rng.random() < 0.95:
                    rows.append((day, lat, lon, "NOAA-20", "VIIRS", power, dn))
                if rng.random() < pa:
                    rows.append((day, lat, lon, "Aqua", "MODIS", power * 1.5, dn))
                if rng.random() < pt:
                    rows.append((day, lat, lon, "Terra", "MODIS", power * 1.5, dn))
    det = pd.DataFrame(rows, columns=["acq_date", "lat", "lon", "satellite", "sensor_family", "frp_mw", "daynight"])
    det["confidence_class"], det["type_known"] = "nominal", True
    for _, d in det.groupby("sensor_family"):
        write_celldays(to_celldays(d, 0.05), s.celldays_dir)
    cov = [{"acq_date": d, "satellite": sat, "status": "ok"} for sat in ("Terra", "Aqua") for d in days]
    cov += [{"acq_date": d, "satellite": "S-NPP", "status": "ok"} for d in days if d >= pd.Timestamp("2012-01-20")]
    cov += [{"acq_date": d, "satellite": "NOAA-20", "status": "ok"} for d in days if d >= pd.Timestamp("2013-01-01")]
    cp = s.path(s["coverage"]["path"])
    cp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cov).to_parquet(cp, index=False)


def test_package_end_to_end(tmp_path):
    s = _settings(tmp_path)
    _world(s)
    run_nodata(s)
    assert (s.processed_dir / "nodata" / "fill_choice.json").exists()
    run_factors(s)
    res = run_resolution(s)
    assert res["adopted"] and (s.processed_dir / "counts" / "fine_monthly.parquet").exists()
    assert "version_test" in res and "region_test" in res
    choice = json.loads((s.processed_dir / "counts" / "resolution_choice.json").read_text())
    assert {"fit_years", "region_overrides", "region_flags"} <= set(choice)

    d = run_daily(s)
    assert "note" not in d, d
    assert d["day_basis"] in ("utc", "local")
    assert set(d["labels_before_2012"]) == {"day", "week", "month"}
    sk = pd.read_csv(s.processed_dir / "daily" / "skill_by_scale.csv")
    cell = sk[(sk["basis"] == "utc") & (sk["units"] == "cell") & (sk["variable"] == "cells")].set_index("scale")
    assert cell.loc["month", "fw_median_error_pct"] <= cell.loc["day", "fw_median_error_pct"]   # months are easier than days
    shifts = pd.read_csv(s.processed_dir / "daily" / "date_shift_share.csv")
    assert shifts["moved"].sum() > 0                    # night passes west of 22.5 W move for Terra

    h = run_harmonise(s)
    assert "note" not in h, h
    for f in ("monthly_regions.csv", "daily_regions.parquet", "scale_tile_months.parquet", "record_info.json"):
        assert (s.processed_dir / "l3" / f).exists(), f
    info = json.loads((s.processed_dir / "l3" / "record_info.json").read_text())
    assert info["provenance"]["settings_sha256"] == settings_hash(s.raw)
    mr = pd.read_csv(s.processed_dir / "l3" / "monthly_regions.csv")
    assert {"global", "area_a", "event:burst_a"} <= set(mr["region"])
    assert (mr["cells_h_est"] >= mr["cells_h"] - 1e-6).all()      # scaling only ever adds the unseen days

    c = run_checks(s)
    assert "summary" in c and "provenance" in c
    assert c["anomaly_rule"].get("adopted") in ("plain", "era-fair"), c["anomaly_rule"]
    ft = c["final_test_2026"]
    assert "predictors" in ft and "Aqua x k" in ft["predictors"], ft
    assert abs(ft["predictors"]["Aqua x k"]["cells"]["global_error_pct"]) < 15          # steady world: Aqua x k holds
    assert (s.processed_dir / "checks" / "final_test_2026.json").exists()
    again = run_checks(s)
    assert "stored result" in again["final_test_2026"]["note"]                           # opened once only
    assert again["coverage"].get("accepted") is not None, again["coverage"]               # a rerun works too
    unc = pd.read_parquet(s.processed_dir / "l3" / "l3_uncertainty.parquet")
    assert "month_range_pct" in unc and (unc["month_range_pct"] >= unc["range_pct"] - 0.11).all()
    cov = c["coverage"]
    assert "cells" in cov, cov
    assert all(0 <= r["calibrated_coverage_pct"] <= 100 for r in cov["cells"])
    sm = pd.DataFrame(c["simple_methods"]).set_index("method")
    assert sm.loc["SHIFT", "score"] > sm.loc["no harmonisation", "score"]
    assert abs(sm.loc["SHIFT", "global_backward_error_pct_cells"]) < abs(sm.loc["no harmonisation", "global_backward_error_pct_cells"])
    ev = c["known_events"]
    assert ev["table"][0]["SHIFT frp hit"] is True                                      # the burst is found
    cp = c["critical_periods"]
    assert cp["events_found_as_dated_runs"]["burst_a"], cp                              # ... as a dated run
    assert "note" in c["burned_area"] and "note" in c["cloud"]                         # no outside data yet

    # outside data as stage external writes it: burned area that follows the (steady) true fire,
    # and a cloud field; the raw stitched record must step against burned area, SHIFT much less
    ext = s.processed_dir / "external"
    ext.mkdir(parents=True, exist_ok=True)
    mr = pd.read_csv(s.processed_dir / "l3" / "monthly_regions.csv")
    mr.assign(ba_ha=mr["MODIS_cells"].clip(lower=1) * 25.0)[["region", "year", "month", "ba_ha"]].to_csv(
        ext / "gfed_monthly_regions.csv", index=False)
    fm = pd.read_parquet(s.processed_dir / "counts" / "fine_monthly.parquet")[["lat0", "lon0", "year", "month"]].drop_duplicates()
    fm.assign(cloud=np.random.default_rng(1).uniform(0.2, 0.9, len(fm))).to_parquet(ext / "cloud_monthly_fine.parquet", index=False)
    c3 = run_checks(s)
    g = pd.DataFrame(c3["burned_area"]["table"]).set_index("region").loc["global"]
    assert g["cells_step_raw_stitched"] > 1.3
    assert abs(np.log(g["cells_step_shift"])) < abs(np.log(g["cells_step_raw_stitched"]))
    assert c3["cloud"]["flag_shown"] in (True, False) and c3["cloud"]["cloudy_share_all_months_pct"] is not None


def test_version_test_prefers_same_version_years_when_processing_changes():
    rng = np.random.default_rng(3)
    rows = []
    for cell in range(40):
        lat0, lon0 = -10 + 2.5 * (cell % 8), 20 + 2.5 * (cell // 8)
        for y in range(2012, 2020):
            for mth in range(1, 13):
                real = 200 * rng.lognormal(0, 0.15)
                ratio = 0.5 if y <= 2017 else 0.35                   # processing change in 2018
                rows += [{"lat0": lat0, "lon0": lon0, "size": "all", "in_mask": False, "group": "S-NPP", "year": y, "month": mth,
                          "active_cells": real, "frp_mw": real * 10},
                         {"lat0": lat0, "lon0": lon0, "size": "all", "in_mask": False, "group": "MODIS", "year": y, "month": mth,
                          "active_cells": real * ratio, "frp_mw": real * ratio * 10}]
    m = pd.DataFrame(rows)
    cubes = {v: Cube(m, v, True, False, 10, list(range(2012, 2020)), 2.5) for v in ("cells", "frp")}
    from shift.tiles import tile_name
    cells = pd.Index(sorted({tile_name(a, b, 2.5) for a, b in zip(m["lat0"], m["lon0"])}))
    w = pd.Series(1.0, index=cells)
    vt = version_test(cubes, [["MODIS", 2000, 2019]], "S-NPP", list(range(2012, 2020)), "fine",
                      {"cells": 30, "frp": 300}, {"cells": 30, "frp": 300}, cells, w,
                      {"segment": "MODIS", "same_version_years": [2012, 2017],
                       "rule": {"gain_points": 1, "error_gain_points": 0.5, "max_score_drop_points": 1}})
    assert vt["chosen"] == "same version", vt["why"]
    assert abs(vt["same_version"]["cells"]["pooled_error_pct"]) < abs(vt["whole_piece"]["cells"]["pooled_error_pct"])


def test_local_day_shift():
    # Aqua night pass (01:30 local) at 90 E happens at 19:30 UTC the day before: local = UTC + 1
    # Terra night pass (22:30 local) at 60 W happens at 02:30 UTC the next day: local = UTC - 1
    # day passes away from the date line keep their date
    sh = local_day_shift([90.0, -60.0, 10.0, 90.0], ["Aqua", "Terra", "S-NPP", "Aqua"], [0, 0, 5, 3], [4, 2, 0, 1])
    assert sh.tolist() == [1, -1, 0, 0]


def test_scale_month_counts_hybrid_and_no_data_days():
    nd, nz = 31, 72
    status = np.full((nd, nz), "primary", dtype=object)
    target = np.full(nd, "MODIS", dtype=object)
    from shift.factors import zone_key
    z = int(zone_key(25.5, -81.5, 30))
    status[:3, z] = "filled: Aqua"
    status[10, z] = "no data"
    own = np.zeros((648, 12), bool)                                  # no tile has its own factor
    tables = {("own", "MODIS <- Aqua", "Aqua", "cells"): own, ("own", "MODIS <- Aqua", "Aqua", "frp"): own}
    sc = scale_month(status, target, tables, {"MODIS <- Aqua": "hybrid"}, 0, nd, 1, 10.0, 30.0)
    row = sc[sc["tile"] == "N20W090"].iloc[0]
    assert row["days_scaled_hybrid"] == 3 and row["days_no_data"] == 1
    assert np.isclose(row["multiplier"], 31 / 27)
    sc2 = scale_month(status, target, tables, {"MODIS <- Aqua": "substitute"}, 0, nd, 1, 10.0, 30.0)
    assert sc2[sc2["tile"] == "N20W090"].iloc[0]["days_scaled_hybrid"] == 0


def test_checks_helpers():
    v = pd.DataFrame({"region": "x", "year": list(range(2003, 2013)), "month": 8,
                      "val": [10, 11, 9, 10, 12, 10, 11, 10, 9, 200]})
    z = zscores(v, "val", list(range(2003, 2013)))
    assert z.iloc[-1] > 5 and abs(z.iloc[0]) < 2                    # the big year is far above the others' normal
    crit = np.array([0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1], bool)
    assert runs_from(np.arange(11), crit, max_gap=2, min_days=3) == [(1, 4)]
    assert wmedian([1, 2, 100], [1, 1, 10]) == 100


def test_external_reducers():
    lat, lon = gfed_grid()
    assert lat[0] == 89.875 and lon[0] == -179.875 and len(lat) == 720
    f = np.zeros((720, 1440))
    f[720 // 2 - 1, 1440 // 2] = 50.0                               # one cell just north of the equator, east of 0
    fine = to_fine(f, lat, lon, 2.5, "sum")
    assert np.isclose(fine["value"].sum(), 50.0)
    hit = fine[fine["value"] > 0].iloc[0]
    assert hit["lat0"] == 0.0 and hit["lon0"] == 0.0
    tot = region_totals(f, lat, lon, {"global": [-180, -90, 180, 90], "box": [-1, -1, 1, 1], "far": [10, 10, 20, 20]})
    assert tot == {"global": 50.0, "box": 50.0, "far": 0.0}
    assert pick_unit(pd.Series([3.5e8, 3.4e8]), 200, 600) == 1.0
    assert pick_unit(pd.Series([3.5e10, 3.4e10]), 200, 600) == 0.01


def test_provenance_stamp(tmp_path):
    s = load_settings()
    st = stamp(s)
    assert len(st["settings_sha256"]) == 64 and st["run_utc"].endswith("Z")
    s2 = load_settings()
    s2.raw["factors"]["seed"] = 999
    assert settings_hash(s2.raw) != st["settings_sha256"]
