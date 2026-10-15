"""Fix 9b and the export stage, on a world built to have the flaw fix 9b exists for.

One 10-degree tile (N20W090) holds two fire areas that MODIS sees very differently: area A
(MODIS sees most fires) and area B (MODIS sees few). Fires burn at a steady rate every year, so
the true 2012 step is 1.0 in both. A single 10-degree factor is right for the tile but wrong for
each area: one jumps up at 2012, the other down. 2.5-degree factors must fix both, the rule must
adopt them, harmonise must use them, and export must write files that read back to the same
numbers.
"""
import gzip
import json

import numpy as np
import pandas as pd
import yaml

from shift.aggregate import to_celldays, write_celldays
from shift.config import DEFAULT_SETTINGS, load_settings
from shift.export import run_export, unpack_daily
from shift.factors import run_factors
from shift.harmonise import run_harmonise
from shift.nodata import run_nodata
from shift.resolution import run_resolution

AREAS = {"area_a": (25.02, -81.48, 0.60, 0.50), "area_b": (28.02, -86.48, 0.15, 0.10)}   # lat, lon, p_aqua, p_terra


def _settings(tmp_path):
    cfg = yaml.safe_load(DEFAULT_SETTINGS.read_text())
    cfg["dataset"] = "global"
    cfg["regions"]["test"] = {"area_a": {"bbox": [-82.5, 24.0, -80.0, 26.5], "regime": "a"},
                              "area_b": {"bbox": [-87.5, 27.5, -85.0, 29.9], "regime": "b"}}
    cfg["counts"]["source_segments"] = [["MODIS", 2000, 2019], ["Aqua", 2020, 2100]]
    cfg["factors"].update({"min_ref_cells_per_month": 20, "overlap_years": [2012, 2015],
                           "backward_fit_years": [2014, 2015], "backward_test_years": [2012, 2013],
                           "bootstrap_samples": 10})
    cfg["resolution"].update({"min_ref_cells_per_month": 10})
    cfg["harmonise"]["gate"].update({"before_years": [2010, 2011], "after_years": [2013, 2014]})
    cfg["nodata"].update({
        "regional": {**cfg["nodata"]["regional"], "min_partner_cells": 5, "min_expected_cells": 8},
        "links": [{"target": "S-NPP", "substitute": "D19", "years": [2012, 2015]},
                  {"target": "MODIS", "substitute": "Aqua", "years": [2010, 2011]}],
        "record": [{"from": "2009-01-01", "to": "2012-01-19", "target": "MODIS", "fill": ["Aqua"]},
                   {"from": "2012-01-20", "to": "2100-12-31", "target": "S-NPP", "fill": ["D19"]}],
        "replicates": 1})
    p = tmp_path / "settings.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return load_settings(p, root=tmp_path)


def _world(s):
    rng = np.random.default_rng(11)
    days = pd.date_range("2009-01-01", "2015-12-31")
    rows = []
    for day in days:
        for name, (blat, blon, pa, pt) in AREAS.items():
            n = int(min(60, max(3, 22 * rng.lognormal(0, 0.5))))
            for f in range(n):
                lat, lon = blat + 0.05 * (f % 12), blon + 0.05 * (f // 12)
                power = float(rng.lognormal(2.3, 0.8))
                if day >= pd.Timestamp("2012-01-20"):
                    rows.append((day, lat, lon, "S-NPP", "VIIRS", power))
                if rng.random() < pa:
                    rows.append((day, lat, lon, "Aqua", "MODIS", power * 1.5))
                if rng.random() < pt:
                    rows.append((day, lat, lon, "Terra", "MODIS", power * 1.5))
    det = pd.DataFrame(rows, columns=["acq_date", "lat", "lon", "satellite", "sensor_family", "frp_mw"])
    det["daynight"], det["confidence_class"], det["type_known"] = "D", "nominal", True
    for _, d in det.groupby("sensor_family"):
        write_celldays(to_celldays(d, 0.05), s.celldays_dir)
    cov = [{"acq_date": d, "satellite": sat, "status": "ok"} for sat in ("Terra", "Aqua") for d in days]
    cov += [{"acq_date": d, "satellite": "S-NPP", "status": "ok"} for d in days if d >= pd.Timestamp("2012-01-20")]
    cp = s.path(s["coverage"]["path"])
    cp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cov).to_parquet(cp, index=False)


def test_fine_factors_fix_regional_steps_and_export_reads_back(tmp_path):
    s = _settings(tmp_path)
    _world(s)
    run_nodata(s)
    run_factors(s)
    coarse = run_harmonise(s)
    assert coarse["factor_resolution_deg"] == 10.0

    rep = run_resolution(s)
    gaps = rep["region_step_gap_pct"]
    assert abs(gaps["coarse"]["area_a"]) > 15 and abs(gaps["coarse"]["area_b"]) > 15      # the flaw
    assert rep["adopted"] and rep["chosen"] in ("fine", "fine_shrunk"), rep["why"]
    fixed = gaps[rep["chosen"]]
    assert abs(fixed["area_a"]) < 8 and abs(fixed["area_b"]) < 8                          # fixed

    fine = run_harmonise(s)
    assert fine["factor_resolution_deg"] == 2.5
    l3 = pd.concat([pd.read_parquet(p) for p in (s.processed_dir / "l3").glob("year=*/month=*/part.parquet")])
    l3["year"] = l3["date"].str[:4].astype(int)
    for name, (blat, blon, *_x) in AREAS.items():
        a = l3[(l3["lat"].sub(blat).abs() < 1.5) & (l3["lon"].sub(blon).abs() < 1.5)].groupby("year")["cells_h"].sum()
        assert 0.85 < a[2011] / a[2013] < 1.15, (name, a[2011] / a[2013])                  # no jump in either area

    ex = run_export(s)
    assert "note" not in ex and ex["fits_github_pages"] and ex["daily_tiles"] >= 1
    out = s.processed_dir / "site_data"
    meta = json.loads((out / "meta.json").read_text())
    assert meta["days"] > 2000 and "N20W090" in meta["tiles"]
    d = unpack_daily(gzip.open(out / "daily" / "N20W090.bin.gz").read())
    in_tile = l3[(l3["lat"] >= 20) & (l3["lat"] < 30) & (l3["lon"] >= -90) & (l3["lon"] < -80)]
    assert np.isclose(d["cells"].sum(), in_tile["cells_h"].sum(), rtol=1e-4)               # reads back exactly
    assert d["cell"].max() < 100 and d["day"].max() < meta["days"]
    zones = np.frombuffer(gzip.open(out / "zones.bin.gz").read(), "u1").reshape(72, meta["days"])
    assert (zones[zones != 255] <= 5).all()
    assert (out / "monthly" / "2013-06.bin.gz").exists() and (s.processed_dir / "site_data.zip").exists()
