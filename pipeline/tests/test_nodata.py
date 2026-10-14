"""Fix 7: the no-data mask, on a synthetic world with known holes.

Two fire tiles (Florida, African savanna). Fire varies a lot from day to day, so scaling a
month up from the days that have data is a poor guess, while a second VIIRS satellite flying
the same orbit is a good one. Holes put in on purpose:
  * S-NPP silent 1-25 Jun 2020 (whole world), NOAA-20 fine     -> filled from NOAA-20
  * S-NPP AND NOAA-20 silent 10-14 Mar 2021, Aqua fine           -> filled from the D19 piece (if adopted) or no data
  * S-NPP missing over Florida only, 10-12 Sep 2019 (coverage table says "ok") -> found as a regional gap
"""
import numpy as np
import pandas as pd
import yaml

from shift.aggregate import to_celldays, write_celldays
from shift.config import DEFAULT_SETTINGS, load_settings
from shift.nodata import Space, bucket_of, decide, fit_k, run_nodata

SATS = {"S-NPP": ("VIIRS", 1.0), "NOAA-20": ("VIIRS", 0.95), "Aqua": ("MODIS", 0.5), "Terra": ("MODIS", 0.4)}


def _settings(tmp_path):
    cfg = yaml.safe_load(DEFAULT_SETTINGS.read_text())
    cfg["dataset"] = "global"
    cfg["counts"]["source_segments"] = [["MODIS", 2000, 2019], ["Aqua", 2020, 2100]]
    cfg["factors"].update({"min_ref_cells_per_month": 20, "overlap_years": [2018, 2021]})
    cfg["nodata"].update({
        "regional": {**cfg["nodata"]["regional"], "min_partner_cells": 5, "min_expected_cells": 8},
        "links": [{"target": "S-NPP", "substitute": "NOAA-20", "years": [2018, 2021]},
                  {"target": "S-NPP", "substitute": "D19", "years": [2018, 2021]}],
        "record": [{"from": "2018-01-01", "to": "2100-12-31", "target": "S-NPP", "fill": ["NOAA-20", "D19"]}],
        "replicates": 2})
    p = tmp_path / "settings.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return load_settings(p, root=tmp_path)


def _world(s):
    rng = np.random.default_rng(70)
    days = pd.date_range("2018-01-01", "2021-12-31")
    snpp_off = set(pd.date_range("2020-06-01", "2020-06-25")) | set(pd.date_range("2021-03-10", "2021-03-14"))
    n20_off = set(pd.date_range("2021-03-10", "2021-03-14"))
    fl_off = set(pd.date_range("2019-09-10", "2019-09-12"))
    rows = []
    for day in days:
        for tile, (blat, blon) in {"fl": (25.02, -81.48), "af": (8.02, 20.02)}.items():
            level = 18 if tile == "fl" else 30
            n = int(min(80, max(1, level * rng.lognormal(0, 0.9))))       # very uneven from day to day
            for f in range(n):
                lat, lon = blat + 0.05 * (f % 16), blon + 0.05 * (f // 16)
                power = float(rng.lognormal(2.3, 1.0))
                for sat, (fam, p) in SATS.items():
                    if sat == "S-NPP" and (day in snpp_off or (tile == "fl" and day in fl_off)):
                        continue
                    if sat == "NOAA-20" and day in n20_off:
                        continue
                    if rng.random() < p:
                        rows.append((day, lat, lon, sat, fam, power * (1.4 if fam == "MODIS" else 1.0)))
    det = pd.DataFrame(rows, columns=["acq_date", "lat", "lon", "satellite", "sensor_family", "frp_mw"])
    det["daynight"], det["confidence_class"], det["type_known"] = "D", "nominal", True
    for _, d in det.groupby("sensor_family"):
        write_celldays(to_celldays(d, 0.05), s.celldays_dir)
    cov = []
    for sat in SATS:
        for day in days:
            off = (sat == "S-NPP" and day in snpp_off) or (sat == "NOAA-20" and day in n20_off)
            cov.append({"acq_date": day, "satellite": sat, "status": "none" if off else "ok"})
    cp = s.path(s["coverage"]["path"])
    cp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cov).to_parquet(cp, index=False)


def test_nodata_end_to_end(tmp_path):
    s = _settings(tmp_path)
    _world(s)
    rep = run_nodata(s)
    out = s.processed_dir / "nodata"

    # regional gap: S-NPP missing over Florida while the coverage table said "ok"
    reg = pd.read_csv(out / "regional_gaps.csv")
    fl = reg[(reg["satellite"] == "S-NPP") & (reg["zone"] == "N00W090")]
    assert {"2019-09-10", "2019-09-11", "2019-09-12"} <= set(fl["date"])
    assert not (reg[reg["zone"] == "N00E000"]["date"].isin(["2019-09-10", "2019-09-11", "2019-09-12"])).any()

    # the NOAA-20 fill beats scaling on simulated outages and is adopted by the rule
    assert rep["chosen"]["S-NPP <- NOAA-20"] in ("substitute", "hybrid")
    sc = pd.read_csv(out / "fill_test_scores.csv")
    e = sc[sc["link"] == "S-NPP <- NOAA-20"].set_index(["variable", "method"])["abs_error_pct"]
    assert e[("frp", "substitute")] < e[("frp", "scale")] < e[("frp", "none")]

    st = pd.read_parquet(out / "zone_day_status.parquet").set_index(["date", "zone"])["status"]
    assert st[("2020-06-10", "N00W090")] == "filled: NOAA-20"
    assert st[("2019-09-11", "N00W090")] == "filled: NOAA-20"      # regional gap filled too
    assert st[("2019-09-11", "N00E000")] == "primary"
    both_off = st[("2021-03-12", "N00E000")]
    if rep["chosen"]["S-NPP <- D19"] == "scale":
        assert both_off == "no data"
    else:
        assert both_off == "filled: D19"
        assert rep["zone_days_left_empty"] == 0

    mt = pd.read_parquet(out / "zone_month_status.parquet")
    jun = mt[(mt["zone"] == "N00W090") & (mt["year"] == 2020) & (mt["month"] == 6)].iloc[0]
    assert jun["filled"] == 25 and jun["primary"] == 5 and jun["month_status"] == "filled"
    full = mt[(mt["zone"] == "N00E000") & (mt["year"] == 2018) & (mt["month"] == 2)].iloc[0]
    assert full["month_status"] == "complete" and full["expected_error_pct"] == 0

    fac = pd.read_parquet(out / "fill_factors.parquet")
    k = fac[(fac["link"] == "S-NPP <- NOAA-20") & (fac["variable"] == "cells") & (fac["tile"] == "N00E020")]["k"]
    assert 0.95 < k.median() < 1.15                                  # NOAA-20 sees 95% of S-NPP's fires


def test_fit_k_fallback_and_buckets():
    sp = Space(10, 30)
    Tm = np.zeros((sp.nt, 2))
    Sm = np.zeros((sp.nt, 2))
    t_rich, t_thin = 3 * 36 + 10, 3 * 36 + 11                           # same 30-degree zone
    Tm[t_rich], Sm[t_rich] = [200, 220], [100, 110]
    Tm[t_thin], Sm[t_thin] = [3, 0], [1, 0]
    k, level = fit_k(Tm, Sm, np.array([True, True]), np.array([1, 2]), sp, 30)
    assert level[t_rich, 0] == 0 and abs(k[t_rich, 0] - 2.0) < 1e-9
    assert level[t_thin, 0] == 2 and abs(k[t_thin, 0] - 203 / 101) < 1e-9   # borrowed from its zone
    assert level[t_rich, 5] == 1                                          # empty month -> own all-year
    assert [bucket_of(n, [[1, 3], [4, 7]]) for n in (1, 3, 4, 7, 8)] == [0, 0, 1, 1, 2]


def test_rule_needs_a_real_gain():
    def sc(sub_f, sub_c, glob=0.0):
        rows = []
        for v, (scale, sub) in {"frp": (20.0, sub_f), "cells": (15.0, sub_c)}.items():
            rows += [{"variable": v, "method": "none", "abs_error_pct": 40.0, "global_error_pct": -30.0},
                     {"variable": v, "method": "scale", "abs_error_pct": scale, "global_error_pct": 0.5},
                     {"variable": v, "method": "substitute", "abs_error_pct": sub, "global_error_pct": glob},
                     {"variable": v, "method": "hybrid", "abs_error_pct": sub + 0.5, "global_error_pct": glob}]
        return pd.DataFrame(rows)
    rule = {"gain_points": 1, "global_error_limit_pct": 5}
    assert decide(sc(10.0, 8.0), rule)[0] == "substitute"
    assert decide(sc(19.5, 8.0), rule)[0] == "scale"          # gain under 1 point
    assert decide(sc(10.0, 16.0), rule)[0] == "scale"         # worse for burning cells
    assert decide(sc(10.0, 8.0, glob=7.0), rule)[0] == "scale"  # biased overall
