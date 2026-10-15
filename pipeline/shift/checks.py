"""Phase 2 checks on the final record (stage `checks`, after harmonise).

Every rule below was fixed in settings `phase2_checks` on 16 Oct 2026, before the first run, and
committed to git (the commit time is the evidence; nothing was registered with a third party).

  1  ANOMALY RULE again, on the final 2.5-degree factors (must-fix 1). Same rule as fix 9
     (harmonise.anomaly_rule): out-of-sample MODIS x k months, leave-one-year-out over the
     production fit years of the MODIS piece, classed against the S-NPP climatology plainly and
     "era-fair"; fire-weighted over the tested cells, and also for the D18 regions.
  2  UNTOUCHED FINAL TEST on Jan-Sep 2026 (must-fix 4): no factor has seen 2026. Aqua x k (the
     production Aqua piece) and NOAA-20 x k (the S-NPP <- NOAA-20 link, which carries the record
     after S-NPP ends) predict S-NPP. PASS if the global error is within +-error_limit_pct for
     both variables. Run ONCE: the result is stored with its provenance and later runs show the
     stored result unless allow_rerun is set (and then say so).
  3  COVERAGE of the 95% ranges (must-fix 5). The factor's range is about the factor, not about a
     single month. Measured: how often the real S-NPP month falls inside the month's band on the
     backward test. Recalibrated: a month band per factor source and fire level, set on the
     leave-one-year-out years and VERIFIED on the backward years; accepted if verified coverage
     lies in `accept`. For areas, the cells' bands are combined as independent unless that
     under-covers, then as fully correlated.
  4  SIMPLE METHODS on the same tests (option A): no harmonisation, one global ratio, one ratio
     per 10-degree region for all months (as IGNIS-style products do), the v1 Florida equation,
     and SHIFT.
  5  KNOWN EVENTS (must-fix 7), list fixed in settings before the run: each event's months classed
     against its own box's climatology from the other years; hit = at least one event month
     unusual or critical in fire power. Compared with the raw stitched record and MODIS alone,
     and with the base rate of flags in all other box-months.
  6  DATED CRITICAL PERIODS (option D): 7-day fire power against the same days of the other years;
     runs of critical days become named, dated periods; known events must appear among them.
  7  BURNED AREA (option B): the ratio of the record to GFED4 burned area (no small fires, one
     sensor family) must not step at 2012 by more than MODIS alone's ratio steps at other years.
  8  CLOUD (option C): does cloud (MERRA-2 CLDTOT) explain scaling errors or low months? The cloud
     flag is shown only if low months are cloudy at least min_ratio times as often as all months.

    python -m shift.run --dataset global --stage checks      (after harmonise; ~10-20 min)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import Settings
from .counts import source_segments
from .daily import wmedian
from .factors import Cube, bootstrap
from .harmonise import Table, check_regions
from .provenance import stamp
from .region import in_bbox
from .resolution import fit_weights, k_model, key_centres, model_spec
from .selection import baseline, classify, score

HIGH = {"unusual", "critical"}
V1 = (0.316162, 24.701681)          # v1 Florida line: VIIRS FRP = a x MODIS FRP + b per cell-day


# ---------------------------------------------------------------- shared helpers

def preds(cube: Cube, g: str, ref: str, test_years, k: np.ndarray | None = None, fn=None) -> pd.DataFrame:
    """Predicted and real reference per cell, year and month: pred = A[g] * k, or fn(j, month)."""
    out = []
    for y in test_years:
        if y not in cube.years or g not in cube.A:
            continue
        j = cube.years.index(y)
        ok = cube.avail[g][:, j] & cube.avail[ref][:, j]
        for mth in np.where(ok)[0]:
            p = fn(j, mth) if fn is not None else cube.A[g][:, mth, j] * k[:, mth]
            out.append(pd.DataFrame({"key": np.arange(len(cube.keys)), "year": y, "month": mth + 1,
                                     "pred": p, "real": cube.A[ref][:, mth, j]}))
    if not out:
        return pd.DataFrame(columns=["key", "region", "year", "month", "pred", "real"])
    p = pd.concat(out, ignore_index=True)
    p["region"] = cube.keys["tile"].to_numpy()[p["key"].to_numpy()]
    return p


def judge_preds(lo: pd.DataFrame, bw: pd.DataFrame, cells, weights: pd.Series, ref: str) -> dict:
    lo, bw = lo[lo["region"].isin(cells)], bw[bw["region"].isin(cells)]
    yrs = sorted(lo["year"].unique())
    base = baseline(lo[["region", "year", "month", "real"]].rename(columns={"real": ref}), ref, yrs, [])
    a, b = score(lo, base, []).set_index("region"), score(bw, base, []).set_index("region")
    t = pd.DataFrame({"loyo": a["agreement_pct"], "backward": b["agreement_pct"], "err": b["total_error_pct"]})
    t["score"] = t[["loyo", "backward"]].mean(axis=1)
    w = weights.reindex(t.index).fillna(0)
    ok = t["score"].notna() & (w > 0)
    med = wmedian(t["err"].abs(), w)
    return {"score": round(float(np.average(t["score"][ok], weights=w[ok])), 2) if ok.any() else None,
            "fw_median_abs_backward_error_pct": None if med is None else round(med, 1),
            "cells_within_10pct": round(float(100 * (t["err"].abs() <= 10).mean()), 1),
            "global_backward_error_pct": round(float(100 * (bw["pred"].sum() / bw["real"].sum() - 1)), 2) if bw["real"].sum() else None}


def area_of(cube: Cube, deg: float, regions: dict) -> dict:
    lat, lon = key_centres(cube, deg)
    return {n: in_bbox(bb, lon, lat) for n, bb in regions.items()}


def zscores(v: pd.DataFrame, value: str, years: list[int]) -> pd.Series:
    """z of log(1 + value) per (region, year, month) against the same region and calendar month in
    the other years of `years` (the year itself is always left out of its own normal)."""
    x = v.assign(lv=np.log1p(v[value].clip(lower=0)))
    inb = x["year"].isin(years)
    s = x[inb].groupby(["region", "month"])["lv"].agg(["sum", "count"])
    s2 = x[inb].assign(q=x["lv"] ** 2).groupby(["region", "month"])["q"].sum()
    j = x.join(s, on=["region", "month"]).join(s2.rename("sq"), on=["region", "month"])
    own = inb.astype(float)
    n = j["count"] - own
    mu = (j["sum"] - own * j["lv"]) / n
    var = (j["sq"] - own * j["lv"] ** 2) / n - mu ** 2
    sd = np.sqrt(np.maximum(var * n / np.maximum(n - 1, 1), 0)).clip(lower=0.05)
    return ((j["lv"] - mu) / sd).where(n >= 3)


# ---------------------------------------------------------------- 1. anomaly rule on the final factors

def anomaly_check(cubes, g, ref, years_fit, all_overlap, spec, prior, min_v, cells, weights, settings, regions_mask) -> tuple[dict, pd.DataFrame]:
    rule = (settings.raw.get("harmonise") or {}).get("anomaly_rule") or {}
    c = cubes["frp"]
    p = pd.concat([preds(c, g, ref, [y], k_model(c, g, ref, np.array([1.0 if (x in years_fit and x != y) else 0.0 for x in c.years]),
                                                    spec, prior["frp"], min_v["frp"])) for y in years_fit], ignore_index=True)
    truth = preds(c, ref, ref, all_overlap, np.ones((len(c.keys), 12)))
    base = baseline(truth[["region", "year", "month", "real"]].rename(columns={"real": ref}), ref, all_overlap, [])
    sigma = None
    bt = settings.processed_dir / "places" / "borrowed_factor_test.csv"
    if bt.exists():
        b = pd.read_csv(bt).set_index("level")
        if "own month" in b.index:
            sigma = float(np.log1p(b.loc["own month", "median_month_abs_error_pct"] / 100) / 0.6745)
    sigma = sigma if sigma is not None else float(rule.get("sigma_fallback_log", 0.18))
    cp = p[p["region"].isin(cells)].merge(base, on=["region", "month"], how="left")
    w = cp["region"].map(weights).fillna(0).to_numpy()
    real_hi = classify(cp["real"], cp["mu"], cp["sd"]).isin(HIGH).to_numpy()
    lr = np.log((cp["pred"] + 10) / (cp["real"] + 10)).abs()
    sigma_fine = float(wmedian(lr, w) / 0.6745) if len(cp) else None

    def rates(cls, rh, ww):
        hi = cls.isin(HIGH).to_numpy()
        fa = float(100 * ww[hi & ~rh].sum() / ww[hi].sum()) if ww[hi].sum() else 0.0
        ca = float(100 * ww[hi & rh].sum() / ww[rh].sum()) if ww[rh].sum() else 0.0
        return round(fa, 1), round(ca, 1), int(hi.sum())
    plain = rates(classify(cp["pred"], cp["mu"], cp["sd"]), real_hi, w)
    fair = rates(classify(cp["pred"], cp["mu"], np.sqrt(cp["sd"] ** 2 + sigma ** 2)), real_hi, w)
    fair_fine = rates(classify(cp["pred"], cp["mu"], np.sqrt(cp["sd"] ** 2 + sigma_fine ** 2)), real_hi, w) if sigma_fine else None
    fa_drop, ca_max = float(rule.get("fa_drop_points", 5)), float(rule.get("caught_max_drop_points", 5))
    adopt = (plain[0] - fair[0] >= fa_drop) and (plain[1] - fair[1] <= ca_max)
    # the same for the D18 regions (sums of cells; unweighted, each region-month counts once)
    reg = []
    for name, m in regions_mask.items():
        if name == "global" or not m.any():
            continue
        keys = set(np.nonzero(m)[0])
        rp = p[p["key"].isin(keys)].groupby(["year", "month"], as_index=False)[["pred", "real"]].sum().assign(region=name)
        rt = truth[truth["key"].isin(keys)].groupby(["year", "month"], as_index=False)["real"].sum().assign(region=name)
        reg.append((rp, rt))
    reg_out = None
    if reg:
        rp = pd.concat([a for a, _ in reg], ignore_index=True)
        rt = pd.concat([b for _, b in reg], ignore_index=True)
        rb = baseline(rt.rename(columns={"real": ref}), ref, all_overlap, [])
        rp = rp.merge(rb, on=["region", "month"], how="left")
        rh = classify(rp["real"], rp["mu"], rp["sd"]).isin(HIGH).to_numpy()
        one = np.ones(len(rp))
        rpl = rates(classify(rp["pred"], rp["mu"], rp["sd"]), rh, one)
        rfa = rates(classify(rp["pred"], rp["mu"], np.sqrt(rp["sd"] ** 2 + sigma ** 2)), rh, one)
        reg_out = {"plain": {"false_alarm_pct": rpl[0], "caught_pct": rpl[1], "flagged_months": rpl[2]},
                   "era_fair": {"false_alarm_pct": rfa[0], "caught_pct": rfa[1], "flagged_months": rfa[2]},
                   "real_unusual_months": int(rh.sum())}
    rep = {"years": [years_fit[0], years_fit[-1]], "cells": int(cp["region"].nunique()), "sigma_log": round(sigma, 3),
           "plain": {"false_alarm_pct": plain[0], "caught_pct": plain[1], "flagged_months": plain[2]},
           "era_fair": {"false_alarm_pct": fair[0], "caught_pct": fair[1], "flagged_months": fair[2]},
           "real_unusual_months": int(real_hi.sum()), "adopted": "era-fair" if adopt else "plain",
           "why": (f"false alarms {plain[0]}% -> {fair[0]}% (needed a drop of >= {fa_drop:g}), "
                   f"caught {plain[1]}% -> {fair[1]}% (allowed drop <= {ca_max:g})"),
           "sensitivity_fine_sigma": None if fair_fine is None else {"sigma_log": round(sigma_fine, 3), "false_alarm_pct": fair_fine[0],
                                                                     "caught_pct": fair_fine[1]},
           "regions": reg_out}
    return rep, p


# ---------------------------------------------------------------- 2. the untouched 2026 test

def final_test(settings, cubes, ref, cells, weights, regions_mask, cfg, fine) -> dict:
    out = settings.processed_dir / "checks"
    lock = out / "final_test_2026.json"
    if lock.exists() and not cfg.get("allow_rerun", False):
        stored = json.loads(lock.read_text())
        return {**stored, "note": "stored result of the one run (allow_rerun is off)"}
    a, b = cfg.get("months", ["2026-01", "2026-09"])
    y = int(a[:4])
    months = list(range(int(a[5:7]), int(b[5:7]) + 1))
    limit = float(cfg.get("error_limit_pct", 10))
    c0 = cubes["cells"]
    if y not in c0.years:
        return {"note": f"no {y} data"}
    j = c0.years.index(y)
    lat, lon = key_centres(c0, fine)
    preds_by = {}
    ff = settings.processed_dir / "counts" / "factors_world_fine.parquet"
    fl = settings.processed_dir / "nodata" / "fill_factors.parquet"
    if ff.exists():
        t = pd.read_parquet(ff)
        t = t[t["segment"].astype(str).str.startswith("Aqua")]
        if len(t):
            preds_by["Aqua x k"] = ("Aqua", {v: Table(t[t["variable"] == v], fine) for v in ("cells", "frp")})
    if fl.exists():
        t = pd.read_parquet(fl)
        t = t[(t["link"] == "S-NPP <- NOAA-20")]
        if len(t):
            deg = float((settings.raw.get("nodata") or {}).get("tile_deg", 10))
            preds_by["NOAA-20 x k"] = ("NOAA-20", {v: Table(t[t["variable"] == v], deg) for v in ("cells", "frp")})
    res = {}
    for name, (grp, tabs) in preds_by.items():
        r = {}
        for var in ("cells", "frp"):
            c = cubes[var]
            if grp not in c.A:
                continue
            rows = []
            for mth in months:
                if not (c.avail[grp][mth - 1, j] and c.avail[ref][mth - 1, j]):
                    continue
                k = tabs[var](lat, lon, np.full(len(lat), mth))
                rows.append(pd.DataFrame({"key": np.arange(len(lat)), "region": c.keys["tile"].to_numpy(), "month": mth,
                                          "pred": c.A[grp][:, mth - 1, j] * k, "real": c.A[ref][:, mth - 1, j]}))
            if not rows:
                continue
            p = pd.concat(rows, ignore_index=True)
            glob = round(float(100 * (p["pred"].sum() / p["real"].sum() - 1)), 2) if p["real"].sum() else None
            reg = {}
            for rn, m in regions_mask.items():
                if rn == "global":
                    continue
                q = p[p["key"].isin(np.nonzero(m)[0])]
                if q["real"].sum() > 0:
                    reg[rn] = round(float(100 * (q["pred"].sum() / q["real"].sum() - 1)), 1)
            cell = p[p["region"].isin(cells)].groupby("region")[["pred", "real"]].sum()
            cell = cell[cell["real"] > 0]
            ce = (100 * (cell["pred"] / cell["real"] - 1)).abs()
            med = wmedian(ce, weights.reindex(ce.index).fillna(0))
            r[var] = {"global_error_pct": glob, "regions_error_pct": reg,
                      "fw_median_abs_cell_error_pct": None if med is None else round(med, 1)}
        r["pass"] = bool(all(r.get(v, {}).get("global_error_pct") is not None and abs(r[v]["global_error_pct"]) <= limit
                             for v in ("cells", "frp")))
        res[name] = r
    result = {"months": [a, b], "error_limit_pct": limit, "predictors": res,
              "pass": bool(res) and all(v["pass"] for v in res.values()), "provenance": stamp(settings)}
    out.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps(result, indent=2, default=str))
    if cfg.get("allow_rerun", False):
        result["note"] = "allow_rerun was set: this is NOT the first run of the 2026 test"
    return result


# ---------------------------------------------------------------- 3. coverage of the ranges

def coverage_check(settings, cubes, g, y0, y1, ref, years, spec, prior, min_v, cells, weights, regions_mask, cfg, fine) -> dict:
    fcfg = settings.raw.get("factors") or {}
    allowed = spec["fit_years"].get(g)
    piece = [y for y in years if int(y0) <= y <= int(y1)]
    prod = [y for y in piece if allowed is None or y in allowed]
    cal_years = [y for y in piece if int(cfg.get("calibrate_years", [2016, 2019])[0]) <= y <= int(cfg.get("calibrate_years", [2016, 2019])[1])]
    ver_years = [y for y in piece if int(cfg.get("verify_years", [2012, 2015])[0]) <= y <= int(cfg.get("verify_years", [2012, 2015])[1])]
    ver_fit = [y for y in prod if y not in ver_years]
    if not cal_years or not ver_years or not ver_fit:
        return {"note": "not enough years for calibration and verification"}
    bins = [0.0] + [float(x) for x in cfg.get("fire_bins_cells", [30, 300])] + [np.inf]
    offset = {"cells": 1.0, "frp": 10.0}
    borrow = spec["variant"] in ("fine_shrunk", "fine_shrunk_seasonal")
    n_boot = int(cfg.get("bootstrap_samples", fcfg.get("bootstrap_samples", 200)))
    lo_acc, hi_acc = (float(x) for x in cfg.get("accept", [90, 99]))
    out_rows, calib, area = [], [], {}
    for var in ("cells", "frp"):
        c = cubes[var]
        wprod = fit_weights(c, g, y0, y1, years, {g: prod})
        c.fit(g, ref, wprod, False, borrow, prior[var], min_v[var])
        level = c.last_level.copy()
        kc = k_model(cubes["cells"], g, ref, fit_weights(cubes["cells"], g, y0, y1, years, {g: prod}), spec, prior["cells"], min_v["cells"])
        # calibration: each calibration year predicted from the other production years
        cal = pd.concat([preds(c, g, ref, [y], k_model(c, g, ref, fit_weights(c, g, y0, y1, years, {g: prod}, drop=y), spec, prior[var], min_v[var]))
                         for y in cal_years], ignore_index=True)
        # verification: backward, fitted without the verify years, with the bootstrap range of that fit
        wv = fit_weights(c, g, y0, y1, years, {g: ver_fit})
        ver = preds(c, g, ref, ver_years, k_model(c, g, ref, wv, spec, prior[var], min_v[var]))
        bs = bootstrap(c, [[g, y0, y1]], ref, years, False, borrow, prior[var], min_v[var], n_boot,
                       int(fcfg.get("seed", 26)), {g: ver_fit})
        _, blo, bhi, _, _ = next(iter(bs.values()))
        with np.errstate(all="ignore"):
            hwf = np.log(np.where((blo > 0) & (bhi > 0), bhi / blo, np.nan)) / 2
        for name, p in (("cal", cal), ("ver", ver)):
            p["level"] = level[p["key"], p["month"] - 1]
            p["pred_cells"] = cubes["cells"].A[g][p["key"], p["month"] - 1, [c.years.index(y) for y in p["year"]]] * kc[p["key"], p["month"] - 1]
            p["bin"] = np.digitize(p["pred_cells"], bins[1:-1])
            p["lerr"] = np.abs(np.log((p["pred"] + offset[var]) / (p["real"] + offset[var])))
            p["w"] = p["region"].map(weights).fillna(0)
        cal = cal[cal["region"].isin(cells) & ((cal["pred"] > 0) | (cal["real"] > 0))]
        ver = ver[ver["region"].isin(cells) & ((ver["pred"] > 0) | (ver["real"] > 0))]
        q = {}
        for (lv, bn), gq in cal.groupby(["level", "bin"]):
            o = np.argsort(gq["lerr"].to_numpy())
            cw = np.cumsum(gq["w"].to_numpy()[o])
            q[(lv, bn)] = float(gq["lerr"].to_numpy()[o][np.searchsorted(cw, 0.95 * cw[-1])]) if cw[-1] > 0 else float(np.quantile(gq["lerr"], 0.95))
            calib.append({"variable": var, "level": int(lv), "fire_bin": int(bn), "month_hw95_log": round(q[(lv, bn)], 4),
                          "n": int(len(gq))})
        fallback = float(np.quantile(cal["lerr"], 0.95)) if len(cal) else 1.0
        hw_fac = hwf[ver["key"], ver["month"] - 1]
        hw_fac = np.nan_to_num(hw_fac, nan=0.0)
        hw_cal = np.fmax(hw_fac, np.array([q.get((a, b), fallback) for a, b in zip(ver["level"], ver["bin"])]))
        inside = lambda hw: (ver["lerr"].to_numpy() <= hw)  # noqa: E731
        ww = ver["w"].to_numpy()
        cov = lambda hw: (round(float(100 * inside(hw).mean()), 1), round(float(100 * (ww * inside(hw)).sum() / ww.sum()), 1) if ww.sum() else None)  # noqa: E731
        f_un, f_w = cov(hw_fac)
        c_un, c_w = cov(hw_cal)
        out_rows.append({"variable": var, "cell_months": int(len(ver)), "factor_band_coverage_pct": f_un,
                         "factor_band_coverage_fw_pct": f_w, "calibrated_coverage_pct": c_un, "calibrated_coverage_fw_pct": c_w,
                         "accepted": bool(lo_acc <= c_un <= hi_acc)})
        # areas: D18 regions, cells' bands combined independently or fully correlated
        vv = ver.assign(dev=ver["pred"] * (np.exp(hw_cal) - 1))
        a_rows = []
        for rn, m in regions_mask.items():
            if rn == "global":
                continue
            sub = vv[vv["key"].isin(np.nonzero(m)[0])].groupby(["year", "month"]).agg(
                pred=("pred", "sum"), real=("real", "sum"), ind=("dev", lambda d: float(np.sqrt((d ** 2).sum()))), cor=("dev", "sum"))
            a_rows.append(sub[sub["real"] > 0])
        if a_rows:
            ar = pd.concat(a_rows)
            err = (ar["pred"] - ar["real"]).abs()
            ind, cor = float(100 * (err <= ar["ind"]).mean()), float(100 * (err <= ar["cor"]).mean())
            area[var] = {"region_months": int(len(ar)), "independent_coverage_pct": round(ind, 1),
                         "correlated_coverage_pct": round(cor, 1), "rule": "independent" if ind >= lo_acc else "correlated"}
    cal_tbl = pd.DataFrame(calib)
    out = settings.processed_dir / "checks"
    out.mkdir(parents=True, exist_ok=True)
    cal_tbl.to_csv(out / "month_range_calibration.csv", index=False)
    accepted = bool(out_rows) and all(r["accepted"] for r in out_rows)
    # month band per factor in the published uncertainty table (cell's typical MODIS-era month)
    up = settings.processed_dir / "l3" / "l3_uncertainty.parquet"
    if accepted and up.exists() and len(cal_tbl):
        u = pd.read_parquet(up).drop(columns=["month_hw95_log", "month_range_pct"], errors="ignore")
        typical = {}
        cc = cubes["cells"]
        era = [cc.years.index(y) for y in cc.years if y <= 2011 and y >= 2003]
        kc_all = k_model(cc, g, ref, fit_weights(cc, g, y0, y1, years, {g: prod}), spec, prior["cells"], min_v["cells"])
        if era:
            mean_cells = cc.A[g][:, :, era].mean(2) * kc_all
            typical = {(t, m + 1): mean_cells[i, m] for i, t in enumerate(cc.keys["tile"]) for m in range(12)}
        lvl_map = {"own month": 0, "own all-year": 1, "zone": 2, "world": 3}
        u["fire_bin"] = np.digitize([typical.get((t, m), 0.0) for t, m in zip(u["tile"], u["month"])], bins[1:-1])
        u["level"] = u["source"].map(lvl_map).fillna(2).astype(int)
        u = u.merge(cal_tbl[["variable", "level", "fire_bin", "month_hw95_log"]], on=["variable", "level", "fire_bin"], how="left")
        u["month_hw95_log"] = np.fmax(u["month_hw95_log"].fillna(u["month_hw95_log"].max()), u["hw95_log"])
        u["month_range_pct"] = (100 * (np.exp(u["month_hw95_log"]) - 1)).round(1)
        u.drop(columns=["fire_bin", "level"]).to_parquet(up, index=False)
    return {"cells": out_rows, "areas": area, "accepted": accepted, "calibrate_years": [cal_years[0], cal_years[-1]],
            "verify_years": [ver_years[0], ver_years[-1]], "verify_fit_years": [ver_fit[0], ver_fit[-1]],
            "written_to_l3_uncertainty": bool(accepted and up.exists())}


# ---------------------------------------------------------------- 4. simple methods

def simple_methods(cubes, g, y0, y1, ref, years, bw_fit, bw_test, spec, prior, min_v, cells, weights) -> pd.DataFrame:
    piece = [y for y in years if int(y0) <= y <= int(y1)]
    cc, cf = cubes["cells"], cubes["frp"]
    same = len(cc.keys) == len(cf.keys) and (cc.keys["tile"].to_numpy() == cf.keys["tile"].to_numpy()).all()

    def k_for(method, c, var, wts):
        S = np.einsum("kmy,my->km", c.A[g], wts[None, :] * (c.avail[g] & c.avail[ref]))
        R = np.einsum("kmy,my->km", c.A[ref], wts[None, :] * (c.avail[g] & c.avail[ref]))
        if method == "no harmonisation":
            return np.ones_like(S)
        if method == "one global ratio":
            return np.full(S.shape, R.sum() / max(S.sum(), 1e-9))
        if method == "one ratio per 10-degree region, all months":
            Sz, Rz = (c.Z @ S).sum(1), (c.Z @ R).sum(1)
            kz = np.where(Sz > 0, Rz / np.maximum(Sz, 1e-9), R.sum() / max(S.sum(), 1e-9))
            return np.broadcast_to((c.Z.T @ kz)[:, None], S.shape).copy()
        return k_model(c, g, ref, wts, spec, prior[var], min_v[var])

    rows = []
    for method in ("no harmonisation", "one global ratio", "one ratio per 10-degree region, all months", "v1 Florida line", "SHIFT"):
        row = {"method": method}
        for var, c in (("cells", cc), ("frp", cf)):
            def run(fit, test):
                wts = np.array([1.0 if y in fit else 0.0 for y in c.years])
                if method == "v1 Florida line":
                    if var == "cells" or not same:
                        return preds(c, g, ref, test, np.ones((len(c.keys), 12)))
                    return preds(c, g, ref, test, fn=lambda j, mth: V1[0] * cf.A[g][:, mth, j] + V1[1] * cc.A[g][:, mth, j])
                return preds(c, g, ref, test, k_for(method, c, var, wts))
            lo = pd.concat([run([x for x in piece if x != y], [y]) for y in piece], ignore_index=True)
            bw = run([y for y in bw_fit if y in piece], [y for y in bw_test if y in piece])
            r = judge_preds(lo, bw, cells, weights, ref)
            row.update({f"{k_}_{var}": v for k_, v in r.items()})
        row["score"] = round(float(np.mean([row["score_cells"], row["score_frp"]])), 2) if row.get("score_cells") is not None else None
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 5. known events and 6. dated critical periods

def series_frame(mr: pd.DataFrame) -> pd.DataFrame:
    mr = mr.copy()
    mr["stitched_frp"] = np.where(mr["year"] < 2012, mr["MODIS_frp"], mr["S-NPP_frp"])
    mr["stitched_cells"] = np.where(mr["year"] < 2012, mr["MODIS_cells"], mr["S-NPP_cells"])
    return mr


def event_check(settings, cfg) -> dict:
    p = settings.processed_dir / "l3" / "monthly_regions.csv"
    events = cfg.get("events") or {}
    if not p.exists() or not events:
        return {"note": "no monthly_regions.csv (rerun harmonise) or no events"}
    base_years = list(range(int(cfg.get("baseline_years", [2003, 2025])[0]), int(cfg.get("baseline_years", [2003, 2025])[1]) + 1))
    mr = series_frame(pd.read_csv(p))
    mr = mr[mr["region"].str.startswith("event:") & mr["year"].isin(base_years)]
    series = {"SHIFT": ("frp_h_est", "cells_h_est"), "raw stitched": ("stitched_frp", "stitched_cells"), "MODIS alone": ("MODIS_frp", "MODIS_cells")}
    z = {}
    for s, (f, c) in series.items():
        z[(s, "frp")] = zscores(mr, f, base_years)
        z[(s, "cells")] = zscores(mr, c, base_years)
    rows, ev_months = [], set()
    for name, e in events.items():
        reg = f"event:{name}"
        ms = [(int(x[:4]), int(x[5:7])) for x in e["months"]]
        ev_months |= {(reg, y, m) for y, m in ms}
        sel = (mr["region"] == reg) & pd.Series(list(zip(mr["year"], mr["month"])), index=mr.index).isin(ms)
        row = {"event": name, "months": ", ".join(e["months"])}
        for (s, var), zz in z.items():
            v = zz[sel]
            row[f"{s} {var} max z"] = round(float(v.max()), 2) if v.notna().any() else None
            row[f"{s} {var} hit"] = bool((v >= 1).any())
        rows.append(row)
    t = pd.DataFrame(rows)
    # base rate: every box once (two events may share a box), without any event month of that box
    box = {f"event:{n}": tuple(e["bbox"]) for n, e in events.items()}
    first = {}
    for reg, b in box.items():
        first.setdefault(b, reg)
    ev_box = {(box[r], y, m) for r, y, m in ev_months}
    other = pd.Series([first.get(box.get(r)) == r and (box.get(r), y, m) not in ev_box
                       for r, y, m in zip(mr["region"], mr["year"], mr["month"])], index=mr.index)
    summary = {}
    for s in series:
        for var in ("frp", "cells"):
            zz = z[(s, var)]
            summary[f"{s} {var}"] = {"hit_rate_pct": round(float(100 * t[f"{s} {var} hit"].mean()), 1),
                                     "base_rate_pct": round(float(100 * (zz[other] >= 1).mean()), 1)}
    return {"table": t.to_dict("records"), "summary": summary}


def runs_from(dates: np.ndarray, crit: np.ndarray, max_gap: int, min_days: int) -> list[tuple[int, int]]:
    idx = np.nonzero(crit)[0]
    out, s, last = [], None, None
    for i in idx:
        if s is None:
            s, last = i, i
        elif i - last <= max_gap + 1:
            last = i
        else:
            out.append((s, last))
            s, last = i, i
    if s is not None:
        out.append((s, last))
    return [(a, b) for a, b in out if crit[a:b + 1].sum() >= min_days]


def critical_periods(settings, cfg) -> dict:
    p = settings.processed_dir / "l3" / "daily_regions.parquet"
    if not p.exists():
        return {"note": "no daily_regions.parquet (rerun harmonise)"}
    c = cfg.get("critical") or {}
    win, zc, half = int(c.get("window_days", 7)), float(c.get("z", 2)), int(c.get("doy_halfwidth", 15))
    min_days, max_gap = int(c.get("min_days", 3)), int(c.get("max_gap_days", 2))
    by = [int(x) for x in cfg.get("baseline_years", [2003, 2025])]
    d = pd.read_parquet(p)
    d["date"] = pd.to_datetime(d["date"])
    runs = []
    for reg, g in d.groupby("region"):
        g = g.sort_values("date").set_index("date")
        v = g["frp_h"].where(g["no_data_share"] <= 0.5)
        roll = v.rolling(win, min_periods=win - 1).sum() * win / v.rolling(win, min_periods=win - 1).count()
        lv = np.log1p(roll)
        doy = np.minimum(lv.index.dayofyear.to_numpy(), 365) - 1
        yr = lv.index.year.to_numpy()
        years = np.arange(by[0], by[1] + 1)
        M = np.full((len(years), 365), np.nan)
        inb = (yr >= by[0]) & (yr <= by[1])
        M[yr[inb] - by[0], doy[inb]] = lv.to_numpy()[inb]
        # window sums over +-half days (circular), per year
        ok = np.isfinite(M)
        X = np.where(ok, M, 0.0)
        S1 = sum(np.roll(X, s, axis=1) for s in range(-half, half + 1))
        S2 = sum(np.roll(X ** 2, s, axis=1) for s in range(-half, half + 1))
        N = sum(np.roll(ok.astype(float), s, axis=1) for s in range(-half, half + 1))
        T1, T2, TN = S1.sum(0), S2.sum(0), N.sum(0)
        z = np.full(len(lv), np.nan)
        for i, (y_, dd, val) in enumerate(zip(yr, doy, lv.to_numpy())):
            if not np.isfinite(val):
                continue
            yi = y_ - by[0]
            own = 0 <= yi < len(years)
            n = TN[dd] - (N[yi, dd] if own else 0)
            if n < 3 * (2 * half + 1):
                continue
            mu = (T1[dd] - (S1[yi, dd] if own else 0)) / n
            var = (T2[dd] - (S2[yi, dd] if own else 0)) / n - mu ** 2
            z[i] = (val - mu) / max(np.sqrt(max(var, 0)), 0.05)
        crit = np.nan_to_num(z) >= zc
        dates = lv.index.to_numpy()
        for a, b in runs_from(dates, crit, max_gap, min_days):
            seg = g.iloc[a:b + 1]
            runs.append({"region": reg, "start": str(lv.index[a].date()), "end": str(lv.index[b].date()),
                         "days": int(b - a + 1), "critical_days": int(crit[a:b + 1].sum()),
                         "peak_date": str(lv.index[a + int(np.nanargmax(z[a:b + 1]))].date()),
                         "peak_z": round(float(np.nanmax(z[a:b + 1])), 2), "fire_power_mw_days": round(float(seg["frp_h"].sum()), 0),
                         "before_2012_20_jan": bool(lv.index[a] < pd.Timestamp("2012-01-20")),
                         "max_no_data_share": round(float(seg["no_data_share"].max()), 2)})
    rt = pd.DataFrame(runs)
    if len(rt):
        rt["name"] = rt["region"].str.replace("event:", "", regex=False) + " " + rt["start"] + " to " + rt["end"]
    out = settings.processed_dir / "checks"
    out.mkdir(parents=True, exist_ok=True)
    rt.to_csv(out / "critical_periods.csv", index=False)
    found = {}
    for name, e in (cfg.get("events") or {}).items():
        reg = f"event:{name}"
        ms = sorted(e["months"])
        lo = pd.Timestamp(ms[0] + "-01")
        hi = pd.Timestamp(ms[-1] + "-01") + pd.offsets.MonthEnd(0)
        hit = rt[(rt["region"] == reg) & (pd.to_datetime(rt["start"]) <= hi) & (pd.to_datetime(rt["end"]) >= lo)] if len(rt) else rt
        found[name] = hit["name"].tolist()[:3] if len(hit) else []
    per_region = rt.groupby("region").size().to_dict() if len(rt) else {}
    return {"runs": int(len(rt)), "runs_per_area": per_region,
            "events_found_as_dated_runs": found,
            "event_detection_rate_pct": round(float(100 * np.mean([bool(v) for v in found.values()])), 1) if found else None}


# ---------------------------------------------------------------- 7. burned area

def gfed_check(settings, cfg) -> dict:
    gp = settings.processed_dir / "external" / "gfed_monthly_regions.csv"
    mp = settings.processed_dir / "l3" / "monthly_regions.csv"
    if not (gp.exists() and mp.exists()):
        return {"note": "no GFED table (run stage external) or no monthly_regions.csv"}
    gc = cfg.get("gfed") or {}
    before, after = [int(x) for x in gc.get("before_years", [2009, 2011])], [int(x) for x in gc.get("after_years", [2013, 2015])]
    b_yrs, a_yrs = list(range(before[0], before[1] + 1)), list(range(after[0], after[1] + 1))
    ba = pd.read_csv(gp).groupby(["region", "year"])["ba_ha"].sum()
    mr = series_frame(pd.read_csv(mp)).groupby(["region", "year"])[["frp_h_est", "cells_h_est", "MODIS_frp", "MODIS_cells",
                                                                    "stitched_frp", "stitched_cells"]].sum()
    j = mr.join(ba, how="inner")
    j = j[j["ba_ha"] > 0]
    y_lo, y_hi = int(j.index.get_level_values("year").min()), int(j.index.get_level_values("year").max())
    width = len(b_yrs)
    rows = []
    for reg, g in j.groupby(level="region"):
        g = g.droplevel("region")
        if not set(b_yrs + a_yrs) <= set(g.index):
            continue

        def step(col, b, a):
            r = g[col] / g["ba_ha"]
            return float(r.reindex(a).mean() / r.reindex(b).mean())
        row = {"region": reg}
        for var in ("frp", "cells"):
            h = step(f"{var}_h_est", b_yrs, a_yrs)
            plac = []
            for s in range(y_lo + width, y_hi - width + 1):
                if s == 2012:
                    continue
                bb, aa = list(range(s - width, s)), list(range(s + 1, s + width + 1))
                if set(bb + aa) <= set(g.index):
                    plac.append(abs(np.log(step(f"MODIS_{var}", bb, aa))))
            row[f"{var}_step_shift"] = round(h, 3)
            row[f"{var}_step_raw_stitched"] = round(step(f"stitched_{var}", b_yrs, a_yrs), 3)
            row[f"{var}_step_modis_alone"] = round(step(f"MODIS_{var}", b_yrs, a_yrs), 3)
            row[f"{var}_largest_other_year_step_modis"] = round(float(np.exp(max(plac))), 3) if plac else None
            row[f"{var}_pass"] = bool(plac) and abs(np.log(h)) <= max(plac)
        rows.append(row)
    t = pd.DataFrame(rows)
    glob = t[t["region"] == "global"]
    verdict = bool(len(glob) and glob["frp_pass"].iloc[0] and glob["cells_pass"].iloc[0])
    reg = t[(t["region"] != "global") & ~t["region"].str.startswith("event:")]
    return {"table": t.to_dict("records"), "global_pass": verdict,
            "regions_passing_both": f"{int((reg['frp_pass'] & reg['cells_pass']).sum())} of {len(reg)}" if len(reg) else None,
            "years": [y_lo, y_hi]}


# ---------------------------------------------------------------- 8. cloud

def cloud_check(settings, cubes, ref, loyo_frp: pd.DataFrame, cells, weights, cfg, all_overlap, fine) -> dict:
    cp = settings.processed_dir / "external" / "cloud_monthly_fine.parquet"
    if not cp.exists():
        return {"note": "no cloud table (run stage external)"}
    cc = cfg.get("cloud") or {}
    pct, min_ratio = float(cc.get("percentile", 90)), float(cc.get("min_ratio", 1.5))
    by = [int(x) for x in cfg.get("baseline_years", [2003, 2025])]
    cl = pd.read_parquet(cp)
    from .tiles import tile_name
    cl["region"] = [tile_name(a, b, fine) for a, b in zip(cl["lat0"], cl["lon0"])]
    inb = cl["year"].between(by[0], by[1])
    thr = cl[inb].groupby(["region", "month"])["cloud"].quantile(pct / 100).rename("thr")
    clim = cl[inb].groupby(["region", "month"])["cloud"].mean().rename("clim")
    cl = cl.join(thr, on=["region", "month"]).join(clim, on=["region", "month"])
    cl["cloudy"] = cl["cloud"] >= cl["thr"]
    cl["anom"] = cl["cloud"] - cl["clim"]
    # (i) do scaling errors grow with cloud?
    p = loyo_frp[loyo_frp["region"].isin(cells)].merge(cl[["region", "year", "month", "anom", "cloudy"]],
                                                        on=["region", "year", "month"], how="inner")
    err = None
    if len(p):
        p["lerr"] = np.abs(np.log((p["pred"] + 10) / (p["real"] + 10)))
        p["tercile"] = pd.qcut(p["anom"], 3, labels=["clearer than usual", "usual", "cloudier than usual"])
        w = p["region"].map(weights).fillna(0)
        err = {str(k): round(float(100 * (np.exp(wmedian(g["lerr"], w[g.index])) - 1)), 1) for k, g in p.groupby("tercile", observed=True)}
    # (ii) are low S-NPP months cloudy more often than other months?
    c = cubes["frp"]
    truth = preds(c, ref, ref, all_overlap, np.ones((len(c.keys), 12)))
    truth = truth[truth["region"].isin(cells)]
    base = baseline(truth[["region", "year", "month", "real"]].rename(columns={"real": ref}), ref, all_overlap, [])
    t = truth.merge(base, on=["region", "month"]).merge(cl[["region", "year", "month", "cloudy"]], on=["region", "year", "month"], how="inner")
    t["cls"] = classify(t["real"], t["mu"], t["sd"])
    base_rate = float(100 * t["cloudy"].mean()) if len(t) else None
    low_rate = float(100 * t.loc[t["cls"] == "low", "cloudy"].mean()) if (t["cls"] == "low").any() else None
    adopt = bool(base_rate and low_rate is not None and low_rate >= min_ratio * base_rate)
    out = settings.processed_dir / "checks"
    out.mkdir(parents=True, exist_ok=True)
    if adopt:
        cl[cl["cloudy"]][["lat0", "lon0", "year", "month", "cloud", "thr"]].to_parquet(out / "cloud_flags.parquet", index=False)
    return {"scaling_error_by_cloud_pct": err, "cloudy_share_all_months_pct": None if base_rate is None else round(base_rate, 1),
            "cloudy_share_low_months_pct": None if low_rate is None else round(low_rate, 1),
            "flag_shown": adopt, "rule": f"flag shown if low months are cloudy at least {min_ratio:g} times as often as all months"}


# ---------------------------------------------------------------- the stage

def run_checks(settings: Settings) -> dict:
    cfg = settings.raw.get("phase2_checks") or {}
    fcfg = settings.raw.get("factors") or {}
    ref = fcfg.get("reference", "S-NPP")
    parent = float(fcfg.get("tile_deg", 10))
    prior = {"cells": float(fcfg.get("prior_cells", 30)), "frp": float(fcfg.get("prior_frp_mw", 300))}
    min_v = {"cells": float(fcfg.get("min_cells_per_month", 30)), "frp": float(fcfg.get("min_frp_mw_per_month", 300))}
    yrs = lambda v: list(range(int(v[0]), int(v[1]) + 1))  # noqa: E731
    years = yrs(fcfg.get("overlap_years", [2012, 2025]))
    bw_fit, bw_test = yrs(fcfg.get("backward_fit_years", [2016, 2025])), yrs(fcfg.get("backward_test_years", [2012, 2015]))
    fp = settings.processed_dir / "counts" / "fine_monthly.parquet"
    if not fp.exists():
        return {"stage": "checks", "note": "no counts/fine_monthly.parquet: run stage resolution first"}
    spec = model_spec(settings)
    fine = spec["fine_deg"]
    segments = source_segments(settings.raw.get("counts") or {})
    g, y0, y1 = segments[0]
    m = pd.read_parquet(fp)
    all_years = list(range(int(m["year"].min()), int(m["year"].max()) + 1))
    cubes = {v: Cube(m, v, True, False, parent, all_years, fine) for v in ("cells", "frp")}
    for c in cubes.values():
        c.tile_deg = fine
    from .tiles import tile_name
    r = m[(m["group"] == ref) & m["year"].isin(years) & ~m["in_mask"]]
    r = r.assign(cell=[tile_name(a, b, fine) for a, b in zip(r["lat0"], r["lon0"])])
    per_cell = r.groupby("cell")["active_cells"].sum() / (12 * len(years))
    cells = per_cell[per_cell >= float((settings.raw.get("resolution") or {}).get("min_ref_cells_per_month", 10))].index
    weights = r.groupby("cell")["active_cells"].sum()
    regions = {k: v for k, v in check_regions(settings).items() if not k.startswith("event:")}
    rmask = area_of(cubes["cells"], fine, regions)
    allowed = spec["fit_years"].get(g)
    fit_span = [y for y in years if int(y0) <= y <= int(y1) and (allowed is None or y in allowed)]
    report = {"stage": "checks", "model": {k: v for k, v in spec.items()}, "tested_cells": int(len(cells))}
    out = settings.processed_dir / "checks"
    out.mkdir(parents=True, exist_ok=True)

    def part(name, fn):
        print(f"[checks] {name} ...", flush=True)
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - one failing check must not hide the others
            return {"error": f"{type(e).__name__}: {e}"}

    anom = part("1 anomaly rule on the final factors",
                lambda: anomaly_check(cubes, g, ref, fit_span, years, spec, prior, min_v, cells, weights, settings, rmask))
    loyo_frp = pd.DataFrame()
    if isinstance(anom, tuple):
        anom, loyo_frp = anom
    report["anomaly_rule"] = anom
    report["final_test_2026"] = part("2 untouched 2026 test", lambda: final_test(settings, cubes, ref, cells, weights, rmask,
                                                                                    cfg.get("final_test") or {}, fine))
    report["coverage"] = part("3 coverage of the ranges", lambda: coverage_check(settings, cubes, g, y0, y1, ref, years, spec, prior,
                                                                                   min_v, cells, weights, rmask, cfg.get("coverage") or {}, fine))
    sm = part("4 simple methods", lambda: simple_methods(cubes, g, y0, y1, ref, years, bw_fit, bw_test, spec, prior, min_v, cells, weights))
    if isinstance(sm, pd.DataFrame):
        sm.to_csv(out / "simple_methods.csv", index=False)
        report["simple_methods"] = sm.to_dict("records")
    else:
        report["simple_methods"] = sm
    report["known_events"] = part("5 known events", lambda: event_check(settings, cfg))
    report["critical_periods"] = part("6 dated critical periods", lambda: critical_periods(settings, cfg))
    report["burned_area"] = part("7 GFED burned area", lambda: gfed_check(settings, cfg))
    report["cloud"] = part("8 cloud", lambda: cloud_check(settings, cubes, ref, loyo_frp, cells, weights, cfg, years, fine))

    # earlier stages of this package, read back for one summary
    def rep(name):
        p = settings.processed_dir / f"report_{name}.json"
        return json.loads(p.read_text()) if p.exists() else {}
    res, dly, har = rep("resolution"), rep("daily"), rep("harmonise")
    summary = {
        "MF2 hybrid fills": {"fill_methods": har.get("fill_methods"), "tile_days_scaled": har.get("tile_days_scaled_hybrid")},
        "MF3 2018 version": (res.get("version_test") or {}).get("chosen"),
        "MF6 time scales before 2012": dly.get("labels_before_2012"),
        "F day basis": dly.get("day_basis"),
        "E regions": {"overrides": [(o["region"], o["variant"]) for o in (res.get("region_test") or {}).get("overrides", [])],
                      "flagged": [(f["region"], f["step_gap_pct"]) for f in (res.get("region_test") or {}).get("flags", [])]},
        "gate 2012": (har.get("gate_2012_step") or {}).get("pass"),
        "MF1 anomaly": anom.get("adopted") if isinstance(anom, dict) else None,
        "MF4 2026 test": (report["final_test_2026"] or {}).get("pass"),
        "MF5 ranges recalibrated": (report["coverage"] or {}).get("accepted"),
        "MF7 events hit (SHIFT frp)": ((report["known_events"] or {}).get("summary") or {}).get("SHIFT frp"),
        "D events as dated runs": (report["critical_periods"] or {}).get("event_detection_rate_pct"),
        "B burned area (global)": (report["burned_area"] or {}).get("global_pass"),
        "C cloud flag": (report["cloud"] or {}).get("flag_shown"),
    }
    report["summary"] = summary
    report["provenance"] = stamp(settings)
    (out / "phase2_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    with pd.option_context("display.width", 240, "display.max_columns", 30, "display.max_rows", 80):
        a = report["anomaly_rule"]
        print(f"[checks] 1 ANOMALY RULE: {a.get('adopted')} | {a.get('why', a.get('error', a.get('note')))}", flush=True)
        ft = report["final_test_2026"]
        print(f"[checks] 2 FINAL 2026 TEST: {'PASS' if ft.get('pass') else 'FAIL'} {json.dumps(ft.get('predictors', ft), default=str)}", flush=True)
        print(f"[checks] 3 COVERAGE: {json.dumps(report['coverage'], default=str)}", flush=True)
        if isinstance(sm, pd.DataFrame):
            print("[checks] 4 SIMPLE METHODS (same tests):\n" + sm.to_string(index=False), flush=True)
        ke = report["known_events"]
        if "table" in ke:
            print("[checks] 5 KNOWN EVENTS:\n" + pd.DataFrame(ke["table"]).to_string(index=False), flush=True)
            print(f"[checks] 5 hit rates: {json.dumps(ke['summary'])}", flush=True)
        else:
            print(f"[checks] 5 KNOWN EVENTS: {ke}", flush=True)
        print(f"[checks] 6 DATED CRITICAL PERIODS: {json.dumps(report['critical_periods'], default=str)}", flush=True)
        ba = report["burned_area"]
        if "table" in ba:
            print("[checks] 7 BURNED AREA ratio steps at 2012:\n" + pd.DataFrame(ba["table"]).to_string(index=False), flush=True)
        print(f"[checks] 7 global pass: {ba.get('global_pass')}; regions: {ba.get('regions_passing_both')} {ba.get('note', ba.get('error', ''))}", flush=True)
        print(f"[checks] 8 CLOUD: {json.dumps(report['cloud'], default=str)}", flush=True)
        print("[checks] PHASE 2 SUMMARY:\n" + "\n".join(f"  {k}: {v}" for k, v in summary.items()), flush=True)
    return report
