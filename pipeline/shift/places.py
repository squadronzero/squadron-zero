"""Phase 2, fix 8: per-place work. Where the worldwide choices are weakest, test place-by-place
alternatives, each under a rule written before the run (settings `places.rule`), and measure
how good the factors for thin places really are.

Five parts, in this order (later parts use what earlier parts keep):

  B. Stronger non-fire mask in FLARE TILES. D22 removes cells NASA labelled as non-fire in at
     least 2 years. In gas-flare regions that left errors (Libya -42%, Komi +33%, western
     Siberia +56%): flares labelled in only one year, and flare light spilling into the next
     grid cell (a MODIS pixel at the swath edge is up to 2 x 5 km, a grid cell 5.5 km).
     Variants, applied ONLY in flare tiles (tiles where the D22 mask holds at least
     `flare_share` of the tile's MODIS burning cells):
        mask1      also remove cells labelled non-fire in a single year
        mask1ring  also remove the 8 neighbours of every labelled cell
     Rule (same as D22): kept if the fire-weighted fire power score drops by no more than
     `mask_max_drop_points` AND the share of tiles within +-error_limit_pct rises.
  A. MODIS source per tile. D19 (merged Terra + Aqua to 2019, Aqua from 2020) won on the
     fire-weighted score, but merged MODIS everywhere was best in half the tiles. Choosing per
     tile on the same tests that then score it would flatter the choice, so the test is NESTED:
     each tile picks its source on leave-one-year-out agreement over `selection_years`
     (2016-2025) only, and all options are then judged on the backward test (fit 2016-2025,
     predict 2012-2015), which the choice never saw. Rule: the per-tile choice is adopted only
     if its fire-weighted backward agreement beats D19 everywhere by more than
     `source_gain_points` AND it has at least as many tiles within +-error_limit_pct.
  C. How good are BORROWED factors? 237 thin tiles get factors from their own all-year value,
     their 30-degree zone or the world. Every tested tile is treated as if it were thin: its
     factor is rebuilt from its zone and from the world WITHOUT the tile itself, then used to
     predict 2012-2015 from 2016-2019. The measured errors replace guesses in the fix 9
     uncertainty.
  D. Confidence drift (from fix 5). Where the gap between S-NPP's and MODIS's "low confidence"
     share changes over the years (boreal Canada and Siberia), a factor fitted on 2012-2019
     carries that change back into 2002-2011. Extra uncertainty per tile = |gap trend| x
     `extrapolation_years`.
  E. Spotlight on one place (`spotlight`, Bangladesh): its own backward test and a first
     harmonised monthly calendar 2003-2025 with unusual and critical months marked.

    python -m shift.run --dataset global --stage places     (after factors and confidence)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings
from .counts import GROUPS, source_segments
from .coverage import bad_days
from .factors import Cube, evaluate, monthly_world
from .grid import grid_shape
from .region import test_regions
from .selection import baseline, classify, predict, score
from .static_mask import masked_cells
from .tiles import tile_name

LEVELS = ["own month", "own all-year", "zone (without the tile)", "world (without the tile)"]


# ---------------------------------------------------------------- B. mask tiers

def mask_tiers(settings: Settings) -> pd.Series | None:
    """Label cells by how strongly NASA marked them as non-fire: 'm2' (labelled in at least
    min_years years: the D22 mask), 'm1' (labelled in fewer years), 'ring' (next to a labelled
    cell). Unlisted cells are clean. None if the scan stage's per-year files are missing."""
    cfg = settings["static_mask"]
    src = settings.path(cfg["sources_dir"])
    fams = [k for k, v in settings["fetch"]["products"].items() if v.get("scan", True)]
    files = [f for f in sorted(src.glob("*.parquet")) if f.stem.rsplit("_", 1)[0] in fams
             and f.stem.rsplit("_", 1)[-1].isdigit()] if src.exists() else []
    if not files:
        m2 = masked_cells(settings)
        return None if m2 is None else pd.Series("m2", index=pd.Index(m2, name="cell_id"))
    allp = pd.concat([pd.read_parquet(f, columns=["cell_id"]).assign(year=int(f.stem.rsplit("_", 1)[1]))
                      for f in files], ignore_index=True)
    n_years = allp.groupby("cell_id")["year"].nunique()
    tier = pd.Series(np.where(n_years >= int(cfg.get("min_years", 2)), "m2", "m1"), index=n_years.index)
    n_rows, n_cols = grid_shape(settings.resolution)
    row, col = np.divmod(tier.index.to_numpy("int64"), n_cols)
    ring = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r = row + dr
            ok = (r >= 0) & (r < n_rows)
            ring.append(r[ok] * n_cols + (col[ok] + dc) % n_cols)
    ring = np.setdiff1d(np.unique(np.concatenate(ring)), tier.index.to_numpy("int64"))
    out = pd.concat([tier, pd.Series("ring", index=ring)])
    out.index.name = "cell_id"
    return out


def variant_frame(m: pd.DataFrame, flare_tiles: set, drop_in_flare: list[str]) -> pd.DataFrame:
    """D22 everywhere (drop 'm2'), plus the extra tiers dropped in flare tiles only."""
    tile = m["tile"]
    drop = m["size"].eq("m2") | (m["size"].isin(drop_in_flare) & tile.isin(flare_tiles))
    return m[~drop].assign(in_mask=False)


# ---------------------------------------------------------------- A. source per tile (nested)

def by_tile_scores(cube: Cube, segments, ref, fit_years, test_years, prior, min_v, base) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = cube.predict(segments, ref, fit_years, test_years, False, False, prior, min_v)
    return p, score(p, base, []).set_index("region") if len(p) else pd.DataFrame()


def loyo_tiles(cube: Cube, segments, ref, years, prior, min_v, base) -> pd.DataFrame:
    parts = [cube.predict(segments, ref, [x for x in years if x != y], [y], False, False, prior, min_v) for y in years]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame()
    return score(pd.concat(parts, ignore_index=True), base, []).set_index("region")


def summarise(p: pd.DataFrame, base: pd.DataFrame, tiles, weights: pd.Series, limit: float) -> dict:
    p = p[p["region"].isin(tiles)]
    s = score(p, base, []).set_index("region")
    w = weights.reindex(s.index).fillna(0)
    ok = s["agreement_pct"].notna() & (w > 0)
    tot = p["real"].sum()
    return {"backward_agreement": round(float(np.average(s["agreement_pct"][ok], weights=w[ok])), 2) if ok.any() else None,
            "tiles_within_limit_pct": round(float(100 * (s["total_error_pct"].abs() <= limit).mean()), 1) if len(s) else None,
            "global_backward_error_pct": round(float(100 * (p["pred"].sum() / tot - 1)), 2) if tot > 0 else None,
            "tiles": int(ok.sum()), "per_tile": s}


def decide_source(d19: dict, per_tile: dict, rule: dict) -> tuple[str, str]:
    gain = float(rule.get("source_gain_points", 2))
    a0, a1 = d19["backward_agreement"], per_tile["backward_agreement"]
    t0, t1 = d19["tiles_within_limit_pct"], per_tile["tiles_within_limit_pct"]
    if a1 is not None and a0 is not None and a1 > a0 + gain and t1 >= t0:
        return "per tile", f"per-tile choice {a1} vs D19 {a0} (+{a1 - a0:.2f}), tiles within limit {t1}% vs {t0}%"
    return "D19 everywhere", (f"per-tile choice does not beat D19 by more than {gain:g} points with at least as many "
                              f"tiles within the limit (per tile {a1}, D19 {a0}; tiles {t1}% vs {t0}%)")


# ---------------------------------------------------------------- C. borrowed factors, tested honestly

def loto_borrowed(cube: Cube, src: str, ref: str, fit_years, test_years, min_v: float,
                  tiles, weights: pd.Series) -> pd.DataFrame:
    """Each tested tile predicted from fit_years to test_years with its own factor and with
    factors borrowed from its zone and the world WITHOUT the tile. Errors per tile and level,
    for the tile's 4-year total and for each calendar month's total."""
    yi = {y: i for i, y in enumerate(cube.years)}
    fy = [yi[y] for y in fit_years if y in yi]
    ty = [yi[y] for y in test_years if y in yi]
    if not fy or not ty or src not in cube.A or ref not in cube.A:
        return pd.DataFrame()
    S_, R_ = cube.A[src], cube.A[ref]
    W = (cube.avail[src] & cube.avail[ref]).astype(float)                 # month x year
    S = np.einsum("kmy,my->km", S_[:, :, fy], W[:, fy])
    R = np.einsum("kmy,my->km", R_[:, :, fy], W[:, fy])
    St = np.einsum("kmy,my->km", S_[:, :, ty], W[:, ty])                  # test-year MODIS per month
    Rt = np.einsum("kmy,my->km", R_[:, :, ty], W[:, ty])                  # test-year truth
    zone = cube.Z.argmax(axis=0)
    Sz, Rz = cube.Z @ S, cube.Z @ R
    Sw, Rw = S.sum(0), R.sum(0)
    names = cube.keys["tile"].to_numpy()
    rows = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in np.nonzero(np.isin(names, list(tiles)))[0]:
            s, r = S[i], R[i]
            k_all = r.sum() / s.sum() if s.sum() >= min_v else np.nan
            ks = {LEVELS[0]: np.where(s >= min_v, r / np.where(s > 0, s, 1), k_all),
                  LEVELS[1]: np.full(12, k_all),
                  LEVELS[2]: (Rz[zone[i]] - r) / (Sz[zone[i]] - s),
                  LEVELS[3]: (Rw - r) / (Sw - s)}
            truth_m, truth = Rt[i], Rt[i].sum()
            if truth <= 0:
                continue
            for lev, k in ks.items():
                k = np.where(np.isfinite(k) & (k > 0), k, np.nan)
                pred_m = St[i] * k
                okm = np.isfinite(pred_m) & (truth_m > 0)
                if not okm.any():
                    continue
                pred = np.nansum(pred_m)
                rows.append({"tile": names[i], "level": lev, "fire": float(weights.get(names[i], 0)),
                             "total_error_pct": 100 * (pred / truth - 1),
                             "month_abs_error_pct": float(100 * np.average(np.abs(pred_m[okm] / truth_m[okm] - 1),
                                                                          weights=truth_m[okm])),
                             "log_ratio": float(np.log(pred / truth)) if pred > 0 else np.nan})
    return pd.DataFrame(rows)


def loto_summary(t: pd.DataFrame, limit: float) -> pd.DataFrame:
    out = []
    for lev in LEVELS:
        d = t[t["level"] == lev].dropna(subset=["log_ratio"])
        if d.empty:
            continue
        w = d["fire"].to_numpy()
        a = d["total_error_pct"].abs()
        order = np.argsort(a.to_numpy())
        cw = np.cumsum(w[order]) / w.sum() if w.sum() else np.linspace(0, 1, len(d))
        out.append({"level": lev, "tiles": int(len(d)),
                    "fw_median_abs_total_error_pct": round(float(a.to_numpy()[order][np.searchsorted(cw, 0.5)]), 1),
                    "median_abs_total_error_pct": round(float(a.median()), 1),
                    "tiles_within_limit_pct": round(float(100 * (a <= limit).mean()), 1),
                    "median_month_abs_error_pct": round(float(d["month_abs_error_pct"].median()), 1),
                    # measured 95% half-width on the log scale: what fix 9 uses for borrowed factors
                    "half_width_95_log": round(float(np.percentile(d["log_ratio"].abs(), 95)), 3)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- E. spotlight

def spotlight(settings: Settings, name: str, segments, ref: str, years, bw_fit, bw_test,
              min_cells: int, comparable_start: int) -> dict | None:
    path = settings.processed_dir / "counts" / "daily.parquet"
    if not path.exists() or name not in test_regions(settings):
        return None
    d = pd.read_parquet(path)
    d = d[d["region"] == name]
    if d.empty or "frp_mw" not in d:
        return None
    d["acq_date"] = pd.to_datetime(d["acq_date"])
    sats = sorted({s for g, *_ in segments for s in GROUPS[g]} | set(GROUPS[ref]))
    bad = bad_days(settings, sats)
    if bad is not None and len(bad):
        d = d[~d["acq_date"].isin(bad)]
    d = d.assign(year=d["acq_date"].dt.year, month=d["acq_date"].dt.month)
    out = {}
    for var, col in (("frp", "frp_mw"), ("cells", "active_cells")):
        w = d.pivot_table(index=["region", "year", "month"], columns="group", values=col, aggfunc="sum").reset_index()
        base = baseline(w, ref, years, [])
        p = predict(w, segments, ref, bw_fit, bw_test, min_cells if var == "cells" else 300, [])
        s = score(p, base, [])
        out[var] = s.iloc[0].to_dict() if len(s) else {}
        if var == "frp":
            # first harmonised calendar: MODIS-era months = D19 MODIS piece x k (fitted on all
            # overlap years inside its span), from 2012 the real S-NPP record
            g, y0, y1 = segments[0]
            span = [y for y in years if int(y0) <= y <= int(y1)]
            fit = w[w["year"].isin(span)].dropna(subset=[g, ref])
            k = fit.groupby("month")[ref].sum() / fit.groupby("month")[g].sum()
            pre = w[(w["year"] >= comparable_start) & (w["year"] < 2012)].dropna(subset=[g])
            cal = pd.concat([pre.assign(value=pre[g] * pre["month"].map(k), source=f"{g} x k"),
                             w[w["year"] >= 2012].dropna(subset=[ref]).assign(value=lambda x: x[ref], source=ref)])
            cal = cal[cal["year"] <= max(years)][["year", "month", "value", "source"]].sort_values(["year", "month"])
            v = np.log1p(cal["value"].clip(lower=0))
            st = cal.assign(v=v).groupby("month")["v"].agg(mu="mean", sd="std")
            st["sd"] = st["sd"].clip(lower=0.05)
            cal["class"] = classify(cal["value"], cal["month"].map(st["mu"]), cal["month"].map(st["sd"]))
            cal["k"] = np.where(cal["source"] == ref, 1.0, cal["month"].map(k))
            out["calendar"] = cal.reset_index(drop=True)
            out["k_by_month"] = k.round(3).to_dict()
    return out


# ---------------------------------------------------------------- the stage

def run_places(settings: Settings) -> dict:
    cfg = settings.raw.get("places") or {}
    fcfg = settings.raw.get("factors") or {}
    rule = cfg.get("rule") or {}
    limit = float(rule.get("error_limit_pct", 10))
    ref = fcfg.get("reference", "S-NPP")
    tile_deg, zone_deg = float(fcfg.get("tile_deg", 10)), float(fcfg.get("zone_deg", 30))
    min_ref = float(fcfg.get("min_ref_cells_per_month", 50))
    prior = {"cells": float(fcfg.get("prior_cells", 30)), "frp": float(fcfg.get("prior_frp_mw", 300))}
    min_v = {"cells": float(fcfg.get("min_cells_per_month", 30)), "frp": float(fcfg.get("min_frp_mw_per_month", 300))}
    span = lambda key, d, c=fcfg: list(range(int(c.get(key, d)[0]), int(c.get(key, d)[1]) + 1))  # noqa: E731
    years = span("overlap_years", [2012, 2025])
    bw_fit, bw_test = span("backward_fit_years", [2016, 2025]), span("backward_test_years", [2012, 2015])
    sel_years = span("selection_years", [2016, 2025], cfg)
    d19 = source_segments(settings.raw.get("counts") or {})
    merged = [["MODIS", 2000, 2100]]
    groups = {g: GROUPS[g] for g in sorted({g for g, *_ in d19} | {"MODIS", ref})}
    bad = bad_days(settings, sorted({s for g in groups for s in GROUPS[g]})) if fcfg.get("exclude_gap_days", True) else None
    out = settings.processed_dir / "places"
    out.mkdir(parents=True, exist_ok=True)

    tiers = mask_tiers(settings)
    print(f"[places] reading the record once ({0 if tiers is None else len(tiers):,} labelled or neighbouring cells); "
          "this prints nothing for several minutes", flush=True)
    m = monthly_world(settings, groups, tile_deg, bad, tiers, None)
    if m.empty:
        return {"stage": "places", "note": "no cell-days found"}
    m["tile"] = [tile_name(a, b, tile_deg) for a, b in zip(m["lat0"], m["lon0"])]
    all_years = list(range(int(m["year"].min()), int(m["year"].max()) + 1))

    # tested tiles and fire weights exactly as stage factors (D22 data)
    clean = m[m["size"] != "m2"]
    r = clean[(clean["group"] == ref) & clean["year"].isin(years)]
    per_tile = r.groupby("tile")["active_cells"].sum() / (12 * len(years))
    tiles = per_tile[per_tile >= min_ref].index
    weights = r.groupby("tile")["active_cells"].sum()

    # flare tiles: share of the tile's MODIS burning cells that sit in D22 cells
    mo = m[(m["group"] == "MODIS") & m["year"].isin(years)]
    tot = mo.groupby("tile")["active_cells"].sum()
    in_m2 = mo[mo["size"] == "m2"].groupby("tile")["active_cells"].sum().reindex(tot.index, fill_value=0)
    share = (in_m2 / tot).fillna(0)
    flare_tiles = set(share[share >= float(cfg.get("flare_share", 0.2))].index)

    def cube(frame, var):
        return Cube(frame, var, False, False, zone_deg, all_years, tile_deg)

    def ev(frame, var="frp", segs=d19):
        return evaluate(cube(frame, var), segs, ref, years, bw_fit, bw_test, False, False, prior[var], min_v[var],
                        tiles, weights, limit)

    # ---- B. mask strength in flare tiles
    steps, frames = [], {}
    frames["D22"] = variant_frame(m, flare_tiles, [])
    cur_name, cur = "D22", ev(frames["D22"])
    steps.append({"step": "D22 (current)", **{k: v for k, v in cur.items() if k != "per_tile"}, "kept": True})
    d22_res = cur
    max_drop = float(rule.get("mask_max_drop_points", 1))
    cur_extra = []
    for label, add in (("+ cells labelled non-fire in only 1 year (flare tiles only)", "m1"),
                       ("+ the 8 neighbours of every labelled cell (flare tiles only)", "ring")):
        extra = cur_extra + [add]
        name = "+".join(["D22"] + extra)
        frames[name] = variant_frame(m, flare_tiles, extra)
        res = ev(frames[name])
        keep = (res["fire_weighted_score"] is not None and cur["fire_weighted_score"] is not None
                and res["fire_weighted_score"] >= cur["fire_weighted_score"] - max_drop
                and res["tiles_within_limit_pct"] > cur["tiles_within_limit_pct"])
        steps.append({"step": label, **{k: v for k, v in res.items() if k != "per_tile"}, "kept": bool(keep)})
        if keep:
            cur_name, cur, cur_extra = name, res, extra
    steps = pd.DataFrame(steps)
    chosen_frame = frames[cur_name]
    watch = [t for t in fcfg.get("watch_tiles", []) if t in cur["per_tile"].index]
    watch_tbl = pd.DataFrame({"D22_backward_error_pct": d22_res["per_tile"]["backward_error_pct"].reindex(watch),
                              "chosen_backward_error_pct": cur["per_tile"]["backward_error_pct"].reindex(watch),
                              "flare_tile": [t in flare_tiles for t in watch]}).round(2)

    # ---- A. MODIS source per tile, nested
    c = cube(chosen_frame, "frp")
    truth = c.predict(d19, ref, years, years, False, False, prior["frp"], min_v["frp"])
    base = baseline(truth[["region", "year", "month", "real"]].rename(columns={"real": ref}), ref, years, [])
    base_sel = baseline(truth[["region", "year", "month", "real"]].rename(columns={"real": ref}), ref, sel_years, [])
    lo_d19 = loyo_tiles(c, d19, ref, sel_years, prior["frp"], min_v["frp"], base_sel)
    lo_mer = loyo_tiles(c, merged, ref, sel_years, prior["frp"], min_v["frp"], base_sel)
    pick = pd.Series("D19", index=lo_d19.index)
    both = lo_d19.index.intersection(lo_mer.index)
    pick.loc[both] = np.where(lo_mer.loc[both, "agreement_pct"] > lo_d19.loc[both, "agreement_pct"], "merged", "D19")
    bw_d19 = c.predict(d19, ref, bw_fit, bw_test, False, False, prior["frp"], min_v["frp"])
    bw_mer = c.predict(merged, ref, bw_fit, bw_test, False, False, prior["frp"], min_v["frp"])
    reg_pick = bw_d19["region"].map(pick).fillna("D19")
    bw_per = pd.concat([bw_d19[reg_pick == "D19"], bw_mer[bw_mer["region"].map(pick).fillna("D19") == "merged"]])
    res_src = {"D19 everywhere": summarise(bw_d19, base, tiles, weights, limit),
               "merged everywhere": summarise(bw_mer, base, tiles, weights, limit),
               f"per tile (chosen on {sel_years[0]}-{sel_years[-1]} only)": summarise(bw_per, base, tiles, weights, limit)}
    src_choice, src_why = decide_source(res_src["D19 everywhere"], res_src[f"per tile (chosen on {sel_years[0]}-{sel_years[-1]} only)"], rule)
    src_tbl = pd.DataFrame([{"option": k, **{a: b for a, b in v.items() if a != "per_tile"}} for k, v in res_src.items()])
    picks = pick[pick.index.isin(tiles)]

    # ---- C. borrowed factors, leave the tile out
    g0, y0, y1 = d19[0]
    fit_c = [y for y in bw_fit if int(y0) <= y <= int(y1)]
    test_c = [y for y in bw_test if int(y0) <= y <= int(y1)]
    loto = loto_borrowed(c, g0, ref, fit_c, test_c, min_v["frp"], tiles, weights)
    loto_tbl = loto_summary(loto, limit) if len(loto) else pd.DataFrame()

    # ---- D. confidence drift uncertainty
    conf_path = settings.processed_dir / "counts" / "confidence_tile_asymmetry.csv"
    ext = float(cfg.get("extrapolation_years", 9))
    conf = None
    if conf_path.exists():
        ct = pd.read_csv(conf_path, index_col=0)
        if "gap_trend_pct_points_per_year" in ct:
            conf = (ct["gap_trend_pct_points_per_year"].abs() * ext).rename("confidence_drift_pct").round(2)

    # ---- per-tile table for fix 9
    tile_tbl = pd.DataFrame(index=pd.Index(sorted(set(m["tile"])), name="tile"))
    tile_tbl["tested"] = tile_tbl.index.isin(tiles)
    tile_tbl["flare_tile"] = tile_tbl.index.isin(flare_tiles)
    tile_tbl["d22_share_of_modis_cells"] = share.reindex(tile_tbl.index).round(3)
    tile_tbl["mask"] = np.where(tile_tbl["flare_tile"], cur_name, "D22")
    tile_tbl["modis_source"] = (pick.reindex(tile_tbl.index).fillna("D19") if src_choice == "per tile" else "D19")
    tile_tbl["source_loyo_pick_2016_2025"] = pick.reindex(tile_tbl.index)
    tile_tbl["backward_error_pct"] = cur["per_tile"]["backward_error_pct"].reindex(tile_tbl.index)
    if conf is not None:
        tile_tbl["confidence_drift_pct"] = conf.reindex(tile_tbl.index)

    # ---- E. spotlight
    spot_name = cfg.get("spotlight", "bangladesh")
    rec_start = int(str((settings.raw.get("record") or {}).get("comparable_start", "2003"))[:4])
    spot = spotlight(settings, spot_name, d19, ref, years, bw_fit, bw_test,
                     int((settings.raw.get("counts") or {}).get("min_cells_per_month", 30)), rec_start)
    region_tiles = []
    if spot_name in test_regions(settings):
        w_, s_, e_, n_ = test_regions(settings)[spot_name]["bbox"]
        for la in np.arange(np.floor(s_ / tile_deg) * tile_deg, n_, tile_deg):
            for lo in np.arange(np.floor(w_ / tile_deg) * tile_deg, e_, tile_deg):
                region_tiles.append(tile_name(la, lo, tile_deg))

    steps.to_csv(out / "mask_steps.csv", index=False)
    watch_tbl.to_csv(out / "mask_watch_tiles.csv")
    src_tbl.to_csv(out / "source_per_tile_test.csv", index=False)
    if len(loto):
        loto.to_csv(out / "borrowed_factor_test_per_tile.csv", index=False)
        loto_tbl.to_csv(out / "borrowed_factor_test.csv", index=False)
    tile_tbl.to_csv(out / "place_choices.csv")
    if spot and "calendar" in spot:
        spot["calendar"].to_csv(out / f"{spot_name}_calendar.csv", index=False)

    flagged = []
    if spot and "calendar" in spot:
        cal = spot["calendar"]
        flagged = cal[cal["class"].isin(["unusual", "critical"])][["year", "month", "class", "source"]].to_dict("records")
    report = {"stage": "places", "flare_tiles": sorted(flare_tiles), "tiles_tested": int(len(tiles)),
              "mask_chosen": cur_name, "mask_steps": steps.to_dict("records"),
              "source_choice": src_choice, "source_why": src_why,
              "source_options": src_tbl.to_dict("records"),
              "source_picks_tested_tiles": picks.value_counts().to_dict(),
              "borrowed_factor_test": loto_tbl.to_dict("records") if len(loto_tbl) else None,
              "confidence_drift_tiles_over_5pct": (int((conf > 5).sum()) if conf is not None else None),
              "spotlight": None if not spot else {
                  "region": spot_name, "tiles": region_tiles,
                  "backward_test_frp": {k: spot["frp"].get(k) for k in ("total_error_pct", "agreement_pct", "high_caught_pct", "false_alarm_pct", "months")},
                  "backward_test_cells": {k: spot["cells"].get(k) for k in ("total_error_pct", "agreement_pct", "high_caught_pct", "months")},
                  "k_frp_by_month": spot.get("k_by_month"),
                  "unusual_or_critical_months": flagged}}
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 80):
        print(f"[places] {len(tiles)} tested tiles; {len(flare_tiles)} flare tiles "
              f"(D22 cells hold >= {100 * float(cfg.get('flare_share', 0.2)):.0f}% of MODIS burning cells): {sorted(flare_tiles)}", flush=True)
        print("[places] B. mask strength in flare tiles (pre-registered rule, fire power):\n"
              + steps.drop(columns=["per_tile"], errors="ignore").to_string(index=False), flush=True)
        print(f"[places] B. CHOSEN mask: {cur_name}", flush=True)
        if len(watch_tbl):
            print("[places] watch tiles, backward fire power error %:\n" + watch_tbl.to_string(), flush=True)
        print(f"[places] A. MODIS source, judged on the backward test {bw_test[0]}-{bw_test[-1]} (never used for the choice):\n"
              + src_tbl.to_string(index=False), flush=True)
        print(f"[places] A. tested tiles picking each source on {sel_years[0]}-{sel_years[-1]}: {picks.value_counts().to_dict()}", flush=True)
        print(f"[places] A. RULE: {src_choice} | {src_why}", flush=True)
        if len(loto_tbl):
            print(f"[places] C. borrowed factors tested honestly (each tile rebuilt without itself, {fit_c[0]}-{fit_c[-1]} -> {test_c[0]}-{test_c[-1]}):\n"
                  + loto_tbl.to_string(index=False), flush=True)
        if conf is not None:
            print(f"[places] D. confidence drift: {int((conf > 5).sum())} tiles get more than 5% extra uncertainty; largest:\n"
                  + conf.sort_values(ascending=False).head(8).to_string(), flush=True)
        if spot:
            print(f"[places] E. {spot_name} (tiles {region_tiles}) backward test {bw_test[0]}-{bw_test[-1]}, fire power: "
                  f"error {spot['frp'].get('total_error_pct')}%, agreement {spot['frp'].get('agreement_pct')}%, "
                  f"unusual months caught {spot['frp'].get('high_caught_pct')}%; burning cells: error "
                  f"{spot['cells'].get('total_error_pct')}%, agreement {spot['cells'].get('agreement_pct')}%", flush=True)
            print(f"[places] E. {spot_name} unusual or critical months 2003-2025 ({len(flagged)}):\n"
                  + (pd.DataFrame(flagged).to_string(index=False) if flagged else "none"), flush=True)
    return report
