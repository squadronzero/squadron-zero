"""D21 for every place on Earth: production factors, gap fixes and refinements.

D21 (decided 14 Oct): MODIS values are put on the S-NPP scale with a ratio per place and
calendar month, k = S-NPP total / MODIS total, on the D19 source pieces (merged MODIS to
2019, Aqua from 2020). The worldwide check confirmed it, and showed four gaps:

  1. Non-fire heat in older years. NASA labelled about 14x more gas flares and industry from
     2018, so older MODIS still holds flares later data drops; the worst tiles are oil and gas
     regions. Fix: optionally remove the 67,810 known non-fire cells in EVERY year.
  2. Thin places. 237 tiles have too little fire to fit their own factor. Fix: a fallback
     chain (tile month -> tile all year -> 30 degree zone month -> world month) for all
     variants, and, as a refinement, blending every tile towards its zone ("borrowing").
  3. Small fires behave differently (selection test 6). Refinement: separate ratios for
     small-fire and large-fire cells.
  4. One odd month can swing a factor. Refinement: blend each month with its neighbours.
  Plus: the Aqua piece is fitted on all its years (2020-2025), not 2020-2021 only.

Every option is tested on the same 10-degree tiles with leave-one-year-out and backward tests,
and kept or dropped by the pre-registered rules in settings `factors.rule`, in this order:
mask, size split, borrowing, smoothing. The chosen variant is then fitted on all overlap years
and written as a factor table for every tile with fire, for burning cells AND fire power (so
counts and fire power stay consistent), with a bootstrap range from resampling whole years.

    python -m shift.run --dataset global --stage factors      (after counts; replaces nothing)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings
from .counts import GROUPS, source_segments
from .coverage import bad_days
from .selection import baseline, drift, score, size_classes
from .static_mask import masked_cells
from .tiles import tile_name

COLS = ["acq_date", "cell_id", "cell_lat", "cell_lon", "satellite", "frp_sum_mw"]
VARS = {"cells": "active_cells", "frp": "frp_mw"}


# ---------------------------------------------------------------- one pass over the record

def monthly_world(settings: Settings, groups: dict, size_deg: float, exclude, labels: pd.Series | None,
                  mask: np.ndarray | None) -> pd.DataFrame:
    """Burning cells and fire power per tile, fire-size class, in-mask flag, group, year, month."""
    root = settings.celldays_dir
    wanted = {s for v in groups.values() for s in v}
    ex = pd.DatetimeIndex(pd.to_datetime(pd.Index(exclude))).normalize() if exclude is not None and len(exclude) else None
    mask = np.asarray(mask if mask is not None else [], dtype="int64")
    parts = []
    for part in sorted(root.glob("sensor_family=*/year=*/month=*")) if root.exists() else []:
        d = pd.read_parquet(part, columns=COLS)
        d["satellite"] = d["satellite"].astype(str)
        d = d[d["satellite"].isin(wanted)]
        if d.empty:
            continue
        d["acq_date"] = pd.to_datetime(d["acq_date"]).dt.normalize()
        if ex is not None:
            d = d[~d["acq_date"].isin(ex)]
        for group, sats in groups.items():
            g = d[d["satellite"].isin(sats)]
            if g.empty:
                continue
            if len(sats) > 1:
                g = (g.groupby(["acq_date", "cell_id"], as_index=False)
                      .agg(cell_lat=("cell_lat", "first"), cell_lon=("cell_lon", "first"), frp_sum_mw=("frp_sum_mw", "sum")))
            cid = g["cell_id"].to_numpy("int64")
            t = pd.DataFrame({
                "lat0": np.floor(g["cell_lat"].to_numpy() / size_deg) * size_deg,
                "lon0": np.floor(g["cell_lon"].to_numpy() / size_deg) * size_deg,
                "size": (g["cell_id"].map(labels).fillna("unlabelled").to_numpy() if labels is not None else "all"),
                "in_mask": np.isin(cid, mask) if len(mask) else False,
                "year": g["acq_date"].dt.year.to_numpy(), "month": g["acq_date"].dt.month.to_numpy(),
                "frp": g["frp_sum_mw"].to_numpy()})
            s = t.groupby(["lat0", "lon0", "size", "in_mask", "year", "month"], as_index=False).agg(
                active_cells=("frp", "size"), frp_mw=("frp", "sum"))
            s["group"] = group
            parts.append(s)
    if not parts:
        return pd.DataFrame()
    m = pd.concat(parts, ignore_index=True)
    return m.groupby(["lat0", "lon0", "size", "in_mask", "group", "year", "month"], as_index=False)[["active_cells", "frp_mw"]].sum()


# ---------------------------------------------------------------- the factor engine (dense arrays)

class Cube:
    """Dense arrays value[key, month, year] per group for one variant (mask on/off, size split
    on/off). Keys are tiles, or tile x size class. Zones and the world are sums of keys."""

    def __init__(self, m: pd.DataFrame, var: str, use_mask: bool, size_split: bool, zone_deg: float,
                 years: list[int], tile_deg: float):
        x = m[~m["in_mask"]] if use_mask else m
        x = x if size_split else x.assign(size="all")
        x = x[x["size"] != "unlabelled"] if size_split else x
        x = x.groupby(["lat0", "lon0", "size", "group", "year", "month"], as_index=False)[VARS[var]].sum()
        keys = x[["lat0", "lon0", "size"]].drop_duplicates().sort_values(["lat0", "lon0", "size"]).reset_index(drop=True)
        self.keys = keys.assign(tile=[tile_name(a, b, tile_deg) for a, b in zip(keys["lat0"], keys["lon0"])])
        kidx = {tuple(r): i for i, r in enumerate(keys.itertuples(index=False, name=None))}
        self.years = list(years)
        yidx = {y: i for i, y in enumerate(self.years)}
        x = x[x["year"].isin(yidx)]
        self.A = {}
        for g, d in x.groupby("group"):
            a = np.zeros((len(keys), 12, len(self.years)))
            ki = [kidx[t] for t in zip(d["lat0"], d["lon0"], d["size"])]
            a[ki, d["month"].to_numpy() - 1, [yidx[y] for y in d["year"]]] = d[VARS[var]].to_numpy(float)
            self.A[g] = a
        # a group "exists" in a month-year if it saw anything anywhere that month
        self.avail = {g: a.sum(axis=0) > 0 for g, a in self.A.items()}
        zl = np.floor(keys["lat0"].to_numpy() / zone_deg) * zone_deg
        zo = np.floor(keys["lon0"].to_numpy() / zone_deg) * zone_deg
        zkey = pd.Series(list(zip(zl, zo, keys["size"])))
        zid, self.zones = pd.factorize(zkey)
        self.Z = np.zeros((len(self.zones), len(keys)))
        self.Z[zid, np.arange(len(keys))] = 1.0
        gid, self.globals_ = pd.factorize(keys["size"])
        self.G = np.zeros((len(self.globals_), len(keys)))
        self.G[gid, np.arange(len(keys))] = 1.0
        # which world row (size class) each zone belongs to
        self.zone_gid = np.array([list(self.globals_).index(z[2]) for z in self.zones])

    def fit(self, src: str, ref: str, wts: np.ndarray, smooth: bool, borrow: bool, prior: float, min_v: float) -> np.ndarray:
        """k[key, month] from year weights `wts` (0 = year not used)."""
        W = wts[None, :] * (self.avail[src] & self.avail[ref])          # month x year
        S = np.einsum("kmy,my->km", self.A[src], W)
        R = np.einsum("kmy,my->km", self.A[ref], W)
        if smooth:
            S = 0.25 * np.roll(S, 1, axis=1) + 0.5 * S + 0.25 * np.roll(S, -1, axis=1)
            R = 0.25 * np.roll(R, 1, axis=1) + 0.5 * R + 0.25 * np.roll(R, -1, axis=1)
        Sz, Rz = self.Z @ S, self.Z @ R
        Sg, Rg = self.G @ S, self.G @ R
        with np.errstate(divide="ignore", invalid="ignore"):
            kg = np.where(Sg >= min_v, Rg / Sg, (Rg.sum(1) / Sg.sum(1))[:, None])        # world, per size
            kg_k = self.G.T @ np.nan_to_num(kg)                                           # back to keys
            if borrow:
                kz = (Rz + prior * np.nan_to_num(kg)[self.zone_gid]) / (Sz + prior)
                kz_k = self.Z.T @ kz
                return (R + prior * kz_k) / (S + prior)
            kz = np.where(Sz >= min_v, Rz / Sz, np.nan)
            kz_k = self.Z.T @ np.nan_to_num(kz, nan=0.0)
            kz_ok = self.Z.T @ (Sz >= min_v)
            k_all = np.where(S.sum(1) >= min_v, R.sum(1) / S.sum(1), np.nan)[:, None]
            k = np.where(S >= min_v, R / S, k_all)
            k = np.where(np.isnan(k), np.where(kz_ok > 0, kz_k, kg_k), k)
            return k

    def predict(self, segments, ref, fit_years, test_years, smooth, borrow, prior, min_v) -> pd.DataFrame:
        """Predicted and real reference per key, year, month for `test_years`."""
        out = []
        for g, y0, y1 in segments:
            if g not in self.A or ref not in self.A:
                continue
            span = [y for y in self.years if int(y0) <= y <= int(y1)]
            fy = [y for y in fit_years if y in span] or list(fit_years)
            ty = [y for y in test_years if y in span]
            if not ty:
                continue
            wts = np.array([1.0 if y in fy else 0.0 for y in self.years])
            k = self.fit(g, ref, wts, smooth, borrow, prior, min_v)
            for y in ty:
                j = self.years.index(y)
                ok = self.avail[g][:, j] & self.avail[ref][:, j]
                pred = self.A[g][:, :, j] * k
                real = self.A[ref][:, :, j]
                for mth in np.where(ok)[0]:
                    out.append(pd.DataFrame({"key": np.arange(len(self.keys)), "year": y, "month": mth + 1,
                                             "pred": pred[:, mth], "real": real[:, mth]}))
        if not out:
            return pd.DataFrame(columns=["region", "year", "month", "pred", "real"])
        p = pd.concat(out, ignore_index=True)
        p["region"] = self.keys["tile"].to_numpy()[p["key"].to_numpy()]
        return p.groupby(["region", "year", "month"], as_index=False)[["pred", "real"]].sum()


# ---------------------------------------------------------------- testing a variant

def evaluate(cube: Cube, segments, ref, years, bw_fit, bw_test, smooth, borrow, prior, min_v,
             tiles: pd.Index, weights: pd.Series, limit: float) -> dict:
    lo = pd.concat([cube.predict(segments, ref, [x for x in years if x != y], [y], smooth, borrow, prior, min_v)
                    for y in years], ignore_index=True)
    bw = cube.predict(segments, ref, bw_fit, bw_test, smooth, borrow, prior, min_v)
    lo, bw = lo[lo["region"].isin(tiles)], bw[bw["region"].isin(tiles)]
    real = lo[["region", "year", "month", "real"]].rename(columns={"real": ref})
    base = baseline(real, ref, years, [])
    a = score(lo, base, []).set_index("region")
    b = score(bw, base, []).set_index("region")
    dr = drift(lo, []).set_index("region")
    t = pd.DataFrame({"loyo": a["agreement_pct"], "backward": b["agreement_pct"],
                      "backward_error_pct": b["total_error_pct"], "drift": dr["drift_pct_per_year"]})
    t["score"] = t[["loyo", "backward"]].mean(axis=1)
    w = weights.reindex(t.index).fillna(0)
    ok = t["score"].notna() & (w > 0)
    return {"fire_weighted_score": round(float(np.average(t["score"][ok], weights=w[ok])), 2) if ok.any() else None,
            "median_score": round(float(t["score"].median()), 2),
            "tiles_within_limit_pct": round(float(100 * (t["backward_error_pct"].abs() <= limit).mean()), 1),
            "fire_weighted_abs_drift": round(float(np.average(t["drift"].abs().fillna(0)[ok], weights=w[ok])), 3) if ok.any() else None,
            "tiles": int(ok.sum()), "per_tile": t}


def bootstrap(cube: Cube, segments, ref, years, smooth, borrow, prior, min_v, n: int, seed: int) -> dict:
    """Final factors fitted on all overlap years, plus a 95% range from resampling whole years."""
    rng = np.random.default_rng(seed)
    res = {}
    for g, y0, y1 in segments:
        if g not in cube.A:
            continue
        span = [y for y in years if int(y0) <= y <= int(y1) and y in cube.years]
        if not span:
            continue
        idx = [cube.years.index(y) for y in span]
        base_w = np.zeros(len(cube.years))
        base_w[idx] = 1.0
        k = cube.fit(g, ref, base_w, smooth, borrow, prior, min_v)
        draws = []
        for _ in range(n):
            w = np.zeros(len(cube.years))
            w[idx] = rng.multinomial(len(idx), [1 / len(idx)] * len(idx))
            draws.append(cube.fit(g, ref, w, smooth, borrow, prior, min_v))
        draws = np.stack(draws)
        with np.errstate(all="ignore"):
            lo, hi = np.nanpercentile(draws, 2.5, axis=0), np.nanpercentile(draws, 97.5, axis=0)
        res[f"{g} {y0}-{y1}"] = (k, lo, hi, span)
    return res


# ---------------------------------------------------------------- the stage

def run_factors(settings: Settings) -> dict:
    cfg = settings.raw.get("factors") or {}
    ref = cfg.get("reference", "S-NPP")
    tile_deg, zone_deg = float(cfg.get("tile_deg", 10)), float(cfg.get("zone_deg", 30))
    min_ref = float(cfg.get("min_ref_cells_per_month", 50))
    prior = {"cells": float(cfg.get("prior_cells", 30)), "frp": float(cfg.get("prior_frp_mw", 300))}
    min_v = {"cells": float(cfg.get("min_cells_per_month", 30)), "frp": float(cfg.get("min_frp_mw_per_month", 300))}
    rule = cfg.get("rule") or {}
    gain, mask_drop, limit = float(rule.get("refinement_gain_points", 2)), float(rule.get("mask_max_drop_points", 1)), float(rule.get("error_limit_pct", 10))
    span = lambda key, d: list(range(int(cfg.get(key, d)[0]), int(cfg.get(key, d)[1]) + 1))  # noqa: E731
    years = span("overlap_years", [2012, 2025])
    bw_fit, bw_test = span("backward_fit_years", [2016, 2025]), span("backward_test_years", [2012, 2015])
    segments = source_segments(settings.raw.get("counts") or {})
    groups = {g: GROUPS[g] for g in sorted({g for g, *_ in segments} | {ref})}
    sats = sorted({s for g in groups for s in GROUPS[g]})
    bad = bad_days(settings, sats) if cfg.get("exclude_gap_days", True) else None

    labels, threshold = size_classes(settings, ref, years)
    mask = masked_cells(settings)
    m = monthly_world(settings, groups, tile_deg, bad, labels if len(labels) else None, mask)
    if m.empty:
        return {"stage": "factors", "note": "no cell-days found"}
    out = settings.processed_dir / "counts"
    out.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out / "factors_world_monthly.parquet", index=False)

    tn = pd.Series([tile_name(a, b, tile_deg) for a, b in zip(m["lat0"], m["lon0"])], index=m.index)
    r = m[(m["group"] == ref) & m["year"].isin(years)].assign(tile=tn)
    per_tile = r.groupby("tile")["active_cells"].sum() / (12 * len(years))
    tiles = per_tile[per_tile >= min_ref].index
    weights = r.groupby("tile")["active_cells"].sum()

    cubes = {}

    def cube(var, use_mask, size_split):
        key = (var, use_mask, size_split)
        if key not in cubes:
            cubes[key] = Cube(m, var, use_mask, size_split, zone_deg, list(range(min(m["year"]), max(m["year"]) + 1)), tile_deg)
        return cubes[key]

    def ev(v, var="frp"):
        return evaluate(cube(var, v["mask"], v["size"]), segments, ref, years, bw_fit, bw_test, v["smooth"], v["borrow"],
                        prior[var], min_v[var], tiles, weights, limit)

    # The pre-registered sequence, judged on fire power (D21)
    steps = []
    cur = {"mask": False, "size": False, "borrow": False, "smooth": False}
    cur_res = ev(cur)
    steps.append({"step": "plain D21", **{k: v for k, v in cur_res.items() if k != "per_tile"}, "kept": True})
    if mask is not None and len(mask):
        cand = {**cur, "mask": True}
        res = ev(cand)
        keep = (res["fire_weighted_score"] >= cur_res["fire_weighted_score"] - mask_drop
                and res["tiles_within_limit_pct"] > cur_res["tiles_within_limit_pct"])
        steps.append({"step": "+ non-fire mask in every year", **{k: v for k, v in res.items() if k != "per_tile"}, "kept": bool(keep)})
        if keep:
            cur, cur_res = cand, res
    for name, flag in (("+ fire-size split", "size"), ("+ borrowing from zone", "borrow"), ("+ seasonal smoothing", "smooth")):
        if flag == "size" and not len(labels):
            continue
        cand = {**cur, flag: True}
        res = ev(cand)
        keep = res["fire_weighted_score"] > cur_res["fire_weighted_score"] + gain
        steps.append({"step": name, **{k: v for k, v in res.items() if k != "per_tile"}, "kept": bool(keep)})
        if keep:
            cur, cur_res = cand, res
    steps = pd.DataFrame(steps)
    cells_res = ev(cur, "cells")

    # Production factors: chosen variant, all overlap years, both variables, bootstrap range
    rows = []
    for var in ("cells", "frp"):
        c = cube(var, cur["mask"], cur["size"])
        bs = bootstrap(c, segments, ref, years, cur["smooth"], cur["borrow"], prior[var], min_v[var],
                       int(cfg.get("bootstrap_samples", 200)), int(cfg.get("seed", 26)))
        for seg, (k, lo, hi, span_used) in bs.items():
            for i, key in c.keys.iterrows():
                for mth in range(12):
                    rows.append({"segment": seg, "fit_years": f"{span_used[0]}-{span_used[-1]}", "tile": key["tile"],
                                 "lat": key["lat0"] + tile_deg / 2, "lon": key["lon0"] + tile_deg / 2, "size": key["size"],
                                 "month": mth + 1, "variable": var, "k": k[i, mth], "k_lo": lo[i, mth], "k_hi": hi[i, mth]})
    fac = pd.DataFrame(rows)
    fac["range_pct"] = (100 * (fac["k_hi"] - fac["k_lo"]) / fac["k"]).round(1)
    fac["tested_tile"] = fac["tile"].isin(tiles)
    fac.to_parquet(out / "factors_world.parquet", index=False)
    steps.to_csv(out / "factors_variant_steps.csv", index=False)
    cur_res["per_tile"].assign(fire_cells_per_month=per_tile.round(1)).to_csv(out / "factors_tile_scores_frp.csv")
    cells_res["per_tile"].to_csv(out / "factors_tile_scores_cells.csv")

    # flare regions before and after (names of tiles from the worldwide check)
    watch = list(cfg.get("watch_tiles", []))
    plain = ev({"mask": False, "size": False, "borrow": False, "smooth": False})["per_tile"]
    watch_tbl = pd.DataFrame({"plain_backward_error_pct": plain["backward_error_pct"].reindex(watch),
                              "final_backward_error_pct": cur_res["per_tile"]["backward_error_pct"].reindex(watch)}).round(2)

    frp_tiles = fac[(fac["variable"] == "frp")]
    report = {"stage": "factors", "chosen": cur, "fire_size_threshold_mw": round(threshold, 2) if threshold == threshold else None,
              "tiles_with_fire": int(fac["tile"].nunique()), "tiles_tested": int(len(tiles)),
              "masked_cells": int(len(mask)) if mask is not None else 0,
              "final_frp": {k: v for k, v in cur_res.items() if k != "per_tile"},
              "final_cells": {k: v for k, v in cells_res.items() if k != "per_tile"},
              "steps": steps.to_dict("records"),
              "factor_rows": int(len(fac)),
              "median_range_pct_frp_tested": float(frp_tiles[frp_tiles["tested_tile"]]["range_pct"].median()),
              "median_range_pct_frp_untested": float(frp_tiles[~frp_tiles["tested_tile"]]["range_pct"].median()) if (~frp_tiles["tested_tile"]).any() else None}
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 60):
        print(f"[factors] {report['tiles_with_fire']} tiles with fire, {len(tiles)} tested; {report['masked_cells']} non-fire cells in the mask", flush=True)
        print("[factors] pre-registered sequence on fire power (each step kept only if the rule allows):\n"
              + steps.to_string(index=False), flush=True)
        print(f"[factors] CHOSEN: {cur}", flush=True)
        print(f"[factors] burning cells with the same choice: fire-weighted score {cells_res['fire_weighted_score']}, "
              f"tiles within +-{limit:g}%: {cells_res['tiles_within_limit_pct']}%", flush=True)
        if len(watch):
            print("[factors] flare and weak tiles from the worldwide check, backward fire power error % (plain -> final):\n"
                  + watch_tbl.to_string(), flush=True)
        print(f"[factors] factor table: {len(fac)} rows (segment x tile x size x month x variable); median 95% range of the "
              f"fire power factor: tested tiles {report['median_range_pct_frp_tested']}%, thin tiles {report['median_range_pct_frp_untested']}%", flush=True)
    return report
