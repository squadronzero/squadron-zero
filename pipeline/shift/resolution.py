"""Phase 2, fix 9b: how fine must a factor be? (stage `resolution`)

Fix 9 passed its global gate (the world shows no 2012 jump), but regions inside a 10-degree
tile can still jump: Java +83%, Florida -26%, Bangladesh -15% (burning cells, harmonised step
vs MODIS alone). A 10-degree tile mixes different fires (Java sits in a tile dominated by
Sumatra's peat fires, Florida in one with Yucatan farm burning), so the tile's factor is right
for the tile and wrong for parts of it.

Three factor resolutions are compared, all judged at the SAME fine scale (2.5-degree cells),
because a calendar for a small area needs the factor of that area:

  coarse        the 10-degree tile factor (current, D21), applied to every fine cell in it
  fine          each 2.5-degree cell's own factor, with the fallback chain
                (own month -> own all-year -> its 10-degree tile -> world)
  fine_shrunk   each fine cell's factor pulled towards its 10-degree tile in proportion to how
                little fire it has (prior_cells / prior_frp_mw): thin cells take the tile
                value, busy cells their own

Tests (same as stage factors, on fine cells with at least min_ref_cells_per_month S-NPP
burning cells a month): leave-one-year-out and backward (fit 2016-2025, predict 2012-2015),
fire-weighted category agreement for burning cells and fire power; plus the 2012 step in every
D18 test region (harmonised vs MODIS alone).

RULE FIXED BEFORE THE RUN (settings resolution.rule, 15 Oct 2026, committed to git): the
best fine variant replaces the coarse factor only if (1) its score (mean of the fire-weighted
burning-cell and fire-power scores at fine scale) is higher by more than gain_points, (2) its
mean absolute 2012-step gap over the D18 regions is lower, and (3) its global backward error is
within +-error_limit_pct for both variables. If adopted, the production table (bootstrap 95%
ranges, as stage factors) is written to counts/factors_world_fine.parquet and stage harmonise
uses it automatically; the measured error of each borrowed level sets its uncertainty.

    python -m shift.run --dataset global --stage resolution   (after factors; ~15-25 min)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings
from .counts import GROUPS, source_segments
from .coverage import bad_days
from .factors import Cube, bootstrap, data_loss_keys, final_mask, monthly_world
from .tiles import tile_name
from .region import in_bbox, test_regions
from .selection import baseline, score
from .static_mask import masked_cells

VARIANTS = ("coarse", "fine", "fine_shrunk")
# extra variants tested only for regions whose 2012 step stays outside the limit (option E)
EXTRA_VARIANTS = ("fine_seasonal", "fine_shrunk_seasonal", "all_year")
LEVEL_NAMES = ["own month", "own all-year", "zone", "world"]


def _sums(cube: Cube, g: str, ref: str, wts: np.ndarray):
    W = wts[None, :] * (cube.avail[g] & cube.avail[ref])
    return np.einsum("kmy,my->km", cube.A[g], W), np.einsum("kmy,my->km", cube.A[ref], W)


def k_variant(cube: Cube, g: str, ref: str, wts: np.ndarray, variant: str, prior: float, min_v: float) -> np.ndarray:
    """k[fine cell, month] for one variant. Zones of the cube are the 10-degree parents."""
    if variant == "fine":
        return cube.fit(g, ref, wts, False, False, prior, min_v)
    if variant == "fine_shrunk":
        return cube.fit(g, ref, wts, False, True, prior, min_v)
    if variant == "fine_seasonal":                      # each month blended with its neighbours
        return cube.fit(g, ref, wts, True, False, prior, min_v)
    if variant == "fine_shrunk_seasonal":
        return cube.fit(g, ref, wts, True, True, prior, min_v)
    if variant == "all_year":                           # one factor for every month, own or parent
        S, R = _sums(cube, g, ref, wts)
        Sz, Rz = cube.Z @ S, cube.Z @ R
        with np.errstate(divide="ignore", invalid="ignore"):
            kw = R.sum() / max(S.sum(), 1e-9)
            kp = np.where(Sz.sum(1) >= min_v, Rz.sum(1) / Sz.sum(1), kw)
            kp_k = cube.Z.T @ kp
            k = np.where(S.sum(1) >= min_v, R.sum(1) / S.sum(1), kp_k)
        return np.broadcast_to(k[:, None], S.shape).copy()
    S, R = _sums(cube, g, ref, wts)                       # coarse: the parent's own chain
    Sz, Rz = cube.Z @ S, cube.Z @ R
    with np.errstate(divide="ignore", invalid="ignore"):
        kw = np.where(S.sum(0) > 0, R.sum(0) / S.sum(0), np.nan)
        kw = np.where(np.isfinite(kw), kw, R.sum() / max(S.sum(), 1e-9))
        kall = np.where(Sz.sum(1) >= min_v, Rz.sum(1) / Sz.sum(1), np.nan)[:, None]
        kz = np.where(Sz >= min_v, Rz / Sz, kall)
        kz = np.where(np.isfinite(kz), kz, kw[None, :])
    return cube.Z.T @ kz


def predict_v(cube: Cube, segments, ref, fit_years, test_years, variant, prior, min_v) -> pd.DataFrame:
    out = []
    for g, y0, y1 in segments:
        if g not in cube.A or ref not in cube.A:
            continue
        span = [y for y in cube.years if int(y0) <= y <= int(y1)]
        fy = [y for y in fit_years if y in span] or list(fit_years)
        ty = [y for y in test_years if y in span]
        if not ty:
            continue
        wts = np.array([1.0 if y in fy else 0.0 for y in cube.years])
        k = k_variant(cube, g, ref, wts, variant, prior, min_v)
        for y in ty:
            j = cube.years.index(y)
            ok = cube.avail[g][:, j] & cube.avail[ref][:, j]
            pred, real = cube.A[g][:, :, j] * k, cube.A[ref][:, :, j]
            for mth in np.where(ok)[0]:
                out.append(pd.DataFrame({"key": np.arange(len(cube.keys)), "year": y, "month": mth + 1,
                                         "pred": pred[:, mth], "real": real[:, mth]}))
    if not out:
        return pd.DataFrame(columns=["region", "year", "month", "pred", "real"])
    p = pd.concat(out, ignore_index=True)
    p["region"] = cube.keys["tile"].to_numpy()[p["key"].to_numpy()]
    return p.groupby(["region", "year", "month"], as_index=False)[["pred", "real"]].sum()


def judge(cube, segments, ref, years, bw_fit, bw_test, variant, prior, min_v, cells, weights, limit) -> dict:
    lo = pd.concat([predict_v(cube, segments, ref, [x for x in years if x != y], [y], variant, prior, min_v)
                    for y in years], ignore_index=True)
    bw = predict_v(cube, segments, ref, bw_fit, bw_test, variant, prior, min_v)
    lo, bw = lo[lo["region"].isin(cells)], bw[bw["region"].isin(cells)]
    base = baseline(lo[["region", "year", "month", "real"]].rename(columns={"real": ref}), ref, years, [])
    a, b = score(lo, base, []).set_index("region"), score(bw, base, []).set_index("region")
    t = pd.DataFrame({"loyo": a["agreement_pct"], "backward": b["agreement_pct"], "err": b["total_error_pct"]})
    t["score"] = t[["loyo", "backward"]].mean(axis=1)
    w = weights.reindex(t.index).fillna(0)
    ok = t["score"].notna() & (w > 0)
    tot = bw.groupby("region")[["pred", "real"]].sum()
    lr = np.log(tot["pred"] / tot["real"]).replace([np.inf, -np.inf], np.nan).dropna()
    errs = t["err"].abs().dropna().sort_values()
    we = w.reindex(errs.index)
    return {"score": round(float(np.average(t["score"][ok], weights=w[ok])), 2) if ok.any() else None,
            "fw_median_abs_error_pct": (round(float(errs[we.cumsum() >= we.sum() / 2].iloc[0]), 1) if we.sum() else None),
            "cells_within_limit_pct": round(float(100 * (t["err"].abs() <= limit).mean()), 1),
            "global_backward_error_pct": round(float(100 * (bw["pred"].sum() / bw["real"].sum() - 1)), 2) if bw["real"].sum() else None,
            "half_width_95_log": round(float(np.percentile(lr.abs(), 95)), 3) if len(lr) else None,
            "cells": int(ok.sum())}


def region_steps(cube, segments, ref, years, variant, prior, min_v, regions, before, after, fit_years=None) -> dict:
    """Harmonised 2012 step per region with this variant's factors (fitted on all overlap years
    of the MODIS piece, or the version test's years) against the step MODIS alone shows."""
    g, y0, y1 = segments[0]
    wts = fit_weights(cube, g, y0, y1, years, fit_years)
    k = k_variant(cube, g, ref, wts, variant, prior, min_v)
    yi = [cube.years.index(y) for y in before if y in cube.years]
    ai = [cube.years.index(y) for y in after if y in cube.years]
    if len(yi) != len(before) or len(ai) != len(after) or g not in cube.A:
        return {}
    lat = cube.keys["lat0"].to_numpy() + cube.tile_deg / 2
    lon = cube.keys["lon0"].to_numpy() + cube.tile_deg / 2
    mod_b = cube.A[g][:, :, yi].sum(2)
    harm_b = (mod_b * k).sum(1)
    snpp_a = cube.A[ref][:, :, ai].sum(2).sum(1)
    mod_a = cube.A[g][:, :, ai].sum(2).sum(1)
    out = {}
    for name, reg in regions.items():
        m = in_bbox(reg["bbox"], lon, lat)
        if not m.any() or harm_b[m].sum() <= 0 or mod_b[m].sum() <= 0:
            continue
        h = snpp_a[m].sum() / harm_b[m].sum()
        mo = mod_a[m].sum() / mod_b[m].sum().sum()
        out[name] = round(float(100 * (h / mo - 1)), 1)
    return out


def key_centres(cube: Cube, deg: float) -> tuple[np.ndarray, np.ndarray]:
    return cube.keys["lat0"].to_numpy() + deg / 2, cube.keys["lon0"].to_numpy() + deg / 2


def fit_weights(cube: Cube, g: str, y0: int, y1: int, years: list[int], fit_years: dict | None,
                drop: int | None = None) -> np.ndarray:
    """Year weights of the production fit of piece g: its overlap years, limited to the years the
    version test chose (fit_years[g]), without `drop` (a held-out year)."""
    allowed = (fit_years or {}).get(g)
    use = [y for y in years if int(y0) <= y <= int(y1) and (allowed is None or y in allowed) and y != drop]
    return np.array([1.0 if y in use else 0.0 for y in cube.years])


# ---------------------------------------------------------------- the production model, shared by later stages

def model_spec(settings: Settings) -> dict:
    """The final model as decided by this stage: variant, fit years per D19 piece (version test)
    and region overrides (option E). Later stages (daily, checks) build k with k_model so every
    test uses exactly the production model."""
    import json
    out = settings.processed_dir / "counts"
    spec = {"variant": "coarse", "fine_deg": float((settings.raw.get("resolution") or {}).get("fine_deg", 2.5)),
            "fit_years": {}, "overrides": [], "flags": []}
    rp = out / "resolution_choice.json"
    if rp.exists():
        r = json.loads(rp.read_text())
        spec["variant"] = r.get("chosen", "coarse") if r.get("adopted") else "coarse"
        spec["fit_years"] = {g: list(range(int(a), int(b) + 1)) for g, (a, b) in (r.get("fit_years") or {}).items()}
        spec["overrides"] = r.get("region_overrides") or []
        spec["flags"] = r.get("region_flags") or []
    return spec


def k_model(cube: Cube, g: str, ref: str, wts: np.ndarray, spec: dict, prior: float, min_v: float) -> np.ndarray:
    k = k_variant(cube, g, ref, wts, spec["variant"], prior, min_v)
    if spec.get("overrides"):
        lat, lon = key_centres(cube, spec["fine_deg"])
        for o in spec["overrides"]:
            m = in_bbox(o["bbox"], lon, lat)
            if m.any():
                k[m] = k_variant(cube, g, ref, wts, o["variant"], prior, min_v)[m]
    return k


# ---------------------------------------------------------------- must-fix 3: the 2018 MODIS processing change

def version_test(cubes: dict, segments, ref, years, variant, prior, min_v, cells, weights, vcfg) -> dict:
    """Same-version fit years vs the whole piece, each test year predicted leave-one-year-out
    (see settings resolution.version_test)."""
    seg = str(vcfg.get("segment", "MODIS"))
    piece = [s for s in segments if s[0] == seg]
    if not piece:
        return {"note": f"no piece {seg}"}
    g, y0, y1 = piece[0]
    whole = [y for y in years if int(y0) <= y <= int(y1)]
    sv = vcfg.get("same_version_years", [2012, 2017])
    same = [y for y in whole if int(sv[0]) <= y <= int(sv[1])]
    if len(same) < 3 or same == whole:
        return {"segment": seg, "note": "same-version years equal the whole piece or fewer than 3: nothing to test",
                "chosen": "whole piece", "fit_years": [whole[0], whole[-1]] if whole else None}
    res = {}
    for name, pool in (("same_version", same), ("whole_piece", whole)):
        row = {}
        for var in ("cells", "frp"):
            c = cubes[var]
            p = pd.concat([predict_v(c, [piece[0]], ref, [x for x in pool if x != y], [y], variant, prior[var], min_v[var])
                           for y in same], ignore_index=True)
            p = p[p["region"].isin(cells)]
            base = baseline(p[["region", "year", "month", "real"]].rename(columns={"real": ref}), ref, same, [])
            t = score(p, base, []).set_index("region")
            w = weights.reindex(t.index).fillna(0)
            ok = t["agreement_pct"].notna() & (w > 0)
            row[var] = {"score": round(float(np.average(t["agreement_pct"][ok], weights=w[ok])), 2) if ok.any() else None,
                        "pooled_error_pct": round(float(100 * (p["pred"].sum() / p["real"].sum() - 1)), 2) if p["real"].sum() else None}
        row["score"] = round(float(np.mean([row["cells"]["score"], row["frp"]["score"]])), 2)
        res[name] = row
    rule = vcfg.get("rule") or {}
    gain, egain, sdrop = (float(rule.get("gain_points", 1)), float(rule.get("error_gain_points", 0.5)),
                          float(rule.get("max_score_drop_points", 1)))
    a, b = res["same_version"], res["whole_piece"]
    closer = all(abs(b[v]["pooled_error_pct"]) - abs(a[v]["pooled_error_pct"]) > egain for v in ("cells", "frp"))
    adopt = bool(a["score"] > b["score"] + gain or (closer and a["score"] >= b["score"] - sdrop))
    # the size of the step itself: MODIS per S-NPP burning cell, by year
    ratio = {}
    for y in whole:
        j = cubes["cells"].years.index(y)
        sr = cubes["cells"].A[ref][:, :, j].sum()
        if sr > 0 and g in cubes["cells"].A:
            ratio[int(y)] = round(float(cubes["cells"].A[g][:, :, j].sum() / sr), 4)
    return {"segment": seg, "same_version_years": [same[0], same[-1]], "whole_piece_years": [whole[0], whole[-1]],
            "same_version": a, "whole_piece": b, "modis_per_snpp_cell_by_year": ratio,
            "chosen": "same version" if adopt else "whole piece",
            "fit_years": [same[0], same[-1]] if adopt else [whole[0], whole[-1]],
            "why": (f"score {a['score']} vs {b['score']} (switch if higher by more than {gain:g}); pooled error cells "
                    f"{a['cells']['pooled_error_pct']}% vs {b['cells']['pooled_error_pct']}%, fire power "
                    f"{a['frp']['pooled_error_pct']}% vs {b['frp']['pooled_error_pct']}% (or switch if both closer to "
                    f"zero by more than {egain:g} points with the score at most {sdrop:g} lower)")}


# ---------------------------------------------------------------- option E: regions still jumping at 2012

def region_test(cubes: dict, segments, ref, years, bw_fit, bw_test, fit_years, current, prior, min_v,
                regions, before, after, deg, ecfg) -> tuple[list, list, list]:
    """For each test region whose 2012 step gap (current variant, production fit years) is outside
    +-step_limit_pct: other variants judged on the region's own step gap, backward error and
    leave-one-year-out month error. A variant replaces the current one IN THAT REGION only if it
    brings the step gap inside the limit AND is no worse on the other two tests; otherwise the
    region is flagged and its measured step gap is added to its uncertainty."""
    limit = float(ecfg.get("step_limit_pct", 10))
    cands = [v for v in ecfg.get("variants", list(VARIANTS) + list(EXTRA_VARIANTS)) if v != current]
    g, y0, y1 = segments[0]
    lat, lon = key_centres(cubes["cells"], deg)
    names = cubes["cells"].keys["tile"].to_numpy()
    piece = [y for y in years if int(y0) <= y <= int(y1)]
    rows, overrides, flags = [], [], []

    def step(v, m):
        c = cubes["cells"]
        k = k_variant(c, g, ref, fit_weights(c, g, y0, y1, years, fit_years), v, prior["cells"], min_v["cells"])
        yi = [c.years.index(y) for y in before if y in c.years]
        ai = [c.years.index(y) for y in after if y in c.years]
        hb = (c.A[g][m][:, :, yi].sum(2) * k[m]).sum()
        mb, ma = c.A[g][m][:, :, yi].sum(), c.A[g][m][:, :, ai].sum()
        sa = c.A[ref][m][:, :, ai].sum()
        return float(100 * ((sa / hb) / (ma / mb) - 1)) if min(hb, mb, ma, sa) > 0 else None

    def tests(v, m):
        sel = set(names[m])
        out = {}
        for var in ("cells", "frp"):
            c = cubes[var]
            bw = predict_v(c, [segments[0]], ref, bw_fit, bw_test, v, prior[var], min_v[var])
            bw = bw[bw["region"].isin(sel)]
            out[f"backward_error_{var}_pct"] = (round(float(100 * (bw["pred"].sum() / bw["real"].sum() - 1)), 2)
                                                if bw["real"].sum() else None)
            lo = pd.concat([predict_v(c, [segments[0]], ref, [x for x in piece if x != y], [y], v, prior[var], min_v[var])
                            for y in piece], ignore_index=True)
            lo = lo[lo["region"].isin(sel)].groupby(["year", "month"])[["pred", "real"]].sum()
            lo = lo[lo["real"] > 0]
            out[f"loyo_month_error_{var}_pct"] = (round(float(100 * np.median(np.abs(lo["pred"] / lo["real"] - 1))), 1)
                                                  if len(lo) else None)
        return out

    for name, reg in regions.items():
        if name == "global":
            continue
        m = in_bbox(reg["bbox"], lon, lat)
        if not m.any():
            continue
        cur_gap = step(current, m)
        if cur_gap is None or abs(cur_gap) <= limit:
            rows.append({"region": name, "variant": current, "step_gap_pct": None if cur_gap is None else round(cur_gap, 1),
                         "tested": False})
            continue
        cur = {"region": name, "variant": current, "step_gap_pct": round(cur_gap, 1), "tested": True, **tests(current, m)}
        rows.append(cur)
        best = None
        for v in cands:
            r = {"region": name, "variant": v, "step_gap_pct": None, "tested": True}
            gap = step(v, m)
            r["step_gap_pct"] = None if gap is None else round(gap, 1)
            r.update(tests(v, m))
            rows.append(r)
            if gap is None or abs(gap) > limit:
                continue
            bw_ok = all(r[f"backward_error_{x}_pct"] is not None and cur[f"backward_error_{x}_pct"] is not None
                        and abs(r[f"backward_error_{x}_pct"]) <= abs(cur[f"backward_error_{x}_pct"]) for x in ("cells", "frp"))
            lo_ok = all(r[f"loyo_month_error_{x}_pct"] is not None and cur[f"loyo_month_error_{x}_pct"] is not None
                        and r[f"loyo_month_error_{x}_pct"] <= cur[f"loyo_month_error_{x}_pct"] for x in ("cells", "frp"))
            if bw_ok and lo_ok and (best is None or abs(gap) < abs(best[1])):
                best = (v, gap)
        if best:
            overrides.append({"region": name, "bbox": reg["bbox"], "variant": best[0], "step_gap_pct": round(best[1], 1)})
        else:
            flags.append({"region": name, "bbox": reg["bbox"], "step_gap_pct": round(cur_gap, 1),
                          "extra_hw95_log": round(abs(float(np.log1p(cur_gap / 100))), 4)})
    return rows, overrides, flags


def variant_bootstrap(cube: Cube, g, ref, base_w: np.ndarray, variant, prior, min_v, n, seed, floor_hw: float):
    """k and a 95% range for an override variant: resampled whole years, at least floor_hw (log)."""
    rng = np.random.default_rng(seed)
    idx = np.nonzero(base_w)[0]
    k = k_variant(cube, g, ref, base_w, variant, prior, min_v)
    draws = []
    for _ in range(n):
        w = np.zeros(len(base_w))
        w[idx] = rng.multinomial(len(idx), [1 / len(idx)] * len(idx))
        draws.append(k_variant(cube, g, ref, w, variant, prior, min_v))
    with np.errstate(all="ignore"):
        d = np.stack(draws)
        lo, hi = np.nanpercentile(d, 2.5, axis=0), np.nanpercentile(d, 97.5, axis=0)
        hw = np.log(np.where((lo > 0) & (hi > 0), hi / lo, np.nan)) / 2
    hw = np.fmax(np.nan_to_num(hw, nan=floor_hw), floor_hw)
    return k, k * np.exp(-hw), k * np.exp(hw)


def run_resolution(settings: Settings) -> dict:
    cfg = settings.raw.get("resolution") or {}
    fcfg = settings.raw.get("factors") or {}
    rule = cfg.get("rule") or {}
    ref = fcfg.get("reference", "S-NPP")
    parent = float(fcfg.get("tile_deg", 10))
    fine = float(cfg.get("fine_deg", 2.5))
    min_ref = float(cfg.get("min_ref_cells_per_month", 10))
    prior = {"cells": float(fcfg.get("prior_cells", 30)), "frp": float(fcfg.get("prior_frp_mw", 300))}
    min_v = {"cells": float(fcfg.get("min_cells_per_month", 30)), "frp": float(fcfg.get("min_frp_mw_per_month", 300))}
    span = lambda key, d: list(range(int(fcfg.get(key, d)[0]), int(fcfg.get(key, d)[1]) + 1))  # noqa: E731
    years = span("overlap_years", [2012, 2025])
    bw_fit, bw_test = span("backward_fit_years", [2016, 2025]), span("backward_test_years", [2012, 2015])
    gate = (settings.raw.get("harmonise") or {}).get("gate") or {}
    yr = lambda k, d: list(range(int(gate.get(k, d)[0]), int(gate.get(k, d)[1]) + 1))  # noqa: E731
    before, after = yr("before_years", [2009, 2011]), yr("after_years", [2013, 2015])
    limit = float(rule.get("error_limit_pct", 10))
    segments = source_segments(settings.raw.get("counts") or {})
    groups = {g: GROUPS[g] for g in sorted({g for g, *_ in segments} | {ref})}
    bad = bad_days(settings, sorted({s for g in groups for s in GROUPS[g]})) if fcfg.get("exclude_gap_days", True) else None
    # Terra and NOAA-20 are read as well (never used by the fits here) so later stages can reuse
    # this one read of the record (counts/fine_monthly.parquet); the gap days stay those of the fit groups
    read_groups = {**groups, **{g: GROUPS[g] for g in ("Terra", "NOAA-20")}}
    mask = final_mask(settings, masked_cells(settings), parent)
    print(f"[resolution] reading the record once at {fine:g} degrees; this prints nothing for several minutes", flush=True)
    m = monthly_world(settings, read_groups, fine, bad, None, mask, drop_zone_days=data_loss_keys(settings))
    if m.empty:
        return {"stage": "resolution", "note": "no cell-days found"}
    out = settings.processed_dir / "counts"
    out.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out / "fine_monthly.parquet", index=False)
    all_years = list(range(int(m["year"].min()), int(m["year"].max()) + 1))
    cubes = {v: Cube(m, v, True, False, parent, all_years, fine) for v in ("cells", "frp")}
    for c in cubes.values():
        c.tile_deg = fine
    r = m[(m["group"] == ref) & m["year"].isin(years) & ~m["in_mask"]]
    r = r.assign(cell=[tile_name(a, b, fine) for a, b in zip(r["lat0"], r["lon0"])])
    per_cell = r.groupby("cell")["active_cells"].sum() / (12 * len(years))
    cells = per_cell[per_cell >= min_ref].index
    weights = r.groupby("cell")["active_cells"].sum()
    regions = {"global": {"bbox": [-180, -90, 180, 90]}, **test_regions(settings)}

    # ---- 1. the three resolutions, under the rule fixed before the first run (unchanged)
    res, steps = {}, {}
    for v in VARIANTS:
        res[v] = {var: judge(cubes[var], segments, ref, years, bw_fit, bw_test, v, prior[var], min_v[var],
                             cells, weights, limit) for var in ("cells", "frp")}
        steps[v] = region_steps(cubes["cells"], segments, ref, years, v, prior["cells"], min_v["cells"],
                                regions, before, after)
        print(f"[resolution] variant {v} scored", flush=True)
    rows = []
    for v in VARIANTS:
        reg = {k: x for k, x in steps[v].items() if k != "global"}
        rows.append({"variant": v, "score": round(np.mean([res[v]["cells"]["score"], res[v]["frp"]["score"]]), 2),
                     "score_cells": res[v]["cells"]["score"], "score_frp": res[v]["frp"]["score"],
                     "fw_median_abs_error_frp_pct": res[v]["frp"]["fw_median_abs_error_pct"],
                     "cells_within_limit_frp_pct": res[v]["frp"]["cells_within_limit_pct"],
                     "global_backward_error_cells_pct": res[v]["cells"]["global_backward_error_pct"],
                     "global_backward_error_frp_pct": res[v]["frp"]["global_backward_error_pct"],
                     "mean_abs_region_step_gap_pct": round(float(np.mean(np.abs(list(reg.values())))), 1) if reg else None,
                     "global_step_gap_pct": steps[v].get("global")})
    tbl = pd.DataFrame(rows)
    gain = float(rule.get("gain_points", 1))
    base = tbl.set_index("variant").loc["coarse"]
    cand = tbl[tbl["variant"] != "coarse"].sort_values("score", ascending=False).iloc[0]
    ok_err = all(x is not None and abs(x) <= limit for x in (cand["global_backward_error_cells_pct"], cand["global_backward_error_frp_pct"]))
    adopt = bool(cand["score"] > base["score"] + gain
                 and cand["mean_abs_region_step_gap_pct"] is not None
                 and base["mean_abs_region_step_gap_pct"] is not None
                 and cand["mean_abs_region_step_gap_pct"] < base["mean_abs_region_step_gap_pct"] and ok_err)
    why = (f"{cand['variant']}: score {cand['score']} vs coarse {base['score']} (needed > +{gain:g}); mean |regional "
           f"2012 step gap| {cand['mean_abs_region_step_gap_pct']}% vs {base['mean_abs_region_step_gap_pct']}% (needed lower); "
           f"global backward error {cand['global_backward_error_cells_pct']}% / {cand['global_backward_error_frp_pct']}% "
           f"(needed within +-{limit:g}%)")
    chosen = str(cand["variant"]) if adopt else "coarse"
    tbl.to_csv(out / "resolution_variants.csv", index=False)

    # ---- 2. must-fix 3: fit years of the MODIS piece (2018 processing change), rule fixed before the run
    vt = version_test(cubes, segments, ref, years, chosen, prior, min_v, cells, weights, cfg.get("version_test") or {})
    fit_years = {}
    if vt.get("fit_years") and vt.get("chosen") == "same version":
        fit_years[vt["segment"]] = list(range(vt["fit_years"][0], vt["fit_years"][1] + 1))
    print(f"[resolution] VERSION TEST: {vt.get('chosen')} | {vt.get('why', vt.get('note'))}", flush=True)

    # 2012 step per region with the production fit years
    steps_final = region_steps(cubes["cells"], segments, ref, years, chosen, prior["cells"], min_v["cells"],
                               regions, before, after, fit_years)

    # ---- 3. option E: regions still outside the step limit, rule fixed before the run
    e_rows, overrides, flags = (region_test(cubes, segments, ref, years, bw_fit, bw_test, fit_years, chosen, prior, min_v,
                                            regions, before, after, fine, cfg.get("region_test") or {})
                                if adopt else ([], [], []))
    e_tbl = pd.DataFrame(e_rows)
    e_tbl.to_csv(out / "resolution_region_test.csv", index=False)

    step_tbl = pd.DataFrame({**steps, "final": steps_final}).rename_axis("region")
    step_tbl.to_csv(out / "resolution_region_steps.csv")

    measured = None
    if adopt:
        borrow = chosen == "fine_shrunk"
        hw_coarse = res["coarse"]["frp"]["half_width_95_log"]
        rows = []
        for var in ("cells", "frp"):
            c = cubes[var]
            bs = bootstrap(c, segments, ref, years, False, borrow, prior[var], min_v[var],
                           int(fcfg.get("bootstrap_samples", 200)), int(fcfg.get("seed", 26)), fit_years)
            lat, lon = key_centres(c, fine)
            for seg, (k, lo, hi, span_used, level) in bs.items():
                k, lo, hi = k.copy(), lo.copy(), hi.copy()
                src = np.array(LEVEL_NAMES)[level.astype(int)]
                g, y0, y1 = [x for x in segments if f"{x[0]} {x[1]}-{x[2]}" == seg][0]
                for o in overrides:
                    mm = in_bbox(o["bbox"], lon, lat)
                    if mm.any():
                        ko, lo_o, hi_o = variant_bootstrap(c, g, ref, fit_weights(c, g, y0, y1, years, fit_years), o["variant"],
                                                           prior[var], min_v[var], int(fcfg.get("bootstrap_samples", 200)),
                                                           int(fcfg.get("seed", 26)), float(hw_coarse or 0.0))
                        k[mm], lo[mm], hi[mm] = ko[mm], lo_o[mm], hi_o[mm]
                        src[mm] = f"override: {o['variant']}"
                for i, key in c.keys.iterrows():
                    for mth in range(12):
                        rows.append({"segment": seg, "fit_years": f"{span_used[0]}-{span_used[-1]}", "tile": key["tile"],
                                     "lat": key["lat0"] + fine / 2, "lon": key["lon0"] + fine / 2, "size": "all",
                                     "month": mth + 1, "variable": var, "k": k[i, mth], "k_lo": lo[i, mth],
                                     "k_hi": hi[i, mth], "source": src[i, mth]})
        fac = pd.DataFrame(rows)
        fac.to_parquet(out / "factors_world_fine.parquet", index=False)
        # honest uncertainty for borrowed levels: a fine cell using its 10-degree parent is as good
        # as the coarse variant measured at fine scale; the world level is never better than that
        measured = {"zone": hw_coarse, "world": hw_coarse}
    import json
    (out / "resolution_choice.json").write_text(json.dumps(
        {"adopted": adopt, "chosen": chosen, "fine_deg": fine, "measured_half_width_95_log": measured,
         "fit_years": {g: [ys[0], ys[-1]] for g, ys in fit_years.items()},
         "region_overrides": overrides, "region_flags": flags}, default=str))
    if not adopt and (out / "factors_world_fine.parquet").exists():
        (out / "factors_world_fine.parquet").unlink()          # never leave a stale fine table behind
    report = {"stage": "resolution", "fine_deg": fine, "parent_deg": parent, "cells_tested": int(len(cells)),
              "variants": tbl.to_dict("records"), "region_step_gap_pct": steps, "region_step_gap_final_pct": steps_final,
              "adopted": adopt, "chosen": chosen, "why": why, "measured_half_width_95_log": measured,
              "version_test": vt, "region_test": {"overrides": overrides, "flags": flags,
                                                  "table": e_tbl.to_dict("records") if len(e_tbl) else []}}
    with pd.option_context("display.width", 230, "display.max_columns", 20, "display.max_rows", 60):
        print(f"[resolution] {len(cells)} fine cells of {fine:g} degrees tested", flush=True)
        print("[resolution] variants, judged at fine scale (rule fixed before the run):\n" + tbl.to_string(index=False), flush=True)
        print(f"[resolution] RULE: {'ADOPTED ' + chosen if adopt else 'coarse factors stay'} | {why}", flush=True)
        if "same_version" in vt:
            print("[resolution] version test (each same-version year predicted leave-one-year-out):\n"
                  + pd.DataFrame({k: {"score": vt[k]["score"], "cells_error_pct": vt[k]["cells"]["pooled_error_pct"],
                                      "frp_error_pct": vt[k]["frp"]["pooled_error_pct"]}
                                  for k in ("same_version", "whole_piece")}).T.to_string(), flush=True)
            print(f"[resolution] MODIS per S-NPP burning cell by year: {vt['modis_per_snpp_cell_by_year']}", flush=True)
        print("[resolution] 2012 step gap % per region (harmonised vs MODIS alone, burning cells; 'final' = production fit years):\n"
              + step_tbl.to_string(), flush=True)
        if len(e_tbl):
            print("[resolution] regions outside the step limit, other variants (rule fixed before the run):\n"
                  + e_tbl.to_string(index=False), flush=True)
        print(f"[resolution] region overrides: {overrides or 'none'}; flagged (wider range): "
              f"{[(f['region'], f['step_gap_pct']) for f in flags] or 'none'}", flush=True)
    return report
