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

PRE-REGISTERED RULE (settings resolution.rule, fixed 15 Oct 2026 before the first run): the
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
from .region import in_bbox, test_regions
from .selection import baseline, score
from .static_mask import masked_cells

VARIANTS = ("coarse", "fine", "fine_shrunk")
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


def region_steps(cube, segments, ref, years, variant, prior, min_v, regions, before, after) -> dict:
    """Harmonised 2012 step per region with this variant's factors (fitted on all overlap years
    of the MODIS piece) against the step MODIS alone shows."""
    g, y0, y1 = segments[0]
    span = [y for y in years if int(y0) <= y <= int(y1)]
    wts = np.array([1.0 if y in span else 0.0 for y in cube.years])
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
    mask = final_mask(settings, masked_cells(settings), parent)
    print(f"[resolution] reading the record once at {fine:g} degrees; this prints nothing for several minutes", flush=True)
    m = monthly_world(settings, groups, fine, bad, None, mask, drop_zone_days=data_loss_keys(settings))
    if m.empty:
        return {"stage": "resolution", "note": "no cell-days found"}
    all_years = list(range(int(m["year"].min()), int(m["year"].max()) + 1))
    cubes = {v: Cube(m, v, True, False, parent, all_years, fine) for v in ("cells", "frp")}
    for c in cubes.values():
        c.tile_deg = fine
    keys = cubes["cells"].keys
    r = m[(m["group"] == ref) & m["year"].isin(years) & ~m["in_mask"]]
    from .tiles import tile_name
    r = r.assign(cell=[tile_name(a, b, fine) for a, b in zip(r["lat0"], r["lon0"])])
    per_cell = r.groupby("cell")["active_cells"].sum() / (12 * len(years))
    cells = per_cell[per_cell >= min_ref].index
    weights = r.groupby("cell")["active_cells"].sum()
    regions = {"global": {"bbox": [-180, -90, 180, 90]}, **test_regions(settings)}

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

    out = settings.processed_dir / "counts"
    out.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out / "resolution_variants.csv", index=False)
    step_tbl = pd.DataFrame(steps).rename_axis("region")
    step_tbl.to_csv(out / "resolution_region_steps.csv")

    measured = None
    if adopt:
        borrow = chosen == "fine_shrunk"
        rows = []
        for var in ("cells", "frp"):
            c = cubes[var]
            bs = bootstrap(c, segments, ref, years, False, borrow, prior[var], min_v[var],
                           int(fcfg.get("bootstrap_samples", 200)), int(fcfg.get("seed", 26)))
            for seg, (k, lo, hi, span_used, level) in bs.items():
                for i, key in c.keys.iterrows():
                    for mth in range(12):
                        rows.append({"segment": seg, "fit_years": f"{span_used[0]}-{span_used[-1]}", "tile": key["tile"],
                                     "lat": key["lat0"] + fine / 2, "lon": key["lon0"] + fine / 2, "size": "all",
                                     "month": mth + 1, "variable": var, "k": k[i, mth], "k_lo": lo[i, mth],
                                     "k_hi": hi[i, mth], "source": LEVEL_NAMES[int(level[i, mth])]})
        fac = pd.DataFrame(rows)
        fac.to_parquet(out / "factors_world_fine.parquet", index=False)
        # honest uncertainty for borrowed levels: a fine cell using its 10-degree parent is as good
        # as the coarse variant measured at fine scale; the world level is never better than that
        hw = res["coarse"]["frp"]["half_width_95_log"]
        measured = {"zone": hw, "world": hw}
    import json
    (out / "resolution_choice.json").write_text(json.dumps(
        {"adopted": adopt, "chosen": chosen, "fine_deg": fine, "measured_half_width_95_log": measured}))
    if not adopt and (out / "factors_world_fine.parquet").exists():
        (out / "factors_world_fine.parquet").unlink()          # never leave a stale fine table behind
    report = {"stage": "resolution", "fine_deg": fine, "parent_deg": parent, "cells_tested": int(len(cells)),
              "variants": tbl.to_dict("records"), "region_step_gap_pct": steps,
              "adopted": adopt, "chosen": chosen, "why": why, "measured_half_width_95_log": measured}
    with pd.option_context("display.width", 230, "display.max_columns", 20, "display.max_rows", 40):
        print(f"[resolution] {len(cells)} fine cells of {fine:g} degrees tested", flush=True)
        print("[resolution] variants, judged at fine scale (pre-registered rule):\n" + tbl.to_string(index=False), flush=True)
        print("[resolution] 2012 step gap % per region (harmonised vs MODIS alone, burning cells):\n"
              + step_tbl.to_string(), flush=True)
        print(f"[resolution] RULE: {'ADOPTED ' + chosen if adopt else 'coarse factors stay'} | {why}", flush=True)
    return report
