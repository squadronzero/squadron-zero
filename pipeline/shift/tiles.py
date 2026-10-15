"""Worldwide check: do the Phase 2 choices hold everywhere, not only in the ten test regions?

The record itself always covers the whole planet. But the choices so far (D19: which MODIS
satellite calibrates the record; D21: how fire power is harmonised) were tested on ten
regions plus the world as one block. Ten boxes cannot represent every fire regime, and the
world as one block is dominated by African savanna. This stage covers the whole Earth with
a grid of tiles (10 x 10 degrees by default) and repeats the fair tests in every tile that
has enough fire:

  burning cells (D19)   each MODIS source candidate, leave-one-year-out and backward
  fire power   (D21)    ratio of all fire power  vs  the v1 Florida line on merged MODIS

Every tile gets its own factors (fitted inside that tile), so this is also the first look at
calibration for every place on Earth (fix 8). Results are summarised three ways: share of
tiles where each option is best, fire-weighted average (big fire tiles count more) and
plain median, plus the worst tiles by name of their centre.

A rule fixed before the run (settings `worldwide.rule`) says when a decision counts as confirmed.

    python -m shift.run --dataset global --stage worldwide     (after `counts`)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings
from .counts import GROUPS, source_segments
from .coverage import bad_days
from .selection import DEFAULT_CANDIDATES, baseline, drift, loyo, predict, score

COLS = ["acq_date", "cell_id", "cell_lat", "cell_lon", "satellite", "frp_sum_mw"]


def tile_name(lat0: float, lon0: float, size: float) -> str:
    """'N20E080' style label for the tile's south-west corner."""
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    return f"{ns}{abs(int(lat0)):02d}{ew}{abs(int(lon0)):03d}"


def monthly_tiles(settings: Settings, groups: dict, size: float, exclude, drop_cells=None) -> pd.DataFrame:
    """Active cells, fire power and detections per tile, satellite group, year and month,
    with the excluded (outage) days left out. One partition at a time."""
    root = settings.celldays_dir
    wanted = {s for v in groups.values() for s in v}
    ex = pd.DatetimeIndex(pd.to_datetime(pd.Index(exclude))).normalize() if exclude is not None and len(exclude) else None
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
        if drop_cells is not None and len(drop_cells):
            d = d[~d["cell_id"].isin(drop_cells)]        # D22: non-fire cells removed in every year
        for group, sats in groups.items():
            g = d[d["satellite"].isin(sats)]
            if g.empty:
                continue
            if len(sats) > 1:
                g = (g.groupby(["acq_date", "cell_id"], as_index=False)
                      .agg(cell_lat=("cell_lat", "first"), cell_lon=("cell_lon", "first"),
                           frp_sum_mw=("frp_sum_mw", "sum")))
            lat0 = np.floor(g["cell_lat"].to_numpy() / size) * size
            lon0 = np.floor(g["cell_lon"].to_numpy() / size) * size
            t = pd.DataFrame({"lat0": lat0, "lon0": lon0, "year": g["acq_date"].dt.year.to_numpy(),
                              "month": g["acq_date"].dt.month.to_numpy(), "frp": g["frp_sum_mw"].to_numpy()})
            s = t.groupby(["lat0", "lon0", "year", "month"], as_index=False).agg(active_cells=("frp", "size"), frp_mw=("frp", "sum"))
            s["group"] = group
            parts.append(s)
    if not parts:
        return pd.DataFrame(columns=["region", "lat0", "lon0", "group", "year", "month", "active_cells", "frp_mw"])
    m = pd.concat(parts, ignore_index=True)
    m = m.groupby(["lat0", "lon0", "group", "year", "month"], as_index=False)[["active_cells", "frp_mw"]].sum()
    m["region"] = [tile_name(a, b, size) for a, b in zip(m["lat0"], m["lon0"])]
    return m


def _wide(m: pd.DataFrame, value: str) -> pd.DataFrame:
    return m.pivot_table(index=["region", "year", "month"], columns="group", values=value, aggfunc="sum").reset_index()


def _tests(w, cands, ref, years, bw_fit, bw_test, min_v):
    base = baseline(w, ref, years, [])
    rows, preds = [], {}
    for name, segs in cands.items():
        lo = loyo(w, segs, ref, years, min_v, [])
        bw = predict(w, segs, ref, bw_fit, bw_test, min_v, [])
        preds[name] = lo
        a = score(lo, base, []).set_index("region") if len(lo) else pd.DataFrame()
        b = score(bw, base, []).set_index("region") if len(bw) else pd.DataFrame()
        dr = drift(lo, []).set_index("region") if len(lo) else pd.DataFrame()
        for region in sorted(set(a.index) | set(b.index)):
            rows.append({"region": region, "candidate": name,
                         "agreement_loyo_pct": a["agreement_pct"].get(region, np.nan) if len(a) else np.nan,
                         "agreement_backward_pct": b["agreement_pct"].get(region, np.nan) if len(b) else np.nan,
                         "backward_error_pct": b["total_error_pct"].get(region, np.nan) if len(b) else np.nan,
                         "drift_pct_per_year": dr["drift_pct_per_year"].get(region, np.nan) if len(dr) else np.nan})
    out = pd.DataFrame(rows)
    out["score"] = out[["agreement_loyo_pct", "agreement_backward_pct"]].mean(axis=1)
    return out


def _summary(res: pd.DataFrame, weights: pd.Series, limit: float) -> pd.DataFrame:
    best = res.loc[res.groupby("region")["score"].idxmax().dropna()]
    share_best = best["candidate"].value_counts(normalize=True) * 100
    rows = []
    for name, r in res.groupby("candidate"):
        r = r.set_index("region")
        wv = weights.reindex(r.index).fillna(0)
        ok = r["score"].notna()
        rows.append({"candidate": name, "tiles": int(ok.sum()),
                     "fire_weighted_score": round(float(np.average(r["score"][ok], weights=wv[ok])), 2) if wv[ok].sum() else None,
                     "median_score": round(float(r["score"].median()), 2),
                     "best_in_pct_of_tiles": round(float(share_best.get(name, 0.0)), 1),
                     "tiles_backward_within_limit_pct": round(float(100 * (r["backward_error_pct"].abs() <= limit).mean()), 1),
                     "fire_weighted_abs_drift": round(float(np.average(r["drift_pct_per_year"].abs().fillna(0)[ok], weights=wv[ok])), 3) if wv[ok].sum() else None})
    return pd.DataFrame(rows).sort_values("fire_weighted_score", ascending=False)


def _verdict(summary: pd.DataFrame, chosen: str, margin: float) -> tuple[str, str]:
    top = summary.iloc[0]
    me = summary.set_index("candidate").loc[chosen]
    gap = float(top["fire_weighted_score"] - me["fire_weighted_score"])
    if top["candidate"] == chosen:
        return "CONFIRMED", f"{chosen} has the highest fire-weighted score worldwide"
    if gap <= margin:
        return "CONFIRMED", f"{chosen} is within {gap:.1f} points of the best ({top['candidate']})"
    return "CHECK", f"{top['candidate']} beats {chosen} by {gap:.1f} points worldwide"


def run_worldwide(settings: Settings) -> dict:
    cfg = settings.raw.get("worldwide") or {}
    sel = settings.raw.get("selection") or {}
    frp_cfg = settings.raw.get("frp") or {}
    ref = cfg.get("reference", "S-NPP")
    size = float(cfg.get("tile_deg", 10))
    min_ref = float(cfg.get("min_ref_cells_per_month", 50))
    rule = cfg.get("rule") or {}
    margin = float(rule.get("confirm_margin_points", 2))
    limit = float(rule.get("error_limit_pct", 10))
    span = lambda key, d: list(range(int(cfg.get(key, d)[0]), int(cfg.get(key, d)[1]) + 1))  # noqa: E731
    years = span("overlap_years", [2012, 2025])
    bw_fit, bw_test = span("backward_fit_years", [2016, 2025]), span("backward_test_years", [2012, 2015])

    cands = sel.get("candidates") or DEFAULT_CANDIDATES
    d19 = source_segments(settings.raw.get("counts") or {})
    chosen19 = next((n for n, s in cands.items() if [[g, int(a), int(b)] for g, a, b in s] == d19), None)
    groups = {g: GROUPS[g] for g in sorted({g for s in cands.values() for g, *_ in s} | {ref})}
    sats = sorted({s for g in groups for s in GROUPS[g]})
    bad = bad_days(settings, sats) if cfg.get("exclude_gap_days", True) else None

    from .static_mask import masked_cells
    mask = masked_cells(settings) if (settings.raw.get("nonfire") or {}).get("mask_all_years", False) else None
    m = monthly_tiles(settings, groups, size, bad, mask)
    if m.empty:
        return {"stage": "worldwide", "note": "no cell-days found"}
    out = settings.processed_dir / "counts"
    out.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out / "worldwide_tiles_monthly.parquet", index=False)

    # Tiles with enough fire to test: average S-NPP active cells per month over the overlap.
    r = m[(m["group"] == ref) & m["year"].isin(years)]
    per_tile = r.groupby("region")["active_cells"].sum() / (12 * len(years))
    testable = per_tile[per_tile >= min_ref].index
    weights = r.groupby("region")["active_cells"].sum()
    mt = m[m["region"].isin(testable)]
    centre = m.groupby("region")[["lat0", "lon0"]].first() + size / 2

    # D19 on burning cells
    w_cells = _wide(mt, "active_cells")
    res19 = _tests(w_cells, cands, ref, years, bw_fit, bw_test, int(sel.get("min_cells_per_month", 30)))
    sum19 = _summary(res19, weights, limit)
    v19 = _verdict(sum19, chosen19, margin) if chosen19 else ("CHECK", "D19 source is not among the candidates")

    # D21 on fire power: ratio of all fire power (D19 pieces) vs v1 line on merged MODIS
    w_frp = _wide(mt, "frp_mw")
    res21 = _tests(w_frp, {"ratio_totals": d19}, ref, years, bw_fit, bw_test, float(frp_cfg.get("min_frp_mw_per_month", 300)))
    a, b = (frp_cfg.get("v1_equation") or [0.316162, 24.701681])
    base = baseline(w_frp, ref, years, [])
    v1rows = []
    wc = w_cells.set_index(["region", "year", "month"])
    for test, ty in (("loyo", years), ("backward", bw_test)):
        t = w_frp[w_frp["year"].isin(ty)].dropna(subset=["MODIS", ref])
        cells = wc["MODIS"].reindex(pd.MultiIndex.from_frame(t[["region", "year", "month"]])).to_numpy(float)
        p = t[["region", "year", "month"]].assign(pred=a * t["MODIS"].to_numpy(float) + b * cells, real=t[ref].to_numpy(float)).dropna()
        sc = score(p, base, []).set_index("region")
        v1rows.append((test, sc, drift(p, []).set_index("region") if test == "loyo" else None))
    lo, bwv = v1rows[0][1], v1rows[1][1]
    dr = v1rows[0][2]
    v1 = pd.DataFrame({"region": lo.index, "candidate": "v1_equation",
                       "agreement_loyo_pct": lo["agreement_pct"].to_numpy(),
                       "agreement_backward_pct": bwv["agreement_pct"].reindex(lo.index).to_numpy(),
                       "backward_error_pct": bwv["total_error_pct"].reindex(lo.index).to_numpy(),
                       "drift_pct_per_year": dr["drift_pct_per_year"].reindex(lo.index).to_numpy()})
    v1["score"] = v1[["agreement_loyo_pct", "agreement_backward_pct"]].mean(axis=1)
    res21 = pd.concat([res21, v1], ignore_index=True)
    sum21 = _summary(res21, weights, limit)
    v21 = _verdict(sum21, "ratio_totals", margin)

    for res in (res19, res21):
        res["lat"] = res["region"].map(centre["lat0"])
        res["lon"] = res["region"].map(centre["lon0"])
        res["fire_cells_per_month"] = res["region"].map(per_tile).round(1)
    res19.to_csv(out / "worldwide_d19_tiles.csv", index=False)
    res21.to_csv(out / "worldwide_d21_tiles.csv", index=False)

    def worst(res, name, n=12):
        x = res[res["candidate"] == name].sort_values("score").head(n)
        return x[["region", "lat", "lon", "fire_cells_per_month", "score", "backward_error_pct"]]

    total_tiles = m["region"].nunique()
    report = {"stage": "worldwide", "tile_deg": size, "nonfire_mask_all_years": mask is not None, "tiles_with_fire": int(total_tiles),
              "tiles_tested": int(len(testable)),
              "fire_share_in_tested_tiles_pct": round(float(100 * weights.reindex(testable).sum() / weights.sum()), 2),
              "rule": rule,
              "d19": {"chosen": chosen19, "verdict": v19[0], "why": v19[1], "summary": sum19.to_dict("records")},
              "d21": {"chosen": "ratio_totals", "verdict": v21[0], "why": v21[1], "summary": sum21.to_dict("records")}}
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 100):
        print(f"[worldwide] non-fire mask in every year (D22): {'on' if mask is not None else 'off'}", flush=True)
        print(f"[worldwide] {total_tiles} tiles of {size:g} degrees have fire; {len(testable)} have enough to test "
              f"(>= {min_ref:g} S-NPP burning cells a month on average), holding {report['fire_share_in_tested_tiles_pct']}% of all fire", flush=True)
        print(f"[worldwide] D19 burning cells, every tested tile:\n{sum19.to_string(index=False)}", flush=True)
        print(f"[worldwide] D19 verdict: {v19[0]} ({v19[1]})", flush=True)
        print(f"[worldwide] D21 fire power, every tested tile:\n{sum21.to_string(index=False)}", flush=True)
        print(f"[worldwide] D21 verdict: {v21[0]} ({v21[1]})", flush=True)
        if chosen19:
            print(f"[worldwide] weakest tiles for D19 ({chosen19}), burning cells:\n{worst(res19, chosen19).to_string(index=False)}", flush=True)
        print(f"[worldwide] weakest tiles for D21 (ratio_totals), fire power:\n{worst(res21, 'ratio_totals').to_string(index=False)}", flush=True)
    return report
