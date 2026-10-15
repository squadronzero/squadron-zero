"""Phase 2 fix 5: the confidence asymmetry.

Every detection comes with a confidence label, but the two sensors define it differently:

  * MODIS gives a number from 0 to 100. SHIFT calls < 30 "low" (MODIS C6.1 user guide).
  * VIIRS gives a class directly: low, nominal or high. VIIRS "low" is mostly daytime pixels
    in sun-glint areas or with a weak temperature signal, a different test from MODIS.

Since Phase 1 the pipeline drops "low" for both sensors at ingest. That removes about 4-5% of
MODIS detections but about 11% of VIIRS detections (fetch markers). A ratio per place and month
(D21) absorbs a difference that stays the same over the years; it cannot absorb one that changes
from year to year, which would show up as a false trend. Fix 5 measures this and tests,
under a rule fixed before the run, whether keeping low-confidence detections (for one or both
sensors) gives a better record.

Two stages:

  confscan    downloads the FIRMS yearly world files again (same files as fetch) and keeps
              ONLY the low-confidence vegetation-fire detections, as cell-days in their own
              folder (`celldays_low`). The main record is not touched. Resumable.
  confidence  measures the low-confidence share per satellite, year and tile, and scores the
              variants with the same tiles, tests and D22 mask as stage `factors`.

    python -m shift.run --dataset global --stage confscan
    python -m shift.run --dataset global --stage confidence
"""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .counts import GROUPS, source_segments
from .coverage import bad_days
from .factors import Cube, evaluate, monthly_world
from .fetch import download, process_zip, product_file, zip_url
from .static_mask import masked_cells
from .tiles import tile_name

DIAG_GROUPS = ["Terra", "Aqua", "MODIS", "S-NPP", "NOAA-20"]


def low_dir(settings: Settings) -> Path:
    cfg = settings.raw.get("confidence") or {}
    return settings.processed_dir / cfg.get("low_subdir", "celldays_low")


def low_settings(settings: Settings) -> Settings:
    """The same settings, but keeping ONLY low-confidence detections. Every other filter
    (vegetation fires only, bad values, static mask for data without a type) is unchanged."""
    raw = copy.deepcopy(settings.raw)
    raw["filters"]["keep_confidence"] = ["low"]
    return Settings(raw=raw, root=settings.root)


# ---------------------------------------------------------------- stage confscan

def run_confscan(settings: Settings, years: list[int] | None = None, families: list[str] | None = None,
                 force: bool = False) -> dict:
    cfg = settings["fetch"]
    done_dir = settings.processed_dir / "_confscan_done"
    fetch_done = settings.processed_dir / "_fetch_done"
    done_dir.mkdir(parents=True, exist_ok=True)
    out_dir = low_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = Path(cfg["tmp_dir"]) if cfg.get("tmp_dir") else Path(tempfile.gettempdir())
    ls = low_settings(settings)

    report = {"stage": "confscan", "dataset": settings.dataset, "years": {}, "missing": [], "skipped": [],
              "different_file_from_fetch": []}
    for product, spec in cfg["products"].items():
        if families and product not in families:
            continue
        family = spec.get("family", product)
        filename = product_file(spec)
        for year in range(int(spec["first_year"]), int(cfg["last_year"]) + 1):
            if years and year not in years:
                continue
            key = f"{product}_{year}"
            marker = done_dir / f"{key}.json"
            if marker.exists() and not force:
                report["skipped"].append(key)
                report["years"][key] = {k: v for k, v in json.loads(marker.read_text()).items() if k != "partitions"}
                continue
            t0 = time.time()
            work = Path(tempfile.mkdtemp(prefix=f"shift_conf_{key}_", dir=scratch_root))
            try:
                fmarker = fetch_done / f"{key}.json"
                finfo = json.loads(fmarker.read_text()) if fmarker.exists() else {}
                urls = [finfo["url"]] if finfo.get("url") else [zip_url(cfg["base_url"], p, year) for p in spec["prefixes"]]
                zip_path, used, fp = work / f"{key}.zip", None, None
                for url in urls:
                    fp = download(url, zip_path)
                    if fp:
                        used = url
                        break
                if not used:
                    report["missing"].append(key)
                    print(f"[confscan] {key}: no file found", flush=True)
                    continue
                if marker.exists():  # forced rerun: remove only this product's old files
                    old = json.loads(marker.read_text())
                    for rel in old.get("partitions", []):
                        (out_dir / rel / filename).unlink(missing_ok=True)
                    marker.unlink()
                same = (fp.get("sha256") == finfo.get("sha256")) if finfo.get("sha256") else None
                if same is False:
                    report["different_file_from_fetch"].append(key)
                partitions, counts = process_zip(zip_path, family, ls, work, out_dir, spec.get("satellites"), filename)
                info = {"url": used, "family": family, "file": filename, "sha256": fp["sha256"],
                        "same_file_as_fetch": same, "rows_read": counts.get("rows_read", 0),
                        "low_kept": counts.get("rows_kept", 0), "low_celldays": counts.get("celldays", 0),
                        "main_kept": finfo.get("rows_kept"), "partitions": partitions,
                        "seconds": round(time.time() - t0, 1)}
                tot = info["low_kept"] + (info["main_kept"] or 0)
                info["low_share_of_kept_pct"] = round(100 * info["low_kept"] / tot, 2) if tot else None
                marker.write_text(json.dumps(info, indent=1))
                report["years"][key] = {k: v for k, v in info.items() if k != "partitions"}
                print(f"[confscan] {key}: {info['rows_read']:,} rows read, {info['low_kept']:,} low-confidence fire "
                      f"detections kept ({info['low_share_of_kept_pct']}% of all fire detections), "
                      f"{info['low_celldays']:,} cell-days, same file as fetch: {same}, {info['seconds']} s", flush=True)
            finally:
                shutil.rmtree(work, ignore_errors=True)
    shares = pd.DataFrame([{"key": k, **v} for k, v in report["years"].items()])
    if len(shares):
        report["low_share_by_year"] = shares.set_index("key")["low_share_of_kept_pct"].to_dict()
    return report


# ---------------------------------------------------------------- stage confidence

def _yearly_share(main: pd.DataFrame, allc: pd.DataFrame, var: str, by: list[str],
                  none_is_missing: bool = False) -> pd.DataFrame:
    """Low-confidence share. With none_is_missing, a group that gained nothing at all from the
    low-confidence folder counts as NOT SCANNED (NaN), not as 0%: confscan covers the yearly
    files only, so 2026 (form downloads) has no low-confidence data and must not pull trends down."""
    a = main.groupby(by)[var].sum()
    b = allc.groupby(by)[var].sum()
    t = pd.DataFrame({"kept": a, "all": b}).fillna(0)
    t["low_share_pct"] = np.where(t["all"] > 0, 100 * (1 - t["kept"] / t["all"]), np.nan)
    if none_is_missing:
        t.loc[t["all"] <= t["kept"], "low_share_pct"] = np.nan
    return t.reset_index()


def _slope(x: pd.Series, y: pd.Series) -> float:
    ok = y.notna()
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 3 else np.nan


def run_confidence(settings: Settings) -> dict:
    low_root = low_dir(settings)
    if not low_root.exists() or not any(low_root.glob("sensor_family=*")):
        return {"stage": "confidence", "note": "no low-confidence cell-days found: run stage confscan first"}
    fcfg = settings.raw.get("factors") or {}
    ccfg = settings.raw.get("confidence") or {}
    ref = fcfg.get("reference", "S-NPP")
    tile_deg, zone_deg = float(fcfg.get("tile_deg", 10)), float(fcfg.get("zone_deg", 30))
    min_ref = float(fcfg.get("min_ref_cells_per_month", 50))
    prior = {"cells": float(fcfg.get("prior_cells", 30)), "frp": float(fcfg.get("prior_frp_mw", 300))}
    min_v = {"cells": float(fcfg.get("min_cells_per_month", 30)), "frp": float(fcfg.get("min_frp_mw_per_month", 300))}
    span = lambda key, d: list(range(int(fcfg.get(key, d)[0]), int(fcfg.get(key, d)[1]) + 1))  # noqa: E731
    years = span("overlap_years", [2012, 2025])
    bw_fit, bw_test = span("backward_fit_years", [2016, 2025]), span("backward_test_years", [2012, 2015])
    rule = ccfg.get("rule") or {}
    gain, limit = float(rule.get("gain_points", 2)), float(rule.get("error_limit_pct", 10))
    segments = source_segments(settings.raw.get("counts") or {})
    use_mask = bool((settings.raw.get("nonfire") or {}).get("mask_all_years", False))

    names = sorted({g for g, *_ in segments} | {ref} | set(DIAG_GROUPS))
    groups = {g: GROUPS[g] for g in names}
    bad = bad_days(settings, sorted({s for g in groups for s in GROUPS[g]})) if fcfg.get("exclude_gap_days", True) else None
    mask = masked_cells(settings)
    print("[confidence] reading the record twice (as used now, and with low-confidence detections added); "
          "this prints nothing for several minutes", flush=True)
    m_main = monthly_world(settings, groups, tile_deg, bad, None, mask)
    m_all = monthly_world(settings, groups, tile_deg, bad, None, mask, extra_root=low_root)
    for m in (m_main, m_all):
        m["tile"] = [tile_name(a, b, tile_deg) for a, b in zip(m["lat0"], m["lon0"])]

    # -------- 1. how much is "low", per satellite and year (fires only: D22 cells left out)
    key = ["group", "year"]
    fm, fa = m_main[~m_main["in_mask"]], m_all[~m_all["in_mask"]]
    share = _yearly_share(fm, fa, "active_cells", key, True).rename(columns={"low_share_pct": "low_share_cells_pct"})
    share_frp = _yearly_share(fm, fa, "frp_mw", key, True)[key + ["low_share_pct"]].rename(columns={"low_share_pct": "low_share_frp_pct"})
    share = share.merge(share_frp, on=key)[key + ["low_share_cells_pct", "low_share_frp_pct"]]
    share = share[share["group"].isin(DIAG_GROUPS) & (share["year"] >= 2003)
                  & share["low_share_cells_pct"].notna()].round(2)
    trend = {g: {"cells_pct_points_per_year": round(_slope(d["year"], d["low_share_cells_pct"]), 3),
                 "frp_pct_points_per_year": round(_slope(d["year"], d["low_share_frp_pct"]), 3),
                 "mean_cells_pct": round(float(d["low_share_cells_pct"].mean()), 2),
                 "years": f"{int(d['year'].min())}-{int(d['year'].max())}"}
             for g, d in share.groupby("group")}

    # -------- 2. the asymmetry per tile: S-NPP low share minus the D19 source's low share
    def d19(x):
        sel = np.zeros(len(x), bool)
        for g, y0, y1 in segments:
            sel |= ((x["group"] == g) & x["year"].between(int(y0), int(y1))).to_numpy()
        return x[sel]

    by_t = ["tile", "year"]
    s_src = _yearly_share(d19(fm), d19(fa), "active_cells", by_t).rename(columns={"low_share_pct": "src_low_pct"})
    s_ref = _yearly_share(fm[fm["group"] == ref], fa[fa["group"] == ref], "active_cells", by_t).rename(
        columns={"low_share_pct": "ref_low_pct"})
    asym = s_src[["tile", "year", "src_low_pct"]].merge(s_ref[["tile", "year", "ref_low_pct"]], on=by_t)
    asym = asym[asym["year"].isin(years)]
    asym["gap_pct_points"] = asym["ref_low_pct"] - asym["src_low_pct"]

    # tiles tested: the same as stage factors (enough S-NPP fire, D22 cells not counting)
    r = m_main[(m_main["group"] == ref) & m_main["year"].isin(years) & (~m_main["in_mask"] if use_mask else True)]
    per_tile = r.groupby("tile")["active_cells"].sum() / (12 * len(years))
    tiles = per_tile[per_tile >= min_ref].index
    weights = r.groupby("tile")["active_cells"].sum()

    tile_rows = {}
    for t, d in asym[asym["tile"].isin(tiles)].groupby("tile"):
        tile_rows[t] = {"mean_gap_pct_points": d["gap_pct_points"].mean(),
                        "gap_trend_pct_points_per_year": _slope(d["year"], d["gap_pct_points"]),
                        "ref_low_pct": d["ref_low_pct"].mean(), "src_low_pct": d["src_low_pct"].mean()}
    tile_tbl = pd.DataFrame.from_dict(tile_rows, orient="index",
                                      columns=["mean_gap_pct_points", "gap_trend_pct_points_per_year", "ref_low_pct", "src_low_pct"])
    tile_tbl.index.name = "tile"
    tile_tbl["fire_cells"] = weights.reindex(tile_tbl.index)
    tile_tbl = tile_tbl.round(3).sort_values("gap_trend_pct_points_per_year", key=np.abs, ascending=False)
    w = tile_tbl["fire_cells"].fillna(0)
    ok = tile_tbl["gap_trend_pct_points_per_year"].notna() & (w > 0)
    fw_gap_trend = float(np.average(tile_tbl["gap_trend_pct_points_per_year"][ok].abs(), weights=w[ok])) if ok.any() else None
    fw_gap = float(np.average(tile_tbl["mean_gap_pct_points"][ok], weights=w[ok])) if ok.any() else None

    # -------- 3. the variants, scored like stage factors (rule fixed before the run)
    variants = ccfg.get("variants") or {"current": ["main", "main"], "all": ["all", "all"],
                                         "modis_all": ["all", "main"], "viirs_all": ["main", "all"]}
    src_groups = sorted({g for g, *_ in segments})
    ally = list(range(int(min(m_main["year"])), int(max(m_main["year"])) + 1))
    rows, per_tile_scores = [], {}
    for name, (src_from, ref_from) in variants.items():
        pick = {"main": m_main, "all": m_all}
        mv = pd.concat([pick[src_from][pick[src_from]["group"].isin(src_groups)],
                        pick[ref_from][pick[ref_from]["group"] == ref]], ignore_index=True)
        res = {}
        for var in ("frp", "cells"):
            c = Cube(mv, var, use_mask, False, zone_deg, ally, tile_deg)
            res[var] = evaluate(c, segments, ref, years, bw_fit, bw_test, False, False, prior[var], min_v[var],
                                tiles, weights, limit)
        per_tile_scores[name] = res["frp"]["per_tile"]
        both = [res[v]["fire_weighted_score"] for v in ("frp", "cells")]
        rows.append({"variant": name, "modis": src_from, "viirs": ref_from,
                     "score": round(float(np.mean(both)), 2) if None not in both else None,
                     "score_frp": res["frp"]["fire_weighted_score"], "score_cells": res["cells"]["fire_weighted_score"],
                     "tiles_within_limit_frp_pct": res["frp"]["tiles_within_limit_pct"],
                     "tiles_within_limit_cells_pct": res["cells"]["tiles_within_limit_pct"],
                     "global_backward_error_frp_pct": res["frp"]["global_backward_error_pct"],
                     "global_backward_error_cells_pct": res["cells"]["global_backward_error_pct"],
                     "false_trend_frp": res["frp"]["fire_weighted_abs_drift"],
                     "false_trend_cells": res["cells"]["fire_weighted_abs_drift"]})
    tbl = pd.DataFrame(rows)

    def eligible(row):
        e = [row["global_backward_error_frp_pct"], row["global_backward_error_cells_pct"]]
        return all(x is not None and abs(x) <= limit for x in e)

    tbl["eligible"] = tbl.apply(eligible, axis=1)
    cur = tbl[tbl["variant"] == "current"].iloc[0]
    challengers = tbl[(tbl["variant"] != "current") & tbl["eligible"]
                      & (tbl["score"] > cur["score"] + gain)
                      & (tbl["tiles_within_limit_frp_pct"] >= cur["tiles_within_limit_frp_pct"])]
    if len(challengers):
        win = challengers.sort_values("score", ascending=False).iloc[0]
        why = (f"{win['variant']} beats the current filter by {win['score'] - cur['score']:.2f} points "
               f"(needed more than {gain:g}), is eligible and keeps at least as many tiles within +-{limit:g}%")
    else:
        win = cur
        best = tbl[tbl["variant"] != "current"].sort_values("score", ascending=False).iloc[0]
        why = (f"no variant beats the current filter by more than {gain:g} points while eligible "
               f"(best other: {best['variant']}, {best['score'] - cur['score']:+.2f} points)")

    out = settings.processed_dir / "counts"
    out.mkdir(parents=True, exist_ok=True)
    share.to_csv(out / "confidence_low_share_yearly.csv", index=False)
    tile_tbl.to_csv(out / "confidence_tile_asymmetry.csv")
    tbl.to_csv(out / "confidence_variants.csv", index=False)
    pd.DataFrame({k: v["backward_error_pct"] for k, v in per_tile_scores.items()}).round(2).to_csv(
        out / "confidence_tile_backward_error_frp.csv")

    report = {"stage": "confidence", "winner": str(win["variant"]), "why": why,
              "low_share_trend": trend,
              "fire_weighted_mean_gap_pct_points": round(fw_gap, 2) if fw_gap is not None else None,
              "fire_weighted_abs_gap_trend_pct_points_per_year": round(fw_gap_trend, 3) if fw_gap_trend is not None else None,
              "tiles_tested": int(len(tiles)), "variants": tbl.to_dict("records"),
              "note": "2026 form downloads (near-real-time) are not in confscan; added only if a variant other than current wins"}
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 80):
        print("[confidence] low-confidence share of burning cells, %, per satellite and year (fire power share is in the CSV):\n"
              + share.pivot(index="year", columns="group", values="low_share_cells_pct").to_string(), flush=True)
        print("[confidence] trend of that share (percentage points per year):\n" + pd.DataFrame(trend).T.to_string(), flush=True)
        if fw_gap is not None:
            print(f"[confidence] asymmetry over {len(tiles)} tested tiles: S-NPP drops {fw_gap:+.2f} points more than "
                  f"MODIS (fire-weighted); that gap changes by {fw_gap_trend:.3f} points a year on average", flush=True)
        print("[confidence] tiles where the gap changes most:\n" + tile_tbl.head(10).to_string(), flush=True)
        print("[confidence] variants (rule fixed before the run):\n" + tbl.to_string(index=False), flush=True)
        print(f"[confidence] RESULT: {win['variant']} | {why}", flush=True)
    return report
