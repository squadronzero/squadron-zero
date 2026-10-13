"""Phase 2, fix 1: detection-count harmonisation (removes the "2012 cliff").

The problem. VIIRS (375 m pixels) reports about 4 times as many fire detections as
MODIS (1 km pixels) for the same fires: one fire lights up several small VIIRS pixels,
and VIIRS also sees small fires MODIS misses. A raw count record that switches from
MODIS to VIIRS in 2012 therefore jumps upward for instrument reasons, not because the
world started burning more.

The method, in two steps, both checked on years the model never saw:

  1. Collapse. Count ACTIVE CELL-DAYS (a 0.05 deg cell with at least one detection that
     day) instead of detections. Several pixels on one fire become one active cell, so
     the finer sensor cannot inflate the count. MODIS Terra and Aqua are merged first:
     a cell-day is MODIS-active if either satellite saw fire there.
  2. Scale. VIIRS still sees some fires MODIS misses, so over the 2012-2021 overlap we
     measure, per test region and calendar month, how many VIIRS active cell-days there
     are per MODIS active cell-day (factor k). The MODIS-era record (2002-2011) is then
     put on the VIIRS scale as MODIS x k. Days when any of the satellites was not fully
     working (qc coverage table) are left out of the fit.

Validation on 2022-2025: predicted VIIRS (MODIS x k) against real VIIRS, per region and
month, plus the size of the 2012 step in the raw and the harmonised record.

    python -m shift.run --dataset global --stage counts
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings
from .coverage import bad_days
from .region import in_bbox, test_regions

# Satellite groups the series are built for. "MODIS" = Terra and Aqua merged per cell-day.
GROUPS = {"MODIS": ["Terra", "Aqua"], "Aqua": ["Aqua"], "Terra": ["Terra"],
          "S-NPP": ["S-NPP"], "NOAA-20": ["NOAA-20"]}
GLOBAL = {"bbox": [-180.0, -90.0, 180.0, 90.0], "regime": "whole world"}
COLS = ["acq_date", "cell_id", "cell_lat", "cell_lon", "satellite", "n_detections", "frp_sum_mw"]


def daily_series(settings: Settings, groups: dict | None = None,
                 cell_class: pd.Series | None = None) -> pd.DataFrame:
    """Per region, day and satellite group: detections and active cell-days.
    Reads the cell-day record one partition at a time (memory stays small).
    With `cell_class` (a label per cell_id), the counts are split by that label as well
    (column `size`); cells without a label get "unlabelled"."""
    groups = groups or GROUPS
    regions = {"global": GLOBAL, **test_regions(settings)}
    root = settings.celldays_dir
    keys = ["acq_date"] if cell_class is None else ["acq_date", "size"]
    parts = []
    for part in sorted(root.glob("sensor_family=*/year=*/month=*")) if root.exists() else []:
        d = pd.read_parquet(part, columns=COLS)
        d["satellite"] = d["satellite"].astype(str)
        if not d["satellite"].isin({s for v in groups.values() for s in v}).any():
            continue
        for group, sats in groups.items():
            g = d[d["satellite"].isin(sats)]
            if g.empty:
                continue
            if len(sats) > 1:   # one cell-day counts once however many satellites saw it
                g = (g.groupby(["acq_date", "cell_id"], as_index=False)
                      .agg(cell_lat=("cell_lat", "first"), cell_lon=("cell_lon", "first"),
                           n_detections=("n_detections", "sum"), frp_sum_mw=("frp_sum_mw", "sum")))
            if cell_class is not None:
                g = g.assign(size=g["cell_id"].map(cell_class).fillna("unlabelled").astype(str))
            for name, reg in regions.items():
                r = g if name == "global" else g[in_bbox(reg["bbox"], g["cell_lon"], g["cell_lat"])]
                if r.empty:
                    continue
                s = r.groupby(keys).agg(detections=("n_detections", "sum"),
                                        active_cells=("cell_id", "size"),
                                        frp_mw=("frp_sum_mw", "sum")).reset_index()
                s["region"], s["group"] = name, group
                parts.append(s)
    cols = ["region", "group"] + keys
    if not parts:
        return pd.DataFrame(columns=cols + ["detections", "active_cells", "frp_mw"])
    out = pd.concat(parts, ignore_index=True)
    out["acq_date"] = pd.to_datetime(out["acq_date"]).dt.normalize()
    # A day can sit in two partitions only if it came from two sources; add them up.
    return (out.groupby(cols, as_index=False)[["detections", "active_cells", "frp_mw"]]
               .sum().astype({"detections": "int64", "active_cells": "int64"}))


def _monthly(daily: pd.DataFrame, exclude: pd.DatetimeIndex | None) -> pd.DataFrame:
    d = daily if exclude is None or len(exclude) == 0 else daily[~daily["acq_date"].isin(exclude)]
    d = d.assign(year=d["acq_date"].dt.year, month=d["acq_date"].dt.month)
    cols = [c for c in ("detections", "active_cells", "frp_mw") if c in d.columns]
    return d.groupby(["region", "group", "year", "month"], as_index=False)[cols].sum()


def _years(start: str, end: str) -> list[int]:
    return list(range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1))


def fit_factors(monthly: pd.DataFrame, source: str, reference: str, years: list[int],
                min_cells: int) -> pd.DataFrame:
    """k per region and calendar month = reference active cells / source active cells over
    `years`. Months with fewer than `min_cells` source cells use the region's all-year k."""
    m = monthly[monthly["year"].isin(years)]
    piv = m.pivot_table(index=["region", "year", "month"], columns="group", values="active_cells",
                        aggfunc="sum").reset_index()
    if source not in piv or reference not in piv:
        return pd.DataFrame(columns=["region", "month", "k", "k_region", "source_cells", "used"])
    piv = piv.dropna(subset=[source, reference])          # both sensors in the record that month
    rows = []
    for region, r in piv.groupby("region"):
        k_region = r[reference].sum() / r[source].sum() if r[source].sum() > 0 else np.nan
        for month in range(1, 13):
            mm = r[r["month"] == month]
            src = mm[source].sum()
            ok = src >= min_cells
            k = mm[reference].sum() / src if ok else k_region
            rows.append({"region": region, "month": month, "k": float(k), "k_region": float(k_region),
                         "source_cells": int(src), "used": "month" if ok else "region"})
    return pd.DataFrame(rows)


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan")


def validate(monthly: pd.DataFrame, factors: pd.DataFrame, source: str, reference: str,
             years: list[int]) -> pd.DataFrame:
    """Predicted reference (source x k) against real reference, region-months in `years`."""
    m = monthly[monthly["year"].isin(years)]
    piv = m.pivot_table(index=["region", "year", "month"], columns="group",
                        values=["active_cells", "detections"], aggfunc="sum")
    rows = []
    if ("active_cells", source) not in piv or ("active_cells", reference) not in piv:
        return pd.DataFrame(rows)
    piv = piv.dropna(subset=[("active_cells", source), ("active_cells", reference)]).reset_index()
    piv.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in piv.columns]
    piv = piv.merge(factors[["region", "month", "k"]], on=["region", "month"], how="left")
    for region, r in piv.groupby("region"):
        y = r[f"active_cells_{reference}"].to_numpy(float)
        x = r[f"active_cells_{source}"].to_numpy(float)
        p = x * r["k"].to_numpy(float)
        yd = r[f"detections_{reference}"].to_numpy(float)
        xd = r[f"detections_{source}"].to_numpy(float)
        rows.append({
            "region": region, "months": int(len(r)),
            "r2_harmonised": round(_r2(y, p), 4),
            "r2_raw_detections": round(_r2(yd, xd), 4),
            "total_error_pct": round(float(100 * (p.sum() - y.sum()) / y.sum()), 2) if y.sum() else None,
            "median_abs_error_pct": round(float(100 * np.median(np.abs(p - y) / np.maximum(y, 1))), 1),
            "corr_detections": round(float(np.corrcoef(xd, yd)[0, 1]), 4) if len(r) > 2 else None,
            "corr_active_cells": round(float(np.corrcoef(x, y)[0, 1]), 4) if len(r) > 2 else None,
            "raw_ratio_detections": round(float(yd.sum() / xd.sum()), 3) if xd.sum() else None,
            "raw_ratio_active_cells": round(float(y.sum() / x.sum()), 3) if x.sum() else None,
        })
    return pd.DataFrame(rows)


def step_at_switch(monthly: pd.DataFrame, factors: pd.DataFrame, source: str, reference: str,
                   switch_year: int, years_each_side: int = 3) -> pd.DataFrame:
    """Size of the jump where the record switches from MODIS to VIIRS: mean yearly count in
    the years after the switch / before it. Raw = MODIS detections then VIIRS detections.
    Harmonised = MODIS active cells x k then VIIRS active cells. 1.0 = no jump."""
    before = list(range(switch_year - years_each_side, switch_year))
    after = list(range(switch_year + 1, switch_year + 1 + years_each_side))  # skip the partial first year
    k = factors.set_index(["region", "month"])["k"]
    rows = []
    for region, r in monthly.groupby("region"):
        src = r[(r["group"] == source) & r["year"].isin(before)]
        ref = r[(r["group"] == reference) & r["year"].isin(after)]
        if src.empty or ref.empty:
            continue
        kk = np.array([k.get((region, mth), np.nan) for mth in src["month"]], dtype=float)
        raw_before = src["detections"].sum() / len(before)
        raw_after = ref["detections"].sum() / len(after)
        harm_before = np.nansum(src["active_cells"].to_numpy(float) * kk) / len(before)
        harm_after = ref["active_cells"].sum() / len(after)
        # The real change in fire activity between the two windows, seen by MODIS alone
        # (same sensor both sides). A good harmonisation makes step_harmonised close to it.
        src_after = r[(r["group"] == source) & r["year"].isin(after)]["active_cells"].sum() / len(after)
        src_before = src["active_cells"].sum() / len(before)
        rows.append({"region": region, "years_before": f"{before[0]}-{before[-1]}",
                     "years_after": f"{after[0]}-{after[-1]}",
                     "step_raw": round(float(raw_after / raw_before), 3) if raw_before else None,
                     "step_harmonised": round(float(harm_after / harm_before), 3) if harm_before else None,
                     "real_change_modis": round(float(src_after / src_before), 3) if src_before else None})
    return pd.DataFrame(rows)


def run_counts(settings: Settings) -> dict:
    cfg = settings.raw.get("counts") or {}
    source, reference = cfg.get("source", "MODIS"), cfg.get("reference", "S-NPP")
    min_cells = int(cfg.get("min_cells_per_month", 30))
    periods = settings["periods"]
    cal = _years(periods["calibration_start"], periods["calibration_end"])
    hold = _years(periods["holdout_start"], periods["holdout_end"])

    daily = daily_series(settings)
    if daily.empty:
        return {"stage": "counts", "note": "no cell-days found"}
    sats = sorted({s for g in (source, reference) for s in GROUPS[g]})
    bad = bad_days(settings, sats) if cfg.get("exclude_gap_days", True) else None
    monthly = _monthly(daily, bad)
    factors = fit_factors(monthly, source, reference, cal, min_cells)
    val = validate(monthly, factors, source, reference, hold)
    step = step_at_switch(_monthly(daily, None), factors, source, reference,
                          pd.Timestamp(periods["viirs_start"]).year)

    out = settings.processed_dir / "counts"
    out.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out / "daily.parquet", index=False)
    factors.to_csv(out / "factors.csv", index=False)
    val.to_csv(out / "validation_holdout.csv", index=False)
    step.to_csv(out / "step_2012.csv", index=False)

    k_year = factors.groupby("region")["k_region"].first().round(3).to_dict() if len(factors) else {}
    report = {
        "stage": "counts", "source": source, "reference": reference,
        "calibration_years": [cal[0], cal[-1]], "holdout_years": [hold[0], hold[-1]],
        "outage_days_excluded": None if bad is None else int(len(bad)),
        "k_per_region": k_year,
        "holdout": val.to_dict("records"),
        "step_2012": step.to_dict("records"),
        "outputs": [str(p.relative_to(settings.processed_dir)) for p in sorted(out.iterdir())],
    }
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"[counts] holdout ({hold[0]}-{hold[-1]}), harmonised MODIS vs real VIIRS:\n"
              + (val.to_string(index=False) if len(val) else "none"), flush=True)
        print("[counts] jump at the 2012 switch (1.0 = none):\n"
              + (step.to_string(index=False) if len(step) else "none"), flush=True)
    return report


# ---------------------------------------------------------------- fix 2: NOAA-20 continuity

def run_continuity(settings: Settings) -> dict:
    """Phase 2, fix 2: tie VIIRS NOAA-20 to VIIRS S-NPP (D8).

    NASA stops delivering Suomi NPP data on 1 Nov 2026. NOAA-20 carries the same VIIRS
    instrument, so over the 2018-2026 overlap we measure, per test region and calendar
    month, S-NPP active cells (and fire power) per NOAA-20 active cell (fit years), test it
    on later years, and show how NOAA-20 fills the S-NPP outage of 28 Apr-2 Jun 2026.
    Reads the daily series written by the `counts` stage (run that first).
    """
    cfg = settings.raw.get("continuity") or {}
    source, reference = cfg.get("source", "NOAA-20"), cfg.get("reference", "S-NPP")
    fit = list(range(int(cfg.get("fit_years", [2019, 2023])[0]), int(cfg.get("fit_years", [2019, 2023])[1]) + 1))
    test = list(range(int(cfg.get("test_years", [2024, 2025])[0]), int(cfg.get("test_years", [2024, 2025])[1]) + 1))
    min_cells = int(cfg.get("min_cells_per_month", 30))
    out = settings.processed_dir / "counts"
    path = out / "daily.parquet"
    if not path.exists():
        return {"stage": "continuity", "note": "no counts/daily.parquet: run the counts stage first"}
    daily = pd.read_parquet(path)
    daily["acq_date"] = pd.to_datetime(daily["acq_date"])
    if "frp_mw" not in daily.columns:
        return {"stage": "continuity", "note": "daily.parquet has no fire power column: rerun the counts stage"}
    bad = bad_days(settings, [source, reference]) if cfg.get("exclude_gap_days", True) else None
    monthly = _monthly(daily, bad)
    factors = fit_factors(monthly, source, reference, fit, min_cells)
    val = validate(monthly, factors, source, reference, test)

    # Fire power ratio per region (all fit months together) and its error on the test years.
    m = monthly[monthly["group"].isin([source, reference])]
    piv = m.pivot_table(index=["region", "year", "month"], columns="group", values="frp_mw", aggfunc="sum").dropna()
    frp_rows = []
    for region, r in piv.reset_index().groupby("region"):
        f, t = r[r["year"].isin(fit)], r[r["year"].isin(test)]
        if f[source].sum() <= 0:
            continue
        k_frp = f[reference].sum() / f[source].sum()
        err = 100 * (t[source].sum() * k_frp - t[reference].sum()) / t[reference].sum() if t[reference].sum() else None
        frp_rows.append({"region": region, "k_frp": round(float(k_frp), 4),
                         "test_total_frp_error_pct": None if err is None else round(float(err), 2)})
    frp = pd.DataFrame(frp_rows)

    # Gap fill: days when S-NPP was silent or low but NOAA-20 was fine.
    from .coverage import load_coverage
    cov = load_coverage(settings)
    gap_rows = []
    if cov is not None:
        st = cov.pivot_table(index="acq_date", columns="satellite", values="status", aggfunc="first")
        if reference in st and source in st:
            gap_days = st.index[(st[reference] != "ok") & (st[source] == "ok")]
            k = factors.set_index(["region", "month"])["k"]
            g = daily[(daily["group"] == source) & daily["acq_date"].isin(gap_days)].copy()
            g["k"] = [k.get((r, d.month), np.nan) for r, d in zip(g["region"], g["acq_date"])]
            g["snpp_equivalent_cells"] = (g["active_cells"] * g["k"]).round(1)
            gap_rows = g
    gap = gap_rows if isinstance(gap_rows, pd.DataFrame) else pd.DataFrame(gap_rows)

    factors.to_csv(out / "continuity_factors.csv", index=False)
    val.to_csv(out / "continuity_validation.csv", index=False)
    frp.to_csv(out / "continuity_frp.csv", index=False)
    if len(gap):
        gap.to_csv(out / "continuity_gapfill.csv", index=False)

    glob = gap[gap["region"] == "global"] if len(gap) else gap
    report = {
        "stage": "continuity", "source": source, "reference": reference,
        "fit_years": [fit[0], fit[-1]], "test_years": [test[0], test[-1]],
        "days_excluded": None if bad is None else int(len(bad)),
        "k_cells_per_region": factors.groupby("region")["k_region"].first().round(3).to_dict() if len(factors) else {},
        "frp": frp.to_dict("records"),
        "test": val.to_dict("records"),
        "gapfill": {"days": int(glob["acq_date"].nunique()) if len(glob) else 0,
                    "first": str(glob["acq_date"].min().date()) if len(glob) else None,
                    "last": str(glob["acq_date"].max().date()) if len(glob) else None},
    }
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"[continuity] test ({test[0]}-{test[-1]}), NOAA-20 x k vs real S-NPP:\n"
              + (val.to_string(index=False) if len(val) else "none"), flush=True)
        print("[continuity] fire power ratio S-NPP / NOAA-20:\n" + (frp.to_string(index=False) if len(frp) else "none"), flush=True)
    return report


# ---------------------------------------------------------------- fix 3: which MODIS satellite

def yearly_ratio(monthly: pd.DataFrame, source: str, reference: str, years: list[int]) -> pd.DataFrame:
    """Reference active cells / source active cells per region and year. A stable sensor
    gives a flat line; orbital drift shows up as a trend."""
    m = monthly[monthly["year"].isin(years) & monthly["group"].isin([source, reference])]
    piv = (m.pivot_table(index=["region", "year", "month"], columns="group", values="active_cells", aggfunc="sum")
             .dropna().reset_index())
    if source not in piv or reference not in piv:
        return pd.DataFrame(columns=["region", "year", "ratio"])
    y = piv.groupby(["region", "year"], as_index=False)[[source, reference]].sum()
    y["ratio"] = (y[reference] / y[source]).round(4)
    return y[["region", "year", "ratio"]]


def drift_pct_per_year(ratios: pd.DataFrame) -> dict:
    """Straight-line trend of the yearly ratio, as % of its mean per year, per region."""
    out = {}
    for region, r in ratios.groupby("region"):
        if len(r) < 3:
            continue
        slope = np.polyfit(r["year"].to_numpy(float), r["ratio"].to_numpy(float), 1)[0]
        out[region] = round(float(100 * slope / r["ratio"].mean()), 2)
    return out


def run_sources(settings: Settings) -> dict:
    """Phase 2, fix 3: calibrate from Aqua alone, Terra alone, or both merged?

    Terra's orbit has been drifting to an earlier overpass time since 2020 (end of mission
    manoeuvres), so it sees fewer afternoon fires each year, and that drift lands in the
    2022-2025 holdout. Aqua is more stable. For each candidate the stage fits the same
    k-per-region-and-month as fix 1, with the SAME left-out days for every candidate (any of
    Terra, Aqua, S-NPP not fully working), then compares holdout error, the 2012 step and
    the drift of the yearly S-NPP/candidate ratio. Reads counts/daily.parquet (run `counts` first).
    """
    cfg = settings.raw.get("sources") or {}
    candidates = cfg.get("candidates", ["MODIS", "Aqua", "Terra"])
    reference = cfg.get("reference", "S-NPP")
    min_cells = int(cfg.get("min_cells_per_month", 30))
    periods = settings["periods"]
    cal = _years(periods["calibration_start"], periods["calibration_end"])
    hold = _years(periods["holdout_start"], periods["holdout_end"])
    out = settings.processed_dir / "counts"
    path = out / "daily.parquet"
    if not path.exists():
        return {"stage": "sources", "note": "no counts/daily.parquet: run the counts stage first"}
    daily = pd.read_parquet(path)
    daily["acq_date"] = pd.to_datetime(daily["acq_date"])

    sats = sorted({s for g in candidates + [reference] for s in GROUPS[g]})
    bad = bad_days(settings, sats) if cfg.get("exclude_gap_days", True) else None
    monthly = _monthly(daily, bad)
    switch = pd.Timestamp(periods["viirs_start"]).year
    overlap = list(range(cal[0], hold[-1] + 1))

    vals, steps, ratios, drift, factors = [], [], [], {}, []
    for src in candidates:
        f = fit_factors(monthly, src, reference, cal, min_cells)
        v = validate(monthly, f, src, reference, hold)
        s = step_at_switch(_monthly(daily, None), f, src, reference, switch)
        r = yearly_ratio(monthly, src, reference, overlap)
        for df in (f, v, s, r):
            if len(df):
                df.insert(0, "source", src)
        factors.append(f); vals.append(v); steps.append(s); ratios.append(r)
        drift[src] = drift_pct_per_year(r.drop(columns="source")) if len(r) else {}
    val = pd.concat(vals, ignore_index=True)
    step = pd.concat(steps, ignore_index=True)
    ratio = pd.concat(ratios, ignore_index=True)
    fac = pd.concat(factors, ignore_index=True)

    # One line per candidate: how well it does across the test regions (global kept separate).
    summary = {}
    for src in candidates:
        v = val[val["source"] == src]
        reg = v[v["region"] != "global"]
        g = v[v["region"] == "global"]
        summary[src] = {
            "global_r2": float(g["r2_harmonised"].iloc[0]) if len(g) else None,
            "global_error_pct": float(g["total_error_pct"].iloc[0]) if len(g) else None,
            "regions_median_r2": round(float(reg["r2_harmonised"].median()), 4) if len(reg) else None,
            "regions_mean_abs_error_pct": round(float(reg["total_error_pct"].abs().mean()), 2) if len(reg) else None,
            "regions_worst_abs_error_pct": round(float(reg["total_error_pct"].abs().max()), 2) if len(reg) else None,
            "global_drift_pct_per_year": drift[src].get("global"),
        }
    best = min((s for s in summary if summary[s]["regions_mean_abs_error_pct"] is not None),
               key=lambda s: summary[s]["regions_mean_abs_error_pct"], default=None)

    val.to_csv(out / "sources_validation.csv", index=False)
    step.to_csv(out / "sources_step.csv", index=False)
    ratio.to_csv(out / "sources_yearly_ratio.csv", index=False)
    fac.to_csv(out / "sources_factors.csv", index=False)

    report = {"stage": "sources", "candidates": candidates, "reference": reference,
              "calibration_years": [cal[0], cal[-1]], "holdout_years": [hold[0], hold[-1]],
              "days_excluded": None if bad is None else int(len(bad)),
              "summary": summary, "lowest_regional_error": best,
              "drift_pct_per_year": drift,
              "holdout": val.to_dict("records"), "step": step.to_dict("records")}
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        err = val.pivot(index="region", columns="source", values="total_error_pct")[candidates]
        r2 = val.pivot(index="region", columns="source", values="r2_harmonised")[candidates]
        st = step.pivot(index="region", columns="source", values="step_harmonised")[candidates]
        st["real_change"] = step[step["source"] == candidates[0]].set_index("region")["real_change_modis"]
        dr = pd.DataFrame(drift)[candidates]
        print(f"[sources] holdout ({hold[0]}-{hold[-1]}) total error % (S-NPP predicted from each):\n{err.to_string()}", flush=True)
        print(f"[sources] holdout R2:\n{r2.to_string()}", flush=True)
        print(f"[sources] 2012 step after harmonisation (1.0 = none; compare real_change):\n{st.to_string()}", flush=True)
        print(f"[sources] drift of yearly S-NPP/source ratio, % per year ({overlap[0]}-{overlap[-1]}):\n{dr.to_string()}", flush=True)
    return report
