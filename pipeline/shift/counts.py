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
COLS = ["acq_date", "cell_id", "cell_lat", "cell_lon", "satellite", "n_detections"]


def daily_series(settings: Settings) -> pd.DataFrame:
    """Per region, day and satellite group: detections and active cell-days.
    Reads the cell-day record one partition at a time (memory stays small)."""
    regions = {"global": GLOBAL, **test_regions(settings)}
    root = settings.celldays_dir
    parts = []
    for part in sorted(root.glob("sensor_family=*/year=*/month=*")) if root.exists() else []:
        d = pd.read_parquet(part, columns=COLS)
        d["satellite"] = d["satellite"].astype(str)
        for group, sats in GROUPS.items():
            g = d[d["satellite"].isin(sats)]
            if g.empty:
                continue
            if len(sats) > 1:   # one cell-day counts once however many satellites saw it
                g = (g.groupby(["acq_date", "cell_id"], as_index=False)
                      .agg(cell_lat=("cell_lat", "first"), cell_lon=("cell_lon", "first"),
                           n_detections=("n_detections", "sum")))
            for name, reg in regions.items():
                r = g if name == "global" else g[in_bbox(reg["bbox"], g["cell_lon"], g["cell_lat"])]
                if r.empty:
                    continue
                s = r.groupby("acq_date").agg(detections=("n_detections", "sum"),
                                              active_cells=("cell_id", "size")).reset_index()
                s["region"], s["group"] = name, group
                parts.append(s)
    if not parts:
        return pd.DataFrame(columns=["region", "group", "acq_date", "detections", "active_cells"])
    out = pd.concat(parts, ignore_index=True)
    out["acq_date"] = pd.to_datetime(out["acq_date"]).dt.normalize()
    # A day can sit in two partitions only if it came from two sources; add them up.
    return (out.groupby(["region", "group", "acq_date"], as_index=False)[["detections", "active_cells"]]
               .sum().astype({"detections": "int64", "active_cells": "int64"}))


def _monthly(daily: pd.DataFrame, exclude: pd.DatetimeIndex | None) -> pd.DataFrame:
    d = daily if exclude is None or len(exclude) == 0 else daily[~daily["acq_date"].isin(exclude)]
    d = d.assign(year=d["acq_date"].dt.year, month=d["acq_date"].dt.month)
    return d.groupby(["region", "group", "year", "month"], as_index=False)[["detections", "active_cells"]].sum()


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
