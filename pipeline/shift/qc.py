"""QC stage: data health check over the whole cell-day record.

Reads only four columns of every cell-day partition and builds, per satellite:

  * a daily coverage table (date, satellite, detections, cells, FRP, status), where
    status is "none" (no detections at all inside the satellite's operating period),
    "low" (below qc.low_fraction x the median of the surrounding qc.window_days) or "ok".
    This table is the base of the Phase 2 "no data is not no fire" mask;
  * gap runs: consecutive "none"/"low" days, longest first;
  * yearly and monthly totals, the Terra:Aqua ratio and the VIIRS:MODIS ratio;
  * before/after checks for possible processing changes (qc.step_checks).

Outputs go to processed/<dataset>/qc/. Nothing in the record is changed.

    python -m shift.run --dataset global --stage qc
"""
from __future__ import annotations

import pandas as pd

from .config import Settings

COLS = ["acq_date", "satellite", "n_detections", "frp_sum_mw"]
MODIS_SATS = {"Terra", "Aqua"}


def daily_totals(settings: Settings) -> pd.DataFrame:
    root = settings.celldays_dir
    parts = []
    for part in sorted(root.glob("sensor_family=*/year=*/month=*")) if root.exists() else []:
        family = part.parent.parent.name.split("=", 1)[1]
        d = pd.read_parquet(part, columns=COLS)
        g = d.groupby(["acq_date", "satellite"], observed=True).agg(
            detections=("n_detections", "sum"), cells=("n_detections", "size"),
            frp_mw=("frp_sum_mw", "sum")).reset_index()
        g["sensor_family"] = family
        parts.append(g)
    if not parts:
        return pd.DataFrame(columns=["acq_date", "satellite", "detections", "cells", "frp_mw", "sensor_family"])
    out = pd.concat(parts, ignore_index=True)
    # A day can be split over two partitions only if it came from two sources; add them up.
    return (out.groupby(["acq_date", "satellite", "sensor_family"], as_index=False)
               [["detections", "cells", "frp_mw"]].sum())


def coverage(daily: pd.DataFrame, window: int, low_fraction: float) -> pd.DataFrame:
    rows = []
    for sat, d in daily.groupby("satellite"):
        d = d.set_index("acq_date").sort_index()
        full = pd.date_range(d.index.min(), d.index.max(), freq="D")
        d = d.reindex(full)
        d["satellite"] = sat
        d["sensor_family"] = d["sensor_family"].ffill().bfill()
        d[["detections", "cells", "frp_mw"]] = d[["detections", "cells", "frp_mw"]].fillna(0)
        med = d["detections"].rolling(window, center=True, min_periods=max(5, window // 3)).median()
        status = pd.Series("ok", index=d.index)
        status[d["detections"] < low_fraction * med] = "low"
        status[d["detections"] == 0] = "none"
        d["rolling_median"] = med
        d["status"] = status
        rows.append(d.rename_axis("acq_date").reset_index())
    return pd.concat(rows, ignore_index=True) if rows else daily.assign(status=[])


def gap_runs(cov: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sat, d in cov.groupby("satellite"):
        d = d.sort_values("acq_date")
        bad = d["status"] != "ok"
        run_id = (bad != bad.shift()).cumsum()
        for _, r in d[bad].groupby(run_id[bad]):
            out.append({"satellite": sat, "start": r["acq_date"].min().date().isoformat(),
                        "end": r["acq_date"].max().date().isoformat(), "days": len(r),
                        "days_none": int((r["status"] == "none").sum()),
                        "detections_in_gap": int(r["detections"].sum())})
    g = pd.DataFrame(out, columns=["satellite", "start", "end", "days", "days_none", "detections_in_gap"])
    return g.sort_values(["days", "start"], ascending=[False, True]).reset_index(drop=True)


def yearly_table(cov: pd.DataFrame) -> pd.DataFrame:
    c = cov.assign(year=cov["acq_date"].dt.year)
    y = c.pivot_table(index="year", columns="satellite", values="detections", aggfunc="sum").fillna(0)
    days_ok = c[c["status"] == "ok"].pivot_table(index="year", columns="satellite", values="detections",
                                                  aggfunc="size").fillna(0)
    out = pd.DataFrame(index=y.index)
    for sat in y.columns:
        out[f"{sat}_detections"] = y[sat].astype("int64")
        out[f"{sat}_days_ok"] = days_ok.get(sat, pd.Series(0, index=y.index)).reindex(y.index).fillna(0).astype(int)
    if {"Terra", "Aqua"} <= set(y.columns):
        out["terra_to_aqua"] = (y["Terra"] / y["Aqua"].where(y["Aqua"] > 0)).round(3)
    modis = y[[s for s in y.columns if s in MODIS_SATS]].sum(axis=1)
    if "S-NPP" in y.columns:
        out["viirs_to_modis"] = (y["S-NPP"] / modis.where(modis > 0)).where(y["S-NPP"] > 0).round(3)
    if {"S-NPP", "NOAA-20"} <= set(y.columns):
        # Two copies of the same VIIRS instrument: the ratio should sit near a steady value.
        out["noaa20_to_snpp"] = (y["NOAA-20"] / y["S-NPP"].where(y["S-NPP"] > 0)).where(y["NOAA-20"] > 0).round(3)
    return out.reset_index()


def step_check(cov: pd.DataFrame, family: str, at: str, years_each_side: int) -> dict:
    """Compare the family's share of detections against VIIRS (or its own level when no
    VIIRS) for the same calendar months before and after a date."""
    at = pd.Timestamp(at)
    c = cov[cov["status"] == "ok"].assign(month=cov["acq_date"].dt.to_period("M"))
    fam = c[c["sensor_family"] == family].groupby("month")["detections"].sum()
    ref = c[c["satellite"] == "S-NPP"].groupby("month")["detections"].sum()
    before = (at - pd.DateOffset(years=years_each_side)).to_period("M"), (at - pd.DateOffset(days=1)).to_period("M")
    after = at.to_period("M"), (at + pd.DateOffset(years=years_each_side) - pd.DateOffset(days=1)).to_period("M")
    ratio = (fam / ref).dropna()

    def window(s, lo_hi):
        return s[(s.index >= lo_hi[0]) & (s.index <= lo_hi[1])]
    rb, ra = window(ratio, before), window(ratio, after)
    if rb.empty or ra.empty:
        return {"family": family, "at": at.date().isoformat(), "note": "not enough data on both sides"}
    change = (ra.median() / rb.median() - 1) * 100
    return {"family": family, "at": at.date().isoformat(), "measure": f"{family} detections per VIIRS detection, monthly",
            "median_before": round(float(rb.median()), 4), "median_after": round(float(ra.median()), 4),
            "change_pct": round(float(change), 1), "months_before": int(len(rb)), "months_after": int(len(ra))}


def mask_overlap(settings: Settings) -> pd.DataFrame | None:
    """Share of kept fire detections per satellite and year that sit inside cells of the
    static-source mask. These are real (labelled vegetation) fires near gas flares,
    industry or volcanoes: what the mask would remove if applied to the whole record.
    Reported per satellite, because the VIIRS family holds two satellites from 2018."""
    from .static_mask import masked_cells
    cells = masked_cells(settings)
    root = settings.celldays_dir
    if cells is None or not root.exists():
        return None
    rows = []
    for part in sorted(root.glob("sensor_family=*/year=*/month=*")):
        family = part.parent.parent.name.split("=", 1)[1]
        year = int(part.parent.name.split("=", 1)[1])
        d = pd.read_parquet(part, columns=["cell_id", "satellite", "n_detections", "frp_sum_mw"])
        d["in_mask"] = d["n_detections"].where(d["cell_id"].isin(cells), 0)
        d["frp_in_mask"] = d["frp_sum_mw"].where(d["cell_id"].isin(cells), 0.0)
        g = d.groupby("satellite", observed=True)[["n_detections", "in_mask", "frp_sum_mw", "frp_in_mask"]].sum()
        for sat, r in g.iterrows():
            rows.append({"family": family, "satellite": sat, "year": year,
                         "detections": int(r["n_detections"]), "in_mask": int(r["in_mask"]),
                         "frp": float(r["frp_sum_mw"]), "frp_in_mask": float(r["frp_in_mask"])})
    t = pd.DataFrame(rows).groupby(["family", "satellite", "year"], as_index=False).sum()
    t["in_mask_pct"] = (100 * t["in_mask"] / t["detections"]).round(2)
    t["frp_in_mask_pct"] = (100 * t["frp_in_mask"] / t["frp"]).round(2)
    return t


def run_qc(settings: Settings) -> dict:
    cfg = settings["qc"]
    daily = daily_totals(settings)
    if daily.empty:
        return {"stage": "qc", "dataset": settings.dataset, "note": "no cell-days found"}
    cov = coverage(daily, int(cfg["window_days"]), float(cfg["low_fraction"]))
    gaps = gap_runs(cov)
    years = yearly_table(cov)
    monthly = (cov.assign(month=cov["acq_date"].dt.to_period("M").astype(str))
                  .pivot_table(index="month", columns="satellite", values="detections", aggfunc="sum").fillna(0)
                  .astype("int64").reset_index())
    steps = [step_check(cov, s["family"], s["at"], int(s["years_each_side"])) for s in cfg.get("step_checks", [])]

    out = settings.processed_dir / "qc"
    out.mkdir(parents=True, exist_ok=True)
    cov.to_parquet(out / "daily_coverage.parquet", index=False)
    gaps.to_csv(out / "gaps.csv", index=False)
    years.to_csv(out / "yearly.csv", index=False)
    monthly.to_csv(out / "monthly.csv", index=False)

    overlap = mask_overlap(settings)
    if overlap is not None:
        overlap.to_csv(out / "mask_overlap.csv", index=False)

    status_counts = cov.groupby(["satellite", "status"]).size().unstack(fill_value=0)
    report = {
        "stage": "qc", "dataset": settings.dataset,
        "date_range": [cov["acq_date"].min().date().isoformat(), cov["acq_date"].max().date().isoformat()],
        "days_by_status": {s: {k: int(v) for k, v in r.items()} for s, r in status_counts.iterrows()},
        "longest_gaps": gaps.head(15).to_dict("records"),
        "step_checks": steps,
        "real_fires_in_mask_pct": (None if overlap is None else
                                   {f: round(100 * g["in_mask"].sum() / g["detections"].sum(), 2)
                                    for f, g in overlap.groupby("satellite")}),
        "outputs": [str(p.relative_to(settings.processed_dir)) for p in sorted(out.iterdir())],
    }
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("[qc] yearly totals and ratios:\n" + years.to_string(index=False), flush=True)
        print("[qc] longest gaps:\n" + (gaps.head(15).to_string(index=False) if len(gaps) else "none"), flush=True)
    print("[qc] step checks:", steps, flush=True)
    return report
