"""Phase 2 must-fix 6 and option F: how fine in TIME can the MODIS years be trusted, and should
a day be the UTC day or the local solar day? (stage `daily`, after resolution, before harmonise)

The factors are tested on MONTH totals. The calendar also shows days and weeks. Here the final
model (stage resolution: variant, fit years, region choices) is fitted on the later overlap years
of the MODIS piece and used to predict S-NPP day by day in the earlier ones (backward test, out
of sample), on the tested 2.5-degree cells and the D18 regions, at three time scales: day, week
(Monday to Sunday) and calendar month. Days when a satellite was not fully working, and zone-days
judged data loss (D25), are left out for both sides.

Measures per cell or region, then fire-weighted (S-NPP burning cells) over cells:
  error        median of |predicted - real| / max(real, 1) over periods with fire on either side
  correlation  Pearson correlation of log(1 + value), all periods
  detection    share of periods with fire on either side that have fire on both

RULES FIXED 15 OCT 2026 BEFORE THE RUN (settings daily.rule):
  LABELS     a time scale counts as TESTED for the MODIS years if the fire-weighted error over the
             tested cells is at most error_limit_pct for both variables; finer scales are shown
             as "indicative" before 20 Jan 2012.
  DAY BASIS  the local solar day replaces the UTC day only if the daily error falls by more than
             gain_points for both variables and the daily correlation does not fall for either.

    python -m shift.run --dataset global --stage daily      (after resolution; ~10-20 min)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import Settings
from .counts import GROUPS, source_segments
from .coverage import bad_days
from .factors import Cube, data_loss_keys, final_mask, zone_key
from .harmonise import check_regions
from .region import in_bbox
from .resolution import k_model, model_spec
from .solar import local_day_shift
from .static_mask import masked_cells

COLS = ["acq_date", "cell_id", "cell_lat", "cell_lon", "satellite", "frp_sum_mw", "n_day", "n_night"]
SCALES = ("day", "week", "month")


def wmedian(x, w) -> float | None:
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = np.isfinite(x) & (w > 0)
    if not ok.any():
        return None
    o = np.argsort(x[ok])
    xs, ws = x[ok][o], w[ok][o]
    return float(xs[np.searchsorted(np.cumsum(ws), ws.sum() / 2)])


def unit_metrics(p: pd.DataFrame) -> pd.DataFrame:
    """p: unit, period, pred, real (one variable). Metrics per unit."""
    p = p.assign(lp=np.log1p(p["pred"]), lr=np.log1p(p["real"]))
    fire = (p["pred"] > 0) | (p["real"] > 0)
    e = p[fire].assign(err=lambda x: (x["pred"] - x["real"]).abs() / np.maximum(x["real"], 1.0),
                       both=lambda x: (x["pred"] > 0) & (x["real"] > 0))
    err = e.groupby("unit")["err"].median() * 100
    det = e.groupby("unit")["both"].mean() * 100

    def corr(g):
        return g["lp"].corr(g["lr"]) if g["lp"].std() > 0 and g["lr"].std() > 0 else np.nan
    cor = p.groupby("unit")[["lp", "lr"]].apply(corr)
    return pd.DataFrame({"error_pct": err, "corr": cor, "detection_pct": det, "real_total": p.groupby("unit")["real"].sum()})


def period_of(dates: pd.Series, scale: str) -> pd.Series:
    if scale == "day":
        return dates
    if scale == "week":
        return dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")
    return dates.dt.to_period("M").dt.start_time


def run_daily(settings: Settings) -> dict:
    cfg = settings.raw.get("daily") or {}
    fcfg = settings.raw.get("factors") or {}
    rule = cfg.get("rule") or {}
    ref = fcfg.get("reference", "S-NPP")
    parent = float(fcfg.get("tile_deg", 10))
    prior = {"cells": float(fcfg.get("prior_cells", 30)), "frp": float(fcfg.get("prior_frp_mw", 300))}
    min_v = {"cells": float(fcfg.get("min_cells_per_month", 30)), "frp": float(fcfg.get("min_frp_mw_per_month", 300))}
    yrs = lambda v: list(range(int(v[0]), int(v[1]) + 1))  # noqa: E731
    overlap = yrs(fcfg.get("overlap_years", [2012, 2025]))
    test_years, fit_years = yrs(cfg.get("test_years", [2012, 2015])), yrs(cfg.get("fit_years", [2016, 2019]))
    hours = cfg.get("overpass_hours")
    fp = settings.processed_dir / "counts" / "fine_monthly.parquet"
    if not fp.exists():
        return {"stage": "daily", "note": "no counts/fine_monthly.parquet: run stage resolution first"}
    spec = model_spec(settings)
    fine = spec["fine_deg"]
    segments = source_segments(settings.raw.get("counts") or {})
    g, y0, y1 = segments[0]
    m = pd.read_parquet(fp)
    all_years = list(range(int(m["year"].min()), int(m["year"].max()) + 1))
    cubes = {v: Cube(m, v, True, False, parent, all_years, fine) for v in ("cells", "frp")}
    wts = np.array([1.0 if (y in fit_years and int(y0) <= y <= int(y1)) else 0.0 for y in all_years])
    k = {v: k_model(cubes[v], g, ref, wts, spec, prior[v], min_v[v]) for v in ("cells", "frp")}
    keys = cubes["cells"].keys
    nlon = int(round(360 / fine))
    fkey = lambda lat, lon: (np.floor((np.asarray(lat, float) + 90) / fine).astype("int64") * nlon  # noqa: E731
                             + np.floor((np.asarray(lon, float) + 180) / fine).astype("int64"))
    kidx = np.full(int(round(180 / fine)) * nlon, -1, dtype="int64")
    kidx[fkey(keys["lat0"] + fine / 2, keys["lon0"] + fine / 2)] = np.arange(len(keys))
    r = m[(m["group"] == ref) & m["year"].isin(overlap) & ~m["in_mask"]]
    r = r.assign(cell=fkey(r["lat0"] + fine / 2, r["lon0"] + fine / 2))
    per_cell = r.groupby("cell")["active_cells"].sum() / (12 * len(overlap))
    tested = set(per_cell[per_cell >= float((settings.raw.get("resolution") or {}).get("min_ref_cells_per_month", 10))].index)
    regions = {k_: v for k_, v in check_regions(settings).items() if not k_.startswith("event:")}

    sats = sorted({s for gg in (g, ref) for s in GROUPS[gg]})
    bad = bad_days(settings, sats) if fcfg.get("exclude_gap_days", True) else None
    bad = set(pd.DatetimeIndex(bad).normalize()) if bad is not None else set()
    loss = data_loss_keys(settings)
    loss = set(loss.tolist()) if loss is not None else set()
    mask = final_mask(settings, masked_cells(settings), parent)
    mask = np.asarray(mask if mask is not None else [], dtype="int64")
    root = settings.celldays_dir
    cells_rows, reg_rows, shift_rows = {b: [] for b in ("utc", "local")}, {b: [] for b in ("utc", "local")}, []
    print(f"[daily] backward test: fit {fit_years[0]}-{fit_years[-1]}, predict {test_years[0]}-{test_years[-1]} "
          f"day by day; {len(tested)} tested cells", flush=True)
    for y in test_years:
        for mo in range(1, 13):
            parts = [root / f"sensor_family={f}" / f"year={y}" / f"month={mo}" for f in ("MODIS", "VIIRS")]
            parts = [p for p in parts if p.exists()]
            if not parts:
                continue
            d = pd.concat([pd.read_parquet(p, columns=COLS) for p in parts], ignore_index=True)
            d["satellite"] = d["satellite"].astype(str)
            d = d[d["satellite"].isin(sats)]
            if len(mask):
                d = d[~np.isin(d["cell_id"].to_numpy("int64"), mask)]
            d["acq_date"] = pd.to_datetime(d["acq_date"]).dt.normalize()
            d = d[~d["acq_date"].isin(bad)]
            if loss:
                key = d["acq_date"].to_numpy().astype("datetime64[D]").astype("int64") * 100 + zone_key(d["cell_lat"], d["cell_lon"])
                d = d[~np.isin(key, list(loss))]
            if d.empty:
                continue
            sh = local_day_shift(d["cell_lon"], d["satellite"], d["n_day"], d["n_night"], hours)
            night = (d["n_night"] > d["n_day"]).to_numpy()
            shift_rows.append(pd.DataFrame({"satellite": d["satellite"].to_numpy(), "night": night, "moved": sh != 0})
                              .groupby(["satellite", "night"]).agg(cell_days=("moved", "size"), moved=("moved", "sum")).reset_index())
            for basis in ("utc", "local"):
                x = d.assign(date=d["acq_date"] + pd.to_timedelta(sh if basis == "local" else 0, unit="D"))
                out = []
                for grp, kind in ((g, "pred"), (ref, "real")):
                    gg = x[x["satellite"].isin(GROUPS[grp])]
                    gg = gg.groupby(["date", "cell_id"], as_index=False).agg(
                        cell_lat=("cell_lat", "first"), cell_lon=("cell_lon", "first"), frp=("frp_sum_mw", "sum"))
                    fk = fkey(gg["cell_lat"], gg["cell_lon"])
                    if kind == "pred":
                        ki = kidx[fk]
                        mi = gg["date"].dt.month.to_numpy() - 1
                        ok = ki >= 0
                        c_ = np.where(ok, k["cells"][np.maximum(ki, 0), mi], np.nan)
                        f_ = np.where(ok, k["frp"][np.maximum(ki, 0), mi], np.nan)
                        gg = gg.assign(cells=c_, frp=gg["frp"] * f_)[ok]
                        fk = fk[ok]
                    else:
                        gg = gg.assign(cells=1.0)
                    gg = gg.assign(unit=fk, kind=kind)
                    out.append(gg)
                x = pd.concat(out, ignore_index=True)
                c = x[x["unit"].isin(tested)].pivot_table(index=["unit", "date"], columns="kind", values=["cells", "frp"],
                                                         aggfunc="sum", fill_value=0.0)
                c.columns = [f"{a}_{b}" for a, b in c.columns]
                cells_rows[basis].append(c.reset_index())
                for name, bb in regions.items():
                    mm = np.ones(len(x), bool) if name == "global" else in_bbox(bb, x["cell_lon"], x["cell_lat"])
                    if mm.any():
                        rr = x[mm].pivot_table(index="date", columns="kind", values=["cells", "frp"], aggfunc="sum", fill_value=0.0)
                        rr.columns = [f"{a}_{b}" for a, b in rr.columns]
                        reg_rows[basis].append(rr.reset_index().assign(unit=name))
        print(f"[daily] {y} read", flush=True)

    # the days every unit is judged on: test-year days not left out; a local day needs its UTC day and both neighbours
    days = pd.date_range(f"{test_years[0]}-01-01", f"{test_years[-1]}-12-31")
    ok_utc = days[~days.isin(list(bad))]
    ok_local = ok_utc[ok_utc.isin(ok_utc - pd.Timedelta(days=1)) & ok_utc.isin(ok_utc + pd.Timedelta(days=1))]
    valid = {"utc": ok_utc, "local": ok_local}
    vals = ["cells_pred", "cells_real", "frp_pred", "frp_real"]
    summary, per_region = [], []
    for basis in ("utc", "local"):
        for unit_type, rows in (("cell", cells_rows[basis]), ("region", reg_rows[basis])):
            if not rows:
                continue
            x = pd.concat(rows, ignore_index=True).reindex(columns=["unit", "date", *vals]).fillna(0.0)
            x = x.groupby(["unit", "date"], as_index=False)[vals].sum()
            units = x["unit"].unique()
            full = pd.MultiIndex.from_product([units, valid[basis]], names=["unit", "date"])
            x = x.set_index(["unit", "date"]).reindex(full, fill_value=0.0).reset_index()
            w = x.groupby("unit")["cells_real"].sum()
            if unit_type == "region":
                # every region counts once (the world's daily total would otherwise outweigh them all);
                # the world is reported on its own line
                w = pd.Series(np.where(w.index == "global", 0.0, 1.0), index=w.index)
            for scale in SCALES:
                xs = x.assign(period=period_of(x["date"], scale)).groupby(["unit", "period"], as_index=False)[vals].sum()
                for var in ("cells", "frp"):
                    um = unit_metrics(xs.rename(columns={f"{var}_pred": "pred", f"{var}_real": "real"})[["unit", "period", "pred", "real"]])
                    ww = w.reindex(um.index).fillna(0)
                    summary.append({"basis": basis, "units": unit_type, "scale": scale, "variable": var, "n": int((ww > 0).sum()),
                                    "fw_median_error_pct": None if wmedian(um["error_pct"], ww) is None else round(wmedian(um["error_pct"], ww), 1),
                                    "fw_mean_corr": round(float(np.average(um["corr"].fillna(0), weights=ww)), 3) if ww.sum() else None,
                                    "fw_detection_pct": round(float(np.average(um["detection_pct"].fillna(0), weights=ww)), 1) if ww.sum() else None})
                    if unit_type == "region":
                        per_region.append(um.reset_index().assign(basis=basis, scale=scale, variable=var))
                        if "global" in um.index:
                            gm = um.loc["global"]
                            summary.append({"basis": basis, "units": "world", "scale": scale, "variable": var, "n": 1,
                                            "fw_median_error_pct": round(float(gm["error_pct"]), 1),
                                            "fw_mean_corr": round(float(gm["corr"]), 3), "fw_detection_pct": round(float(gm["detection_pct"]), 1)})
    sm = pd.DataFrame(summary)
    out = settings.processed_dir / "daily"
    out.mkdir(parents=True, exist_ok=True)
    sm.to_csv(out / "skill_by_scale.csv", index=False)
    if per_region:
        pd.concat(per_region, ignore_index=True).round(3).to_csv(out / "skill_regions.csv", index=False)
    shifts = pd.concat(shift_rows, ignore_index=True).groupby(["satellite", "night"], as_index=False).sum() if shift_rows else pd.DataFrame()
    if len(shifts):
        shifts["moved_pct"] = (100 * shifts["moved"] / shifts["cell_days"]).round(2)
        shifts.to_csv(out / "date_shift_share.csv", index=False)

    lim, gain = float(rule.get("error_limit_pct", 25)), float(rule.get("gain_points", 1))
    get = lambda b, u, sc, v, col: sm[(sm["basis"] == b) & (sm["units"] == u) & (sm["scale"] == sc) & (sm["variable"] == v)][col].iloc[0]  # noqa: E731
    labels = {}
    for sc in SCALES:
        try:
            ok = all(get("utc", "cell", sc, v, "fw_median_error_pct") is not None
                     and get("utc", "cell", sc, v, "fw_median_error_pct") <= lim for v in ("cells", "frp"))
        except IndexError:
            ok = False
        labels[sc] = "tested" if ok else "indicative"
    finest = next((sc for sc in SCALES if labels[sc] == "tested"), None)
    try:
        better = all(get("utc", "cell", "day", v, "fw_median_error_pct") - get("local", "cell", "day", v, "fw_median_error_pct") > gain
                     and get("local", "cell", "day", v, "fw_mean_corr") >= get("utc", "cell", "day", v, "fw_mean_corr")
                     for v in ("cells", "frp"))
    except (IndexError, TypeError):
        better = False
    basis = "local" if better else "utc"
    (out / "day_basis.json").write_text(json.dumps({"basis": basis}))
    (out / "label_rule.json").write_text(json.dumps({"labels_before_2012": labels, "finest_tested_scale": finest,
                                                     "error_limit_pct": lim}))
    report = {"stage": "daily", "test_years": [test_years[0], test_years[-1]], "fit_years": [fit_years[0], fit_years[-1]],
              "tested_cells": len(tested), "labels_before_2012": labels, "finest_tested_scale": finest,
              "day_basis": basis, "skill": sm.to_dict("records"),
              "date_shift_share": shifts.to_dict("records") if len(shifts) else []}
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 80):
        print("[daily] skill of MODIS x k against S-NPP by time scale (out of sample, fire-weighted):\n"
              + sm.to_string(index=False), flush=True)
        if len(shifts):
            print("[daily] cell-days whose local solar date differs from the UTC date:\n" + shifts.to_string(index=False), flush=True)
        print(f"[daily] LABEL RULE (error <= {lim:g}% for both variables): {labels}; finest tested scale: {finest}", flush=True)
        print(f"[daily] DAY BASIS RULE: {basis}", flush=True)
    return report
