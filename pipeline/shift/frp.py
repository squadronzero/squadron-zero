"""Phase 2, fix 4: the intercept problem in fire power (FRP) harmonisation.

v1 calibrated fire power with one straight line fitted to Florida cell-days where both
sensors saw fire:  VIIRS_FRP = 0.316 x MODIS_FRP + 24.7 MW.
Two things go wrong when that line is used to rebuild a record:

  * The intercept. Every MODIS cell-day gets +24.7 MW whatever it saw, so a region's total
    depends on how many cells burned more than on how much they burned, and a place with
    many tiny fires is inflated.
  * Missing fires. The line is fitted only where BOTH sensors saw fire. VIIRS also sees many
    small fires MODIS misses; those never enter the fit, so the line cannot put them back.

Methods compared (each predicts a region-month VIIRS S-NPP fire power total from MODIS,
with the D19 source pieces: merged MODIS to 2019, Aqua from 2020):

  v1_equation     the v1 Florida line, unchanged, applied everywhere to merged Terra + Aqua
                  (the input it was fitted on), in every year
  ols_cellday     a straight line with intercept, refitted per region on paired cell-days
  origin_cellday  a line through zero (no intercept), per region, paired cell-days
  ratio_paired    total VIIRS / total MODIS fire power over paired cell-days, per region
  ratio_totals    total VIIRS / total MODIS fire power over ALL cell-days, per region and
                  calendar month (the fix 1 method applied to fire power)

The same tests as fix 3b (leave-one-year-out, backward, forward, category agreement,
false trend) and a pre-registered rule (settings `frp.rule`, decision D21) pick the method.

    python -m shift.run --dataset global --stage frp        (after `counts`)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings
from .counts import GLOBAL, GROUPS, _monthly, source_segments
from .coverage import bad_days
from .region import in_bbox, test_regions
from .selection import baseline, drift, pick_winner, predict, score

METHODS = ["v1_equation", "ols_cellday", "origin_cellday", "ratio_paired", "ratio_totals"]
LINEAR = {"v1_equation", "ols_cellday", "origin_cellday", "ratio_paired"}
STAT_COLS = ["n", "sx", "sy", "sxx", "sxy"]


def pair_stats(settings: Settings, groups: list[str], ref: str, years: list[int],
               exclude: pd.DatetimeIndex | None) -> pd.DataFrame:
    """Sums for a straight-line fit (n, sum x, sum y, sum x^2, sum xy) of reference cell-day
    fire power (y) on source cell-day fire power (x), over cell-days where both saw fire,
    per region, source group, year and month. Read one month at a time."""
    regions = {"global": GLOBAL, **test_regions(settings)}
    root = settings.celldays_dir
    cols = ["acq_date", "cell_id", "cell_lat", "cell_lon", "satellite", "frp_sum_mw"]
    rows = []
    for year in years:
        for month in range(1, 13):
            m = root / "sensor_family=MODIS" / f"year={year}" / f"month={month}"
            v = root / "sensor_family=VIIRS" / f"year={year}" / f"month={month}"
            if not (m.exists() and v.exists()):
                continue
            dm = pd.read_parquet(m, columns=cols)
            dv = pd.read_parquet(v, columns=cols)
            dv = dv[dv["satellite"].astype(str).isin(GROUPS[ref])]
            if exclude is not None and len(exclude):
                dm = dm[~pd.to_datetime(dm["acq_date"]).isin(exclude)]
                dv = dv[~pd.to_datetime(dv["acq_date"]).isin(exclude)]
            y = dv.groupby(["acq_date", "cell_id"], as_index=False)["frp_sum_mw"].sum().rename(columns={"frp_sum_mw": "y"})
            for g in groups:
                src = dm[dm["satellite"].astype(str).isin(GROUPS[g])]
                if src.empty:
                    continue
                x = (src.groupby(["acq_date", "cell_id"], as_index=False)
                        .agg(cell_lat=("cell_lat", "first"), cell_lon=("cell_lon", "first"), x=("frp_sum_mw", "sum")))
                p = x.merge(y, on=["acq_date", "cell_id"], how="inner")
                for name, reg in regions.items():
                    r = p if name == "global" else p[in_bbox(reg["bbox"], p["cell_lon"], p["cell_lat"])]
                    if r.empty:
                        continue
                    xv, yv = r["x"].to_numpy(float), r["y"].to_numpy(float)
                    rows.append({"region": name, "group": g, "year": year, "month": month, "n": len(r),
                                 "sx": xv.sum(), "sy": yv.sum(), "sxx": (xv * xv).sum(), "sxy": (xv * yv).sum()})
    return pd.DataFrame(rows, columns=["region", "group", "year", "month"] + STAT_COLS)


def fit_line(stats: pd.DataFrame, group: str, years: list[int], method: str, v1: tuple[float, float]) -> pd.DataFrame:
    """Slope a and intercept b per region for a linear method, from the summed statistics."""
    s = stats[(stats["group"] == group) & stats["year"].isin(years)].groupby("region")[STAT_COLS].sum()
    if method == "v1_equation":
        return pd.DataFrame({"a": v1[0], "b": v1[1]}, index=s.index if len(s) else pd.Index([], name="region"))
    if method == "ols_cellday":
        den = s["n"] * s["sxx"] - s["sx"] ** 2
        a = (s["n"] * s["sxy"] - s["sx"] * s["sy"]) / den.where(den > 0)
        return pd.DataFrame({"a": a, "b": (s["sy"] - a * s["sx"]) / s["n"]})
    if method == "origin_cellday":
        return pd.DataFrame({"a": s["sxy"] / s["sxx"].where(s["sxx"] > 0), "b": 0.0})
    if method == "ratio_paired":
        return pd.DataFrame({"a": s["sy"] / s["sx"].where(s["sx"] > 0), "b": 0.0})
    raise ValueError(method)


def predict_linear(wf: pd.DataFrame, wc: pd.DataFrame, stats: pd.DataFrame, segments: list, ref: str,
                   method: str, fit_years: list[int], test_years: list[int], v1) -> pd.DataFrame:
    """Predicted region-month reference fire power = a x source fire power + b x source active cells."""
    out = []
    for group, y0, y1 in segments:
        span = set(range(int(y0), int(y1) + 1))
        fy = [y for y in fit_years if y in span] or list(fit_years)
        ty = [y for y in test_years if y in span]
        if not ty or group not in wf:
            continue
        line = fit_line(stats, group, fy, method, v1)
        t = wf[wf["year"].isin(ty)].dropna(subset=[group, ref])
        if t.empty or line.empty:
            continue
        cells = wc.set_index(["region", "year", "month"])[group].reindex(pd.MultiIndex.from_frame(t[["region", "year", "month"]])).to_numpy(float)
        a = line["a"].reindex(t["region"]).to_numpy(float)
        b = line["b"].reindex(t["region"]).to_numpy(float)
        pred = a * t[group].to_numpy(float) + b * cells
        out.append(t[["region", "year", "month"]].assign(pred=pred, real=t[ref].to_numpy(float),
                                                         from_intercept=b * cells))
    if not out:
        return pd.DataFrame(columns=["region", "year", "month", "pred", "real", "from_intercept"])
    return pd.concat(out, ignore_index=True).dropna(subset=["pred"])


def run_frp(settings: Settings) -> dict:
    cfg = settings.raw.get("frp") or {}
    ref = cfg.get("reference", "S-NPP")
    segments = source_segments(settings.raw.get("counts") or {})
    v1 = tuple(cfg.get("v1_equation", [0.316162, 24.701681]))
    min_frp = float(cfg.get("min_frp_mw_per_month", 300))
    span = lambda key, d: list(range(int(cfg.get(key, d)[0]), int(cfg.get(key, d)[1]) + 1))  # noqa: E731
    years = span("overlap_years", [2012, 2025])
    bw_fit, bw_test = span("backward_fit_years", [2016, 2025]), span("backward_test_years", [2012, 2015])
    fw_fit, fw_test = span("forward_fit_years", [2012, 2021]), span("forward_test_years", [2022, 2025])
    rule = cfg.get("rule") or {}

    out = settings.processed_dir / "counts"
    path = out / "daily.parquet"
    if not path.exists():
        return {"stage": "frp", "note": "no counts/daily.parquet: run the counts stage first"}
    daily = pd.read_parquet(path)
    daily["acq_date"] = pd.to_datetime(daily["acq_date"])
    if "frp_mw" not in daily.columns:
        return {"stage": "frp", "note": "daily.parquet has no fire power column: rerun the counts stage"}
    groups = sorted({g for g, *_ in segments})
    sats = sorted({s for g in groups for s in GROUPS[g]} | set(GROUPS[ref]))
    bad = bad_days(settings, sats) if cfg.get("exclude_gap_days", True) else None

    d = daily if bad is None or len(bad) == 0 else daily[~daily["acq_date"].isin(bad)]
    d = d.assign(year=d["acq_date"].dt.year, month=d["acq_date"].dt.month)
    m = d.groupby(["region", "group", "year", "month"], as_index=False)[["active_cells", "frp_mw"]].sum()
    wf = m.pivot_table(index=["region", "year", "month"], columns="group", values="frp_mw", aggfunc="sum").reset_index()
    wc = m.pivot_table(index=["region", "year", "month"], columns="group", values="active_cells", aggfunc="sum").reset_index()
    base = baseline(wf, ref, years, [])

    stats = pair_stats(settings, groups, ref, years, bad)

    # The v1 line was fitted on Terra + Aqua merged, so it is always applied to merged MODIS
    # (its own input). Feeding it Aqua alone after 2019 would halve its input and be unfair.
    v1_segments = [["MODIS", 1900, 2100]]

    def run(method, fit, test):
        if method == "ratio_totals":
            return predict(wf, segments, ref, fit, test, min_frp, [])
        segs = v1_segments if method == "v1_equation" else segments
        return predict_linear(wf, wc, stats, segs, ref, method, fit, test, v1)

    scores, drifts, intercept_share = [], [], []
    for method in METHODS:
        lo = [run(method, [x for x in years if x != y], [y]) for y in years]
        lo = pd.concat([p for p in lo if len(p)], ignore_index=True) if any(len(p) for p in lo) else pd.DataFrame()
        for test, p in (("leave_one_year_out", lo), ("backward", run(method, bw_fit, bw_test)),
                        ("forward", run(method, fw_fit, fw_test))):
            sc = score(p, base, [])
            if len(sc):
                scores.append(sc.assign(method=method, test=test))
        if len(lo):
            drifts.append(drift(lo, []).assign(method=method))
            if "from_intercept" in lo:
                share = lo.groupby("region")[["from_intercept", "pred"]].sum()
                intercept_share.append((100 * share["from_intercept"] / share["pred"]).round(1).rename(method))
    scores = pd.concat([x for x in scores if len(x) and not x.isna().all().all()] or scores, ignore_index=True)
    drifts = pd.concat(drifts, ignore_index=True)
    ishare = pd.concat(intercept_share, axis=1) if intercept_share else pd.DataFrame()

    summ = []
    for method in METHODS:
        s = scores[scores["method"] == method]
        reg = s[s["region"] != "global"]
        lo, bw = reg[reg["test"] == "leave_one_year_out"], reg[reg["test"] == "backward"]
        g_bw = s[(s["region"] == "global") & (s["test"] == "backward")]
        g_dr = drifts[(drifts["method"] == method) & (drifts["region"] == "global")]
        a_lo, a_bw = lo["agreement_pct"].mean(), bw["agreement_pct"].mean()
        summ.append({"candidate": method,
                     "score": round(float(np.nanmean([a_lo, a_bw])), 2),
                     "agreement_loyo_pct": round(float(a_lo), 1), "agreement_backward_pct": round(float(a_bw), 1),
                     "high_caught_loyo_pct": round(float(lo["high_caught_pct"].mean()), 1),
                     "mean_abs_error_loyo_pct": round(float(lo["total_error_pct"].abs().mean()), 2),
                     "mean_abs_error_backward_pct": round(float(bw["total_error_pct"].abs().mean()), 2),
                     "global_backward_error_pct": float(g_bw["total_error_pct"].iloc[0]) if len(g_bw) else np.nan,
                     "global_drift_pct_per_year": float(g_dr["drift_pct_per_year"].iloc[0]) if len(g_dr) else np.nan})
    summary = pd.DataFrame(summ)
    winner, why = pick_winner(summary, rule)
    summary["eligible"] = summary["global_backward_error_pct"].abs() <= float(rule.get("max_global_backward_error_pct", 10))

    # Lines fitted on all overlap years, for the record (what the intercept looks like per region).
    lines = []
    for group, y0, y1 in segments:
        fy = [y for y in years if int(y0) <= y <= int(y1)]
        if not fy:
            continue
        for method in ["ols_cellday", "origin_cellday", "ratio_paired"]:
            ln = fit_line(stats, group, fy, method, v1).reset_index()
            lines.append(ln.assign(segment=f"{group} {y0}-{y1}", method=method))
    lines = pd.concat(lines, ignore_index=True) if lines else pd.DataFrame()
    pairs = stats.groupby(["region", "group"])["n"].sum().unstack() if len(stats) else pd.DataFrame()

    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "frp_scores.csv", index=False)
    drifts.to_csv(out / "frp_drift.csv", index=False)
    summary.to_csv(out / "frp_summary.csv", index=False)
    stats.to_csv(out / "frp_pair_stats.csv", index=False)
    if len(lines):
        lines.to_csv(out / "frp_lines.csv", index=False)
    if len(ishare):
        ishare.to_csv(out / "frp_intercept_share.csv")

    report = {"stage": "frp", "reference": ref, "source_segments": segments, "rule": rule,
              "days_excluded": None if bad is None else int(len(bad)),
              "winner": winner, "why": why,
              "summary": summary.replace({np.nan: None}).to_dict("records"),
              "intercept_share_pct": ishare.replace({np.nan: None}).to_dict() if len(ishare) else {},
              "paired_cell_days": pairs.fillna(0).astype(int).to_dict() if len(pairs) else {}}

    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        def table(test, col):
            t = scores[scores["test"] == test].pivot(index="region", columns="method", values=col)
            return t[[c for c in METHODS if c in t]].to_string()
        print("[frp] SUMMARY (score = mean category agreement %, leave-one-year-out and backward, test regions)\n"
              + summary.to_string(index=False), flush=True)
        print(f"[frp] category agreement %, leave-one-year-out ({years[0]}-{years[-1]})\n" + table("leave_one_year_out", "agreement_pct"), flush=True)
        print(f"[frp] backward (fit {bw_fit[0]}-{bw_fit[-1]}, predict {bw_test[0]}-{bw_test[-1]}) total fire power error %\n" + table("backward", "total_error_pct"), flush=True)
        print(f"[frp] forward (fit {fw_fit[0]}-{fw_fit[-1]}, predict {fw_test[0]}-{fw_test[-1]}) total fire power error %\n" + table("forward", "total_error_pct"), flush=True)
        dt = drifts.pivot(index="region", columns="method", values="drift_pct_per_year")
        print("[frp] false trend (% per year; 0 = none)\n" + dt[[c for c in METHODS if c in dt]].to_string(), flush=True)
        if len(ishare):
            print("[frp] share of predicted fire power that comes from the intercept alone, %\n" + ishare.to_string(), flush=True)
        if len(lines):
            ols = lines[lines["method"] == "ols_cellday"].pivot(index="region", columns="segment", values=["a", "b"]).round(3)
            print("[frp] straight line with intercept fitted per region (a = slope, b = intercept MW)\n" + ols.to_string(), flush=True)
        print(f"[frp] WINNER by the pre-registered rule: {winner} ({why})", flush=True)
    return report
