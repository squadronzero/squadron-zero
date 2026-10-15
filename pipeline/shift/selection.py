"""Phase 2, fix 3b: choose the MODIS calibration source with fair, pre-registered tests (D19).

SHIFT's job is a fire calendar: is this month normal, unusual or critical for this place?
So the source that calibrates the MODIS years must (1) add no false jumps or trends,
(2) give the same verdict the newer VIIRS sensor would, and (3) work BACKWARDS, because
VIIRS-era factors are applied to the MODIS-only years 2002-2011.

Candidates (settings `selection.candidates`): Terra+Aqua merged, Aqua alone, Terra alone,
and merged until Terra's drift then Aqua alone. Every candidate uses the same method as
fix 1 (factor k = S-NPP active cells / source active cells per region and calendar month)
and the same left-out days.

Tests
  1. backward      fit on late overlap years, predict the early ones (how the product is used)
  2. categories    real S-NPP and predicted months put in low / normal / unusual / critical
                   against the same S-NPP baseline; how often they agree, how many real
                   unusual-or-critical months are caught, how many false alarms
  3. leave-one-year-out   each overlap year predicted from all the others
  4. MODIS-era     Aqua/Terra ratio 2003-2025 (instrument stability before VIIRS existed)
                   and the size of the mid-2002 step when Aqua joins Terra
  5. bootstrap     years resampled: a 95% range for every k
  6. fire size     tests 1-3 repeated on small-fire and large-fire cells separately

The winner is picked by the rule in `selection.rule`, written down before the first run.

    python -m shift.run --dataset global --stage selection     (after `counts`)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings
from .counts import GROUPS, _monthly, _r2, daily_series
from .coverage import bad_days

CLASSES = ["low", "normal", "unusual", "critical"]
DEFAULT_CANDIDATES = {
    "MODIS": [["MODIS", 2000, 2100]],
    "Aqua": [["Aqua", 2000, 2100]],
    "Terra": [["Terra", 2000, 2100]],
    "MODIS_then_Aqua": [["MODIS", 2000, 2019], ["Aqua", 2020, 2100]],
}


# ---------------------------------------------------------------- building blocks

def wide_table(monthly: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """One row per region (and size), year and month; one active-cells column per group."""
    idx = ["region"] + (by or []) + ["year", "month"]
    return monthly.pivot_table(index=idx, columns="group", values="active_cells", aggfunc="sum").reset_index()


def fit_k(w: pd.DataFrame, src: str, ref: str, years: list[int], min_cells: int, by: list[str]) -> pd.Series:
    """k per (region[, size], month) from the rows in `years`; thin months use the all-year k."""
    if src not in w or ref not in w:
        return pd.Series(dtype=float)
    sub = w[w["year"].isin(years)].dropna(subset=[src, ref])
    keys = ["region"] + by
    if sub.empty:
        return pd.Series(dtype=float)
    m = sub.groupby(keys + ["month"])[[src, ref]].sum()
    r = sub.groupby(keys)[[src, ref]].sum()
    k_reg = (r[ref] / r[src].where(r[src] > 0)).rename("k_region")
    m = m.join(k_reg, on=keys)
    k = (m[ref] / m[src].where(m[src] > 0)).where(m[src] >= min_cells, m["k_region"])
    return k


def predict(w: pd.DataFrame, segments: list, ref: str, fit_years: list[int], test_years: list[int],
            min_cells: int, by: list[str]) -> pd.DataFrame:
    """Predicted reference active cells for `test_years`. Each segment [group, from, to] is
    fitted on the fit years inside its own span (all fit years if it has none there)."""
    keys = ["region"] + by
    out = []
    for group, y0, y1 in segments:
        span = [y for y in range(int(y0), int(y1) + 1)]
        fy = [y for y in fit_years if y in span] or list(fit_years)
        ty = [y for y in test_years if y in span]
        if not ty or group not in w:
            continue
        k = fit_k(w, group, ref, fy, min_cells, by)
        t = w[w["year"].isin(ty)].dropna(subset=[group, ref])
        if t.empty or k.empty:
            continue
        kk = k.reindex(pd.MultiIndex.from_frame(t[keys + ["month"]])).to_numpy()
        out.append(t[keys + ["year", "month"]].assign(pred=t[group].to_numpy(float) * kk,
                                                      real=t[ref].to_numpy(float)))
    if not out:
        return pd.DataFrame(columns=keys + ["year", "month", "pred", "real"])
    return pd.concat(out, ignore_index=True).dropna(subset=["pred"])


def baseline(w: pd.DataFrame, ref: str, years: list[int], by: list[str]) -> pd.DataFrame:
    """Typical level per (region[, size], calendar month) from REAL S-NPP: mean and spread
    of log(1 + active cells). The same baseline judges real and predicted months."""
    keys = ["region"] + by + ["month"]
    sub = w[w["year"].isin(years)].dropna(subset=[ref])
    v = np.log1p(sub[ref].astype(float))
    b = sub.assign(v=v).groupby(keys)["v"].agg(mu="mean", sd="std").reset_index()
    b["sd"] = b["sd"].where(b["sd"] > 0.05, 0.05)        # never divide by ~0
    return b


def classify(values: pd.Series, mu: pd.Series, sd: pd.Series) -> pd.Series:
    z = (np.log1p(values.astype(float).clip(lower=0)) - mu) / sd
    return pd.cut(z, [-np.inf, -1, 1, 2, np.inf], labels=CLASSES, right=False).astype(str)


def score(pred: pd.DataFrame, base: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Error and category scores per (region[, size])."""
    keys = ["region"] + by
    if pred.empty:
        return pd.DataFrame(columns=keys)
    p = pred.merge(base, on=keys + ["month"], how="left")
    p["cls_real"] = classify(p["real"], p["mu"], p["sd"])
    p["cls_pred"] = classify(p["pred"], p["mu"], p["sd"])
    high = {"unusual", "critical"}
    rows = []
    for key, r in p.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        y, x = r["real"].to_numpy(float), r["pred"].to_numpy(float)
        real_high, pred_high = r["cls_real"].isin(high), r["cls_pred"].isin(high)
        rows.append({**dict(zip(keys, key)), "months": int(len(r)),
                     "total_error_pct": round(float(100 * (x.sum() - y.sum()) / y.sum()), 2) if y.sum() else None,
                     "r2": round(_r2(y, x), 4),
                     "median_abs_error_pct": round(float(100 * np.median(np.abs(x - y) / np.maximum(y, 1))), 1),
                     "agreement_pct": round(float(100 * (r["cls_real"] == r["cls_pred"]).mean()), 1),
                     "real_high_months": int(real_high.sum()),
                     "high_caught_pct": round(float(100 * pred_high[real_high].mean()), 1) if real_high.any() else None,
                     "false_alarm_pct": round(float(100 * (~real_high[pred_high]).mean()), 1) if pred_high.any() else None})
    return pd.DataFrame(rows)


def drift(pred: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Trend of the yearly real/predicted ratio, % per year. 0 = no false trend."""
    keys = ["region"] + by
    rows = []
    for key, r in pred.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        y = r.groupby("year")[["pred", "real"]].sum()
        y = y[y["pred"] > 0]
        if len(y) < 3:
            continue
        ratio = y["real"] / y["pred"]
        slope = np.polyfit(y.index.to_numpy(float), ratio.to_numpy(float), 1)[0]
        rows.append({**dict(zip(keys, key)), "drift_pct_per_year": round(float(100 * slope / ratio.mean()), 2)})
    return pd.DataFrame(rows)


def loyo(w, segments, ref, years, min_cells, by):
    """Leave-one-year-out: each year predicted from all the others."""
    parts = [predict(w, segments, ref, [x for x in years if x != y], [y], min_cells, by) for y in years]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def bootstrap_k(w, segments, ref, years, min_cells, n, seed) -> pd.DataFrame:
    """95% range of each k when the fit years are resampled (whole years at a time)."""
    rng = np.random.default_rng(seed)
    rows = []
    for group, y0, y1 in segments:
        span = [y for y in years if int(y0) <= y <= int(y1)]
        if len(span) < 3 or group not in w:
            continue
        sub = w[w["year"].isin(span)].dropna(subset=[group, ref])
        weights = rng.multinomial(len(span), [1 / len(span)] * len(span), size=n)   # n x years
        for (region, month), r in sub.groupby(["region", "month"]):
            yr = r.groupby("year")[[group, ref]].sum().reindex(span, fill_value=0)
            src, rf = weights @ yr[group].to_numpy(float), weights @ yr[ref].to_numpy(float)
            ok = src >= min_cells
            if ok.mean() < 0.5:          # thin month: the all-year k is used instead
                continue
            kb = rf[ok] / src[ok]
            k = yr[ref].sum() / yr[group].sum()
            lo, hi = np.percentile(kb, [2.5, 97.5])
            rows.append({"segment": group, "region": region, "month": month, "k": round(float(k), 4),
                         "k_lo": round(float(lo), 4), "k_hi": round(float(hi), 4),
                         "range_pct": round(float(100 * (hi - lo) / k), 1) if k else None})
    return pd.DataFrame(rows)


def modis_era(monthly: pd.DataFrame, years: list[int], stable_until: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aqua/Terra and merged/Terra yearly ratios, before VIIRS existed and after.
    A steady Aqua/Terra ratio means neither MODIS instrument was degrading."""
    m = monthly[monthly["group"].isin(["Terra", "Aqua", "MODIS"]) & monthly["year"].isin(years)]
    piv = m.pivot_table(index=["region", "year", "month"], columns="group", values="active_cells",
                        aggfunc="sum").dropna().reset_index()
    if not {"Terra", "Aqua", "MODIS"} <= set(piv.columns):
        return pd.DataFrame(), pd.DataFrame()
    y = piv.groupby(["region", "year"], as_index=False)[["Terra", "Aqua", "MODIS"]].sum()
    y["aqua_to_terra"] = (y["Aqua"] / y["Terra"]).round(4)
    y["merged_to_terra"] = (y["MODIS"] / y["Terra"]).round(4)
    rows = []
    for region, r in y.groupby("region"):
        row = {"region": region}
        for label, part in (("stable", r[r["year"] <= stable_until]), ("late", r[r["year"] > stable_until])):
            if len(part) >= 3:
                s = np.polyfit(part["year"].to_numpy(float), part["aqua_to_terra"].to_numpy(float), 1)[0]
                row[f"aqua_terra_drift_{label}_pct_per_year"] = round(float(100 * s / part["aqua_to_terra"].mean()), 2)
        first = r[r["year"].between(years[0], years[0] + 2)]
        row["step_2002_merged_vs_terra"] = round(float(first["MODIS"].sum() / first["Terra"].sum()), 3) if len(first) else None
        rows.append(row)
    return y, pd.DataFrame(rows)


def size_classes(settings: Settings, ref: str, years: list[int]) -> tuple[pd.Series, float]:
    """Label every cell 'small' or 'large' by its typical fire power as seen by the reference
    sensor (mean fire power per active cell-day over the overlap years). The label belongs to
    the PLACE, so MODIS and VIIRS are split the same way. Threshold = median over cells."""
    root = settings.celldays_dir
    acc = []
    for part in sorted(root.glob("sensor_family=*/year=*/month=*")):
        y = int(part.parent.name.split("=")[1])
        if y not in years:
            continue
        d = pd.read_parquet(part, columns=["cell_id", "satellite", "frp_sum_mw"])
        d = d[d["satellite"].astype(str).isin(GROUPS[ref])]
        if d.empty:
            continue
        acc.append(d.groupby("cell_id")["frp_sum_mw"].agg(["sum", "size"]))
        if len(acc) >= 24:
            acc = [pd.concat(acc).groupby(level=0).sum()]
    if not acc:
        return pd.Series(dtype=str), float("nan")
    t = pd.concat(acc).groupby(level=0).sum()
    mean_frp = t["sum"] / t["size"]
    thr = float(mean_frp.median())
    return pd.Series(np.where(mean_frp < thr, "small", "large"), index=mean_frp.index), thr


def pick_winner(summary: pd.DataFrame, rule: dict) -> tuple[str | None, str]:
    """Apply the pre-registered rule (settings `selection.rule`)."""
    s = summary.copy()
    limit = float(rule.get("max_global_backward_error_pct", 10))
    s["eligible"] = s["global_backward_error_pct"].abs() <= limit
    ok = s[s["eligible"] & s["score"].notna()]
    if ok.empty:
        return None, f"no candidate within {limit}% global backward error"
    best = ok["score"].max()
    margin = float(rule.get("tie_margin_points", 2))
    close = ok[ok["score"] >= best - margin]
    win = close.loc[close["global_drift_pct_per_year"].abs().idxmin(), "candidate"]
    why = (f"highest score {best:.1f}" if len(close) == 1 else
           f"{len(close)} candidates within {margin} points of the best score {best:.1f}; "
           f"lowest global drift wins")
    return str(win), why


# ---------------------------------------------------------------- the stage

def run_selection(settings: Settings) -> dict:
    cfg = settings.raw.get("selection") or {}
    ref = cfg.get("reference", "S-NPP")
    cands = cfg.get("candidates") or DEFAULT_CANDIDATES
    min_cells = int(cfg.get("min_cells_per_month", 30))
    ov = cfg.get("overlap_years", [2012, 2025])
    years = list(range(int(ov[0]), int(ov[1]) + 1))
    span = lambda key, d: list(range(int(cfg.get(key, d)[0]), int(cfg.get(key, d)[1]) + 1))  # noqa: E731
    bw_fit, bw_test = span("backward_fit_years", [2016, 2025]), span("backward_test_years", [2012, 2015])
    fw_fit, fw_test = span("forward_fit_years", [2012, 2021]), span("forward_test_years", [2022, 2025])
    rule = cfg.get("rule") or {}

    out = settings.processed_dir / "counts"
    path = out / "daily.parquet"
    if not path.exists():
        return {"stage": "selection", "note": "no counts/daily.parquet: run the counts stage first"}
    daily = pd.read_parquet(path)
    daily["acq_date"] = pd.to_datetime(daily["acq_date"])
    sats = sorted({s for segs in cands.values() for g, *_ in segs for s in GROUPS[g]} | set(GROUPS[ref]))
    bad = bad_days(settings, sats) if cfg.get("exclude_gap_days", True) else None
    monthly = _monthly(daily, bad)
    w = wide_table(monthly)
    base = baseline(w, ref, years, [])

    # Tests 1-3 on all fires
    scores, drifts, preds = [], [], []
    for name, segs in cands.items():
        for test, p in (("leave_one_year_out", loyo(w, segs, ref, years, min_cells, [])),
                        ("backward", predict(w, segs, ref, bw_fit, bw_test, min_cells, [])),
                        ("forward", predict(w, segs, ref, fw_fit, fw_test, min_cells, []))):
            sc = score(p, base, [])
            if len(sc):
                scores.append(sc.assign(candidate=name, test=test))
            if test == "leave_one_year_out" and len(p):
                drifts.append(drift(p, []).assign(candidate=name))
                preds.append(p.assign(candidate=name))
    scores = pd.concat([x for x in scores if len(x) and not x.isna().all().all()] or scores, ignore_index=True)
    drifts = pd.concat(drifts, ignore_index=True) if drifts else pd.DataFrame(columns=["region", "candidate", "drift_pct_per_year"])

    # Summary per candidate and the pre-registered choice
    summ = []
    for name in cands:
        s = scores[scores["candidate"] == name]
        reg = s[s["region"] != "global"]
        lo = reg[reg["test"] == "leave_one_year_out"]
        bw = reg[reg["test"] == "backward"]
        g_bw = s[(s["region"] == "global") & (s["test"] == "backward")]
        g_dr = drifts[(drifts["candidate"] == name) & (drifts["region"] == "global")]
        a_lo = lo["agreement_pct"].mean() if len(lo) else np.nan
        a_bw = bw["agreement_pct"].mean() if len(bw) else np.nan
        summ.append({"candidate": name,
                     "score": round(float(np.nanmean([a_lo, a_bw])), 2) if not (np.isnan(a_lo) and np.isnan(a_bw)) else None,
                     "agreement_loyo_pct": round(float(a_lo), 1) if not np.isnan(a_lo) else None,
                     "agreement_backward_pct": round(float(a_bw), 1) if not np.isnan(a_bw) else None,
                     "high_caught_loyo_pct": round(float(lo["high_caught_pct"].mean()), 1) if len(lo) else None,
                     "false_alarm_loyo_pct": round(float(lo["false_alarm_pct"].mean()), 1) if len(lo) else None,
                     "mean_abs_error_loyo_pct": round(float(lo["total_error_pct"].abs().mean()), 2) if len(lo) else None,
                     "mean_abs_error_backward_pct": round(float(bw["total_error_pct"].abs().mean()), 2) if len(bw) else None,
                     "global_backward_error_pct": float(g_bw["total_error_pct"].iloc[0]) if len(g_bw) else np.nan,
                     "global_drift_pct_per_year": float(g_dr["drift_pct_per_year"].iloc[0]) if len(g_dr) else np.nan})
    summary = pd.DataFrame(summ)
    winner, why = pick_winner(summary, rule)
    summary["eligible"] = summary["global_backward_error_pct"].abs() <= float(rule.get("max_global_backward_error_pct", 10))

    # Per-region best by leave-one-year-out agreement (information for fix 8, not the choice)
    lo_all = scores[scores["test"] == "leave_one_year_out"]
    per_region = (lo_all.assign(abs_err=lo_all["total_error_pct"].abs())
                  .sort_values(["region", "agreement_pct", "abs_err"], ascending=[True, False, True])
                  .groupby("region").head(1)[["region", "candidate", "agreement_pct", "total_error_pct"]])

    # Test 4: MODIS-era stability
    era_years, era = modis_era(_monthly(daily, bad_days(settings, ["Terra", "Aqua"]) if cfg.get("exclude_gap_days", True) else None),
                               list(range(int(cfg.get("modis_first_year", 2002)), int(ov[1]) + 1)),
                               int(cfg.get("terra_stable_until", 2019)))

    # Test 5: bootstrap ranges of k
    boots = []
    for name, segs in cands.items():
        b = bootstrap_k(w, segs, ref, years, min_cells, int(cfg.get("bootstrap_samples", 1000)), int(cfg.get("seed", 26)))
        if len(b):
            boots.append(b.assign(candidate=name))
    boot = pd.concat(boots, ignore_index=True) if boots else pd.DataFrame()

    # Test 6: small-fire and large-fire cells separately
    size_scores, threshold = pd.DataFrame(), None
    if cfg.get("fire_size_split", True):
        labels, threshold = size_classes(settings, ref, years)
        if len(labels):
            groups = {g: GROUPS[g] for g in sorted({g for segs in cands.values() for g, *_ in segs} | {ref})}
            ds = daily_series(settings, groups=groups, cell_class=labels)
            ds = ds[ds["size"] != "unlabelled"]
            ms = _monthly_by(ds, bad)
            ws = wide_table(ms, ["size"])
            bs = baseline(ws, ref, years, ["size"])
            parts = []
            for name, segs in cands.items():
                for test, p in (("leave_one_year_out", loyo(ws, segs, ref, years, min_cells, ["size"])),
                                ("backward", predict(ws, segs, ref, bw_fit, bw_test, min_cells, ["size"]))):
                    sc = score(p, bs, ["size"])
                    if len(sc):
                        parts.append(sc.assign(candidate=name, test=test))
            size_scores = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # Save
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "selection_scores.csv", index=False)
    drifts.to_csv(out / "selection_drift.csv", index=False)
    summary.to_csv(out / "selection_summary.csv", index=False)
    per_region.to_csv(out / "selection_best_per_region.csv", index=False)
    if preds:
        pd.concat(preds, ignore_index=True).to_csv(out / "selection_loyo_predictions.csv", index=False)
    if len(era):
        era.to_csv(out / "selection_modis_era.csv", index=False)
        era_years.to_csv(out / "selection_modis_era_yearly.csv", index=False)
    if len(boot):
        boot.to_csv(out / "selection_bootstrap_k.csv", index=False)
    if len(size_scores):
        size_scores.to_csv(out / "selection_fire_size.csv", index=False)

    boot_width = (boot.groupby(["candidate", "region"])["range_pct"].median().unstack(0).round(1)
                  if len(boot) else pd.DataFrame())
    report = {"stage": "selection", "reference": ref, "candidates": cands, "rule": rule,
              "overlap_years": [years[0], years[-1]], "backward": {"fit": [bw_fit[0], bw_fit[-1]], "test": [bw_test[0], bw_test[-1]]},
              "days_excluded": None if bad is None else int(len(bad)),
              "winner": winner, "why": why,
              "summary": summary.replace({np.nan: None}).to_dict("records"),
              "best_per_region_loyo": per_region.to_dict("records"),
              "modis_era": era.replace({np.nan: None}).to_dict("records") if len(era) else [],
              "fire_size_threshold_mw": None if threshold is None else round(threshold, 2)}

    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        names = list(cands)
        def table(test, col):
            t = scores[scores["test"] == test].pivot(index="region", columns="candidate", values=col)
            return t[[c for c in names if c in t]].to_string()
        print("[selection] SUMMARY (score = mean category agreement %, leave-one-year-out and backward, test regions)\n"
              + summary.to_string(index=False), flush=True)
        print(f"[selection] test 2+3: category agreement %, leave-one-year-out ({years[0]}-{years[-1]})\n" + table("leave_one_year_out", "agreement_pct"), flush=True)
        print(f"[selection] test 1: backward (fit {bw_fit[0]}-{bw_fit[-1]}, predict {bw_test[0]}-{bw_test[-1]}) total error %\n" + table("backward", "total_error_pct"), flush=True)
        print("[selection] test 1+2: backward category agreement %\n" + table("backward", "agreement_pct"), flush=True)
        print("[selection] unusual/critical months caught %, leave-one-year-out\n" + table("leave_one_year_out", "high_caught_pct"), flush=True)
        dt = drifts.pivot(index="region", columns="candidate", values="drift_pct_per_year")
        print("[selection] false trend (drift of real/predicted, % per year; 0 = none)\n" + dt[[c for c in names if c in dt]].to_string(), flush=True)
        if len(era):
            print("[selection] test 4: MODIS-era stability (Aqua/Terra ratio drift; 2002 step when Aqua joins)\n" + era.to_string(index=False), flush=True)
        if len(boot_width):
            print("[selection] test 5: bootstrap 95% range of k, median over months, % of k\n" + boot_width[[c for c in names if c in boot_width]].to_string(), flush=True)
        if len(size_scores):
            size_scores = size_scores[size_scores["region"] != "global"] if (size_scores["region"] != "global").any() else size_scores
            ag = (size_scores.groupby(["test", "size", "candidate"])["agreement_pct"].mean().unstack().round(1))
            er = (size_scores.assign(e=size_scores["total_error_pct"].abs())
                  .groupby(["test", "size", "candidate"])["e"].mean().unstack().round(2))
            print(f"[selection] test 6: fire size (threshold {threshold:.1f} MW per cell-day), mean agreement % over regions\n"
                  + ag[[c for c in names if c in ag]].to_string(), flush=True)
            print("[selection] test 6: mean absolute total error % over regions\n" + er[[c for c in names if c in er]].to_string(), flush=True)
        print("[selection] best candidate per region (leave-one-year-out agreement; information for fix 8)\n"
              + per_region.to_string(index=False), flush=True)
        print(f"[selection] WINNER by the pre-registered rule: {winner} ({why})", flush=True)
    return report


def _monthly_by(daily: pd.DataFrame, exclude) -> pd.DataFrame:
    d = daily if exclude is None or len(exclude) == 0 else daily[~daily["acq_date"].isin(exclude)]
    d = d.assign(year=d["acq_date"].dt.year, month=d["acq_date"].dt.month)
    return d.groupby(["region", "size", "group", "year", "month"], as_index=False)[["active_cells"]].sum()
