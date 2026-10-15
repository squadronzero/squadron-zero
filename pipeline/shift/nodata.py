"""Phase 2, fix 7: the no-data mask. "No data" must never look like "no fire".

A satellite record has holes: whole days when a satellite was silent or sent only part of
its data (the qc coverage table: 15 gaps of 3+ days, the longest S-NPP 28 Apr-2 Jun 2026),
and days when part of the world is missing even though the global total looks fine (lost
data over one region). If the calendar simply adds up what it has, every hole looks like a
quiet fire day. This stage finds every hole and fills or flags it, in four steps:

  1. Whole-day gaps per satellite: the qc coverage table ("none" and "low" days).
  2. Regional gaps: per 30-degree zone and day, each satellite is compared with a partner
     that flies close in time (Terra/Aqua, S-NPP/NOAA-20, S-NPP/MODIS before NOAA-20).
     When the partner sees plenty of fire but the satellite sees under `low_fraction` of
     what their usual ratio says, that zone-day is a regional gap for the satellite.
  3. Filling, tested before it is trusted. For each link (target <- substitute: S-NPP <-
     NOAA-20, S-NPP <- the D19 MODIS piece, merged MODIS <- Aqua, merged MODIS <- Terra),
     outages are simulated on days when both were working: a run of days is hidden, then the
     month total is rebuilt three ways and compared with the truth:
        none        the hidden days count as zero (what a naive record does)
        scale       the month is scaled up by days available / days in the month
        substitute  the hidden days are taken from the substitute x k (ratio per 10-degree
                    tile and calendar month, fitted leaving the test year out, with the
                    fallback chain tile month -> tile all year -> 30-degree zone -> world)
        hybrid      substitute where the tile has its own monthly factor, scale elsewhere
     Every gap length is tested equally often (1-3, 4-7, 8-14 and 15-27 days). A
     The rule fixed before the run (settings `nodata.rule`) decides per link.
  4. The record's availability, per zone and day: the record's own satellite ("primary"),
     filled from a substitute in the order of `nodata.record`, or "no data". Per zone and
     month: how many days are primary, filled and missing, and the expected error of the
     month total from the test; months past `month_error_limit_pct` are withheld.

What this stage cannot see: cloud. FIRMS lists fires only, so a cloudy day and a day with
no fire look the same. That needs a cloud product (fix 7b).

    python -m shift.run --dataset global --stage nodata     (after qc; reads the record once)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings
from .counts import GROUPS, source_segments
from .coverage import load_coverage
from .static_mask import masked_cells
from .tiles import tile_name

COLS = ["acq_date", "cell_id", "cell_lat", "cell_lon", "satellite", "frp_sum_mw"]
SATS = ["Terra", "Aqua", "S-NPP", "NOAA-20"]
GRP = ["Terra", "Aqua", "MODIS", "S-NPP", "NOAA-20"]
VARS = ("cells", "frp")
DEFAULT_BUCKETS = [[1, 3], [4, 7], [8, 14], [15, 27]]
METHODS = ("none", "scale", "substitute", "hybrid")


# ---------------------------------------------------------------- the record as daily tile arrays

class Space:
    """10-degree tiles and 30-degree zones on a fixed world grid."""

    def __init__(self, tile_deg: float, zone_deg: float):
        self.tile_deg, self.zone_deg = tile_deg, zone_deg
        self.nlat, self.nlon = int(round(180 / tile_deg)), int(round(360 / tile_deg))
        self.nt = self.nlat * self.nlon
        per = int(round(zone_deg / tile_deg))
        self.zlat, self.zlon = int(np.ceil(self.nlat / per)), int(np.ceil(self.nlon / per))
        self.nz = self.zlat * self.zlon
        li, lo = np.divmod(np.arange(self.nt), self.nlon)
        self.zone_of_tile = (li // per) * self.zlon + (lo // per)
        self.tile_lat0 = -90 + li * tile_deg
        self.tile_lon0 = -180 + lo * tile_deg
        self.Z = np.zeros((self.nz, self.nt), np.float32)
        self.Z[self.zone_of_tile, np.arange(self.nt)] = 1.0

    def tile_of(self, lat, lon) -> np.ndarray:
        li = np.clip(np.floor((np.asarray(lat, float) + 90) / self.tile_deg).astype(int), 0, self.nlat - 1)
        lo = np.clip(np.floor((np.asarray(lon, float) + 180) / self.tile_deg).astype(int), 0, self.nlon - 1)
        return li * self.nlon + lo

    def tile_names(self) -> np.ndarray:
        return np.array([tile_name(a, b, self.tile_deg) for a, b in zip(self.tile_lat0, self.tile_lon0)])

    def zone_names(self) -> np.ndarray:
        zl, zo = np.divmod(np.arange(self.nz), self.zlon)
        return np.array([tile_name(-90 + a * self.zone_deg, -180 + b * self.zone_deg, self.zone_deg)
                         for a, b in zip(zl, zo)])


def daily_tiles(settings: Settings, space: Space, start: pd.Timestamp, ndays: int,
                drop_cells: np.ndarray | None) -> dict:
    """Burning cells and fire power per satellite group, tile and day (dense float32 arrays,
    shape [group, tile, day]). Terra and Aqua merged per cell-day for the MODIS group.
    Non-fire cells (D22) removed in every year when `drop_cells` is given."""
    root = settings.celldays_dir
    A = {v: np.zeros((len(GRP), space.nt, ndays), np.float32) for v in VARS}
    drop = np.asarray(drop_cells if drop_cells is not None else [], dtype="int64")
    s0 = np.datetime64(start.date(), "D")
    for part in sorted(root.glob("sensor_family=*/year=*/month=*")) if root.exists() else []:
        d = pd.read_parquet(part, columns=COLS)
        d["satellite"] = d["satellite"].astype(str)
        d = d[d["satellite"].isin(SATS)]
        if len(drop):
            d = d[~np.isin(d["cell_id"].to_numpy("int64"), drop)]
        if d.empty:
            continue
        d["day"] = (pd.to_datetime(d["acq_date"]).to_numpy().astype("datetime64[D]") - s0).astype(int)
        d = d[(d["day"] >= 0) & (d["day"] < ndays)]
        for gi, g in enumerate(GRP):
            x = d[d["satellite"].isin(GROUPS[g])]
            if x.empty:
                continue
            if len(GROUPS[g]) > 1:          # one cell-day counts once however many satellites saw it
                x = x.groupby(["day", "cell_id"], as_index=False).agg(
                    cell_lat=("cell_lat", "first"), cell_lon=("cell_lon", "first"), frp_sum_mw=("frp_sum_mw", "sum"))
            key = space.tile_of(x["cell_lat"], x["cell_lon"]).astype("int64") * ndays + x["day"].to_numpy("int64")
            s = pd.DataFrame({"k": key, "f": x["frp_sum_mw"].to_numpy(float)}).groupby("k")["f"].agg(["size", "sum"])
            idx = s.index.to_numpy()
            A["cells"][gi].reshape(-1)[idx] += s["size"].to_numpy(np.float32)
            A["frp"][gi].reshape(-1)[idx] += s["sum"].to_numpy(np.float32)
    return A


def ok_days(cov: pd.DataFrame, start: pd.Timestamp, ndays: int) -> dict:
    """True on days the satellite was fully working ("ok" in the qc coverage table). A day
    with no row (before launch, after the end, or simply missing) is not ok."""
    ok = {s: np.zeros(ndays, bool) for s in SATS}
    for sat, c in cov.groupby("satellite"):
        if sat not in ok:
            continue
        i = (c["acq_date"] - start).dt.days.to_numpy()
        v = (i >= 0) & (i < ndays)
        ok[sat][i[v]] = (c["status"].to_numpy() == "ok")[v]
    ok["MODIS"] = ok["Terra"] & ok["Aqua"]
    return ok


# ---------------------------------------------------------------- step 2: regional gaps

def regional_gaps(Zc: dict, ok: dict, partners: dict, window: int, min_partner: float,
                  low_fraction: float, min_expected: float = 0.0, severe_fraction: float = 0.0,
                  min_zones: int = 10 ** 6) -> tuple[dict, pd.DataFrame]:
    """Zone-days when a satellite saw far less fire than its partner says it should.
    Zc: burning cells per group, zone and day. The partner used on a day is the first one in
    the list that was fully working that day. A zone-day is judged only when the partner saw at
    least `min_partner` burning cells AND the satellite was expected to see at least
    `min_expected` (so a handful of fires seen by one satellite and not the other is noise,
    not a data gap).

    D25 (15 Oct, from the first global run): a shortfall under `low_fraction` is only a DATA
    LOSS when it is near-total (under `severe_fraction` of what was expected) or when the same
    satellite falls short in at least `min_zones` zones on the same day. A moderate shortfall in
    one zone is the normal difference between two looks a few hours apart (cloud moves, fires
    start and stop): it is kept in the file as kind "weather" and not treated as a gap."""
    nz, nd = next(iter(Zc.values())).shape
    bad = {g: np.zeros((nz, nd), bool) for g in GRP}
    rows = []
    for sat, plist in partners.items():
        chosen = np.full(nd, -1)
        for i, p in enumerate(plist):
            chosen[(chosen == -1) & ok[p] & ok[sat]] = i
        for i, p in enumerate(plist):
            a, b = Zc[sat].astype(float), Zc[p].astype(float)
            valid = (b >= min_partner) & ok[sat][None, :] & ok[p][None, :]
            ratio = np.where(valid, a / np.maximum(b, 1.0), np.nan)
            med = pd.DataFrame(ratio.T).rolling(window, center=True, min_periods=5).median().to_numpy().T
            expected = np.where(np.isfinite(med), med, 0.0) * b
            f = (valid & np.isfinite(med) & (med > 0) & (expected >= min_expected)
                 & (a < low_fraction * expected) & (chosen == i)[None, :])
            if not f.any():
                continue
            many = f.sum(axis=0) >= min_zones                         # same satellite, same day, many zones
            loss = f & ((a < severe_fraction * expected) | many[None, :])
            bad[sat] |= loss
            for z, d in zip(*np.nonzero(f)):
                rows.append({"satellite": sat, "partner": p, "zone": int(z), "day": int(d),
                             "cells": int(a[z, d]), "expected": round(float(med[z, d] * b[z, d]), 1),
                             "kind": "data loss" if loss[z, d] else "weather"})
    bad["MODIS"] = bad["Terra"] | bad["Aqua"]
    return bad, pd.DataFrame(rows, columns=["satellite", "partner", "zone", "day", "cells", "expected", "kind"])


# ---------------------------------------------------------------- step 3: fill links, tested

class Series:
    """One satellite group or a piecewise source (the D19 MODIS piece) as daily tile arrays,
    with its usable zone-days and the piece each day belongs to (each piece gets its own k)."""

    def __init__(self, name, A, usable, segments, day_year):
        self.name = name
        if segments is None:
            gi = GRP.index(name)
            self.vals = {v: A[v][gi] for v in VARS}
            self.use = usable[name]
            self.seg = np.zeros(len(day_year), int)
            self.seg_names = [name]
        else:
            self.vals = {v: np.zeros_like(A[v][0]) for v in VARS}
            self.use = np.zeros_like(usable[GRP[0]])
            self.seg = np.full(len(day_year), -1)
            self.seg_names = []
            for i, (g, y0, y1) in enumerate(segments):
                on = (day_year >= int(y0)) & (day_year <= int(y1))
                gi = GRP.index(g)
                for v in VARS:
                    self.vals[v][:, on] = A[v][gi][:, on]
                self.use[:, on] = usable[g][:, on]
                self.seg[on] = i
                self.seg_names.append(f"{g} {y0}-{y1}")


def month_sums(x: np.ndarray, ym_slices: list) -> np.ndarray:
    """[tile, ym] totals from a [tile, day] array."""
    return np.stack([x[:, a:b].sum(axis=1) for a, b in ym_slices], axis=1)


def fit_k(Tm, Sm, sel_ym, ym_month, space: Space, min_v: float):
    """k[tile, month] = target / substitute over the selected year-months, with the fallback
    chain tile month -> tile all year -> 30-degree zone month -> world month.
    Level: 0 own month, 1 own all-year, 2 zone, 3 world."""
    T = np.zeros((space.nt, 12))
    S = np.zeros((space.nt, 12))
    for m in range(12):
        cols = sel_ym & (ym_month == m + 1)
        T[:, m] = Tm[:, cols].sum(axis=1)
        S[:, m] = Sm[:, cols].sum(axis=1)
    Tz, Sz = space.Z @ T, space.Z @ S
    Tw, Sw = T.sum(0), S.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        kw = np.where(Sw > 0, Tw / Sw, np.nan)
        kw = np.where(np.isnan(kw), Tw.sum() / Sw.sum() if Sw.sum() > 0 else 1.0, kw)
        kz = np.where(Sz >= min_v, Tz / Sz, np.nan)[space.zone_of_tile]
        kall = np.where(S.sum(1) >= min_v, T.sum(1) / S.sum(1), np.nan)[:, None] * np.ones((1, 12))
        own = S >= min_v
        k = np.where(own, T / np.where(own, S, 1), kall)
        level = np.where(own, 0, np.where(~np.isnan(kall), 1, np.where(~np.isnan(kz), 2, 3)))
        k = np.where(np.isnan(k), np.where(np.isnan(kz), kw[None, :], kz), k)
    return k, level


def bucket_of(n, buckets) -> int:
    for i, (a, b) in enumerate(buckets):
        if a <= n <= b:
            return i
    return len(buckets)            # longer than the last bucket: not testable, month withheld


def test_link(T: Series, S: Series, years: list[int], tested: np.ndarray, space: Space, cal: dict,
              min_v: dict, buckets, replicates: int, rng) -> tuple[pd.DataFrame, dict]:
    """Simulated outages on days when target and substitute both worked. Returns one row per
    tested tile x month x replicate x variable with the truth and each method's estimate,
    plus the factors fitted on all link years (for production)."""
    day_year, ym_slices, ym_year, ym_month = cal["day_year"], cal["ym_slices"], cal["ym_year"], cal["ym_month"]
    in_years = np.isin(day_year, years)
    elig_z = T.use & S.use & in_years[None, :]
    elig = elig_z[space.zone_of_tile]                       # tile x day
    Tm = {v: month_sums(T.vals[v] * elig, ym_slices) for v in VARS}
    Sm = {v: month_sums(S.vals[v] * elig, ym_slices) for v in VARS}
    ym_seg = np.array([S.seg[a] if b > a else -1 for a, b in ym_slices])
    # production factors: all link years, per piece
    final = {}
    for si, sname in enumerate(S.seg_names):
        sel = np.isin(ym_year, years) & (ym_seg == si)
        if sel.any():
            final[sname] = {v: fit_k(Tm[v], Sm[v], sel, ym_month, space, min_v[v]) for v in VARS}
    rows = []
    tiles = np.nonzero(tested)[0]
    for y in years:
        ym_idx = [i for i in range(len(ym_slices)) if ym_year[i] == y and ym_slices[i][1] > ym_slices[i][0]]
        if not ym_idx:
            continue
        si = ym_seg[ym_idx[0]]
        fit_sel = np.isin(ym_year, [x for x in years if x != y]) & (ym_seg == si)
        if not fit_sel.any():
            continue
        ks = {v: fit_k(Tm[v], Sm[v], fit_sel, ym_month, space, min_v[v]) for v in VARS}
        for i in ym_idx:
            a, b = ym_slices[i]
            nd = b - a
            m = ym_month[i]
            for r in range(replicates):
                lo, hi = buckets[rng.integers(len(buckets))]
                L = int(rng.integers(lo, min(hi, nd - 1) + 1)) if nd > 1 else 1
                s0 = int(rng.integers(0, nd - L + 1))
                hid = np.zeros(nd, bool)
                hid[s0:s0 + L] = True
                e = elig[tiles, a:b]
                n_e, n_h = e.sum(1), (e & hid[None, :]).sum(1)
                n_o = n_e - n_h
                keep = (n_h > 0) & (n_o > 0)
                if not keep.any():
                    continue
                for v in VARS:
                    t = T.vals[v][tiles, a:b]
                    true = (t * e).sum(1)
                    obs = (t * (e & ~hid[None, :])).sum(1)
                    s_hid = (S.vals[v][tiles, a:b] * (e & hid[None, :])).sum(1)
                    k, level = ks[v][0][tiles, m - 1], ks[v][1][tiles, m - 1]
                    scale = obs * n_e / np.maximum(n_o, 1)
                    sub = obs + k * s_hid
                    hyb = np.where(level == 0, sub, scale)
                    rows.append(pd.DataFrame({
                        "tile": tiles[keep], "year": y, "month": m, "replicate": r, "variable": v,
                        "hidden_days": n_h[keep], "true": true[keep], "none": obs[keep], "scale": scale[keep],
                        "substitute": sub[keep], "hybrid": hyb[keep]}))
    cols = ["tile", "year", "month", "replicate", "variable", "hidden_days", "true", *METHODS]
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)), final


def score_methods(t: pd.DataFrame, buckets, limit_tile: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per method and variable: fire-weighted absolute error of month totals (% of the true
    total, pooled over tested tiles, i.e. the fire-weighted mean of per-tile errors), global
    total error, share of tiles within +-limit; and the same error by gap length."""
    out, byb = [], []
    if t.empty:
        return pd.DataFrame(), pd.DataFrame()
    t = t.assign(bucket=[bucket_of(n, buckets) for n in t["hidden_days"]])
    for v, d in t.groupby("variable"):
        tot = d["true"].sum()
        for meth in METHODS:
            err = (d[meth] - d["true"]).abs()
            per_tile = err.groupby(d["tile"]).sum() / d["true"].groupby(d["tile"]).sum().replace(0, np.nan)
            out.append({"variable": v, "method": meth,
                        "abs_error_pct": round(float(100 * err.sum() / tot), 2) if tot > 0 else None,
                        "global_error_pct": round(float(100 * (d[meth].sum() / tot - 1)), 2) if tot > 0 else None,
                        "tiles_within_limit_pct": round(float(100 * (100 * per_tile.dropna() <= limit_tile).mean()), 1)
                        if per_tile.notna().any() else None})
            for bi, g in d.groupby("bucket"):
                gt = g["true"].sum()
                lab = f"{buckets[bi][0]}-{buckets[bi][1]} days" if bi < len(buckets) else "longer"
                byb.append({"variable": v, "method": meth, "bucket": int(bi), "gap_length": lab, "months": int(len(g)),
                            "abs_error_pct": round(float(100 * (g[meth] - g["true"]).abs().sum() / gt), 2) if gt > 0 else None})
    return pd.DataFrame(out), pd.DataFrame(byb)


def decide(sc: pd.DataFrame, rule: dict) -> tuple[str, str]:
    """RULE FIXED BEFORE THE RUN (settings nodata.rule, 15 Oct 2026, committed to git).
    Default for a gap is `scale`. A fill method (substitute or hybrid) is adopted for a link
    only if, on the simulated outages, its fire-weighted absolute error of month totals is
    lower than scale's by more than gain_points for fire power, AND not higher for burning
    cells, AND its global total error is within +-global_error_limit_pct for both. If both
    fill methods qualify, the lower fire power error wins; on a tie, hybrid (the more cautious)."""
    gain, lim = float(rule.get("gain_points", 1)), float(rule.get("global_error_limit_pct", 5))
    if sc.empty:
        return "scale", "no test months"
    e = sc.set_index(["variable", "method"])
    base_f, base_c = e.loc[("frp", "scale"), "abs_error_pct"], e.loc[("cells", "scale"), "abs_error_pct"]
    ok = []
    for meth in ("substitute", "hybrid"):
        f, c = e.loc[("frp", meth), "abs_error_pct"], e.loc[("cells", meth), "abs_error_pct"]
        gf, gc = e.loc[("frp", meth), "global_error_pct"], e.loc[("cells", meth), "global_error_pct"]
        if f < base_f - gain and c <= base_c and abs(gf) <= lim and abs(gc) <= lim:
            ok.append((f, meth))
    if not ok:
        return "scale", (f"no fill method beats scale by more than {gain:g} points on fire power with burning cells "
                         f"no worse and global error within +-{lim:g}% (scale: fire power {base_f}%, cells {base_c}%)")
    f, meth = min(ok)
    return meth, f"{meth}: fire power error {f}% vs scale {base_f}%; burning cells {e.loc[('cells', meth), 'abs_error_pct']}% vs {base_c}%"


# ---------------------------------------------------------------- the stage

def run_nodata(settings: Settings) -> dict:
    cfg = settings.raw.get("nodata") or {}
    fcfg = settings.raw.get("factors") or {}
    space = Space(float(cfg.get("tile_deg", 10)), float(cfg.get("zone_deg", 30)))
    buckets = [list(map(int, b)) for b in cfg.get("gap_buckets", DEFAULT_BUCKETS)]
    rule = cfg.get("rule") or {}
    month_limit = float(rule.get("month_error_limit_pct", 25))
    min_v = {"cells": float(fcfg.get("min_cells_per_month", 30)), "frp": float(fcfg.get("min_frp_mw_per_month", 300))}

    cov = load_coverage(settings)
    if cov is None:
        return {"stage": "nodata", "note": "no coverage table: run the qc stage on the global dataset first"}
    start = cov["acq_date"].min().normalize()
    end = cov["acq_date"].max().normalize()
    days = pd.date_range(start, end, freq="D")
    nd = len(days)
    day_year = days.year.to_numpy()
    ym = pd.PeriodIndex(days, freq="M")
    uniq = ym.unique()
    first = np.searchsorted(ym.asi8, uniq.asi8)
    ym_slices = [(int(a), int(b)) for a, b in zip(first, list(first[1:]) + [nd])]
    cal = {"day_year": day_year, "ym_slices": ym_slices,
           "ym_year": np.array([p.year for p in uniq]), "ym_month": np.array([p.month for p in uniq])}

    use_mask = bool((settings.raw.get("nonfire") or {}).get("mask_all_years", False))
    mask = masked_cells(settings) if use_mask else None
    print(f"[nodata] reading the record once ({start.date()} to {end.date()}, {space.nt} tiles); "
          "this prints nothing for several minutes", flush=True)
    A = daily_tiles(settings, space, start, nd, mask)
    ok = ok_days(cov, start, nd)
    # The newest day of a form download is cut off at the time NASA processed the request, so
    # it is never complete: the last `drop_last_days` day(s) of the record count as no data.
    n_drop = int(cfg.get("drop_last_days", 1))
    if n_drop > 0:
        for sat in SATS + ["MODIS"]:
            ok[sat][nd - n_drop:] = False

    # ---- step 2: regional gaps
    reg = cfg.get("regional") or {}
    Zc = {g: space.Z @ A["cells"][GRP.index(g)] for g in GRP}
    partners = {k: list(v) for k, v in (reg.get("partners") or
                {"Terra": ["Aqua"], "Aqua": ["Terra"], "S-NPP": ["NOAA-20", "MODIS"], "NOAA-20": ["S-NPP"]}).items()}
    bad, flags = regional_gaps(Zc, ok, partners, int(reg.get("window_days", 31)),
                               float(reg.get("min_partner_cells", 40)), float(reg.get("low_fraction", 0.25)),
                               float(reg.get("min_expected_cells", 20)), float(reg.get("severe_fraction", 0.10)),
                               int(reg.get("min_zones_same_day", 3)))
    usable = {g: ok[g][None, :] & ~bad[g] for g in GRP}          # zone x day

    # ---- step 3: links, tested under the rule
    names = space.tile_names()
    znames = space.zone_names()
    ref = GRP.index(fcfg.get("reference", "S-NPP"))
    ov = [int(x) for x in fcfg.get("overlap_years", [2012, 2025])]
    on = (day_year >= ov[0]) & (day_year <= ov[1])
    ref_use = usable[GRP[ref]][space.zone_of_tile]
    per_month = (A["cells"][ref][:, on] * ref_use[:, on]).sum(1) / (12 * (ov[1] - ov[0] + 1))
    tested = per_month >= float(fcfg.get("min_ref_cells_per_month", 50))

    d19 = source_segments(settings.raw.get("counts") or {})
    rng = np.random.default_rng(int(cfg.get("seed", 7)))
    reps = int(cfg.get("replicates", 3))
    links = cfg.get("links") or [
        {"target": "S-NPP", "substitute": "NOAA-20", "years": [2019, 2025]},
        {"target": "S-NPP", "substitute": "D19", "years": [2012, 2025]},
        {"target": "MODIS", "substitute": "Aqua", "years": [2003, 2019]},
        {"target": "MODIS", "substitute": "Terra", "years": [2003, 2019]}]

    def series(name):
        return Series(name, A, usable, d19 if name == "D19" else None, day_year)

    results, scores, bys, fac_rows, chosen = {}, [], [], [], {}
    for ln in links:
        tgt, sub = ln["target"], ln["substitute"]
        yrs = list(range(int(ln["years"][0]), int(ln["years"][1]) + 1))
        lname = f"{tgt} <- {sub}"
        T, S = series(tgt), series(sub)
        t, final = test_link(T, S, yrs, tested, space, cal, min_v, buckets, reps, rng)
        sc, byb = score_methods(t, buckets, float(rule.get("tile_limit_pct", 10)))
        meth, why = decide(sc, rule)
        chosen[lname] = meth
        results[lname] = {"method": meth, "why": why, "years": [yrs[0], yrs[-1]],
                          "test_months": int(t[t["variable"] == "frp"].shape[0]) if len(t) else 0}
        if len(sc):
            scores.append(sc.assign(link=lname))
            bys.append(byb.assign(link=lname))
        for seg, fv in final.items():
            for v, (k, level) in fv.items():
                ti = np.nonzero(k.sum(1) > 0)[0]
                for m in range(12):
                    fac_rows.append(pd.DataFrame({"link": lname, "segment": seg, "variable": v, "tile": names[ti],
                                                  "month": m + 1, "k": k[ti, m],
                                                  "source": np.array(["own month", "own all-year", "zone", "world"])[level[ti, m]]}))
    scores = pd.concat(scores, ignore_index=True) if scores else pd.DataFrame()
    bys = pd.concat(bys, ignore_index=True) if bys else pd.DataFrame()
    facs = pd.concat(fac_rows, ignore_index=True) if fac_rows else pd.DataFrame()

    # ---- step 4: the record's availability per zone and day
    eras = cfg.get("record") or [
        {"from": "2002-01-01", "to": "2002-07-03", "target": "Terra", "fill": []},
        {"from": "2002-07-04", "to": "2012-01-19", "target": "MODIS", "fill": ["Aqua", "Terra"]},
        {"from": "2012-01-20", "to": "2100-12-31", "target": "S-NPP", "fill": ["NOAA-20", "D19"]}]
    d19_use = series("D19").use
    use_any = {**usable, "D19": d19_use}
    status = np.full((space.nz, nd), "", dtype=object)
    target_of = np.full(nd, "", dtype=object)
    for e in eras:
        on = (days >= pd.Timestamp(e["from"])) & (days <= pd.Timestamp(e["to"]))
        if not on.any():
            continue
        tg = e["target"]
        target_of[on] = tg
        st = np.where(use_any[tg][:, on], "primary", "")
        for f in e.get("fill", []):
            if chosen.get(f"{tg} <- {f}", "scale") == "scale":
                continue
            st = np.where((st == "") & use_any[f][:, on], f"filled: {f}", st)
        status[:, on] = np.where(st == "", "no data", st)
    zd = pd.DataFrame({"date": np.tile(days, space.nz), "zone": np.repeat(znames, nd),
                       "target": np.tile(target_of, space.nz), "status": status.reshape(-1)})
    zd = zd[zd["target"] != ""]
    # zones that never hold a burning cell (open ocean, Antarctica) are left out of the summaries
    fire_z = (space.Z @ A["cells"][GRP.index("MODIS")]).sum(1) + (space.Z @ A["cells"][ref]).sum(1) > 0
    zd_fire = zd[zd["zone"].isin(znames[fire_z])]

    # month table with expected error from the test
    def err_lookup(link, meth, n, extrapolate=False):
        """Expected fire power error of a month total with n days filled (or scaled), from the
        simulated outages. Longer than the longest tested gap: a FILL is extrapolated in
        proportion to the days filled (substitution error grows with the share of the month it
        covers); SCALING a month from almost nothing is not trusted (inf)."""
        if bys.empty:
            return np.nan
        b = bucket_of(n, buckets)
        sel = (bys["link"] == link) & (bys["method"] == meth) & (bys["variable"] == "frp")
        if b >= len(buckets):
            if not extrapolate:
                return np.inf
            lo, hi = buckets[-1]
            x = bys[sel & (bys["bucket"] == len(buckets) - 1)]
            return float(x["abs_error_pct"].iloc[0]) * n / ((lo + hi) / 2) if len(x) else np.nan
        x = bys[sel & (bys["bucket"] == b)]
        return float(x["abs_error_pct"].iloc[0]) if len(x) else np.nan

    # scale errors depend only on how much the target varies from day to day; a target with no
    # link of its own (Terra before Aqua) borrows the first link's scale errors
    scale_link = {}
    for ln in links:
        scale_link.setdefault(ln["target"], f"{ln['target']} <- {ln['substitute']}")
    any_link = f"{links[0]['target']} <- {links[0]['substitute']}" if links else ""
    z = zd_fire.assign(year=zd_fire["date"].dt.year, month=zd_fire["date"].dt.month,
                       primary=zd_fire["status"] == "primary", filled=zd_fire["status"].str.startswith("filled"),
                       nodata=zd_fire["status"] == "no data")
    mt = z.groupby(["zone", "year", "month", "target"], as_index=False).agg(
        days=("status", "size"), primary=("primary", "sum"), filled=("filled", "sum"), no_data=("nodata", "sum"),
        fill_source=("status", lambda s: ", ".join(sorted({x.split(": ")[1] for x in s if x.startswith("filled")}))))
    exp = []
    for r in mt.itertuples(index=False):
        a = 0.0
        if r.filled:
            src = r.fill_source.split(", ")[0]
            ln = f"{r.target} <- {src}"
            a = err_lookup(ln, chosen.get(ln, "scale"), int(r.filled), extrapolate=True)
        b = 0.0
        if r.no_data:
            b = np.inf if r.no_data >= r.days else err_lookup(scale_link.get(r.target, any_link), "scale", int(r.no_data))
        a = 0.0 if not np.isfinite(a) else a
        b = 0.0 if (b is None or np.isnan(b)) else b
        exp.append(float(np.hypot(a, b)) if np.isfinite(b) else np.inf)
    mt["expected_error_pct"] = np.round(exp, 2)
    mt["month_status"] = np.select(
        [mt["no_data"] + mt["filled"] == 0,
         (mt["no_data"] == 0) & (mt["expected_error_pct"] <= month_limit),
         mt["expected_error_pct"] <= month_limit],
        ["complete", "filled", "estimated"], "no data")

    # before / after: zone-days in fire zones without the record's own satellite, and left empty
    summ = zd_fire.groupby("target")["status"].value_counts().unstack(fill_value=0)
    gaps_before = int((zd_fire["status"] != "primary").sum())
    gaps_after = int((zd_fire["status"] == "no data").sum())
    whole = zd_fire.groupby("date")["status"].apply(lambda s: (s == "no data").all())
    empty_days = whole[whole].index
    runs = []
    if len(empty_days):
        grp = (pd.Series(empty_days).diff().dt.days != 1).cumsum()
        for _, r in pd.Series(empty_days).groupby(grp.to_numpy()):
            runs.append({"start": r.min().date().isoformat(), "end": r.max().date().isoformat(), "days": int(len(r)),
                         "target": str(target_of[(r.min() - start).days])})
    runs = pd.DataFrame(runs, columns=["start", "end", "days", "target"]).sort_values("days", ascending=False) if runs else pd.DataFrame(columns=["start", "end", "days", "target"])

    loss_flags = flags[flags["kind"] == "data loss"] if len(flags) else flags
    reg_summary = (flags.assign(loss=flags["kind"] == "data loss").groupby("satellite").agg(
        candidates=("day", "size"), data_loss_zone_days=("loss", "sum"),
        data_loss_days=("day", lambda d: loss_flags.loc[loss_flags.index.intersection(d.index), "day"].nunique()))
        if len(flags) else pd.DataFrame())
    if len(flags):
        flags = flags.assign(date=days[flags["day"].to_numpy()].date, zone=znames[flags["zone"].to_numpy()]).drop(columns="day")

    out = settings.processed_dir / "nodata"
    out.mkdir(parents=True, exist_ok=True)
    zd.assign(date=zd["date"].dt.date.astype(str)).to_parquet(out / "zone_day_status.parquet", index=False)
    mt.to_parquet(out / "zone_month_status.parquet", index=False)
    if len(facs):
        facs.to_parquet(out / "fill_factors.parquet", index=False)
    if len(scores):
        scores.to_csv(out / "fill_test_scores.csv", index=False)
        bys.to_csv(out / "fill_test_by_gap_length.csv", index=False)
    flags.to_csv(out / "regional_gaps.csv", index=False)
    import json
    (out / "fill_choice.json").write_text(json.dumps({"chosen": chosen, "why": {k: v["why"] for k, v in results.items()}}, indent=2))
    runs.to_csv(out / "remaining_empty_days.csv", index=False)

    ms = mt[mt["year"] >= int(str((settings.raw.get("record") or {}).get("comparable_start", "2003"))[:4])]
    report = {"stage": "nodata", "dataset": settings.dataset, "date_range": [str(start.date()), str(end.date())],
              "tiles_tested": int(tested.sum()), "fire_zones": int(fire_z.sum()),
              "links": results, "chosen": chosen,
              "regional_gaps": {s: {"candidates": int(r["candidates"]), "data_loss_zone_days": int(r["data_loss_zone_days"]),
                                    "data_loss_days": int(r["data_loss_days"])} for s, r in reg_summary.iterrows()},
              "last_days_dropped": n_drop,
              "zone_days_without_own_satellite": gaps_before, "zone_days_left_empty": gaps_after,
              "whole_world_empty_days": int(len(empty_days)),
              "status_by_target": {t: {k: int(v) for k, v in r.items()} for t, r in summ.iterrows()},
              "months_by_status": ms["month_status"].value_counts().to_dict(),
              "cloud": "not visible in FIRMS fire detections; needs a cloud product (fix 7b)"}
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 80):
        print(f"[nodata] {int(tested.sum())} tested tiles, {int(fire_z.sum())} zones with fire", flush=True)
        print("[nodata] regional shortfalls (candidates) and those judged data loss (D25):\n"
              + (reg_summary.to_string() if len(reg_summary) else "none"), flush=True)
        if len(scores):
            print("[nodata] simulated outages, month total error by method (fire-weighted, %):\n"
                  + scores.pivot_table(index=["link", "variable"], columns="method", values="abs_error_pct")[list(METHODS)].to_string(), flush=True)
            print("[nodata] fire power error by gap length (%):\n"
                  + bys[bys["variable"] == "frp"].pivot_table(index=["link", "gap_length"], columns="method",
                                                               values="abs_error_pct")[list(METHODS)].to_string(), flush=True)
        for ln, r in results.items():
            print(f"[nodata] RULE {ln}: {r['method']} | {r['why']}", flush=True)
        print("[nodata] record availability (zone-days in zones with fire):\n" + summ.to_string(), flush=True)
        print(f"[nodata] zone-days without the record's own satellite: {gaps_before}; left as 'no data' after filling: {gaps_after}", flush=True)
        print("[nodata] days with no data anywhere:\n" + (runs.head(15).to_string(index=False) if len(runs) else "none"), flush=True)
        print("[nodata] zone-months from 2003 by status:\n" + ms["month_status"].value_counts().to_string(), flush=True)
    return report
