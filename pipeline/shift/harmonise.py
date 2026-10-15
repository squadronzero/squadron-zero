"""Phase 2, fix 9: the harmonised record (L3) and the final model (D7).

Everything decided in Phase 2 is applied here, once, to every fire on Earth since 2002:

  * the record's own satellite per period: Terra only (Jan-Jun 2002, flagged not comparable,
    D20), merged Terra + Aqua MODIS (Jul 2002 - 19 Jan 2012), S-NPP (from 20 Jan 2012, D8);
  * MODIS put on the S-NPP scale with a ratio per 10-degree tile and calendar month, for
    burning cells AND fire power (D21), on the D19 source piece, fitted with the D22 + D26
    non-fire mask and without the D25 data-loss zone-days (stage factors, rerun first);
  * gaps filled per zone and day exactly as stage nodata decided (D25): S-NPP from NOAA-20 or
    from the D19 MODIS piece, merged MODIS from Aqua or Terra, each with its tested factor;
    "no data" stays empty and is never shown as zero fire;
  * non-fire cells (D22 + D26) removed in every year.

Output (Drive, processed/global/l3/year=Y/month=M/part.parquet), one row per 0.25-degree cell
and day with fire:
  date, cell25, lat, lon, cells_h (S-NPP-equivalent burning 0.05-degree cells), frp_h
  (S-NPP-equivalent fire power, MW), n_raw (0.05-degree cells actually seen), source
  plus l3_uncertainty.parquet (95% range per tile, month, piece and variable) and the zone-day
  availability already written by stage nodata.

Checks written before the run (settings harmonise.gate and harmonise.anomaly_rule):
  GATE  the harmonised world shows no instrument jump at the 2012 switch: the step
        (mean 2013-2015 / mean 2009-2011) is within +-step_tolerance_pct of the step MODIS alone
        shows over the same years, for burning cells and for fire power.
  RULE  "unusual" months in the MODIS years must not be noise from the scaling. On the overlap
        years, out-of-sample MODIS x k months (leave-one-year-out) are classed against the
        same S-NPP climatology two ways: plain, and "era-fair" (the month's spread widened by
        the measured month error of a factor, from stage places). Era-fair is adopted for the
        calendar if its false-alarm rate is lower by at least fa_drop_points while the share of
        real unusual months it catches falls by no more than caught_max_drop_points.

    python -m shift.run --dataset global --stage harmonise   (after factors rerun; ~30-50 min)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings
from .counts import GROUPS, source_segments
from .coverage import bad_days
from .factors import Cube, data_loss_keys, final_mask, monthly_world, zone_key
from .region import in_bbox, test_regions
from .selection import baseline, classify
from .static_mask import masked_cells
from .tiles import tile_name

COLS = ["acq_date", "cell_id", "cell_lat", "cell_lon", "satellite", "frp_sum_mw"]
SOURCES = {0: "S-NPP", 1: "NOAA-20 fill", 2: "D19 MODIS fill", 3: "MODIS x k",
           4: "Aqua fill x k", 5: "Terra fill x k", 6: "Terra only x k (not comparable)"}
HIGH = {"unusual", "critical"}


def tile_key(lat, lon, deg: float) -> np.ndarray:
    nlon = int(round(360 / deg))
    li = np.floor((np.asarray(lat, float) + 90) / deg).astype("int64")
    lo = np.floor((np.asarray(lon, float) + 180) / deg).astype("int64")
    return li * nlon + lo


def key_of_name(name: str, deg: float) -> int:
    lat0 = int(name[1:3]) * (1 if name[0] == "N" else -1)
    lon0 = int(name[4:7]) * (1 if name[3] == "E" else -1)
    return int(tile_key(lat0 + deg / 2, lon0 + deg / 2, deg))


class Table:
    """k per (tile, month) as a dense array with a per-month world fallback (median over tiles)
    for tiles the table does not list."""

    def __init__(self, df: pd.DataFrame, deg: float):
        nt = int(round(180 / deg)) * int(round(360 / deg))
        a = np.full((nt, 12), np.nan)
        if len(df):
            keys = np.array([key_of_name(t, deg) for t in df["tile"]])
            a[keys, df["month"].to_numpy(int) - 1] = df["k"].to_numpy(float)
        fb = np.nanmedian(np.where(np.isfinite(a) & (a > 0), a, np.nan), axis=0) if np.isfinite(a).any() else np.ones(12)
        fb = np.where(np.isfinite(fb), fb, 1.0)
        self.a = np.where(np.isfinite(a) & (a > 0), a, fb[None, :])

    def __call__(self, tkey: np.ndarray, month: np.ndarray) -> np.ndarray:
        return self.a[tkey, month - 1]


def load_tables(settings: Settings, tile_deg: float) -> dict:
    """Factor tables: MODIS -> S-NPP per D19 piece (stage factors) and the fill links (nodata)."""
    out = {}
    fw = settings.processed_dir / "counts" / "factors_world.parquet"
    if fw.exists():
        f = pd.read_parquet(fw)
        f = f[f["size"].astype(str) == "all"] if "size" in f else f
        for (seg, var), g in f.groupby(["segment", "variable"]):
            out[("factor", seg, var)] = Table(g, tile_deg)
    ff = settings.processed_dir / "nodata" / "fill_factors.parquet"
    if ff.exists():
        f = pd.read_parquet(ff)
        for (link, seg, var), g in f.groupby(["link", "segment", "variable"]):
            out[("fill", link, seg, var)] = Table(g, tile_deg)
    return out


def status_arrays(settings: Settings, start: pd.Timestamp, nd: int, zone_deg: float):
    """Per day: the record's target satellite; per day and zone: the status code."""
    p = settings.processed_dir / "nodata" / "zone_day_status.parquet"
    if not p.exists():
        return None, None
    z = pd.read_parquet(p)
    day = (pd.to_datetime(z["date"]) - start).dt.days.to_numpy()
    lat0 = z["zone"].str[1:3].astype(int) * np.where(z["zone"].str[0] == "N", 1, -1)
    lon0 = z["zone"].str[4:7].astype(int) * np.where(z["zone"].str[3] == "E", 1, -1)
    zk = zone_key(lat0 + 0.5, lon0 + 0.5, zone_deg)
    nz = int(round(180 / zone_deg)) * int(round(360 / zone_deg))
    ok = (day >= 0) & (day < nd)
    status = np.full((nd, nz), "", dtype=object)
    status[day[ok], zk[ok]] = z["status"].to_numpy()[ok]
    target = np.full(nd, "", dtype=object)
    target[day[ok]] = z["target"].to_numpy()[ok]
    return status, target


def segment_for(year: int, segments) -> tuple[str, str]:
    for g, y0, y1 in segments:
        if int(y0) <= year <= int(y1):
            return str(g), f"{g} {y0}-{y1}"
    g, y0, y1 = segments[0]
    return str(g), f"{g} {y0}-{y1}"


def group_frame(d: pd.DataFrame, name: str) -> pd.DataFrame:
    g = d[d["satellite"].isin(GROUPS[name])]
    if len(GROUPS[name]) > 1 and len(g):
        g = g.groupby(["day", "cell_id"], as_index=False).agg(
            cell_lat=("cell_lat", "first"), cell_lon=("cell_lon", "first"), frp_sum_mw=("frp_sum_mw", "sum"))
    return g


def harmonise_month(d: pd.DataFrame, status, target, tables: dict, segments, tile_deg: float, zone_deg: float,
                    res25: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """L3 rows for one month of cell-days, and the raw comparators (merged MODIS and S-NPP
    burning cells and fire power per cell-day, for the 2012 step check)."""
    d = d.assign(zk=zone_key(d["cell_lat"], d["cell_lon"], zone_deg))
    st = status[d["day"].to_numpy(), d["zk"].to_numpy()]
    tg = target[d["day"].to_numpy()]
    d = d.assign(st=st, tg=tg)
    parts = []

    def take(group, mask_fn, src, k_cells, k_frp):
        g = group_frame(d[mask_fn], group)
        if g.empty:
            return
        tk = tile_key(g["cell_lat"], g["cell_lon"], tile_deg)
        kc, kf = k_cells(tk, g), k_frp(tk, g)
        parts.append(pd.DataFrame({"day": g["day"].to_numpy(), "cell_lat": g["cell_lat"].to_numpy(),
                                   "cell_lon": g["cell_lon"].to_numpy(), "cells_h": kc,
                                   "frp_h": g["frp_sum_mw"].to_numpy(float) * kf, "src": src}))

    year = int(d["year"].iloc[0])
    month = int(d["month"].iloc[0])
    mvec = lambda g: np.full(len(g), month)  # noqa: E731
    one = lambda tk, g: np.ones(len(g))  # noqa: E731

    def fac(seg, var):
        t = tables.get(("factor", seg, var))
        return (lambda tk, g: t(tk, mvec(g))) if t is not None else one

    def fill(link, seg, var, then=None):
        t = tables.get(("fill", link, seg, var))
        f = (lambda tk, g: t(tk, mvec(g))) if t is not None else one
        return (lambda tk, g: f(tk, g) * then(tk, g)) if then is not None else f

    _, mseg = segment_for(min(year, 2011), segments)          # the MODIS-era piece
    dg, dseg = segment_for(year, segments)                      # the D19 piece for this year
    sel = lambda t, s: ((d["tg"] == t) & (d["st"] == s)).to_numpy()  # noqa: E731
    # S-NPP era
    take("S-NPP", sel("S-NPP", "primary"), 0, one, one)
    take("NOAA-20", sel("S-NPP", "filled: NOAA-20"), 1,
         fill("S-NPP <- NOAA-20", "NOAA-20", "cells"), fill("S-NPP <- NOAA-20", "NOAA-20", "frp"))
    take(dg, sel("S-NPP", "filled: D19"), 2,
         fill("S-NPP <- D19", dseg, "cells"), fill("S-NPP <- D19", dseg, "frp"))
    # MODIS era
    take("MODIS", sel("MODIS", "primary"), 3, fac(mseg, "cells"), fac(mseg, "frp"))
    take("Aqua", sel("MODIS", "filled: Aqua"), 4,
         fill("MODIS <- Aqua", "Aqua", "cells", fac(mseg, "cells")), fill("MODIS <- Aqua", "Aqua", "frp", fac(mseg, "frp")))
    take("Terra", sel("MODIS", "filled: Terra"), 5,
         fill("MODIS <- Terra", "Terra", "cells", fac(mseg, "cells")), fill("MODIS <- Terra", "Terra", "frp", fac(mseg, "frp")))
    # Terra only (Jan-Jun 2002)
    take("Terra", sel("Terra", "primary"), 6, fac(mseg, "cells"), fac(mseg, "frp"))

    if parts:
        x = pd.concat(parts, ignore_index=True)
        n25 = int(round(360 / res25))
        r = np.floor((x["cell_lat"].to_numpy() + 90) / res25).astype("int64")
        c = np.floor((x["cell_lon"].to_numpy() + 180) / res25).astype("int64")
        x["cell25"] = r * n25 + c
        out = x.groupby(["day", "cell25", "src"], as_index=False).agg(
            cells_h=("cells_h", "sum"), frp_h=("frp_h", "sum"), n_raw=("cells_h", "size"),
            lat=("cell_lat", "mean"), lon=("cell_lon", "mean"))
        out["lat"] = (-90 + (out["cell25"] // n25 + 0.5) * res25).astype("float32")
        out["lon"] = (-180 + (out["cell25"] % n25 + 0.5) * res25).astype("float32")
    else:
        out = pd.DataFrame(columns=["day", "cell25", "src", "cells_h", "frp_h", "n_raw", "lat", "lon"])
    raw = []
    for name in ("MODIS", "S-NPP"):
        g = group_frame(d, name)
        if len(g):
            raw.append(pd.DataFrame({"group": name, "cell_lat": g["cell_lat"].to_numpy(),
                                     "cell_lon": g["cell_lon"].to_numpy(), "frp": g["frp_sum_mw"].to_numpy(float)}))
    return out, (pd.concat(raw, ignore_index=True) if raw else pd.DataFrame())


def region_sums(lat, lon, values: dict, regions: dict) -> dict:
    out = {}
    for name, reg in regions.items():
        m = np.ones(len(lat), bool) if name == "global" else in_bbox(reg["bbox"], lon, lat)
        out[name] = {k: float(np.asarray(v)[m].sum()) for k, v in values.items()}
    return out


def uncertainty_table(settings: Settings, tile_deg: float) -> pd.DataFrame:
    """95% range of each factor (log half-width): the bootstrap range from stage factors, at
    least the honestly measured range for borrowed factors (stage places, part C), plus the
    confidence drift per tile (places, part D) for the MODIS-era piece."""
    fw = settings.processed_dir / "counts" / "factors_world.parquet"
    if not fw.exists():
        return pd.DataFrame()
    f = pd.read_parquet(fw)
    f = f[f["size"].astype(str) == "all"] if "size" in f else f
    with np.errstate(divide="ignore", invalid="ignore"):
        hw = np.log(np.where((f["k_lo"] > 0) & (f["k_hi"] > 0), f["k_hi"] / f["k_lo"], np.nan)) / 2
    f = f.assign(hw95_log=hw)
    bt = settings.processed_dir / "places" / "borrowed_factor_test.csv"
    if bt.exists():
        b = pd.read_csv(bt).set_index("level")["half_width_95_log"]
        meas = {"own all-year": b.get("own all-year"), "zone": b.get("zone (without the tile)"),
                "world": b.get("world (without the tile)")}
        for src, v in meas.items():
            if v is not None and np.isfinite(v):
                m = f["source"] == src
                f.loc[m, "hw95_log"] = np.fmax(f.loc[m, "hw95_log"], v)
    pc = settings.processed_dir / "places" / "place_choices.csv"
    if pc.exists():
        p = pd.read_csv(pc)
        if "confidence_drift_pct" in p:
            drift = np.log1p(f["tile"].map(p.set_index("tile")["confidence_drift_pct"]).fillna(0) / 100)
            modis = f["segment"].astype(str).str.startswith("MODIS")
            f.loc[modis, "hw95_log"] = np.sqrt(f.loc[modis, "hw95_log"] ** 2 + drift[modis] ** 2)
    f["range_pct"] = (100 * (np.exp(f["hw95_log"]) - 1)).round(1)
    return f[["segment", "tile", "month", "variable", "k", "source", "hw95_log", "range_pct"]]


def anomaly_test(settings: Settings, tile_deg: float, zone_deg: float, rule: dict) -> dict:
    """Plain vs era-fair classification of out-of-sample MODIS x k months (see module doc)."""
    fcfg = settings.raw.get("factors") or {}
    ref = fcfg.get("reference", "S-NPP")
    segs = source_segments(settings.raw.get("counts") or {})
    span = lambda key, dflt: list(range(int(fcfg.get(key, dflt)[0]), int(fcfg.get(key, dflt)[1]) + 1))  # noqa: E731
    years = span("overlap_years", [2012, 2025])
    g0, y0, y1 = segs[0]
    test_years = [y for y in years if int(y0) <= y <= int(y1)]
    groups = {g: GROUPS[g] for g in sorted({g for g, *_ in segs} | {ref})}
    bad = bad_days(settings, sorted({s for g in groups for s in GROUPS[g]})) if fcfg.get("exclude_gap_days", True) else None
    mask = final_mask(settings, masked_cells(settings), tile_deg)
    m = monthly_world(settings, groups, tile_deg, bad, None, mask, drop_zone_days=data_loss_keys(settings))
    if m.empty or len(test_years) < 3:
        return {"note": "not enough data"}
    m["tile"] = [tile_name(a, b, tile_deg) for a, b in zip(m["lat0"], m["lon0"])]
    r = m[(m["group"] == ref) & m["year"].isin(years) & ~m["in_mask"]]
    per_tile = r.groupby("tile")["active_cells"].sum() / (12 * len(years))
    tiles = per_tile[per_tile >= float(fcfg.get("min_ref_cells_per_month", 50))].index
    weights = r.groupby("tile")["active_cells"].sum()
    all_years = list(range(int(m["year"].min()), int(m["year"].max()) + 1))
    cube = Cube(m, "frp", True, False, zone_deg, all_years, tile_deg)
    prior, min_v = float(fcfg.get("prior_frp_mw", 300)), float(fcfg.get("min_frp_mw_per_month", 300))
    p = pd.concat([cube.predict(segs, ref, [x for x in years if x != y], [y], False, False, prior, min_v)
                   for y in test_years], ignore_index=True)
    p = p[p["region"].isin(tiles)]
    truth = cube.predict(segs, ref, years, years, False, False, prior, min_v)
    base = baseline(truth[["region", "year", "month", "real"]].rename(columns={"real": ref}), ref, years, [])
    p = p.merge(base, on=["region", "month"], how="left")
    sigma = None
    bt = settings.processed_dir / "places" / "borrowed_factor_test.csv"
    if bt.exists():
        b = pd.read_csv(bt).set_index("level")
        if "own month" in b.index:
            sigma = float(np.log1p(b.loc["own month", "median_month_abs_error_pct"] / 100) / 0.6745)
    if sigma is None:
        sigma = float(rule.get("sigma_fallback_log", 0.18))
    w = p["region"].map(weights).fillna(0).to_numpy()
    real_hi = classify(p["real"], p["mu"], p["sd"]).isin(HIGH).to_numpy()

    def rates(cls):
        hi = cls.isin(HIGH).to_numpy()
        fa = float(100 * w[hi & ~real_hi].sum() / w[hi].sum()) if w[hi].sum() else 0.0
        ca = float(100 * w[hi & real_hi].sum() / w[real_hi].sum()) if w[real_hi].sum() else 0.0
        return round(fa, 1), round(ca, 1), int(hi.sum())
    plain = rates(classify(p["pred"], p["mu"], p["sd"]))
    fair = rates(classify(p["pred"], p["mu"], np.sqrt(p["sd"] ** 2 + sigma ** 2)))
    fa_drop, ca_max = float(rule.get("fa_drop_points", 5)), float(rule.get("caught_max_drop_points", 5))
    adopt = (plain[0] - fair[0] >= fa_drop) and (plain[1] - fair[1] <= ca_max)
    return {"years": [test_years[0], test_years[-1]], "tiles": int(len(tiles)), "sigma_log": round(sigma, 3),
            "plain": {"false_alarm_pct": plain[0], "caught_pct": plain[1], "flagged_months": plain[2]},
            "era_fair": {"false_alarm_pct": fair[0], "caught_pct": fair[1], "flagged_months": fair[2]},
            "real_unusual_months": int(real_hi.sum()),
            "adopted": "era-fair" if adopt else "plain",
            "why": (f"false alarms {plain[0]}% -> {fair[0]}% (needed a drop of >= {fa_drop:g}), "
                    f"caught {plain[1]}% -> {fair[1]}% (allowed drop <= {ca_max:g})")}


def run_harmonise(settings: Settings) -> dict:
    cfg = settings.raw.get("harmonise") or {}
    fcfg = settings.raw.get("factors") or {}
    tile_deg = float(fcfg.get("tile_deg", 10))
    zone_deg = float((settings.raw.get("nodata") or {}).get("zone_deg", 30))
    res25 = float(cfg.get("resolution_deg", 0.25))
    segments = source_segments(settings.raw.get("counts") or {})
    root = settings.celldays_dir
    out_dir = settings.processed_dir / cfg.get("out_subdir", "l3")
    zp = settings.processed_dir / "nodata" / "zone_day_status.parquet"
    if not zp.exists():
        return {"stage": "harmonise", "note": "no zone_day_status: run stage nodata first"}
    if not (settings.processed_dir / "counts" / "factors_world.parquet").exists():
        return {"stage": "harmonise", "note": "no factors_world: run stage factors first"}
    zdates = pd.to_datetime(pd.read_parquet(zp, columns=["date"])["date"])
    start, end = zdates.min().normalize(), zdates.max().normalize()
    nd = (end - start).days + 1
    status, target = status_arrays(settings, start, nd, zone_deg)
    tables = load_tables(settings, tile_deg)
    _, mseg0 = segment_for(2011, segments)
    missing = [k for k in (("factor", mseg0, "cells"), ("factor", mseg0, "frp")) if k not in tables]
    if missing:
        return {"stage": "harmonise", "note": f"factor tables missing {missing}: rerun stage factors"}
    mask = final_mask(settings, masked_cells(settings), tile_deg)
    mask = np.asarray(mask if mask is not None else [], dtype="int64")
    regions = {"global": {"bbox": [-180, -90, 180, 90]}, **test_regions(settings)}

    months = sorted({(int(p.parent.name.split("=")[1]), int(p.name.split("=")[1]))
                     for p in root.glob("sensor_family=*/year=*/month=*")}) if root.exists() else []
    print(f"[harmonise] building L3 at {res25} degrees for {len(months)} months "
          f"({start.date()} to {end.date()}); {len(mask):,} non-fire cells removed in every year", flush=True)
    yearly, rows_written, src_counts = {}, 0, {}
    for i, (y, mo) in enumerate(months):
        parts = [root / f"sensor_family={f}" / f"year={y}" / f"month={mo}" for f in ("MODIS", "VIIRS")]
        parts = [p for p in parts if p.exists()]
        d = pd.concat([pd.read_parquet(p, columns=COLS) for p in parts], ignore_index=True)
        d["satellite"] = d["satellite"].astype(str)
        d = d[d["satellite"].isin(["Terra", "Aqua", "S-NPP", "NOAA-20"])]
        if len(mask):
            d = d[~np.isin(d["cell_id"].to_numpy("int64"), mask)]
        d["day"] = (pd.to_datetime(d["acq_date"]) - start).dt.days
        d = d[(d["day"] >= 0) & (d["day"] < nd)]
        if d.empty:
            continue
        d["year"], d["month"] = y, mo
        l3, raw = harmonise_month(d, status, target, tables, segments, tile_deg, zone_deg, res25)
        if len(l3):
            l3["date"] = (start + pd.to_timedelta(l3["day"], unit="D")).dt.date.astype(str)
            od = out_dir / f"year={y}" / f"month={mo}"
            od.mkdir(parents=True, exist_ok=True)
            l3 = l3.astype({"cell25": "int32", "src": "int8", "cells_h": "float32", "frp_h": "float32", "n_raw": "int32"})
            l3[["date", "cell25", "lat", "lon", "cells_h", "frp_h", "n_raw", "src"]].to_parquet(od / "part.parquet", index=False)
            rows_written += len(l3)
            for s, n in l3["src"].value_counts().items():
                src_counts[SOURCES[int(s)]] = src_counts.get(SOURCES[int(s)], 0) + int(n)
            hs = region_sums(l3["lat"], l3["lon"], {"cells_h": l3["cells_h"], "frp_h": l3["frp_h"]}, regions)
            for reg, v in hs.items():
                a = yearly.setdefault((reg, y), {})
                for k, val in v.items():
                    a[k] = a.get(k, 0.0) + val
        for name in ("MODIS", "S-NPP"):
            g = raw[raw["group"] == name] if len(raw) else raw
            if len(g):
                rs = region_sums(g["cell_lat"], g["cell_lon"], {f"{name}_cells": np.ones(len(g)), f"{name}_frp": g["frp"]}, regions)
                for reg, v in rs.items():
                    a = yearly.setdefault((reg, y), {})
                    for k, val in v.items():
                        a[k] = a.get(k, 0.0) + val
        if (i + 1) % 24 == 0:
            print(f"[harmonise] {i + 1}/{len(months)} months done ({y}-{mo:02d})", flush=True)

    yt = pd.DataFrame([{"region": r, "year": y, **v} for (r, y), v in yearly.items()]).sort_values(["region", "year"])
    out_dir.mkdir(parents=True, exist_ok=True)
    yt.to_csv(out_dir / "yearly_totals.csv", index=False)

    # GATE: the 2012 step
    gate = cfg.get("gate") or {}
    tol = float(gate.get("step_tolerance_pct", 5))
    yr = lambda k, dflt: list(range(int(gate.get(k, dflt)[0]), int(gate.get(k, dflt)[1]) + 1))  # noqa: E731
    before, after = yr("before_years", [2009, 2011]), yr("after_years", [2013, 2015])
    steps = []
    for reg, g in yt.groupby("region"):
        g = g.set_index("year")
        if not (set(before) | set(after)) <= set(g.index) or "MODIS_cells" not in g:
            continue
        if not (g["MODIS_cells"].reindex(before).fillna(0).mean() > 0 and g["cells_h"].reindex(before).fillna(0).mean() > 0):
            continue                                   # no fire in this region in those years
        row = {"region": reg}
        for var, raw_col in (("cells", "MODIS_cells"), ("frp", "MODIS_frp")):
            h = g[f"{var}_h"]
            row[f"{var}_step_harmonised"] = round(float(h[after].mean() / h[before].mean()), 3)
            row[f"{var}_step_modis_only"] = round(float(g[raw_col][after].mean() / g[raw_col][before].mean()), 3)
            row[f"{var}_gap_pct"] = round(100 * (row[f"{var}_step_harmonised"] / row[f"{var}_step_modis_only"] - 1), 1)
        if "S-NPP_cells" in g:
            raw_step = (g["S-NPP_cells"].reindex(after).mean()) / g["MODIS_cells"][before].mean()
            row["cells_step_raw_switch"] = round(float(raw_step), 3)
        steps.append(row)
    steps = pd.DataFrame(steps)
    steps.to_csv(out_dir / "step_2012.csv", index=False)
    glob = steps[steps["region"] == "global"].iloc[0] if (len(steps) and (steps["region"] == "global").any()) else None
    gate_pass = bool(glob is not None and abs(glob["cells_gap_pct"]) <= tol and abs(glob["frp_gap_pct"]) <= tol)

    unc = uncertainty_table(settings, tile_deg)
    if len(unc):
        unc.to_parquet(out_dir / "l3_uncertainty.parquet", index=False)
    anom = anomaly_test(settings, tile_deg, zone_deg, cfg.get("anomaly_rule") or {})

    gy = yt[yt["region"] == "global"].set_index("year")
    report = {"stage": "harmonise", "resolution_deg": res25, "rows": int(rows_written),
              "months": len(months), "rows_by_source": src_counts,
              "gate_2012_step": {"pass": gate_pass, "tolerance_pct": tol,
                                 "global": None if glob is None else {k: glob[k] for k in glob.index if k != "region"}},
              "uncertainty_median_range_pct": (None if unc.empty else
                                               {v: float(g["range_pct"].median()) for v, g in unc.groupby("variable")}),
              "anomaly_rule": anom,
              "final_model_D7": ("MODIS x k per 10-degree tile and calendar month (D21) on the D19 pieces, fitted with "
                                 "the D22+D26 mask and without D25 data-loss zone-days; S-NPP from 2012 with NOAA-20 and "
                                 "D19 fills (D25); fallback chain tile month -> tile all year -> zone -> world")}
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 80):
        print(f"[harmonise] L3 rows written: {rows_written:,}; by source: {src_counts}", flush=True)
        print("[harmonise] global yearly totals (harmonised, S-NPP scale):\n"
              + gy[["cells_h", "frp_h"]].round(0).astype("int64").to_string(), flush=True)
        print(f"[harmonise] 2012 step (mean {after[0]}-{after[-1]} / mean {before[0]}-{before[-1]}), harmonised vs MODIS alone:\n"
              + (steps.to_string(index=False) if len(steps) else "not enough years"), flush=True)
        print(f"[harmonise] GATE (global step within +-{tol:g}% of MODIS alone, both variables): "
              f"{'PASS' if gate_pass else 'FAIL'}", flush=True)
        if len(unc):
            print("[harmonise] factor 95% range, median % by variable and source:\n"
                  + unc.groupby(["variable", "source"])["range_pct"].median().to_string(), flush=True)
        print(f"[harmonise] anomaly RULE: {anom.get('adopted')} | {anom.get('why', anom.get('note'))}", flush=True)
    return report
