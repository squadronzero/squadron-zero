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

Phase 2 package (16 Oct):
  * fills follow the method stage nodata chose for each link (D25): "substitute" fills every
    gap; "hybrid" fills only where the 10-degree tile has its own monthly fill factor (both
    variables) and leaves the rest to SCALING, recorded per tile and month in
    scale_tile_months.parquet (days seen, days scaled, multiplier). Daily rows stay observations;
    the *_est columns of the region tables apply the multipliers (the month as if fully seen);
  * dates are UTC, or the local solar day when stage daily's rule adopted it (option F);
  * monthly_regions.csv and daily_regions.csv: harmonised and raw sums per test region, known-
    event box and the world, for stage checks;
  * the anomaly rule is no longer run here: stage checks runs it on the final factors.

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
from .factors import final_mask, zone_key
from .provenance import stamp
from .region import in_bbox, test_regions
from .solar import local_day_shift
from .static_mask import masked_cells
from .tiles import tile_name

COLS = ["acq_date", "cell_id", "cell_lat", "cell_lon", "satellite", "frp_sum_mw", "n_day", "n_night"]
SOURCES = {0: "S-NPP", 1: "NOAA-20 fill", 2: "D19 MODIS fill", 3: "MODIS x k",
           4: "Aqua fill x k", 5: "Terra fill x k", 6: "Terra only x k (not comparable)"}


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
        self.deg = deg
        nt = int(round(180 / deg)) * int(round(360 / deg))
        a = np.full((nt, 12), np.nan)
        if len(df):
            keys = np.array([key_of_name(t, deg) for t in df["tile"]])
            a[keys, df["month"].to_numpy(int) - 1] = df["k"].to_numpy(float)
        fb = np.nanmedian(np.where(np.isfinite(a) & (a > 0), a, np.nan), axis=0) if np.isfinite(a).any() else np.ones(12)
        fb = np.where(np.isfinite(fb), fb, 1.0)
        self.a = np.where(np.isfinite(a) & (a > 0), a, fb[None, :])

    def __call__(self, lat, lon, month: np.ndarray) -> np.ndarray:
        return self.a[tile_key(lat, lon, self.deg), month - 1]


def factor_source(settings: Settings) -> tuple:
    """Which MODIS -> S-NPP factor table the record uses: the fine table from stage resolution
    when its rule (fixed before the run) adopted it (fix 9b), else the 10-degree table from stage factors.
    Returns (path, degrees, measured half-widths for borrowed levels or None)."""
    import json
    fcfg = settings.raw.get("factors") or {}
    coarse = (settings.processed_dir / "counts" / "factors_world.parquet", float(fcfg.get("tile_deg", 10)), None)
    if (settings.raw.get("harmonise") or {}).get("use_fine_factors", "auto") is False:
        return coarse
    rp = settings.processed_dir / "counts" / "resolution_choice.json"
    fp = settings.processed_dir / "counts" / "factors_world_fine.parquet"
    if not (rp.exists() and fp.exists()):
        return coarse
    rep = json.loads(rp.read_text())
    if not rep.get("adopted"):
        return coarse
    return fp, float(rep.get("fine_deg", 2.5)), rep.get("measured_half_width_95_log")


def load_tables(settings: Settings, fill_deg: float, factor_path=None, factor_deg: float | None = None) -> dict:
    """Factor tables: MODIS -> S-NPP per D19 piece (stage factors or resolution) and the fill
    links (stage nodata, on 10-degree tiles)."""
    out = {}
    fw = factor_path or settings.processed_dir / "counts" / "factors_world.parquet"
    if fw.exists():
        f = pd.read_parquet(fw)
        f = f[f["size"].astype(str) == "all"] if "size" in f else f
        for (seg, var), g in f.groupby(["segment", "variable"]):
            out[("factor", seg, var)] = Table(g, factor_deg or fill_deg)
    ff = settings.processed_dir / "nodata" / "fill_factors.parquet"
    if ff.exists():
        f = pd.read_parquet(ff)
        for (link, seg, var), g in f.groupby(["link", "segment", "variable"]):
            out[("fill", link, seg, var)] = Table(g, fill_deg)
            out[("own", link, seg, var)] = own_table(g, fill_deg)
    return out


def own_table(df: pd.DataFrame, deg: float) -> np.ndarray:
    """True per (tile, month) where the fill factor is the tile's own monthly factor (hybrid)."""
    nt = int(round(180 / deg)) * int(round(360 / deg))
    a = np.zeros((nt, 12), bool)
    if len(df) and "source" in df:
        d = df[df["source"].astype(str) == "own month"]
        if len(d):
            keys = np.array([key_of_name(t, deg) for t in d["tile"]])
            a[keys, d["month"].to_numpy(int) - 1] = True
    return a


def fill_methods(settings: Settings) -> dict:
    """The method stage nodata chose per link (D25): nodata/fill_choice.json, else the saved
    nodata report. Setting harmonise.fill_method = substitute forces substitution everywhere."""
    import json
    forced = (settings.raw.get("harmonise") or {}).get("fill_method", "auto")
    for p in (settings.processed_dir / "nodata" / "fill_choice.json", settings.processed_dir / "report_nodata.json"):
        if p.exists():
            chosen = json.loads(p.read_text()).get("chosen") or {}
            return {k: ("substitute" if forced == "substitute" and v != "scale" else v) for k, v in chosen.items()}
    return {}


def day_basis(settings: Settings) -> str:
    """utc or local: harmonise.day_basis, where auto = stage daily's decision."""
    import json
    b = str((settings.raw.get("harmonise") or {}).get("day_basis", "auto"))
    if b in ("utc", "local"):
        return b
    p = settings.processed_dir / "daily" / "day_basis.json"
    return json.loads(p.read_text()).get("basis", "utc") if p.exists() else "utc"


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
                    res25: float, methods: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """L3 rows for one month of cell-days, and the raw comparators (merged MODIS and S-NPP
    burning cells and fire power per cell-day, for the 2012 step check).

    `d["uday"]` (UTC day) decides the zone-day status; `d["day"]` (UTC or local solar day) is the
    day the value is written to. A link whose method is "hybrid" keeps a filled cell-day only where
    its 10-degree tile has its own monthly fill factor for both variables; the rest is scaled."""
    methods = methods or {}
    uday = d["uday"].to_numpy() if "uday" in d else d["day"].to_numpy()
    d = d.assign(zk=zone_key(d["cell_lat"], d["cell_lon"], zone_deg))
    st = status[uday, d["zk"].to_numpy()]
    tg = target[uday]
    d = d.assign(st=st, tg=tg)
    parts = []

    year = int(d["year"].iloc[0])
    month = int(d["month"].iloc[0])

    def own_both(link, seg):
        oc, of = tables.get(("own", link, seg, "cells")), tables.get(("own", link, seg, "frp"))
        if oc is None or of is None:
            return None
        return lambda g: (oc & of)[tile_key(g["cell_lat"].to_numpy(), g["cell_lon"].to_numpy(), tile_deg), month - 1]

    def take(group, mask_fn, src, k_cells, k_frp, link=None, seg=None):
        g = group_frame(d[mask_fn], group)
        if g.empty:
            return
        if link is not None and methods.get(link) == "hybrid":
            own = own_both(link, seg)
            if own is not None:
                g = g[own(g)]
                if g.empty:
                    return
        kc, kf = k_cells(g), k_frp(g)
        parts.append(pd.DataFrame({"day": g["day"].to_numpy(), "cell_lat": g["cell_lat"].to_numpy(),
                                   "cell_lon": g["cell_lon"].to_numpy(), "cells_h": kc,
                                   "frp_h": g["frp_sum_mw"].to_numpy(float) * kf, "src": src}))

    mvec = lambda g: np.full(len(g), month)  # noqa: E731
    one = lambda g: np.ones(len(g))  # noqa: E731

    def look(t):
        return lambda g: t(g["cell_lat"].to_numpy(), g["cell_lon"].to_numpy(), mvec(g))

    def fac(seg, var):
        t = tables.get(("factor", seg, var))
        return look(t) if t is not None else one

    def fill(link, seg, var, then=None):
        t = tables.get(("fill", link, seg, var))
        f = look(t) if t is not None else one
        return (lambda g: f(g) * then(g)) if then is not None else f

    _, mseg = segment_for(min(year, 2011), segments)          # the MODIS-era piece
    dg, dseg = segment_for(year, segments)                      # the D19 piece for this year
    sel = lambda t, s: ((d["tg"] == t) & (d["st"] == s)).to_numpy()  # noqa: E731
    # S-NPP era
    take("S-NPP", sel("S-NPP", "primary"), 0, one, one)
    take("NOAA-20", sel("S-NPP", "filled: NOAA-20"), 1,
         fill("S-NPP <- NOAA-20", "NOAA-20", "cells"), fill("S-NPP <- NOAA-20", "NOAA-20", "frp"),
         "S-NPP <- NOAA-20", "NOAA-20")
    take(dg, sel("S-NPP", "filled: D19"), 2,
         fill("S-NPP <- D19", dseg, "cells"), fill("S-NPP <- D19", dseg, "frp"), "S-NPP <- D19", dseg)
    # MODIS era
    take("MODIS", sel("MODIS", "primary"), 3, fac(mseg, "cells"), fac(mseg, "frp"))
    take("Aqua", sel("MODIS", "filled: Aqua"), 4,
         fill("MODIS <- Aqua", "Aqua", "cells", fac(mseg, "cells")), fill("MODIS <- Aqua", "Aqua", "frp", fac(mseg, "frp")),
         "MODIS <- Aqua", "Aqua")
    take("Terra", sel("MODIS", "filled: Terra"), 5,
         fill("MODIS <- Terra", "Terra", "cells", fac(mseg, "cells")), fill("MODIS <- Terra", "Terra", "frp", fac(mseg, "frp")),
         "MODIS <- Terra", "Terra")
    # Terra only (Jan-Jun 2002)
    take("Terra", sel("Terra", "primary"), 6, fac(mseg, "cells"), fac(mseg, "frp"))

    if parts:
        x = pd.concat(parts, ignore_index=True)
        n25 = int(round(360 / res25))
        r = np.floor((x["cell_lat"].to_numpy() + 90) / res25).astype("int64")
        c = np.floor((x["cell_lon"].to_numpy() + 180) / res25).astype("int64")
        x["cell25"] = r * n25 + c
        out = x.groupby(["day", "cell25", "src"], as_index=False).agg(
            cells_h=("cells_h", "sum"), frp_h=("frp_h", "sum"), n_raw=("cells_h", "size"))
        out["lat"] = (-90 + (out["cell25"] // n25 + 0.5) * res25).astype("float32")
        out["lon"] = (-180 + (out["cell25"] % n25 + 0.5) * res25).astype("float32")
    else:
        out = pd.DataFrame(columns=["day", "cell25", "src", "cells_h", "frp_h", "n_raw", "lat", "lon"])
    raw = []
    for name in ("MODIS", "S-NPP"):
        g = group_frame(d, name)
        if len(g):
            raw.append(pd.DataFrame({"group": name, "day": g["day"].to_numpy(), "cell_lat": g["cell_lat"].to_numpy(),
                                     "cell_lon": g["cell_lon"].to_numpy(), "frp": g["frp_sum_mw"].to_numpy(float)}))
    return out, (pd.concat(raw, ignore_index=True) if raw else pd.DataFrame())


def scale_month(status, target, tables: dict, methods: dict, day0: int, day1: int, month: int,
                tile_deg: float, zone_deg: float) -> pd.DataFrame:
    """Per 10-degree tile for one month: days in the record, days left to scaling (hybrid fills
    where the tile has no own factor, plus "no data" days of its zone) and the multiplier that
    turns the month's observed total into a full-month estimate."""
    nlat, nlon = int(round(180 / tile_deg)), int(round(360 / tile_deg))
    tk = np.arange(nlat * nlon)
    tlat = -90 + (tk // nlon + 0.5) * tile_deg
    tlon = -180 + (tk % nlon + 0.5) * tile_deg
    zk = zone_key(tlat, tlon, zone_deg)
    st = status[day0:day1]
    rec = (target[day0:day1] != "").sum()
    if rec == 0:
        return pd.DataFrame()
    scaled = np.zeros(len(tk))
    for link, meth in methods.items():
        if meth != "hybrid":
            continue
        tgt, sub = [x.strip() for x in link.split("<-")]
        on = target[day0:day1] == tgt
        filled = (st[on] == f"filled: {sub}").sum(0)                   # per zone
        seg = sub if sub != "D19" else None
        own = None
        for key, t in tables.items():
            if len(key) == 4 and key[0] == "own" and key[1] == link and (seg is None or key[2] == seg):
                own = t[:, month - 1] if own is None else own & t[:, month - 1]
        if own is None:
            own = np.zeros(len(tk), bool)
        scaled += np.where(own, 0, filled[zk])
    nodata = (st == "no data").sum(0)[zk]
    seen = rec - scaled - nodata
    mult = np.where(seen > 0, rec / np.maximum(seen, 1), 1.0)
    keep = (scaled + nodata) > 0
    return pd.DataFrame({"tile": [tile_name(a - tile_deg / 2, b - tile_deg / 2, tile_deg) for a, b in zip(tlat[keep], tlon[keep])],
                         "tile_key": tk[keep], "month": month, "days": int(rec), "days_scaled_hybrid": scaled[keep].astype(int),
                         "days_no_data": nodata[keep].astype(int), "multiplier": mult[keep]})


def region_sums(lat, lon, values: dict, regions: dict) -> dict:
    out = {}
    for name, reg in regions.items():
        m = np.ones(len(lat), bool) if name == "global" else in_bbox(reg["bbox"], lon, lat)
        out[name] = {k: float(np.asarray(v)[m].sum()) for k, v in values.items()}
    return out


def uncertainty_table(settings: Settings, factor_path=None, measured: dict | None = None) -> pd.DataFrame:
    """95% range of each factor (log half-width): the bootstrap range from stage factors, at
    least the honestly measured range for borrowed factors (stage places, part C), plus the
    confidence drift per tile (places, part D) for the MODIS-era piece."""
    fw = factor_path or settings.processed_dir / "counts" / "factors_world.parquet"
    if not fw.exists():
        return pd.DataFrame()
    f = pd.read_parquet(fw)
    f = f[f["size"].astype(str) == "all"] if "size" in f else f
    with np.errstate(divide="ignore", invalid="ignore"):
        hw = np.log(np.where((f["k_lo"] > 0) & (f["k_hi"] > 0), f["k_hi"] / f["k_lo"], np.nan)) / 2
    f = f.assign(hw95_log=hw)
    bt = settings.processed_dir / "places" / "borrowed_factor_test.csv"
    if measured:                             # fine factors: borrowed levels measured by stage resolution
        for src, v in measured.items():
            if v is not None and np.isfinite(v):
                m = f["source"] == src
                f.loc[m, "hw95_log"] = np.fmax(f.loc[m, "hw95_log"], v)
    elif bt.exists():
        b = pd.read_csv(bt).set_index("level")["half_width_95_log"]
        meas = {"own all-year": b.get("own all-year"), "zone": b.get("zone (without the tile)"),
                "world": b.get("world (without the tile)")}
        for src, v in meas.items():
            if v is not None and np.isfinite(v):
                m = f["source"] == src
                f.loc[m, "hw95_log"] = np.fmax(f.loc[m, "hw95_log"], v)
    import json
    rp = settings.processed_dir / "counts" / "resolution_choice.json"
    if measured and rp.exists():                # option E: regions still jumping at 2012 carry that gap
        for fl in json.loads(rp.read_text()).get("region_flags") or []:
            m = in_bbox(fl["bbox"], f["lon"], f["lat"])
            f.loc[m, "hw95_log"] = np.sqrt(f.loc[m, "hw95_log"] ** 2 + float(fl["extra_hw95_log"]) ** 2)
    pc = settings.processed_dir / "places" / "place_choices.csv"
    if pc.exists():
        p = pd.read_csv(pc)
        if "confidence_drift_pct" in p:
            drift = np.log1p(f["tile"].map(p.set_index("tile")["confidence_drift_pct"]).fillna(0) / 100)
            modis = f["segment"].astype(str).str.startswith("MODIS")
            f.loc[modis, "hw95_log"] = np.sqrt(f.loc[modis, "hw95_log"] ** 2 + drift[modis] ** 2)
    f["range_pct"] = (100 * (np.exp(f["hw95_log"]) - 1)).round(1)
    return f[["segment", "tile", "lat", "lon", "month", "variable", "k", "source", "hw95_log", "range_pct"]]


def check_regions(settings: Settings) -> dict:
    """Areas summed for stage checks: the world, the D18 test regions and the known-event boxes
    (settings phase2_checks.events), each {name: bbox}."""
    out = {"global": [-180.0, -90.0, 180.0, 90.0]}
    out.update({k: v["bbox"] for k, v in test_regions(settings).items()})
    for name, e in ((settings.raw.get("phase2_checks") or {}).get("events") or {}).items():
        out[f"event:{name}"] = [float(x) for x in e["bbox"]]
    return out


def area_sums(day, lat, lon, values: dict, regions: dict) -> pd.DataFrame:
    """Daily sums of `values` per bbox area."""
    out = []
    base = pd.DataFrame({"day": np.asarray(day), **{k: np.asarray(v, float) for k, v in values.items()}})
    for name, bb in regions.items():
        m = np.ones(len(base), bool) if name == "global" else in_bbox(bb, lon, lat)
        if m.any():
            out.append(base[m].groupby("day", as_index=False).sum().assign(region=name))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def run_harmonise(settings: Settings) -> dict:
    import json
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
    fpath, fdeg, measured = factor_source(settings)
    tables = load_tables(settings, tile_deg, fpath, fdeg)
    methods = fill_methods(settings)
    basis_name = day_basis(settings)
    hours = (settings.raw.get("daily") or {}).get("overpass_hours")
    print(f"[harmonise] MODIS -> S-NPP factors: {fpath.name} ({fdeg:g}-degree cells); fill methods {methods or 'not found'}; "
          f"days: {basis_name}", flush=True)
    if not methods:
        return {"stage": "harmonise", "note": "fill methods of stage nodata not found (nodata/fill_choice.json or "
                                              "report_nodata.json): rerun stage nodata"}
    _, mseg0 = segment_for(2011, segments)
    missing = [k for k in (("factor", mseg0, "cells"), ("factor", mseg0, "frp")) if k not in tables]
    if missing:
        return {"stage": "harmonise", "note": f"factor tables missing {missing}: rerun stage factors"}
    mask = final_mask(settings, masked_cells(settings), tile_deg)
    mask = np.asarray(mask if mask is not None else [], dtype="int64")
    regions = check_regions(settings)

    months = sorted({(int(p.parent.name.split("=")[1]), int(p.name.split("=")[1]))
                     for p in root.glob("sensor_family=*/year=*/month=*")}) if root.exists() else []
    print(f"[harmonise] building L3 at {res25} degrees for {len(months)} months "
          f"({start.date()} to {end.date()}); {len(mask):,} non-fire cells removed in every year", flush=True)
    rows_written, src_counts = 0, {}
    daily_parts, scale_parts, spill = [], [], {}
    write = lambda l3, od: l3[["date", "cell25", "lat", "lon", "cells_h", "frp_h", "n_raw", "src"]].to_parquet(od / "part.parquet", index=False)  # noqa: E731
    for i, (y, mo) in enumerate(months):
        parts = [root / f"sensor_family={f}" / f"year={y}" / f"month={mo}" for f in ("MODIS", "VIIRS")]
        parts = [p for p in parts if p.exists()]
        d = pd.concat([pd.read_parquet(p, columns=COLS) for p in parts], ignore_index=True)
        d["satellite"] = d["satellite"].astype(str)
        d = d[d["satellite"].isin(["Terra", "Aqua", "S-NPP", "NOAA-20"])]
        if len(mask):
            d = d[~np.isin(d["cell_id"].to_numpy("int64"), mask)]
        d["uday"] = (pd.to_datetime(d["acq_date"]) - start).dt.days
        d = d[(d["uday"] >= 0) & (d["uday"] < nd)]
        if d.empty:
            continue
        d["day"] = d["uday"] + (local_day_shift(d["cell_lon"], d["satellite"], d["n_day"], d["n_night"], hours)
                                if basis_name == "local" else 0)
        d = d[(d["day"] >= 0) & (d["day"] < nd)]
        d["year"], d["month"] = y, mo
        l3, raw = harmonise_month(d, status, target, tables, segments, tile_deg, zone_deg, res25, methods)
        m0 = (pd.Timestamp(y, mo, 1) - start).days
        m1 = (pd.Timestamp(y, mo, 1) + pd.offsets.MonthBegin(1) - start).days
        sc = scale_month(status, target, tables, methods, max(m0, 0), min(m1, nd), mo, tile_deg, zone_deg)
        mult = np.ones(int(round(180 / tile_deg)) * int(round(360 / tile_deg)))
        if len(sc):
            scale_parts.append(sc.assign(year=y))
            mult[sc["tile_key"].to_numpy()] = sc["multiplier"].to_numpy()
        if len(l3):
            l3["date"] = (start + pd.to_timedelta(l3["day"], unit="D")).dt.date.astype(str)
            l3 = l3.astype({"cell25": "int32", "src": "int8", "cells_h": "float32", "frp_h": "float32", "n_raw": "int32"})
            for s_, n in l3["src"].value_counts().items():
                src_counts[SOURCES[int(s_)]] = src_counts.get(SOURCES[int(s_)], 0) + int(n)
            rows_written += len(l3)
            mul = mult[tile_key(l3["lat"], l3["lon"], tile_deg)]
            daily_parts.append(area_sums(l3["day"], l3["lat"], l3["lon"],
                                         {"cells_h": l3["cells_h"], "frp_h": l3["frp_h"],
                                          "cells_h_est": l3["cells_h"] * mul, "frp_h_est": l3["frp_h"] * mul},
                                         regions))
            own = l3["date"].str[:7] == f"{y:04d}-{mo:02d}"
            for (yy, mm), g in l3[~own].groupby([l3["date"].str[:4].astype(int), l3["date"].str[5:7].astype(int)]):
                spill.setdefault((int(yy), int(mm)), []).append(g)
            if own.any():
                od = out_dir / f"year={y}" / f"month={mo}"
                od.mkdir(parents=True, exist_ok=True)
                write(l3[own], od)
        if len(raw):
            daily_parts.append(area_sums(raw["day"], raw["cell_lat"], raw["cell_lon"],
                                         {"MODIS_cells": (raw["group"] == "MODIS").astype(float),
                                          "MODIS_frp": raw["frp"].where(raw["group"] == "MODIS", 0.0),
                                          "S-NPP_cells": (raw["group"] == "S-NPP").astype(float),
                                          "S-NPP_frp": raw["frp"].where(raw["group"] == "S-NPP", 0.0)}, regions))
        if (i + 1) % 24 == 0:
            print(f"[harmonise] {i + 1}/{len(months)} months done ({y}-{mo:02d})", flush=True)
    for (yy, mm), gs in spill.items():                 # local solar days that fall in the next/previous month
        od = out_dir / f"year={yy}" / f"month={mm}"
        od.mkdir(parents=True, exist_ok=True)
        old = [pd.read_parquet(od / "part.parquet")] if (od / "part.parquet").exists() else []
        x = pd.concat(old + gs, ignore_index=True).groupby(["date", "cell25", "src"], as_index=False).agg(
            lat=("lat", "first"), lon=("lon", "first"), cells_h=("cells_h", "sum"), frp_h=("frp_h", "sum"), n_raw=("n_raw", "sum"))
        write(x.astype({"cell25": "int32", "src": "int8", "cells_h": "float32", "frp_h": "float32", "n_raw": "int32"}), od)

    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["cells_h", "frp_h", "cells_h_est", "frp_h_est", "MODIS_cells", "MODIS_frp", "S-NPP_cells", "S-NPP_frp"]
    dl = pd.concat([x for x in daily_parts if len(x)], ignore_index=True)
    dl = dl.groupby(["region", "day"], as_index=False)[[c for c in cols if c in dl]].sum()
    dl = dl.reindex(columns=["region", "day", *cols]).fillna(0.0)
    dl["date"] = (start + pd.to_timedelta(dl["day"], unit="D")).dt.date.astype(str)
    dl["year"], dl["month"] = dl["date"].str[:4].astype(int), dl["date"].str[5:7].astype(int)
    # share of each bbox area's 30-degree zones with no data that day (for the dated runs of stage checks)
    nd_share = []
    for name, bb in regions.items():
        zl = [z for la in np.arange(-90, 90, zone_deg) for lo in np.arange(-180, 180, zone_deg)
              if la < bb[3] and la + zone_deg > bb[1] and lo < bb[2] and lo + zone_deg > bb[0]
              for z in [int(zone_key(la + 0.5 * zone_deg, lo + 0.5 * zone_deg, zone_deg))]]
        share = (status[:, zl] == "no data").mean(1) if zl else np.zeros(nd)
        nd_share.append(pd.DataFrame({"region": name, "day": np.arange(nd), "no_data_share": share}))
    full = pd.concat(nd_share, ignore_index=True)
    full["date"] = (start + pd.to_timedelta(full["day"], unit="D")).dt.date.astype(str)
    daily = full.merge(dl.drop(columns=["date", "year", "month"]), on=["region", "day"], how="left").fillna(0.0)
    daily = daily[["region", "date", "cells_h", "frp_h", "MODIS_cells", "MODIS_frp", "S-NPP_cells", "S-NPP_frp", "no_data_share"]]
    daily.to_parquet(out_dir / "daily_regions.parquet", index=False)
    monthly = dl.groupby(["region", "year", "month"], as_index=False)[cols].sum()
    monthly.to_csv(out_dir / "monthly_regions.csv", index=False)
    scales = pd.concat(scale_parts, ignore_index=True) if scale_parts else pd.DataFrame()
    scales.to_parquet(out_dir / "scale_tile_months.parquet", index=False)

    yt = monthly.groupby(["region", "year"], as_index=False)[cols].sum()
    yt = yt[~yt["region"].str.contains(":")].sort_values(["region", "year"])
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

    unc = uncertainty_table(settings, fpath, measured)
    if len(unc):
        unc.to_parquet(out_dir / "l3_uncertainty.parquet", index=False)
    gy = yt[yt["region"] == "global"].set_index("year")
    prov = stamp(settings)
    hybrid_days = int(scales["days_scaled_hybrid"].sum()) if len(scales) else 0
    record = {"provenance": prov, "day_basis": basis_name, "fill_methods": methods, "factor_table": fpath.name,
              "factor_resolution_deg": fdeg, "sources": {str(k): v for k, v in SOURCES.items()},
              "scaled_tile_days_hybrid": hybrid_days,
              "note": ("Daily rows are observations. Months with gap days left to scaling: multiply by "
                       "scale_tile_months.multiplier of the 10-degree tile (already applied in the *_est columns).")}
    (out_dir / "record_info.json").write_text(json.dumps(record, indent=2, default=str))
    report = {"stage": "harmonise", "resolution_deg": res25, "factor_resolution_deg": fdeg, "rows": int(rows_written),
              "months": len(months), "rows_by_source": src_counts, "day_basis": basis_name, "fill_methods": methods,
              "tile_days_scaled_hybrid": hybrid_days,
              "gate_2012_step": {"pass": gate_pass, "tolerance_pct": tol,
                                 "global": None if glob is None else {k: glob[k] for k in glob.index if k != "region"}},
              "uncertainty_median_range_pct": (None if unc.empty else
                                               {v: float(g["range_pct"].median()) for v, g in unc.groupby("variable")}),
              "anomaly_rule": {"note": "run by stage checks on the final factors"},
              "final_model_D7": ("MODIS x k per 2.5-degree cell and calendar month (pulled towards its 10-degree tile; D28) "
                                 "on the D19 pieces, fitted with the D22+D26 mask and without D25 data-loss zone-days, with "
                                 "the version-test fit years and region choices of stage resolution; S-NPP from 2012 with "
                                 "NOAA-20 and D19 fills by the methods of D25")}
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 80):
        print(f"[harmonise] L3 rows written: {rows_written:,}; by source: {src_counts}", flush=True)
        print(f"[harmonise] tile-days left to scaling by hybrid fills: {hybrid_days:,}", flush=True)
        print("[harmonise] global yearly totals (harmonised, S-NPP scale):\n"
              + gy[["cells_h", "frp_h"]].round(0).astype("int64").to_string(), flush=True)
        print(f"[harmonise] 2012 step (mean {after[0]}-{after[-1]} / mean {before[0]}-{before[-1]}), harmonised vs MODIS alone:\n"
              + (steps.to_string(index=False) if len(steps) else "not enough years"), flush=True)
        print(f"[harmonise] GATE (global step within +-{tol:g}% of MODIS alone, both variables): "
              f"{'PASS' if gate_pass else 'FAIL'}", flush=True)
        if len(unc):
            print("[harmonise] factor 95% range, median % by variable and source:\n"
                  + unc.groupby(["variable", "source"])["range_pct"].median().to_string(), flush=True)
    return report
