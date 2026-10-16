"""Phase 2 options B and C: outside data for the checks (stage `external`, a download stage).

B. INDEPENDENT TRUTH. GFED4 monthly burned area WITHOUT small fires (GFED4.0_MQ_YYYYMM_BA.hdf,
   0.25 degrees, hectares; ORNL DAAC, doi:10.3334/ORNLDAAC/1293; 1995-2016). It is mapped from
   MODIS surface reflectance (MCD64A1), one sensor family on both sides of 2012, so a fire record
   with an instrument jump at 2012 would show a jump in its ratio to burned area. (GFED4.1s adds
   small fires estimated FROM MODIS active fires, which would not be independent: not used.)
C. CLOUD. MERRA-2 monthly mean total cloud area fraction CLDTOT (M2TMNXRAD v5.12.4,
   0.5 x 0.625 degrees; GES DISC). A cloudy month and a fire-free month look the same in fire
   detections; stage checks tests whether cloud explains low months and scaling errors.

Both are reduced at once to small tables and the downloaded files are deleted:
  external/gfed_monthly_regions.csv   burned area (ha) per month for the world, the D18 regions
                                      and the known-event boxes
  external/gfed_monthly_fine.parquet  burned area (ha) per 2.5-degree cell and month (cells with any)
  external/cloud_monthly_fine.parquet CLDTOT per 2.5-degree cell and month (mean of MERRA-2 cells)

Earthdata Login: through the environment variables EARTHDATA_USERNAME and EARTHDATA_PASSWORD,
set from Colab Secrets in the notebook (never typed into a cell, a file or a chat). GES DISC data
also needs "NASA GESDISC DATA ARCHIVE" approved once under Applications in the Earthdata profile.

    pip install earthaccess pyhdf
    python -m shift.run --dataset global --stage external        (~10-30 min, needs internet)
"""
from __future__ import annotations

import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .harmonise import check_regions
from .region import in_bbox

GFED_PATTERN = re.compile(r"GFED4\.0_MQ_(\d{4})(\d{2})_BA\.hdf$")
MERRA_PATTERN = re.compile(r"\.(\d{4})(\d{2})\.nc4$")


# ---------------------------------------------------------------- reducers (pure, tested)

def gfed_grid() -> tuple[np.ndarray, np.ndarray]:
    """Centres of the GFED 0.25-degree grid; row 0 is the north edge (89.875 N)."""
    lat = 89.875 - 0.25 * np.arange(720)
    lon = -179.875 + 0.25 * np.arange(1440)
    return lat, lon


def to_fine(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, deg: float, how: str) -> pd.DataFrame:
    """Sum or mean of a lat x lon field over `deg` cells: rows lat0, lon0, value."""
    la = np.floor((np.clip(lat, -89.999, 89.999) + 90) / deg).astype("int64")
    lo = np.floor((np.clip(lon, -180, 179.999) + 180) / deg).astype("int64")
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    df = pd.DataFrame({"li": LA.ravel(), "lo": LO.ravel(), "v": field.ravel().astype(float)})
    df = df[np.isfinite(df["v"])]
    g = df.groupby(["li", "lo"], as_index=False)["v"].agg(how)
    return pd.DataFrame({"lat0": -90 + g["li"] * deg, "lon0": -180 + g["lo"] * deg, "value": g["v"]})


def region_totals(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, regions: dict) -> dict:
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    out = {}
    for name, bb in regions.items():
        m = np.ones(LA.shape, bool) if name == "global" else in_bbox(bb, LO, LA)
        out[name] = float(np.nansum(field[m]))
    return out


def pick_unit(yearly_global_ha: pd.Series, lo_mha: float, hi_mha: float) -> float:
    """GFED4 global burned area is about 350 Mha a year. The multiplier (1, 0.01 or 100) that puts
    the stored values in that range; raises if none does, so a wrong unit can never pass silently."""
    med = float(yearly_global_ha.median()) / 1e6
    for f in (1.0, 0.01, 100.0):
        if lo_mha <= med * f <= hi_mha:
            return f
    raise ValueError(f"GFED global burned area {med:.1f} Mha a year fits no known unit")


# ---------------------------------------------------------------- downloads

def _login():
    import earthaccess
    auth = earthaccess.login(strategy="environment")
    if not getattr(auth, "authenticated", False):
        raise RuntimeError("Earthdata login failed: set EARTHDATA_USERNAME and EARTHDATA_PASSWORD from Colab Secrets")
    return earthaccess


def _links(earthaccess, short_name: str, version: str | None, y0: int, y1: int) -> list[str]:
    kw = {"short_name": short_name, "temporal": (f"{y0}-01-01", f"{y1}-12-31")}
    if version:
        kw["version"] = version
    out = []
    for g in earthaccess.search_data(**kw):
        out += [u for u in g.data_links(access="external") if u.startswith("https")]
    return sorted(set(out))


def _read_hdf4(path: Path, name: str) -> np.ndarray:
    from pyhdf.SD import SD, SDC
    f = SD(str(path), SDC.READ)
    names = [n for n in f.datasets() if name.lower() in n.lower() and "uncert" not in n.lower()]
    if not names:
        raise KeyError(f"{name} not in {path.name}: {list(f.datasets())}")
    sds = f.select(names[0])
    a = sds.get().astype(float)
    att = sds.attributes()
    a = a * float(att.get("scale_factor", 1.0)) + float(att.get("add_offset", 0.0))
    f.end()
    return np.where(a >= 0, a, np.nan)


def gfed(settings: Settings, cfg: dict, earthaccess, tmp: Path) -> dict:
    y0, y1 = (int(v) for v in cfg.get("years", [2003, 2016]))
    deg = float(cfg.get("cell_deg", 2.5))
    links = [u for u in _links(earthaccess, cfg.get("short_name", "fire_emissions_v4_R1_1293"), None, y0, y1)
             if GFED_PATTERN.search(u) and y0 <= int(GFED_PATTERN.search(u).group(1)) <= y1]
    if not links:
        return {"note": "no GFED4.0_MQ files found"}
    lat, lon = gfed_grid()
    regions = check_regions(settings)
    reg_rows, fine_rows = [], []
    print(f"[external] GFED4 burned area: {len(links)} monthly files", flush=True)
    for i in range(0, len(links), 24):
        batch = links[i:i + 24]
        paths = _retry(lambda: earthaccess.download(batch, str(tmp)), f"GFED batch {i // 24 + 1}")
        for p in map(Path, paths):
            mt = GFED_PATTERN.search(p.name)
            if not mt:
                continue
            y, mo = int(mt.group(1)), int(mt.group(2))
            ba = _read_hdf4(p, "BurnedArea")
            reg_rows += [{"region": r, "year": y, "month": mo, "ba_ha": v} for r, v in region_totals(ba, lat, lon, regions).items()]
            f = to_fine(ba, lat, lon, deg, "sum")
            fine_rows.append(f[f["value"] > 0].assign(year=y, month=mo))
            p.unlink()
        print(f"[external] GFED {min(i + 24, len(links))}/{len(links)}", flush=True)
    reg = pd.DataFrame(reg_rows)
    got = reg[["year", "month"]].drop_duplicates().shape[0] if len(reg) else 0
    if got != len(links):              # a download that silently returned fewer files must not pass
        raise RuntimeError(f"GFED: {got} of {len(links)} months read; run the stage again with --force")
    fine = pd.concat(fine_rows, ignore_index=True).rename(columns={"value": "ba_ha"})
    unit = pick_unit(reg[reg["region"] == "global"].groupby("year")["ba_ha"].sum(), 200, 600)
    reg["ba_ha"] *= unit
    fine["ba_ha"] *= unit
    out = settings.processed_dir / "external"
    reg.to_csv(out / "gfed_monthly_regions.csv", index=False)
    fine.to_parquet(out / "gfed_monthly_fine.parquet", index=False)
    yearly = (reg[reg["region"] == "global"].groupby("year")["ba_ha"].sum() / 1e6).round(1)
    return {"files": len(links), "unit_multiplier": unit, "global_burned_area_mha_by_year": yearly.to_dict(),
            "source": "GFED4.0 monthly burned area without small fires, ORNL DAAC doi:10.3334/ORNLDAAC/1293"}


def _retry(fn, what: str, waits=(5, 15, 30, 60, 120, 240)):
    """Run fn(); on failure wait and try again (NASA servers sometimes answer 502/503 for a while)."""
    last = None
    for i, w in enumerate((0,) + tuple(waits)):
        if w:
            print(f"[external] {what}: attempt {i + 1} after {w} s ({type(last).__name__}: {str(last)[:120]})", flush=True)
            time.sleep(w)
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"{what} failed after {len(waits) + 1} attempts: {type(last).__name__}: {last}")


def _merra_field(session, url: str, var: str, tmp: Path, earthaccess) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """CLDTOT, lat, lon of one monthly file and how it was fetched: an OPeNDAP subset (about 1 MB)
    when the server allows it, otherwise the whole file (both retried on server errors)."""
    import h5py
    p = tmp / "merra.nc4"
    od = url.replace("https://data.gesdisc.earthdata.nasa.gov/data/", "https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/")
    how = None
    if od != url:
        try:
            r = session.get(f"{od}.nc4?{var},lat,lon", timeout=120)
            if r.status_code == 200 and r.content[:4] == b"\x89HDF":
                p.write_bytes(r.content)
                how = "opendap"
        except Exception:  # noqa: BLE001 - fall back to the whole file
            how = None
    if how is None:
        def full():
            got = earthaccess.download([url], str(tmp))
            if not got:
                raise RuntimeError("no file returned")
            return Path(got[0])
        p = _retry(full, f"MERRA-2 {Path(url).name}")
        how = "file"
    with h5py.File(p, "r") as f:
        a = np.asarray(f[var][...], float).squeeze()
        lat, lon = np.asarray(f["lat"][...], float), np.asarray(f["lon"][...], float)
        fill = f[var].attrs.get("_FillValue")
    p.unlink()
    if fill is not None:
        a = np.where(np.isclose(a, np.asarray(fill).ravel()[0]), np.nan, a)
    return a, lat, lon, how


def merra(settings: Settings, cfg: dict, earthaccess, tmp: Path) -> dict:
    """Month by month into external/merra_parts/ (a rerun skips months already done), then one table."""
    y0, y1 = (int(v) for v in cfg.get("years", [2003, 2026]))
    deg = float(cfg.get("cell_deg", 2.5))
    var = str(cfg.get("variable", "CLDTOT"))
    links = _retry(lambda: [u for u in _links(earthaccess, cfg.get("short_name", "M2TMNXRAD"), str(cfg.get("version", "5.12.4")), y0, y1)
                            if MERRA_PATTERN.search(u)], "MERRA-2 search")
    if not links:
        return {"note": "no MERRA-2 files found"}
    parts = settings.processed_dir / "external" / "merra_parts"
    parts.mkdir(parents=True, exist_ok=True)
    session = earthaccess.get_requests_https_session()
    how, failed = {"opendap": 0, "file": 0, "already done": 0}, []
    print(f"[external] MERRA-2 {var}: {len(links)} monthly files", flush=True)
    for i, u in enumerate(links):
        mt = MERRA_PATTERN.search(u)
        y, mo = int(mt.group(1)), int(mt.group(2))
        pp = parts / f"{y:04d}{mo:02d}.parquet"
        if pp.exists():
            how["already done"] += 1
            continue
        try:
            a, lat, lon, h = _merra_field(session, u, var, tmp, earthaccess)
        except Exception as e:  # noqa: BLE001 - record and go on; a rerun picks the month up again
            failed.append(f"{y:04d}-{mo:02d}: {str(e)[:160]}")
            continue
        how[h] += 1
        to_fine(a, lat, lon, deg, "mean").rename(columns={"value": "cloud"}).assign(year=y, month=mo).to_parquet(pp, index=False)
        if (i + 1) % 24 == 0:
            print(f"[external] MERRA-2 {i + 1}/{len(links)} ({how})", flush=True)
    files = sorted(parts.glob("*.parquet"))
    if not files:
        return {"error": "no MERRA-2 month could be read", "failed": failed}
    c = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if not (0 <= c["cloud"].min() and c["cloud"].max() <= 1.0001):
        raise ValueError(f"{var} outside 0-1: {c['cloud'].min()} to {c['cloud'].max()}")
    if failed:                           # never leave a table that looks complete when it is not
        (settings.processed_dir / "external" / "cloud_monthly_fine.parquet").unlink(missing_ok=True)
        return {"error": f"{len(failed)} of {len(links)} months failed after retries: run the stage again "
                         "(finished months are kept)", "failed": failed[:10], "fetched": how}
    c.to_parquet(settings.processed_dir / "external" / "cloud_monthly_fine.parquet", index=False)
    return {"files": len(links), "variable": var, "mean_cloud": round(float(c["cloud"].mean()), 3),
            "months": int(c[["year", "month"]].drop_duplicates().shape[0]), "fetched": how,
            "source": "MERRA-2 M2TMNXRAD v5.12.4 CLDTOT (GMAO), NASA GES DISC"}


def run_external(settings: Settings, years=None, families=None, force: bool = False) -> dict:
    cfg = settings.raw.get("external") or {}
    out = settings.processed_dir / "external"
    out.mkdir(parents=True, exist_ok=True)
    want = [f.lower() for f in families] if families else ["gfed", "merra2"]
    report = {"stage": "external"}
    try:
        ea = _login()
    except Exception as e:  # noqa: BLE001
        return {**report, "note": f"no Earthdata login: {e}"}
    tmp = Path(tempfile.mkdtemp(prefix="shift_external_"))
    try:
        for name, fn, done in (("gfed", gfed, "gfed_monthly_regions.csv"), ("merra2", merra, "cloud_monthly_fine.parquet")):
            if name not in want:
                continue
            if (out / done).exists() and not force:
                report[name] = {"note": f"already done ({done}); use --force to redo"}
                continue
            try:
                report[name] = fn(settings, {**cfg.get(name, {}), "cell_deg": cfg.get("cell_deg", 2.5)}, ea, tmp)
            except Exception as e:  # noqa: BLE001 - one source failing must not lose the other
                report[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[external] {name}: {report[name]}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return report
