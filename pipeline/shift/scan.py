"""Scan stage: one read-only pass over the FIRMS yearly world files.

The fetch stage keeps only cleaned cell-days. This stage downloads each yearly file again
(fast on Colab) and records what fetch does not keep:

  * data version labels per year (a change of version can mean a change of processing,
    which can put a hidden step into the record),
  * the location of every non-fire heat source FIRMS labels itself (type 1 volcano,
    2 static land source such as gas flares and industry, 3 offshore), counted per grid
    cell and year. These build the static-source mask used for data without a type field
    (near-real-time files) and are the raw material for a "non-fire heat sources" layer,
  * a fingerprint (sha256, size, Last-Modified) of each file, added to the fetch marker
    when the file is the same size as the one fetch used.

Nothing in the cell-day record is changed. Resumable: finished years leave a marker in
processed/<dataset>/_scan_done/.

    python -m shift.run --dataset global --stage scan
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from .config import Settings
from .fetch import download, zip_url
from .grid import cell_centre, cell_index

NON_FIRE_TYPES = {1: "volcano", 2: "static_land_source", 3: "offshore"}
COLUMNS = ["latitude", "longitude", "version", "type"]


def static_dir(settings: Settings) -> Path:
    return settings.path(settings["static_mask"]["sources_dir"])


def scan_zip(zip_path: Path, resolution: float, chunk_rows: int) -> tuple[dict, pd.DataFrame]:
    versions: Counter = Counter()
    static_parts = []
    rows = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        for member in members:
            with zf.open(member) as fh:
                for ch in pd.read_csv(fh, usecols=lambda c: c.lower() in COLUMNS, dtype=str,
                                      chunksize=chunk_rows, keep_default_na=False):
                    ch.columns = [c.lower() for c in ch.columns]
                    rows += len(ch)
                    versions.update(ch["version"].str.strip().value_counts().to_dict())
                    if "type" not in ch:
                        continue
                    t = pd.to_numeric(ch["type"], errors="coerce")
                    nf = ch[t.isin(list(NON_FIRE_TYPES))]
                    if nf.empty:
                        continue
                    cells = cell_index(pd.to_numeric(nf["latitude"]), pd.to_numeric(nf["longitude"]), resolution)
                    static_parts.append(pd.DataFrame({"cell_id": cells, "type": t[nf.index].astype("int8")})
                                        .value_counts().rename("n_detections").reset_index())
    static = (pd.concat(static_parts).groupby(["cell_id", "type"], as_index=False)["n_detections"].sum()
              if static_parts else pd.DataFrame({"cell_id": pd.Series(dtype="int64"),
                                                 "type": pd.Series(dtype="int8"),
                                                 "n_detections": pd.Series(dtype="int64")}))
    return {"files": len(members), "rows": rows, "versions": dict(versions)}, static


def build_static_mask(settings: Settings) -> dict:
    """Combine the per-year static-source files into one mask of grid cells."""
    cfg = settings["static_mask"]
    src = static_dir(settings)
    families = [k for k, v in settings["fetch"]["products"].items() if v.get("scan", True)]
    pat = re.compile(r"^(%s)_(\d{4})$" % "|".join(map(re.escape, families)))
    files = sorted(f for f in src.glob("*.parquet") if pat.match(f.stem)) if src.exists() else []
    if not files:
        return {"cells": 0, "note": "no static-source files yet; run the scan stage"}
    allp = pd.concat([pd.read_parquet(f).assign(year=int(f.stem.rsplit("_", 1)[1])) for f in files], ignore_index=True)
    per_cell = allp.groupby("cell_id").agg(
        n_years=("year", "nunique"),
        n_detections=("n_detections", "sum"),
        main_type=("type", lambda s: int(s.mode().iloc[0])),
    ).reset_index()
    mask = per_cell[per_cell["n_years"] >= int(cfg["min_years"])].copy()
    mask["cell_lat"], mask["cell_lon"] = cell_centre(mask["cell_id"], settings.resolution)
    out = settings.path(cfg["path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    mask.to_parquet(out, index=False)
    return {"cells": int(len(mask)), "min_years": int(cfg["min_years"]),
            "by_type": {NON_FIRE_TYPES[k]: int(v) for k, v in mask["main_type"].value_counts().items()},
            "path": str(out)}


def run_scan(settings: Settings, years: list[int] | None = None, families: list[str] | None = None,
             force: bool = False) -> dict:
    cfg = settings["fetch"]
    done_dir = settings.processed_dir / "_scan_done"
    fetch_done = settings.processed_dir / "_fetch_done"
    done_dir.mkdir(parents=True, exist_ok=True)
    out_dir = static_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = Path(cfg["tmp_dir"]) if cfg.get("tmp_dir") else Path(tempfile.gettempdir())
    chunk_rows = int(settings["processing"]["chunk_rows"])

    report = {"stage": "scan", "dataset": settings.dataset, "years": {}, "missing": [], "skipped": []}
    for family, spec in cfg["products"].items():
        if (families and family not in families) or not spec.get("scan", True):
            continue
        for year in range(int(spec["first_year"]), int(cfg["last_year"]) + 1):
            if years and year not in years:
                continue
            key = f"{family}_{year}"
            marker = done_dir / f"{key}.json"
            if marker.exists() and not force:
                report["skipped"].append(key)
                report["years"][key] = json.loads(marker.read_text())
                continue
            t0 = time.time()
            work = Path(tempfile.mkdtemp(prefix=f"shift_scan_{key}_", dir=scratch_root))
            try:
                fmarker = fetch_done / f"{key}.json"
                finfo = json.loads(fmarker.read_text()) if fmarker.exists() else {}
                urls = [finfo["url"]] if finfo.get("url") else [zip_url(cfg["base_url"], p, year) for p in spec["prefixes"]]
                zip_path, used, fp = work / f"{key}.zip", None, None
                for url in urls:
                    fp = download(url, zip_path)
                    if fp:
                        used = url
                        break
                if not used:
                    report["missing"].append(key)
                    print(f"[scan] {key}: no file found", flush=True)
                    continue
                info, static = scan_zip(zip_path, settings.resolution, chunk_rows)
                static.to_parquet(out_dir / f"{family}_{year}.parquet", index=False)
                same = bool(finfo) and round(fp["bytes"] / 1e6, 1) == finfo.get("zip_mb")
                if same and "sha256" not in finfo:
                    finfo.update({"sha256": fp["sha256"], "bytes": fp["bytes"],
                                  "last_modified": fp["last_modified"], "fingerprint_from": "scan"})
                    fmarker.write_text(json.dumps(finfo, indent=1))
                result = {"url": used, **fp, "same_file_as_fetch": same if finfo else None, **info,
                          "static_cells": int(static["cell_id"].nunique()),
                          "static_detections": {NON_FIRE_TYPES[int(k)]: int(v) for k, v in
                                                static.groupby("type")["n_detections"].sum().items()},
                          "seconds": round(time.time() - t0, 1)}
                marker.write_text(json.dumps(result, indent=1))
                report["years"][key] = result
                vers = ", ".join(f"{v} ({n:,})" for v, n in sorted(info["versions"].items()))
                print(f"[scan] {key}: versions {vers} | non-fire cells {result['static_cells']:,} | "
                      f"same file as fetch: {result['same_file_as_fetch']} | {result['seconds']} s", flush=True)
            finally:
                shutil.rmtree(work, ignore_errors=True)

    # Version summary: which label each family uses, year by year, and where it changes.
    changes = {}
    for family in {k.split("_")[0] for k in report["years"]}:
        seq = sorted((int(k.split("_")[1]), max(v["versions"], key=v["versions"].get))
                     for k, v in report["years"].items() if k.startswith(family + "_") and v.get("versions"))
        mixed = [y for y, _ in seq if len(report["years"][f"{family}_{y}"]["versions"]) > 1]
        changes[family] = {"labels": sorted({lab for _, lab in seq}),
                           "changes_at": [y for (y0, a), (y, b) in zip(seq, seq[1:]) if a != b],
                           "years_with_mixed_labels": mixed}
    report["version_summary"] = changes
    report["static_mask"] = build_static_mask(settings)
    print("[scan] version summary:", json.dumps(changes), flush=True)
    print("[scan] static mask:", json.dumps(report["static_mask"]), flush=True)
    return report
