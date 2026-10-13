"""Fetch stage: FIRMS yearly world files -> Level 2 cell-days, one year at a time.

FIRMS publishes one zip per sensor per year containing every country's detections
(`<prefix>_<year>_all_countries.zip`, one CSV per country inside). For each year this stage:

  1. downloads the zip to local scratch space (not Drive),
  2. reads each country CSV through the normal adapter and filters,
  3. aggregates to 0.05 deg cell-days and merges cells that cross country borders,
  4. writes the cell-day partitions to the processed folder (Drive),
  5. writes a small "done" marker and deletes the zip.

Raw detections are never stored, which keeps Drive usage small. Because every year leaves
a marker, an interrupted run (Colab disconnect) simply continues where it stopped.

    python -m shift.run --stage fetch                     # every missing year, both sensors
    python -m shift.run --stage fetch --years 2020        # one year (test)
    python -m shift.run --stage fetch --family MODIS      # one sensor
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from .adapters import get_adapter
from .aggregate import combine_partials, to_celldays, write_celldays
from .config import Settings
from .ingest import apply_filters

USER_AGENT = "SHIFT-pipeline/0.0 (NASA Space Apps 2026, Team Squadron Zero)"


def zip_url(base_url: str, prefix: str, year: int) -> str:
    return f"{base_url.rstrip('/')}/{prefix}_{year}_all_countries.zip"


def download(url: str, dest: Path, retries: int = 3) -> dict | None:
    """Download url to dest.

    Returns a fingerprint {sha256, bytes, last_modified} so every result can be traced to
    the exact file NASA served, or None if the file does not exist (403/404).
    """
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            h = hashlib.sha256()
            n = 0
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
                last_modified = resp.headers.get("Last-Modified") if hasattr(resp, "headers") else None
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    h.update(block)
                    out.write(block)
                    n += len(block)
            return {"sha256": h.hexdigest(), "bytes": n, "last_modified": last_modified}
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if isinstance(getattr(e, "reason", None), FileNotFoundError):
                return None
            err = e
        time.sleep(3 * attempt)
    raise RuntimeError(f"download failed after {retries} attempts: {url} ({err})")


def process_zip(zip_path: Path, family: str, settings: Settings, work: Path, out_dir: Path) -> tuple[list[str], dict]:
    """Turn one yearly world zip into cell-day partitions in out_dir.

    Country files are aggregated one by one into partial cell-days on local scratch,
    split by month. Each month is then merged on its own, so memory use depends on the
    busiest month, not the whole year.
    """
    adapter = get_adapter(settings["inputs"]["adapter"])
    chunk_rows = int(settings["processing"]["chunk_rows"])
    counts: Counter = Counter()
    staging = work / "partials"
    n_part = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        counts["files"] = len(members)
        for member in members:
            path = Path(zf.extract(member, work))
            for frame in adapter.read(path, settings, chunk_rows):
                if (frame["sensor_family"] != family).any():
                    raise ValueError(f"{member}: expected {family} data")
                counts["rows_read"] += len(frame)
                kept, rejected = apply_filters(frame, settings)
                counts.update(rejected)
                counts["rows_kept"] += len(kept)
                if len(kept):
                    part = to_celldays(kept, settings.resolution)
                    for month, m in part.groupby("month"):
                        d = staging / f"month={int(month)}"
                        d.mkdir(parents=True, exist_ok=True)
                        m.to_parquet(d / f"p{n_part:06d}.parquet", index=False)
                        n_part += 1
            path.unlink()
    written: list[str] = []
    counts["celldays"] = 0
    for d in sorted(staging.glob("month=*"), key=lambda p: int(p.name.split("=")[1])):
        cd = combine_partials(pd.concat([pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))],
                                        ignore_index=True))
        written += write_celldays(cd, out_dir)
        counts["celldays"] += len(cd)
        shutil.rmtree(d)
    return written, dict(counts)


def run_fetch(settings: Settings, years: list[int] | None = None, families: list[str] | None = None,
              force: bool = False) -> dict:
    cfg = settings["fetch"]
    done_dir = settings.processed_dir / "_fetch_done"
    done_dir.mkdir(parents=True, exist_ok=True)
    out_dir = settings.celldays_dir
    scratch_root = Path(cfg["tmp_dir"]) if cfg.get("tmp_dir") else Path(tempfile.gettempdir())

    report = {"stage": "fetch", "dataset": settings.dataset, "years": {}, "missing": [], "skipped": []}
    for family, spec in cfg["products"].items():
        if families and family not in families:
            continue
        for year in range(int(spec["first_year"]), int(cfg["last_year"]) + 1):
            if years and year not in years:
                continue
            key = f"{family}_{year}"
            marker = done_dir / f"{key}.json"
            if marker.exists() and not force:
                report["skipped"].append(key)
                continue
            t0 = time.time()
            work = Path(tempfile.mkdtemp(prefix=f"shift_{key}_", dir=scratch_root))
            try:
                zip_path, used, fp = work / f"{key}.zip", None, None
                for prefix in spec["prefixes"]:
                    url = zip_url(cfg["base_url"], prefix, year)
                    fp = download(url, zip_path)
                    if fp:
                        used = url
                        break
                if not used:
                    report["missing"].append(key)
                    print(f"[fetch] {key}: no file found (tried {spec['prefixes']})", flush=True)
                    continue
                size_mb = round(zip_path.stat().st_size / 1e6, 1)
                if marker.exists():  # forced rerun: remove this year's old partitions first
                    for rel in json.loads(marker.read_text()).get("partitions", []):
                        shutil.rmtree(out_dir / rel, ignore_errors=True)
                    marker.unlink()
                partitions, counts = process_zip(zip_path, family, settings, work, out_dir)
                info = {"url": used, "zip_mb": size_mb, "sha256": fp["sha256"], "bytes": fp["bytes"],
                        "last_modified": fp["last_modified"], **counts, "partitions": partitions,
                        "seconds": round(time.time() - t0, 1)}
                marker.write_text(json.dumps(info, indent=1))
                report["years"][key] = {k: v for k, v in info.items() if k != "partitions"}
                print(f"[fetch] {key}: {size_mb} MB, {counts.get('rows_read', 0):,} rows read, "
                      f"{counts.get('rows_kept', 0):,} kept, "
                      f"{counts.get('celldays', 0):,} cell-days, {info['seconds']} s", flush=True)
            finally:
                shutil.rmtree(work, ignore_errors=True)
    return report
