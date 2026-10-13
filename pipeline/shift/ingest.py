"""Stage 1: raw files -> cleaned Level 1 detections (Parquet).

Steps
  1. The adapter converts each raw file to Level 1 columns, chunk by chunk.
  2. Rows failing a filter are dropped and counted by reason.
  3. Surviving rows go to a staging area, split by sensor, year and month.
  4. Each month is de-duplicated (archive rows win over NRT) and written once.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from . import region, static_mask
from .adapters import get_adapter
from .config import Settings
from .contract import L1_COLUMNS, conform, validate_l1

DEDUP_KEYS = ["satellite", "acq_datetime_utc", "lat", "lon"]
PARTITIONS = ["sensor_family", "year", "month"]


def apply_filters(df: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, Counter]:
    """Drop rows that fail any filter. Each row is counted once, under the first reason it fails."""
    f = settings["filters"]
    coords_ok = df["lat"].between(-90, 90) & df["lon"].between(-180, 180)
    in_region = region.inside(settings, df["lon"].fillna(0), df["lat"].fillna(0))
    outside = (pd.Series(~in_region, index=df.index) if in_region is not None
               else pd.Series(False, index=df.index))
    unknown = ~df["type_known"]
    static = pd.Series(False, index=df.index)
    if unknown.any():
        hit = static_mask.in_mask(settings, df["lat"], df["lon"])
        if hit is not None:
            static = pd.Series(hit, index=df.index) & unknown
    reasons: list[tuple[str, pd.Series]] = [
        ("bad_datetime", df["acq_datetime_utc"].isna()),
        ("bad_coordinates", ~coords_ok),
        ("outside_region", outside),
        ("unknown_satellite", ~df["satellite"].isin(["Terra", "Aqua", "S-NPP", "NOAA-20", "NOAA-21"])),
        ("bad_frp", df["frp_mw"].isna() | ~df["frp_mw"].between(f["frp_min_mw"], f["frp_max_mw"])),
        ("bad_daynight", ~df["daynight"].isin(["D", "N"])),
        ("confidence_filtered", ~df["confidence_class"].isin(f["keep_confidence"])),
        ("non_vegetation_type", df["type_known"] & ~df["fire_type"].isin(f["keep_fire_types"])),
        ("static_source_masked", static),
        ("unknown_type_dropped", (~df["type_known"]) & (not f["keep_unknown_type"])),
    ]
    keep = pd.Series(True, index=df.index)
    counts: Counter = Counter()
    for name, bad in reasons:
        hit = keep & bad
        counts[name] += int(hit.sum())
        keep &= ~bad
    return df[keep], counts


def _expand_zips(files: list[Path], settings: Settings):
    """Yield CSV paths. A FIRMS .zip is unpacked one CSV at a time into local scratch
    space (never into raw/), processed, then deleted, so Drive only holds the zip."""
    for path in files:
        if path.suffix.lower() != ".zip":
            yield path
            continue
        tmp_root = settings["fetch"].get("tmp_dir") if "fetch" in settings.raw else None
        work = Path(tempfile.mkdtemp(prefix="shift_zip_", dir=tmp_root))
        try:
            with zipfile.ZipFile(path) as zf:
                for member in sorted(n for n in zf.namelist() if n.lower().endswith(".csv")):
                    target = work / Path(member).name
                    with zf.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
                    yield target
                    target.unlink(missing_ok=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)


def run_ingest(settings: Settings, files: list[Path] | None = None) -> dict:
    adapter = get_adapter(settings["inputs"]["adapter"])
    chunk_rows = int(settings["processing"]["chunk_rows"])
    if files is None:
        files = sorted({p for pat in settings["inputs"]["patterns"] for p in settings.raw_dir.glob(pat)})
    if not files:
        # Nothing to ingest (for example the global dataset before any recent-months
        # download). Clear old detections so aggregate does not reuse stale data.
        shutil.rmtree(settings.detections_dir, ignore_errors=True)
        return {"stage": "ingest", "files": {}, "totals": {"rows_read": 0, "rows_final": 0},
                "note": f"no input files in {settings.raw_dir}"}

    out_dir = settings.detections_dir
    staging = settings.processed_dir / "_staging_detections"
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    staging.mkdir(parents=True)

    report = {"stage": "ingest", "files": {}, "totals": Counter()}
    for path in _expand_zips(files, settings):
        file_counts: Counter = Counter()
        for frame in adapter.read(path, settings, chunk_rows):
            file_counts["rows_read"] += len(frame)
            kept, rejected = apply_filters(frame, settings)
            file_counts.update(rejected)
            file_counts["rows_kept"] += len(kept)
            if len(kept):
                kept.to_parquet(staging, partition_cols=PARTITIONS, index=False)
        report["files"][path.name] = dict(file_counts)
        report["totals"].update(file_counts)

    dup_total = 0
    for part in sorted(staging.glob("sensor_family=*/year=*/month=*")):
        df = pd.read_parquet(part)
        family = part.parent.parent.name.split("=", 1)[1]
        year = int(part.parent.name.split("=", 1)[1])
        month = int(part.name.split("=", 1)[1])
        df["sensor_family"], df["year"], df["month"] = family, year, month
        # Archive rows sort before NRT rows, so keep="first" prefers archive.
        df["_nrt"] = df["source_product"].str.endswith("_NRT")
        df = df.sort_values(["_nrt", *DEDUP_KEYS], kind="stable")
        before = len(df)
        df = df.drop_duplicates(DEDUP_KEYS, keep="first").drop(columns="_nrt")
        dup_total += before - len(df)
        df = conform(df.reset_index(drop=True), L1_COLUMNS)
        validate_l1(df)
        target = out_dir / f"sensor_family={family}" / f"year={year}" / f"month={month}"
        target.mkdir(parents=True, exist_ok=True)
        df.drop(columns=PARTITIONS).to_parquet(target / "part-0.parquet", index=False)

    shutil.rmtree(staging, ignore_errors=True)
    report["totals"]["duplicates_removed"] = dup_total
    report["totals"]["rows_final"] = report["totals"]["rows_kept"] - dup_total
    report["totals"] = dict(report["totals"])
    return report


def read_detections(settings: Settings, **filters) -> pd.DataFrame:
    """Read Level 1 back with partition columns restored, e.g. read_detections(s, sensor_family='MODIS')."""
    return _read_partitioned(settings.detections_dir, L1_COLUMNS, filters)


def _read_partitioned(root: Path, columns: dict[str, str], filters: dict) -> pd.DataFrame:
    if not root.exists():
        return conform(pd.DataFrame({c: pd.Series(dtype=t) for c, t in columns.items()}), columns)
    pq_filters = [(k, "==", v) for k, v in filters.items()] or None
    df = pd.read_parquet(root, filters=pq_filters)
    for col in PARTITIONS:
        df[col] = df[col].astype(str)
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    return conform(df, columns)


def write_report(report: dict, settings: Settings, name: str) -> Path:
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    path = settings.processed_dir / f"report_{name}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path
