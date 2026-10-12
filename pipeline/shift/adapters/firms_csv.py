"""Adapter for NASA FIRMS active-fire CSV files (archive and NRT, MODIS and VIIRS).

Converts each file to Level 1 columns (unfiltered). Filtering happens in ingest.py
so every rejected row can be counted by reason.

If the official data arrives with different column names, add the new names to
COLUMN_ALIASES. If the format is completely different, write a new adapter module
with the same `read()` signature and point `inputs.adapter` in settings.yaml at it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from ..contract import L1_COLUMNS, conform

# Alternative spellings mapped to the FIRMS standard names.
COLUMN_ALIASES = {
    "lat": "latitude",
    "lon": "longitude",
    "long": "longitude",
    "acq_hhmm": "acq_time",
    "fire_type": "type",
}

SATELLITE_NAMES = {
    # MODIS
    "terra": "Terra", "t": "Terra",
    "aqua": "Aqua", "a": "Aqua",
    # VIIRS
    "n": "S-NPP", "npp": "S-NPP", "snpp": "S-NPP", "suomi npp": "S-NPP", "s-npp": "S-NPP",
    "1": "NOAA-20", "n20": "NOAA-20", "noaa-20": "NOAA-20", "noaa20": "NOAA-20", "j1": "NOAA-20",
    "2": "NOAA-21", "n21": "NOAA-21", "noaa-21": "NOAA-21", "noaa21": "NOAA-21", "j2": "NOAA-21",
}

VIIRS_CONFIDENCE = {"l": "low", "low": "low", "n": "nominal", "nominal": "nominal", "h": "high", "high": "high"}


def detect_family(columns: list[str]) -> str:
    cols = {c.lower() for c in columns}
    if "bright_ti4" in cols:
        return "VIIRS"
    if "brightness" in cols:
        return "MODIS"
    raise ValueError(f"cannot tell MODIS from VIIRS from columns: {sorted(cols)}")


def read(path: Path, settings, chunk_rows: int) -> Iterator[pd.DataFrame]:
    """Yield Level 1 frames (unfiltered) from one FIRMS CSV file."""
    path = Path(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    family = detect_family(header)
    is_nrt = "nrt" in path.name.lower()
    mapping = settings["confidence_mapping"]

    for chunk in pd.read_csv(path, dtype=str, chunksize=chunk_rows, keep_default_na=False):
        chunk.columns = [COLUMN_ALIASES.get(c.lower(), c.lower()) for c in chunk.columns]
        yield _to_l1(chunk, family, is_nrt, path.name, mapping)


def _to_l1(raw: pd.DataFrame, family: str, is_nrt: bool, file_name: str, mapping: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)

    hhmm = raw["acq_time"].str.strip().str.replace(":", "", regex=False).str.zfill(4)
    stamp = raw["acq_date"].str.strip() + " " + hhmm
    out["acq_datetime_utc"] = pd.to_datetime(stamp, format="%Y-%m-%d %H%M", errors="coerce", utc=True)
    out["acq_date"] = out["acq_datetime_utc"].dt.tz_localize(None).dt.normalize()
    out["year"] = out["acq_datetime_utc"].dt.year.fillna(0).astype("int16")
    out["month"] = out["acq_datetime_utc"].dt.month.fillna(0).astype("int8")

    out["lat"] = pd.to_numeric(raw["latitude"], errors="coerce")
    out["lon"] = pd.to_numeric(raw["longitude"], errors="coerce")
    out["frp_mw"] = pd.to_numeric(raw["frp"], errors="coerce")
    out["scan_km"] = pd.to_numeric(raw.get("scan"), errors="coerce") if "scan" in raw else np.nan
    out["track_km"] = pd.to_numeric(raw.get("track"), errors="coerce") if "track" in raw else np.nan

    out["sensor_family"] = family
    sat = raw["satellite"].str.strip().str.lower().map(SATELLITE_NAMES)
    out["satellite"] = sat.fillna("UNKNOWN")

    conf_raw = raw["confidence"].str.strip()
    out["confidence_raw"] = conf_raw
    if family == "MODIS":
        num = pd.to_numeric(conf_raw, errors="coerce")
        cls = np.where(num < mapping["modis_low_below"], "low",
                       np.where(num >= mapping["modis_high_from"], "high", "nominal"))
        out["confidence_class"] = pd.Series(cls, index=raw.index).where(num.notna(), "UNKNOWN")
    else:
        out["confidence_class"] = conf_raw.str.lower().map(VIIRS_CONFIDENCE).fillna("UNKNOWN")

    out["daynight"] = raw["daynight"].str.strip().str.upper()

    if "type" in raw.columns:
        ftype = pd.to_numeric(raw["type"], errors="coerce")
        out["type_known"] = ftype.notna()
        out["fire_type"] = ftype.fillna(-1).astype("int8")
    else:
        out["type_known"] = False
        out["fire_type"] = np.int8(-1)

    kind = "NRT" if is_nrt else "archive"
    out["source_product"] = family + "_" + out["satellite"].str.replace("-", "", regex=False) + "_" + kind
    out["source_file"] = file_name
    return conform(out, L1_COLUMNS)
