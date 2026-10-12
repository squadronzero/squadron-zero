"""Stage 2: Level 1 detections -> Level 2 cell-days on the SHIFT grid."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from .config import Settings
from .contract import L1_COLUMNS, L2_COLUMNS, conform, validate_l2
from .grid import cell_centre, cell_index
from .ingest import PARTITIONS, _read_partitioned

GROUP_KEYS = ["acq_date", "cell_id", "satellite"]
SUM_COLS = ["n_detections", "frp_sum_mw", "n_day", "n_night", "n_high_conf", "n_type_unknown"]
MANIFEST = "_aggregate_partitions.json"


def empty_celldays() -> pd.DataFrame:
    return conform(pd.DataFrame({c: pd.Series(dtype=t) for c, t in L2_COLUMNS.items()}), L2_COLUMNS)


def to_celldays(det: pd.DataFrame, resolution: float) -> pd.DataFrame:
    """Aggregate a batch of detections (one sensor family) to cell-days."""
    if det.empty:
        return empty_celldays()
    d = det.assign(
        cell_id=cell_index(det["lat"], det["lon"], resolution),
        is_day=(det["daynight"] == "D").astype("int32"),
        is_night=(det["daynight"] == "N").astype("int32"),
        is_high=(det["confidence_class"] == "high").astype("int32"),
        is_unknown=(~det["type_known"]).astype("int32"),
    )
    g = d.groupby(GROUP_KEYS, observed=True, sort=True).agg(
        sensor_family=("sensor_family", "first"),
        n_detections=("frp_mw", "size"),
        frp_sum_mw=("frp_mw", "sum"),
        frp_mean_mw=("frp_mw", "mean"),
        frp_max_mw=("frp_mw", "max"),
        n_day=("is_day", "sum"),
        n_night=("is_night", "sum"),
        n_high_conf=("is_high", "sum"),
        n_type_unknown=("is_unknown", "sum"),
    ).reset_index()
    g["cell_lat"], g["cell_lon"] = cell_centre(g["cell_id"], resolution)
    g["year"] = g["acq_date"].dt.year
    g["month"] = g["acq_date"].dt.month
    return conform(g, L2_COLUMNS)


def combine_partials(parts: pd.DataFrame) -> pd.DataFrame:
    """Merge cell-day rows that were aggregated separately (for example one per country file).

    A grid cell on a border can appear in two files on the same day; this adds the counts
    and sums, takes the larger maximum and recomputes the mean, giving the same result as
    aggregating all detections together.
    """
    if parts.empty:
        return empty_celldays()
    g = parts.groupby(GROUP_KEYS, observed=True, sort=True).agg(
        sensor_family=("sensor_family", "first"),
        cell_lat=("cell_lat", "first"),
        cell_lon=("cell_lon", "first"),
        frp_max_mw=("frp_max_mw", "max"),
        **{c: (c, "sum") for c in SUM_COLS},
    ).reset_index()
    g["frp_mean_mw"] = g["frp_sum_mw"] / g["n_detections"]
    g["year"] = g["acq_date"].dt.year
    g["month"] = g["acq_date"].dt.month
    return conform(g, L2_COLUMNS)


def write_celldays(cd: pd.DataFrame, out_dir: Path) -> list[str]:
    """Write cell-days as one Parquet file per sensor family, year and month. Returns partition paths."""
    written = []
    if cd.empty:
        return written
    for (family, year, month), part in cd.groupby(["sensor_family", "year", "month"], observed=True):
        part = conform(part.reset_index(drop=True), L2_COLUMNS)
        validate_l2(part)
        rel = f"sensor_family={family}/year={int(year)}/month={int(month)}"
        target = out_dir / rel
        target.mkdir(parents=True, exist_ok=True)
        part.drop(columns=PARTITIONS).to_parquet(target / "part-0.parquet", index=False)
        written.append(rel)
    return written


def fetched_partitions(settings: Settings) -> set[str]:
    """Partitions written by the fetch stage (they must never be overwritten by aggregate)."""
    done = settings.processed_dir / "_fetch_done"
    out: set[str] = set()
    for marker in done.glob("*.json") if done.exists() else []:
        out.update(json.loads(marker.read_text()).get("partitions", []))
    return out


def run_aggregate(settings: Settings) -> dict:
    src = settings.detections_dir
    out_dir = settings.celldays_dir
    manifest = settings.processed_dir / MANIFEST

    # Remove only what this stage wrote last time; partitions from the fetch stage stay.
    if manifest.exists():
        for rel in json.loads(manifest.read_text()):
            shutil.rmtree(out_dir / rel, ignore_errors=True)
    protected = fetched_partitions(settings)

    report = {"stage": "aggregate", "resolution_deg": settings.resolution, "partitions": 0,
              "detections_in": 0, "celldays_out": 0}
    written: list[str] = []
    for part in sorted(src.glob("sensor_family=*/year=*/month=*")) if src.exists() else []:
        family = part.parent.parent.name.split("=", 1)[1]
        year = int(part.parent.name.split("=", 1)[1])
        month = int(part.name.split("=", 1)[1])
        rel = f"sensor_family={family}/year={year}/month={month}"
        if rel in protected:
            raise RuntimeError(
                f"{rel} already comes from the yearly world files (fetch stage). "
                "Raw form downloads must start after fetch.last_year to avoid double counting.")
        det = pd.read_parquet(part).assign(sensor_family=family, year=year, month=month)
        det = conform(det, L1_COLUMNS)
        cd = to_celldays(det, settings.resolution)
        written += write_celldays(cd, out_dir)
        report["partitions"] += 1
        report["detections_in"] += len(det)
        report["celldays_out"] += len(cd)

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(sorted(set(written)), indent=1))
    return report


def read_celldays(settings: Settings, **filters) -> pd.DataFrame:
    return _read_partitioned(settings.celldays_dir, L2_COLUMNS, filters)
