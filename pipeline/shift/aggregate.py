"""Stage 2: Level 1 detections -> Level 2 cell-days on the SHIFT grid."""
from __future__ import annotations

import shutil

import pandas as pd

from .config import Settings
from .contract import L1_COLUMNS, L2_COLUMNS, conform, validate_l2
from .grid import cell_centre, cell_index
from .ingest import PARTITIONS, _read_partitioned

GROUP_KEYS = ["acq_date", "cell_id", "satellite"]


def to_celldays(det: pd.DataFrame, resolution: float) -> pd.DataFrame:
    """Aggregate one batch of detections (one sensor family, one month) to cell-days."""
    if det.empty:
        return conform(pd.DataFrame({c: pd.Series(dtype=t) for c, t in L2_COLUMNS.items()}), L2_COLUMNS)
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


def run_aggregate(settings: Settings) -> dict:
    src = settings.detections_dir
    out_dir = settings.celldays_dir
    shutil.rmtree(out_dir, ignore_errors=True)
    report = {"stage": "aggregate", "resolution_deg": settings.resolution, "partitions": 0,
              "detections_in": 0, "celldays_out": 0}
    for part in sorted(src.glob("sensor_family=*/year=*/month=*")):
        family = part.parent.parent.name.split("=", 1)[1]
        year = int(part.parent.name.split("=", 1)[1])
        month = int(part.name.split("=", 1)[1])
        det = pd.read_parquet(part).assign(sensor_family=family, year=year, month=month)
        det = conform(det, L1_COLUMNS)
        cd = to_celldays(det, settings.resolution)
        validate_l2(cd)
        target = out_dir / f"sensor_family={family}" / f"year={year}" / f"month={month}"
        target.mkdir(parents=True, exist_ok=True)
        cd.drop(columns=PARTITIONS).to_parquet(target / "part-0.parquet", index=False)
        report["partitions"] += 1
        report["detections_in"] += len(det)
        report["celldays_out"] += len(cd)
    return report


def read_celldays(settings: Settings, **filters) -> pd.DataFrame:
    return _read_partitioned(settings.celldays_dir, L2_COLUMNS, filters)

