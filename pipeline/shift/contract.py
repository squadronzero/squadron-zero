"""Schemas for Level 1 (detections) and Level 2 (cell-days). See docs/DATA_CONTRACT.md."""
from __future__ import annotations

import pandas as pd

L1_COLUMNS: dict[str, str] = {
    "acq_datetime_utc": "datetime64[ns, UTC]",
    "acq_date": "datetime64[ns]",
    "year": "int16",
    "month": "int8",
    "lat": "float64",
    "lon": "float64",
    "sensor_family": "string",
    "satellite": "string",
    "frp_mw": "float64",
    "confidence_class": "string",
    "confidence_raw": "string",
    "daynight": "string",
    "fire_type": "int8",
    "type_known": "bool",
    "scan_km": "float64",
    "track_km": "float64",
    "source_product": "string",
    "source_file": "string",
}

L2_COLUMNS: dict[str, str] = {
    "acq_date": "datetime64[ns]",
    "year": "int16",
    "month": "int8",
    "cell_id": "int64",
    "cell_lat": "float64",
    "cell_lon": "float64",
    "sensor_family": "string",
    "satellite": "string",
    "n_detections": "int32",
    "frp_sum_mw": "float64",
    "frp_mean_mw": "float64",
    "frp_max_mw": "float64",
    "n_day": "int32",
    "n_night": "int32",
    "n_high_conf": "int32",
    "n_type_unknown": "int32",
}

SENSOR_FAMILIES = {"MODIS", "VIIRS"}
SATELLITES = {"Terra", "Aqua", "S-NPP", "NOAA-20", "NOAA-21"}
CONFIDENCE_CLASSES = {"low", "nominal", "high"}


class ContractError(ValueError):
    pass


def conform(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Return df with exactly the contract columns, in order, with contract dtypes."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ContractError(f"missing columns: {missing}")
    return df[list(columns)].astype(columns)


def validate_l1(df: pd.DataFrame) -> None:
    _check_columns(df, L1_COLUMNS)
    if len(df) == 0:
        return
    _check(df["lat"].between(-90, 90).all(), "lat out of range")
    _check(df["lon"].between(-180, 180).all(), "lon out of range")
    _check(df["sensor_family"].isin(SENSOR_FAMILIES).all(), "unknown sensor_family")
    _check(df["satellite"].isin(SATELLITES).all(), "unknown satellite")
    _check(df["confidence_class"].isin(CONFIDENCE_CLASSES).all(), "unknown confidence_class")
    _check(df["daynight"].isin({"D", "N"}).all(), "daynight must be D or N")
    _check((df["frp_mw"] >= 0).all(), "negative FRP")


def validate_l2(df: pd.DataFrame) -> None:
    _check_columns(df, L2_COLUMNS)
    if len(df) == 0:
        return
    _check((df["n_detections"] > 0).all(), "cell-day with zero detections")
    _check((df["n_day"] + df["n_night"] == df["n_detections"]).all(), "day + night != detections")
    _check((df["frp_max_mw"] >= df["frp_mean_mw"] - 1e-9).all(), "frp_max < frp_mean")
    _check(not df.duplicated(["acq_date", "cell_id", "satellite"]).any(), "duplicate cell-day rows")


def _check_columns(df: pd.DataFrame, columns: dict[str, str]) -> None:
    if list(df.columns) != list(columns):
        raise ContractError(f"columns differ from contract: {list(df.columns)}")


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)
