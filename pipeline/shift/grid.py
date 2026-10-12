"""Regular lat/lon grid used for every SHIFT product (see docs/DATA_CONTRACT.md)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def grid_shape(resolution: float) -> tuple[int, int]:
    return int(round(180 / resolution)), int(round(360 / resolution))


def cell_index(lat: pd.Series | np.ndarray, lon: pd.Series | np.ndarray, resolution: float) -> np.ndarray:
    """Return the int64 cell_id for each lat/lon pair."""
    n_rows, n_cols = grid_shape(resolution)
    row = np.floor((np.asarray(lat, dtype="float64") + 90.0) / resolution).astype("int64")
    col = np.floor((np.asarray(lon, dtype="float64") + 180.0) / resolution).astype("int64")
    row = np.clip(row, 0, n_rows - 1)
    col = np.clip(col, 0, n_cols - 1)
    return row * n_cols + col


def cell_centre(cell_id: pd.Series | np.ndarray, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (cell_lat, cell_lon) of the cell centres."""
    _, n_cols = grid_shape(resolution)
    cid = np.asarray(cell_id, dtype="int64")
    row, col = np.divmod(cid, n_cols)
    return -90.0 + (row + 0.5) * resolution, -180.0 + (col + 0.5) * resolution
