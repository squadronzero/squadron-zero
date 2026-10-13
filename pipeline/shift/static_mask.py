"""Static heat-source mask for detections that carry no FIRMS `type` field.

Archive files label gas flares, industry, volcanoes and offshore platforms (type 1-3), so
the normal filter removes them. Near-real-time (NRT) files have no `type` field, so those
sources would slip through for the most recent months, and NASA's newest data is mostly NRT.

The scan stage records every grid cell FIRMS labelled as a non-fire source in the yearly
archive files. Cells labelled in at least `static_mask.min_years` different years form the
mask. An unlabelled detection that falls in a masked cell is dropped and counted as
`static_source_masked`. Labelled (archive) detections are never touched by the mask.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .grid import cell_index


def mask_path(settings) -> Path | None:
    cfg = settings.raw.get("static_mask") or {}
    if not cfg.get("enabled", True) or not cfg.get("path"):
        return None
    return settings.path(cfg["path"])


@lru_cache(maxsize=4)
def _load(path_str: str, mtime: float) -> np.ndarray:
    return np.sort(pd.read_parquet(path_str, columns=["cell_id"])["cell_id"].to_numpy(dtype="int64"))


def masked_cells(settings) -> np.ndarray | None:
    """Sorted array of masked cell ids, or None when no mask has been built yet."""
    path = mask_path(settings)
    if path is None or not path.exists():
        return None
    return _load(str(path), path.stat().st_mtime)


def in_mask(settings, lat: pd.Series, lon: pd.Series) -> np.ndarray | None:
    cells = masked_cells(settings)
    if cells is None or len(cells) == 0:
        return None
    ids = cell_index(lat.fillna(0), lon.fillna(0), settings.resolution)
    ids = np.asarray(ids, dtype="int64")
    pos = np.searchsorted(cells, ids)
    pos[pos >= len(cells)] = len(cells) - 1
    return cells[pos] == ids
