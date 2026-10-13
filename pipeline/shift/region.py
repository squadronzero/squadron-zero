"""Optional region boundary per dataset.

`regions.clip` in settings.yaml maps a dataset name to a GeoJSON file. Detections
outside that boundary are dropped in ingest and counted as `outside_region`.
Datasets without an entry (for example `global`) are not clipped.

Why: the v1 Florida prototype's FIRMS request covered Florida's state outline,
not the whole 24.5-31.0 N, 87.6-80.0 W box. The box also takes in parts of
Georgia and Alabama, which v1 never saw. Clipping to the outline reproduces
v1's record counts to within 0.2% (checked 13 Oct 2026).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np


def region_path(settings) -> Path | None:
    clip = (settings.raw.get("regions") or {}).get("clip") or {}
    value = clip.get(settings.dataset)
    return settings.path(value) if value else None


@lru_cache(maxsize=8)
def _load(path_str: str):
    from shapely.geometry import shape
    from shapely.ops import unary_union

    data = json.loads(Path(path_str).read_text(encoding="utf-8"))
    feats = data["features"] if data.get("type") == "FeatureCollection" else [data]
    geoms = [shape(f["geometry"] if "geometry" in f else f) for f in feats]
    return unary_union(geoms)


def inside(settings, lon, lat) -> np.ndarray | None:
    """Boolean array: True where the point is inside the dataset's boundary.
    Returns None when the dataset has no boundary (keep everything)."""
    path = region_path(settings)
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"region boundary for dataset '{settings.dataset}' not found: {path}")
    from shapely import contains_xy

    return contains_xy(_load(str(path)), np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))


def test_regions(settings) -> dict[str, dict]:
    """Phase 2 test regions (D18) from settings: name -> {bbox: [w, s, e, n], regime}."""
    out = {}
    for name, spec in ((settings.raw.get("regions") or {}).get("test") or {}).items():
        w, s, e, n = (float(v) for v in spec["bbox"])
        if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
            raise ValueError(f"test region {name}: bad bbox {spec['bbox']}")
        out[name] = {"bbox": [w, s, e, n], "regime": str(spec.get("regime", ""))}
    return out


def in_bbox(bbox, lon, lat) -> np.ndarray:
    """True where (lon, lat) falls inside bbox [west, south, east, north]."""
    w, s, e, n = bbox
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    return (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
