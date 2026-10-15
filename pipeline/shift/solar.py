"""Local solar day of a fire observation (Phase 2 option F).

FIRMS dates every detection in UTC. A satellite crosses a place at a fixed LOCAL solar time, so
the UTC date of a pass differs from the local date wherever the pass falls on the other side of
UTC midnight: night passes of Aqua, S-NPP and NOAA-20 (about 01:30 local) east of 22.5 degrees E
carry the previous UTC date; Terra's night pass (about 22:30 local) west of 22.5 degrees W
carries the next one; day passes only near the date line. A cell-day is moved by whole days
using the nominal local overpass hour of its satellite and whether most of its detections were
made by day or by night (settings daily.overpass_hours).
"""
from __future__ import annotations

import numpy as np

DEFAULT_HOURS = {"Terra": [10.5, 22.5], "Aqua": [13.5, 1.5], "S-NPP": [13.5, 1.5], "NOAA-20": [13.5, 1.5]}


def local_day_shift(lon, satellite, n_day, n_night, hours: dict | None = None) -> np.ndarray:
    """Days to add to the UTC date to get the local solar date (-1, 0 or +1)."""
    hours = hours or DEFAULT_HOURS
    lon = np.asarray(lon, dtype=float)
    night = np.asarray(n_night, dtype=float) > np.asarray(n_day, dtype=float)
    sat = np.asarray(satellite).astype(str)
    t = np.full(len(lon), 13.5)
    for s, (hd, hn) in hours.items():
        m = sat == s
        t[m] = np.where(night[m], float(hn), float(hd))
    utc_hour = t - lon / 15.0
    return (-np.floor(utc_hour / 24.0)).astype("int64")
