"""Satellite coverage from the qc stage: which days each satellite was fully working.

The qc stage writes processed/global/qc/daily_coverage.parquet with one row per
satellite per day and a status of "ok", "low" (well below normal) or "none" (silent).
Any analysis that combines satellites should drop days when one of them was not "ok",
otherwise an outage looks like a change in fire activity. The calendar uses the same
table to show "no data" instead of "no fire".
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def coverage_path(settings) -> Path | None:
    value = (settings.raw.get("coverage") or {}).get("path")
    return settings.path(value) if value else None


def load_coverage(settings) -> pd.DataFrame | None:
    path = coverage_path(settings)
    if path is None or not path.exists():
        return None
    cov = pd.read_parquet(path, columns=["acq_date", "satellite", "status"])
    cov["acq_date"] = pd.to_datetime(cov["acq_date"]).dt.normalize()
    return cov


def bad_days(settings, satellites: list[str]) -> pd.DatetimeIndex | None:
    """Days when at least one of `satellites` was not fully working ("low" or "none").
    None when no coverage table exists yet (run the qc stage on the global dataset)."""
    cov = load_coverage(settings)
    if cov is None:
        return None
    sub = cov[cov["satellite"].isin(satellites)]
    bad = sub.loc[sub["status"] != "ok", "acq_date"]
    # A day with no row at all for a satellite is also missing, but only inside the period
    # when every listed satellite was flying (VIIRS starts in 2012, so 2002-2011 never counts).
    first = sub.groupby("satellite")["acq_date"].min().max()
    last = sub.groupby("satellite")["acq_date"].max().min()
    span = pd.date_range(first, last, freq="D")
    present = sub.groupby("acq_date")["satellite"].nunique()
    missing = span[~span.isin(present[present == len(set(satellites))].index)]
    return pd.DatetimeIndex(sorted(set(bad) | set(missing)))
