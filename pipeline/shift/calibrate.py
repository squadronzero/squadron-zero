"""Stage 3: fit the MODIS -> VIIRS FRP calibration (v1 method) and validate it.

Method (from the v1 Florida prototype, decision D11):
  1. Pair MODIS and VIIRS cell-days where both sensors detected fire in the same
     cell on the same day (MODIS = Terra + Aqua summed, VIIRS = S-NPP).
  2. Fit ordinary least squares on the calibration window:
         VIIRS_FRP = slope x MODIS_FRP + intercept
  3. Apply it to held-out years and report errors at several scales, from a single
     cell-day up to a whole region per month (the scale the calendar uses).

Known limitation, tracked for Phase 2: pairs only include cell-days where BOTH
sensors detected fire. Cell-days seen only by VIIRS (small fires MODIS misses)
are not in the fit. Checking calibrated totals against all VIIRS detections needs
the full downloaded record, not just the overlap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate import read_celldays
from .config import Settings

PAIR_COLUMNS = ["acq_date", "cell_lat", "cell_lon", "frp_modis", "count_modis", "frp_viirs", "count_viirs"]


# ---------------------------------------------------------------- pairing

def build_pairs(celldays: pd.DataFrame, modis_sats: list[str], viirs_sats: list[str]) -> pd.DataFrame:
    """Join MODIS and VIIRS cell-days that share a cell and a day."""
    def side(sats: list[str], tag: str) -> pd.DataFrame:
        d = celldays[celldays["satellite"].isin(sats)]
        return (d.groupby(["acq_date", "cell_id"], observed=True)
                 .agg(cell_lat=("cell_lat", "first"), cell_lon=("cell_lon", "first"),
                      **{f"frp_{tag}": ("frp_sum_mw", "sum"), f"count_{tag}": ("n_detections", "sum")})
                 .reset_index())
    m, v = side(modis_sats, "modis"), side(viirs_sats, "viirs")
    pairs = m.merge(v.drop(columns=["cell_lat", "cell_lon"]), on=["acq_date", "cell_id"], how="inner")
    return pairs[PAIR_COLUMNS].sort_values(["acq_date", "cell_lat", "cell_lon"]).reset_index(drop=True)


def load_v1_overlap(path: str | Path) -> pd.DataFrame:
    """Read the v1 Florida overlap CSV into the same shape as build_pairs()."""
    df = pd.read_csv(path, parse_dates=["acq_date"])
    return df.rename(columns={"grid_lat": "cell_lat", "grid_lon": "cell_lon"})[PAIR_COLUMNS]


def split_window(pairs: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    d = pairs["acq_date"]
    return pairs[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]


# ---------------------------------------------------------------- model

def fit_ols(pairs: pd.DataFrame) -> dict:
    if len(pairs) < 2:
        raise ValueError("need at least 2 paired cell-days to fit")
    slope, intercept = np.polyfit(pairs["frp_modis"].to_numpy(float), pairs["frp_viirs"].to_numpy(float), 1)
    return {
        "method": "ols",
        "formula": "VIIRS_FRP = slope * MODIS_FRP + intercept",
        "slope": float(slope),
        "intercept": float(intercept),
        "n_pairs": int(len(pairs)),
        "date_start": str(pairs["acq_date"].min().date()),
        "date_end": str(pairs["acq_date"].max().date()),
    }


def apply_model(model: dict, frp_modis) -> np.ndarray:
    return model["slope"] * np.asarray(frp_modis, dtype=float) + model["intercept"]


# ---------------------------------------------------------------- validation

def _group_keys(df: pd.DataFrame, scale: dict) -> list:
    keys = [df["acq_date"] if scale["time"] == "day" else df["acq_date"].dt.to_period("M")]
    space = scale["space_deg"]
    if space is None:
        keys += [df["cell_lat"].round(6), df["cell_lon"].round(6)]
    elif space != "all":
        keys += [np.floor(df["cell_lat"] / space), np.floor(df["cell_lon"] / space)]
    return keys


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan")


def validate(pairs: pd.DataFrame, model: dict, scales: list[dict]) -> pd.DataFrame:
    """Error metrics of calibrated MODIS vs native VIIRS, aggregated to each scale."""
    df = pairs.assign(pred=apply_model(model, pairs["frp_modis"]))
    rows = []
    for scale in scales:
        g = df.groupby(_group_keys(df, scale)).agg(v=("frp_viirs", "sum"), p=("pred", "sum"), m=("frp_modis", "sum"))
        y, p, m = g["v"].to_numpy(), g["p"].to_numpy(), g["m"].to_numpy()
        rows.append({
            "scale": scale["name"],
            "n_units": len(g),
            "r2_calibrated": _r2(y, p),
            "r2_raw_modis": _r2(y, m),
            "rmse_mw": float(np.sqrt(((y - p) ** 2).mean())),
            "mae_mw": float(np.abs(y - p).mean()),
            "bias_mw": float((p - y).mean()),
            "total_error_pct": float(100 * (p.sum() - y.sum()) / y.sum()),
            "median_abs_error_pct": float(100 * np.median(np.abs(y - p) / y)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- stage

def run_calibrate(settings: Settings) -> dict:
    cfg, periods = settings["calibration"], settings["periods"]
    cd = read_celldays(settings)
    pairs = build_pairs(cd, cfg["modis_satellites"], cfg["viirs_satellites"])
    train = split_window(pairs, periods["calibration_start"], periods["calibration_end"])
    test = split_window(pairs, periods["holdout_start"], periods["holdout_end"])

    if len(train) < 2:
        return {"stage": "calibrate", "pairs_total": len(pairs), "pairs_train": len(train),
                "skipped": "fewer than 2 paired cell-days in the calibration window"}
    model = fit_ols(train)
    out = settings.processed_dir / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    pairs.to_parquet(out / "pairs.parquet", index=False)

    report = {"stage": "calibrate", "model": model, "pairs_total": len(pairs),
              "pairs_train": len(train), "pairs_holdout": len(test)}
    if len(test) >= 2:
        val = validate(test, model, cfg["validation_scales"])
        val.to_csv(out / "validation_holdout.csv", index=False)
        report["validation_holdout"] = val.round(4).to_dict(orient="records")
    else:
        report["validation_holdout"] = "no paired cell-days in the holdout window"
    return report
