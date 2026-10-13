"""Cross-check stage: two independent download routes must agree.

The Florida dataset came from the FIRMS form; the global dataset came from the FIRMS
yearly world files (fetch stage). For the same years, cells well inside Florida must hold
the same cell-days in both. A good match proves the yearly files are complete and that
nothing was lost or double-counted at country borders.

Only cells whose centre lies at least `checks.crosscheck.inset_deg` inside the boundary
are compared, because the Florida dataset was clipped detection by detection while
the global dataset is not clipped at all.

    python -m shift.run --dataset global --stage crosscheck
"""
from __future__ import annotations

import pandas as pd

from .config import Settings
from .region import _load as load_region

KEY = ["acq_date", "cell_id", "satellite"]
VALS = ["n_detections", "frp_sum_mw"]


def _read(root, years, bounds):
    lo_lon, lo_lat, hi_lon, hi_lat = bounds
    flt = [("cell_lat", ">=", lo_lat), ("cell_lat", "<=", hi_lat), ("cell_lon", ">=", lo_lon), ("cell_lon", "<=", hi_lon)]
    parts = []
    for part in sorted(root.glob("sensor_family=*/year=*/month=*")) if root.exists() else []:
        year = int(part.parent.name.split("=", 1)[1])
        if not years[0] <= year <= years[1]:
            continue
        d = pd.read_parquet(part, columns=KEY + VALS + ["cell_lat", "cell_lon"], filters=flt)
        d["sensor_family"] = part.parent.parent.name.split("=", 1)[1]
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEY + VALS + ["cell_lat", "cell_lon", "sensor_family"])


def run_crosscheck(settings: Settings) -> dict:
    from shapely import contains_xy

    cfg = settings["checks"]["crosscheck"]
    ref = Settings(raw={**settings.raw, "dataset": cfg["reference_dataset"]}, root=settings.root)
    region = load_region(str(settings.path(cfg["region"])))
    inner = region.buffer(-float(cfg["inset_deg"]))
    years = tuple(int(y) for y in cfg["years"])

    g = _read(settings.celldays_dir, years, region.bounds)
    f = _read(ref.celldays_dir, years, region.bounds)
    if f.empty:
        return {"stage": "crosscheck", "note": f"no cell-days in the {cfg['reference_dataset']} dataset"}
    # Compare only satellites the reference has. The global record also holds VIIRS NOAA-20
    # (D8), which the Florida download does not, and would otherwise look like extra data.
    g = g[g["satellite"].isin(set(f["satellite"].unique()))]
    g = g[contains_xy(inner, g["cell_lon"].to_numpy(float), g["cell_lat"].to_numpy(float))]
    f = f[contains_xy(inner, f["cell_lon"].to_numpy(float), f["cell_lat"].to_numpy(float))]

    m = g.merge(f, on=KEY, how="outer", suffixes=("_global", "_florida"), indicator=True)
    m["sensor_family"] = m["sensor_family_global"].fillna(m["sensor_family_florida"])
    m["year"] = pd.to_datetime(m["acq_date"]).dt.year
    both = m["_merge"] == "both"
    m["same_count"] = both & (m["n_detections_global"] == m["n_detections_florida"])
    m["same_frp"] = both & ((m["frp_sum_mw_global"] - m["frp_sum_mw_florida"]).abs() <= 0.05)

    rows = []
    for (fam, yr), d in m.groupby(["sensor_family", "year"]):
        b = d["_merge"] == "both"
        rows.append({"family": fam, "year": int(yr), "celldays_global": int((d["_merge"] != "right_only").sum()),
                     "celldays_florida": int((d["_merge"] != "left_only").sum()), "in_both": int(b.sum()),
                     "same_count_pct": round(100 * d["same_count"].sum() / max(b.sum(), 1), 2),
                     "same_frp_pct": round(100 * d["same_frp"].sum() / max(b.sum(), 1), 2),
                     "detections_global": int(d["n_detections_global"].sum()),
                     "detections_florida": int(d["n_detections_florida"].sum())})
    table = pd.DataFrame(rows)
    tot = {k: int(table[k].sum()) for k in ["celldays_global", "celldays_florida", "in_both",
                                             "detections_global", "detections_florida"]}
    tot["same_count_pct"] = round(100 * m["same_count"].sum() / max(both.sum(), 1), 2)
    tot["same_frp_pct"] = round(100 * m["same_frp"].sum() / max(both.sum(), 1), 2)
    tot["detections_diff_pct"] = round(100 * (tot["detections_global"] / max(tot["detections_florida"], 1) - 1), 3)
    verdict = ("PASS" if tot["same_count_pct"] >= 99 and abs(tot["detections_diff_pct"]) <= 1 else "CHECK")

    out = settings.processed_dir / "qc"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "crosscheck_florida.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("[crosscheck] global yearly files vs Florida form download, cells inside Florida:\n"
              + table.to_string(index=False), flush=True)
    print(f"[crosscheck] verdict: {verdict}", flush=True)
    return {"stage": "crosscheck", "years": list(years), "verdict": verdict, "totals": tot,
            "output": "qc/crosscheck_florida.csv"}
