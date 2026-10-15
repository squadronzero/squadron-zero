"""Command-line entry point. One command rebuilds every product from the raw files.

    python -m shift.run                          # ingest, aggregate, calibrate (dataset from settings)
    python -m shift.run --dataset florida        # same, for the Florida v1 check
    python -m shift.run --stage ingest           # one stage
    python -m shift.run --stage fetch            # download yearly world files (global dataset)
    python -m shift.run --stage fetch --years 2020 --family MODIS
    python -m shift.run --dataset global --stage scan   # versions, non-fire sources, fingerprints
    python -m shift.run --dataset global --stage qc     # data health check
    python -m shift.run --dataset global --stage crosscheck  # yearly files vs Florida download
    python -m shift.run --dataset global --stage counts      # Phase 2 fix 1: detection-count harmonisation
    python -m shift.run --dataset global --stage continuity  # Phase 2 fix 2: tie NOAA-20 to S-NPP (after counts)
    python -m shift.run --dataset global --stage sources     # Phase 2 fix 3: Aqua, Terra or both? (after counts)
    python -m shift.run --dataset global --stage selection   # Phase 2 fix 3b: six fair tests pick the MODIS source (D19)
    python -m shift.run --dataset global --stage frp         # Phase 2 fix 4: the intercept problem in fire power (D21)
    python -m shift.run --dataset global --stage worldwide   # D19 and D21 re-tested on a 10-degree grid over the whole Earth
    python -m shift.run --dataset global --stage factors     # D21 for every place on Earth: gap fixes, refinements, factor table
    python -m shift.run --dataset global --stage confscan    # Phase 2 fix 5: low-confidence fire cell-days, own folder (download)
    python -m shift.run --dataset global --stage confidence  # Phase 2 fix 5: confidence asymmetry, variants scored
    python -m shift.run --dataset global --stage nodata      # Phase 2 fix 7: no-data mask, regional gaps, tested gap filling
    python -m shift.run --dataset global --stage places      # Phase 2 fix 8: per-place choices, borrowed factors tested, spotlight
    python -m shift.run --settings path/to/other_settings.yaml

Run from the `pipeline/` folder (or add it to PYTHONPATH).
"""
from __future__ import annotations

import argparse
import json
import time

from .aggregate import run_aggregate
from .calibrate import run_calibrate
from .config import load_settings
from .counts import run_continuity, run_counts, run_sources
from .crosscheck import run_crosscheck
from .fetch import run_fetch
from .frp import run_frp
from .ingest import run_ingest, write_report
from .qc import run_qc
from .scan import run_scan
from .selection import run_selection
from .tiles import run_worldwide
from .factors import run_factors
from .confidence import run_confidence, run_confscan
from .nodata import run_nodata
from .places import run_places

STAGES = {
    "ingest": run_ingest,
    "aggregate": run_aggregate,
    "calibrate": run_calibrate,
    # Phase 2 adds: "harmonise" (build the L3 record), per-region calibration
    # Phase 3 adds: "calendar" (climatology, anomalies, critical periods)
}
# Stages that only run when asked for by name (downloads and checks).
EXTRA_STAGES = {"fetch": run_fetch, "scan": run_scan, "qc": run_qc, "crosscheck": run_crosscheck,
                "counts": run_counts, "continuity": run_continuity, "sources": run_sources,
                "selection": run_selection, "frp": run_frp,
                "worldwide": run_worldwide, "factors": run_factors,
                "confscan": run_confscan, "confidence": run_confidence,
                "nodata": run_nodata, "places": run_places}
# What to print at the end of each stage (the full report is always saved as JSON).
SUMMARY = {"scan": ["version_summary", "static_mask", "missing"],
           "qc": ["date_range", "days_by_status", "step_checks", "real_fires_in_mask_pct"],
           "crosscheck": ["verdict", "totals"],
           "counts": ["k_per_region", "outage_days_excluded"],
           "continuity": ["k_cells_per_region", "gapfill", "days_excluded"],
           "sources": ["summary", "lowest_regional_error", "days_excluded"],
           "selection": ["winner", "why", "fire_size_threshold_mw", "days_excluded"],
           "frp": ["winner", "why", "days_excluded"],
           "worldwide": ["tiles_with_fire", "tiles_tested", "fire_share_in_tested_tiles_pct"],
           "factors": ["chosen", "final_frp", "final_cells", "factor_rows"],
           "confscan": ["low_share_by_year", "missing", "different_file_from_fetch"],
           "confidence": ["winner", "why", "low_share_trend", "fire_weighted_mean_gap_pct_points",
                          "fire_weighted_abs_gap_trend_pct_points_per_year"],
           "nodata": ["chosen", "links", "regional_gaps", "zone_days_without_own_satellite",
                      "zone_days_left_empty", "whole_world_empty_days", "months_by_status"],
           "places": ["mask_chosen", "source_choice", "source_why", "source_picks_tested_tiles",
                      "borrowed_factor_test", "confidence_drift_tiles_over_5pct"]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="SHIFT pipeline")
    ap.add_argument("--settings", help="settings YAML (default: config/settings.yaml)")
    ap.add_argument("--dataset", help="dataset folder to use: global or florida (default: from settings)")
    ap.add_argument("--stage", choices=list(STAGES) + list(EXTRA_STAGES), help="run only this stage")
    ap.add_argument("--years", type=int, nargs="+", help="fetch/scan/confscan only: years to download")
    ap.add_argument("--family", nargs="+", help="fetch/scan/confscan only: products from fetch.products (MODIS, VIIRS, VIIRS_NOAA20)")
    ap.add_argument("--force", action="store_true", help="fetch/scan/confscan only: redo years already done")
    args = ap.parse_args(argv)

    settings = load_settings(args.settings, dataset=args.dataset)
    print(f"dataset: {settings.dataset}  raw: {settings.raw_dir}  processed: {settings.processed_dir}")
    for name in ([args.stage] if args.stage else list(STAGES)):
        t0 = time.time()
        if name in ("fetch", "scan", "confscan"):
            report = EXTRA_STAGES[name](settings, years=args.years, families=args.family, force=args.force)
        elif name in EXTRA_STAGES:
            report = EXTRA_STAGES[name](settings)
        else:
            report = STAGES[name](settings)
        report["seconds"] = round(time.time() - t0, 1)
        path = write_report(report, settings, name)
        print(f"[{name}] done in {report['seconds']} s -> {path}")
        shown = ({k: report.get(k) for k in SUMMARY[name]} if name in SUMMARY
                 else report.get("totals", report))
        print(json.dumps(shown, indent=2, default=str))


if __name__ == "__main__":
    main()
