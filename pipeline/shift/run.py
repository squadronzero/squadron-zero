"""Command-line entry point. One command rebuilds every product from the raw files.

    python -m shift.run                          # ingest, aggregate, calibrate (dataset from settings)
    python -m shift.run --dataset florida        # same, for the Florida v1 check
    python -m shift.run --stage ingest           # one stage
    python -m shift.run --stage fetch            # download yearly world files (global dataset)
    python -m shift.run --stage fetch --years 2020 --family MODIS
    python -m shift.run --dataset global --stage scan   # versions, non-fire sources, fingerprints
    python -m shift.run --dataset global --stage qc     # data health check
    python -m shift.run --dataset global --stage crosscheck  # yearly files vs Florida download
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
from .crosscheck import run_crosscheck
from .fetch import run_fetch
from .ingest import run_ingest, write_report
from .qc import run_qc
from .scan import run_scan

STAGES = {
    "ingest": run_ingest,
    "aggregate": run_aggregate,
    "calibrate": run_calibrate,
    # Phase 2 adds: "harmonise" (build the L3 record), per-region calibration
    # Phase 3 adds: "calendar" (climatology, anomalies, critical periods)
}
# Stages that only run when asked for by name (downloads and checks).
EXTRA_STAGES = {"fetch": run_fetch, "scan": run_scan, "qc": run_qc, "crosscheck": run_crosscheck}
# What to print at the end of each stage (the full report is always saved as JSON).
SUMMARY = {"scan": ["version_summary", "static_mask", "missing"],
           "qc": ["date_range", "days_by_status", "step_checks", "real_fires_in_mask_pct"],
           "crosscheck": ["verdict", "totals"]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="SHIFT pipeline")
    ap.add_argument("--settings", help="settings YAML (default: config/settings.yaml)")
    ap.add_argument("--dataset", help="dataset folder to use: global or florida (default: from settings)")
    ap.add_argument("--stage", choices=list(STAGES) + list(EXTRA_STAGES), help="run only this stage")
    ap.add_argument("--years", type=int, nargs="+", help="fetch/scan only: years to download")
    ap.add_argument("--family", nargs="+", help="fetch/scan only: products from fetch.products (MODIS, VIIRS, VIIRS_NOAA20)")
    ap.add_argument("--force", action="store_true", help="fetch/scan only: redo years already done")
    args = ap.parse_args(argv)

    settings = load_settings(args.settings, dataset=args.dataset)
    print(f"dataset: {settings.dataset}  raw: {settings.raw_dir}  processed: {settings.processed_dir}")
    for name in ([args.stage] if args.stage else list(STAGES)):
        t0 = time.time()
        if name in ("fetch", "scan"):
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
