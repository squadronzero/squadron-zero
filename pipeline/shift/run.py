"""Command-line entry point. One command rebuilds every product from the raw files.

    python -m shift.run                          # ingest, aggregate, calibrate (dataset from settings)
    python -m shift.run --dataset florida        # same, for the Florida v1 check
    python -m shift.run --stage ingest           # one stage
    python -m shift.run --stage fetch            # download yearly world files (global dataset)
    python -m shift.run --stage fetch --years 2020 --family MODIS
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
from .fetch import run_fetch
from .ingest import run_ingest, write_report

STAGES = {
    "ingest": run_ingest,
    "aggregate": run_aggregate,
    "calibrate": run_calibrate,
    # Phase 2 adds: "harmonise" (build the L3 record), per-region calibration
    # Phase 3 adds: "calendar" (climatology, anomalies, critical periods)
}
# Stages that only run when asked for by name (long downloads).
EXTRA_STAGES = {"fetch": run_fetch}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="SHIFT pipeline")
    ap.add_argument("--settings", help="settings YAML (default: config/settings.yaml)")
    ap.add_argument("--dataset", help="dataset folder to use: global or florida (default: from settings)")
    ap.add_argument("--stage", choices=list(STAGES) + list(EXTRA_STAGES), help="run only this stage")
    ap.add_argument("--years", type=int, nargs="+", help="fetch only: years to download")
    ap.add_argument("--family", choices=["MODIS", "VIIRS"], nargs="+", help="fetch only: sensor families")
    ap.add_argument("--force", action="store_true", help="fetch only: redo years already done")
    args = ap.parse_args(argv)

    settings = load_settings(args.settings, dataset=args.dataset)
    print(f"dataset: {settings.dataset}  raw: {settings.raw_dir}  processed: {settings.processed_dir}")
    for name in ([args.stage] if args.stage else list(STAGES)):
        t0 = time.time()
        if name == "fetch":
            report = run_fetch(settings, years=args.years, families=args.family, force=args.force)
        else:
            report = STAGES[name](settings)
        report["seconds"] = round(time.time() - t0, 1)
        path = write_report(report, settings, name)
        print(f"[{name}] done in {report['seconds']} s -> {path}")
        print(json.dumps(report.get("totals", report), indent=2, default=str))


if __name__ == "__main__":
    main()
