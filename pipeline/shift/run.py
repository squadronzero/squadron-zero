"""Command-line entry point. One command rebuilds every product from the raw files.

    python -m shift.run                 # all stages
    python -m shift.run --stage ingest  # one stage
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
from .ingest import run_ingest, write_report

STAGES = {
    "ingest": run_ingest,
    "aggregate": run_aggregate,
    "calibrate": run_calibrate,
    # Phase 2 adds: "harmonise" (build the L3 record), per-region calibration
    # Phase 3 adds: "calendar" (climatology, anomalies, critical periods)
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="SHIFT pipeline")
    ap.add_argument("--settings", help="settings YAML (default: config/settings.yaml)")
    ap.add_argument("--stage", choices=list(STAGES), help="run only this stage")
    args = ap.parse_args(argv)

    settings = load_settings(args.settings)
    for name in ([args.stage] if args.stage else list(STAGES)):
        t0 = time.time()
        report = STAGES[name](settings)
        report["seconds"] = round(time.time() - t0, 1)
        path = write_report(report, settings, name)
        print(f"[{name}] done in {report['seconds']} s -> {path}")
        print(json.dumps(report.get("totals", report), indent=2, default=str))


if __name__ == "__main__":
    main()
