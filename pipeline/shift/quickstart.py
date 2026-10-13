"""Two-minute demo: run the whole method on a small real Florida sample.

    cd pipeline
    pip install -r requirements.txt
    python -m shift.quickstart

Uses samples/florida_sample.zip (real NASA FIRMS archive detections for Florida,
2020 and 2023 only, about 57,000 rows). 2020 fits the v1-style calibration;
2023 tests it. No downloads and no Google Drive: everything is written to a fresh
temporary folder, and nothing from the main data folders (coverage table, static-source
mask) is read, so the result is the same on every machine.
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from .aggregate import run_aggregate
from .calibrate import run_calibrate
from .config import REPO_ROOT, load_settings
from .ingest import run_ingest, write_report

SAMPLE = REPO_ROOT / "samples" / "florida_sample.zip"


def main() -> None:
    t0 = time.time()
    settings = load_settings(dataset="sample")
    # The sample covers two years only: fit on 2020, test on 2023.
    settings.raw["periods"].update({"calibration_start": "2020-01-01", "calibration_end": "2020-12-31",
                                    "holdout_start": "2023-01-01", "holdout_end": "2023-12-31"})
    work = Path(tempfile.mkdtemp(prefix="shift_quickstart_"))
    settings.raw["inputs"]["raw_dir"] = str(work / "raw")
    settings.raw["outputs"]["processed_dir"] = str(work / "processed")
    settings.raw["static_mask"]["enabled"] = False      # self-contained: no global mask
    settings.raw["calibration"]["exclude_gap_days"] = False  # no global coverage table
    raw = settings.raw_dir
    raw.mkdir(parents=True)
    shutil.copy(SAMPLE, raw / SAMPLE.name)
    print(f"SHIFT quickstart: {SAMPLE.name} -> {raw}")

    for name, fn in [("ingest", run_ingest), ("aggregate", run_aggregate), ("calibrate", run_calibrate)]:
        report = fn(settings)
        write_report(report, settings, name)
        if name == "ingest":
            t = report["totals"]
            print(f"[ingest]    {t['rows_read']:,} detections read, {t['rows_final']:,} kept "
                  f"({t.get('outside_region', 0):,} outside Florida, {t.get('confidence_filtered', 0):,} low confidence, "
                  f"{t.get('non_vegetation_type', 0):,} non-fire heat)")
        elif name == "aggregate":
            print(f"[aggregate] {report['celldays_out']:,} cell-days on the 0.05 deg grid")
        else:
            m = report["model"]
            print(f"[calibrate] VIIRS_FRP = {m['slope']:.4f} x MODIS_FRP + {m['intercept']:.2f}  "
                  f"(fitted on {m['n_pairs']:,} cell-days seen by both sensors in 2020)")
            for v in report.get("validation_holdout", []):
                print(f"            2023 test, {v['scale']:<14} R2 {v['r2_calibrated']:+.3f} "
                      f"(raw MODIS {v['r2_raw_modis']:+.3f})")
    print(f"Done in {time.time() - t0:.0f} s. Reports: {settings.processed_dir}")
    print("Full method: docs/SCIENTIFIC_SPECIFICATION.md. Full global run: docs/COLAB_GUIDE.md.")


if __name__ == "__main__":
    main()
