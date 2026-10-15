# SHIFT runbook: stage order

Every stage is run from `pipeline/` with `python -m shift.run --dataset global --stage <name>`.
Each writes `data/processed/global/report_<name>.json` with a provenance stamp (git commit,
whether code or settings had uncommitted changes, SHA-256 of the settings, Python, numpy and
pandas versions, UTC time). Rules that decide anything are written in `config/settings.yaml`
and committed to git BEFORE the run they judge; the git history is the evidence of that order.
Nothing is registered with an outside registry.

## Full rebuild from raw files (only squadronzero10, on Colab)

| # | Stage | Needs | Writes (under data/processed/global/) | Time |
| --- | --- | --- | --- | --- |
| 1 | fetch | internet (FIRMS yearly files) | celldays/ | ~81 min (13 Oct) |
| 2 | ingest, aggregate (2026 form ZIPs in raw/global/) | raw ZIPs | celldays/ | ~2 min |
| 3 | scan | internet | static_sources/, version table | see its report |
| 4 | qc | celldays | qc/daily_coverage.parquet | see its report |
| 5 | confscan, confidence | internet; celldays | celldays_low/, confidence/ | confidence 836 s; confscan see its report |
| 6 | nodata | qc | nodata/ (zone-day status, fill factors, fill_choice.json) | 938 s (15 Oct) |
| 7 | places | factors (first pass), confidence | places/ | 335 s |
| 8 | factors | nodata, places | counts/factors_world.parquet | 523 s |
| 9 | resolution | factors | counts/factors_world_fine.parquet, resolution_choice.json, fine_monthly.parquet | 371 s before this package; longer now |
| 10 | daily | resolution | daily/ (skill, label_rule.json, day_basis.json) | ~10-20 min |
| 11 | external | Earthdata Login (Colab Secrets) | external/ (burned area, cloud) | ~10-30 min |
| 12 | harmonise | resolution, daily, nodata | l3/ (record, region tables, scale table, uncertainty, record_info.json) | 1,110 s before this package |
| 13 | checks | harmonise, external | checks/ (summary, 2026 lock, critical periods, calibration) | ~10-20 min |
| 14 | export | harmonise, checks | site_data/, site_data.zip (website paused) | ~5 min |

Order matters at three places: `nodata` before `factors` (data-loss zone-days are left out of
the fits); `resolution` and `daily` before `harmonise` (fit years, region choices and the day
basis are decided there); `checks` last, because it judges the finished record.

## Things that are run once

- The untouched 2026 test in `checks` stores its result in `checks/final_test_2026.json` with
  its provenance. Later runs print the stored result. Setting `phase2_checks.final_test.allow_rerun`
  to true reruns it and the report then says it is not the first run.

## After the final S-NPP download (29 Oct)

Replace the three 2026 ZIPs in `raw/global/`, then rerun from step 2 (ingest/aggregate for 2026),
qc, nodata, factors, resolution, daily, harmonise, checks. The 2026 test keeps its stored first
result; the new months only extend the record.
