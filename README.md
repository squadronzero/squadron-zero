# MODIS-VIIRS-Harmonized-Global-Burning-Activity-Calendar-NASA-Space-Apps-Challenge-2026

**SHIFT** (Satellite Harmonization & Integrated Fire Tracking) by Team Squadron Zero, for the NASA Space Apps Challenge 2026, theme *The Next Frontier*, challenge 9: **Harmonization of MODIS and VIIRS Hot Spots**.

SHIFT turns MODIS (2000 onwards) and VIIRS (2012 onwards) active-fire detections into one consistent, uncertainty-aware burning-activity calendar for any area on Earth.

Why now: NASA stops delivering Suomi NPP data on 1 November 2026, and the Terra and Aqua MODIS
missions are near their end. A fire record that runs from 2000 to today only stays usable if the
sensors are tied together.

Version **0.0**. The system is built so that new source data can be swapped in without changing anything downstream: only the adapter and `config/settings.yaml` should need to change.

## Repository layout

```
config/settings.yaml        every tunable value (grid, filters, periods, paths)
docs/
  SCIENTIFIC_SPECIFICATION.md  scientific questions and scope
  DATA_CONTRACT.md             exact format of every dataset (L1, L2, L3)
  COLAB_GUIDE.md               how to run the pipeline on Google Colab
  AI_USE.md                    every AI tool used, what it did, what the team did
pipeline/
  shift/adapters/firms_csv.py  converts FIRMS CSV files to Level 1
  shift/ingest.py              stage 1: clean, filter, de-duplicate -> L1 detections
  shift/aggregate.py           stage 2: grid to 0.05 deg cell-days -> L2
  shift/fetch.py               downloads FIRMS yearly world files straight to cell-days
  shift/scan.py                re-reads yearly files: versions, non-fire sources, fingerprints
  shift/qc.py                  data health check (coverage, gaps, ratios)
  shift/crosscheck.py          global yearly files vs Florida form download
  shift/counts.py              Phase 2 fix 1: detection-count harmonisation (the 2012 jump)
  shift/selection.py           Phase 2 fix 3b: six tests choose the MODIS calibration source (D19)
  shift/frp.py                 Phase 2 fix 4: fire power without the v1 intercept problem (D21)
  shift/tiles.py               Worldwide check of D19 and D21 on a grid covering the whole Earth
  shift/factors.py             D21 factors for every place on Earth, gap fixes and refinements
  shift/confidence.py          Fix 5: low-confidence cell-days (confscan) and the confidence asymmetry test
  shift/nodata.py              Fix 7: no-data mask: regional gaps, gap filling tested on simulated outages, availability per zone and day
  shift/harmonise.py           Fix 9: the harmonised record L3 (0.25 deg, daily, S-NPP scale) with sources, uncertainty, 2012 gate and era-fair anomaly rule
  shift/resolution.py          Fix 9b: 10-degree vs 2.5-degree factors judged at fine scale, regional 2012 steps, rule fixed before the run
  shift/export.py              Phase 3: static website files (daily per 1-degree cell, monthly 0.25-degree map, availability, uncertainty, Not-a-fire layer, validation) + site_data.zip
  shift/places.py              Fix 8: per-place choices (flare-tile mask, MODIS source per tile, nested), borrowed factors tested, confidence drift, spotlight region
  shift/daily.py               Phase 2 package: skill of the MODIS years by day, week and month; UTC day vs local solar day
  shift/external.py            Phase 2 package: GFED4 burned area and MERRA-2 cloud, downloaded and reduced (download stage)
  shift/checks.py              Phase 2 package: anomaly rule on final factors, untouched 2026 test, range coverage, simple methods, known events, dated critical periods, burned area, cloud
  shift/solar.py               local solar day of a satellite pass
  shift/provenance.py          git commit, settings hash and software versions stamped into every report
  shift/run.py                 one command runs every stage
  shift/quickstart.py          2-minute demo on samples/florida_sample.zip
  tests/                       automated checks
samples/                       small real FIRMS sample for the quickstart
data/                          not in git; raw files live on shared Google Drive
```

Planned: `api/` (FastAPI) and `web/` (React + MapLibre) in later phases.

## Try it in 2 minutes (no download, no account)

```bash
git clone https://github.com/squadronzero/squadron-zero.git && cd squadron-zero/pipeline
pip install -r requirements.txt
python -m shift.quickstart
```

This runs the whole method on `samples/florida_sample.zip`, about 57,000 real NASA FIRMS
detections over Florida for 2020 and 2023: clean and filter, grid to 0.05 deg cell-days,
fit the MODIS -> VIIRS calibration on 2020, and test it on 2023. It finishes in seconds
and prints the formula and the test scores at four scales. Because it uses one year to fit,
its formula differs from the full 2012-2021 fit (slope 0.317, intercept 24.8).

## Quick start (full pipeline)

```bash
cd pipeline
pip install -r requirements.txt
python -m pytest -q        # all tests pass
python -m shift.run --dataset florida   # reads data/raw/florida/, writes data/processed/florida/
```

On Colab, follow `docs/COLAB_GUIDE.md`.

## Pipeline stages

| Stage | Input | Output | Status |
| --- | --- | --- | --- |
| fetch | FIRMS yearly world zips: MODIS 2002+, VIIRS S-NPP 2012+, VIIRS NOAA-20 2018+ | L2 cell-days (global, no raw kept) | Done |
| scan | FIRMS yearly world zips (re-read) | version labels, non-fire source cells + static-source mask, file fingerprints | Done |
| qc | L2 | daily coverage, data gaps, yearly ratios, version step check | Done |
| crosscheck | L2 global + L2 florida | download-route agreement, PASS/CHECK | Done |
| ingest | FIRMS CSV (archive + NRT) | L1 detections | Done |
| aggregate | L1 | L2 cell-days, 0.05 deg | Done |
| calibrate | L2 overlap 2012-2021, satellite outage days left out | v1 OLS model + validation at 4 scales | Done (v1 method) |
| quickstart | `samples/florida_sample.zip` | the whole method in seconds, in a temporary folder | Done |
| counts | L2 global | detection-count harmonisation: active cell-days, MODIS scaled to VIIRS per region and month, 2012 jump measured | Phase 2, fix 1 |
| continuity | counts daily series | NOAA-20 tied to S-NPP per region and month (cells and fire power), tested on 2024-2025, S-NPP outage days filled | Phase 2, fix 2 |
| sources | counts daily series | fix 1 fitted from MODIS merged, Aqua alone and Terra alone; holdout error, 2012 step and orbital drift compared | Phase 2, fix 3 |
| selection | counts daily series + L2 | six fair tests (backward, category agreement, leave-one-year-out, MODIS-era stability, bootstrap, fire size) choose the MODIS calibration source by a rule fixed before the run | Phase 2, fix 3b (D19) |
| frp | counts daily series + L2 pairs | fire power harmonisation: v1 line, line with intercept, line through zero, paired ratio and ratio of totals compared with the same tests and rule | Phase 2, fix 4 (D21) |
| worldwide | L2 global | D19 and D21 re-tested on a 10° grid of tiles over the whole Earth; confirm rule fixed before the run; first factors for every place | Phase 2 (fix 8 start) |
| factors | L2 global + static mask | D21 production factors for every 10° tile (burning cells and fire power, each D19 piece on all its years, bootstrap range); non-fire mask in every year, fire-size split, zone borrowing and seasonal smoothing tested by rule fixed before the runs | Phase 2 (D21, D22, fixes 6 and 8) |
| confscan | FIRMS yearly world files | Low-confidence vegetation-fire detections only, as cell-days in their own folder (`celldays_low`); main record untouched | Phase 2 (fix 5) |
| confidence | L2 global + `celldays_low` | Low-confidence share per satellite, year and tile; keeping low for MODIS, VIIRS or both scored against the current filter by a rule fixed before the run | Phase 2 (fix 5) |
| nodata | L2 global + qc coverage table | Regional gaps (a satellite far below its partner in one 30-degree zone); fill links (S-NPP <- NOAA-20, S-NPP <- D19 MODIS, MODIS <- Aqua or Terra) tested on simulated outages under a rule fixed before the run; status per zone and day (primary, filled, no data) and per zone and month with an expected error | Phase 2 (fix 7) |
| places | L2 global + static sources + counts + confidence outputs | Stronger non-fire mask in flare tiles; MODIS source per tile chosen on 2016-2025 and judged on 2012-2015; borrowed factors rebuilt without each tile and tested; confidence drift per tile; Bangladesh spotlight with a first harmonised calendar. Every choice under a rule fixed before the run | Phase 2 (fix 8) |
| resolution | L2 global + factors | 2.5-degree factors (D28), the 2018 MODIS version test, the regional 2012-step test (option E); writes `counts/fine_monthly.parquet` for later stages | Phase 2 |
| daily | L2 + resolution | skill of MODIS x k by day, week and month (out of sample); labels for pre-2012 time scales; UTC vs local solar day | Phase 2 |
| external | NASA Earthdata (GFED4, MERRA-2) | burned area and cloud per month, reduced to small tables | Phase 2 |
| harmonise | L2 + all choices above | L3 harmonised record, hybrid fills with scaling, region tables, provenance | Phase 2 |
| checks | L3 + fine monthly + outside data | final checks of Phase 2 (see `docs/RUNBOOK.md`) | Phase 2 |
| calendar | L3 | climatology, anomalies, critical periods | Phase 3 |

Each stage writes `data/processed/<dataset>/report_<stage>.json` with row counts and rejection reasons, so every number in the app can be traced back.

## NASA data sources used

| Dataset | What we use it for | How |
| --- | --- | --- |
| NASA FIRMS MODIS Collection 6.1 active fires (Terra and Aqua, 1 km), https://firms.modaps.eosdis.nasa.gov/ | The long record, 2000 onwards | Yearly world archive files 2002-2025 plus a form download for 2026; cleaned, gridded to 0.05 deg cell-days |
| NASA FIRMS VIIRS 375 m active fires (Suomi NPP), https://firms.modaps.eosdis.nasa.gov/ | The reference sensor the MODIS record is calibrated to, 2012 onwards | Same route as MODIS; paired with MODIS on the same cell and day to fit and test the calibration |
| NASA FIRMS VIIRS 375 m active fires (NOAA-20), https://firms.modaps.eosdis.nasa.gov/ | Continuity: NASA stops delivering Suomi NPP data on 1 Nov 2026; NOAA-20 carries the same instrument (2018 onwards) | Same route; compared with S-NPP over 2018-2026 (yearly detection ratio 0.97-1.07) |

Dataset citations (as requested by NASA FIRMS):

- MODIS Collection 6 Hotspot / Active Fire Detections MCD14ML distributed from NASA FIRMS. https://doi.org/10.5067/FIRMS/MODIS/MCD14ML
- MODIS Collection 61 NRT Hotspot / Active Fire Detections MCD14DL distributed from NASA FIRMS. https://doi.org/10.5067/FIRMS/MODIS/MCD14DL.NRT.0061
- NRT VIIRS 375 m Active Fire product VNP14IMGT distributed from NASA FIRMS. https://doi.org/10.5067/FIRMS/VIIRS/VNP14IMGT_NRT.002
- NRT VIIRS 375 m Active Fire product VJ114IMGTDL_NRT distributed from NASA FIRMS. https://doi.org/10.5067/FIRMS/VIIRS/VJ114IMGT_NRT.002

We acknowledge the use of data from NASA's Fire Information for Resource Management System
(FIRMS, https://www.earthdata.nasa.gov/firms), part of NASA's Earth Science Data and Information System (ESDIS).

Outside data used only for checks (Phase 2 options B and C):

- GFED4 monthly burned area without small fires (Giglio, Randerson and van der Werf 2013), ORNL DAAC. https://doi.org/10.3334/ORNLDAAC/1293
- MERRA-2 monthly radiation diagnostics M2TMNXRAD v5.12.4, total cloud area fraction CLDTOT (Global Modeling and Assimilation Office), NASA GES DISC.

Other data: Florida state outline (`config/regions/florida.geojson`), a public-domain US state boundary.

## Tools and references

Python, pandas, NumPy, PyArrow, PyYAML, Shapely, pytest; Google Colab and Google Drive;
GitHub Actions. MODIS C6.1 and VIIRS 375 m active-fire user guides (NASA FIRMS).

- Giglio, L., Schroeder, W., Justice, C. O. (2016). The collection 6 MODIS active fire detection
  algorithm and fire products. Remote Sensing of Environment, 178, 31-41.
- Schroeder, W., Oliva, P., Giglio, L., Csiszar, I. A. (2014). The New VIIRS 375 m active fire
  detection data product: Algorithm description and initial assessment. Remote Sensing of
  Environment, 143, 85-96.
- SHIFT v1 Florida prototype (Team Squadron Zero): https://github.com/nasifnaser06-cyber/Project_S.H.I.F.T

## Use of AI

Code, tests and documentation were written with AI help (Claude), and the team used Gemini and
ChatGPT as tutors for running the steps. The method, the decisions and every data run are the
team's. Numbers come only from tested code. Full details: `docs/AI_USE.md`.

## Licence

Apache License 2.0 (see `LICENSE`).

## Website

`web/` holds the SHIFT website (React + MapLibre, static files only). The pipeline's `export` stage
writes `site_data.zip`; it is published as the GitHub release `site-data`, and the workflow
`Website` (`.github/workflows/pages.yml`) downloads it, builds the site and publishes it on GitHub
Pages. See `web/README.md` for running it locally or offline.
