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
| harmonise | L2 + model | L3 harmonised record | Phase 2 |
| validate | L3 + holdout years | metrics per region | Phase 2 |
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
