# MODIS-VIIRS-Harmonized-Global-Burning-Activity-Calendar-NASA-Space-Apps-Challenge-2026

**SHIFT** (Satellite Harmonization & Integrated Fire Tracking) by Team Squadron Zero, for the NASA Space Apps Challenge 2026: *Harmonization of MODIS and VIIRS Hot Spots*.

SHIFT turns MODIS (2000 onwards) and VIIRS (2012 onwards) active-fire detections into one consistent, uncertainty-aware burning-activity calendar for any area on Earth.

Version **0.0**. The system is built so that new source data can be swapped in without changing anything downstream: only the adapter and `config/settings.yaml` should need to change.

## Repository layout

```
config/settings.yaml        every tunable value (grid, filters, periods, paths)
docs/
  SCIENTIFIC_SPECIFICATION.md  scientific questions and scope
  DATA_CONTRACT.md             exact format of every dataset (L1, L2, L3)
  COLAB_GUIDE.md               how to run the pipeline on Google Colab
pipeline/
  shift/adapters/firms_csv.py  converts FIRMS CSV files to Level 1
  shift/ingest.py              stage 1: clean, filter, de-duplicate -> L1 detections
  shift/aggregate.py           stage 2: grid to 0.05 deg cell-days -> L2
  shift/fetch.py               downloads FIRMS yearly world files straight to cell-days
  shift/run.py                 one command runs every stage
  tests/                       automated checks
data/                          not in git; raw files live on shared Google Drive
```

Planned: `api/` (FastAPI) and `web/` (React + MapLibre) in later phases.

## Quick start

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
| fetch | FIRMS yearly world zips | L2 cell-days (global, no raw kept) | Done |
| ingest | FIRMS CSV (archive + NRT) | L1 detections | Done |
| aggregate | L1 | L2 cell-days, 0.05 deg | Done |
| calibrate | L2 overlap 2012-2021 | v1 OLS model + validation at 4 scales | Done (v1 method) |
| harmonise | L2 + model | L3 harmonised record | Phase 2 |
| validate | L3 + holdout years | metrics per region | Phase 2 |
| calendar | L3 | climatology, anomalies, critical periods | Phase 3 |

Each stage writes `data/processed/report_<stage>.json` with row counts and rejection reasons, so every number in the app can be traced back.

## Data sources

- NASA FIRMS MODIS Collection 6.1 active fires (Terra and Aqua)
- NASA FIRMS VIIRS 375 m active fires (Suomi NPP)

## Licence

MIT
