# Running the pipeline on Google Colab

The pipeline runs on free Colab. Data sits in a shared Google Drive folder so every team member uses the same files.

There are two datasets, each with its own folders so they never mix:

| Dataset | Raw files | Used for |
| --- | --- | --- |
| `florida` | `SHIFT/data/raw/florida/` (FIRMS form downloads, v1 box) | Checking the new pipeline against the v1 prototype |
| `global` | Yearly world files fetched automatically; `SHIFT/data/raw/global/` only for recent months | The final product |

## 1. Drive folder (one person, once)

The folder lives in the team account **squadronzero10**, and only that account runs the pipeline (files count against the storage of whoever creates them). Create:

```
SHIFT/
  data/
    raw/
      florida/     Florida FIRMS downloads: the ZIPs as they are, or their CSVs (not both)
      global/      only form downloads after fetch.last_year (1 Jan 2026 onwards), ZIPs as they are
    processed/     the pipeline writes here (florida/ and global/ subfolders)
```

squadronzero10 needs no shortcut. Teammates who only look at outputs get Viewer access; if one ever needs to run Colab with their own account, they add a shortcut (Shared with me > right-click `SHIFT` > Organize > Add shortcut > My Drive). Keep General access set to **Restricted**.

After big reruns, empty the Drive trash: deleted files still count against the 15 GB for 30 days.

## 2. Setup cell (run at the start of every session)

Paste this into ONE cell and run it. It is safe to run again at any time.

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content
!rm -rf squadron-zero
!git clone -q https://github.com/squadronzero/squadron-zero.git
%cd /content/squadron-zero
!pip install -q -r pipeline/requirements.txt

# Replace the repo's empty data folder with a link to the shared Drive folder.
# (The rm is needed: without it the link is created INSIDE data/ and the pipeline finds no files.)
!rm -rf data
!ln -s /content/drive/MyDrive/SHIFT/data data
!ls data/raw

%cd /content/squadron-zero/pipeline
!python -m pytest -q          # every test should pass
```

## 3. Florida check (v1 comparison)

Put the Florida FIRMS downloads into `SHIFT/data/raw/florida/`: the ZIPs unchanged, or their CSVs with the original names (`fire_archive_...`, `fire_nrt_...`). Detections are clipped to Florida's state outline (`config/regions/florida.geojson`, D16) to match v1. Then:

```python
%cd /content/squadron-zero/pipeline
!python -m shift.run --dataset florida
```

Outputs go to `SHIFT/data/processed/florida/`, with a `report_<stage>.json` per stage.

## 4. Global data (yearly world files)

FIRMS publishes one zip per sensor per year with every country's detections. The fetch stage downloads each year to Colab's local disk, turns it into cell-days, saves those to Drive and deletes the download. Drive only ever holds the small processed files.

Test with one year first:

```python
%cd /content/squadron-zero/pipeline
!python -m shift.run --dataset global --stage fetch --years 2020 --family MODIS
```

Then everything (MODIS 2002 to 2025, VIIRS 2012 to 2025):

```python
%cd /content/squadron-zero/pipeline
!python -m shift.run --dataset global --stage fetch
```

- Each finished year leaves a marker in `SHIFT/data/processed/global/_fetch_done/`. If Colab disconnects, rerun the setup cell and the same command: finished years are skipped.
- A year with no file on the server is listed under `missing` in the report. That is expected for the current year.
- Data after `fetch.last_year` (2025) comes from one FIRMS form request per sensor (1 Jan 2026 to today, whole world). Drop the ZIPs unchanged into `SHIFT/data/raw/global/` (no unzipping), then run `!python -m shift.run --dataset global`. The pipeline refuses form data that overlaps the yearly files, so nothing is counted twice.
- Every download is fingerprinted (sha256, size, date) in its `_fetch_done` marker.
- **VIIRS NOAA-20 (D8).** NASA stops delivering Suomi NPP data on 1 Nov 2026, so NOAA-20 is fetched as a third product, `VIIRS_NOAA20` (2018 onwards). It shares the VIIRS folders but writes its own file (`part-noaa20.parquet`), so S-NPP is never overwritten. Test one year, then the rest:
  ```python
  !python -m shift.run --dataset global --stage fetch --family VIIRS_NOAA20 --years 2020
  !python -m shift.run --dataset global --stage fetch --family VIIRS_NOAA20
  ```
  2026 NOAA-20 comes from one FIRMS form request (source "VIIRS NOAA-20 375m", whole world, 1 Jan 2026 to today), dropped unchanged into `SHIFT/data/raw/global/`. The scan stage skips NOAA-20; the static-source mask stays built from MODIS and S-NPP.
- Calibration is skipped for the global dataset for now; per-region calibration comes in Phase 2.

## 5. Checks (scan, qc, crosscheck)

Run after the global fetch. None of them change the cell-day record.

```python
%cd /content/squadron-zero/pipeline
!python -m shift.run --dataset global --stage scan        # about 25 min, resumable
!python -m shift.run --dataset global --stage qc          # a few minutes
!python -m shift.run --dataset global --stage crosscheck  # about 1 min, needs the Florida outputs
```

| Stage | What it does | Output |
| --- | --- | --- |
| `scan` | Re-reads each yearly file once: data version labels per year, cells FIRMS labels as non-fire heat sources (volcano, gas flare/industry, offshore), file fingerprints | `processed/global/_scan_done/`, `processed/global/static_sources/` (incl. `static_cells.parquet`, the static-source mask) |
| `qc` | Data health: detections per satellite per day, silent or low days, gap runs, yearly ratios (Terra:Aqua, VIIRS:MODIS, NOAA-20:S-NPP), before/after checks for the MODIS processing changes in 2018 and 2023, share of real fires inside the static-source mask per satellite | `processed/global/qc/` (`daily_coverage.parquet`, `gaps.csv`, `yearly.csv`, `monthly.csv`, `mask_overlap.csv`) |
| `crosscheck` | Compares the Florida form download with the Florida part of the global yearly files, cell-day by cell-day | `processed/global/qc/crosscheck_florida.csv`, verdict PASS or CHECK |

Once `static_cells.parquet` exists, rerun `--dataset global` and `--dataset florida`: near-real-time detections (which carry no fire-type field) that fall in known non-fire cells are then dropped and counted as `static_source_masked`.

## 6. Two-minute demo

```python
%cd /content/squadron-zero/pipeline
!python -m shift.quickstart
```

Runs the whole method on the small sample in the repo and writes only to a temporary folder,
never to Drive.

## 7. If something goes wrong

| Message | Cause | Fix |
| --- | --- | --- |
| `no input files in .../raw/florida` | Florida CSVs not in the right folder | Put them in `SHIFT/data/raw/florida/` |
| `ls: cannot access 'data/raw'` | No `SHIFT` shortcut in My Drive | Add the shortcut (section 1) and rerun |
| `No module named shift` | Running from the wrong folder | `%cd /content/squadron-zero/pipeline` first |
| `cannot tell MODIS from VIIRS` | A non-FIRMS CSV is in a raw folder | Move that file out |
| `download failed after 3 attempts` | Network hiccup or FIRMS busy | Rerun the same command; finished years are skipped |
| `already comes from the yearly world files` | A form download overlaps a fetched year | Use form downloads only for dates after 2025 |
| Session crashes or "out of memory" | A large file read at once | Lower `processing.chunk_rows` in `config/settings.yaml` (e.g. `250000`) |
