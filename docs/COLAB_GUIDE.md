# Running the pipeline on Google Colab

The pipeline runs on free Colab. Data sits in a shared Google Drive folder so every team member uses the same files.

There are two datasets, each with its own folders so they never mix:

| Dataset | Raw files | Used for |
| --- | --- | --- |
| `florida` | `SHIFT/data/raw/florida/` (FIRMS form downloads, v1 box) | Checking the new pipeline against the v1 prototype |
| `global` | Yearly world files fetched automatically; `SHIFT/data/raw/global/` only for recent months | The final product |

## 1. Shared Drive folder (one person, once)

Create this in Google Drive and share `SHIFT` with the whole team as Editor:

```
SHIFT/
  data/
    raw/
      florida/     Florida FIRMS CSV files, original file names unchanged
      global/      only form downloads after fetch.last_year (e.g. 1 Jan 2026 to today)
    processed/     the pipeline writes here (florida/ and global/ subfolders)
```

Every team member must then open "Shared with me", right-click `SHIFT`, and choose **Organize > Add shortcut > My Drive**. Without the shortcut, Colab cannot see the folder at `MyDrive/SHIFT`.

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

Unzip the Florida FIRMS downloads into `SHIFT/data/raw/florida/`, keeping the original file names (`fire_archive_...`, `fire_nrt_...`). Then:

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
- Data after `fetch.last_year` (2025) comes from one FIRMS form request per sensor (1 Jan 2026 to today, whole world), unzipped into `SHIFT/data/raw/global/`. Then run `!python -m shift.run --dataset global`. The pipeline refuses form data that overlaps the yearly files, so nothing is counted twice.
- Calibration is skipped for the global dataset for now; per-region calibration comes in Phase 2.

## 5. If something goes wrong

| Message | Cause | Fix |
| --- | --- | --- |
| `no input files in .../raw/florida` | Florida CSVs not in the right folder | Put them in `SHIFT/data/raw/florida/` |
| `ls: cannot access 'data/raw'` | No `SHIFT` shortcut in My Drive | Add the shortcut (section 1) and rerun |
| `No module named shift` | Running from the wrong folder | `%cd /content/squadron-zero/pipeline` first |
| `cannot tell MODIS from VIIRS` | A non-FIRMS CSV is in a raw folder | Move that file out |
| `download failed after 3 attempts` | Network hiccup or FIRMS busy | Rerun the same command; finished years are skipped |
| `already comes from the yearly world files` | A form download overlaps a fetched year | Use form downloads only for dates after 2025 |
| Session crashes or "out of memory" | A large file read at once | Lower `processing.chunk_rows` in `config/settings.yaml` (e.g. `250000`) |
