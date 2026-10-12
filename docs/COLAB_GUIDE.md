# Running the pipeline on Google Colab

The pipeline runs on free Colab. Raw data sits in a shared Google Drive folder so every team member uses the same files.

## 1. Shared Drive folder (one person, once)

Create this in Google Drive and share `SHIFT` with the whole team as Editor:

```
SHIFT/
  data/
    raw/         FIRMS CSV files go here, original file names unchanged
    processed/   the pipeline writes here
```

Every team member must then open "Shared with me", right-click `SHIFT`, and choose **Organize > Add shortcut > My Drive**. Without the shortcut, Colab cannot see the folder at `MyDrive/SHIFT`.

## 2. Get the data (one person, once)

1. Open https://firms.modaps.eosdis.nasa.gov/download/ and sign in (Earthdata Login or email code).
2. Request **MODIS Collection 6.1** and **VIIRS S-NPP 375m**, format **CSV**.
3. Unzip the emailed files into `SHIFT/data/raw/`. Keep the original file names (`fire_archive_...`, `fire_nrt_...`); the pipeline uses them to tell archive from NRT.

## 3. Run in a Colab notebook

Paste this into ONE cell and run it. It is safe to run again at any time: it always starts from a fresh copy of the code.

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
!ls data/raw | head

%cd /content/squadron-zero/pipeline
!python -m pytest -q          # every test should pass
```

Then, when raw files are in Drive, run the pipeline in a new cell:

```python
%cd /content/squadron-zero/pipeline
!python -m shift.run          # ingest, aggregate, calibrate
```

Outputs appear in `SHIFT/data/processed/` on Drive, with a `report_<stage>.json` per stage showing rows read, rows rejected by reason, and rows kept.

## 4. If something goes wrong

| Message | Cause | Fix |
| --- | --- | --- |
| `no input files in .../data/raw` | Drive link missing or folder empty | Rerun the setup cell; check `!ls data/raw` lists CSV files |
| `ls: cannot access 'data/raw'` | No `SHIFT` shortcut in My Drive | Add the shortcut (section 1) and rerun |
| `No module named shift` | Running from the wrong folder | `%cd /content/squadron-zero/pipeline` first |
| `cannot tell MODIS from VIIRS` | A non-FIRMS CSV is in `raw/` | Move that file out of `raw/` |
| Session crashes or "out of memory" | Month of data too large | Lower `processing.chunk_rows` in `config/settings.yaml` (e.g. `250000`) |

Colab forgets everything when the session ends. Rerun the setup cell at the start of every session; the data stays safe on Drive.
