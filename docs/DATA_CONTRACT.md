# Data Contract

Version 0.0 · Squadron Zero

This document fixes the shape of every dataset the SHIFT pipeline produces. The website, the API and every analysis read only these formats. When new source data arrives (for example the datasets NASA names in the full challenge statement on 28 Oct 2026), an **adapter** converts it into Level 1. Nothing downstream of Level 1 changes.

```
raw files (any format)  ->  adapter  ->  L1 detections  ->  L2 cell-days  ->  L3 harmonised record  ->  API  ->  website
```

Field names are lower snake case. Times are UTC. Distances are in degrees (WGS84). Power is in megawatts (MW).

---

`<dataset>` is `florida` or `global` (see `config/settings.yaml`). For the global dataset, years up to `fetch.last_year` are built by the fetch stage straight from FIRMS yearly world files, so they exist only at Level 2.

## Level 1: Detections

One row per satellite active-fire detection, after cleaning and filtering.

Storage: Parquet, `data/processed/<dataset>/detections/sensor_family=<MODIS|VIIRS>/year=<YYYY>/month=<M>/`

| Field | Type | Description |
| --- | --- | --- |
| `acq_datetime_utc` | timestamp (UTC) | Acquisition date and time |
| `acq_date` | date | Acquisition date (UTC) |
| `year` | int16 | Acquisition year |
| `month` | int8 | Acquisition month |
| `lat` | float64 | Latitude of pixel centre, -90 to 90 |
| `lon` | float64 | Longitude of pixel centre, -180 to 180 |
| `sensor_family` | string | `MODIS` or `VIIRS` |
| `satellite` | string | `Terra`, `Aqua`, `S-NPP`, `NOAA-20`, `NOAA-21` |
| `frp_mw` | float64 | Fire radiative power, MW |
| `confidence_class` | string | `low`, `nominal` or `high` |
| `confidence_raw` | string | Confidence exactly as delivered by the source |
| `daynight` | string | `D` or `N` |
| `fire_type` | int8 | FIRMS type: 0 vegetation fire, 1 volcano, 2 static land source, 3 offshore; -1 if unknown |
| `type_known` | bool | False when the source carries no type field (NRT files) |
| `scan_km` | float64 | Along-scan pixel size, km |
| `track_km` | float64 | Along-track pixel size, km |
| `source_product` | string | Source product, e.g. `MODIS_C6.1_archive`, `VIIRS_SNPP_NRT` |
| `source_file` | string | File name the row came from |

Rules:

1. Only rows passing the filters in `config/settings.yaml` are kept. If the dataset has a boundary under `regions.clip` (Florida does), rows outside it are dropped and counted as `outside_region`.
2. Duplicate detections (same satellite, time, latitude and longitude) are removed. Archive rows win over NRT rows.
3. Detections with no FIRMS type field (near-real-time files) that fall in a grid cell FIRMS labelled as a non-fire heat source in at least `static_mask.min_years` archive years are dropped as `static_source_masked` (mask built by the scan stage).
4. Every rejected row is counted by reason in the run report.

---

## Level 2: Cell-days

One row per grid cell, per day, per satellite, where at least one detection occurred. Cells with no detection are not stored (the grid is sparse).

Storage: Parquet, `data/processed/<dataset>/celldays/sensor_family=<MODIS|VIIRS>/year=<YYYY>/month=<M>/`

Each partition folder can hold more than one file: `part-0.parquet` (MODIS, VIIRS S-NPP, and
anything built by the aggregate stage) and `part-noaa20.parquet` (VIIRS NOAA-20 from the fetch
stage, D8). Always read the whole folder.

**Never add up the VIIRS family without choosing satellites.** From 2018 the VIIRS family holds
two satellites (S-NPP and NOAA-20) seeing the same fires, so a family total counts most fires
twice. Filter on `satellite` first.

| Field | Type | Description |
| --- | --- | --- |
| `acq_date` | date | Day (UTC) |
| `year` | int16 | Year |
| `month` | int8 | Month |
| `cell_id` | int64 | Grid cell index (see below) |
| `cell_lat` | float64 | Latitude of cell centre |
| `cell_lon` | float64 | Longitude of cell centre |
| `sensor_family` | string | `MODIS` or `VIIRS` |
| `satellite` | string | Satellite name |
| `n_detections` | int32 | Number of detections |
| `frp_sum_mw` | float64 | Sum of FRP |
| `frp_mean_mw` | float64 | Mean FRP |
| `frp_max_mw` | float64 | Maximum FRP |
| `n_day` | int32 | Daytime detections |
| `n_night` | int32 | Night-time detections |
| `n_high_conf` | int32 | High-confidence detections |
| `n_type_unknown` | int32 | Detections with unknown fire type (NRT) |

### Grid definition

Resolution `r` = `grid.resolution_deg` (0.05° in version 0.0).

```
n_cols = round(360 / r)
row    = floor((lat + 90)  / r), clipped to [0, n_rows - 1]
col    = floor((lon + 180) / r), clipped to [0, n_cols - 1]
cell_id  = row * n_cols + col
cell_lat = -90  + (row + 0.5) * r
cell_lon = -180 + (col + 0.5) * r
```

The same `cell_id` always refers to the same place, so outputs from different runs and sensors join directly.

---

## Level 3: Harmonised record (defined in Phase 2)

Will add the calibrated MODIS-era values and the native VIIRS-era values on one scale, with an uncertainty field and a `source_sensor` flag (`MODIS_Calibrated` or `VIIRS_Native`), following the v1 Florida method.

---

## Changing the contract

A change to any field name, type or meaning must be recorded in the roadmap Decision log and in this file, with a version number increase.
