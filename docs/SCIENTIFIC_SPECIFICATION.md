# Scientific Specification

## 0. Scientific Question

### Primary Scientific Question

> **Can active-fire observations from MODIS and VIIRS be transformed into a consistent, uncertainty-aware historical record of global burning activity that enables reliable characterization of normal, unusual, and critical burning patterns across space and time?**

### Supporting Scientific Questions

#### Q1 — Sensor Comparability

How do MODIS and VIIRS active-fire observations differ under comparable observation conditions?

#### Q2 — Harmonization

Can systematic differences between MODIS and VIIRS observations be characterized and reduced sufficiently to construct a consistent multi-sensor historical record?

#### Q3 — Uncertainty

How can uncertainty, observation coverage, and data-quality information be quantified and incorporated into the harmonized record?

#### Q4 — Historical Burning Patterns

What spatial and temporal patterns of burning activity can be characterized from the harmonized multi-sensor record?

#### Q5 — Unusual and Critical Periods

How can historical baselines be used to identify statistically unusual, persistent, or exceptional periods of burning activity?

### Scientific Intent

The project aims to transform observations from multiple satellite active-fire sensors into a scientifically interpretable, harmonized historical record.

The resulting record will support analysis of:

* where burning activity occurs;
* when burning activity occurs;
* how burning activity varies across years and seasons;
* how current or selected periods compare with historical conditions;
* where observations are unusually high or persistent; and
* how confident users should be in each derived result.

The system will distinguish **observed satellite detections** from **derived estimates, statistical classifications, and analytical interpretations**.

### Scientific Scope

The primary scientific focus is the harmonization and analysis of satellite-derived active-fire observations from MODIS and VIIRS.

The project is intended to characterize observed and derived patterns of burning activity. It is **not**, by itself, a direct measurement of total fire occurrence, total burned area, biomass consumed, emissions, or ground conditions.

Where derived quantities are produced, their assumptions, uncertainty, and limitations will be explicitly documented.

### Sensors (version 0.0)

| Sensor | Satellites | Record | Role |
| --- | --- | --- | --- |
| MODIS Collection 6.1, 1 km | Terra (2000+), Aqua (2002+) | to present; both near end of mission | The long historical record that is harmonised |
| VIIRS 375 m | Suomi NPP | 2012 to 2026; NASA stops delivering its data on 1 Nov 2026 | Calibration reference |
| VIIRS 375 m | NOAA-20 | 2018 to present | Continuity reference after Suomi NPP (D8); measured against Suomi NPP over 2018-2026 |

The 2012-2026 overlap of MODIS and VIIRS is the evidence base: calibration is fitted on
2012-2021 and tested on 2022-2025 and 2026, years the model never sees.

