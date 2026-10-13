# Sample data

`florida_sample.zip` holds real NASA FIRMS active-fire archive detections for the Florida
box (24.5-31.0 N, 87.6-80.0 W) for 2020 and 2023 only: MODIS Collection 6.1 (Terra and Aqua)
and VIIRS S-NPP 375 m. It lets anyone run the whole SHIFT method in about a minute:

```bash
cd pipeline
pip install -r requirements.txt
python -m shift.quickstart
```

Source: NASA FIRMS, https://firms.modaps.eosdis.nasa.gov/ (downloaded 12-13 Oct 2026).
FIRMS data are freely available under NASA's open data policy; please cite FIRMS when reusing them.
