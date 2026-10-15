# SHIFT website

React + MapLibre (D6), static files only (D5). The data comes from the pipeline's `export` stage
(`site_data.zip`), published as the GitHub release `site-data`; the workflow
`.github/workflows/pages.yml` downloads it, builds this folder and publishes it on GitHub Pages.

Views: Explore (map, any area, daily calendar, monthly anomaly classes, trend with the 95% range
of the MODIS-era factor, CSV download, "heat that isn't fire" layer), The 2012 switch (raw vs
harmonised), Validation (the pipeline's own test results), Method and data (sources, limits).

Local or offline (the demo works with the wifi off):

    cd web
    npm ci
    mkdir -p public/data && unzip /path/to/site_data.zip -d public/data
    npm run build
    cd dist && python -m http.server 8000      # then open http://localhost:8000

The map basemap (Natural Earth borders) and the font are bundled, so nothing is fetched from the
internet at run time.
