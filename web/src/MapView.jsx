import { useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { feature } from 'topojson-client';
import countries from 'world-atlas/countries-50m.json';

// A basemap that ships with the site, so the map works with the wifi off.
const land = feature(countries, countries.objects.countries);

function style(theme) {
  return {
    version: 8,
    sources: { land: { type: 'geojson', data: land } },
    layers: [
      { id: 'sea', type: 'background', paint: { 'background-color': theme.sea } },
      { id: 'land', type: 'fill', source: 'land', paint: { 'fill-color': theme.land } },
      { id: 'borders', type: 'line', source: 'land', paint: { 'line-color': theme.border, 'line-width': 0.5 } },
    ],
  };
}

const EMPTY = { type: 'FeatureCollection', features: [] };
const boxFeature = (b) => (b ? { type: 'FeatureCollection', features: [{ type: 'Feature', properties: {},
  geometry: { type: 'Polygon', coordinates: [[[b.w, b.s], [b.e, b.s], [b.e, b.n], [b.w, b.n], [b.w, b.s]]] } }] } : EMPTY);

export default function MapView({ theme, fires, notFire, showNotFire, box, drawing, onBox, onDrawEnd }) {
  const el = useRef(null);
  const map = useRef(null);
  const ready = useRef(false);
  const latest = useRef({ onBox, onDrawEnd, drawing });
  latest.current = { onBox, onDrawEnd, drawing };

  useEffect(() => {
    const m = new maplibregl.Map({ container: el.current, style: style(theme), center: [60, 15], zoom: 1.4,
      attributionControl: false, dragRotate: false, pitchWithRotate: false });
    m.touchZoomRotate.disableRotation();
    m.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');
    m.addControl(new maplibregl.AttributionControl({ compact: true,
      customAttribution: 'Fire data: NASA FIRMS (MODIS, VIIRS) · Borders: Natural Earth' }));
    m.on('load', () => {
      m.addSource('fires', { type: 'geojson', data: EMPTY });
      m.addLayer({ id: 'fires', type: 'circle', source: 'fires', paint: {
        'circle-radius': ['interpolate', ['linear'], ['zoom'], 1, 1.1, 4, 2.6, 7, 9],
        'circle-color': ['step', ['get', 'frp'], theme.ramp[0], 50, theme.ramp[1], 250, theme.ramp[2],
          1000, theme.ramp[3], 4000, theme.ramp[4], 15000, theme.ramp[5]],
        'circle-opacity': 0.9 } });
      m.addSource('notfire', { type: 'geojson', data: EMPTY });
      m.addLayer({ id: 'notfire', type: 'circle', source: 'notfire', layout: { visibility: 'none' }, paint: {
        'circle-radius': ['interpolate', ['linear'], ['zoom'], 1, 1.4, 5, 3, 8, 7],
        'circle-color': ['match', ['get', 'type'], 1, theme.volcano, 3, theme.offshore, theme.industry],
        'circle-stroke-color': theme.surface, 'circle-stroke-width': 0.6 } });
      m.addSource('box', { type: 'geojson', data: EMPTY });
      m.addLayer({ id: 'box-fill', type: 'fill', source: 'box', paint: { 'fill-color': theme.ink, 'fill-opacity': 0.06 } });
      m.addLayer({ id: 'box-line', type: 'line', source: 'box', paint: { 'line-color': theme.ink, 'line-width': 1.5 } });
      ready.current = true;
      m.fire('shift-ready');
    });

    const tip = new maplibregl.Popup({ closeButton: false, closeOnClick: false, className: 'tip' });
    m.on('mousemove', 'fires', (e) => {
      const p = e.features[0].properties;
      tip.setLngLat(e.lngLat).setHTML(`<b>${Number(p.frp).toLocaleString()} MW</b> fire power<br>${p.cells} burning cells (S-NPP scale)`).addTo(m);
    });
    m.on('mouseleave', 'fires', () => tip.remove());
    const kinds = { 1: 'Volcano', 2: 'Gas flare or industry', 3: 'Offshore platform' };
    m.on('mousemove', 'notfire', (e) => {
      const p = e.features[0].properties;
      tip.setLngLat(e.lngLat).setHTML(`<b>${kinds[p.type] || 'Static heat source'}</b><br>Labelled non-fire by NASA in ${p.years} year${p.years > 1 ? 's' : ''}. Left out of every year.`).addTo(m);
    });
    m.on('mouseleave', 'notfire', () => tip.remove());

    // drawing a box: press, drag, release
    let start = null;
    m.on('mousedown', (e) => {
      if (!latest.current.drawing) return;
      e.preventDefault(); m.dragPan.disable(); start = e.lngLat;
    });
    m.on('mousemove', (e) => {
      if (!start) return;
      m.getSource('box')?.setData(boxFeature(norm(start, e.lngLat)));
    });
    m.on('mouseup', (e) => {
      if (!start) return;
      const b = norm(start, e.lngLat); start = null; m.dragPan.enable();
      if (b.e - b.w >= 0.5 && b.n - b.s >= 0.5) latest.current.onBox(b);
      latest.current.onDrawEnd();
    });
    map.current = m;
    return () => m.remove();
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  const whenReady = (fn) => {
    const m = map.current;
    if (!m) return;
    if (ready.current) fn(m); else m.once('shift-ready', () => fn(m));
  };

  useEffect(() => whenReady((m) => m.getSource('fires').setData(fires || EMPTY)), [fires]);
  useEffect(() => whenReady((m) => {
    m.getSource('notfire').setData(notFire || EMPTY);
    m.setLayoutProperty('notfire', 'visibility', showNotFire ? 'visible' : 'none');
  }), [notFire, showNotFire]);
  useEffect(() => whenReady((m) => {
    m.getSource('box').setData(boxFeature(box));
    if (box) m.fitBounds([[box.w, box.s], [box.e, box.n]], { padding: 40, maxZoom: 7, duration: 600 });
  }), [box]);
  useEffect(() => { if (map.current) map.current.getCanvas().style.cursor = drawing ? 'crosshair' : ''; }, [drawing]);

  return <div ref={el} className="map" role="application" aria-label="World map of harmonised fire activity" />;
}

function norm(a, b) {
  const r = (x) => Math.round(x * 4) / 4;
  return { w: r(Math.max(-180, Math.min(a.lng, b.lng))), e: r(Math.min(180, Math.max(a.lng, b.lng))),
    s: r(Math.max(-85, Math.min(a.lat, b.lat))), n: r(Math.min(85, Math.max(a.lat, b.lat))) };
}
