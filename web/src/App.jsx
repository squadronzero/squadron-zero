import { useEffect, useMemo, useState } from 'react';
import MapView from './MapView.jsx';
import { Calendar, MonthGrid, Trend } from './Charts.jsx';
import { MAX_SPAN, areaRange, areaSeries, csv, json, monthLayer, monthly, toCSV } from './data.js';
import { Switch2012, Validation, Method } from './Pages.jsx';

// Test regions (D18) and the local case first. Boxes: west, south, east, north.
export const PLACES = [
  { id: 'bangladesh', name: 'Bangladesh', box: [88.0, 20.5, 92.7, 26.7] },
  { id: 'africa_savanna', name: 'Central African savanna', box: [15, 4, 30, 12] },
  { id: 'australia_savanna', name: 'Northern Australia', box: [129, -17, 138, -11] },
  { id: 'amazon_arc', name: 'Amazon deforestation arc', box: [-64, -13, -52, -7] },
  { id: 'siberia_boreal', name: 'Siberian boreal forest', box: [100, 55, 130, 65] },
  { id: 'iberia_med', name: 'Iberian Peninsula', box: [-9.5, 36, 3.5, 44] },
  { id: 'punjab_crop', name: 'Punjab crop burning', box: [73, 27, 80, 32] },
  { id: 'kalimantan_peat', name: 'Kalimantan peatlands', box: [109, -4, 118, 2] },
  { id: 'java_volcanic', name: 'Java', box: [105, -9, 115, -6] },
  { id: 'florida', name: 'Florida', box: [-87.6, 24.5, -80, 31] },
];

const THEMES = {
  light: { name: 'light', surface: '#FCFCFB', ink: '#17283A', muted: '#5B6873', sea: '#C3D3DA', land: '#E4E9E5', border: '#A9B6B9',
    zero: '#ECEFEC', ramp: ['#FBE6D2', '#F6C49C', '#EC9A64', '#D96B3A', '#B4461F', '#7E2A12'],
    cls: { low: '#4F7FB8', normal: '#D4D8D6', unusual: '#E58A55', critical: '#A63A18' },
    line: '#17283A', band: 'rgba(23,40,58,0.14)', nodataBg: '#F1F2F1', nodataLine: '#9AA4A8',
    volcano: '#6B3FA0', industry: '#2E6E73', offshore: '#2F5DA8' },
  dark: { name: 'dark', surface: '#121A22', ink: '#E6ECEF', muted: '#97A3AC', sea: '#0D1A24', land: '#1D2933', border: '#33424E',
    zero: '#1A242D', ramp: ['#4A2A1C', '#7A3A1E', '#B4532A', '#DE7740', '#F7A26E', '#FFD3B0'],
    cls: { low: '#6F9BCB', normal: '#3E4446', unusual: '#B4532A', critical: '#F7A26E' },
    line: '#E6ECEF', band: 'rgba(230,236,239,0.16)', nodataBg: '#1A242D', nodataLine: '#55636E',
    volcano: '#B08AE0', industry: '#5FB3B9', offshore: '#7EA6E6' },
};

function useTheme() {
  const q = window.matchMedia('(prefers-color-scheme: dark)');
  const [dark, setDark] = useState(q.matches);
  useEffect(() => { const f = (e) => setDark(e.matches); q.addEventListener('change', f); return () => q.removeEventListener('change', f); }, []);  // eslint-disable-line
  return THEMES[dark ? 'dark' : 'light'];
}

const VIEWS = [['explore', 'Explore'], ['switch', 'The 2012 switch'], ['validation', 'Validation'], ['method', 'Method and data']];

export default function App() {
  const theme = useTheme();
  const [view, setView] = useState(() => (location.hash.slice(1) in Object.fromEntries(VIEWS) ? location.hash.slice(1) : 'explore'));
  const [meta, setMeta] = useState(null);
  const [error, setError] = useState(null);
  const [month, setMonth] = useState(null);
  const [fires, setFires] = useState(null);
  const [notFire, setNotFire] = useState(null);
  const [showNotFire, setShowNotFire] = useState(false);
  const [place, setPlace] = useState(PLACES[0]);
  const [drawing, setDrawing] = useState(false);
  const [variable, setVariable] = useState('frp');
  const [series, setSeries] = useState(null);
  const [loading, setLoading] = useState(false);
  const [unc, setUnc] = useState(null);
  const [harm, setHarm] = useState(null);

  useEffect(() => { location.hash = view; }, [view]);
  useEffect(() => {
    json('meta.json').then((m) => {
      setMeta(m);
      const end = new Date(m.end + 'T00:00:00Z');
      const prev = new Date(Date.UTC(end.getUTCFullYear(), end.getUTCMonth() - 1, 1));
      setMonth(`${prev.getUTCFullYear()}-${String(prev.getUTCMonth() + 1).padStart(2, '0')}`);
    }).catch(() => setError('The fire data could not be loaded. If you opened the file directly, serve the folder instead (see Method and data).'));
    json('uncertainty.json').then(setUnc).catch(() => {});
    json('validation/report_harmonise.json').then(setHarm).catch(() => {});
  }, []);
  useEffect(() => { if (month) monthLayer(month).then(setFires).catch(() => setFires(null)); }, [month]);
  useEffect(() => {
    if (!showNotFire || notFire) return;
    json('notfire.json').then((d) => setNotFire({ type: 'FeatureCollection', features: d.cells.map(([lat, lon, type, years]) =>
      ({ type: 'Feature', geometry: { type: 'Point', coordinates: [lon, lat] }, properties: { type, years } })) }));
  }, [showNotFire, notFire]);

  const box = place && { w: place.box[0], s: place.box[1], e: place.box[2], n: place.box[3] };
  useEffect(() => {
    if (!meta || !box) return;
    let live = true;
    setLoading(true);
    areaSeries(meta, box).then((s) => { if (live) { setSeries(s); setLoading(false); } })
      .catch(() => { if (live) { setSeries(null); setLoading(false); } });
    return () => { live = false; };
  }, [meta, place]);  // eslint-disable-line react-hooks/exhaustive-deps

  const mon = useMemo(() => (meta && series ? monthly(meta, series, variable) : null), [meta, series, variable]);
  const range = useMemo(() => (series && unc ? areaRange(unc, series.fineW, variable) : null), [series, unc, variable]);
  const falseAlarm = harm?.anomaly_rule?.plain?.false_alarm_pct ?? 27;

  const months = useMemo(() => {
    if (!meta) return [];
    const out = [];
    const s = new Date(meta.start + 'T00:00:00Z'), e = new Date(meta.end + 'T00:00:00Z');
    for (let d = new Date(s); d <= e; d = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 1)))
      out.push(`${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, '0')}`);
    return out;
  }, [meta]);

  const onBox = (b) => {
    const tooBig = b.e - b.w > MAX_SPAN || b.n - b.s > MAX_SPAN;
    if (tooBig) { setError(`Draw a smaller area: at most ${MAX_SPAN} degrees on each side.`); return; }
    setError(null);
    setPlace({ id: 'custom', name: `Your area (${b.s.toFixed(1)} to ${b.n.toFixed(1)}°, ${b.w.toFixed(1)} to ${b.e.toFixed(1)}°)`, box: [b.w, b.s, b.e, b.n] });
  };

  const download = () => {
    const blob = new Blob([toCSV(meta, series)], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `shift_${place.id}_daily.csv`;
    a.click();
  };

  const totals = mon && summary(mon.rows);

  return (
    <div className={`app theme-${theme.name}`}>
      <header className="top">
        <div className="brand">
          <h1>SHIFT</h1>
          <p>One fire record from two generations of NASA satellites, 2002 to today</p>
        </div>
        <nav aria-label="Views">
          {VIEWS.map(([id, label]) => (
            <button key={id} className={view === id ? 'tab on' : 'tab'} aria-current={view === id ? 'page' : undefined} onClick={() => setView(id)}>{label}</button>
          ))}
        </nav>
      </header>

      {view === 'explore' && (
        <main className="explore">
          <section className="mapside">
            <MapView key={theme.name} theme={theme} fires={fires} notFire={notFire} showNotFire={showNotFire}
              box={box} drawing={drawing} onBox={onBox} onDrawEnd={() => setDrawing(false)} />
            <div className="mapbar">
              <label>Map month
                <select value={month || ''} onChange={(e) => setMonth(e.target.value)}>
                  {months.slice().reverse().map((m) => <option key={m} value={m}>{m}</option>)}
                </select>
              </label>
              <label className="check"><input type="checkbox" checked={showNotFire} onChange={(e) => setShowNotFire(e.target.checked)} />
                Show heat that isn't fire</label>
              {showNotFire && <span className="kinds">
                <span className="swatch round" style={{ background: theme.industry }} /> gas flares, industry
                <span className="swatch round" style={{ background: theme.volcano }} /> volcanoes
                <span className="swatch round" style={{ background: theme.offshore }} /> offshore
              </span>}
            </div>
          </section>

          <section className="story" aria-live="polite">
            <div className="picker">
              <label>Place
                <select value={place.id} onChange={(e) => { const p = PLACES.find((x) => x.id === e.target.value); if (p) { setError(null); setPlace(p); } }}>
                  {place.id === 'custom' && <option value="custom">Your area</option>}
                  {PLACES.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
                </select>
              </label>
              <button className={drawing ? 'action on' : 'action'} onClick={() => setDrawing(!drawing)}>
                {drawing ? 'Drag on the map…' : 'Draw an area'}
              </button>
              <div className="seg" role="group" aria-label="Measure">
                <button className={variable === 'frp' ? 'on' : ''} onClick={() => setVariable('frp')}>Fire power</button>
                <button className={variable === 'cells' ? 'on' : ''} onClick={() => setVariable('cells')}>Burning cells</button>
              </div>
            </div>
            {error && <p className="notice" role="alert">{error}</p>}

            <h2>{place.name}</h2>
            {loading && <p className="muted">Loading the daily record…</p>}
            {!loading && series && series.tiles === 0 && <p className="muted">No fire was recorded in this area.</p>}
            {!loading && series && mon && series.tiles > 0 && (
              <>
                <p className="lede">
                  {totals.unusual} unusual and {totals.critical} critical months since {mon.baseline[0]}, judged against its own
                  {' '}{mon.baseline[0]} to {mon.baseline[1]} normal. {totals.nodataDays > 0 ? `${totals.nodataDays} days had no satellite data and are shown as gaps, not as quiet days.` : ''}
                </p>
                <h3>Every day since 2002</h3>
                <Calendar meta={meta} series={series} variable={variable} theme={theme} />
                <h3>Each month against its normal</h3>
                <MonthGrid monthly={mon} variable={variable} theme={theme} falseAlarm={falseAlarm} />
                <h3>Monthly totals</h3>
                <Trend monthly={mon} range={range} variable={variable} theme={theme} />
                <p className="small">Values are on the VIIRS (S-NPP) scale throughout. Before 20 January 2012 they are MODIS
                  observations put on that scale with a factor measured for this place and month; the shaded band is that
                  factor's 95% range. <button className="link" onClick={download}>Download the daily values (CSV)</button></p>
              </>
            )}
          </section>
        </main>
      )}
      {view === 'switch' && <Switch2012 theme={theme} />}
      {view === 'validation' && <Validation />}
      {view === 'method' && <Method meta={meta} />}
    </div>
  );
}

function summary(rows) {
  let unusual = 0, critical = 0, nodataDays = 0;
  for (const r of rows) { if (r.cls === 'unusual') unusual++; if (r.cls === 'critical') critical++; nodataDays += r.nodata; }
  return { unusual, critical, nodataDays };
}

export { csv };
