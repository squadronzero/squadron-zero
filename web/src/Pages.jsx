import { useEffect, useMemo, useState } from 'react';
import { csv, json } from './data.js';

const NAMES = { global: 'Whole world', bangladesh: 'Bangladesh', africa_savanna: 'Central African savanna', australia_savanna: 'Northern Australia',
  amazon_arc: 'Amazon deforestation arc', siberia_boreal: 'Siberian boreal forest', iberia_med: 'Iberian Peninsula',
  punjab_crop: 'Punjab crop burning', kalimantan_peat: 'Kalimantan peatlands', java_volcanic: 'Java', florida: 'Florida' };

// ------------------------------------------------------------------ the 2012 switch: raw vs harmonised
export function Switch2012({ theme }) {
  const [rows, setRows] = useState(null);
  const [steps, setSteps] = useState(null);
  const [region, setRegion] = useState('global');
  const [mode, setMode] = useState('raw');
  useEffect(() => {
    csv('validation/l3__yearly_totals.csv').then(setRows).catch(() => setRows([]));
    csv('validation/l3__step_2012.csv').then(setSteps).catch(() => setSteps([]));
  }, []);
  const data = useMemo(() => (rows || []).filter((r) => r.region === region && r.year >= 2003)
    .map((r) => ({ year: r.year, raw: r.year < 2012 ? r.MODIS_cells : r['S-NPP_cells'], harmonised: r.cells_h })), [rows, region]);
  const step = (steps || []).find((s) => s.region === region);
  if (!rows) return <main className="page"><p className="muted">Loading…</p></main>;
  const lastYear = data.length ? data[data.length - 1].year : null;
  const w = 720, h = 300, l = 56, r = 12, t = 16, b = 28;
  const key = mode;
  const max = Math.max(1, ...data.map((d) => Math.max(d.raw || 0, d.harmonised || 0)));
  const bw = (w - l - r) / Math.max(data.length, 1);
  const y = (v) => t + (1 - v / max) * (h - t - b);
  return (
    <main className="page">
      <h2>Raw records show fires more than doubling in 2012. That is the satellite, not the planet.</h2>
      <p className="lede">In January 2012 the record moves from MODIS (1 km pixels) to VIIRS (375 m pixels), which sees
        small fires MODIS misses. Joined end to end, the two make a cliff. SHIFT measures how each place's fires look
        to both satellites over the years they flew together and puts the older years on the newer scale.</p>
      <div className="picker">
        <label>Region<select value={region} onChange={(e) => setRegion(e.target.value)}>
          {[...new Set(rows.map((r) => r.region))].map((id) => <option key={id} value={id}>{NAMES[id] || id}</option>)}
        </select></label>
        <div className="seg" role="group" aria-label="Record">
          <button className={mode === 'raw' ? 'on' : ''} onClick={() => setMode('raw')}>Raw record</button>
          <button className={mode === 'harmonised' ? 'on' : ''} onClick={() => setMode('harmonised')}>SHIFT harmonised</button>
        </div>
      </div>
      <figure className="chart">
        <svg viewBox={`0 0 ${w} ${h}`} className="wide" role="img" aria-label={`Yearly burning cells, ${mode}`}>
          {[0, 0.5, 1].map((f) => <g key={f}><line x1={l} x2={w - r} y1={y(f * max)} y2={y(f * max)} className="grid" />
            <text x={l - 6} y={y(f * max) + 4} className="axis" textAnchor="end">{(f * max / 1e6).toFixed(1)}M</text></g>)}
          {data.map((d, i) => {
            const v = d[key] || 0;
            const fill = d.year < 2012 ? (mode === 'raw' ? theme.cls.normal : theme.ramp[3]) : theme.ramp[4];
            return <g key={d.year}>
              <rect x={l + i * bw + 1} y={y(v)} width={bw - 2} height={y(0) - y(v)} rx={2} fill={fill}>
                <title>{d.year}: {Math.round(v).toLocaleString()} burning cells{d.year === lastYear ? ' (year not complete)' : ''}</title>
              </rect>
              {(d.year % 3 === 0) && <text x={l + i * bw + bw / 2} y={h - 8} className="axis" textAnchor="middle">{d.year}</text>}
            </g>;
          })}
          <line x1={l + data.findIndex((d) => d.year === 2012) * bw} x2={l + data.findIndex((d) => d.year === 2012) * bw}
            y1={t} y2={h - b} className="switch" />
        </svg>
        <figcaption className="legend">Burning cells per year. Bars before 2012 are MODIS{mode === 'raw' ? ' as observed' : ' on the VIIRS scale'}; from 2012, VIIRS. {lastYear} is not complete.</figcaption>
      </figure>
      {step && (
        <div className="facts">
          <p><b>{step.cells_step_raw_switch?.toFixed(2)}×</b><span>jump at the switch in the raw record (VIIRS 2013 to 2015 against MODIS 2009 to 2011)</span></p>
          <p><b>{step.cells_step_harmonised?.toFixed(2)}×</b><span>the same comparison after SHIFT</span></p>
          <p><b>{step.cells_step_modis_only?.toFixed(2)}×</b><span>the real change, seen by MODIS alone over the same years</span></p>
        </div>
      )}
    </main>
  );
}

// ------------------------------------------------------------------ validation
function Table({ rows, cols, labels, fmt = {} }) {
  if (!rows?.length) return <p className="muted">Not available in this data release.</p>;
  return (
    <div className="tablewrap"><table>
      <thead><tr>{cols.map((c) => <th key={c} scope="col">{labels?.[c] || c}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={i}>{cols.map((c) => <td key={c}>{fmt[c] ? fmt[c](r[c], r) : r[c] ?? ''}</td>)}</tr>)}</tbody>
    </table></div>
  );
}

const one = (v) => (v == null ? '' : `${(+v).toFixed(1)}%`);
const LINKS = { 'S-NPP <- NOAA-20': 'VIIRS S-NPP ← NOAA-20', 'S-NPP <- D19': 'VIIRS S-NPP ← MODIS', 'MODIS <- Aqua': 'MODIS ← Aqua alone', 'MODIS <- Terra': 'MODIS ← Terra alone' };
const LEVELS = { 'own month': 'Its own data, that month', 'own all-year': 'Its own data, whole year', 'zone (without the tile)': 'Its 30° neighbourhood', 'world (without the tile)': 'The whole world' };
const pct = (v) => (v == null ? '' : `${v > 0 ? '+' : ''}${(+v).toFixed(1)}%`);

export function Validation() {
  const [d, setD] = useState({});
  useEffect(() => {
    const get = (k, p, f) => f(`validation/${p}`).then((v) => setD((o) => ({ ...o, [k]: v }))).catch(() => {});
    get('harm', 'report_harmonise.json', json);
    get('steps', 'l3__step_2012.csv', csv);
    get('res', 'counts__resolution_region_steps.csv', csv);
    get('fill', 'nodata__fill_test_scores.csv', csv);
    get('borrow', 'places__borrowed_factor_test.csv', csv);
  }, []);
  const g = d.harm?.gate_2012_step?.global;
  const fillRows = useMemo(() => {
    const out = {};
    (d.fill || []).filter((r) => r.variable === 'frp').forEach((r) => { (out[r.link] ||= { link: r.link })[r.method] = r.abs_error_pct; });
    return Object.values(out);
  }, [d.fill]);
  return (
    <main className="page">
      <h2>Every choice was tested on years the model never saw, under a rule written before the run.</h2>
      {g && (
        <div className="facts">
          <p><b>{pct(g.cells_gap_pct)}</b><span>difference between SHIFT's 2012 step and the real change, burning cells, whole world</span></p>
          <p><b>{pct(g.frp_gap_pct)}</b><span>the same for fire power</span></p>
          <p><b>{d.harm.anomaly_rule?.plain?.caught_pct}%</b><span>of truly unusual months that the MODIS years also flag (false alarms {d.harm.anomaly_rule?.plain?.false_alarm_pct}%)</span></p>
        </div>
      )}
      <h3>No jump at 2012, region by region</h3>
      <p className="small">The 2012 step (average of 2013 to 2015 against 2009 to 2011), SHIFT compared with MODIS alone. A gap near zero means the switch left no mark.</p>
      <Table rows={d.res ? d.res.map((r) => ({ ...r })) : null} cols={['region', 'coarse', 'fine_shrunk']}
        labels={{ region: 'Region', coarse: 'Gap, 10° factors', fine_shrunk: 'Gap, final 2.5° factors' }}
        fmt={{ region: (v) => NAMES[v] || v, coarse: pct, fine_shrunk: pct }} />
      <h3>Filling satellite outages</h3>
      <p className="small">Real outages were simulated on days when both satellites worked, and each month rebuilt. Error of the month's fire power total.</p>
      <Table rows={fillRows} cols={['link', 'none', 'scale', 'substitute', 'hybrid']}
        labels={{ link: 'Gap in ← filled from', none: 'Counted as zero fire', scale: 'Month scaled up', substitute: 'Partner satellite', hybrid: 'Partner, cautious' }}
        fmt={{ link: (v) => LINKS[v] || v, none: one, scale: one, substitute: one, hybrid: one }} />
      <h3>How good is a borrowed factor?</h3>
      <p className="small">Places with little fire borrow their factor from a wider area. Each tested place was rebuilt without its own data to measure that honestly.</p>
      <Table rows={d.borrow} cols={['level', 'fw_median_abs_total_error_pct', 'tiles_within_limit_pct']}
        labels={{ level: 'Factor from', fw_median_abs_total_error_pct: 'Typical error', tiles_within_limit_pct: 'Places within ±10%' }}
        fmt={{ level: (v) => LEVELS[v] || v, fw_median_abs_total_error_pct: one, tiles_within_limit_pct: one }} />
      <p className="small">Every pipeline stage writes a full report; they ship with this site's data and are described in the
        {' '}<a href="https://github.com/squadronzero/squadron-zero">code repository</a>.</p>
    </main>
  );
}

// ------------------------------------------------------------------ method and data
export function Method({ meta }) {
  return (
    <main className="page prose">
      <h2>How SHIFT builds one record from two satellites</h2>
      <p>NASA's MODIS instruments (Terra and Aqua) have mapped active fires since 2000; the VIIRS instruments (Suomi NPP from
        2012, NOAA-20 from 2018) see the same fires with finer pixels and catch many small ones MODIS misses. SHIFT turns every
        detection from NASA FIRMS into a burning 0.05° cell per day, removes heat that is not fire, and puts the MODIS years on
        the VIIRS scale.</p>
      <h3>The steps</h3>
      <ol>
        <li>Count burning cells instead of detections, so a finer sensor cannot inflate the count.</li>
        <li>Remove gas flares, industry, volcanoes and offshore platforms that NASA labelled as non-fire, in every year.</li>
        <li>For each 2.5° area and calendar month, measure the ratio of VIIRS to MODIS fire over 2012 to 2025. Areas with little fire lean on their 10° neighbourhood.</li>
        <li>Use Terra and Aqua together until 2019 and Aqua alone after, when Terra's orbit drifts.</li>
        <li>Fill satellite outages from a partner satellite flying close in time, tested to within 3 to 6 percent. Where no partner saw the area, the day is shown as no data.</li>
        <li>Carry the record past 1 November 2026, when Suomi NPP data delivery ends, with NOAA-20 tied to it.</li>
      </ol>
      <h3>What to keep in mind</h3>
      <ul>
        <li>Cloud and smoke hide fires from every satellite. A cloudy day can look like a quiet one.</li>
        <li>January to June 2002 has Terra only and is not compared with later years.</li>
        <li>Areas are summed on a 1° grid, so a drawn box is accurate to about 100 km at its edges.</li>
        <li>Before 2012 values carry the measured range of their factor, wider where fire is sparse.</li>
        <li>About one in four unusual months in the MODIS years would not be unusual to VIIRS; those months are marked.</li>
        <li>Iberia is the one test region whose 2012 step differs from MODIS alone by more than 10% (about +12%).</li>
      </ul>
      <h3>Data</h3>
      <ul className="refs">
        <li>MODIS Collection 6.1 Active Fire Detections (MCD14ML, MCD14DL), NASA FIRMS. <a href="https://doi.org/10.5067/FIRMS/MODIS/MCD14ML">doi:10.5067/FIRMS/MODIS/MCD14ML</a></li>
        <li>VIIRS 375 m Active Fire, Suomi NPP (VNP14IMGT), NASA FIRMS. <a href="https://doi.org/10.5067/FIRMS/VIIRS/VNP14IMGT_NRT.002">doi:10.5067/FIRMS/VIIRS/VNP14IMGT_NRT.002</a></li>
        <li>VIIRS 375 m Active Fire, NOAA-20 (VJ114IMGT), NASA FIRMS. <a href="https://doi.org/10.5067/FIRMS/VIIRS/VJ114IMGT_NRT.002">doi:10.5067/FIRMS/VIIRS/VJ114IMGT_NRT.002</a></li>
        <li>Giglio, Schroeder and Justice (2016), The collection 6 MODIS active fire detection algorithm and fire products. Remote Sensing of Environment 178.</li>
        <li>Schroeder, Oliva, Giglio and Csiszar (2014), The new VIIRS 375 m active fire detection data product. Remote Sensing of Environment 143.</li>
        <li>Country borders: Natural Earth, via world-atlas.</li>
      </ul>
      <p>We acknowledge the use of data and imagery from NASA's Fire Information for Resource Management System (FIRMS),
        part of NASA's Earth Science Data and Information System (ESDIS).</p>
      {meta && <p className="small">This release covers {meta.start} to {meta.end}, generated {meta.generated_utc.slice(0, 10)}.
        Code, every test and every decision: <a href="https://github.com/squadronzero/squadron-zero">github.com/squadronzero/squadron-zero</a>.
        To run offline, serve the site folder with any static server (for example <code>python -m http.server</code>) and open it in a browser.</p>}
      <p className="small">SHIFT is Team Squadron Zero's entry to the NASA Space Apps Challenge 2026. It is not an official NASA product.</p>
    </main>
  );
}
