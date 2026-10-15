import { useEffect, useMemo, useRef, useState } from 'react';
import { dateOf, dayIndex } from './data.js';

const fmt = (v, variable) => (variable === 'frp' ? `${Math.round(v).toLocaleString()} MW` : `${v.toFixed(v < 10 ? 1 : 0)} cells`);
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// ------------------------------------------------------------------ calendar: years x days of the year
export function Calendar({ meta, series, variable, theme }) {
  const ref = useRef(null);
  const wrap = useRef(null);
  const [tip, setTip] = useState(null);
  const [width, setWidth] = useState(300);
  const v = series[variable];
  const first = dateOf(meta, 0).getUTCFullYear();
  const last = dateOf(meta, meta.days - 1).getUTCFullYear();
  const years = last - first + 1;
  const switchDay = dayIndex(meta, '2012-01-20');

  // magnitude bins from this area's own non-zero days, so a quiet place and a busy one both read
  const bins = useMemo(() => {
    const nz = [];
    for (let k = 0; k < meta.days; k++) if (v[k] > 0 && !series.nodata[k]) nz.push(v[k]);
    nz.sort((a, b) => a - b);
    const q = (p) => (nz.length ? nz[Math.min(nz.length - 1, Math.floor(p * nz.length))] : 0);
    return [q(0.4), q(0.65), q(0.82), q(0.93), q(0.98)];
  }, [v, series.nodata, meta.days]);

  useEffect(() => {
    setWidth(Math.max(280, wrap.current.clientWidth));
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(280, e.contentRect.width)));
    ro.observe(wrap.current);
    return () => ro.disconnect();
  }, []);

  const left = 40, top = 18;
  const cw = (width - left - 4) / 366;
  const ch = Math.max(7, Math.min(14, cw * 3.2));
  const height = top + years * ch + 6;

  useEffect(() => {
    const c = ref.current, dpr = window.devicePixelRatio || 1;
    c.width = width * dpr; c.height = height * dpr;
    const g = c.getContext('2d');
    g.scale(dpr, dpr);
    g.clearRect(0, 0, width, height);
    g.font = '11px "Archivo Variable", sans-serif';
    g.fillStyle = theme.muted;
    g.textBaseline = 'middle';
    MONTHS.forEach((m, i) => {
      const d = Math.round(new Date(Date.UTC(2001, i, 1)) / 864e5 - new Date(Date.UTC(2001, 0, 1)) / 864e5);
      g.fillText(m, left + d * cw, 8);
    });
    for (let y = 0; y < years; y++) {
      if ((first + y) % 2 === 0 || years < 14) g.fillText(String(first + y), 2, top + y * ch + ch / 2);
    }
    const hatch = g.createPattern(hatchTile(theme), 'repeat');
    for (let k = 0; k < meta.days; k++) {
      const d = dateOf(meta, k);
      const y = d.getUTCFullYear() - first;
      const doy = Math.floor((d - Date.UTC(d.getUTCFullYear(), 0, 1)) / 864e5);
      const x = left + doy * cw, yy = top + y * ch;
      if (series.nodata[k]) { g.fillStyle = hatch; g.fillRect(x, yy, Math.max(cw, 1), ch - 1); continue; }
      const val = v[k];
      let i = -1;
      if (val > 0) { i = 0; while (i < 5 && val > bins[i]) i++; }
      g.fillStyle = i < 0 ? theme.zero : theme.ramp[i];
      g.fillRect(x, yy, Math.max(cw, 1), ch - 1);
    }
    // where VIIRS takes over
    const sd = dateOf(meta, switchDay);
    const sy = sd.getUTCFullYear() - first;
    const sx = left + Math.floor((sd - Date.UTC(sd.getUTCFullYear(), 0, 1)) / 864e5) * cw;
    g.strokeStyle = theme.ink; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(left, top + sy * ch - 0.5); g.lineTo(sx, top + sy * ch - 0.5);
    g.lineTo(sx, top + (sy + 1) * ch - 0.5); g.lineTo(width - 4, top + (sy + 1) * ch - 0.5); g.stroke();
  }, [width, height, v, series.nodata, bins, theme, meta, first, years, cw, ch, switchDay]);

  const onMove = (e) => {
    const r = ref.current.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const yr = Math.floor((y - top) / ch), doy = Math.floor((x - left) / cw);
    if (yr < 0 || yr >= years || doy < 0 || doy > 365) return setTip(null);
    const t = Date.UTC(first + yr, 0, 1) + doy * 864e5;
    const k = Math.round((t - Date.parse(meta.start + 'T00:00:00Z')) / 864e5);
    if (k < 0 || k >= meta.days || new Date(t).getUTCFullYear() !== first + yr) return setTip(null);
    setTip({ x, y, k });
  };

  let tipText = null;
  if (tip) {
    const k = tip.k;
    const date = dateOf(meta, k).toISOString().slice(0, 10);
    const src = k < dayIndex(meta, meta.terra_only_until) + 1 ? 'Terra only, not comparable'
      : k < switchDay ? 'MODIS put on the VIIRS scale' : 'VIIRS (S-NPP)';
    tipText = series.nodata[k]
      ? <><b>{date}</b><br />No data: a satellite outage covered this area. Not the same as no fire.</>
      : <><b>{date}</b><br />{fmt(v[k], variable)}<br />{src}{series.filled[k] ? ', partly filled from a partner satellite' : ''}</>;
  }

  return (
    <figure className="chart" ref={wrap}>
      <div className="calendar" onMouseMove={onMove} onMouseLeave={() => setTip(null)}>
        <canvas ref={ref} style={{ width, height }} aria-label="Daily fire calendar, one row per year" role="img" />
        {tip && <div className="hover" style={{ left: Math.min(tip.x + 12, width - 220), top: tip.y + 12 }}>{tipText}</div>}
      </div>
      <figcaption className="legend">
        <span className="swatch" style={{ background: theme.zero }} /> no fire seen
        {theme.ramp.map((c, i) => <span key={c} className="swatch" style={{ background: c }} title={i ? `above ${fmt(bins[i - 1], variable)}` : 'any fire'} />)}
        <span className="legend-text">more fire</span>
        <span className="swatch hatch" /> no data
        <span className="legend-line" /> VIIRS from 20 Jan 2012
      </figcaption>
    </figure>
  );
}

function hatchTile(theme) {
  const c = document.createElement('canvas');
  c.width = c.height = 6;
  const g = c.getContext('2d');
  g.fillStyle = theme.nodataBg; g.fillRect(0, 0, 6, 6);
  g.strokeStyle = theme.nodataLine; g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, 6); g.lineTo(6, 0); g.stroke();
  return c;
}

// ------------------------------------------------------------------ months: anomaly classes
const CLASS_TEXT = { low: 'Below normal', normal: 'Normal', unusual: 'Unusual', critical: 'Critical' };

export function MonthGrid({ monthly, variable, theme, falseAlarm }) {
  const [tip, setTip] = useState(null);
  const rows = monthly.rows;
  const years = [...new Set(rows.map((r) => r.year))];
  const cell = 18, gap = 2, left = 40, top = 16;
  const width = left + 12 * (cell + gap), height = top + years.length * (cell + gap);
  return (
    <figure className="chart">
      <div className="monthgrid">
        <svg width={width} height={height} role="img" aria-label="Monthly anomaly classes, one row per year">
          {MONTHS.map((m, i) => <text key={m} x={left + i * (cell + gap) + cell / 2} y={10} className="axis" textAnchor="middle">{m[0]}</text>)}
          {years.map((y, yi) => <text key={y} x={0} y={top + yi * (cell + gap) + cell * 0.72} className="axis">{y}</text>)}
          <defs>
            <pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="5" height="5" fill={theme.nodataBg} /><line x1="0" y1="0" x2="0" y2="5" stroke={theme.nodataLine} strokeWidth="1.2" />
            </pattern>
          </defs>
          {rows.map((r) => {
            const x = left + r.month * (cell + gap), y = top + years.indexOf(r.year) * (cell + gap);
            const fill = r.cls ? theme.cls[r.cls] : r.incomplete || r.notComparable ? 'url(#hatch)' : theme.zero;
            return (
              <g key={`${r.year}-${r.month}`} onMouseEnter={() => setTip({ r, x, y })} onMouseLeave={() => setTip(null)}>
                <rect x={x} y={y} width={cell} height={cell} rx={3} fill={fill} />
                {r.modisEra && r.cls && r.cls !== 'normal' && r.cls !== 'low' &&
                  <circle cx={x + cell - 4} cy={y + 4} r={2} fill={theme.surface} />}
              </g>
            );
          })}
        </svg>
        {tip && (
          <div className="hover" style={{ left: Math.min(tip.x + 24, width - 40), top: tip.y + 20 }}>
            <b>{MONTHS[tip.r.month]} {tip.r.year}</b><br />
            {fmt(tip.r.value, variable)}<br />
            {tip.r.notComparable ? 'Terra only: not compared with later years'
              : tip.r.incomplete ? `Not classed: ${tip.r.nodata ? `${tip.r.nodata} day(s) without data` : 'month not complete'}`
              : tip.r.cls ? `${CLASS_TEXT[tip.r.cls]} for ${MONTHS[tip.r.month]} (z = ${tip.r.z.toFixed(1)})` : 'Too few years to judge'}
            {tip.r.modisEra && tip.r.cls && (tip.r.cls === 'unusual' || tip.r.cls === 'critical') &&
              <><br />MODIS years: about {Math.round(falseAlarm)}% of such flags are not unusual in VIIRS (measured)</>}
          </div>
        )}
      </div>
      <figcaption className="legend">
        {['low', 'normal', 'unusual', 'critical'].map((c) => <span key={c}><span className="swatch" style={{ background: theme.cls[c] }} /> {CLASS_TEXT[c]}</span>)}
        <span><span className="swatch hatch" /> not classed</span>
        <span><span className="dot" /> before 2012, lower confidence</span>
      </figcaption>
    </figure>
  );
}

// ------------------------------------------------------------------ trend: monthly totals with the factor range
export function Trend({ monthly, range, variable, theme }) {
  const wrap = useRef(null);
  const [width, setWidth] = useState(300);
  const [hover, setHover] = useState(null);
  useEffect(() => {
    setWidth(Math.max(280, wrap.current.clientWidth));
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(280, e.contentRect.width)));
    ro.observe(wrap.current);
    return () => ro.disconnect();
  }, []);
  const rows = monthly.rows;
  const h = 200, l = 54, r = 10, t = 12, b = 24;
  const lo = (row) => (row.modisEra && range?.[row.month] != null ? row.value / (1 + range[row.month] / 100) : row.value);
  const hi = (row) => (row.modisEra && range?.[row.month] != null ? row.value * (1 + range[row.month] / 100) : row.value);
  const max = Math.max(1, ...rows.map(hi));
  const x = (i) => l + (i / Math.max(rows.length - 1, 1)) * (width - l - r);
  const y = (v) => t + (1 - v / max) * (h - t - b);
  const line = rows.map((row, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(row.value).toFixed(1)}`).join('');
  const pre = rows.map((row, i) => [i, row]).filter(([, row]) => row.modisEra);
  const band = pre.length
    ? pre.map(([i, row], j) => `${j ? 'L' : 'M'}${x(i).toFixed(1)},${y(hi(row)).toFixed(1)}`).join('')
      + pre.slice().reverse().map(([i, row]) => `L${x(i).toFixed(1)},${y(lo(row)).toFixed(1)}`).join('') + 'Z'
    : '';
  const si = rows.findIndex((row) => !row.modisEra);
  const ticks = [0, 0.5, 1].map((f) => f * max);
  const yearsShown = rows.map((row, i) => [row, i]).filter(([row]) => row.month === 0 && row.year % 4 === 0);
  const onMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const i = Math.round(((e.clientX - rect.left - l) / (width - l - r)) * (rows.length - 1));
    setHover(i >= 0 && i < rows.length ? i : null);
  };
  const hv = hover != null ? rows[hover] : null;
  return (
    <figure className="chart" ref={wrap}>
      <svg width={width} height={h} onMouseMove={onMove} onMouseLeave={() => setHover(null)} role="img"
        aria-label={`Monthly ${variable === 'frp' ? 'fire power' : 'burning cells'}, with the 95% range of the MODIS years`}>
        {ticks.map((v) => <g key={v}><line x1={l} x2={width - r} y1={y(v)} y2={y(v)} className="grid" />
          <text x={l - 6} y={y(v) + 4} className="axis" textAnchor="end">{compact(v)}</text></g>)}
        {yearsShown.map(([row, i]) => <text key={row.year} x={x(i)} y={h - 6} className="axis" textAnchor="middle">{row.year}</text>)}
        {band && <path d={band} fill={theme.band} />}
        <path d={line} fill="none" stroke={theme.line} strokeWidth="1.6" strokeLinejoin="round" />
        {si > 0 && <g><line x1={x(si)} x2={x(si)} y1={t} y2={h - b} className="switch" />
          <text x={x(si) + 4} y={t + 10} className="axis">VIIRS</text></g>}
        {hv && <g><line x1={x(hover)} x2={x(hover)} y1={t} y2={h - b} className="cross" />
          <circle cx={x(hover)} cy={y(hv.value)} r={4} fill={theme.line} stroke={theme.surface} strokeWidth="2" /></g>}
      </svg>
      {hv && (
        <div className="hover" style={{ left: Math.min(x(hover) + 12, width - 210), top: 8 }}>
          <b>{MONTHS[hv.month]} {hv.year}</b><br />{fmt(hv.value, variable)}
          {hv.modisEra && range?.[hv.month] != null && <><br />95% range {fmt(lo(hv), variable)} to {fmt(hi(hv), variable)}</>}
          {hv.nodata > 0 && <><br />{hv.nodata} day(s) without data</>}
        </div>
      )}
      <figcaption className="legend">
        <span className="legend-line solid" /> monthly total
        <span className="swatch" style={{ background: theme.band }} /> 95% range of the scaling (MODIS years)
      </figcaption>
    </figure>
  );
}

function compact(v) {
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(v >= 1e4 ? 0 : 1)}k`;
  return v.toFixed(0);
}
