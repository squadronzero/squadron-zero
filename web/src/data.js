// Loading and decoding the files written by the pipeline's export stage (pipeline/shift/export.py).
// Every number shown on the site comes from these files; nothing is estimated in the browser
// except sums over the area the user picks and the climatology of that area.

const BASE = 'data/';
const cache = new Map();

async function bytes(path) {
  if (cache.has(path)) return cache.get(path);
  const p = (async () => {
    const res = await fetch(BASE + path);
    if (!res.ok) throw new Error(`${path}: ${res.status}`);
    const buf = new Uint8Array(await res.arrayBuffer());
    // GitHub Pages serves .gz files as they are; some local servers unpack them on the way.
    if (buf[0] === 0x1f && buf[1] === 0x8b) {
      const ds = new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip')));
      return new Uint8Array(await ds.arrayBuffer());
    }
    return buf;
  })();
  cache.set(path, p);
  try { return await p; } catch (e) { cache.delete(path); throw e; }
}

export async function json(path) {
  const res = await fetch(BASE + path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

export async function csv(path) {
  const res = await fetch(BASE + path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  const lines = (await res.text()).trim().split(/\r?\n/);
  const head = lines[0].split(',');
  return lines.slice(1).map((l) => {
    const v = l.split(',');
    const o = {};
    head.forEach((h, i) => { const x = v[i]; o[h] = x === '' || x === undefined ? null : (isNaN(+x) ? x : +x); });
    return o;
  });
}

// ---------------------------------------------------------------- names, as the pipeline writes them
export function tileName(lat0, lon0) {
  const ns = lat0 >= 0 ? 'N' : 'S';
  const ew = lon0 >= 0 ? 'E' : 'W';
  return `${ns}${String(Math.abs(Math.trunc(lat0))).padStart(2, '0')}${ew}${String(Math.abs(Math.trunc(lon0))).padStart(3, '0')}`;
}

// ---------------------------------------------------------------- binary layouts
function daily(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const n = dv.getUint32(0, true);
  let o = 4;
  const cells = new Float32Array(buf.slice(o, o + 4 * n).buffer); o += 4 * n;
  const frp = new Float32Array(buf.slice(o, o + 4 * n).buffer); o += 4 * n;
  const day = new Uint16Array(buf.slice(o, o + 2 * n).buffer); o += 2 * n;
  const cell = buf.subarray(o, o + n); o += n;
  const src = buf.subarray(o, o + n);
  return { n, cells, frp, day, cell, src };
}

export async function monthLayer(ym) {
  const buf = await bytes(`monthly/${ym}.bin.gz`);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const n = dv.getUint32(0, true);
  const ids = new Uint32Array(buf.slice(4, 4 + 4 * n).buffer);
  const cells = new Float32Array(buf.slice(4 + 4 * n, 4 + 8 * n).buffer);
  const frp = new Float32Array(buf.slice(4 + 8 * n, 4 + 12 * n).buffer);
  const features = new Array(n);
  for (let i = 0; i < n; i++) {
    const r = Math.floor(ids[i] / 1440), c = ids[i] % 1440;
    features[i] = { type: 'Feature', geometry: { type: 'Point', coordinates: [-180 + (c + 0.5) * 0.25, -90 + (r + 0.5) * 0.25] },
      properties: { cells: Math.round(cells[i] * 10) / 10, frp: Math.round(frp[i]) } };
  }
  return { type: 'FeatureCollection', features };
}

let zonesP = null;
export function zones(days) {
  if (!zonesP) zonesP = bytes('zones.bin.gz').then((b) => ({ b, days }));
  return zonesP;
}

// ---------------------------------------------------------------- the area
export const MAX_SPAN = 30;   // degrees per side: keeps a query to at most 16 daily files

export function tilesFor(box) {
  const out = [];
  for (let la = Math.floor(box.s / 10) * 10; la < box.n; la += 10)
    for (let lo = Math.floor(box.w / 10) * 10; lo < box.e; lo += 10) out.push({ name: tileName(la, lo), lat0: la, lon0: lo });
  return out;
}

export function zonesFor(box) {
  const out = [];
  for (let la = Math.floor((box.s + 90) / 30); la * 30 - 90 < box.n; la++)
    for (let lo = Math.floor((box.w + 180) / 30); lo * 30 - 180 < box.e; lo++)
      if (la >= 0 && la < 6 && lo >= 0 && lo < 12) out.push(la * 12 + lo);
  return out;
}

// Daily and monthly sums for an area, with where every value came from and whether each day had data.
export async function areaSeries(meta, box) {
  const N = meta.days;
  const cells = new Float64Array(N), frp = new Float64Array(N);
  const filled = new Uint8Array(N);                 // 1 = some of the value was filled from a partner satellite
  const fineW = new Map();                          // 2.5-degree factor cell -> fire before 2012 (for the uncertainty weight)
  const switchDay = dayIndex(meta, '2012-01-20');
  const wanted = tilesFor(box).filter((t) => meta.tiles[t.name]);
  await Promise.all(wanted.map(async (t) => {
    const d = daily(await bytes(`daily/${t.name}.bin.gz`));
    for (let i = 0; i < d.n; i++) {
      const row = Math.floor(d.cell[i] / 10), col = d.cell[i] % 10;
      const lat = t.lat0 + row + 0.5, lon = t.lon0 + col + 0.5;
      if (lat < box.s || lat > box.n || lon < box.w || lon > box.e) continue;
      const k = d.day[i];
      cells[k] += d.cells[i]; frp[k] += d.frp[i];
      const s = d.src[i];
      if (s === 1 || s === 2 || s === 4 || s === 5) filled[k] = 1;
      if (k < switchDay) {
        const key = tileName(Math.floor(lat / 2.5) * 2.5, Math.floor(lon / 2.5) * 2.5);
        fineW.set(key, (fineW.get(key) || 0) + d.cells[i]);
      }
    }
  }));
  // availability: a day is "no data" when any 30-degree zone the area touches had no satellite at all
  const z = await zones(N);
  const nodata = new Uint8Array(N);
  for (const zi of zonesFor(box)) {
    const row = z.b.subarray(zi * N, zi * N + N);
    for (let k = 0; k < N; k++) if (row[k] === 5) nodata[k] = 1;
  }
  return { cells, frp, filled, nodata, fineW, tiles: wanted.length };
}

export function dayIndex(meta, iso) {
  return Math.round((Date.parse(iso + 'T00:00:00Z') - Date.parse(meta.start + 'T00:00:00Z')) / 86400000);
}

export function dateOf(meta, k) {
  return new Date(Date.parse(meta.start + 'T00:00:00Z') + k * 86400000);
}

// Monthly totals, completeness and anomaly classes (the same rule as the pipeline:
// z of log(1 + value) against the same calendar month of the baseline years; below -1 low,
// -1..1 normal, 1..2 unusual, 2 and above critical).
export function monthly(meta, s, variable) {
  const v = s[variable];
  const map = new Map();
  for (let k = 0; k < meta.days; k++) {
    const d = dateOf(meta, k);
    const y = d.getUTCFullYear(), m = d.getUTCMonth();
    const key = y * 12 + m;
    let o = map.get(key);
    if (!o) { o = { year: y, month: m, value: 0, days: 0, nodata: 0, filled: 0 }; map.set(key, o); }
    o.value += v[k]; o.days += 1; o.nodata += s.nodata[k]; o.filled += s.filled[k];
  }
  const rows = [...map.values()].sort((a, b) => a.year - b.year || a.month - b.month);
  const comp = (meta.comparable_start || '2003-01-01').slice(0, 4) | 0;
  const last = rows[rows.length - 1];
  const full = (r) => r.days === new Date(Date.UTC(r.year, r.month + 1, 0)).getUTCDate();
  const lastFullYear = last.month === 11 && full(last) ? last.year : last.year - 1;
  const stats = Array.from({ length: 12 }, () => []);
  for (const r of rows) {
    r.notComparable = r.year < comp;
    r.incomplete = !full(r) || r.nodata > 0;
    r.modisEra = r.year < 2012;
    if (!r.notComparable && !r.incomplete && r.year <= lastFullYear) stats[r.month].push(Math.log1p(r.value));
  }
  const mu = stats.map((a) => a.reduce((x, y) => x + y, 0) / Math.max(a.length, 1));
  const sd = stats.map((a, i) => Math.max(0.05, Math.sqrt(a.reduce((x, y) => x + (y - mu[i]) ** 2, 0) / Math.max(a.length - 1, 1))));
  for (const r of rows) {
    if (r.notComparable || r.incomplete || stats[r.month].length < 5) { r.cls = null; continue; }
    r.z = (Math.log1p(r.value) - mu[r.month]) / sd[r.month];
    r.cls = r.z < -1 ? 'low' : r.z < 1 ? 'normal' : r.z < 2 ? 'unusual' : 'critical';
  }
  return { rows, baseline: [comp, lastFullYear] };
}

// 95% range (%) of the MODIS-era factor for the area and calendar month, weighted by fire.
export function areaRange(unc, fineW, variable) {
  const t = unc?.range_pct?.[variable];
  if (!t) return null;
  const out = new Array(12).fill(null);
  for (let m = 0; m < 12; m++) {
    let w = 0, s = 0;
    for (const [key, wt] of fineW) {
      const r = t[key]?.[m];
      if (r != null && isFinite(r)) { s += r * wt; w += wt; }
    }
    out[m] = w > 0 ? s / w : null;
  }
  return out;
}

export function toCSV(meta, s) {
  const lines = ['date,burning_cells_snpp_equivalent,fire_power_mw_snpp_equivalent,filled_from_partner,no_data'];
  for (let k = 0; k < meta.days; k++) {
    const d = dateOf(meta, k).toISOString().slice(0, 10);
    lines.push(`${d},${s.nodata[k] ? '' : s.cells[k].toFixed(2)},${s.nodata[k] ? '' : s.frp[k].toFixed(1)},${s.filled[k]},${s.nodata[k]}`);
  }
  return lines.join('\n');
}
