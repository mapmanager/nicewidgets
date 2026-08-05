/** Color lookup tables used by scalar and composite raster rendering. */

export const LUT_LABELS = Object.freeze({
  gray: 'Gray', red: 'Red', green: 'Green', cyan: 'Cyan', magenta: 'Magenta',
  yellow: 'Yellow', fire: 'Fire', viridis: 'Viridis', magma: 'Magma',
});

const STOPS = {
  gray: [[0,[0,0,0]],[1,[255,255,255]]],
  red: [[0,[0,0,0]],[1,[255,32,24]]], green: [[0,[0,0,0]],[1,[20,255,80]]],
  cyan: [[0,[0,0,0]],[1,[20,235,255]]], magenta: [[0,[0,0,0]],[1,[255,20,235]]],
  yellow: [[0,[0,0,0]],[1,[255,235,20]]],
  fire: [[0,[0,0,0]],[0.35,[220,0,0]],[0.7,[255,210,0]],[1,[255,255,255]]],
  viridis: [[0,[68,1,84]],[0.33,[49,104,142]],[0.66,[53,183,121]],[1,[253,231,37]]],
  magma: [[0,[0,0,4]],[0.33,[110,30,130]],[0.66,[235,90,95]],[1,[252,253,191]]],
};

function interpolate(t, stops) {
  const value = Math.max(0, Math.min(1, t));
  for (let index = 0; index < stops.length - 1; index += 1) {
    const left = stops[index]; const right = stops[index + 1];
    if (value <= right[0]) {
      const f = (value - left[0]) / Math.max(1e-12, right[0] - left[0]);
      return left[1].map((part, channel) => Math.round(part + (right[1][channel] - part) * f));
    }
  }
  return stops.at(-1)[1];
}

const TABLES = Object.fromEntries(Object.entries(STOPS).map(([name, stops]) => {
  const table = new Uint8ClampedArray(768);
  for (let index = 0; index < 256; index += 1) table.set(interpolate(index / 255, stops), index * 3);
  return [name, table];
}));

/** Return the packed 256-entry RGB table for one LUT. */
export function lutTable(name) {
  return TABLES[name] || TABLES.gray;
}

/** Return an RGB triplet for a normalized scalar value. */
export function sampleLut(name, value) {
  const table = lutTable(name);
  const offset = Math.round(Math.max(0, Math.min(1, value)) * 255) * 3;
  return [table[offset], table[offset + 1], table[offset + 2]];
}
