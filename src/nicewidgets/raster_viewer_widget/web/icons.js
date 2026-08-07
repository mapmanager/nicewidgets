/** Small, offline Lucide icon subset used by the reusable viewer.
 *
 * Icon designs are from Lucide (https://lucide.dev).
 * Copyright (c) 2020, Lucide Contributors; licensed under ISC.
 */

const SVG_NAMESPACE = 'http://www.w3.org/2000/svg';

const ICONS = Object.freeze({
  'columns-2': [['rect', {x: 3, y: 3, width: 18, height: 18, rx: 2}], ['path', {d: 'M12 3v18'}]],
  'rows-2': [['rect', {x: 3, y: 3, width: 18, height: 18, rx: 2}], ['path', {d: 'M3 12h18'}]],
  square: [['rect', {x: 3, y: 3, width: 18, height: 18, rx: 2}]],
  'layers-3': [
    ['path', {d: 'm12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z'}],
    ['path', {d: 'm22 12.5-9.17 4.17a2 2 0 0 1-1.66 0L2 12.5'}],
    ['path', {d: 'm22 17.5-9.17 4.17a2 2 0 0 1-1.66 0L2 17.5'}],
  ],
  menu: [['path', {d: 'M4 12h16'}], ['path', {d: 'M4 6h16'}], ['path', {d: 'M4 18h16'}]],
  'maximize-2': [
    ['path', {d: 'M15 3h6v6'}],
    ['path', {d: 'm21 3-7 7'}],
    ['path', {d: 'm3 21 7-7'}],
    ['path', {d: 'M9 21H3v-6'}],
  ],
  copy: [
    ['rect', {x: 8, y: 8, width: 13, height: 13, rx: 2}],
    ['path', {d: 'M16 8V6a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h2'}],
  ],
  check: [['path', {d: 'm20 6-11 11-5-5'}]],
  plus: [['path', {d: 'M5 12h14'}], ['path', {d: 'M12 5v14'}]],
  x: [['path', {d: 'M18 6 6 18'}], ['path', {d: 'm6 6 12 12'}]],
  pencil: [
    ['path', {d: 'M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 .83-.497z'}],
    ['path', {d: 'm15 5 4 4'}],
  ],
  'trash-2': [
    ['path', {d: 'M3 6h18'}],
    ['path', {d: 'M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6'}],
    ['path', {d: 'M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2'}],
    ['line', {x1: 10, x2: 10, y1: 11, y2: 17}],
    ['line', {x1: 14, x2: 14, y1: 11, y2: 17}],
  ],
  'chart-column-decreasing': [
    ['path', {d: 'M13 17V9'}],
    ['path', {d: 'M18 17v-3'}],
    ['path', {d: 'M3 3v16a2 2 0 0 0 2 2h16'}],
    ['path', {d: 'M8 17V5'}],
  ],
});

export function lucideIcon(name, label) {
  const definition = ICONS[name];
  if (!definition) {
    console.warn(`unknown Lucide icon: ${name}`);
    const fallback = document.createElement('span');
    fallback.className = 'rv-icon-fallback';
    fallback.textContent = (label || name || '?').slice(0, 2);
    fallback.setAttribute('aria-hidden', 'true');
    if (label) fallback.dataset.label = label;
    return fallback;
  }
  const svg = document.createElementNS(SVG_NAMESPACE, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('width', '18');
  svg.setAttribute('height', '18');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '2');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  svg.setAttribute('aria-hidden', 'true');
  for (const [tag, attributes] of definition) {
    const node = document.createElementNS(SVG_NAMESPACE, tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
    svg.append(node);
  }
  if (label) svg.dataset.label = label;
  return svg;
}
