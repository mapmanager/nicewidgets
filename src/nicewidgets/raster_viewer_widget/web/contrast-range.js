/** Display-range popover with optional log-scaled histogram bars. */

import {sampleLut} from './lut.js';

const HISTOGRAM_BINS = 96;
const MAX_SAMPLES = 120000;
const HISTOGRAM_LEFT = 6;
const HISTOGRAM_RIGHT = 294;

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function sampledFiniteValues(values, maxSamples = MAX_SAMPLES) {
  const samples = [];
  const step = Math.max(1, Math.floor(values.length / maxSamples));
  for (let index = 0; index < values.length; index += step) {
    if (Number.isFinite(values[index])) samples.push(values[index]);
  }
  return samples;
}

function percentile(sortedValues, percent) {
  if (sortedValues.length === 0) return Number.NaN;
  const index = clamp(
    Math.round(percent / 100 * (sortedValues.length - 1)),
    0,
    sortedValues.length - 1,
  );
  return sortedValues[index];
}

export function autoRange(values) {
  const sorted = sampledFiniteValues(values).sort((left, right) => left - right);
  let minimum = percentile(sorted, 1);
  let maximum = percentile(sorted, 99.5);
  if (!(maximum > minimum)) {
    const center = Number.isFinite(minimum) ? minimum : 0;
    const delta = Math.max(1e-9, Math.abs(center) * 0.01 || 1);
    minimum = center - delta;
    maximum = center + delta;
  }
  return [minimum, maximum];
}

export function histogramForValues(values, binCount = HISTOGRAM_BINS) {
  const sorted = sampledFiniteValues(values).sort((left, right) => left - right);
  if (sorted.length === 0) return null;
  let domainMin = percentile(sorted, 0.1);
  let domainMax = percentile(sorted, 99.9);
  if (!(domainMax > domainMin)) {
    const delta = Math.max(1e-9, Math.abs(domainMin) * 0.01 || 1);
    domainMin -= delta;
    domainMax += delta;
  }
  const bins = new Uint32Array(binCount);
  const span = domainMax - domainMin;
  for (const value of sorted) {
    const index = clamp(Math.floor((value - domainMin) / span * binCount), 0, binCount - 1);
    bins[index] += 1;
  }
  return {bins, domainMin, domainMax};
}

export function histogramBarColor(lut) {
  const [red, green, blue] = sampleLut(lut, 0.8);
  return `rgb(${red} ${green} ${blue})`;
}

/**
 * Normalize a bin count to a bar height fraction.
 *
 * @param {number} count Bin count (non-negative).
 * @param {number} maxCount Maximum bin count in the histogram.
 * @param {boolean} [logScale=true] When true, use log1p scaling; otherwise linear.
 * @returns {number} Fraction in ``[0, 1]``.
 */
export function histogramBarFraction(count, maxCount, logScale = true) {
  const peak = Math.max(1, Number(maxCount) || 0);
  const value = Math.max(0, Number(count) || 0);
  if (logScale) return Math.log1p(value) / Math.log1p(peak);
  return value / peak;
}

function numberText(value) {
  const magnitude = Math.abs(value);
  if ((magnitude !== 0 && magnitude < 0.001) || magnitude >= 100000) {
    return value.toExponential(5);
  }
  return String(Number(value.toPrecision(8)));
}

export class ContrastRangePopover {
  constructor(root, onRangeChange) {
    this.root = root;
    this.onRangeChange = onRangeChange;
    this.channel = null;
    this.activeButton = null;
    this.histogram = null;
    this.dragHandle = null;
    this.logScale = true;
    this.theme = {
      canvasBackground: '#020617',
      histogramBar: '#64748b',
      histogramMinimum: '#020617',
      histogramMaximum: '#ffffff',
      histogramHandleLine: '#e2e8f0',
    };
    this.build();
    this.bind();
  }

  build() {
    this.element = document.createElement('div');
    this.element.className = 'rv-range-popover';
    this.element.hidden = true;
    this.element.setAttribute('role', 'dialog');
    this.element.setAttribute('aria-label', 'Display range');
    this.canvas = document.createElement('canvas');
    this.canvas.width = 300;
    this.canvas.height = 100;
    this.canvas.className = 'rv-range-histogram';
    const histogramWrap = document.createElement('div');
    histogramWrap.className = 'rv-histogram-wrap';
    histogramWrap.append(this.canvas);
    const fields = document.createElement('div');
    fields.className = 'rv-range-fields';
    this.minimumInput = this.numberInput('Min');
    this.maximumInput = this.numberInput('Max');
    this.autoButton = document.createElement('button');
    this.autoButton.type = 'button';
    this.autoButton.textContent = 'Auto';
    this.logLabel = document.createElement('label');
    this.logLabel.className = 'rv-range-log';
    this.logCheckbox = document.createElement('input');
    this.logCheckbox.type = 'checkbox';
    this.logCheckbox.checked = this.logScale;
    this.logCheckbox.setAttribute('aria-label', 'Log histogram Y scale');
    this.logLabel.append(this.logCheckbox, document.createTextNode('Log'));
    fields.append(
      this.fieldLabel('Min', this.minimumInput),
      this.fieldLabel('Max', this.maximumInput),
      this.autoButton,
      this.logLabel,
    );
    this.element.append(histogramWrap, fields);
    this.mountPopover();
  }

  mountPopover() {
    // Prefer document.body so fixed positioning is not trapped in a viewer stacking context.
    const mount = typeof document !== 'undefined' && document.body
      ? document.body
      : this.root;
    if (mount && this.element.parentElement !== mount) mount.append(this.element);
  }

  numberInput(label) {
    const input = document.createElement('input');
    input.type = 'number';
    input.step = 'any';
    input.setAttribute('aria-label', `${label} display value`);
    return input;
  }

  fieldLabel(text, input) {
    const label = document.createElement('label');
    label.append(text, input);
    return label;
  }

  bind() {
    this.minimumInput.addEventListener('input', () => this.applyNumericRange());
    this.maximumInput.addEventListener('input', () => this.applyNumericRange());
    this.autoButton.addEventListener('click', () => {
      if (!this.channel) return;
      [this.channel.min, this.channel.max] = autoRange(this.channel.data);
      this.syncFields();
      this.onRangeChange();
      this.drawHistogram();
    });
    this.logCheckbox.addEventListener('change', () => {
      this.logScale = Boolean(this.logCheckbox.checked);
      this.drawHistogram();
    });
    this.canvas.addEventListener('pointerdown', event => this.beginDrag(event));
    this.canvas.addEventListener('pointermove', event => this.updateDrag(event));
    this.canvas.addEventListener('pointerup', event => this.endDrag(event));
    this.documentPointerHandler = event => {
      if (this.element.hidden || this.element.contains(event.target)) return;
      if (event.target.closest?.('.rv-range-button')) return;
      this.close();
    };
    this.keyHandler = event => {
      if (event.key === 'Escape') this.close();
    };
    this.resizeHandler = () => this.close();
    document.addEventListener('pointerdown', this.documentPointerHandler);
    document.addEventListener('keydown', this.keyHandler);
    window.addEventListener('resize', this.resizeHandler);
  }

  syncThemeFromRoot() {
    const theme = this.root?.dataset?.theme === 'light' ? 'light' : 'dark';
    this.element.dataset.theme = theme;
  }

  toggle(channel, button) {
    if (!this.element.hidden && this.channel?.id === channel.id) {
      this.close();
      return;
    }
    this.channel = channel;
    this.activeButton = button;
    this.histogram = channel.histogram || histogramForValues(channel.data);
    channel.histogram = this.histogram;
    this.syncThemeFromRoot();
    this.mountPopover();
    this.syncFields();
    this.logCheckbox.checked = this.logScale;
    this.element.hidden = false;
    this.position();
    this.drawHistogram();
  }

  close() {
    this.element.hidden = true;
    this.dragHandle = null;
    this.activeButton = null;
  }

  setTheme(theme) {
    this.theme = theme;
    this.syncThemeFromRoot();
    this.drawHistogram();
  }

  position() {
    if (!this.activeButton) return;
    const rect = this.activeButton.getBoundingClientRect();
    const width = this.element.offsetWidth || 330;
    const height = this.element.offsetHeight || 190;
    const margin = 8;
    const left = clamp(rect.left, margin, Math.max(margin, window.innerWidth - width - margin));
    const below = rect.bottom + 6;
    const top = below + height <= window.innerHeight - margin
      ? below
      : Math.max(margin, rect.top - height - 6);
    this.element.style.left = `${Math.round(left)}px`;
    this.element.style.top = `${Math.round(top)}px`;
  }

  syncFields() {
    if (!this.channel) return;
    this.minimumInput.value = numberText(this.channel.min);
    this.maximumInput.value = numberText(this.channel.max);
  }

  applyNumericRange() {
    if (!this.channel) return;
    const minimum = Number(this.minimumInput.value);
    const maximum = Number(this.maximumInput.value);
    if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || !(maximum > minimum)) return;
    this.channel.min = minimum;
    this.channel.max = maximum;
    this.onRangeChange();
    this.drawHistogram();
  }

  beginDrag(event) {
    if (!this.channel || !this.histogram) return;
    const pixelX = this.eventPixelX(event);
    const minimumX = this.valueToX(this.channel.min);
    const maximumX = this.valueToX(this.channel.max);
    this.dragHandle = Math.abs(pixelX - minimumX) <= Math.abs(pixelX - maximumX)
      ? 'minimum'
      : 'maximum';
    this.canvas.setPointerCapture?.(event.pointerId);
    this.updateDrag(event);
  }

  updateDrag(event) {
    if (!this.channel || !this.histogram || !this.dragHandle) return;
    const fraction = clamp(
      (this.eventPixelX(event) - HISTOGRAM_LEFT) / (HISTOGRAM_RIGHT - HISTOGRAM_LEFT),
      0,
      1,
    );
    const value = this.histogram.domainMin
      + fraction * (this.histogram.domainMax - this.histogram.domainMin);
    const epsilon = Math.max(
      1e-12,
      (this.histogram.domainMax - this.histogram.domainMin) * 1e-6,
    );
    if (this.dragHandle === 'minimum') {
      this.channel.min = Math.min(value, this.channel.max - epsilon);
    } else {
      this.channel.max = Math.max(value, this.channel.min + epsilon);
    }
    this.syncFields();
    this.onRangeChange();
    this.drawHistogram();
    event.preventDefault();
  }

  endDrag(event) {
    if (!this.dragHandle) return;
    this.canvas.releasePointerCapture?.(event.pointerId);
    this.dragHandle = null;
  }

  eventPixelX(event) {
    const rect = this.canvas.getBoundingClientRect();
    return (event.clientX - rect.left) * this.canvas.width / Math.max(1, rect.width);
  }

  valueToX(value) {
    if (!this.histogram) return HISTOGRAM_LEFT;
    const fraction = clamp(
      (value - this.histogram.domainMin)
        / (this.histogram.domainMax - this.histogram.domainMin),
      0,
      1,
    );
    return HISTOGRAM_LEFT + fraction * (HISTOGRAM_RIGHT - HISTOGRAM_LEFT);
  }

  drawHistogram() {
    if (!this.channel || !this.histogram) return;
    const context = this.canvas.getContext('2d');
    const top = 6;
    const bottom = this.canvas.height - 8;
    const maxCount = Math.max(1, ...this.histogram.bins);
    context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    context.fillStyle = this.theme.canvasBackground;
    context.fillRect(0, 0, this.canvas.width, this.canvas.height);
    context.fillStyle = histogramBarColor(this.channel.lut);
    this.histogram.bins.forEach((count, index) => {
      const x0 = HISTOGRAM_LEFT
        + index / this.histogram.bins.length * (HISTOGRAM_RIGHT - HISTOGRAM_LEFT);
      const x1 = HISTOGRAM_LEFT
        + (index + 1) / this.histogram.bins.length * (HISTOGRAM_RIGHT - HISTOGRAM_LEFT);
      const fraction = histogramBarFraction(count, maxCount, this.logScale);
      const height = fraction * (bottom - top);
      context.fillRect(x0, bottom - height, Math.max(1, x1 - x0), height);
    });
    this.drawHandle(this.channel.min, this.theme.histogramMinimum, top, bottom);
    this.drawHandle(this.channel.max, this.theme.histogramMaximum, top, bottom);
  }

  drawHandle(value, fill, top, bottom) {
    const context = this.canvas.getContext('2d');
    const x = this.valueToX(value);
    context.strokeStyle = fill;
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(x, top);
    context.lineTo(x, bottom);
    context.stroke();
    context.fillStyle = fill;
    context.strokeStyle = this.theme.histogramHandleLine;
    context.fillRect(x - 4, top, 8, 9);
    context.strokeRect(x - 4, top, 8, 9);
  }

  destroy() {
    document.removeEventListener('pointerdown', this.documentPointerHandler);
    document.removeEventListener('keydown', this.keyHandler);
    window.removeEventListener('resize', this.resizeHandler);
    this.element.remove();
  }
}
