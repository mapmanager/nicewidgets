/** Interactive raster viewport with fixed plot gutters and axis chrome. */

import {displayToSource, sourceToDisplay} from './orientation.js';

const MIN_HOME_ZOOM = 0.1;
const MAX_HOME_ZOOM = 1000;
const MIN_REGION_PIXELS = 12;
const AXIS_LOCK_PIXELS = 8;
const TICK_COUNT = 5;

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function formatTick(value) {
  const magnitude = Math.abs(value);
  if (magnitude > 0 && (magnitude < 0.001 || magnitude >= 10000)) {
    return value.toExponential(2);
  }
  return String(Number(value.toPrecision(5)));
}

export function dragZoomMode(bitmap, start, current) {
  const deltaX = current.x - start.x;
  const deltaY = current.y - start.y;
  if (Math.hypot(deltaX, deltaY) < AXIS_LOCK_PIXELS) return 'pending';
  if (bitmap.width === bitmap.height) return 'region';
  return Math.abs(deltaX) >= Math.abs(deltaY) ? 'x' : 'y';
}

/**
 * Validate the multiplicative zoom applied for each wheel event.
 *
 * Values closer to 1 zoom more slowly (for example, 1.03). Valid values are
 * greater than 1 and at most 2; the default is 1.06.
 *
 * @param {number} [value=1.06] Requested per-event zoom multiplier.
 * @returns {number} Validated zoom multiplier.
 * @throws {Error} If the value is outside the supported range.
 */
export function normalizeWheelZoomFactor(value = 1.06) {
  const factor = Number(value);
  if (!Number.isFinite(factor) || factor <= 1 || factor > 2) {
    throw new Error('wheelZoomFactor must be greater than 1 and at most 2');
  }
  return factor;
}

export class RasterViewport {
  static margins = {left: 58, right: 12, top: 10, bottom: 38};

  constructor(
    canvas,
    onChange,
    interactionCanvas = null,
    onSliceStep = null,
    wheelZoomFactor = 1.06,
  ) {
    this.canvas = canvas;
    this.interactionCanvas = interactionCanvas || canvas;
    this.wrap = canvas.parentElement;
    this.onChange = onChange;
    this.onSliceStep = onSliceStep;
    this.wheelZoomFactor = normalizeWheelZoomFactor(wheelZoomFactor);
    this.ctx = canvas.getContext('2d');
    this.bitmap = null;
    this.axes = null;
    this.scaleX = 1;
    this.scaleY = 1;
    this.offsetX = 0;
    this.offsetY = 0;
    this.home = null;
    this.drag = null;
    this.wheelEnd = null;
    this.plotOverlay = null;
    this.overlay = null;
    this.theme = {
      canvasBackground: '#020617',
      axisLine: '#64748b',
      axisText: '#cbd5e1',
    };
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.wrap);
    this.bind();
  }

  bind() {
    this.interactionCanvas.addEventListener('wheel', event => {
      const point = this.point(event);
      if (!this.containsPlotPoint(point)) return;
      event.preventDefault();
      if (event.altKey && this.onSliceStep) {
        this.onSliceStep(event.deltaY < 0 ? 1 : -1);
        return;
      }
      const factor = event.deltaY < 0 ? this.wheelZoomFactor : 1 / this.wheelZoomFactor;
      const minimumX = (this.home?.scaleX || this.scaleX) * MIN_HOME_ZOOM;
      const maximumX = (this.home?.scaleX || this.scaleX) * MAX_HOME_ZOOM;
      const minimumY = (this.home?.scaleY || this.scaleY) * MIN_HOME_ZOOM;
      const maximumY = (this.home?.scaleY || this.scaleY) * MAX_HOME_ZOOM;
      const nextX = clamp(this.scaleX * factor, minimumX, maximumX);
      const nextY = clamp(this.scaleY * factor, minimumY, maximumY);
      this.offsetX = point.x - (point.x - this.offsetX) * nextX / this.scaleX;
      this.offsetY = point.y - (point.y - this.offsetY) * nextY / this.scaleY;
      this.scaleX = nextX;
      this.scaleY = nextY;
      this.draw();
      this.emit('wheel', false);
      clearTimeout(this.wheelEnd);
      this.wheelEnd = setTimeout(() => this.emit('wheel', true), 100);
    }, {passive: false});

    this.interactionCanvas.addEventListener('dblclick', event => {
      const point = this.point(event);
      if (!this.containsPlotPoint(point)) return;
      if (this.overlay?.suppressDoubleClick(point)) return;
      event.preventDefault();
      this.reset();
      this.emit('reset', true);
    });

    this.interactionCanvas.addEventListener('pointerdown', event => {
      const point = this.point(event);
      if (event.button !== 0 || !this.containsPlotPoint(point)) return;
      if (this.overlay?.pointerDown(event, point)) {
        this.interactionCanvas.setPointerCapture(event.pointerId);
        return;
      }
      this.interactionCanvas.setPointerCapture(event.pointerId);
      this.drag = {
        start: point,
        last: point,
        pan: event.shiftKey,
        zoomMode: event.shiftKey ? null : 'pending',
      };
      this.wrap.classList.toggle('is-panning', event.shiftKey);
    });

    this.interactionCanvas.addEventListener('pointermove', event => {
      const point = this.clampToPlot(this.point(event));
      if (this.overlay?.pointerMove(event, point)) return;
      if (!this.drag) return;
      if (this.drag.pan) {
        this.offsetX += point.x - this.drag.last.x;
        this.offsetY += point.y - this.drag.last.y;
      } else if (this.drag.zoomMode === 'pending') {
        const nextMode = dragZoomMode(this.bitmap, this.drag.start, point);
        if (nextMode !== 'pending') {
          this.drag.zoomMode = nextMode;
          this.updateDragClass();
        }
      }
      this.drag.last = point;
      this.draw();
      this.emit(this.drag.pan ? 'pan' : 'region', false);
    });

    this.interactionCanvas.addEventListener('pointerup', event => {
      const point = this.clampToPlot(this.point(event));
      // IDLE ROI click-select claims the gesture; skip zoom completion then.
      const overlayClaimed = Boolean(this.overlay?.pointerUp(event, point));
      if (!this.drag) {
        try {
          this.interactionCanvas.releasePointerCapture(event.pointerId);
        } catch (_) {
          // Pointer capture may already have been released by the browser.
        }
        return;
      }
      const drag = this.drag;
      this.drag = null;
      this.clearDragClasses();
      if (!overlayClaimed) {
        if (!drag.pan && drag.zoomMode === 'region') this.zoomRegion(drag.start, drag.last);
        if (!drag.pan && drag.zoomMode === 'x') this.zoomAxis('x', drag.start, drag.last);
        if (!drag.pan && drag.zoomMode === 'y') this.zoomAxis('y', drag.start, drag.last);
        if (drag.pan || drag.zoomMode !== 'pending') {
          this.draw();
          const cause = drag.pan ? 'pan' : drag.zoomMode === 'x'
            ? 'region-x' : drag.zoomMode === 'y' ? 'region-y' : 'region';
          this.emit(cause, true);
        }
      }
      try {
        this.interactionCanvas.releasePointerCapture(event.pointerId);
      } catch (_) {
        // Pointer capture may already have been released by the browser.
      }
    });

    this.interactionCanvas.addEventListener('pointercancel', () => {
      this.drag = null;
      this.overlay?.pointerCancel();
      this.clearDragClasses();
    });
  }

  point(event) {
    const rect = this.interactionCanvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * this.canvas.width / Math.max(1, rect.width),
      y: (event.clientY - rect.top) * this.canvas.height / Math.max(1, rect.height),
    };
  }

  plot() {
    const margins = RasterViewport.margins;
    return {
      left: margins.left,
      top: margins.top,
      width: Math.max(1, this.canvas.width - margins.left - margins.right),
      height: Math.max(1, this.canvas.height - margins.top - margins.bottom),
    };
  }

  containsPlotPoint(point) {
    const plot = this.plot();
    return point.x >= plot.left && point.x <= plot.left + plot.width
      && point.y >= plot.top && point.y <= plot.top + plot.height;
  }

  clampToPlot(point) {
    const plot = this.plot();
    return {
      x: clamp(point.x, plot.left, plot.left + plot.width),
      y: clamp(point.y, plot.top, plot.top + plot.height),
    };
  }

  setImage(bitmap, axes, reset = true) {
    this.bitmap = bitmap;
    this.axes = axes;
    if (reset) this.fit();
    this.draw();
  }

  setAxes(axes) {
    this.axes = axes;
    this.draw();
  }

  setTheme(theme, redraw = true) {
    this.theme = theme;
    if (redraw) this.draw();
  }

  setPhysicalXRange(minimum, maximum, step) {
    if (!this.bitmap) throw new Error('Raster image is unavailable');
    if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum >= maximum) {
      throw new Error('X axis range requires finite minimum < maximum');
    }
    if (!Number.isFinite(step) || step <= 0) throw new Error('X axis step must be positive');
    const fullMaximum = this.bitmap.width * step;
    const clampedMinimum = clamp(minimum, 0, fullMaximum);
    const clampedMaximum = clamp(maximum, 0, fullMaximum);
    if (clampedMinimum >= clampedMaximum) {
      throw new Error('X axis range is outside the displayed dataset extent');
    }
    const x0 = clampedMinimum / step;
    const x1 = clampedMaximum / step;
    const plot = this.plot();
    this.scaleX = plot.width / (x1 - x0);
    this.offsetX = plot.left - x0 * this.scaleX;
    this.draw();
    this.emit('api-x-range', true);
    return {minimum: clampedMinimum, maximum: clampedMaximum};
  }

  setPhysicalYRange(minimum, maximum, step) {
    if (!this.bitmap) throw new Error('Raster image is unavailable');
    if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum >= maximum) {
      throw new Error('Y axis range requires finite minimum < maximum');
    }
    if (!Number.isFinite(step) || step <= 0) throw new Error('Y axis step must be positive');
    const fullMaximum = this.bitmap.height * step;
    const clampedMinimum = clamp(minimum, 0, fullMaximum);
    const clampedMaximum = clamp(maximum, 0, fullMaximum);
    if (clampedMinimum >= clampedMaximum) {
      throw new Error('Y axis range is outside the displayed dataset extent');
    }
    const y0 = clampedMinimum / step;
    const y1 = clampedMaximum / step;
    const plot = this.plot();
    this.scaleY = plot.height / (y1 - y0);
    this.offsetY = plot.top - (this.bitmap.height - y1) * this.scaleY;
    this.draw();
    this.emit('api-y-range', true);
    return {minimum: clampedMinimum, maximum: clampedMaximum};
  }

  setPhysicalRange(xMinimum, xMaximum, xStep, yMinimum, yMaximum, yStep) {
    if (!this.bitmap) throw new Error('Raster image is unavailable');
    if (!Number.isFinite(xMinimum) || !Number.isFinite(xMaximum) || xMinimum >= xMaximum) {
      throw new Error('X axis range requires finite minimum < maximum');
    }
    if (!Number.isFinite(yMinimum) || !Number.isFinite(yMaximum) || yMinimum >= yMaximum) {
      throw new Error('Y axis range requires finite minimum < maximum');
    }
    if (!Number.isFinite(xStep) || xStep <= 0) throw new Error('X axis step must be positive');
    if (!Number.isFinite(yStep) || yStep <= 0) throw new Error('Y axis step must be positive');
    const fullX = this.bitmap.width * xStep;
    const fullY = this.bitmap.height * yStep;
    const x0Phys = clamp(xMinimum, 0, fullX);
    const x1Phys = clamp(xMaximum, 0, fullX);
    const y0Phys = clamp(yMinimum, 0, fullY);
    const y1Phys = clamp(yMaximum, 0, fullY);
    if (x0Phys >= x1Phys) {
      throw new Error('X axis range is outside the displayed dataset extent');
    }
    if (y0Phys >= y1Phys) {
      throw new Error('Y axis range is outside the displayed dataset extent');
    }
    const x0 = x0Phys / xStep;
    const x1 = x1Phys / xStep;
    const y0 = y0Phys / yStep;
    const y1 = y1Phys / yStep;
    const plot = this.plot();
    this.scaleX = plot.width / (x1 - x0);
    this.offsetX = plot.left - x0 * this.scaleX;
    this.scaleY = plot.height / (y1 - y0);
    this.offsetY = plot.top - (this.bitmap.height - y1) * this.scaleY;
    this.draw();
    this.emit('api-physical-range', true);
    return {
      x: {minimum: x0Phys, maximum: x1Phys},
      y: {minimum: y0Phys, maximum: y1Phys},
    };
  }

  setOverlay(overlay) {
    this.overlay = overlay;
    this.overlay.draw();
  }

  setPlotOverlay(overlay) {
    this.plotOverlay = overlay;
    this.plotOverlay.canvas.width = this.canvas.width;
    this.plotOverlay.canvas.height = this.canvas.height;
    this.plotOverlay.draw();
  }

  resize() {
    const width = Math.max(1, this.wrap.clientWidth);
    const height = Math.max(1, this.wrap.clientHeight);
    if (this.canvas.width === width && this.canvas.height === height) return;
    const wasAtHome = !this.home || this.atHome();
    this.canvas.width = width;
    this.canvas.height = height;
    if (this.interactionCanvas !== this.canvas) {
      this.interactionCanvas.width = width;
      this.interactionCanvas.height = height;
    }
    if (this.plotOverlay) {
      this.plotOverlay.canvas.width = width;
      this.plotOverlay.canvas.height = height;
    }
    if (wasAtHome) this.fit();
    this.draw();
  }

  atHome() {
    return this.home && ['scaleX', 'scaleY', 'offsetX', 'offsetY']
      .every(key => Math.abs(this[key] - this.home[key]) < 1e-6);
  }

  fit() {
    if (!this.bitmap) return;
    const plot = this.plot();
    if (this.bitmap.width === this.bitmap.height) {
      const scale = Math.min(
        plot.width / this.bitmap.width,
        plot.height / this.bitmap.height,
      ) * 0.98;
      this.scaleX = scale;
      this.scaleY = scale;
      this.offsetX = plot.left + (plot.width - this.bitmap.width * scale) / 2;
      this.offsetY = plot.top + (plot.height - this.bitmap.height * scale) / 2;
    } else {
      this.scaleX = plot.width / this.bitmap.width;
      this.scaleY = plot.height / this.bitmap.height;
      this.offsetX = plot.left;
      this.offsetY = plot.top;
    }
    this.home = {
      scaleX: this.scaleX,
      scaleY: this.scaleY,
      offsetX: this.offsetX,
      offsetY: this.offsetY,
    };
  }

  reset() {
    if (!this.home) return;
    Object.assign(this, this.home);
    this.draw();
  }

  zoomRegion(start, end) {
    const plot = this.plot();
    const selection = this.selectionRect(start, end);
    if (selection.width < MIN_REGION_PIXELS || selection.height < MIN_REGION_PIXELS) return;
    const x0 = (selection.left - this.offsetX) / this.scaleX;
    const x1 = (selection.left + selection.width - this.offsetX) / this.scaleX;
    const y0 = (selection.top - this.offsetY) / this.scaleY;
    const y1 = (selection.top + selection.height - this.offsetY) / this.scaleY;
    if (this.bitmap.width === this.bitmap.height) {
      const scale = Math.min(plot.width / (x1 - x0), plot.height / (y1 - y0));
      this.scaleX = scale;
      this.scaleY = scale;
    } else {
      this.scaleX = plot.width / (x1 - x0);
      this.scaleY = plot.height / (y1 - y0);
    }
    this.offsetX = plot.left - x0 * this.scaleX;
    this.offsetY = plot.top - y0 * this.scaleY;
  }

  zoomAxis(axis, start, end) {
    const plot = this.plot();
    if (axis === 'x') {
      const minimum = Math.min(start.x, end.x);
      const maximum = Math.max(start.x, end.x);
      if (maximum - minimum < MIN_REGION_PIXELS) return;
      const worldMinimum = (minimum - this.offsetX) / this.scaleX;
      const worldMaximum = (maximum - this.offsetX) / this.scaleX;
      const nextScale = plot.width / (worldMaximum - worldMinimum);
      this.scaleX = nextScale;
      this.offsetX = plot.left - worldMinimum * nextScale;
      return;
    }
    const minimum = Math.min(start.y, end.y);
    const maximum = Math.max(start.y, end.y);
    if (maximum - minimum < MIN_REGION_PIXELS) return;
    const worldMinimum = (minimum - this.offsetY) / this.scaleY;
    const worldMaximum = (maximum - this.offsetY) / this.scaleY;
    const nextScale = plot.height / (worldMaximum - worldMinimum);
    this.scaleY = nextScale;
    this.offsetY = plot.top - worldMinimum * nextScale;
  }

  updateDragClass() {
    this.wrap.classList.toggle('is-region-zooming', this.drag?.zoomMode === 'region');
    this.wrap.classList.toggle('is-axis-x-zooming', this.drag?.zoomMode === 'x');
    this.wrap.classList.toggle('is-axis-y-zooming', this.drag?.zoomMode === 'y');
  }

  clearDragClasses() {
    this.wrap.classList.remove(
      'is-panning',
      'is-region-zooming',
      'is-axis-x-zooming',
      'is-axis-y-zooming',
    );
  }

  selectionRect(start, end) {
    let endX = end.x;
    let endY = end.y;
    if (this.bitmap?.width === this.bitmap?.height) {
      const plot = this.plot();
      const directionX = end.x < start.x ? -1 : 1;
      const directionY = end.y < start.y ? -1 : 1;
      const requestedSide = Math.max(Math.abs(end.x - start.x), Math.abs(end.y - start.y));
      const availableX = directionX < 0 ? start.x - plot.left : plot.left + plot.width - start.x;
      const availableY = directionY < 0 ? start.y - plot.top : plot.top + plot.height - start.y;
      const side = Math.min(requestedSide, availableX, availableY);
      endX = start.x + directionX * side;
      endY = start.y + directionY * side;
    }
    return {
      left: Math.min(start.x, endX),
      top: Math.min(start.y, endY),
      width: Math.abs(endX - start.x),
      height: Math.abs(endY - start.y),
    };
  }

  visibleImageRect() {
    if (!this.bitmap) return null;
    const plot = this.plot();
    const imageLeft = this.offsetX;
    const imageRight = this.offsetX + this.bitmap.width * this.scaleX;
    const imageTop = this.offsetY;
    const imageBottom = this.offsetY + this.bitmap.height * this.scaleY;
    const left = Math.max(plot.left, imageLeft);
    const right = Math.min(plot.left + plot.width, imageRight);
    const top = Math.max(plot.top, imageTop);
    const bottom = Math.min(plot.top + plot.height, imageBottom);
    if (right <= left || bottom <= top) return null;
    return {left, top, width: right - left, height: bottom - top};
  }

  visibleRange() {
    const plot = this.plot();
    const imageLeft = this.offsetX;
    const imageRight = this.offsetX + this.bitmap.width * this.scaleX;
    const imageTop = this.offsetY;
    const imageBottom = this.offsetY + this.bitmap.height * this.scaleY;
    const left = clamp(plot.left, imageLeft, imageRight);
    const right = clamp(plot.left + plot.width, imageLeft, imageRight);
    const top = clamp(plot.top, imageTop, imageBottom);
    const bottom = clamp(plot.top + plot.height, imageTop, imageBottom);
    return {
      x: [
        clamp((left - this.offsetX) / this.scaleX, 0, this.bitmap.width),
        clamp((right - this.offsetX) / this.scaleX, 0, this.bitmap.width),
      ],
      y: [
        clamp(this.bitmap.height - (bottom - this.offsetY) / this.scaleY, 0, this.bitmap.height),
        clamp(this.bitmap.height - (top - this.offsetY) / this.scaleY, 0, this.bitmap.height),
      ],
    };
  }

  displayToCanvas(x, y) {
    return {
      x: this.offsetX + x * this.scaleX,
      y: this.offsetY + (this.bitmap.height - y) * this.scaleY,
    };
  }

  canvasToDisplay(x, y) {
    return {
      x: (x - this.offsetX) / this.scaleX,
      y: this.bitmap.height - (y - this.offsetY) / this.scaleY,
    };
  }

  sourceToCanvas(row, col) {
    const display = sourceToDisplay(row, col);
    return this.displayToCanvas(display.x, display.y);
  }

  canvasToSource(x, y) {
    const display = this.canvasToDisplay(x, y);
    return displayToSource(display.x, display.y);
  }

  emit(cause, final) {
    if (!this.bitmap) return;
    const imageRange = this.visibleRange();
    this.onChange?.({
      cause,
      final,
      image_range: imageRange,
      source_range: {row: [...imageRange.x], col: [...imageRange.y]},
    });
  }

  draw() {
    const context = this.ctx;
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.fillStyle = this.theme.canvasBackground;
    context.fillRect(0, 0, this.canvas.width, this.canvas.height);
    if (!this.bitmap) return;
    const plot = this.plot();
    context.save();
    context.beginPath();
    context.rect(plot.left, plot.top, plot.width, plot.height);
    context.clip();
    context.translate(this.offsetX, this.offsetY + this.bitmap.height * this.scaleY);
    context.scale(this.scaleX, -this.scaleY);
    context.imageSmoothingEnabled = this.scaleX < 1 || this.scaleY < 1;
    context.drawImage(this.bitmap, 0, 0);
    context.restore();
    this.drawRegionGuide();
    this.drawAxes();
    this.plotOverlay?.draw();
    this.overlay?.draw();
  }

  drawRegionGuide() {
    if (!this.drag || this.drag.pan || this.drag.zoomMode === 'pending') return;
    const selection = this.selectionRect(this.drag.start, this.drag.last);
    const imageRect = this.visibleImageRect();
    if (!imageRect) return;
    this.ctx.fillStyle = 'rgba(56, 189, 248, 0.18)';
    this.ctx.strokeStyle = 'rgba(56, 189, 248, 0.9)';
    if (this.drag.zoomMode === 'x') {
      this.ctx.fillRect(selection.left, imageRect.top, selection.width, imageRect.height);
      this.ctx.beginPath();
      this.ctx.moveTo(selection.left + 0.5, imageRect.top);
      this.ctx.lineTo(selection.left + 0.5, imageRect.top + imageRect.height);
      this.ctx.moveTo(selection.left + selection.width + 0.5, imageRect.top);
      this.ctx.lineTo(selection.left + selection.width + 0.5, imageRect.top + imageRect.height);
      this.ctx.stroke();
      return;
    }
    if (this.drag.zoomMode === 'y') {
      this.ctx.fillRect(imageRect.left, selection.top, imageRect.width, selection.height);
      this.ctx.beginPath();
      this.ctx.moveTo(imageRect.left, selection.top + 0.5);
      this.ctx.lineTo(imageRect.left + imageRect.width, selection.top + 0.5);
      this.ctx.moveTo(imageRect.left, selection.top + selection.height + 0.5);
      this.ctx.lineTo(imageRect.left + imageRect.width, selection.top + selection.height + 0.5);
      this.ctx.stroke();
      return;
    }
    this.ctx.fillRect(selection.left, selection.top, selection.width, selection.height);
    this.ctx.strokeRect(
      selection.left + 0.5,
      selection.top + 0.5,
      Math.max(0, selection.width - 1),
      Math.max(0, selection.height - 1),
    );
  }

  drawAxes() {
    if (!this.axes) return;
    const context = this.ctx;
    const axisRect = this.visibleImageRect();
    if (!axisRect) return;
    const range = this.visibleRange();
    const xStart = range.x[0] * this.axes.x.step;
    const xEnd = range.x[1] * this.axes.x.step;
    const yStart = range.y[0] * this.axes.y.step;
    const yEnd = range.y[1] * this.axes.y.step;
    context.save();
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.strokeStyle = this.theme.axisLine;
    context.fillStyle = this.theme.axisText;
    context.lineWidth = 1;
    context.font = '10px system-ui, sans-serif';
    context.strokeRect(
      axisRect.left + 0.5,
      axisRect.top + 0.5,
      Math.max(0, axisRect.width - 1),
      Math.max(0, axisRect.height - 1),
    );
    for (let index = 0; index < TICK_COUNT; index += 1) {
      const fraction = index / (TICK_COUNT - 1);
      const x = axisRect.left + fraction * axisRect.width;
      const y = axisRect.top + axisRect.height - fraction * axisRect.height;
      context.beginPath();
      context.moveTo(x, axisRect.top + axisRect.height);
      context.lineTo(x, axisRect.top + axisRect.height + 5);
      context.moveTo(axisRect.left - 5, y);
      context.lineTo(axisRect.left, y);
      context.stroke();
      context.textAlign = 'center';
      context.textBaseline = 'top';
      context.fillText(
        formatTick(xStart + fraction * (xEnd - xStart)),
        x,
        axisRect.top + axisRect.height + 7,
      );
      context.textAlign = 'right';
      context.textBaseline = 'middle';
      context.fillText(formatTick(yStart + fraction * (yEnd - yStart)), axisRect.left - 8, y);
    }
    const xUnit = this.axes.x.unit ? ` (${this.axes.x.unit})` : '';
    const yUnit = this.axes.y.unit ? ` (${this.axes.y.unit})` : '';
    context.textAlign = 'center';
    context.textBaseline = 'bottom';
    context.fillText(
      `${this.axes.x.label}${xUnit}`,
      axisRect.left + axisRect.width / 2,
      this.canvas.height - 1,
    );
    context.save();
    context.translate(10, axisRect.top + axisRect.height / 2);
    context.rotate(-Math.PI / 2);
    context.fillText(`${this.axes.y.label}${yUnit}`, 0, 0);
    context.restore();
    context.restore();
  }

  destroy() {
    this.resizeObserver.disconnect();
    clearTimeout(this.wheelEnd);
  }
}
