/** Non-interactive physical X/Y plots rendered above raster pixels. */

export const XYPlotMode = Object.freeze({
  MARKERS: 'markers',
  LINES: 'lines',
  LINES_MARKERS: 'lines_markers',
});

const SUPPORTED_MODES = new Set(Object.values(XYPlotMode));

function positiveNumber(value, name, fallback) {
  const normalized = value === undefined ? fallback : Number(value);
  if (!Number.isFinite(normalized) || normalized <= 0) {
    throw new Error(`${name} must be finite and positive`);
  }
  return normalized;
}

function opacityValue(value) {
  const normalized = value === undefined ? 1 : Number(value);
  if (!Number.isFinite(normalized) || normalized < 0 || normalized > 1) {
    throw new Error('plot opacity must be between 0 and 1');
  }
  return normalized;
}

function coordinateValue(value) {
  if (value === null || value === undefined) return null;
  const normalized = Number(value);
  return Number.isFinite(normalized) ? normalized : null;
}

/**
 * Validate and isolate one caller-owned physical X/Y plot specification.
 *
 * Non-finite coordinates become null gaps. Original indices are retained so
 * optional point IDs and future per-point plane coordinates remain aligned.
 *
 * @param {object} specification Caller plot specification.
 * @returns {object} Canonical plot safe for viewer-owned storage.
 */
export function normalizeXYPlot(specification) {
  const plotId = String(specification?.plot_id ?? '').trim();
  if (!plotId) throw new Error('plot_id must not be empty');
  if (!Array.isArray(specification.x) || !Array.isArray(specification.y)) {
    throw new Error('plot x and y must be arrays');
  }
  if (specification.x.length !== specification.y.length) {
    throw new Error('plot x and y must have equal lengths');
  }
  if (specification.coordinate_space !== undefined
      && specification.coordinate_space !== 'physical') {
    throw new Error('only physical XY plot coordinates are supported');
  }
  const mode = specification.mode ?? XYPlotMode.MARKERS;
  if (!SUPPORTED_MODES.has(mode)) throw new Error(`unsupported XY plot mode: ${mode}`);
  const style = specification.style ?? {};
  const color = String(style.color ?? '#facc15').trim();
  if (!color) throw new Error('plot color must not be empty');
  const channelIds = specification.channel_ids === null
    || specification.channel_ids === undefined
    ? null
    : [...new Set(specification.channel_ids.map(value => String(value)))];
  if (channelIds?.some(value => !value)) throw new Error('channel_ids must not contain empty IDs');
  const zIndex = specification.z_index === null || specification.z_index === undefined
    ? null
    : Number(specification.z_index);
  if (zIndex !== null && (!Number.isInteger(zIndex) || zIndex < 0)) {
    throw new Error('z_index must be null or a non-negative integer');
  }
  let pointIds = null;
  if (specification.point_ids !== null && specification.point_ids !== undefined) {
    if (!Array.isArray(specification.point_ids)
        || specification.point_ids.length !== specification.x.length) {
      throw new Error('point_ids must be null or match x/y length');
    }
    pointIds = specification.point_ids.map(value => String(value));
    if (pointIds.some(value => !value) || new Set(pointIds).size !== pointIds.length) {
      throw new Error('point_ids must be non-empty and unique');
    }
  }
  return {
    plotId,
    name: String(specification.name ?? plotId),
    x: specification.x.map(coordinateValue),
    y: specification.y.map(coordinateValue),
    pointIds,
    coordinateSpace: 'physical',
    mode,
    style: {
      color,
      markerSize: positiveNumber(style.marker_size, 'marker_size', 5),
      lineWidth: positiveNumber(style.line_width, 'line_width', 1.5),
      opacity: opacityValue(style.opacity),
    },
    channelIds,
    zIndex,
    visible: specification.visible !== false,
  };
}

/** Convert a physical plot point into display-image sample coordinates. */
export function physicalPointToDisplay(x, y, axes) {
  const xStep = Number(axes?.x?.step);
  const yStep = Number(axes?.y?.step);
  if (!Number.isFinite(xStep) || xStep <= 0 || !Number.isFinite(yStep) || yStep <= 0) {
    throw new Error('physical XY plots require positive display-axis steps');
  }
  return {x: x / xStep, y: y / yStep};
}

export function plotAppliesToPane(plot, channelIds, zIndex) {
  if (!plot.visible) return false;
  if (plot.zIndex !== null && plot.zIndex !== zIndex) return false;
  return plot.channelIds === null
    || channelIds.some(channelId => plot.channelIds.includes(channelId));
}

export class XYPlotOverlay {
  constructor(canvas, viewport, viewer, channelIds) {
    this.canvas = canvas;
    this.context = canvas.getContext('2d');
    this.viewport = viewport;
    this.viewer = viewer;
    this.channelIds = [...channelIds];
  }

  draw() {
    this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    const plots = [...this.viewer.xyPlots.values()].filter(plot => (
      plotAppliesToPane(plot, this.channelIds, this.viewer.zIndex)
    ));
    if (!plots.length) return;
    const plotRect = this.viewport.plot();
    this.context.save();
    this.context.beginPath();
    this.context.rect(plotRect.left, plotRect.top, plotRect.width, plotRect.height);
    this.context.clip();
    for (const plot of plots) this.drawPlot(plot);
    this.context.restore();
  }

  canvasPoint(plot, index) {
    const x = plot.x[index];
    const y = plot.y[index];
    if (x === null || y === null) return null;
    const display = physicalPointToDisplay(x, y, this.viewer.displayAxes);
    return this.viewport.displayToCanvas(display.x, display.y);
  }

  drawPlot(plot) {
    this.context.save();
    this.context.globalAlpha = plot.style.opacity;
    this.context.strokeStyle = plot.style.color;
    this.context.fillStyle = plot.style.color;
    this.context.lineWidth = plot.style.lineWidth;
    if (plot.mode === XYPlotMode.LINES || plot.mode === XYPlotMode.LINES_MARKERS) {
      this.drawLines(plot);
    }
    if (plot.mode === XYPlotMode.MARKERS || plot.mode === XYPlotMode.LINES_MARKERS) {
      this.drawMarkers(plot);
    }
    this.context.restore();
  }

  drawLines(plot) {
    this.context.beginPath();
    let segmentOpen = false;
    for (let index = 0; index < plot.x.length; index += 1) {
      const point = this.canvasPoint(plot, index);
      if (!point) {
        segmentOpen = false;
      } else if (!segmentOpen) {
        this.context.moveTo(point.x, point.y);
        segmentOpen = true;
      } else {
        this.context.lineTo(point.x, point.y);
      }
    }
    this.context.stroke();
  }

  drawMarkers(plot) {
    const radius = plot.style.markerSize / 2;
    for (let index = 0; index < plot.x.length; index += 1) {
      const point = this.canvasPoint(plot, index);
      if (!point) continue;
      this.context.beginPath();
      this.context.arc(point.x, point.y, radius, 0, Math.PI * 2);
      this.context.fill();
    }
  }
}
