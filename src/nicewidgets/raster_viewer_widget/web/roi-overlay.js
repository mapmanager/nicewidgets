/** Mixed rectangle/line ROI drawing, hit-testing, and transactional editing. */

const HANDLE_SIZE = 8;
const HIT_TOLERANCE = 6;
/** Max pointer travel (canvas px) to treat an IDLE ROI press as a click-select. */
const IDLE_CLICK_TOLERANCE = 6;

export const RoiType = Object.freeze({
  RECT: 'rectroi',
  LINE: 'linesegmentroi',
});

export const RoiInteractionState = Object.freeze({
  IDLE: 'idle',
  CREATING: 'creating',
  EDITING: 'editing',
});

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

export function normalizedBounds(bounds, width, height, roundValues = false) {
  const convert = roundValues ? Math.round : Number;
  let rowStart = convert(bounds.row_start ?? bounds.rowStart);
  let rowStop = convert(bounds.row_stop ?? bounds.rowStop);
  let colStart = convert(bounds.col_start ?? bounds.colStart);
  let colStop = convert(bounds.col_stop ?? bounds.colStop);
  [rowStart, rowStop] = [Math.min(rowStart, rowStop), Math.max(rowStart, rowStop)];
  [colStart, colStop] = [Math.min(colStart, colStop), Math.max(colStart, colStop)];
  rowStart = clamp(rowStart, 0, Math.max(0, height - 1));
  rowStop = clamp(rowStop, rowStart + 1, height);
  colStart = clamp(colStart, 0, Math.max(0, width - 1));
  colStop = clamp(colStop, colStart + 1, width);
  return {rowStart, rowStop, colStart, colStop};
}

export function normalizedEndpoints(endpoints, width, height, roundValues = false) {
  const convert = roundValues ? Math.round : Number;
  return {
    row0: clamp(convert(endpoints.row0), 0, height - 1),
    col0: clamp(convert(endpoints.col0), 0, width - 1),
    row1: clamp(convert(endpoints.row1), 0, height - 1),
    col1: clamp(convert(endpoints.col1), 0, width - 1),
  };
}

export function roiFromEnvelope(envelope, width, height) {
  const roiType = String(envelope.roi_type);
  const common = {
    roiId: Number(envelope.roi_id),
    roiType,
    version: String(envelope.version),
    name: String(envelope.name),
    note: String(envelope.note || ''),
  };
  if (!Number.isInteger(common.roiId) || common.roiId <= 0) {
    throw new Error('ROI ID must be a positive integer');
  }
  if (common.version !== '1.0') throw new Error(`unsupported ROI version: ${common.version}`);
  if (roiType === RoiType.RECT) {
    return {...common, bounds: normalizedBounds(envelope.data, width, height, true)};
  }
  if (roiType === RoiType.LINE) {
    return {...common, endpoints: normalizedEndpoints(envelope.data, width, height, true)};
  }
  throw new Error(`unsupported ROI type: ${roiType}`);
}

export function roiEnvelope(roi) {
  const common = {
    roi_id: roi.roiId,
    roi_type: roi.roiType,
    version: '1.0',
    name: roi.name,
    note: roi.note || '',
  };
  if (roi.roiType === RoiType.RECT) {
    return {...common, data: {
      row_start: roi.bounds.rowStart,
      row_stop: roi.bounds.rowStop,
      col_start: roi.bounds.colStart,
      col_stop: roi.bounds.colStop,
    }};
  }
  if (roi.roiType === RoiType.LINE) return {...common, data: {...roi.endpoints}};
  throw new Error(`unsupported ROI type: ${roi.roiType}`);
}

export function moveSourceBounds(bounds, deltaRow, deltaCol, sourceHeight, sourceWidth) {
  const rowSpan = bounds.rowStop - bounds.rowStart;
  const colSpan = bounds.colStop - bounds.colStart;
  const rowStart = clamp(bounds.rowStart + deltaRow, 0, sourceHeight - rowSpan);
  const colStart = clamp(bounds.colStart + deltaCol, 0, sourceWidth - colSpan);
  return {rowStart, rowStop: rowStart + rowSpan, colStart, colStop: colStart + colSpan};
}

export function resizeSourceBounds(
  bounds, handle, deltaRow, deltaCol, sourceHeight, sourceWidth,
) {
  const result = {...bounds};
  if (handle.includes('w')) {
    result.rowStart = clamp(bounds.rowStart + deltaRow, 0, bounds.rowStop - 1);
  }
  if (handle.includes('e')) {
    result.rowStop = clamp(bounds.rowStop + deltaRow, bounds.rowStart + 1, sourceHeight);
  }
  if (handle.includes('s')) {
    result.colStart = clamp(bounds.colStart + deltaCol, 0, bounds.colStop - 1);
  }
  if (handle.includes('n')) {
    result.colStop = clamp(bounds.colStop + deltaCol, bounds.colStart + 1, sourceWidth);
  }
  return result;
}

export function moveLineEndpoints(endpoints, deltaRow, deltaCol, sourceHeight, sourceWidth) {
  const rowDelta = clamp(
    deltaRow,
    -Math.min(endpoints.row0, endpoints.row1),
    sourceHeight - 1 - Math.max(endpoints.row0, endpoints.row1),
  );
  const colDelta = clamp(
    deltaCol,
    -Math.min(endpoints.col0, endpoints.col1),
    sourceWidth - 1 - Math.max(endpoints.col0, endpoints.col1),
  );
  return {
    row0: endpoints.row0 + rowDelta,
    col0: endpoints.col0 + colDelta,
    row1: endpoints.row1 + rowDelta,
    col1: endpoints.col1 + colDelta,
  };
}

export function moveLineEndpoint(
  endpoints, endpoint, deltaRow, deltaCol, sourceHeight, sourceWidth,
) {
  const result = {...endpoints};
  result[`row${endpoint}`] = clamp(
    endpoints[`row${endpoint}`] + deltaRow, 0, sourceHeight - 1,
  );
  result[`col${endpoint}`] = clamp(
    endpoints[`col${endpoint}`] + deltaCol, 0, sourceWidth - 1,
  );
  return result;
}

export function pointToSegmentDistance(point, start, stop) {
  const deltaX = stop.x - start.x;
  const deltaY = stop.y - start.y;
  const lengthSquared = deltaX * deltaX + deltaY * deltaY;
  if (!lengthSquared) return Math.hypot(point.x - start.x, point.y - start.y);
  const fraction = clamp(
    ((point.x - start.x) * deltaX + (point.y - start.y) * deltaY) / lengthSquared,
    0,
    1,
  );
  return Math.hypot(
    point.x - (start.x + fraction * deltaX),
    point.y - (start.y + fraction * deltaY),
  );
}

function copyRoi(roi) {
  return roi.roiType === RoiType.RECT
    ? {...roi, bounds: {...roi.bounds}}
    : {...roi, endpoints: {...roi.endpoints}};
}

export class RoiOverlay {
  constructor(canvas, viewport, viewer) {
    this.canvas = canvas;
    this.context = canvas.getContext('2d');
    this.viewport = viewport;
    this.viewer = viewer;
    this.active = null;
    /** @type {{roiId:number,start:{x:number,y:number}}|null} */
    this.pendingIdleSelect = null;
  }

  screenGeometry(roi) {
    if (roi.roiType === RoiType.RECT) {
      const lowerLeft = this.viewport.sourceToCanvas(roi.bounds.rowStart, roi.bounds.colStart);
      const upperRight = this.viewport.sourceToCanvas(roi.bounds.rowStop, roi.bounds.colStop);
      return {kind: 'rect', left: Math.min(lowerLeft.x, upperRight.x),
        right: Math.max(lowerLeft.x, upperRight.x), top: Math.min(lowerLeft.y, upperRight.y),
        bottom: Math.max(lowerLeft.y, upperRight.y)};
    }
    return {
      kind: 'line',
      start: this.viewport.sourceToCanvas(roi.endpoints.row0 + 0.5, roi.endpoints.col0 + 0.5),
      stop: this.viewport.sourceToCanvas(roi.endpoints.row1 + 0.5, roi.endpoints.col1 + 0.5),
    };
  }

  rectangleHandles(rect) {
    const centerX = (rect.left + rect.right) / 2;
    const centerY = (rect.top + rect.bottom) / 2;
    return {nw: {x: rect.left, y: rect.top}, n: {x: centerX, y: rect.top},
      ne: {x: rect.right, y: rect.top}, e: {x: rect.right, y: centerY},
      se: {x: rect.right, y: rect.bottom}, s: {x: centerX, y: rect.bottom},
      sw: {x: rect.left, y: rect.bottom}, w: {x: rect.left, y: centerY}};
  }

  hitHandle(point, roi) {
    const geometry = this.screenGeometry(roi);
    const handles = geometry.kind === 'rect'
      ? this.rectangleHandles(geometry)
      : {0: geometry.start, 1: geometry.stop};
    for (const [name, center] of Object.entries(handles)) {
      if (Math.abs(point.x - center.x) <= HANDLE_SIZE
          && Math.abs(point.y - center.y) <= HANDLE_SIZE) return name;
    }
    return null;
  }

  hitDistance(point, roi) {
    const geometry = this.screenGeometry(roi);
    if (geometry.kind === 'line') {
      return pointToSegmentDistance(point, geometry.start, geometry.stop);
    }
    const inside = point.x >= geometry.left && point.x <= geometry.right
      && point.y >= geometry.top && point.y <= geometry.bottom;
    if (inside) return 0;
    const deltaX = Math.max(geometry.left - point.x, 0, point.x - geometry.right);
    const deltaY = Math.max(geometry.top - point.y, 0, point.y - geometry.bottom);
    return Math.hypot(deltaX, deltaY);
  }

  hitRoi(point) {
    const candidates = this.viewer.visibleRois()
      .map(roi => ({roi, distance: this.hitDistance(point, roi)}))
      .filter(candidate => candidate.distance <= HIT_TOLERANCE);
    const selected = candidates.find(({roi}) => roi.roiId === this.viewer.selectedRoiId);
    if (selected) return selected.roi;
    candidates.sort((left, right) => left.distance - right.distance);
    return candidates[0]?.roi ?? null;
  }

  pointerDown(event, point) {
    if (!this.viewer.showRois || event.shiftKey) {
      this.pendingIdleSelect = null;
      return false;
    }
    if (this.viewer.roiState === RoiInteractionState.IDLE) {
      // Do not capture: viewport must receive drag for zoom. Click-select on pointerUp.
      const hit = this.hitRoi(point);
      this.pendingIdleSelect = hit
        ? {roiId: hit.roiId, start: {x: point.x, y: point.y}}
        : null;
      return false;
    }
    this.pendingIdleSelect = null;
    // CREATING / EDITING: capture only presses on the draft; outside → viewport zoom/pan.
    const draft = this.viewer.roiDraft;
    if (!draft) return false;
    const sourcePoint = this.viewport.canvasToSource(point.x, point.y);
    const handle = this.hitHandle(point, draft);
    const onBody = this.hitDistance(point, draft) <= HIT_TOLERANCE;
    if (handle === null && !onBody) return false;
    this.viewer.activeEditOverlay = this;
    this.active = {
      kind: handle !== null ? 'resize' : 'move',
      handle,
      start: sourcePoint,
      original: draft.roiType === RoiType.RECT ? {...draft.bounds} : {...draft.endpoints},
    };
    this.viewer.redrawRois();
    return true;
  }

  pointerMove(_event, point) {
    if (!this.active) return false;
    const draft = this.viewer.roiDraft;
    const current = this.viewport.canvasToSource(point.x, point.y);
    const deltaRow = current.row - this.active.start.row;
    const deltaCol = current.col - this.active.start.col;
    if (draft.roiType === RoiType.RECT) {
      draft.bounds = this.active.kind === 'move'
        ? moveSourceBounds(this.active.original, deltaRow, deltaCol,
          this.viewer.dataset.height, this.viewer.dataset.width)
        : resizeSourceBounds(this.active.original, this.active.handle, deltaRow, deltaCol,
          this.viewer.dataset.height, this.viewer.dataset.width);
    } else {
      draft.endpoints = this.active.kind === 'move'
        ? moveLineEndpoints(this.active.original, deltaRow, deltaCol,
          this.viewer.dataset.height, this.viewer.dataset.width)
        : moveLineEndpoint(this.active.original, Number(this.active.handle), deltaRow, deltaCol,
          this.viewer.dataset.height, this.viewer.dataset.width);
    }
    this.viewer.redrawRois();
    return true;
  }

  pointerUp(_event, point) {
    if (this.pendingIdleSelect) {
      const pending = this.pendingIdleSelect;
      this.pendingIdleSelect = null;
      const travel = point
        ? Math.hypot(point.x - pending.start.x, point.y - pending.start.y)
        : 0;
      if (travel <= IDLE_CLICK_TOLERANCE) {
        this.viewer.selectRoi(pending.roiId, {emit: true});
        return true;
      }
      return false;
    }
    if (!this.active) return false;
    this.active = null;
    return true;
  }

  pointerCancel() {
    this.active = null;
    this.pendingIdleSelect = null;
  }

  /**
   * Suppress viewport dblclick-reset only while interacting with an edit draft.
   *
   * IDLE allows reset everywhere (including over a committed ROI). The earlier
   * IDLE-on-ROI suppress was addressing a false “click resets zoom” symptom
   * whose real cause was a host re-select rebuilding single-mode panes.
   *
   * @param {{x:number,y:number}|null} [point] Canvas point under the double-click.
   * @returns {boolean} True when the viewport should ignore the double-click.
   */
  suppressDoubleClick(point = null) {
    if (this.viewer.roiState === RoiInteractionState.IDLE) return false;
    if (!this.viewer.showRois || point == null) return false;
    const draft = this.viewer.roiDraft;
    if (!draft) return false;
    return this.hitHandle(point, draft) !== null
      || this.hitDistance(point, draft) <= HIT_TOLERANCE;
  }

  draw() {
    this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (!this.viewer.showRois) return;
    for (const roi of this.viewer.visibleRois()) this.drawRoi(roi);
  }

  drawRoi(roi) {
    const geometry = this.screenGeometry(roi);
    const selected = roi.roiId === this.viewer.selectedRoiId || roi === this.viewer.roiDraft;
    const editing = roi === this.viewer.roiDraft;
    this.context.save();
    this.context.strokeStyle = selected ? '#fde047' : '#22d3ee';
    this.context.fillStyle = editing ? 'rgba(253, 224, 71, 0.12)' : 'rgba(34, 211, 238, 0.04)';
    this.context.lineWidth = selected ? 2 : 1.5;
    if (geometry.kind === 'rect') {
      this.context.fillRect(geometry.left, geometry.top,
        geometry.right - geometry.left, geometry.bottom - geometry.top);
      this.context.strokeRect(geometry.left + 0.5, geometry.top + 0.5,
        Math.max(0, geometry.right - geometry.left - 1),
        Math.max(0, geometry.bottom - geometry.top - 1));
    } else {
      this.context.beginPath();
      this.context.moveTo(geometry.start.x, geometry.start.y);
      this.context.lineTo(geometry.stop.x, geometry.stop.y);
      this.context.stroke();
    }
    if (selected) {
      const anchor = geometry.kind === 'rect'
        ? {x: geometry.left + 4, y: Math.max(12, geometry.top - 4)}
        : {x: geometry.start.x + 5, y: geometry.start.y - 5};
      this.context.fillStyle = '#fde047';
      this.context.font = '11px system-ui, sans-serif';
      const label = roi.name ? String(roi.name) : String(roi.roiId);
      this.context.fillText(label, anchor.x, anchor.y);
    }
    if (editing && this.viewer.activeEditOverlay === this) {
      const handles = geometry.kind === 'rect'
        ? Object.values(this.rectangleHandles(geometry))
        : [geometry.start, geometry.stop];
      for (const center of handles) {
        this.context.fillStyle = '#fde047';
        this.context.strokeStyle = '#111827';
        this.context.fillRect(center.x - HANDLE_SIZE / 2, center.y - HANDLE_SIZE / 2,
          HANDLE_SIZE, HANDLE_SIZE);
        this.context.strokeRect(center.x - HANDLE_SIZE / 2, center.y - HANDLE_SIZE / 2,
          HANDLE_SIZE, HANDLE_SIZE);
      }
    }
    this.context.restore();
  }

  cloneRoi(roi) { return copyRoi(roi); }
}
