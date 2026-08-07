/** Node tests for public viewer-level visibility and physical X-range APIs. */

import assert from 'node:assert/strict';
import test from 'node:test';

import {
  RASTER_DESCRIPTOR_SCHEMA_VERSION,
  RasterViewer,
  configuredSliceStep,
  sliceControlPaneIndex,
  sliceValueLabel,
} from '../../../src/nicewidgets/raster_viewer_widget/web/raster-viewer.js';
import {RoiInteractionState} from '../../../src/nicewidgets/raster_viewer_widget/web/roi-overlay.js';
import {
  dragZoomMode,
  normalizeWheelZoomFactor,
  RasterViewport,
} from '../../../src/nicewidgets/raster_viewer_widget/web/viewport.js';

test('slice control follows the agreed pane for each layout', () => {
  assert.equal(sliceControlPaneIndex('side', 3), 2);
  assert.equal(sliceControlPaneIndex('stack', 3), 0);
  assert.equal(sliceControlPaneIndex('single', 1), 0);
  assert.equal(sliceControlPaneIndex('composite', 1), 0);
  assert.equal(sliceControlPaneIndex('side', 0), -1);
});

test('slice label reports zero-based current and maximum indices', () => {
  assert.equal(sliceValueLabel(0, 70), '0/69');
  assert.equal(sliceValueLabel(24, 70), '24/69');
});

test('default Z wheel direction moves up toward zero and down toward maximum', () => {
  assert.equal(configuredSliceStep(1), -1);
  assert.equal(configuredSliceStep(-1), 1);
  assert.equal(configuredSliceStep(1, false), 1);
  assert.equal(configuredSliceStep(-1, false), -1);
});

test('wheel zoom factor defaults to a gentle value and rejects unsafe values', () => {
  assert.equal(normalizeWheelZoomFactor(), 1.06);
  assert.equal(normalizeWheelZoomFactor(1.02), 1.02);
  assert.throws(() => normalizeWheelZoomFactor(1), /greater than 1/);
  assert.throws(() => normalizeWheelZoomFactor(Number.NaN), /greater than 1/);
});

test('viewer rejects missing and unsupported descriptor schema versions before loading', async () => {
  const viewer = Object.create(RasterViewer.prototype);
  await assert.rejects(() => viewer.load({}), /schema_version.*missing/);
  await assert.rejects(
    () => viewer.load({schema_version: '1.0'}),
    /schema_version.*1.0/,
  );
  assert.equal(RASTER_DESCRIPTOR_SCHEMA_VERSION, '2.0');
});

test('slice stepping clamps and coalesces rapid wheel navigation', async () => {
  const requests = [];
  const viewer = Object.create(RasterViewer.prototype);
  Object.assign(viewer, {
    dataset: {header: {sizes: {Z: 5}}},
    zIndex: 1,
    zSlider: {value: '1'},
    zOutput: {textContent: '1/4'},
    sliceWheelTimer: null,
    sliceWheelPending: false,
    planeGeneration: 0,
    requestPlaneUpdate: () => { requests.push(viewer.zIndex); },
  });
  assert.equal(viewer.stepSlice(1), true);
  assert.equal(viewer.stepSlice(1), true);
  assert.equal(viewer.zIndex, 3);
  assert.equal(viewer.zOutput.textContent, '3/4');
  assert.deepEqual(requests, [2]);
  await new Promise(resolve => setTimeout(resolve, 90));
  assert.deepEqual(requests, [2, 3]);
  assert.equal(viewer.stepSlice(10), true);
  assert.equal(viewer.zIndex, 4);
  assert.equal(viewer.stepSlice(1), false);
  clearTimeout(viewer.sliceWheelTimer);
});

test('slice wheel targets T when the dataset has no Z dimension', () => {
  const viewer = Object.create(RasterViewer.prototype);
  Object.assign(viewer, {
    dataset: {header: {sizes: {T: 3}}},
    tIndex: 1,
    tSlider: {value: '1'},
    tOutput: {textContent: '1/2'},
    sliceWheelTimer: 1,
    sliceWheelPending: false,
    planeGeneration: 0,
  });
  assert.equal(viewer.stepSlice(1), true);
  assert.equal(viewer.tIndex, 2);
  assert.equal(viewer.tOutput.textContent, '2/2');
  clearTimeout(viewer.sliceWheelTimer);
});

test('programmatic T and Z selection clamp and report the complete selection', async () => {
  const viewer = Object.create(RasterViewer.prototype);
  Object.assign(viewer, {
    dataset: {header: {sizes: {T: 3, Z: 5}}},
    tIndex: 0,
    zIndex: 0,
    plusMinusZ: 2,
    tSlider: {value: '0'},
    tOutput: {textContent: '0/2'},
    zSlider: {value: '0'},
    zOutput: {textContent: '0/4'},
    updatePlanes: async () => true,
  });
  assert.deepEqual(await viewer.setTIndex(20), {
    t_index: 2, z_index: 0, plus_minus_z: 2,
  });
  assert.equal(viewer.tOutput.textContent, '2/2');
  assert.deepEqual(await viewer.setZIndex(-4), {
    t_index: 2, z_index: 0, plus_minus_z: 2,
  });
  assert.equal(viewer.zOutput.textContent, '0/4');
});

test('channel selection is silent for caller API and emits for user selection', () => {
  const emitted = [];
  const viewer = Object.create(RasterViewer.prototype);
  Object.assign(viewer, {
    channels: [{id: 'a'}, {id: 'b'}],
    dataset: {id: 'sample'},
    selected: 'a',
    mode: 'side',
    dispatch: (name, detail) => emitted.push([name, detail]),
  });
  assert.equal(viewer.selectChannel('b', false), 'b');
  assert.deepEqual(emitted, []);
  assert.equal(viewer.selectChannel('a'), 'a');
  assert.equal(emitted[0][0], 'raster-channel-selected');
  assert.equal(emitted[0][1].channel_id, 'a');
});

test('programmatic channel display supports LUT-only and complete range updates', () => {
  const channel = {id: 'a', lut: 'gray', enabled: true, min: 2, max: 8};
  const viewer = Object.create(RasterViewer.prototype);
  Object.assign(viewer, {channels: [channel], render: () => {}, redraw: () => {}});
  viewer.setChannelDisplay('a', {
    lut: 'green', value_min: null, value_max: null, visible: false,
  });
  assert.equal(channel.lut, 'green');
  assert.equal(channel.min, 2);
  assert.equal(channel.enabled, false);
  const applied = viewer.setChannelDisplay('a', {value_min: 3, value_max: 7});
  assert.equal(applied.value_min, 3);
  assert.equal(applied.value_max, 7);
});

test('runtime calibration updates metadata and display axes without loading planes', () => {
  const viewer = Object.create(RasterViewer.prototype);
  Object.assign(viewer, {
    dataset: {
      header: {
        dims: ['T', 'Z', 'Y', 'X'],
        physical_units: [1, 1, 1, 1],
        physical_units_labels: ['frame', 'slice', 'px', 'px'],
      },
      axes: {},
    },
    viewports: [],
  });
  assert.deepEqual(
    viewer.setPhysicalCalibration([0.5, 1, 0.2, 0.3], ['s', 'slice', 'um', 'um']),
    {
      physical_units: [0.5, 1, 0.2, 0.3],
      physical_units_labels: ['s', 'slice', 'um', 'um'],
    },
  );
  assert.equal(viewer.displayAxes.x.step, 0.2);
  assert.equal(viewer.displayAxes.y.step, 0.3);
  assert.throws(
    () => viewer.setPhysicalCalibration([1, 1], ['px', 'px']),
    /must match active dataset dimensions/,
  );
});

test('reset view emits a final API reset for every pane', () => {
  const calls = [];
  const makeViewport = id => ({
    reset: () => calls.push(`${id}:reset`),
    emit: (cause, final) => calls.push(`${id}:${cause}:${final}`),
  });
  const viewer = Object.create(RasterViewer.prototype);
  viewer.viewports = [makeViewport('a'), makeViewport('b')];
  assert.equal(viewer.resetView(), true);
  assert.deepEqual(calls, [
    'a:reset', 'a:api-reset:true', 'b:reset', 'b:api-reset:true',
  ]);
});

test('drag zoom mode locks square regions and nonsquare dominant axes', () => {
  const start = {x: 20, y: 20};
  assert.equal(dragZoomMode({width: 100, height: 100}, start, {x: 24, y: 23}), 'pending');
  assert.equal(dragZoomMode({width: 100, height: 100}, start, {x: 40, y: 30}), 'region');
  assert.equal(dragZoomMode({width: 300, height: 100}, start, {x: 40, y: 30}), 'x');
  assert.equal(dragZoomMode({width: 300, height: 100}, start, {x: 25, y: 40}), 'y');
});

test('axes visibility updates controls and every viewport without resetting', () => {
  const axes = {x: {step: 0.2}, y: {step: 0.5}};
  const applied = [];
  const viewer = Object.create(RasterViewer.prototype);
  Object.assign(viewer, {
    showAxes: true,
    axesInput: {checked: true},
    displayAxes: axes,
    viewports: [{setAxes: value => applied.push(value)}, {setAxes: value => applied.push(value)}],
  });
  assert.equal(viewer.setAxesVisible(false), false);
  assert.equal(viewer.axesInput.checked, false);
  assert.deepEqual(applied, [null, null]);
  assert.equal(viewer.setAxesVisible(true), true);
  assert.equal(applied[2], axes);
});

test('ROI visibility redraws but cannot hide an active edit draft', () => {
  let redraws = 0;
  const viewer = Object.create(RasterViewer.prototype);
  Object.assign(viewer, {
    showRois: true,
    roisInput: {checked: true},
    roiState: RoiInteractionState.IDLE,
    redrawRois: () => { redraws += 1; },
  });
  assert.equal(viewer.setRoisVisible(false), false);
  assert.equal(redraws, 1);
  viewer.roiState = RoiInteractionState.EDITING;
  assert.equal(viewer.setRoisVisible(false), false);
  viewer.showRois = true;
  assert.equal(viewer.setRoisVisible(false), true);
});

test('channel toolbar visibility updates its menu control and complete pane headers', () => {
  const toolbars = [{hidden: false}, {hidden: false}];
  const viewer = Object.create(RasterViewer.prototype);
  Object.assign(viewer, {
    showChannelToolbars: true,
    channelToolbarsInput: {checked: true},
    stage: {querySelectorAll: selector => {
      assert.equal(selector, '.rv-pane-header');
      return toolbars;
    }},
  });
  assert.equal(viewer.setChannelToolbarsVisible(false), false);
  assert.equal(viewer.channelToolbarsInput.checked, false);
  assert.deepEqual(toolbars.map(item => item.hidden), [true, true]);
  assert.equal(viewer.setChannelToolbarsVisible(true), true);
  assert.deepEqual(toolbars.map(item => item.hidden), [false, false]);
});

test('caller-originated committed ROI CRUD is silent and reports outcomes', () => {
  let redraws = 0;
  const viewer = Object.create(RasterViewer.prototype);
  Object.assign(viewer, {
    dataset: {width: 20, height: 10},
    rois: [],
    selectedRoiId: null,
    redrawRois: () => { redraws += 1; },
    dispatch: () => { throw new Error('silent CRUD must not dispatch'); },
  });
  const first = {
    roi_id: 1,
    roi_type: 'rectroi',
    version: '1.0',
    name: '0',
    note: '',
    data: {row_start: 1, row_stop: 5, col_start: 2, col_stop: 8},
  };
  const line = {
    roi_id: 2,
    roi_type: 'linesegmentroi',
    version: '1.0',
    name: '1',
    note: '',
    data: {row0: 1, col0: 2, row1: 8, col1: 19},
  };
  assert.equal(viewer.setRois([first, line]), 2);
  assert.equal(viewer.selectedRoiId, 1);
  assert.equal(viewer.selectRoi(99), false);
  assert.equal(viewer.selectRoi(null), true);
  assert.equal(viewer.addRoi({...first, roi_id: 3, name: '2'}), true);
  assert.equal(viewer.rois.length, 3);
  assert.throws(() => viewer.addRoi(first), /already exists/);
  assert.equal(viewer.updateRoi({...first, name: 'updated'}), true);
  assert.throws(() => viewer.updateRoi({...first, roi_id: 99}), /does not exist/);
  assert.equal(viewer.removeRoi(99), false);
  assert.equal(viewer.removeRoi(1), true);
  assert.equal(viewer.rois.length, 2);
  assert.equal(redraws, 6);
});

test('viewer X range uses physical display-X units for every viewport', () => {
  const calls = [];
  const viewer = Object.create(RasterViewer.prototype);
  Object.assign(viewer, {
    displayAxes: {x: {step: 0.001, label: 'seconds', unit: ''}},
    displayWidth: 30000,
    viewports: [
      {setPhysicalXRange: (...args) => { calls.push(args); return {minimum: 5, maximum: 10}; }},
      {setPhysicalXRange: (...args) => { calls.push(args); return {minimum: 5, maximum: 10}; }},
    ],
  });
  assert.deepEqual(viewer.fullPhysicalXRange(), {
    minimum: 0,
    maximum: 30,
    label: 'seconds',
    unit: '',
  });
  assert.deepEqual(viewer.setXRange(5, 10), {minimum: 5, maximum: 10});
  assert.deepEqual(calls, [[5, 10, 0.001], [5, 10, 0.001]]);
});

test('setLayout remembers multi-channel mode before channels exist', () => {
  const viewer = Object.create(RasterViewer.prototype);
  let rendered = 0;
  Object.assign(viewer, {
    channels: [],
    dataset: null,
    mode: 'side',
    lastMultiChannelMode: 'side',
    layoutForNextLoad: null,
    modeControls: [],
    channelSelect: null,
    render: () => { rendered += 1; },
  });
  assert.equal(viewer.setLayout('composite'), 'composite');
  assert.equal(viewer.mode, 'composite');
  assert.equal(viewer.lastMultiChannelMode, 'composite');
  assert.equal(viewer.layoutForNextLoad, 'composite');
  assert.equal(rendered, 0);
});

test('setLayout single is preserved across the next multi-channel load mode choice', () => {
  const viewer = Object.create(RasterViewer.prototype);
  Object.assign(viewer, {
    channels: [],
    dataset: null,
    mode: 'side',
    lastMultiChannelMode: 'composite',
    layoutForNextLoad: null,
    modeControls: [],
    channelSelect: null,
    render: () => {},
  });
  assert.equal(viewer.setLayout('single'), 'single');
  assert.equal(viewer.layoutForNextLoad, 'single');
  assert.equal(viewer.lastMultiChannelMode, 'composite');
  // Mirror load()'s multi-channel branch.
  viewer.channels = [{id: '0'}, {id: '1'}];
  viewer.mode = viewer.layoutForNextLoad != null
    ? viewer.layoutForNextLoad
    : viewer.lastMultiChannelMode;
  viewer.layoutForNextLoad = null;
  assert.equal(viewer.mode, 'single');
});

test('setTIndex and setZIndex are no-ops when the axis is absent', async () => {
  const viewer = Object.create(RasterViewer.prototype);
  Object.assign(viewer, {
    dataset: {header: {sizes: {}}},
    tIndex: null,
    zIndex: null,
    plusMinusZ: 0,
    updatePlanes: async () => {
      throw new Error('updatePlanes should not run when axis is absent');
    },
  });
  assert.deepEqual(await viewer.setTIndex(3), {
    t_index: null, z_index: null, plus_minus_z: 0,
  });
  assert.deepEqual(await viewer.setZIndex(2), {
    t_index: null, z_index: null, plus_minus_z: 0,
  });
});

test('viewer physical range updates X and Y once per viewport', () => {
  const calls = [];
  const viewer = Object.create(RasterViewer.prototype);
  Object.assign(viewer, {
    displayAxes: {
      x: {step: 0.1, label: 'x', unit: ''},
      y: {step: 0.2, label: 'y', unit: ''},
    },
    viewports: [
      {
        setPhysicalRange: (...args) => {
          calls.push(args);
          return {x: {minimum: 1, maximum: 2}, y: {minimum: 3, maximum: 4}};
        },
      },
    ],
  });
  assert.deepEqual(viewer.setPhysicalRange(1, 2, 3, 4), {
    x: {minimum: 1, maximum: 2},
    y: {minimum: 3, maximum: 4},
  });
  assert.deepEqual(calls, [[1, 2, 0.1, 3, 4, 0.2]]);
});

test('viewport X range clamps physical values and preserves Y transform', () => {
  const viewport = Object.create(RasterViewport.prototype);
  let emitted = null;
  Object.assign(viewport, {
    bitmap: {width: 100, height: 50},
    scaleX: 1,
    scaleY: 7,
    offsetX: 0,
    offsetY: 13,
    plot: () => ({left: 20, top: 0, width: 400, height: 200}),
    draw: () => {},
    emit: (cause, final) => { emitted = {cause, final}; },
  });
  assert.deepEqual(viewport.setPhysicalXRange(-5, 30, 0.2), {
    minimum: 0,
    maximum: 20,
  });
  assert.equal(viewport.scaleX, 4);
  assert.equal(viewport.offsetX, 20);
  assert.equal(viewport.scaleY, 7);
  assert.equal(viewport.offsetY, 13);
  assert.deepEqual(emitted, {cause: 'api-x-range', final: true});
});

test('viewport physical range updates X and Y with one draw', () => {
  const viewport = Object.create(RasterViewport.prototype);
  let draws = 0;
  let emitted = null;
  Object.assign(viewport, {
    bitmap: {width: 100, height: 50},
    scaleX: 1,
    scaleY: 1,
    offsetX: 0,
    offsetY: 0,
    plot: () => ({left: 20, top: 10, width: 400, height: 200}),
    draw: () => { draws += 1; },
    emit: (cause, final) => { emitted = {cause, final}; },
  });
  assert.deepEqual(viewport.setPhysicalRange(0, 20, 0.2, 2, 8, 0.2), {
    x: {minimum: 0, maximum: 20},
    y: {minimum: 2, maximum: 8},
  });
  assert.equal(draws, 1);
  assert.deepEqual(emitted, {cause: 'api-physical-range', final: true});
  assert.equal(viewport.scaleX, 4);
  assert.equal(viewport.offsetX, 20);
  assert.equal(viewport.scaleY, 200 / 30);
});

test('visible range and axes are bounded to the displayed image footprint', () => {
  const viewport = Object.create(RasterViewport.prototype);
  Object.assign(viewport, {
    bitmap: {width: 100, height: 100},
    scaleX: 2,
    scaleY: 2,
    offsetX: 150,
    offsetY: 10,
    plot: () => ({left: 50, top: 10, width: 400, height: 200}),
  });
  assert.deepEqual(viewport.visibleImageRect(), {
    left: 150,
    top: 10,
    width: 200,
    height: 200,
  });
  assert.deepEqual(viewport.visibleRange(), {x: [0, 100], y: [0, 100]});

  viewport.offsetX = -50;
  assert.deepEqual(viewport.visibleImageRect(), {
    left: 50,
    top: 10,
    width: 100,
    height: 200,
  });
  assert.deepEqual(viewport.visibleRange(), {x: [50, 100], y: [0, 100]});
});

test('nonsquare axis zoom changes only its dominant transform', () => {
  const viewport = Object.create(RasterViewport.prototype);
  Object.assign(viewport, {
    scaleX: 2,
    scaleY: 3,
    offsetX: 10,
    offsetY: 20,
    plot: () => ({left: 50, top: 10, width: 400, height: 200}),
  });
  viewport.zoomAxis('x', {x: 100, y: 30}, {x: 300, y: 80});
  assert.equal(viewport.scaleX, 4);
  assert.equal(viewport.offsetX, -130);
  assert.equal(viewport.scaleY, 3);
  assert.equal(viewport.offsetY, 20);

  const xState = {scaleX: viewport.scaleX, offsetX: viewport.offsetX};
  viewport.zoomAxis('y', {x: 100, y: 50}, {x: 150, y: 150});
  assert.equal(viewport.scaleY, 6);
  assert.equal(viewport.offsetY, -50);
  assert.deepEqual(
    {scaleX: viewport.scaleX, offsetX: viewport.offsetX},
    xState,
  );
});
