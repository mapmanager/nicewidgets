/** Node tests for transpose orientation and source-coordinate ROI editing. */

import assert from 'node:assert/strict';
import test from 'node:test';

import {
  DISPLAY_ORIENTATION,
  displayToSource,
  sourceToDisplay,
  transposePlane,
  transposedAxes,
  transposedShape,
} from '../../../src/nicewidgets/raster_viewer_widget/web/orientation.js';
import {
  moveLineEndpoint,
  moveLineEndpoints,
  moveSourceBounds,
  normalizedEndpoints,
  pointToSegmentDistance,
  resizeSourceBounds,
  roiEnvelope,
  roiFromEnvelope,
} from '../../../src/nicewidgets/raster_viewer_widget/web/roi-overlay.js';

test('a nonsquare source plane is transposed without changing dtype', () => {
  assert.deepEqual(DISPLAY_ORIENTATION, {transpose: true, flip_y: true});
  const source = new Uint16Array([
    1, 2, 3,
    4, 5, 6,
  ]);
  const display = transposePlane(source, 2, 3);
  assert.ok(display instanceof Uint16Array);
  assert.deepEqual([...display], [1, 4, 2, 5, 3, 6]);
  assert.deepEqual(transposedShape(2, 3), {width: 2, height: 3});
});

test('float transpose preserves non-finite values', () => {
  const source = new Float32Array([1, Number.NaN, Number.POSITIVE_INFINITY, -2]);
  const display = transposePlane(source, 2, 2);
  assert.equal(display[0], 1);
  assert.equal(display[1], Number.POSITIVE_INFINITY);
  assert.ok(Number.isNaN(display[2]));
  assert.equal(display[3], -2);
});

test('source rows become display X and source columns become display Y', () => {
  assert.deepEqual(sourceToDisplay(17, 4), {x: 17, y: 4});
  assert.deepEqual(displayToSource(17, 4), {row: 17, col: 4});
  const sourceAxes = {
    x: {label: 'um', step: 0.2, unit: ''},
    y: {label: 'seconds', step: 0.001, unit: ''},
  };
  assert.deepEqual(transposedAxes(sourceAxes), {x: sourceAxes.y, y: sourceAxes.x});
});

test('ROI movement remains bounded in source coordinates', () => {
  const original = {rowStart: 10, rowStop: 30, colStart: 5, colStop: 15};
  assert.deepEqual(moveSourceBounds(original, 100, -100, 40, 20), {
    rowStart: 20,
    rowStop: 40,
    colStart: 0,
    colStop: 10,
  });
});

test('screen-direction handles update transposed source edges', () => {
  const original = {rowStart: 10, rowStop: 30, colStart: 5, colStop: 15};
  assert.equal(resizeSourceBounds(original, 'w', 4, 0, 40, 20).rowStart, 14);
  assert.equal(resizeSourceBounds(original, 'e', 4, 0, 40, 20).rowStop, 34);
  assert.equal(resizeSourceBounds(original, 's', 0, 3, 40, 20).colStart, 8);
  assert.equal(resizeSourceBounds(original, 'n', 0, 3, 40, 20).colStop, 18);
});

test('resize handles cannot invert or leave source bounds', () => {
  const original = {rowStart: 10, rowStop: 30, colStart: 5, colStop: 15};
  assert.deepEqual(resizeSourceBounds(original, 'nw', 100, 100, 40, 20), {
    rowStart: 29,
    rowStop: 30,
    colStart: 5,
    colStop: 20,
  });
});

test('line endpoints retain identity and clamp as source pixel indices', () => {
  assert.deepEqual(normalizedEndpoints(
    {row0: 99, col0: -3, row1: -4, col1: 20}, 10, 8, true,
  ), {row0: 7, col0: 0, row1: 0, col1: 9});
});

test('whole-line movement preserves its vector at image boundaries', () => {
  const original = {row0: 2, col0: 3, row1: 6, col1: 8};
  assert.deepEqual(moveLineEndpoints(original, 99, -99, 10, 12), {
    row0: 5, col0: 0, row1: 9, col1: 5,
  });
  assert.deepEqual(moveLineEndpoint(original, 0, -99, 99, 10, 12), {
    row0: 0, col0: 11, row1: 6, col1: 8,
  });
});

test('line envelopes round trip and segment hit distance uses screen pixels', () => {
  const envelope = {
    roi_id: 2, roi_type: 'linesegmentroi', version: '1.0', name: '1', note: '',
    data: {row0: 1, col0: 2, row1: 7, col1: 9},
  };
  assert.deepEqual(roiEnvelope(roiFromEnvelope(envelope, 10, 8)), envelope);
  assert.equal(pointToSegmentDistance(
    {x: 5, y: 3}, {x: 0, y: 0}, {x: 10, y: 0},
  ), 3);
});
