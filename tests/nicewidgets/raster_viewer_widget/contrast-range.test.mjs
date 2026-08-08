/** Node tests for range-popover LUT histogram colors and clipboard helpers. */

import assert from 'node:assert/strict';
import test from 'node:test';

import {
  clipboardCopyAvailable,
} from '../../../src/nicewidgets/raster_viewer_widget/web/clipboard.js';
import {
  histogramBarColor,
  histogramBarFraction,
} from '../../../src/nicewidgets/raster_viewer_widget/web/contrast-range.js';

test('clipboard copy is available when the native host bridge is enabled', () => {
  assert.equal(clipboardCopyAvailable(true), true);
});

test('histogram bar color follows the selected LUT', () => {
  assert.equal(histogramBarColor('red'), 'rgb(204 26 19)');
  assert.equal(histogramBarColor('green'), 'rgb(16 204 64)');
  assert.equal(histogramBarColor('blue'), 'rgb(19 64 204)');
  assert.equal(histogramBarColor('gray'), 'rgb(204 204 204)');
});

test('histogram bar fraction supports log and linear Y scales', () => {
  assert.equal(histogramBarFraction(0, 100, true), 0);
  assert.equal(histogramBarFraction(100, 100, true), 1);
  assert.ok(histogramBarFraction(1, 100, true) > 0);
  // Log lifts small bins relative to linear.
  assert.ok(histogramBarFraction(1, 100, true) > histogramBarFraction(1, 100, false));
  assert.equal(histogramBarFraction(25, 100, false), 0.25);
  assert.equal(histogramBarFraction(100, 100, false), 1);
});