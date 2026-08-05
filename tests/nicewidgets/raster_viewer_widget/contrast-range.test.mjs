/** Node tests for range-popover LUT histogram colors. */

import assert from 'node:assert/strict';
import test from 'node:test';

import {histogramBarColor} from '../../../src/nicewidgets/raster_viewer_widget/web/contrast-range.js';

test('histogram bar color follows the selected LUT', () => {
  assert.equal(histogramBarColor('red'), 'rgb(204 26 19)');
  assert.equal(histogramBarColor('green'), 'rgb(16 204 64)');
  assert.equal(histogramBarColor('blue'), 'rgb(19 64 204)');
  assert.equal(histogramBarColor('gray'), 'rgb(204 204 204)');
});
