/** Node tests for JavaScript color lookup tables. */

import assert from 'node:assert/strict';
import test from 'node:test';

import {LUT_LABELS, lutTable, sampleLut} from '../../../src/nicewidgets/raster_viewer_widget/web/lut.js';

test('blue is a first-class LUT, not a gray fallback', () => {
  assert.equal(LUT_LABELS.blue, 'Blue');
  assert.notEqual(lutTable('blue'), lutTable('gray'));
  assert.deepEqual(sampleLut('blue', 0), [0, 0, 0]);
  assert.deepEqual(sampleLut('blue', 1), [24, 80, 255]);
  const mid = sampleLut('blue', 0.8);
  assert.ok(mid[2] > mid[0] && mid[2] > mid[1]);
});
