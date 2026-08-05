/** Node tests for dataset-scoped browser plane caching. */

import assert from 'node:assert/strict';
import test from 'node:test';

import {PlaneCache} from '../../../src/nicewidgets/raster_viewer_widget/web/plane-cache.js';

test('plane cache deduplicates matching in-flight and completed requests', async () => {
  const cache = new PlaneCache({});
  let fetches = 0;
  const expected = new Uint16Array([1, 2, 3]);
  cache.fetch = async () => {
    fetches += 1;
    return expected;
  };
  const channel = {id: 'channel_1'};
  const selection = {t_index: 2, z_index: 4, plus_minus_z: 1};
  assert.equal(cache.has(channel, selection), false);
  const [first, second] = await Promise.all([
    cache.get(channel, selection),
    cache.get(channel, selection),
  ]);
  const third = await cache.get(channel, selection);
  assert.equal(fetches, 1);
  assert.equal(cache.has(channel, selection), true);
  assert.equal(first, expected);
  assert.equal(second, expected);
  assert.equal(third, expected);
  cache.clear();
});

test('plane cache separates T/Z selections and projection radii', async () => {
  const cache = new PlaneCache({});
  let fetches = 0;
  cache.fetch = async () => new Uint16Array([++fetches]);
  const channel = {id: 'channel_1'};
  await cache.get(channel, {t_index: 0, z_index: 4, plus_minus_z: 0});
  await cache.get(channel, {t_index: 1, z_index: 4, plus_minus_z: 0});
  await cache.get(channel, {t_index: 1, z_index: 5, plus_minus_z: 0});
  await cache.get(channel, {t_index: 1, z_index: 5, plus_minus_z: 1});
  assert.equal(fetches, 4);
  cache.clear();
});

test('clearing a dataset cache aborts requests and discards every entry', async () => {
  const cache = new PlaneCache({});
  cache.fetch = async () => new Uint16Array([1]);
  const channel = {id: 'channel_1'};
  await cache.get(channel, {z_index: 2, plus_minus_z: 0});
  assert.equal(cache.entries.size, 1);
  assert.equal(cache.controller.signal.aborted, false);
  cache.clear();
  assert.equal(cache.controller.signal.aborted, true);
  assert.equal(cache.entries.size, 0);
});
