import assert from 'node:assert/strict';
import test from 'node:test';

import {
  normalizeXYPlot,
  physicalPointToDisplay,
  plotAppliesToPane,
} from '../../../src/nicewidgets/raster_viewer_widget/web/xy-plot-overlay.js';
import {RasterViewer} from '../../../src/nicewidgets/raster_viewer_widget/web/raster-viewer.js';

test('plot normalization preserves indices and converts non-finite values to gaps', () => {
  const plot = normalizeXYPlot({
    plot_id: 'measurements',
    x: [-2, Number.NaN, 12],
    y: [1, 2, Number.POSITIVE_INFINITY],
    point_ids: ['a', 'b', 'c'],
    mode: 'lines_markers',
  });
  assert.deepEqual(plot.x, [-2, null, 12]);
  assert.deepEqual(plot.y, [1, 2, null]);
  assert.deepEqual(plot.pointIds, ['a', 'b', 'c']);
  assert.equal(plot.coordinateSpace, 'physical');
});

test('physical coordinates are not clamped to the image extent', () => {
  const point = physicalPointToDisplay(-4, 25, {
    x: {step: 0.5},
    y: {step: 2.5},
  });
  assert.deepEqual(point, {x: -8, y: 10});
});

test('plot visibility can filter by pane channels and current Z plane', () => {
  const plot = normalizeXYPlot({
    plot_id: 'z-points',
    x: [1],
    y: [2],
    channel_ids: ['channel_1'],
    z_index: 4,
  });
  assert.equal(plotAppliesToPane(plot, ['channel_1'], 4), true);
  assert.equal(plotAppliesToPane(plot, ['channel_2'], 4), false);
  assert.equal(plotAppliesToPane(plot, ['channel_1'], 3), false);
});

test('viewer X/Y plot API supports strict CRUD and visibility by plot ID', () => {
  let redraws = 0;
  const viewer = {
    xyPlots: new Map(),
    viewports: [{plotOverlay: {draw: () => { redraws += 1; }}}],
  };
  for (const name of [
    'addXYPlot', 'updateXYPlot', 'removeXYPlot', 'showXYPlot',
    'hideXYPlot', 'setXYPlotVisible', 'redrawXYPlots',
  ]) viewer[name] = RasterViewer.prototype[name];

  assert.equal(viewer.addXYPlot({plot_id: 'one', x: [1], y: [2]}), 'one');
  assert.throws(
    () => viewer.addXYPlot({plot_id: 'one', x: [1], y: [2]}),
    /already exists/,
  );
  assert.equal(viewer.hideXYPlot('one'), true);
  assert.equal(viewer.xyPlots.get('one').visible, false);
  assert.equal(viewer.showXYPlot('missing'), false);
  assert.equal(viewer.updateXYPlot({
    plot_id: 'one', x: [3], y: [4], mode: 'lines',
  }), true);
  assert.deepEqual(viewer.xyPlots.get('one').x, [3]);
  assert.throws(
    () => viewer.updateXYPlot({plot_id: 'missing', x: [], y: []}),
    /does not exist/,
  );
  assert.equal(viewer.removeXYPlot('one'), true);
  assert.equal(viewer.removeXYPlot('one'), false);
  assert.equal(redraws, 4);
});
