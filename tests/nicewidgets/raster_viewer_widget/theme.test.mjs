/** Node tests for the viewer theme contract. */

import assert from 'node:assert/strict';
import test from 'node:test';

import {normalizeViewerTheme, ViewerTheme} from '../../../src/nicewidgets/raster_viewer_widget/web/theme.js';

test('viewer accepts exactly the light and dark themes', () => {
  assert.equal(normalizeViewerTheme(ViewerTheme.LIGHT), 'light');
  assert.equal(normalizeViewerTheme(ViewerTheme.DARK), 'dark');
  assert.throws(() => normalizeViewerTheme('auto'), /unsupported viewer theme/);
});
