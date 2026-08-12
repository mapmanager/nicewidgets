/** Thin instance-scoped NiceGUI adapter for the framework-independent viewer. */

/**
 * Bump when raster-viewer.css or the entry JS module changes.
 *
 * Web browsers cache NiceGUI static assets aggressively; native/pywebview often
 * does not. Versioning the CSS URL (same as the JS entry) keeps app and web
 * chrome in sync without divergent code paths.
 */
const RASTER_VIEWER_ASSETS_VERSION = 'axis-plotly-align-1';

const stylesheetPromises = new Map();

function ensureStylesheet(url) {
  if (stylesheetPromises.has(url)) return stylesheetPromises.get(url);
  // Drop older raster-viewer stylesheets so SPA remounts cannot keep stale rules
  // (e.g. pre-fix margin-left:auto) stacked with the new sheet.
  for (const link of document.querySelectorAll('link[data-raster-viewer-css]')) {
    const previous = link.dataset.rasterViewerCss;
    if (previous && previous !== url) {
      link.remove();
      stylesheetPromises.delete(previous);
    }
  }
  const promise = new Promise((resolve, reject) => {
    const stylesheet = document.createElement('link');
    stylesheet.rel = 'stylesheet';
    stylesheet.href = url;
    stylesheet.dataset.rasterViewerCss = url;
    stylesheet.addEventListener('load', resolve, {once: true});
    stylesheet.addEventListener('error', () => reject(
      new Error('raster viewer stylesheet failed to load'),
    ), {once: true});
    document.head.append(stylesheet);
  });
  stylesheetPromises.set(url, promise);
  return promise;
}

export default {
  template: '<div ref="host" class="raster-viewer-component-host" style="width:100%;height:100%;min-height:0"></div>',
  props: {
    descriptorUrl: {type: String, default: ''},
    initialTheme: {type: String, default: 'dark'},
    initialLayout: {type: String, default: 'auto'},
    initialAxesVisible: {type: Boolean, default: true},
    initialRoisVisible: {type: Boolean, default: true},
    initialChannelToolbarsVisible: {type: Boolean, default: true},
    initialRoiToolbarVisible: {type: Boolean, default: true},
    roiChromeEnabled: {type: Boolean, default: true},
    roiHostMode: {type: String, default: 'local'},
    invertSliceWheel: {type: Boolean, default: true},
    wheelZoomFactor: {type: Number, default: 1.06},
    // Default on so native/pywebview Copy view works even if the Python prop
    // is late or string-coerced; JS still prefers the browser Clipboard API.
    hostClipboardBridge: {type: Boolean, default: true},
    resourcePath: {type: String, required: true},
  },
  mounted() {
    this.readyPromise = this.initializeViewer().catch(error => {
      if (!this.viewer) this.$el.dispatchEvent(new CustomEvent('raster-error', {
        bubbles: true,
        detail: {message: error instanceof Error ? error.message : String(error)},
      }));
      throw error;
    });
  },
  beforeUnmount() {
    this.viewer?.destroy();
    this.viewer = null;
  },
  methods: {
    async initializeViewer() {
      const stylesheetUrl = (
        `${this.resourcePath}/raster-viewer.css?v=${RASTER_VIEWER_ASSETS_VERSION}`
      );
      await ensureStylesheet(stylesheetUrl);
      const module = await import(
        `${this.resourcePath}/raster-viewer.js?v=${RASTER_VIEWER_ASSETS_VERSION}`
      );
      this.viewer = new module.RasterViewer(this.$refs.host, {
        theme: this.initialTheme,
        invertSliceWheel: this.invertSliceWheel,
        wheelZoomFactor: this.wheelZoomFactor,
        roiHostMode: this.roiHostMode,
        roiToolbarVisible: this.initialRoiToolbarVisible,
        roiChromeEnabled: this.roiChromeEnabled,
        hostClipboardBridge: this.hostClipboardBridge,
      });
      if (this.descriptorUrl) await this.fetchAndLoad(this.descriptorUrl, this.viewer);
      this.viewer.setAxesVisible(this.initialAxesVisible);
      this.viewer.setRoisVisible(this.roiChromeEnabled && this.initialRoisVisible);
      this.viewer.setChannelToolbarsVisible(this.initialChannelToolbarsVisible);
      this.viewer.setRoiToolbarVisible(this.roiChromeEnabled && this.initialRoiToolbarVisible);
      if (this.initialLayout !== 'auto') this.viewer.setLayout(this.initialLayout);
      return true;
    },
    async getViewer() {
      await this.readyPromise;
      if (!this.viewer) throw new Error('raster viewer has been destroyed');
      return this.viewer;
    },
    async loadDescriptorUrl(url) {
      const viewer = await this.getViewer();
      return this.fetchAndLoad(url, viewer);
    },
    async fetchAndLoad(url, viewer) {
      try {
        const response = await fetch(url, {cache: 'no-store'});
        if (!response.ok) throw new Error(`dataset request failed: ${response.status}`);
        const descriptor = await response.json();
        await viewer.load(descriptor);
        return descriptor.id;
      } catch (error) {
        viewer.dispatch('raster-error', {
          message: error instanceof Error ? error.message : String(error),
        });
        throw error;
      }
    },
    async setTheme(value) { return (await this.getViewer()).setTheme(value); },
    async setLayout(value) { return (await this.getViewer()).setLayout(value); },
    async setXRange(minimum, maximum) {
      return (await this.getViewer()).setXRange(minimum, maximum);
    },
    async setYRange(minimum, maximum) {
      return (await this.getViewer()).setYRange(minimum, maximum);
    },
    async setPhysicalRange(xMinimum, xMaximum, yMinimum, yMaximum) {
      return (await this.getViewer()).setPhysicalRange(
        xMinimum, xMaximum, yMinimum, yMaximum,
      );
    },
    async setAxesVisible(value) { return (await this.getViewer()).setAxesVisible(value); },
    async setRoisVisible(value) { return (await this.getViewer()).setRoisVisible(value); },
    async setChannelToolbarsVisible(value) {
      return (await this.getViewer()).setChannelToolbarsVisible(value);
    },
    async setRoiToolbarVisible(value) {
      return (await this.getViewer()).setRoiToolbarVisible(value);
    },
    async setZIndex(value) { return (await this.getViewer()).setZIndex(value); },
    async setTIndex(value) { return (await this.getViewer()).setTIndex(value); },
    async setSlidingZ(enabled, radius) {
      return (await this.getViewer()).setSlidingZ(enabled, radius);
    },
    async clear() { return (await this.getViewer()).clear(); },
    async resetView() { return (await this.getViewer()).resetView(); },
    async resetXRange() { return (await this.getViewer()).resetXRange(); },
    async setPhysicalCalibration(units, labels) {
      return (await this.getViewer()).setPhysicalCalibration(units, labels);
    },
    async selectChannel(channelId) {
      return (await this.getViewer()).selectChannel(channelId, false);
    },
    async setChannelDisplay(channelId, display) {
      return (await this.getViewer()).setChannelDisplay(channelId, display);
    },
    async addXYPlot(value) { return (await this.getViewer()).addXYPlot(value); },
    async updateXYPlot(value) { return (await this.getViewer()).updateXYPlot(value); },
    async removeXYPlot(value) { return (await this.getViewer()).removeXYPlot(value); },
    async showXYPlot(value) { return (await this.getViewer()).showXYPlot(value); },
    async hideXYPlot(value) { return (await this.getViewer()).hideXYPlot(value); },
    async setRois(value) { return (await this.getViewer()).setRois(value); },
    async addRoi(value) { return (await this.getViewer()).addRoi(value); },
    async updateRoi(value) { return (await this.getViewer()).updateRoi(value); },
    async removeRoi(value) { return (await this.getViewer()).removeRoi(value); },
    async selectRoi(value) { return (await this.getViewer()).selectRoi(value); },
    async beginRoiCreate(value) { return (await this.getViewer()).beginRoiCreate(value); },
    async beginRoiEdit(value) { return (await this.getViewer()).beginRoiEdit(value); },
    async commitRoiEdit() { return (await this.getViewer()).commitRoiEdit(); },
    async cancelRoiEdit() { return (await this.getViewer()).cancelRoiEdit(); },
    async completeRoiCommit(value) {
      return (await this.getViewer()).completeRoiCommit(value);
    },
  },
};
