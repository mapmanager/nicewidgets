/** Self-contained multi-channel raster viewer with generated toolbar. */

import {autoRange, ContrastRangePopover} from './contrast-range.js';
import {
  clipboardCopyAvailable,
  clipboardImageSupported,
  composeViewportPngDataUrl,
  copyViewportToClipboard,
} from './clipboard.js?v=copy-1';
import {lucideIcon} from './icons.js?v=roi-chrome-1';
import {LUT_LABELS, lutTable} from './lut.js';
import {
  DISPLAY_ORIENTATION,
  transposedAxes,
  transposedShape,
} from './orientation.js';
import {
  normalizedBounds,
  normalizedEndpoints,
  roiEnvelope,
  roiFromEnvelope,
  RoiInteractionState,
  RoiOverlay,
  RoiType,
} from './roi-overlay.js?v=idle-dblclick-1';
import {RasterViewport} from './viewport.js?v=roi-click-1';
import {PlaneCache} from './plane-cache.js';
import {normalizeViewerTheme, readChromeTheme, ViewerTheme} from './theme.js';
import {ViewerTooltip} from './tooltip.js';
import {normalizeXYPlot, XYPlotOverlay} from './xy-plot-overlay.js';

let viewerInstanceCounter = 0;
/** @type {RasterViewer|null} Last pointer-activated viewer for digit layout hotkeys. */
let keyboardActiveViewer = null;
export const RASTER_DESCRIPTOR_SCHEMA_VERSION = '2.0';

/**
 * True when a key event target is an editable control that should keep the keystroke.
 *
 * @param {EventTarget|null} target Event target.
 * @returns {boolean} Whether layout hotkeys should ignore this event.
 */
function isEditableKeyboardTarget(target) {
  if (!(target instanceof Element)) return false;
  return Boolean(target.closest('input, textarea, select, [contenteditable=""], [contenteditable="true"]'));
}

/**
 * @typedef {object} RasterAxis
 * @property {string} label Axis label.
 * @property {number} step Positive physical distance per sample.
 * @property {string} unit Physical unit label, or an empty string.
 */

/**
 * @typedef {object} RasterChannelResource
 * @property {string} id Stable dataset-local channel identity.
 * @property {number} index Zero-based display index.
 * @property {string} label Human-readable name used in tooltips.
 * @property {'uint16'|'float32'} dtype Scalar transport dtype.
 * @property {'raw-u16-le'|'raw-f32-le'} encoding Little-endian binary encoding.
 * @property {number} byte_length Expected decoded 2D plane byte length.
 * @property {{lut:string,value_min:number|null,value_max:number|null,visible:boolean}} display
 *   Initial channel presentation state.
 * @property {string} data_url Same-origin or absolute 2D plane endpoint.
 */

/**
 * @typedef {object} RasterDescriptor
 * @property {'2.0'} schema_version Exact descriptor contract version.
 * @property {string} id Stable dataset identity.
 * @property {string} label Human-readable dataset label.
 * @property {{dims:Array<'Y'|'X'|'Z'|'T'>,sizes:Object<string,number>,dtype:string,num_channels:number,physical_units:number[],physical_units_labels:string[]}} header
 *   Snake-case channel-independent metadata. T and Z are optional axes before Y/X.
 * @property {number} width Source X/column count for each returned plane.
 * @property {number} height Source Y/row count for each returned plane.
 * @property {'row-major'} layout Binary sample layout.
 * @property {'little'} endianness Binary byte order.
 * @property {{transpose:true,flip_y:true}} display_orientation Required display transform.
 * @property {{x:RasterAxis,y:RasterAxis}} axes Source-plane axes transposed for display.
 * @property {RoiEnvelope[]} [rois] Initial committed mixed-shape ROI snapshot.
 * @property {RasterChannelResource[]} channels Same-shaped channel resources.
 */

/**
 * @typedef {object} RectRoiData
 * @property {number} row_start Inclusive source-array row edge.
 * @property {number} row_stop Exclusive source-array row edge.
 * @property {number} col_start Inclusive source-array column edge.
 * @property {number} col_stop Exclusive source-array column edge.
 */

/**
 * @typedef {object} RectRoiEnvelope
 * @property {number} roi_id Stable positive committed identity.
 * @property {'rectroi'} roi_type Rectangle discriminator.
 * @property {'1.0'} version ROI schema version.
 * @property {string} name Display name.
 * @property {string} note Optional user note.
 * @property {RectRoiData} data Integer half-open source-array bounds.
 */

/**
 * @typedef {object} LineRoiEnvelope
 * @property {number} roi_id Stable positive committed identity.
 * @property {'linesegmentroi'} roi_type Line-segment discriminator.
 * @property {'1.0'} version ROI schema version.
 * @property {string} name Human-readable display name.
 * @property {string} note Optional user note.
 * @property {{row0:number,col0:number,row1:number,col1:number}} data
 *   Integer endpoints identifying source-array pixel centers.
 */

/** @typedef {RectRoiEnvelope|LineRoiEnvelope} RoiEnvelope */

/**
 * @typedef {object} RoiCreateSpecification
 * @property {'rectroi'|'linesegmentroi'} roi_type Shape discriminator.
 * @property {string} name Proposed display name.
 * @property {string} [note] Optional user note.
 * @property {RectRoiData|{row0:number,col0:number,row1:number,col1:number}} data
 *   Initial source-array geometry for the uncommitted draft.
 */

/**
 * @typedef {object} XYPlotSpecification
 * @property {string} plot_id Stable dataset-local plot identity.
 * @property {number[]} x Physical display-X coordinates; non-finite values become gaps.
 * @property {number[]} y Physical display-Y coordinates; non-finite values become gaps.
 * @property {string} [name] Optional human-readable plot name.
 * @property {'markers'|'lines'|'lines_markers'} [mode='markers'] Drawing mode.
 * @property {{color?:string,marker_size?:number,line_width?:number,opacity?:number}} [style]
 *   Screen-space visual styling.
 * @property {boolean} [visible=true] Initial visibility.
 * @property {string[]|null} [channel_ids=null] Pane-channel filter, or every pane.
 * @property {number|null} [z_index=null] Whole-plot Z filter, or every plane.
 * @property {string[]|null} [point_ids=null] Stable identities reserved for future interaction.
 * @property {'physical'} [coordinate_space='physical'] Coordinate-space discriminator.
 */

function renderBitmap(width, height, channels) {
  const image = new ImageData(width, height);
  const pixels = image.data;
  const activeChannels = channels.filter(channel => channel.enabled).map(channel => ({
    data: channel.data,
    inverseSpan: 1 / Math.max(1e-12, channel.max - channel.min),
    minimum: channel.min,
    opacity: channel.opacity,
    table: lutTable(channel.lut),
  }));
  for (let index = 0; index < width * height; index += 1) {
    let red = 0;
    let green = 0;
    let blue = 0;
    for (const channel of activeChannels) {
      const normalized = (channel.data[index] - channel.minimum) * channel.inverseSpan;
      const tableIndex = Math.round(Math.max(0, Math.min(1, normalized)) * 255) * 3;
      red += channel.table[tableIndex] * channel.opacity;
      green += channel.table[tableIndex + 1] * channel.opacity;
      blue += channel.table[tableIndex + 2] * channel.opacity;
    }
    const offset = index * 4;
    pixels[offset] = red;
    pixels[offset + 1] = green;
    pixels[offset + 2] = blue;
    pixels[offset + 3] = 255;
  }
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  canvas.getContext('2d').putImageData(image, 0, 0);
  return canvas;
}

export function sliceControlPaneIndex(mode, paneCount) {
  if (!Number.isInteger(paneCount) || paneCount < 1) return -1;
  return mode === 'stack' ? 0 : paneCount - 1;
}

export function sliceValueLabel(index, size) {
  return `${index}/${size - 1}`;
}

/** Return the configured Z-step for the viewport's legacy signed wheel delta. */
export function configuredSliceStep(delta, invertSliceWheel = true) {
  return invertSliceWheel ? -delta : delta;
}

export class RasterViewer {
  /**
   * Create one self-contained raster viewer inside a host element.
   *
   * @param {HTMLElement} host Empty element owned by this viewer instance.
   * @param {{
   *   theme?:'light'|'dark',
   *   invertSliceWheel?:boolean,
   *   wheelZoomFactor?:number,
   *   roiHostMode?:'local'|'delegated',
   *   roiToolbarVisible?:boolean,
   *   hostClipboardBridge?:boolean,
   * }} [options]
   *   Initial presentation and interaction options. `wheelZoomFactor` defaults
   *   to 1.06; values closer to 1 zoom more slowly (for example, 1.03), and its
   *   valid range is greater than 1 through 2 inclusive. `roiHostMode` defaults
   *   to `local` (JS mutates its own ROI list). Use `delegated` when a host owns
   *   ROI truth and must accept request events before silent `*Roi` APIs run.
   *   `hostClipboardBridge` enables Copy view when the browser Clipboard API is
   *   unavailable (NiceGUI native / pywebview) by emitting PNG bytes to Python.
   */
  constructor(host, options = {}) {
    this.host = host;
    this.instanceId = ++viewerInstanceCounter;
    this.options = options;
    this.theme = normalizeViewerTheme(options.theme || ViewerTheme.DARK);
    this.invertSliceWheel = options.invertSliceWheel !== false;
    this.wheelZoomFactor = options.wheelZoomFactor ?? 1.06;
    this.roiHostMode = options.roiHostMode === 'delegated' ? 'delegated' : 'local';
    this.roiChromeEnabled = options.roiChromeEnabled !== false;
    // NiceGUI/Vue may deliver booleans as real bools or as "true"/"false" strings.
    this.hostClipboardBridge = options.hostClipboardBridge === true
      || options.hostClipboardBridge === 'true';
    this.showRoiToolbar = this.roiChromeEnabled && options.roiToolbarVisible !== false;
    this.dataset = null;
    this.channels = [];
    this.mode = 'side';
    this.lastMultiChannelMode = 'side';
    /** @type {'side'|'stack'|'single'|'composite'|null} Host layout for the next load. */
    this.layoutForNextLoad = null;
    this.selected = '';
    this.showAxes = true;
    this.showRois = this.roiChromeEnabled;
    this.showChannelToolbars = true;
    this.viewports = [];
    this.rois = [];
    this.xyPlots = new Map();
    this.selectedRoiId = null;
    this.nextLocalRoiId = 1;
    this.roiState = RoiInteractionState.IDLE;
    this.roiDraft = null;
    this.activeEditOverlay = null;
    this.modeControls = [];
    this.paneControls = [];
    this.roiIdleControls = [];
    this.roiEditControls = [];
    this.planeCache = null;
    this.planeGeneration = 0;
    this.tIndex = null;
    this.zIndex = null;
    this.plusMinusZ = 0;
    this.savedSlidingRadius = 1;
    this.planeTimer = null;
    this.sliceWheelTimer = null;
    this.sliceWheelPending = false;
    this.pointerHandler = event => {
      if (this.optionsMenu?.open && !this.optionsMenu.contains(event.target)) {
        this.optionsMenu.open = false;
      }
    };
    this.hostPointerHandler = () => {
      keyboardActiveViewer = this;
    };
    this.keyHandler = event => this.onDocumentKeyDown(event);
    document.addEventListener('keydown', this.keyHandler);
    document.addEventListener('pointerdown', this.pointerHandler);
    this.buildShell();
    this.host.addEventListener('pointerdown', this.hostPointerHandler);
    this.tooltip = new ViewerTooltip(this.host, this.instanceId);
  }

  /**
   * Document key handler: Escape for chrome, layout/reset hotkeys when activated.
   *
   * @param {KeyboardEvent} event Browser keydown event.
   * @returns {void}
   */
  onDocumentKeyDown(event) {
    if (event.key === 'Escape' && this.roiState !== RoiInteractionState.IDLE) {
      if (this.roiHostMode === 'delegated') this.requestRoiEditCancel();
      else this.cancelRoiEdit();
      return;
    }
    if (event.key === 'Escape' && this.optionsMenu?.open) {
      this.optionsMenu.open = false;
      return;
    }
    if (keyboardActiveViewer !== this) return;
    if (isEditableKeyboardTarget(event.target)) return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (this.handleChannelLayoutHotkey(event.key) || this.handleResetViewHotkey(event.key)) {
      event.preventDefault();
    }
  }

  /**
   * Apply multi-channel layout hotkeys: ``1``/``2`` → one-channel 0/1, ``3`` → composite.
   *
   * Inactive when the dataset has fewer than two channels. Requires this viewer
   * to be the pointer-activated keyboard target (see host ``pointerdown``).
   *
   * @param {string} key ``event.key`` value.
   * @returns {boolean} True when the key was handled.
   */
  handleChannelLayoutHotkey(key) {
    if (!this.dataset || this.channels.length < 2) return false;
    if (key === '1' || key === '2') {
      const channel = this.channels[key === '1' ? 0 : 1];
      if (!channel) return false;
      this.setLayout('single');
      this.selectChannel(channel.id);
      this.toolbarAction('view-mode', {mode: 'single'});
      return true;
    }
    if (key === '3') {
      this.setLayout('composite');
      this.toolbarAction('view-mode', {mode: 'composite'});
      return true;
    }
    return false;
  }

  /**
   * Enter/Return triggers the same action as Viewer options → Reset view.
   *
   * @param {string} key ``event.key`` value.
   * @returns {boolean} True when the key was handled.
   */
  handleResetViewHotkey(key) {
    if (key !== 'Enter') return false;
    if (!this.dataset) return false;
    this.resetView();
    this.toolbarAction('reset-view', {});
    if (this.optionsMenu) this.optionsMenu.open = false;
    return true;
  }

  buildShell() {
    this.host.classList.add('rv-root');
    this.host.dataset.theme = this.theme;
    this.host.replaceChildren();
    this.toolbar = document.createElement('div');
    this.toolbar.className = 'rv-toolbar';
    this.stage = document.createElement('div');
    this.stage.className = 'rv-stage';
    this.host.append(this.toolbar, this.stage);
    this.rangePopover = new ContrastRangePopover(this.host, () => this.redraw('user'));
    this.chromeTheme = readChromeTheme(this.host);
    this.rangePopover.setTheme(this.chromeTheme);
  }

  /**
   * Apply viewer chrome theme without changing raster or ROI analysis colors.
   *
   * @param {'light'|'dark'} theme Requested theme.
   * @returns {'light'|'dark'} Applied normalized theme.
   */
  setTheme(theme) {
    this.theme = normalizeViewerTheme(theme);
    this.host.dataset.theme = this.theme;
    this.chromeTheme = readChromeTheme(this.host);
    this.rangePopover.setTheme(this.chromeTheme);
    for (const viewport of this.viewports) viewport.setTheme(this.chromeTheme);
    return this.theme;
  }

  /**
   * Apply one supported channel layout and rebuild the view panes.
   *
   * @param {'side'|'stack'|'single'|'composite'} layout Requested layout.
   * @returns {'side'|'stack'|'single'|'composite'} Applied layout.
   */
  setLayout(layout) {
    const supported = new Set(['side', 'stack', 'single', 'composite']);
    if (!supported.has(layout)) throw new Error(`unsupported viewer layout: ${layout}`);
    if (this.channels.length === 1) layout = 'single';
    this.mode = layout;
    // Remember host intent for the next load, including ``single`` (which must
    // not be lost when load would otherwise fall back to lastMultiChannelMode).
    this.layoutForNextLoad = layout;
    // Always remember multi-channel layouts even before channels exist so the
    // next loadSource first-paints with the host-requested mode (no flash).
    if (layout !== 'single') this.lastMultiChannelMode = layout;
    for (const input of this.modeControls) {
      if (input.type === 'radio') input.checked = input.value === layout;
    }
    // Match buildToolbar: one-channel datasets never show the channel <select>
    // (CloudScope set_layout('single') before load_source must not flash it).
    if (this.channelSelect) {
      this.channelSelect.hidden = this.channels.length === 1 || layout !== 'single';
    }
    this.syncRoiToolbarDivider();
    if (this.dataset && this.channels.length > 0) this.render();
    return this.mode;
  }

  /**
   * Load a dataset descriptor, fetch its initial planes, and replace viewer state.
   *
   * Initial committed ROIs are read from `descriptor.rois`. Dataset changes clear
   * the decoded plane cache and cancel any active ROI draft.
   *
   * @param {RasterDescriptor} descriptor Complete snake-case raster descriptor.
   * @returns {Promise<void>} Resolves after initial planes and UI are ready.
   * @throws {Error} If orientation, channel bytes, dtype, or metadata are invalid.
   */
  async load(descriptor) {
    if (descriptor?.schema_version !== RASTER_DESCRIPTOR_SCHEMA_VERSION) {
      throw new Error(
        `unsupported raster descriptor schema_version: ${descriptor?.schema_version ?? 'missing'}`,
      );
    }
    clearTimeout(this.sliceWheelTimer);
    this.sliceWheelTimer = null;
    this.sliceWheelPending = false;
    this.rangePopover.close();
    this.planeCache?.clear();
    this.xyPlots.clear();
    this.planeCache = new PlaneCache(descriptor, detail => {
      this.performanceMetric('plane-fetch', detail);
    });
    this.planeGeneration += 1;
    this.dataset = descriptor;
    const orientation = descriptor.display_orientation || DISPLAY_ORIENTATION;
    if (!orientation.transpose || !orientation.flip_y) {
      throw new Error('viewer requires transpose plus flip-Y display orientation');
    }
    const displayShape = transposedShape(descriptor.height, descriptor.width);
    this.displayWidth = displayShape.width;
    this.displayHeight = displayShape.height;
    this.displayAxes = transposedAxes(descriptor.axes);
    this.roiState = RoiInteractionState.IDLE;
    this.roiDraft = null;
    this.activeEditOverlay = null;
    const tSize = descriptor.header?.sizes?.T ?? null;
    const zSize = descriptor.header?.sizes?.Z ?? null;
    this.tIndex = tSize === null ? null : 0;
    this.zIndex = zSize === null ? null : 0;
    this.plusMinusZ = 0;
    this.channels = descriptor.channels.map(metadata => ({
      ...metadata,
      data: null,
      min: metadata.display?.value_min ?? 0,
      max: metadata.display?.value_max ?? 1,
      explicitRange: metadata.display?.value_min != null && metadata.display?.value_max != null,
      lut: metadata.display?.lut || 'gray',
      opacity: 1,
      enabled: metadata.display?.visible !== false,
      histogram: null,
    }));
    await this.updatePlanes({initializeContrast: true, render: false});
    if (this.channels.length === 1) {
      this.mode = 'single';
    } else if (this.layoutForNextLoad != null) {
      this.mode = this.layoutForNextLoad;
    } else {
      this.mode = this.lastMultiChannelMode;
    }
    this.layoutForNextLoad = null;
    this.setRois(descriptor.rois || []);
    this.selected = this.channels[0].id;
    this.buildToolbar();
    this.render();
    this.dispatch('raster-ready', {
      dataset_id: descriptor.id,
      x_axis: this.fullPhysicalXRange(),
    });
  }

  buildToolbar() {
    this.toolbar.replaceChildren();
    this.modeControls = [];
    this.layoutControls = null;
    this.slidingZControls = null;
    this.roiToolbarDivider = null;
    this.roiToolbar = null;
    this.tSlider = null;
    this.tOutput = null;
    this.zSlider = null;
    this.zOutput = null;

    // Viewer options leftmost; layout / Sliding-Z / ROI strip follow in document order.
    const menu = document.createElement('details');
    menu.className = 'rv-options-menu';
    const menuButton = document.createElement('summary');
    menuButton.dataset.rvTooltip = 'Viewer options';
    menuButton.setAttribute('aria-label', 'Viewer options');
    menuButton.append(lucideIcon('menu', 'Viewer options'));
    const menuPanel = document.createElement('div');
    menuPanel.className = 'rv-options-panel';
    menu.append(menuButton, menuPanel);
    this.optionsMenu = menu;
    this.toolbar.append(menu);

    this.axesInput = this.visibilityControl(menuPanel, 'Axes', this.showAxes, visible => {
      this.setAxesVisible(visible);
      this.toolbarAction('axes', {visible});
    });
    if (this.roiChromeEnabled) {
      this.roisInput = this.visibilityControl(menuPanel, 'ROIs', this.showRois, visible => {
        this.setRoisVisible(visible);
        this.toolbarAction('rois', {visible});
      });
    }
    this.channelToolbarsInput = this.visibilityControl(
      menuPanel,
      'Channel Toolbars',
      this.showChannelToolbars,
      visible => {
        this.setChannelToolbarsVisible(visible);
        this.toolbarAction('channel-toolbars', {visible});
      },
    );
    if (this.roiChromeEnabled) {
      this.roiToolbarInput = this.visibilityControl(
        menuPanel,
        'ROI Toolbar',
        this.showRoiToolbar,
        visible => {
          this.setRoiToolbarVisible(visible);
          this.toolbarAction('roi-toolbar', {visible});
        },
      );
    }

    const resetButton = document.createElement('button');
    resetButton.type = 'button';
    resetButton.className = 'rv-menu-action';
    resetButton.setAttribute('aria-label', 'Reset view');
    resetButton.append(lucideIcon('maximize-2', ''), 'Reset view');
    resetButton.addEventListener('click', () => {
      this.resetView();
      this.toolbarAction('reset-view', {});
      menu.open = false;
    });
    menuPanel.append(resetButton);
    this.modeControls.push(resetButton);

    const modes = [
      ['side', 'Side by side', 'columns-2'],
      ['stack', 'Stacked', 'rows-2'],
      ['single', 'One channel', 'square'],
      ['composite', 'Composite', 'layers-3'],
    ];
    const layoutGroup = document.createElement('div');
    layoutGroup.className = 'rv-layout-controls';
    layoutGroup.setAttribute('role', 'radiogroup');
    layoutGroup.setAttribute('aria-label', 'Channel layout');
    layoutGroup.hidden = this.channels.length === 1;
    this.layoutControls = layoutGroup;
    for (const [value, text, icon] of modes) {
      const label = document.createElement('label');
      label.className = 'rv-icon-radio';
      label.dataset.rvTooltip = text;
      const input = document.createElement('input');
      input.type = 'radio';
      input.name = `rv-mode-${this.instanceId}`;
      input.value = value;
      input.checked = this.mode === value;
      input.setAttribute('aria-label', text);
      this.modeControls.push(input);
      input.addEventListener('change', () => {
        this.mode = value;
        this.lastMultiChannelMode = value;
        this.channelSelect.hidden = value !== 'single';
        this.syncRoiToolbarDivider();
        this.render();
        this.toolbarAction('view-mode', {mode: value});
      });
      label.append(input, lucideIcon(icon, text));
      layoutGroup.append(label);
    }
    this.toolbar.append(layoutGroup);

    const channelSelect = document.createElement('select');
    channelSelect.setAttribute('aria-label', 'Selected channel');
    for (const channel of this.channels) {
      const option = document.createElement('option');
      option.value = channel.id;
      option.textContent = String(channel.index);
      channelSelect.append(option);
    }
    channelSelect.value = this.selected;
    channelSelect.addEventListener('change', () => {
      this.selectChannel(channelSelect.value);
    });
    channelSelect.hidden = this.channels.length === 1 || this.mode !== 'single';
    this.toolbar.append(channelSelect);
    this.channelSelect = channelSelect;
    this.modeControls.push(channelSelect);

    this.buildSlidingZControls();

    if (this.roiChromeEnabled) this.buildRoiToolbar();
    this.tooltip.refresh();

  }

  /**
   * Build the top-toolbar ROI strip: dropdown + add/delete/edit/commit/cancel.
   *
   * Visibility is independent of per-pane channel toolbars but belongs to the
   * same conceptual chrome toolbar. In `delegated` mode, action buttons emit
   * request events only; in `local` mode they mutate the in-viewer ROI list.
   */
  buildRoiToolbar() {
    this.roiIdleControls = [];
    this.roiEditControls = [];
    const divider = document.createElement('div');
    divider.className = 'rv-toolbar-divider';
    divider.setAttribute('aria-hidden', 'true');
    this.roiToolbarDivider = divider;

    const strip = document.createElement('div');
    strip.className = 'rv-roi-toolbar';
    strip.hidden = !this.showRoiToolbar;
    this.roiToolbar = strip;

    const select = document.createElement('select');
    select.className = 'rv-roi-select';
    select.setAttribute('aria-label', 'Selected ROI');
    select.addEventListener('change', () => {
      const value = select.value === '' ? null : Number(select.value);
      this.selectRoi(value, {emit: true, source: 'dropdown'});
    });
    strip.append(select);
    this.roiSelect = select;
    this.roiIdleControls.push(select);

    const addButton = this.roiIconButton('plus', 'Add ROI', () => this.requestRoiAdd());
    const deleteButton = this.roiIconButton('trash-2', 'Delete ROI', () => this.requestRoiDelete());
    const editButton = this.roiIconButton('pencil', 'Edit ROI', () => this.requestRoiEdit());
    this.roiAddButton = addButton;
    this.roiDeleteButton = deleteButton;
    this.roiEditButton = editButton;
    this.roiIdleControls.push(addButton, deleteButton, editButton);
    strip.append(addButton, deleteButton, editButton);

    const commitButton = this.roiIconButton('check', 'Commit ROI edit', () => {
      this.requestRoiEditCommit();
    });
    commitButton.classList.add('rv-toolbar-icon-button--commit');
    const cancelButton = this.roiIconButton('x', 'Cancel ROI edit', () => {
      if (this.roiHostMode === 'delegated') this.requestRoiEditCancel();
      else this.cancelRoiEdit();
    });
    cancelButton.classList.add('rv-toolbar-icon-button--cancel');
    this.roiCommitButton = commitButton;
    this.roiCancelButton = cancelButton;
    this.roiEditControls.push(commitButton, cancelButton);
    strip.append(commitButton, cancelButton);

    this.toolbar.append(divider, strip);
    this.syncRoiToolbar();
    this.syncRoiToolbarDivider();
  }

  /**
   * Show a thin divider before the ROI strip when it sits beside other chrome.
   *
   * Hidden when the ROI toolbar is off, or when layout / channel select /
   * Sliding-Z are all hidden (typical one-channel toolbar).
   *
   * @returns {void}
   */
  syncRoiToolbarDivider() {
    if (!this.roiToolbarDivider) return;
    const roiVisible = Boolean(this.showRoiToolbar && this.roiToolbar && !this.roiToolbar.hidden);
    const neighborVisible = Boolean(
      (this.layoutControls && !this.layoutControls.hidden)
      || (this.channelSelect && !this.channelSelect.hidden)
      || (this.slidingZControls && !this.slidingZControls.hidden),
    );
    this.roiToolbarDivider.hidden = !(roiVisible && neighborVisible);
  }

  roiIconButton(icon, label, onClick) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'rv-toolbar-icon-button';
    button.dataset.rvTooltip = label;
    button.setAttribute('aria-label', label);
    button.append(lucideIcon(icon, label));
    button.addEventListener('click', onClick);
    return button;
  }

  /**
   * Show or hide the top-toolbar ROI CRUD strip.
   *
   * @param {boolean} visible Desired visibility.
   * @returns {boolean} Applied visibility.
   */
  setRoiToolbarVisible(visible) {
    if (!this.roiChromeEnabled) {
      this.showRoiToolbar = false;
      if (this.roiToolbar) this.roiToolbar.hidden = true;
      this.syncRoiToolbarDivider();
      return false;
    }
    this.showRoiToolbar = Boolean(visible);
    if (this.roiToolbar) this.roiToolbar.hidden = !this.showRoiToolbar;
    if (this.roiToolbarInput) this.roiToolbarInput.checked = this.showRoiToolbar;
    this.syncRoiToolbarDivider();
    return this.showRoiToolbar;
  }

  /**
   * Refresh ROI dropdown options, selection, and idle/edit button enablement.
   */
  syncRoiToolbar() {
    if (!this.roiSelect) return;
    const previous = this.roiSelect.value;
    this.roiSelect.replaceChildren();
    const empty = document.createElement('option');
    empty.value = '';
    empty.textContent = this.rois.length ? 'ROI' : 'No ROIs';
    this.roiSelect.append(empty);
    for (const roi of this.rois) {
      const option = document.createElement('option');
      option.value = String(roi.roiId);
      option.textContent = roi.name ? `${roi.roiId}: ${roi.name}` : String(roi.roiId);
      this.roiSelect.append(option);
    }
    const selected = this.selectedRoiId == null ? '' : String(this.selectedRoiId);
    this.roiSelect.value = this.rois.some(roi => String(roi.roiId) === selected)
      ? selected
      : '';
    if (this.roiSelect.value !== previous && this.roiSelect.value === '' && selected !== '') {
      this.roiSelect.value = '';
    }
    this.updateRoiToolbarEnabled();
  }

  updateRoiToolbarEnabled() {
    const editing = this.roiState !== RoiInteractionState.IDLE;
    const hasSelection = this.selectedRoiId != null;
    // Keep idle chrome visible during edit so Commit/Cancel append without reshuffling.
    for (const control of this.roiIdleControls) {
      control.hidden = false;
      if (control === this.roiDeleteButton || control === this.roiEditButton) {
        control.disabled = editing || !hasSelection;
      } else {
        control.disabled = editing;
      }
    }
    for (const control of this.roiEditControls) {
      control.hidden = !editing;
      control.disabled = !editing;
    }
  }

  /**
   * Request (or locally perform) instant rectangular ROI add.
   *
   * @returns {boolean} Whether a request was emitted or a local ROI was added.
   */
  requestRoiAdd() {
    if (!this.dataset || this.roiState !== RoiInteractionState.IDLE) return false;
    if (this.roiHostMode === 'delegated') {
      this.dispatch('raster-roi-add-request', {
        dataset_id: this.dataset.id,
        preferred_type: RoiType.RECT,
      });
      return true;
    }
    const envelope = this._localSuggestedRectEnvelope();
    this.addRoi(envelope);
    this.selectRoi(envelope.roi_id, {emit: true, source: 'toolbar'});
    return true;
  }

  /**
   * Request (or locally perform) deletion of the selected ROI.
   *
   * @returns {boolean} Whether a request was emitted or a local ROI was removed.
   */
  requestRoiDelete() {
    if (!this.dataset || this.roiState !== RoiInteractionState.IDLE || this.selectedRoiId == null) {
      return false;
    }
    const roiId = this.selectedRoiId;
    if (this.roiHostMode === 'delegated') {
      this.dispatch('raster-roi-delete-request', {
        dataset_id: this.dataset.id,
        roi_id: roiId,
      });
      return true;
    }
    const ids = this.rois.map(roi => roi.roiId);
    const index = ids.indexOf(roiId);
    this.removeRoi(roiId);
    const remaining = this.rois.map(roi => roi.roiId);
    const nextId = remaining.length
      ? remaining[Math.min(Math.max(index, 0), remaining.length - 1)]
      : null;
    this.selectRoi(nextId, {emit: true, source: 'toolbar'});
    return true;
  }

  /**
   * Request (or locally begin) editing the selected ROI.
   *
   * @returns {boolean} Whether a request was emitted or local edit started.
   */
  requestRoiEdit() {
    if (!this.dataset || this.roiState !== RoiInteractionState.IDLE || this.selectedRoiId == null) {
      return false;
    }
    const roiId = this.selectedRoiId;
    if (this.roiHostMode === 'delegated') {
      this.dispatch('raster-roi-edit-request', {
        dataset_id: this.dataset.id,
        roi_id: roiId,
      });
      return true;
    }
    return this.beginRoiEdit(roiId);
  }

  /**
   * Commit the active draft: emit a proposal in delegated mode, install locally otherwise.
   *
   * @returns {boolean} Whether commit progressed.
   */
  requestRoiEditCommit() {
    if (this.roiState === RoiInteractionState.IDLE || !this.roiDraft) return false;
    if (this.roiHostMode === 'delegated') return this.commitRoiEdit();
    if (this.roiState === RoiInteractionState.CREATING) {
      const envelope = this._localSuggestedRectEnvelope();
      if (this.roiDraft.roiType === RoiType.RECT) {
        envelope.data = {
          row_start: this.roiDraft.bounds.rowStart,
          row_stop: this.roiDraft.bounds.rowStop,
          col_start: this.roiDraft.bounds.colStart,
          col_stop: this.roiDraft.bounds.colStop,
        };
        envelope.name = this.roiDraft.name;
        envelope.note = this.roiDraft.note || '';
      }
      return this.completeRoiCommit(envelope);
    }
    return this.completeRoiCommit(roiEnvelope(this.roiDraft));
  }

  /**
   * Request host cancellation of an active create/edit draft (delegated mode).
   *
   * @returns {boolean} Whether a cancel request was emitted.
   */
  requestRoiEditCancel() {
    if (!this.dataset || this.roiState === RoiInteractionState.IDLE) return false;
    this.dispatch('raster-roi-edit-cancel-request', {
      dataset_id: this.dataset.id,
      roi_id: this.roiDraft?.roiId ?? this.selectedRoiId,
    });
    return true;
  }

  _localSuggestedRectEnvelope() {
    const width = this.dataset.width;
    const height = this.dataset.height;
    const rectWidth = Math.max(1, Math.floor(width / 4));
    const rectHeight = Math.max(1, Math.floor(height / 4));
    const colStart = Math.floor((width - rectWidth) / 2);
    const rowStart = Math.floor((height - rectHeight) / 2);
    const roiId = this.nextLocalRoiId;
    this.nextLocalRoiId += 1;
    return {
      roi_id: roiId,
      roi_type: RoiType.RECT,
      version: '1.0',
      name: String(roiId - 1),
      note: '',
      data: {
        row_start: rowStart,
        row_stop: rowStart + rectHeight,
        col_start: colStart,
        col_stop: colStart + rectWidth,
      },
    };
  }

  _refreshLocalRoiIdCounter() {
    let maximum = 0;
    for (const roi of this.rois) maximum = Math.max(maximum, Number(roi.roiId) || 0);
    this.nextLocalRoiId = maximum + 1;
  }

  buildSlidingZControls() {
    const zSize = this.dataset.header?.sizes?.Z;
    if (!Number.isInteger(zSize) || zSize < 1) return;
    const controls = document.createElement('div');
    controls.className = 'rv-sliding-z-controls';
    this.slidingZControls = controls;
    const slidingLabel = document.createElement('label');
    slidingLabel.className = 'rv-sliding-z';
    const enabled = document.createElement('input');
    enabled.type = 'checkbox';
    enabled.setAttribute('aria-label', 'Sliding-Z maximum projection');
    const radius = document.createElement('input');
    radius.type = 'number';
    radius.min = '0';
    radius.max = String(Math.min(10, zSize - 1));
    radius.step = '1';
    radius.value = String(this.savedSlidingRadius);
    radius.disabled = true;
    radius.setAttribute('aria-label', 'Sliding-Z plus or minus slices');
    enabled.addEventListener('change', () => {
      radius.disabled = !enabled.checked;
      this.plusMinusZ = enabled.checked ? Number(radius.value) : 0;
      this.requestPlaneUpdate();
    });
    radius.addEventListener('change', () => {
      const normalized = Math.max(0, Math.min(Number(radius.max), Number(radius.value)));
      radius.value = String(normalized);
      this.savedSlidingRadius = normalized;
      this.plusMinusZ = enabled.checked ? normalized : 0;
      this.requestPlaneUpdate();
    });
    slidingLabel.append(enabled, 'Sliding-Z ±', radius);
    controls.append(slidingLabel);
    this.toolbar.append(controls);
    this.modeControls.push(enabled, radius);
    this.slidingZInput = enabled;
    this.slidingZRadiusInput = radius;
  }

  createSliceControl(dimensionName) {
    const size = this.dataset.header?.sizes?.[dimensionName];
    if (!Number.isInteger(size) || size < 1) return null;
    const property = dimensionName === 'T' ? 'tIndex' : 'zIndex';
    const control = document.createElement('label');
    control.className = 'rv-slice-control';
    const dimension = document.createElement('span');
    dimension.className = 'rv-slice-dimension';
    dimension.textContent = dimensionName;
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = '0';
    slider.max = String(size - 1);
    slider.step = '1';
    slider.value = String(this[property]);
    slider.setAttribute('aria-label', `${dimensionName} plane`);
    const value = document.createElement('output');
    value.textContent = sliceValueLabel(this[property], size);
    let pointerActive = false;
    slider.addEventListener('pointerdown', () => {
      pointerActive = true;
    });
    slider.addEventListener('pointerup', () => {
      pointerActive = false;
    });
    slider.addEventListener('pointercancel', () => {
      pointerActive = false;
    });
    slider.addEventListener('input', () => {
      this[property] = Number(slider.value);
      value.textContent = sliceValueLabel(slider.value, size);
      if (pointerActive) this.schedulePlaneUpdate();
      else this.requestPlaneUpdate();
    });
    slider.addEventListener('change', () => this.requestPlaneUpdate());
    control.append(dimension, slider, value);
    this.paneControls.push(slider);
    this[`${dimensionName.toLowerCase()}Slider`] = slider;
    this[`${dimensionName.toLowerCase()}Output`] = value;
    return control;
  }

  /**
   * Select a zero-based Z plane through the normal cache/fetch pipeline.
   *
   * When the active dataset has no Z dimension this is a no-op and returns the
   * current plane selection (hosts may call it unconditionally on reconnect).
   *
   * @param {number} zIndex Requested plane index; finite values are truncated and clamped.
   * @returns {Promise<{t_index:number|null,z_index:number|null,plus_minus_z:number}>} Applied selection.
   */
  async setZIndex(zIndex) {
    const zSize = this.dataset?.header?.sizes?.Z;
    // No-op when Z is absent so hosts can restore session plane indices safely.
    if (!Number.isInteger(zSize)) return this.planeSelection();
    const normalized = Math.max(0, Math.min(zSize - 1, Math.trunc(Number(zIndex))));
    this.zIndex = normalized;
    if (this.zSlider) this.zSlider.value = String(normalized);
    if (this.zOutput) this.zOutput.textContent = sliceValueLabel(normalized, zSize);
    await this.updatePlanes();
    return this.planeSelection();
  }

  /**
   * Select a zero-based T plane through the normal cache/fetch pipeline.
   *
   * When the active dataset has no T dimension this is a no-op and returns the
   * current plane selection (hosts may call it unconditionally on reconnect).
   *
   * @param {number} tIndex Requested index; finite values are truncated and clamped.
   * @returns {Promise<{t_index:number|null,z_index:number|null,plus_minus_z:number}>}
   *   Complete applied plane selection.
   */
  async setTIndex(tIndex) {
    const tSize = this.dataset?.header?.sizes?.T;
    // No-op when T is absent so hosts can restore session plane indices safely.
    if (!Number.isInteger(tSize)) return this.planeSelection();
    const normalized = Math.max(0, Math.min(tSize - 1, Math.trunc(Number(tIndex))));
    this.tIndex = normalized;
    if (this.tSlider) this.tSlider.value = String(normalized);
    if (this.tOutput) this.tOutput.textContent = sliceValueLabel(normalized, tSize);
    await this.updatePlanes();
    return this.planeSelection();
  }

  /**
   * Move the current Z plane by one signed direction, used by Alt/Option+wheel.
   *
   * @param {number} delta Positive selects the next plane; negative selects the previous.
   * @returns {boolean} Whether the selected plane changed.
   */
  stepSlice(delta) {
    const dimensionName = Number.isInteger(this.dataset.header?.sizes?.Z) ? 'Z' : 'T';
    const size = this.dataset.header?.sizes?.[dimensionName];
    const property = dimensionName === 'Z' ? 'zIndex' : 'tIndex';
    if (!Number.isInteger(size) || this[property] === null) return false;
    const direction = Math.sign(Number(delta));
    if (direction === 0) return false;
    const nextIndex = Math.max(0, Math.min(size - 1, this[property] + direction));
    if (nextIndex === this[property]) return false;
    this[property] = nextIndex;
    const slider = this[`${dimensionName.toLowerCase()}Slider`];
    const output = this[`${dimensionName.toLowerCase()}Output`];
    if (slider) slider.value = String(nextIndex);
    if (output) output.textContent = sliceValueLabel(nextIndex, size);
    if (this.sliceWheelTimer === null) {
      this.requestPlaneUpdate();
      this.sliceWheelTimer = setTimeout(() => this.flushSliceWheel(), 75);
    } else {
      this.planeGeneration += 1;
      this.sliceWheelPending = true;
    }
    return true;
  }

  flushSliceWheel() {
    this.sliceWheelTimer = null;
    if (!this.sliceWheelPending) return;
    this.sliceWheelPending = false;
    this.requestPlaneUpdate();
  }

  /**
   * Enable or disable a backend-computed centered Sliding-Z maximum projection.
   *
   * @param {boolean} enabled Whether projection is active.
   * @param {number} [plusMinusSlices=1] Non-negative radius, clamped to the demo limit.
   * @returns {Promise<{t_index:number|null,z_index:number,plus_minus_z:number}>} Applied selection.
   * @throws {Error} If the active dataset has no Z dimension.
   */
  async setSlidingZ(enabled, plusMinusSlices = 1) {
    const zSize = this.dataset.header?.sizes?.Z;
    if (!Number.isInteger(zSize)) throw new Error('active dataset has no Z dimension');
    const maximum = Math.min(10, zSize - 1);
    const radius = Math.max(0, Math.min(maximum, Math.trunc(Number(plusMinusSlices))));
    this.savedSlidingRadius = radius;
    this.plusMinusZ = enabled ? radius : 0;
    if (this.slidingZInput) this.slidingZInput.checked = Boolean(enabled);
    if (this.slidingZRadiusInput) {
      this.slidingZRadiusInput.value = String(radius);
      this.slidingZRadiusInput.disabled = !enabled;
    }
    await this.updatePlanes();
    return this.planeSelection();
  }

  planeSelection() {
    return {
      z_index: this.zIndex,
      t_index: this.tIndex,
      plus_minus_z: this.plusMinusZ,
    };
  }

  schedulePlaneUpdate() {
    clearTimeout(this.planeTimer);
    this.planeTimer = setTimeout(() => this.requestPlaneUpdate(), 75);
  }

  async requestPlaneUpdate() {
    try {
      return await this.updatePlanes();
    } catch (error) {
      if (error?.name !== 'AbortError') {
        this.dispatch('raster-error', {
          message: error instanceof Error ? error.message : String(error),
        });
      }
      return false;
    }
  }

  async updatePlanes(options = {}) {
    clearTimeout(this.planeTimer);
    const started = performance.now();
    const generation = ++this.planeGeneration;
    const selection = this.planeSelection();
    const cacheHits = this.channels.filter(channel => this.planeCache.has(channel, selection)).length;
    const planes = await Promise.all(this.channels.map(channel => (
      this.planeCache.get(channel, selection)
    )));
    if (generation !== this.planeGeneration) return false;
    this.channels.forEach((channel, index) => {
      channel.data = planes[index];
      channel.histogram = null;
      if (options.initializeContrast && !channel.explicitRange) {
        [channel.min, channel.max] = autoRange(channel.data);
      }
    });
    if (options.render !== false) {
      const renderStarted = performance.now();
      this.redraw('user');
      const completed = performance.now();
      this.performanceMetric('plane-update', {
        ...selection,
        cache_hits: cacheHits,
        channel_count: this.channels.length,
        load_ms: renderStarted - started,
        render_ms: completed - renderStarted,
        total_ms: completed - started,
      });
      this.dispatch('raster-plane-change', {
        dataset_id: this.dataset.id,
        ...selection,
      });
    }
    return true;
  }

  visibilityControl(container, text, checked, onChange) {
    const label = document.createElement('label');
    label.className = 'rv-radio';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = checked;
    input.setAttribute('aria-label', text);
    input.addEventListener('change', () => onChange(input.checked));
    label.append(input, text);
    container.append(label);
    this.modeControls.push(input);
    return input;
  }

  /** @param {boolean} visible Desired axes visibility. @returns {boolean} Applied value. */
  setAxesVisible(visible) {
    this.showAxes = Boolean(visible);
    if (this.axesInput) this.axesInput.checked = this.showAxes;
    for (const viewport of this.viewports) {
      viewport.setAxes(this.showAxes ? this.displayAxes : null);
    }
    return this.showAxes;
  }

  /**
   * Show or hide committed ROI overlays without changing ROI data.
   *
   * @param {boolean} visible Desired visibility.
   * @returns {boolean} Applied value; active edit drafts cannot be hidden.
   */
  setRoisVisible(visible) {
    if (!this.roiChromeEnabled) {
      this.showRois = false;
      if (this.roisInput) this.roisInput.checked = false;
      this.redrawRois();
      return false;
    }
    if (!visible && this.roiState !== RoiInteractionState.IDLE) return this.showRois;
    this.showRois = Boolean(visible);
    if (this.roisInput) this.roisInput.checked = this.showRois;
    this.redrawRois();
    return this.showRois;
  }

  /**
   * Show or hide every complete pane-header toolbar.
   *
   * The toolbar includes all channel controls and Copy view. The setting is
   * retained when a layout change rebuilds the pane DOM.
   *
   * @param {boolean} visible Desired toolbar visibility.
   * @returns {boolean} Applied visibility.
   */
  setChannelToolbarsVisible(visible) {
    this.showChannelToolbars = Boolean(visible);
    if (this.channelToolbarsInput) {
      this.channelToolbarsInput.checked = this.showChannelToolbars;
    }
    for (const toolbar of this.stage.querySelectorAll('.rv-pane-header')) {
      toolbar.hidden = !this.showChannelToolbars;
    }
    return this.showChannelToolbars;
  }

  fullPhysicalXRange() {
    const step = this.displayAxes?.x?.step;
    if (!Number.isFinite(step) || step <= 0) throw new Error('X axis step must be positive');
    return {
      minimum: 0,
      maximum: this.displayWidth * step,
      label: this.displayAxes.x.label,
      unit: this.displayAxes.x.unit,
    };
  }

  /**
   * Apply one physical X range to every visible pane while preserving Y transforms.
   *
   * @param {number} minimum Requested physical lower edge.
   * @param {number} maximum Requested physical upper edge.
   * @returns {{minimum:number,maximum:number}|null} Last pane's bounded applied range.
   */
  setXRange(minimum, maximum) {
    let applied = null;
    for (const viewport of this.viewports) {
      applied = viewport.setPhysicalXRange(
        Number(minimum),
        Number(maximum),
        this.displayAxes.x.step,
      );
    }
    return applied;
  }

  /**
   * Apply one physical Y range to every visible pane while preserving X transforms.
   *
   * @param {number} minimum Requested physical lower edge.
   * @param {number} maximum Requested physical upper edge.
   * @returns {{minimum:number,maximum:number}|null} Last pane's bounded applied range.
   */
  setYRange(minimum, maximum) {
    let applied = null;
    for (const viewport of this.viewports) {
      applied = viewport.setPhysicalYRange(
        Number(minimum),
        Number(maximum),
        this.displayAxes.y.step,
      );
    }
    return applied;
  }

  /**
   * Apply physical X and Y ranges to every pane in one transform update.
   *
   * @param {number} xMinimum Requested physical X lower edge.
   * @param {number} xMaximum Requested physical X upper edge.
   * @param {number} yMinimum Requested physical Y lower edge.
   * @param {number} yMaximum Requested physical Y upper edge.
   * @returns {{x:{minimum:number,maximum:number},y:{minimum:number,maximum:number}}|null}
   *     Last pane's bounded applied ranges.
   */
  setPhysicalRange(xMinimum, xMaximum, yMinimum, yMaximum) {
    let applied = null;
    for (const viewport of this.viewports) {
      applied = viewport.setPhysicalRange(
        Number(xMinimum),
        Number(xMaximum),
        this.displayAxes.x.step,
        Number(yMinimum),
        Number(yMaximum),
        this.displayAxes.y.step,
      );
    }
    return applied;
  }

  /**
   * Return the full physical Y extent of the loaded image.
   *
   * @returns {{minimum:number,maximum:number,label:string,unit:string}|null}
   */
  fullPhysicalYRange() {
    if (!this.dataset) return null;
    const step = this.displayAxes.y.step;
    if (!Number.isFinite(step) || step <= 0) throw new Error('Y axis step must be positive');
    return {
      minimum: 0,
      maximum: this.displayHeight * step,
      label: this.displayAxes.y.label,
      unit: this.displayAxes.y.unit,
    };
  }

  /**
   * Restore the full X and Y image extent in every pane.
   *
   * @returns {boolean} True after all current panes are reset.
   */
  resetView() {
    this.viewports.forEach(viewport => {
      viewport.reset();
      viewport.emit('api-reset', true);
    });
    return true;
  }

  /**
   * Restore only the full physical X extent while preserving Y.
   *
   * @returns {{minimum:number,maximum:number}|null} Applied physical X range.
   */
  resetXRange() {
    const full = this.fullPhysicalXRange();
    return this.setXRange(full.minimum, full.maximum);
  }

  /**
   * Update runtime calibration without reloading or invalidating pixel planes.
   *
   * @param {number[]} physicalUnits Positive spacing aligned with descriptor dims.
   * @param {string[]} physicalUnitsLabels Labels aligned with descriptor dims.
   * @returns {{physical_units:number[],physical_units_labels:string[]}} Applied calibration.
   * @throws {Error} If lengths differ from dims or a spacing is invalid.
   */
  setPhysicalCalibration(physicalUnits, physicalUnitsLabels) {
    const dims = this.dataset?.header?.dims;
    if (!Array.isArray(dims) || physicalUnits.length !== dims.length
      || physicalUnitsLabels.length !== dims.length) {
      throw new Error('physical calibration must match active dataset dimensions');
    }
    const units = physicalUnits.map(Number);
    if (units.some(value => !Number.isFinite(value) || value <= 0)) {
      throw new Error('physical calibration values must be finite and positive');
    }
    this.dataset.header.physical_units = units;
    this.dataset.header.physical_units_labels = physicalUnitsLabels.map(String);
    const yIndex = dims.indexOf('Y');
    const xIndex = dims.indexOf('X');
    this.dataset.axes = {
      x: {label: String(physicalUnitsLabels[xIndex]), step: units[xIndex], unit: ''},
      y: {label: String(physicalUnitsLabels[yIndex]), step: units[yIndex], unit: ''},
    };
    this.displayAxes = transposedAxes(this.dataset.axes);
    for (const viewport of this.viewports) {
      viewport.setImage(renderBitmap(this.displayWidth, this.displayHeight, viewport.group),
        this.showAxes ? this.displayAxes : null, false);
    }
    return {physical_units: units, physical_units_labels: physicalUnitsLabels.map(String)};
  }

  /**
   * Select the active logical channel.
   *
   * @param {string} channelId Dataset-local channel identity.
   * @param {boolean} [notify=true] Whether to emit a user-facing selection event.
   * @returns {string} Applied channel identity.
   * @throws {Error} If the channel does not exist.
   */
  selectChannel(channelId, notify = true) {
    const channel = this.channels.find(candidate => candidate.id === String(channelId));
    if (!channel) throw new Error(`unknown channel: ${channelId}`);
    // Re-selecting the same channel must not rebuild panes: hosts sync channel
    // after ROI selection, and render() → setImage(reset) would wipe zoom.
    const changed = this.selected !== channel.id;
    this.selected = channel.id;
    if (this.channelSelect) this.channelSelect.value = channel.id;
    if (changed && this.mode === 'single') this.render();
    if (notify) this.dispatch('raster-channel-selected', {
      dataset_id: this.dataset.id,
      channel_id: channel.id,
    });
    return channel.id;
  }

  /**
   * Apply a partial display update to one channel without reloading pixels.
   *
   * Null contrast limits preserve the current range; non-null limits must be
   * supplied together.
   *
   * @param {string} channelId Dataset-local channel identity.
   * @param {{lut?:string,value_min?:number|null,value_max?:number|null,visible?:boolean}} display
   *   Desired partial presentation state.
   * @returns {{channel_id:string,lut:string,value_min:number,value_max:number,visible:boolean}}
   *   Applied complete display snapshot.
   * @throws {Error} If the channel or contrast range is invalid.
   */
  setChannelDisplay(channelId, display) {
    const channel = this.channels.find(candidate => candidate.id === String(channelId));
    if (!channel) throw new Error(`unknown channel: ${channelId}`);
    if (display.lut !== undefined) channel.lut = String(display.lut);
    if (display.visible !== undefined) channel.enabled = Boolean(display.visible);
    if (display.value_min != null || display.value_max != null) {
      if (display.value_min == null || display.value_max == null) {
        throw new Error('channel display limits must be supplied together');
      }
      const minimum = Number(display.value_min);
      const maximum = Number(display.value_max);
      if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum >= maximum) {
        throw new Error('channel display range must be finite and increasing');
      }
      channel.min = minimum;
      channel.max = maximum;
      channel.explicitRange = true;
    }
    this.render();
    this.redraw('programmatic');
    return this.channelDisplaySnapshot(channel);
  }

  channelDisplaySnapshot(channel) {
    return {
      channel_id: channel.id,
      lut: channel.lut,
      value_min: channel.min,
      value_max: channel.max,
      visible: channel.enabled,
    };
  }

  createChannelControl(channel) {
    const row = document.createElement('div');
    row.className = 'rv-channel-control';

    const name = document.createElement('span');
    name.className = 'rv-channel-name';
    name.textContent = String(channel.index);
    name.setAttribute('role', 'button');
    name.tabIndex = 0;
    const select = () => {
      if (this.selected !== channel.id) this.selectChannel(channel.id);
    };
    name.addEventListener('click', select);
    name.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') select();
    });

    const enabled = document.createElement('input');
    enabled.type = 'checkbox';
    enabled.checked = channel.enabled;
    enabled.setAttribute('aria-label', `${channel.label} visible`);
    enabled.addEventListener('change', () => {
      channel.enabled = enabled.checked;
      this.redraw('user');
    });

    const lut = document.createElement('select');
    lut.setAttribute('aria-label', `${channel.label} color LUT`);
    for (const [value, text] of Object.entries(LUT_LABELS)) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = text;
      lut.append(option);
    }
    lut.value = channel.lut;
    lut.addEventListener('change', () => {
      channel.lut = lut.value;
      this.redraw('user');
    });

    const rangeButton = document.createElement('button');
    rangeButton.type = 'button';
    rangeButton.className = 'rv-range-button';
    rangeButton.dataset.rvTooltip = 'Set contrast';
    rangeButton.setAttribute('aria-label', `${channel.label} set contrast`);
    rangeButton.append(lucideIcon('chart-column-decreasing', 'Set contrast'));
    rangeButton.addEventListener('click', () => {
      this.rangePopover.toggle(channel, rangeButton);
    });

    row.append(name, enabled, lut, rangeButton);
    this.paneControls.push(enabled, lut, rangeButton);
    return row;
  }

  destroyViews() {
    this.viewports.forEach(viewport => viewport.destroy());
    this.viewports = [];
    this.stage.replaceChildren();
  }

  /**
   * Snapshot pan/zoom from the first pane for layout rebuilds.
   *
   * Hotkeys and toolbar layout changes call ``render()``, which destroys
   * canvases. Without a snapshot, ``addPane`` → ``setImage(..., true)`` would
   * ``fit()`` and wipe the user's zoom.
   *
   * @returns {{scaleX:number,scaleY:number,offsetX:number,offsetY:number,home:object|null}|null}
   *   Transform to restore, or ``null`` when there is no prior pane (first paint).
   */
  captureViewportTransform() {
    const viewport = this.viewports[0];
    if (!viewport?.bitmap) return null;
    return {
      scaleX: viewport.scaleX,
      scaleY: viewport.scaleY,
      offsetX: viewport.offsetX,
      offsetY: viewport.offsetY,
      home: viewport.home ? {...viewport.home} : null,
    };
  }

  /**
   * Apply a captured transform to every current pane and redraw.
   *
   * @param {{scaleX:number,scaleY:number,offsetX:number,offsetY:number,home:object|null}} transform
   *   Snapshot from ``captureViewportTransform``.
   * @returns {void}
   */
  restoreViewportTransform(transform) {
    for (const viewport of this.viewports) {
      viewport.scaleX = transform.scaleX;
      viewport.scaleY = transform.scaleY;
      viewport.offsetX = transform.offsetX;
      viewport.offsetY = transform.offsetY;
      viewport.home = transform.home ? {...transform.home} : viewport.home;
      viewport.draw();
    }
  }

  /**
   * Replace all committed ROIs from an authoritative caller snapshot.
   *
   * This method is silent and never emits a user-originated ROI callback.
   *
   * @param {RoiEnvelope[]} envelopes Complete committed collection.
   * @returns {number} Number of installed ROIs.
   */
  setRois(envelopes) {
    if (!this.dataset) return 0;
    this.rois = envelopes.map(envelope => roiFromEnvelope(
      envelope,
      this.dataset.width,
      this.dataset.height,
    ));
    if (!this.rois.some(roi => roi.roiId === this.selectedRoiId)) {
      this.selectedRoiId = this.rois[0]?.roiId ?? null;
    }
    this._refreshLocalRoiIdCounter();
    this.syncRoiToolbar();
    this.redrawRois();
    return this.rois.length;
  }

  _installRoi(envelope) {
    const roi = roiFromEnvelope(envelope, this.dataset.width, this.dataset.height);
    const index = this.rois.findIndex(item => item.roiId === roi.roiId);
    if (index >= 0) this.rois[index] = roi;
    else this.rois.push(roi);
    this._refreshLocalRoiIdCounter();
    this.syncRoiToolbar();
    this.redrawRois();
    return true;
  }

  /**
   * Add one committed ROI from an authoritative caller without emitting an event.
   *
   * @param {RoiEnvelope} envelope Complete committed ROI.
   * @returns {boolean} True after successful validation and installation.
   * @throws {Error} If the ROI ID already exists.
   */
  addRoi(envelope) {
    const roiId = Number(envelope.roi_id);
    if (this.rois.some(item => item.roiId === roiId)) {
      throw new Error(`ROI ${envelope.roi_id} already exists`);
    }
    return this._installRoi(envelope);
  }

  /**
   * Replace one committed ROI without emitting a user-originated event.
   *
   * @param {RoiEnvelope} envelope Complete committed ROI.
   * @returns {boolean} True after successful validation and replacement.
   * @throws {Error} If the ROI ID does not exist.
   */
  updateRoi(envelope) {
    const roiId = Number(envelope.roi_id);
    if (!this.rois.some(item => item.roiId === roiId)) {
      throw new Error(`ROI ${envelope.roi_id} does not exist`);
    }
    return this._installRoi(envelope);
  }

  /**
   * Remove one committed ROI silently.
   *
   * @param {number} roiId Stable ROI identity.
   * @returns {boolean} Whether a matching ROI was removed.
   */
  removeRoi(roiId) {
    const previousLength = this.rois.length;
    this.rois = this.rois.filter(roi => roi.roiId !== Number(roiId));
    if (this.selectedRoiId === Number(roiId)) this.selectedRoiId = null;
    this.syncRoiToolbar();
    this.redrawRois();
    return this.rois.length !== previousLength;
  }

  /**
   * Select a committed ROI or clear selection.
   *
   * Caller synchronization is silent by default. Set `options.emit` only for a
   * genuine user-originated interaction.
   *
   * @param {number|null} roiId Stable ROI identity, or null to clear selection.
   * @param {{emit?:boolean,source?:string}} [options] Optional event behavior.
   * @returns {boolean} Whether the requested selection was valid and applied.
   */
  selectRoi(roiId, options = {}) {
    const normalized = roiId === null ? null : Number(roiId);
    if (normalized !== null && !this.rois.some(roi => roi.roiId === normalized)) return false;
    this.selectedRoiId = normalized;
    this.syncRoiToolbar();
    this.redrawRois();
    if (options.emit) {
      this.dispatch('raster-roi-select', {
        dataset_id: this.dataset.id,
        roi_id: normalized,
        source: options.source || 'pointer',
      });
    }
    return true;
  }

  /**
   * Start an uncommitted rectangle or line-segment creation draft.
   *
   * @param {RoiCreateSpecification} specification Initial draft metadata and bounds.
   * @returns {boolean} False when another ROI interaction is already active.
   */
  beginRoiCreate(specification) {
    if (this.roiState !== RoiInteractionState.IDLE) return false;
    const roiType = String(specification.roi_type);
    if (![RoiType.RECT, RoiType.LINE].includes(roiType)) {
      throw new Error(`unsupported ROI creation type: ${roiType}`);
    }
    this.roiState = RoiInteractionState.CREATING;
    this.roiDraft = {
      roiId: null,
      roiType,
      version: '1.0',
      name: String(specification.name),
      note: String(specification.note || ''),
    };
    if (roiType === RoiType.RECT) {
      this.roiDraft.bounds = normalizedBounds(
        specification.data, this.dataset.width, this.dataset.height, true,
      );
    } else {
      this.roiDraft.endpoints = normalizedEndpoints(
        specification.data, this.dataset.width, this.dataset.height, true,
      );
    }
    this.activeEditOverlay = this.viewports[0]?.overlay || null;
    this.setRoiControlsDisabled(true);
    this.emitRoiState();
    this.redrawRois();
    return true;
  }

  /**
   * Start transactional editing of one committed ROI.
   *
   * @param {number} roiId Stable committed identity.
   * @returns {boolean} Whether editing started.
   */
  beginRoiEdit(roiId) {
    if (this.roiState !== RoiInteractionState.IDLE) return false;
    const roi = this.rois.find(item => item.roiId === Number(roiId));
    if (!roi) return false;
    this.selectedRoiId = roi.roiId;
    this.roiState = RoiInteractionState.EDITING;
    this.roiDraft = this.viewports[0]?.overlay?.cloneRoi(roi)
      || (roi.roiType === RoiType.RECT
        ? {...roi, bounds: {...roi.bounds}}
        : {...roi, endpoints: {...roi.endpoints}});
    this.activeEditOverlay = this.viewports[0]?.overlay || null;
    this.setRoiControlsDisabled(true);
    this.emitRoiState();
    this.syncRoiToolbar();
    this.redrawRois();
    return true;
  }

  /**
   * Emit one creation/edit proposal for the active draft.
   *
   * The draft remains active until an authoritative caller invokes
   * `completeRoiCommit` or cancellation.
   *
   * @returns {boolean} Whether a proposal was emitted.
   */
  commitRoiEdit() {
    if (!this.roiDraft || this.roiState === RoiInteractionState.IDLE) return false;
    if (this.roiDraft.roiType === RoiType.RECT) {
      this.roiDraft.bounds = normalizedBounds(
        this.roiDraft.bounds, this.dataset.width, this.dataset.height, true,
      );
    } else {
      this.roiDraft.endpoints = normalizedEndpoints(
        this.roiDraft.endpoints, this.dataset.width, this.dataset.height, true,
      );
    }
    if (this.roiState === RoiInteractionState.CREATING) {
      this.dispatch('raster-roi-create', {
        dataset_id: this.dataset.id,
        name: this.roiDraft.name,
        note: this.roiDraft.note,
        roi_type: this.roiDraft.roiType,
        data: roiEnvelope({...this.roiDraft, roiId: 0}).data,
      });
    } else {
      this.dispatch('raster-roi-edit-commit', {
        dataset_id: this.dataset.id,
        roi: roiEnvelope(this.roiDraft),
      });
    }
    return true;
  }

  /**
   * Install a canonical ROI returned by the authoritative owner and end editing.
   *
   * @param {RoiEnvelope} envelope Validated committed ROI.
   * @returns {boolean} Always true after installation.
   */
  completeRoiCommit(envelope) {
    this._installRoi(envelope);
    this.selectedRoiId = Number(envelope.roi_id);
    this.finishRoiInteraction();
    return true;
  }

  /** @returns {boolean} Whether an active ROI draft was cancelled. */
  cancelRoiEdit() {
    if (this.roiState === RoiInteractionState.IDLE) return false;
    this.finishRoiInteraction();
    return true;
  }

  finishRoiInteraction() {
    this.roiState = RoiInteractionState.IDLE;
    this.roiDraft = null;
    this.activeEditOverlay = null;
    this.setRoiControlsDisabled(false);
    this.emitRoiState();
    this.syncRoiToolbar();
    this.redrawRois();
  }

  setRoiControlsDisabled(disabled) {
    for (const control of this.modeControls) control.disabled = disabled;
    for (const control of this.paneControls) control.disabled = disabled;
    if (!disabled && this.slidingZRadiusInput) {
      this.slidingZRadiusInput.disabled = !this.slidingZInput.checked;
    }
    this.updateRoiToolbarEnabled();
    this.rangePopover.close();
  }

  emitRoiState() {
    this.dispatch('raster-roi-state-change', {
      dataset_id: this.dataset.id,
      state: this.roiState,
      roi_id: this.roiDraft?.roiId ?? null,
    });
  }

  visibleRois() {
    if (!this.roiDraft) return this.rois;
    if (this.roiState === RoiInteractionState.CREATING) return [...this.rois, this.roiDraft];
    return this.rois.map(roi => roi.roiId === this.roiDraft.roiId ? this.roiDraft : roi);
  }

  redrawRois() {
    for (const viewport of this.viewports) viewport.overlay?.draw();
  }

  /**
   * Add one non-interactive physical X/Y plot.
   *
   * @param {XYPlotSpecification} specification Complete plot with a unique `plot_id`.
   * @returns {string} Installed plot ID.
   * @throws {Error} If validation fails or the ID already exists.
   */
  addXYPlot(specification) {
    const plot = normalizeXYPlot(specification);
    if (this.xyPlots.has(plot.plotId)) throw new Error(`XY plot ${plot.plotId} already exists`);
    this.xyPlots.set(plot.plotId, plot);
    this.redrawXYPlots();
    return plot.plotId;
  }

  /**
   * Replace one existing X/Y plot while retaining its stable ID.
   *
   * @param {XYPlotSpecification} specification Complete replacement plot specification.
   * @returns {boolean} True after replacement.
   * @throws {Error} If validation fails or the ID does not exist.
   */
  updateXYPlot(specification) {
    const plot = normalizeXYPlot(specification);
    if (!this.xyPlots.has(plot.plotId)) {
      throw new Error(`XY plot ${plot.plotId} does not exist`);
    }
    this.xyPlots.set(plot.plotId, plot);
    this.redrawXYPlots();
    return true;
  }

  removeXYPlot(plotId) {
    const removed = this.xyPlots.delete(String(plotId));
    if (removed) this.redrawXYPlots();
    return removed;
  }

  /** Show an existing plot without changing its data. */
  showXYPlot(plotId) {
    return this.setXYPlotVisible(plotId, true);
  }

  /** Hide an existing plot without removing it. */
  hideXYPlot(plotId) {
    return this.setXYPlotVisible(plotId, false);
  }

  setXYPlotVisible(plotId, visible) {
    const normalizedId = String(plotId);
    const plot = this.xyPlots.get(normalizedId);
    if (!plot) return false;
    if (plot.visible !== visible) {
      this.xyPlots.set(normalizedId, {...plot, visible});
      this.redrawXYPlots();
    }
    return true;
  }

  redrawXYPlots() {
    for (const viewport of this.viewports) viewport.plotOverlay?.draw();
  }

  render() {
    // Preserve pan/zoom across layout/channel pane rebuilds (1/2/3 hotkeys,
    // toolbar radios, selectChannel in single mode). First paint has no
    // snapshot and keeps addPane's fit() behavior.
    const transform = this.captureViewportTransform();
    this.destroyViews();
    this.rangePopover.close();
    this.paneControls = [];
    this.tSlider = null;
    this.tOutput = null;
    this.zSlider = null;
    this.zOutput = null;
    this.stage.className = `rv-stage rv-${this.mode}`;
    let groups;
    if (this.mode === 'composite') {
      groups = [this.channels];
    } else if (this.mode === 'single') {
      const selected = this.channels.find(channel => channel.id === this.selected);
      groups = [[selected || this.channels[0]]];
    } else {
      groups = this.channels.map(channel => [channel]);
    }
    const slicePaneIndex = sliceControlPaneIndex(this.mode, groups.length);
    groups.forEach((group, index) => {
      const showSliceControl = index === slicePaneIndex;
      this.addPane(group, showSliceControl, transform == null);
    });
    if (transform) this.restoreViewportTransform(transform);
  }

  redraw(cause = 'render') {
    for (const viewport of this.viewports) {
      const image = renderBitmap(this.displayWidth, this.displayHeight, viewport.group);
      viewport.setImage(image, this.showAxes ? this.displayAxes : null, false);
    }
    this.dispatch('raster-display-change', {
      dataset_id: this.dataset.id,
      cause,
      channels: this.channels.map(channel => ({
        channel_id: channel.id,
        visible: channel.enabled,
        lut: channel.lut,
        value_min: channel.min,
        value_max: channel.max,
        opacity: channel.opacity,
      })),
    });
  }

  addPane(group, showSliceControl = false, resetView = true) {
    const pane = document.createElement('section');
    pane.className = 'rv-pane';
    const header = document.createElement('div');
    header.className = 'rv-pane-header';
    header.hidden = !this.showChannelToolbars;
    const channelControls = document.createElement('div');
    channelControls.className = 'rv-pane-channel-controls';
    for (const channel of group) channelControls.append(this.createChannelControl(channel));
    const copyButton = document.createElement('button');
    copyButton.type = 'button';
    copyButton.className = 'rv-copy-button';
    copyButton.dataset.rvTooltip = 'Copy view to clipboard';
    copyButton.setAttribute('aria-label', 'Copy view to clipboard');
    copyButton.append(lucideIcon('copy', 'Copy view to clipboard'));
    copyButton.disabled = !clipboardCopyAvailable(this.hostClipboardBridge);
    if (copyButton.disabled) {
      copyButton.dataset.rvTooltip = 'Image clipboard access is unavailable in this browser context';
    }
    header.append(channelControls, copyButton);
    const body = document.createElement('div');
    body.className = 'rv-pane-body';
    const wrap = document.createElement('div');
    wrap.className = 'rv-canvas-wrap';
    const canvas = document.createElement('canvas');
    canvas.className = 'rv-raster-canvas';
    const plotCanvas = document.createElement('canvas');
    plotCanvas.className = 'rv-xy-plot-canvas';
    const overlayCanvas = document.createElement('canvas');
    overlayCanvas.className = 'rv-roi-canvas';
    wrap.append(canvas, plotCanvas, overlayCanvas);
    body.append(wrap);
    if (showSliceControl) {
      for (const dimensionName of this.dataset.header.dims.filter(dim => dim === 'T' || dim === 'Z')) {
        const sliceControl = this.createSliceControl(dimensionName);
        if (sliceControl) body.append(sliceControl);
      }
    }
    this.toolbar.hidden = false;
    pane.append(header, body);
    this.stage.append(pane);
    const viewport = new RasterViewport(canvas, detail => this.dispatch(
      'raster-view-change',
      {
        dataset_id: this.dataset.id,
        channels: group.map(channel => channel.id),
        ...detail,
        physical_range: {
          x: {
            minimum: detail.image_range.x[0] * this.displayAxes.x.step,
            maximum: detail.image_range.x[1] * this.displayAxes.x.step,
            label: this.displayAxes.x.label,
            unit: this.displayAxes.x.unit,
          },
          y: {
            minimum: detail.image_range.y[0] * this.displayAxes.y.step,
            maximum: detail.image_range.y[1] * this.displayAxes.y.step,
            label: this.displayAxes.y.label,
            unit: this.displayAxes.y.unit,
          },
        },
      },
    ), overlayCanvas, this.zIndex === null && this.tIndex === null ? null : delta => this.stepSlice(
      configuredSliceStep(delta, this.invertSliceWheel),
    ), this.wheelZoomFactor);
    viewport.setTheme(this.chromeTheme, false);
    copyButton.addEventListener('click', async () => {
      copyButton.disabled = true;
      try {
        const channelIds = group.map(channel => channel.id);
        if (clipboardImageSupported()) {
          await copyViewportToClipboard(viewport);
        } else if (this.hostClipboardBridge) {
          const pngDataUrl = await composeViewportPngDataUrl(viewport);
          this.dispatch('raster-copy-view-request', {
            dataset_id: this.dataset.id,
            png_data_url: pngDataUrl,
            channels: channelIds,
          });
        } else {
          throw new Error('Image clipboard access is unavailable in this browser context');
        }
        copyButton.replaceChildren(lucideIcon('check', 'Copied'));
        copyButton.dataset.rvTooltip = 'Copied';
        this.toolbarAction('copy-view', {channels: channelIds});
        setTimeout(() => {
          copyButton.replaceChildren(lucideIcon('copy', 'Copy view to clipboard'));
          copyButton.dataset.rvTooltip = 'Copy view to clipboard';
          copyButton.disabled = !clipboardCopyAvailable(this.hostClipboardBridge);
        }, 1000);
      } catch (error) {
        copyButton.disabled = !clipboardCopyAvailable(this.hostClipboardBridge);
        this.dispatch('raster-error', {
          message: error instanceof Error ? error.message : String(error),
        });
      }
    });
    viewport.group = group;
    this.viewports.push(viewport);
    const image = renderBitmap(this.displayWidth, this.displayHeight, group);
    // ``resetView`` is false when ``render()`` will restore a captured transform.
    viewport.setImage(image, this.showAxes ? this.displayAxes : null, resetView);
    const plotOverlay = new XYPlotOverlay(
      plotCanvas,
      viewport,
      this,
      group.map(channel => channel.id),
    );
    viewport.setPlotOverlay(plotOverlay);
    const overlay = new RoiOverlay(overlayCanvas, viewport, this);
    viewport.setOverlay(overlay);
    this.tooltip.refresh();
  }

  dispatch(name, detail) {
    this.host.dispatchEvent(new CustomEvent(name, {bubbles: true, detail}));
  }

  toolbarAction(action, detail) {
    this.dispatch('raster-toolbar-action', {dataset_id: this.dataset.id, action, ...detail});
  }

  performanceMetric(phase, detail) {
    const metric = {dataset_id: this.dataset?.id ?? null, phase, ...detail};
    console.debug('[RasterViewer performance]', metric);
    this.dispatch('raster-performance', metric);
  }

  /**
   * Clear the active dataset and all dataset-scoped browser state.
   *
   * In-flight requests are aborted; planes, ROIs, plots, selection, and pane
   * DOM are released while the reusable viewer shell remains mounted.
   *
   * @returns {boolean} Always true after the viewer becomes empty.
   */
  clear() {
    clearTimeout(this.planeTimer);
    clearTimeout(this.sliceWheelTimer);
    this.planeGeneration += 1;
    this.planeCache?.clear();
    this.planeCache = null;
    this.rangePopover.close();
    this.xyPlots.clear();
    this.rois = [];
    this.channels = [];
    this.dataset = null;
    this.tIndex = null;
    this.zIndex = null;
    this.plusMinusZ = 0;
    this.selected = '';
    this.destroyViews();
    this.toolbar.replaceChildren();
    return true;
  }

  /** Release observers, listeners, cached requests, canvases, and host contents. */
  destroy() {
    clearTimeout(this.planeTimer);
    clearTimeout(this.sliceWheelTimer);
    this.planeCache?.clear();
    this.destroyViews();
    this.rangePopover.destroy();
    this.tooltip.destroy();
    document.removeEventListener('keydown', this.keyHandler);
    document.removeEventListener('pointerdown', this.pointerHandler);
    this.host.removeEventListener('pointerdown', this.hostPointerHandler);
    if (keyboardActiveViewer === this) keyboardActiveViewer = null;
    this.host.replaceChildren();
    this.host.classList.remove('rv-root');
  }
}
