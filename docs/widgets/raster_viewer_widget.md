# RasterViewerWidget

`RasterViewerWidget` is an instance-scoped NiceGUI component backed by a
framework-independent JavaScript canvas viewer. It supports uint16 and float32
data, multiple channels, T/Z navigation, Sliding-Z maximum projections,
physical axes, typed rectangle and line ROIs, and non-interactive X/Y overlays.

The existing [`PlotlyRasterViewer`](raster_viewer.md) remains a separate widget.

## Chrome notes

Three chrome layers (names match Viewer options toggles where applicable):

1. **Viewer options** — hamburger menu, pinned leftmost on the top toolbar.
   Toggles Axes / ROIs / Channel Toolbars / ROI Toolbar and Reset view.
2. **Top toolbar (rest)** — channel layout radios (hidden for one channel),
   channel select (single mode), Sliding-Z, then a thin divider (when those
   neighbors are visible) and the **ROI Toolbar** strip
   (add/delete/edit/commit/cancel) in normal left-to-right flow.
3. **Channel Toolbars** — per-pane header rows (channel index, enable, LUT,
   Set contrast) plus **Copy view to clipboard**; shown/hidden together via
   the Channel Toolbars option. Copy uses the browser Clipboard API on web,
   and on NiceGUI native/pywebview composes a PNG in JS then writes via
   ``nicewidgets.utils.clipboard.copy_png_bytes_to_native_clipboard``
   (requires the optional desktop extras: ``pyperclipimg`` + Pillow).

Also:

- ROI edit **Commit** / **Cancel** use muted green / red icon styling.
- **Set contrast** opens a histogram dialog on `document.body` (so it stacks
  above sibling viewers). A **Log** checkbox (default on) switches histogram
  Y scaling between log and linear.
- After a pointer click/press inside a viewer (ignored while typing in inputs):
  with **two or more channels**, keys ``1`` / ``2`` switch to one-channel view
  for channel 0 / 1, and ``3`` switches to composite; **Enter** runs
  Viewer options → Reset view (same as the menu action).

## Embed

```python
import numpy as np

from nicewidgets.raster_viewer_widget import RasterViewerWidget

image = np.arange(20 * 512 * 512, dtype=np.uint16).reshape(20, 512, 512)
viewer = RasterViewerWidget.from_array(
    image,
    dims=('Z', 'Y', 'X'),
    physical_units=(0.75, 0.4, 0.2),
    physical_units_labels=('um', 'um', 'um'),
)
viewer.classes('w-full h-[70vh]')
```

Source arrays remain in NumPy row-major coordinates. Browser display applies
the widget's defined transpose and display Y-flip. Dataset descriptors use
snake_case and exclude the channel axis from `shape` and `dims`.

## Public API

Import public models and the widget from `nicewidgets.raster_viewer_widget`.
Namespaced runtime APIs are available through `viewer.channels`,
`viewer.rois`, and `viewer.xy_plots`.

::: nicewidgets.raster_viewer_widget.RasterViewerWidget
    options:
      show_root_heading: true
      heading_level: 3

## Example

```bash
uv run python -m examples.raster_viewer_widget.main
```
