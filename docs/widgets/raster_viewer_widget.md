# RasterViewerWidget

`RasterViewerWidget` is an instance-scoped NiceGUI component backed by a
framework-independent JavaScript canvas viewer. It supports uint16 and float32
data, multiple channels, T/Z navigation, Sliding-Z maximum projections,
physical axes, typed rectangle and line ROIs, and non-interactive X/Y overlays.

The existing [`PlotlyRasterViewer`](raster_viewer.md) remains a separate widget.

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
