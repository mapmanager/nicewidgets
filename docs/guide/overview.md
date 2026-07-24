# Package overview

NiceWidgets owns reusable UI behavior built on NiceGUI. Applications own
orchestration and domain logic.

Design rule:

```text
nicewidgets owns reusable UI behavior.
applications own orchestration and domain interpretation.
```

## Main areas

| Area | Role |
|------|------|
| `raster_viewer` | Plotly multiresolution raster viewer, ROI and trace overlays |
| `plotly_plot` | General Plotly line/measurement plot widget |
| `echart_widget` | ECharts line/plot widget |
| `table_widget` / `tree_widget` | AG Grid table and tree wrappers |
| `nicepool` | DataFrame-driven plot pool UI |
| `image_toolbar_widget` / `contrast_widget` | Image toolbar and contrast controls |
| `upload_widget` | File upload normalization |
| `utils` | Logging, clipboard, desktop detection helpers |

## Import style

```python
from nicewidgets.table_widget.table_widget import TableWidget
from nicewidgets.raster_viewer.frontend.plotly_viewer import PlotlyRasterViewer
```

Do not rely on package-level re-exports.

## Optional desktop extras

Native clipboard and pywebview desktop detection tolerate missing optional
packages. Install them when building desktop apps:

```bash
uv sync --extra desktop
```

That extra group includes `pyperclip`, `pyperclipimg`, and `pywebview`.
