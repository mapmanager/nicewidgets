# Examples

Runnable demos live under `examples/` in the repository. They are **not** part
of the installable wheel.

## Table widget

```bash
uv run python examples/table_widget/demo_app.py
```

## Raster viewer

```bash
uv run python examples/raster_viewer/nicegui_raster_demo.py
uv run python examples/raster_viewer/nicegui_y_axis_demo.py
```

The main raster demo combines `PlotlyRasterViewer` with `ImageToolbarWidget`
(channel select, ROI add/delete/edit) and `ContrastWidget` (color LUT, Auto,
intensity range). It is organized as a template for host applications:

- `sample_data.py`: synthetic multi-channel datasets (pure data, no NiceGUI)
- `demo_controller.py`: state plus intent-to-viewer wiring
- `nicegui_raster_demo.py`: thin page layout and demo-only controls

## Scripts

Focused manual diagnostics live under `scripts/`:

```bash
uv run python scripts/try_plotly_plot_widget.py
uv run python scripts/try_plotly_plot_event_overlays.py
```
