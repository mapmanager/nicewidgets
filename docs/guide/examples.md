# Examples

Runnable demos live under `examples/` in the repository. They are **not** part
of the installable wheel.

## Combined demo app

Run all three product demos from one browser-based app:

```bash
uv run python -m examples.main_demo.main
```

Open [http://localhost:8080](http://localhost:8080). The home page links to
independent `/raster`, `/table`, and `/nicepool` routes. Separate routes keep
the Plotly and AG Grid demos isolated and provide stable deep links.

To run the same app with Docker Compose:

```bash
docker compose -f examples/main_demo/docker-compose.yml up --build
```

The combined app imports each demo's reusable `page.py` builder. The standalone
entry points below call the same builders, so their content does not diverge.

## Table widget

```bash
uv run python examples/table_widget/demo_app.py
```

The demo shows the optional 1-based index column, AG Grid Enterprise category
grouping, row selection, editable cells, and a custom context menu. Enterprise
grouping requires a host-provided AG Grid license in production. Reusable page
composition and sample rows live in `examples/table_widget/page.py`.

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
- `page.py`: reusable page layout and demo-only controls
- `nicegui_raster_demo.py`: standalone route and `ui.run` entry point

## NicePool

```bash
uv run python examples/nicepool/demo_app.py
```

The demo drives `NicePool` with a synthetic velocity-pool `DataFrame` (one row
per file/channel/ROI). It shows pre-filter dropdowns, plot-type controls, named
presets, the optional data table, and linked selection between the table and
plots. Demo-only controls cover dataset switching, programmatic selection
(`select_points_by_row_ids`), and a dark-mode toggle. It follows the host
application template:

- `sample_data.py`: synthetic pool rows as `list[dict]` -> `DataFrame`, using the
  `acqstore` velocity-pool schema (pure data, no NiceGUI)
- `demo_controller.py`: widget config, callbacks, and public-API wiring
- `page.py`: reusable page layout and demo-only controls
- `demo_app.py`: standalone route and `ui.run` entry point

The `list[dict]` sample data mirrors what `acqstore`'s `AcqImageList` analysis
pools return, so a later revision can swap in `velocity_analysis_pool.get_dataframe()`
without changing the widget wiring.

## Scripts

Focused manual diagnostics live under `scripts/`:

```bash
uv run python scripts/try_plotly_plot_widget.py
uv run python scripts/try_plotly_plot_event_overlays.py
```
