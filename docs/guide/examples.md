# Examples

Runnable demos live under `examples/` in the repository. They are **not** part
of the installable wheel.

Host applications that embed Plotly or AG Grid widgets should also read
[Layout and sizing](layout-and-sizing.md) (parent height / flex collapse).

Widget narrative + API pages: [Widgets](../widgets/index.md).

## Combined demo app

Run all product demos from one browser-based app:

```bash
uv run python -m examples.main_demo.main
```

Open [http://localhost:8080](http://localhost:8080). The home page links to
independent `/raster`, `/raster-widget`, `/table`, `/nicepool`, `/plotly`, and
`/tree` routes. Separate routes keep the visualization demos isolated and
provide stable deep links.

The shared top toolbar shows the `NiceWidgets` brand, demo links, a Dark switch
(Quasar chrome via `ui.dark_mode` plus each page's widget theme API), and a Docs
link to the published MkDocs site.

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

## RasterViewerWidget

```bash
uv run python -m examples.raster_viewer_widget.main
```

This separate example exercises the canvas-based `RasterViewerWidget` with
synthetic uint16 and float32 datasets, multichannel layouts, T/Z navigation,
Sliding-Z projections, typed rectangle and line ROIs, and physical X/Y plot
overlays. Optional TIFF data is documented in
`examples/raster_viewer_widget/data/README.md`.
It is also mounted at `/raster-widget` in the combined demo app.

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

## PlotlyPlotWidget

```bash
uv run python examples/plotly_plot/demo_app.py
```

The demo exercises the core host API: a continuous `scattergl` line, sparse peak
markers, programmatic x-range set/reset, `on_x_range_changed`, one draggable
horizontal measurement line with `on_measurement_changed`, and the built-in
right-click display menu. Layout:

- `sample_data.py`: synthetic time/intensity sequences (pure data, no NiceGUI)
- `demo_controller.py`: widget config, callbacks, and public-API wiring
- `page.py`: reusable page layout and demo-only controls
- `demo_app.py`: standalone route and `ui.run` entry point

## TreeWidget

```bash
uv run python examples/tree_widget/demo_app.py
```

The demo shows an AG Grid Enterprise file/ROI tree with selection, expand/collapse,
and a custom context menu. Layout:

- `sample_data.py`: synthetic hierarchy rows (pure data, no NiceGUI)
- `page.py`: reusable page layout and demo-only controls
- `demo_app.py`: standalone route and `ui.run` entry point
