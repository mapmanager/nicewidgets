# Widgets

These pages document how to embed each primary NiceWidgets building block in a
NiceGUI host application: construction, configuration, and the public API.

## Available guides

| Widget | Role |
|--------|------|
| [Raster viewer](raster_viewer.md) | Multiresolution Plotly image viewer |
| [RasterViewerWidget](raster_viewer_widget.md) | Canvas viewer with channels, planes, ROIs, and X/Y overlays |
| [PlotlyPlotWidget](plotly_plot.md) | Scientific traces and measurements |
| [NicePool](nicepool.md) | DataFrame-driven plot pool |
| [TableWidget](table_widget.md) | AG Grid table |
| [TreeWidget](tree_widget.md) | AG Grid tree |

## Shared hosting notes

Plotly and AG Grid widgets need a **definite parent height**. See
[Layout and sizing](../guide/layout-and-sizing.md) before debugging empty plots
or collapsed grids.

Runnable demos live under [`examples/`](../guide/examples.md) (not part of the
installable wheel). Screenshots from the combined demo are in
`docs/assets/widgets/`.
