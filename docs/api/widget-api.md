# NiceWidgets widget API notes

Per-widget narrative guides and auto-generated class docs live under
[Widgets](../widgets/index.md). This page keeps cross-cutting contracts
consumers rely on (update policy, context menus, hosting rules).

For empty plots / collapsed grids, start with
[Layout and sizing](../guide/layout-and-sizing.md).

## Purpose

NiceWidgets provides reusable NiceGUI widgets for scientific and desktop
applications. The widgets should be framework-level building blocks, not
application-specific orchestration.

[CloudScope](https://mapmanager.github.io/cloudscope/) is one known consumer.
This document records the public widget contracts consumers rely on.

Design rule:

```text
nicewidgets owns reusable UI behavior.
applications own orchestration.
domain packages own domain interpretation.
```

## PlotlyRasterViewer

Responsibilities:

```text
- display 2D raster/image data
- handle Plotly/NiceGUI integration
- provide efficient image updates
- expose public ROI overlay APIs
- expose future axis range callbacks for linked views
```

External callers should call public methods only. They should not mutate
internal Plotly dictionaries directly except through the widget API.

## Plotly update policy

Full `plotly.update()` is expensive because it can push the full plot data/image payload. Use it only for full figure rebuilds.

For ROI-only changes, use browser-side incremental updates such as:

```text
Plotly.relayout(..., {shapes: [...]})
```

The widget should own this implementation detail.

External callers should simply call:

```text
viewer.set_rois(...)
viewer.select_roi(...)
viewer.add_roi(...)
viewer.delete_roi(...)
```

## Plotly raster viewer context menu

The Plotly raster viewer owns its right-click menu and exposes display toggles through public widget APIs. Callers may provide initial display defaults with:

```python
PlotlyRasterViewer(
    display_options=PlotlyRasterViewerDisplayOptions(
        show_plotly_toolbar=False,
        show_rois=True,
        show_trace_overlays=True,
        show_axis_labels=False,
        theme='light',
    )
)
```

The context menu currently provides:

```text
- ROIs
- Traces
- Axis Labels
- Plotly Toolbar
- Copy To Clipboard
```

Checked items are prefixed with `✓`. Visibility toggles must not delete source widget state. ROI visibility is applied through Plotly shape `visible`; trace overlay visibility is applied through Plotly trace `visible`; axis display is updated through `Plotly.relayout` and includes title text, tick labels, tick marks, axis line, zero line, and grid lines; toolbar visibility is updated through Plotly config. Plotly light/dark theme changes are also applied through `Plotly.relayout` without rebuilding the raster image.

Public display API:

```python
set_roi_overlays_visible(visible: bool) -> None
set_trace_overlays_visible(visible: bool) -> None
set_axis_labels_visible(visible: bool) -> None
set_plotly_toolbar_visible(visible: bool) -> None
set_theme(theme: Literal['light', 'dark']) -> None
set_dark_mode(enabled: bool) -> None
copy_plot_to_clipboard() -> None
```

## ROI overlay API

Current/desired public API:

```python
set_rois(rois: Sequence[RectRoiOverlay]) -> None
select_roi(roi_id: int | None) -> None
add_roi(roi: RectRoiOverlay) -> None
delete_roi(roi_id: int) -> None
```

ROI shapes should be stored in Plotly layout `shapes` and identified with:

```text
name = "roi:<roi_id>"
```

ROI style should be configured centrally, not scattered:

```python
RectRoiStyleConfig(
    line_width=..., line_color=..., fill_color=...,
    selected_line_width=..., selected_line_color=..., selected_fill_color=...,
)
```

ROI overlay support currently targets rectangular ROIs only. Line-segment ROI display can be added later with separate shape/model APIs.

## ROI overlay performance rule

ROI-only updates should:

```text
- mutate layout.shapes in the widget's local plotly dict
- call Plotly.relayout for shapes only
- avoid NiceGUI plotly.update()
```

This rule prevents expensive redraws and avoids resending large image payloads.

## Future ROI editing API

Planned editing flow:

```text
application enters ROI edit mode
PlotlyRasterViewer enables selected shape editing
user drags/resizes shape
widget emits preview bounds
application controller stores pending bounds
toolbar OK commits
cancel restores model shapes
```

nicewidgets should expose clean callbacks/events such as:

```text
on_roi_bounds_preview(roi_id, bounds)
set_roi_editing(enabled, roi_id)
get_current_roi_bounds(roi_id)
```

External callers should not parse raw Plotly relayout payloads if nicewidgets can hide that complexity.

## EChartWidget

Responsibilities:

```text
- display one or more 1D/2D chart series
- provide simple line plot API
- update chart options cleanly
- future axis range/datazoom events
```

Current/desired public API:

```python
set_line_data(x, y, x_label, y_label, title=None) -> None
clear() -> None
set_x_axis_limits(x_min, x_max) -> None
reset_x_axis_limits() -> None
```

External callers should pass prepared series data into EChartWidget. They should
not construct low-level ECharts options except through widget APIs.

## Axis range linking

Applications may need a primary image x-axis and an analysis plot x-axis linked.

nicewidgets responsibilities:

```text
PlotlyRasterViewer:
  emit x-axis range changes from Plotly relayout
  accept programmatic x-axis range updates

EChartWidget:
  emit x-axis range changes from datazoom/axis interactions
  accept programmatic x-axis range updates
```

Application responsibility:

```text
translate widget callbacks into app-level axis range events
avoid feedback loops by tagging event source view_id
```

Suggested application event:

```text
XAxisRangeChanged(source_view_id, x_min, x_max)
```

This should be implemented after the widgets expose stable callback APIs.

## ImageToolbarWidget

Responsibilities:

```text
- channel selection UI
- ROI selection UI
- ROI CRUD/edit lifecycle buttons
- emit widget-level toolbar intents
```

Applications translate nicewidgets toolbar intents into application intents.

Example:

```text
ImageToolbarRoiAddRequestIntent -> AddRoiIntent
ImageToolbarSelectRoiIntent -> SelectRoiIntent
```

The widget should not mutate application or domain state directly.

## Widget state APIs

Reusable widgets should provide external setter methods that update internal/UI state without emitting user-intent callbacks. This prevents feedback loops.

Examples:

```text
set_file_ext(...)
set_roi_options_and_selection_ext(...)
set_enabled_ext(...)
```

Pattern:

```text
user action -> emits intent callback
programmatic state sync -> no intent callback
```

## Stress points and improvement areas

### Plotly DOM id handling is fragile

Any browser-side JS must use the established widget DOM-id pattern. Do not guess new selectors. Tests should validate generated JS strings where practical, but real browser behavior still needs manual testing.

### Avoid async public APIs unless required

Simple widget state updates should be synchronous where possible. Async widget APIs caused test/runtime friction in ROI overlay work.

### Axis linking is not finished

Plotly and ECharts need stable range callback APIs before applications should implement full axis linking.

### Keep widgets application-agnostic

If a widget imports application events, controllers, or state, that is an architecture smell.

### Test widgets at option/model boundary

Most widget tests should assert generated option dictionaries, styles, and JS payloads rather than requiring a browser.
