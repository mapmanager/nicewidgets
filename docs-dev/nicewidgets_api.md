# nicewidgets API Used by CloudScope

## Purpose

nicewidgets provides reusable GUI widgets used by CloudScope. The widgets should be framework-level building blocks, not CloudScope-specific application logic.

Design rule:

```text
nicewidgets owns reusable UI behavior.
CloudScope owns application orchestration.
acqstore owns domain interpretation.
```

## PlotlyRasterViewer

Used by:

```text
PrimaryImageView
ReferenceImageView
```

Responsibilities:

```text
- display 2D raster/image data
- handle Plotly/NiceGUI integration
- provide efficient image updates
- expose public ROI overlay APIs
- expose future axis range callbacks for linked views
```

CloudScope should call public methods only. It should not mutate internal Plotly dictionaries directly except through the widget API.

## Plotly update policy

Full `plotly.update()` is expensive because it can push the full plot data/image payload. Use it only for full figure rebuilds.

For ROI-only changes, use browser-side incremental updates such as:

```text
Plotly.relayout(..., {shapes: [...]})
```

The widget should own this implementation detail.

CloudScope should simply call:

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
CloudScope enters ROI edit mode
PlotlyRasterViewer enables selected shape editing
user drags/resizes shape
widget emits preview bounds
CloudScope controller stores pending bounds
toolbar OK commits
cancel restores model shapes
```

nicewidgets should expose clean callbacks/events such as:

```text
on_roi_bounds_preview(roi_id, bounds)
set_roi_editing(enabled, roi_id)
get_current_roi_bounds(roi_id)
```

CloudScope should not parse raw Plotly relayout payloads if nicewidgets can hide that complexity.

## EChartWidget

Used by:

```text
AcqAnalysisPlotView
```

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

CloudScope should pass `AnalysisPlotData` from acqstore directly into EChartWidget. It should not construct low-level ECharts options except through widget APIs.

## Axis range linking

CloudScope needs the primary image x-axis and analysis plot x-axis linked.

nicewidgets responsibilities:

```text
PlotlyRasterViewer:
  emit x-axis range changes from Plotly relayout
  accept programmatic x-axis range updates

EChartWidget:
  emit x-axis range changes from datazoom/axis interactions
  accept programmatic x-axis range updates
```

CloudScope responsibility:

```text
translate widget callbacks into app-level axis range events
avoid feedback loops by tagging event source view_id
```

Suggested CloudScope event:

```text
XAxisRangeChanged(source_view_id, x_min, x_max)
```

This should be implemented after the widgets expose stable callback APIs.

## ImageToolbarWidget

Used by:

```text
ImageToolbarView
```

Responsibilities:

```text
- channel selection UI
- ROI selection UI
- ROI CRUD/edit lifecycle buttons
- emit widget-level toolbar intents
```

CloudScope translates nicewidgets toolbar intents into CloudScope application intents.

Example:

```text
ImageToolbarRoiAddRequestIntent -> AddRoiIntent
ImageToolbarSelectRoiIntent -> SelectRoiIntent
```

The widget should not mutate acqstore or CloudScope state directly.

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

Plotly and ECharts need stable range callback APIs before CloudScope should implement full axis linking.

### Keep widgets CloudScope-agnostic

If a widget imports CloudScope events/controllers/state, that is an architecture smell.

### Test widgets at option/model boundary

Most widget tests should assert generated option dictionaries, styles, and JS payloads rather than requiring a browser.
