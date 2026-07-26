# Layout and sizing

NiceWidgets Plotly and AG Grid hosts are sensitive to **parent height**. If the
enclosing NiceGUI layout does not give the widget a real CSS height, the
browser paints a collapsed (often empty) plot or grid. Remounting the page can
appear to “fix” it because the second paint happens after the flex shell has
settled.

This is a common NiceGUI hosting issue for:

- `PlotlyPlotWidget`
- `PlotlyRasterViewer`
- `TableWidget` / `TreeWidget` (`ui.aggrid`)

## Rules of thumb

1. **Prefer an explicit height** on the widget root when the page is
   content-sized (for example Tailwind `h-96`, or a fixed `style='height: …'`).
2. **Do not rely on `h-full` alone** unless a parent in the chain has a definite
   height (`h-screen`, flex child with `flex-1 min-h-0`, splitter pane, etc.).
3. **Percentage height (`height: 100%`) only resolves when the parent height is
   definite.** A parent whose height is `auto` / content-sized makes `h-full`
   children collapse toward zero.
4. **Flex shells need `min-h-0`** on flex children that scroll or shrink;
   without it, nested plots/grids often fail to receive height on first paint.
5. **Plotly may need a resize nudge** after the first SPA navigation into a
   route (`Plotly.Plots.resize`), once the container has a non-zero client size.

## Pattern that collapses

```python
# Parent has no definite height (content-sized column).
with ui.column().classes('w-full'):          # height: auto
    plot = PlotlyPlotWidget(...)
    # Widget root defaults include h-full → 100% of auto ≈ 0 on first paint.
    plot.container.classes('w-full h-full')
```

The same failure mode shows up inside `flex-1 min-h-0 overflow-auto` shells when
an intermediate child also uses `h-full` without contributing a fixed height.

## Pattern that works

Give the plot/grid an **explicit** height and drop conflicting `h-full`:

```python
plot = PlotlyPlotWidget(x_label='Time (s)', y_label='Signal')
plot.container.classes(remove='h-full')
plot.container.classes(add='w-full h-96')
plot.add_trace(name='signal', x=xs, y=ys)
```

For AG Grid tables/trees, size the **parent** you pass to `build()`:

```python
with ui.column().classes('w-full').style('height: 24rem;'):
    table.build()
```

`TableWidget.build()` already supplies a default ~24rem parent when you omit
one; prefer an explicit host height in application layouts.

## Full-height application shell

When you truly want the widget to fill the viewport:

```python
with ui.column().classes('w-full h-screen min-h-0 gap-0'):
    with ui.row().classes('shrink-0'):
        ui.label('Toolbar')
    with ui.column().classes('w-full flex-1 min-h-0'):
        # Now h-full on the child can resolve against flex-1.
        plot = PlotlyPlotWidget(...)
        plot.container.classes('w-full h-full min-h-0')
```

Checklist:

- outer: definite height (`h-screen` or similar)
- flex children that shrink: `min-h-0`
- only one scrolling region owns `overflow-auto`
- avoid stacking several `h-full` wrappers that all mean “100% of auto”

## First SPA navigation blank plot

Symptom: navigating from a home index card into a demo route shows an empty
Plotly area; clicking the same route again in the toolbar shows the plot.

Cause: first client navigation mounts the widget while the flex/`h-full` chain
still resolves to ~0px. Remount gets a real size.

Mitigations used in `examples/plotly_plot`:

1. Remove default `h-full` and apply a fixed height (`h-96`).
2. After mount, nudge Plotly:

```javascript
const el = document.querySelector('.nw-plotly-plot .js-plotly-plot');
if (el && window.Plotly && window.Plotly.Plots) {
  window.Plotly.Plots.resize(el);
}
```

## See also

- [Examples](examples.md)
- [Widgets](../widgets/index.md)
