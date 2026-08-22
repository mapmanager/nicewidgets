# State and presets

NicePool uses explicit JSON-compatible state rather than treating controls as
the source of truth. TypeScript interfaces describe authoring-time types,
runtime validators reject incompatible data atomically, and the version-one
JSON Schemas in `schemas/` provide the cross-language contract.

## Workspace state

`NicePoolState` stores `schemaVersion`, layout, active plot index, and exactly
four independent `PlotState` objects. Layout is one of `1x1`, `1x2`, `2x1`, or
`2x2`. Hidden plot slots remain intact when the layout changes. One shared
control panel edits only the active visible plot.

Defaults must be derived after dataset validation because valid X, Y, Group,
and Color-by choices depend on the table schema. `setData` therefore creates a
fresh default workspace and clears selection; it never attempts to carry plot
state across datasets.

Plot-type-specific fields remain present while inactive. This keeps the state
simple, stable, and easy for Python or TypeScript clients to construct.

Display fields are also plot state: legend visibility and position, Plotly
toolbar visibility, combined axis chrome, and horizontal/vertical grid lines.
They are represented by `showLegend`, `legendPosition`,
`showPlotlyToolbar`, `showAxes`, `showHorizontalGrid`, and
`showVerticalGrid`. The default legend position is `bottom`.

Light/dark theme, splitter positions, and the open/closed Display options panel
are workspace presentation settings rather than analytical plot state. They do
not enter presets or summaries. The default theme is dark.

## Plot presets

A `PlotPreset` has a version, name, and one complete `PlotState`. Applying it
replaces only the active plot. The standalone demo may persist presets in
browser local storage, but persistence is optional presentation behavior; the
engine only validates and applies values supplied by its caller.

Preset validation is dataset-aware. A preset referring to an absent or
incompatible column fails as a whole rather than partially changing the plot.
Complete workspace save/load is intentionally deferred until a concrete client
workflow requires it.
