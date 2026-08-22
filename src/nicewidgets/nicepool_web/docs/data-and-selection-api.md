# Data and selection API

## Dataset replacement

`setData(input)` is a complete reinitialization boundary. It validates and
replaces the authoritative dataset, then clears selection, filters, plot state,
prepared data, summaries, and dataset-derived UI state. Construction options and
host event subscriptions remain installed.

Rows are rectangular JSON-compatible records containing only `string`, finite
`number`, `boolean`, or `null` values. The configured row-ID column must exist
and contain unique, non-empty string or finite-number values. IDs are normalized
to strings. Invalid input rejects the complete replacement.

## Missing values

`null` is the only missing-value representation. Every row must include every
dataset column; adapters must turn an absent CSV/table cell into `null`.
`undefined`, `NaN`, infinity, arrays, objects, and dates are rejected.

The authoritative dataset retains rows containing `null`. A particular plot
projection omits a row only when that plot requires a missing X, Y, Group, or
Color-by value. An inactive filter includes missing rows, while `null` is not
offered as a filter choice. NicePool does not invent a synthetic “(missing)”
category.

## Selection

Selection contains an optional primary row and a multi-row set:

```ts
interface NicePoolSelection {
  primaryRowId: string | null
  selectedRowIds: readonly string[]
}
```

Host calls do not emit user-selection events. Plot interactions emit
`selection-change` from the Vue component and `nicepool-selection-change` from
the Custom Element. Ordinary filtering preserves hidden selections; `setData`
clears all selection.

An area selection replaces the shared selected-row set. A point click replaces
it with exactly one primary row. Clearing selection publishes an empty set to
every Plotly view. Each change also advances Plotly's `selectionrevision`, so
Plotly cannot retain stale multi-selected points as internal interaction state.
Selection events emitted by `Plotly.react` during programmatic synchronization
are ignored; only genuine Plotly user events update the authoritative model.

The Custom Element also emits `nicepool-state-change`,
`nicepool-presets-change`, `nicepool-theme-change`, and `nicepool-data-reset`.
Host-facing methods include
`setState`, `getState`, `setPlotPresets`, and `getPlotPresets` in addition to the
data and selection methods. `setTheme('dark' | 'light')` and `getTheme()` expose
the presentation theme without changing serialized workspace state.

## NiceGUI boundary

`nicegui_demo.py` demonstrates the intended Python boundary. It converts pandas
missing values to JSON `null`, serves the built Custom Element from NiceGUI,
and exchanges data, state, and selection only through public element methods
and custom events. The Python host does not prepare plots or own selection.
