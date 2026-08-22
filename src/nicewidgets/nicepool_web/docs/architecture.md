# Architecture

NicePool Web uses one directional dependency flow:

```text
core <- prepared plots <- Plotly adapter <- Vue widget <- demo application
                                      `---- Custom Element
```

The core imports neither Vue, Plotly, nor browser DOM APIs. It owns the
authoritative dataset, four-slot workspace state, and selection. Vue renders
snapshots and relays user intent. Plotly payloads are decoded only in the
Plotly adapter. Browser persistence belongs to the standalone Vue host rather
than the core, so embedded clients may provide their own storage.

Theme, splitter positions, and control typography are presentation settings,
not plot state. Theme reaches Plotly only through its adapter. The widget
exposes `--nicepool-control-font-size` for consistent control
sizing. Dragging the vertical divider changes control width; dragging the
divider below the plot region changes plot height. Neither operation changes
`NicePoolState` or enters summaries and presets.

## Shared preparation boundary

Each plot is prepared once. Plotly rendering and plot-summary generation consume
the same immutable prepared data. This prevents the duplicated filtering,
coercion, grouping, category ordering, and histogram binning found in the Python
implementation.

## Deliberate non-goals

This implementation does not provide a general dataframe library, service
container, plugin framework, event bus, core persistence system, virtualized
data table, or incremental row mutation. Complete workspace presets are also
deferred. These are not required for the current reusable architecture.
