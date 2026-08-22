# Plot and summary semantics

Generic aggregation and plot summaries are separate systems. Both reuse pure
descriptive-statistics functions, but only a plot summary describes data
represented by one configured plot.

```text
dataset -> filters/transforms -> prepared plot data
                                  |-> Plotly specification
                                  `-> PlotSummary
```

A summary includes plot parameters, group-level aggregate rows, and long-form
represented rows. It is not reconstructed from Plotly traces. Hidden raw swarm
points remain in the summary when their distribution is represented by plot
overlays.

Swarm positions use a documented deterministic string hash seeded by row ID,
category, color group, and `jitterSeed`. Exact Python hash positions are not a
compatibility target because Python hash randomization is process-dependent.

Histograms compute NicePool-owned bin edges once and share them between Plotly
rendering and summary generation. Every group uses the same edges. Histogram
summaries retain the contributing source rows and include a separate bin table
with counts and cumulative proportions.

## Scatter vocabulary

- **X column** supplies the horizontal coordinate and may be numeric or categorical.
- **Y column** supplies the vertical measurement and must be numeric.
- **Color by** optionally divides points into colored traces using one categorical column.

With `Color by = None`, scatter renders one series and one overall aggregate
summary. NicePool never applies a hidden default color grouping.

## Swarm vocabulary

- **Group** determines the categorical X-axis positions.
- **Y column** supplies the numeric vertical measurement.
- **Color by** optionally subdivides points within each X-axis group using color
  and horizontal offset.

Scatter and swarm share color-trace preparation, selection identity, and
summary primitives. Their X-coordinate preparation remains plot-specific.

## Box and violin vocabulary

Box and violin use the same **Group**, numeric **Y column**, and optional
**Color by** contract as swarm. They reuse one prepared categorical
distribution and add Q1, Q3, and IQR to the shared descriptive summary.

## Histogram vocabulary

- **X column** supplies the numeric observations.
- **Group** and **Color by** optionally subdivide those observations.
- **Histogram bins** is an integer from 1 through 200 and defaults to 50.

Histogram and cumulative histogram use common prepared bins. Cumulative
proportions are normalized independently within each group/color combination.
Interactive bin-to-row selection is intentionally deferred; prepared source
rows retain stable IDs so it can be added without changing the data contract.

## Shared controls

Absolute value transformation and the extreme-value threshold apply during
plot preparation without mutating the authoritative table. Point size applies
to scatter and swarm raw points. Swarm can independently show raw points, its
mean marker, and one error-bar type: standard deviation (SD) or standard error
(SE). Plot types that do not render points will omit the point-size control when
they are introduced.

Display options change Plotly presentation only. Hiding Axes jointly hides axis
titles, axis lines, tick marks, and tick values. Grid lines remain independently
controllable: the X-axis grid produces vertical lines and the Y-axis grid
produces horizontal lines. These fields do not alter prepared points or plot
summaries.
