# NicePool Web Rewrite — Development Roadmap

**Status:** architecture / implementation handoff  
**Source of truth for the current implementation:** `src/nicewidgets/nicepool/` from `nicewidgets-20260819-v1.zip`  
**Target:** fully featured standalone static-web NicePool implemented in TypeScript/JavaScript, with Plotly.js and a framework-neutral public API suitable for thin parent clients.

---

## 1. Purpose of this document

This document is the handoff for rewriting the existing Python/NiceGUI/pandas NicePool widget as a standalone web widget.

The current implementation lives in:

```text
src/nicewidgets/nicepool/
```

Only that directory was examined for this roadmap. No unrelated repository code is assumed to define NicePool behavior.

The rewrite must preserve the useful behavior and scientific/statistical semantics of the current Python implementation while changing the runtime architecture from:

```text
pandas DataFrame
    ↓
Python DataFrameProcessor / algorithms
    ↓
Python FigureGenerator
    ↓
Plotly.py figure dicts
    ↓
NiceGUI controls + NiceGUI Plotly component
```

to:

```text
CSV / rows / caller-supplied data
    ↓
pure TypeScript data model + transformations
    ↓
pure TypeScript statistics / plot preparation
    ↓
Plotly.js figure specs
    ↓
standalone web UI
```

The new widget must run on a static website. It must not require Python, NiceGUI, FastAPI, a server, Pyodide, SQL, or DuckDB for its normal operation.

This roadmap is specifically for **NicePool**. It does not define a rewrite of the separate raster/image viewer.

---

# 2. Executive architecture decisions

The first implementation should use:

- **TypeScript** for all NicePool data processing, state, statistics, selection, configuration, and public API.
- **Plotly.js** for plotting.
- **Vue 3** as the internal UI implementation unless implementation experience gives a concrete reason to change it.
- A **framework-neutral public JavaScript API** so callers do not need to know or use Vue.
- A small CSV parser such as **Papa Parse** for CSV ingestion.
- Ordinary typed JavaScript objects/arrays as the core tabular representation.
- Explicit NicePool-owned statistical utilities rather than relying on the defaults of a dataframe library.
- JSON-serializable plot and widget state.
- Stable row IDs as a required first-class concept.

The initial implementation should **not** use Danfo, Arquero, DuckDB-Wasm, or browser Polars as a foundational dependency.

Arquero remains the preferred fallback if actual implementation proves that repeated dataframe-style transformations are becoming awkward or error-prone.

Danfo remains an option if a future requirement strongly favors pandas-like syntax, but a mechanical pandas-to-Danfo port is not the preferred architecture.

The key principle is:

> Reproduce the behavior and contracts of NicePool, not the incidental pandas implementation details.

---

# 3. Why pure TypeScript is the current recommendation

The current NicePool implementation uses pandas heavily, but the actual operations required by the widget are relatively constrained.

The examined code uses pandas/numpy primarily for:

- filtering rows by categorical values;
- converting values to strings for categorical comparison;
- numeric coercion with invalid values becoming missing;
- dropping missing values;
- selecting columns;
- detecting numeric/categorical columns;
- extracting unique values;
- grouping;
- count/min/max/mean/median/sum/std/sem/CV;
- simple sorting;
- histograms and cumulative histograms;
- temporary tabular structures used during plot preparation;
- interval differences and instantaneous frequency;
- selection calculations;
- output summary tables.

NicePool does **not** appear to require a broad general-purpose pandas compatibility layer for its normal plotting path. The source reviewed here does not depend on complex dataframe joins, MultiIndex workflows, arbitrary pandas `apply`, resampling, rolling windows, or similarly difficult pandas features.

For NicePool, a small typed TypeScript data-processing layer is therefore likely to be simpler, more stable, easier to test, and more browser-native than carrying a dataframe runtime into the application.

---

# 4. Current Python implementation map

The rewrite should be traceable to the current Python source. The following files are the important current responsibilities.

## 4.1 `nice_pool.py`

Current public wrapper:

```text
NicePool
```

Responsibilities include:

- public NicePool construction;
- building the UI;
- relayout;
- updating/replacing the dataframe;
- dark-mode forwarding;
- programmatic row-ID selection.

### Web mapping

Target public facade:

```ts
NicePool
```

or:

```ts
createNicePool(container, options)
```

The public web object should own lifecycle and expose a framework-neutral API.

---

## 4.2 `config.py`

Current public configuration layer:

```text
NicePoolConfig
resolve_pre_filter_columns()
DEFAULT_AUTO_PRE_FILTER_COLUMNS = ("accept", "channel", "roi_id")
```

Important current configuration includes:

- `pre_filter_columns`;
- `unique_row_id_col`;
- `db_type`;
- initial plot state/config;
- callbacks;
- whether to show save controls;
- whether to show selection feedback;
- whether to show the dataframe table;
- configuration persistence;
- dark mode;
- saved plot presets.

### Web mapping

Create a typed:

```ts
export interface NicePoolOptions
```

and a separate:

```ts
export interface NicePoolFeatures
```

for optional UI capabilities.

The new API should preserve the idea that the caller can explicitly specify pre-filter columns or allow NicePool to auto-detect conventional columns such as:

```text
accept
channel
roi_id
```

Missing optional pre-filter columns should be ignored with a warning/event rather than crashing the widget.

---

## 4.3 `plot_state.py`

Current plot types:

```text
scatter
swarm
box_plot
violin
histogram
cumulative_histogram
grouped
```

Current `PlotState` includes:

- pre-filter selections;
- x column;
- y column;
- plot type;
- group column;
- color grouping;
- grouped statistic;
- CV epsilon;
- histogram bins;
- absolute-value transform;
- swarm jitter;
- swarm color-group offset;
- remove-value threshold;
- mean display;
- std/SEM display;
- std vs SEM;
- line widths;
- raw-point visibility;
- point size;
- legend visibility.

It is already explicitly serializable through `to_dict()` / `from_dict()`.

### Web mapping

This becomes a native TypeScript type.

Recommended:

```ts
export type PlotType =
  | 'scatter'
  | 'swarm'
  | 'box_plot'
  | 'violin'
  | 'histogram'
  | 'cumulative_histogram'
  | 'grouped';

export type GroupStatistic =
  | 'mean'
  | 'median'
  | 'sum'
  | 'count'
  | 'std'
  | 'sem'
  | 'min'
  | 'max'
  | 'cv';

export interface PlotState {
  preFilter: Record<string, string>;
  xColumn: string;
  yColumn: string;
  plotType: PlotType;

  groupColumn: string | null;
  colorGrouping: string | null;
  yStatistic: GroupStatistic;
  cvEpsilon: number;

  histogramBins: number;
  useAbsoluteValue: boolean;

  swarmJitterAmount: number;
  swarmGroupOffset: number;

  useRemoveValues: boolean;
  removeValuesThreshold: number | null;

  showMean: boolean;
  showStdSem: boolean;
  stdSemType: 'std' | 'sem';

  meanLineWidth: number;
  errorLineWidth: number;

  showRaw: boolean;
  pointSize: number;
  showLegend: boolean;
}
```

Field names may use Python-compatible snake_case if direct config compatibility is considered more valuable. This is not architecturally important. Pick one convention early and provide one explicit serializer/migration layer.

**Recommendation:** use idiomatic camelCase internally, but define a versioned JSON schema and migration functions. Do not let persistence format leak throughout the implementation.

---

# 5. Data model

## 5.1 Core row representation

Do not create a fake pandas DataFrame API.

Use:

```ts
export type NicePoolValue =
  | string
  | number
  | boolean
  | null;

export type NicePoolRow =
  Record<string, NicePoolValue>;
```

Internally, rows may retain additional source metadata if useful, but the scientific table contract should remain simple.

For stronger typing in specific callers, generic types can be supported:

```ts
export class NicePool<Row extends NicePoolRow = NicePoolRow> {
  ...
}
```

NicePool itself must remain capable of loading arbitrary CSV schemas.

---

## 5.2 Stable row IDs

The current Python implementation uses `unique_row_id_col` for:

- table selection;
- plot-click-to-row mapping;
- linked plot selection;
- programmatic selection;
- mapping Plotly points back to source rows.

That requirement should be preserved and strengthened.

The web rewrite should require or derive a stable row identifier.

Recommended option:

```ts
uniqueRowIdColumn: string
```

Default may remain compatible with the public Python NicePool default (`pool_row_id`) if that is the public wrapper behavior the project wants to preserve.

If the specified ID column is absent:

- **strict mode:** initialization fails with a clear data-contract error;
- optionally, NicePool may support generated internal row IDs, but generated IDs are unsuitable for durable cross-component selection and should not be the default for integrated applications.

For thin-client integration, stable caller-owned row IDs are strongly preferred.

---

# 6. CSV and external data loading

CSV parsing should be separate from NicePool's scientific processing.

Recommended architecture:

```text
CSV URL / File / text
      ↓
CsvLoader
      ↓
NicePoolRow[]
      ↓
NicePool core
```

NicePool should also accept already-parsed rows directly.

Recommended public sources:

```ts
type NicePoolDataSource =
  | { type: 'rows'; rows: NicePoolRow[] }
  | { type: 'csv-url'; url: string; csvOptions?: CsvOptions }
  | { type: 'csv-file'; file: File; csvOptions?: CsvOptions }
  | { type: 'csv-text'; text: string; csvOptions?: CsvOptions };
```

The browser should use `fetch()` for remote CSV.

CORS remains the responsibility of the dataset host.

A lightweight parser such as Papa Parse is appropriate because CSV syntax/parsing and NicePool transformations are separate concerns.

NicePool must not depend on the CSV parser's inferred types being scientifically correct. Numeric coercion should happen explicitly in the NicePool processing layer.

---

# 7. Port of `DataFrameProcessor`

Current source:

```text
dataframe_processor.py
```

This is one of the clearest boundaries to preserve.

Create:

```text
src/core/DataProcessor.ts
```

or:

```text
src/data/NicePoolDataProcessor.ts
```

Its responsibility is deterministic transformations over source rows.

## 7.1 `get_pre_filter_values()`

Python behavior:

- obtains values for a pre-filter column;
- ignores missing values;
- presents values as selectable categorical choices.

Target:

```ts
getPreFilterValues(column: string): string[]
```

Requirements:

- predictable string normalization;
- deterministic ordering matching agreed Python behavior;
- no accidental `"null"` / `"undefined"` choices;
- tests against fixtures from Python.

---

## 7.2 `filter_by_pre_filters()`

Python uses the `(none)` sentinel defined in:

```text
pre_filter_conventions.py
```

and compares categorical values through string conversion.

Target:

```ts
filterByPreFilters(
  rows: NicePoolRow[],
  selections: Record<string, string>
): NicePoolRow[]
```

Preserve:

```text
PRE_FILTER_NONE = "(none)"
```

at least for config compatibility unless deliberately migrated.

Do not special-case `roi_id` in the core filtering algorithm. `roi_id` is simply one possible configured pre-filter column.

---

## 7.3 numeric coercion

Python currently relies on:

```python
pd.to_numeric(..., errors="coerce")
```

The web version must define this behavior explicitly.

Create:

```ts
toFiniteNumber(value: unknown): number | null
```

Do not use naive `Number(value)` without defining edge cases.

Examples that require tests:

- `null`;
- `""`;
- whitespace;
- numeric strings;
- `"nan"`;
- `"NaN"`;
- `"inf"`;
- `"Infinity"`;
- booleans;
- malformed strings.

The intended contract should follow the behavior NicePool actually depends on, not necessarily every pandas conversion edge case.

---

## 7.4 absolute-value processing

Current `PlotState.use_absolute_value` transforms numeric plotting values.

Implement this as an explicit processing option.

The transformation must be applied in the same stages as the Python implementation so summaries and plots agree.

---

## 7.5 remove-values threshold

Current state:

```text
use_remove_values
remove_values_threshold
```

The current Python code removes/masks values outside the threshold before plotting/statistics.

Implement this in the shared numeric-processing path so:

- raw traces;
- summaries;
- group statistics;
- selection;

use consistent transformed values.

Do not duplicate threshold logic independently in each plot renderer.

---

## 7.6 row-ID index

Current:

```python
build_row_id_index()
```

The web implementation should normally use a `Map`.

Example:

```ts
Map<string, number[]>
```

Use an array of indices rather than assuming IDs occur only once unless uniqueness is explicitly validated.

Because the configuration calls this a unique row ID, initialization should validate uniqueness and report duplicates.

---

## 7.7 grouped statistics

Current `calculate_group_stats()` and algorithm modules calculate:

- count;
- min;
- max;
- mean;
- median;
- std;
- sem;
- CV.

Implement NicePool-owned functions in:

```text
src/core/statistics.ts
```

Recommended:

```ts
count()
sum()
mean()
median()
sampleVariance()
sampleStd()
sem()
min()
max()
coefficientOfVariation()
```

Do not hide these inside rendering code.

---

# 8. Statistical compatibility requirements

Scientific/statistical compatibility is an acceptance criterion.

## 8.1 Standard deviation

Current code explicitly uses:

```python
ddof=1
```

for sample standard deviation in relevant summaries.

TypeScript must implement the same formula:

```text
variance = Σ(x - mean)^2 / (n - 1)
```

for `n >= 2`.

Behavior for `n < 2` must match the Python code path being ported. Do not arbitrarily substitute population standard deviation.

---

## 8.2 SEM

SEM is:

```text
sample_std / sqrt(n)
```

with the same small-sample behavior as the current implementation.

Tests must compare against Python output.

---

## 8.3 CV

Current NicePool uses:

```text
std / mean
```

with a configurable epsilon test:

```text
abs(mean) < cv_epsilon → missing/NaN
```

Preserve this.

---

## 8.4 missing values

JavaScript has both `null` and `NaN`.

Do not let this become inconsistent across the codebase.

Recommended internal rule:

- parsed/categorical missing value: `null`;
- invalid numeric input after coercion: `null`;
- statistics helpers operate only on `number[]` containing finite numbers;
- optional result that pandas would express as NaN is represented internally as `null`;
- conversion to Plotly may use `null` where appropriate.

This gives a cleaner JSON model than persisting JavaScript `NaN`.

Compatibility tests should compare semantically rather than requiring JSON to encode NaN.

---

## 8.5 ordering

Pandas grouping often introduces deterministic sorted ordering.

Group and category ordering affects:

- swarm x positions;
- box/violin categories;
- grouped plots;
- legends;
- summary tables.

Define ordering explicitly.

Do not rely on incidental object property order.

Where Python currently sorts groups, match it.

---

# 9. Plot algorithms

Current source:

```text
figure_generator.py
algorithms/group_plot.py
algorithms/swarm_stats.py
algorithms/intv_stats.py
```

The rewrite should split scientific preparation from Plotly construction.

Recommended architecture:

```text
core algorithm
    ↓
typed PlotModel
    ↓
Plotly trace/layout builder
```

Do not let algorithms directly manipulate Vue components.

---

# 10. Scatter plot

Current scatter behavior includes:

- x/y extraction;
- numeric axes;
- categorical considerations;
- raw points;
- color/grouping paths;
- mean and error overlays;
- Plotly customdata carrying row IDs;
- selected-point overlays;
- plot summary generation.

Target split:

```text
prepareScatterData()
buildScatterSummary()
buildScatterFigure()
```

Every source point that can be clicked or selected should carry its stable row ID in Plotly `customdata` or a similarly explicit field.

---

# 11. Swarm plot

Swarm is a custom NicePool behavior and should be treated as a first-class algorithm.

Current code includes:

- categorical grouping;
- optional color grouping;
- group offsets;
- jitter;
- group statistics;
- raw points;
- mean/std/SEM overlays;
- mapping displayed x positions back to rows.

Target split:

```text
prepareSwarmData()
computeSwarmPositions()
computeSwarmStats()
buildSwarmFigure()
```

Define an intermediate type such as:

```ts
interface SwarmPoint {
  rowId: string;
  group: string;
  colorGroup: string | null;
  value: number;
  categoryCenter: number;
  renderedX: number;
}
```

Jitter should be deterministic unless the current Python behavior intentionally changes on every replot.

If the Python jitter is random, the web implementation should consider seeded deterministic jitter as an improvement, but this is a behavior change and must be decided explicitly during implementation.

**Senior-dev recommendation:** deterministic seeded jitter keyed by row ID is preferable because plots remain stable across UI state updates. Before changing behavior, verify whether current Python jitter is deterministic.

---

# 12. Box plot and violin plot

Port the current grouping and color-group behavior.

The data-preparation functions should be shared with swarm where possible.

Do not duplicate categorical validation across swarm/box/violin.

Use Plotly.js native box and violin traces once the NicePool grouping model has been prepared.

---

# 13. Histogram

Current state exposes:

```text
histogram_bins
```

The implementation includes histogram summary/export behavior.

Avoid relying blindly on Plotly's auto-binning if exact compatibility with Python/Numpy histograms matters.

Recommended:

- calculate bins and counts in TypeScript;
- feed explicit bin-derived results to Plotly as needed;
- test edge conventions against the Python implementation.

Create a deterministic:

```ts
histogram(values, binCount)
```

utility whose bin edge conventions are documented and tested.

---

# 14. Cumulative histogram

Current implementation explicitly supports:

```text
cumulative_histogram
```

Do not implement this merely as a Plotly toggle if that changes the current calculated outputs.

Prepare histogram data in the TypeScript algorithm layer, then produce cumulative counts/percentages according to existing NicePool behavior.

---

# 15. Grouped plot

Current grouped plot supports statistics:

```text
mean
median
sum
count
std
sem
min
max
cv
```

and explicit categorical-group validation.

Target:

```ts
prepareGroupedPlot()
groupedAggregate()
buildGroupedStatsTable()
buildGroupedFigure()
```

The aggregate table should be a reusable data product, not something reconstructed from Plotly traces.

This enables:

- plotting;
- copy/export;
- testing;
- caller callbacks;
- future downloadable CSV/TSV.

---

# 16. Interval statistics

Current:

```text
algorithms/intv_stats.py
```

This code is not merely UI glue. It defines a scientific transformation.

Current behavior includes:

- filtering by `roi_id`;
- filtering by `rel_path`;
- filtering by `event_type`;
- time-series numeric coercion;
- preserving existing chronological row order rather than sorting;
- computing IEI by difference from the previous event;
- instantaneous frequency = `1 / iei`;
- first interval missing;
- zero IEI treated as detection error;
- zero IEI and associated instantaneous frequency masked before aggregation;
- tracking `n_original`;
- aggregate count/min/max/mean/std/sem/CV;
- parsing `rel_path` into metadata components.

Port this into a standalone:

```text
src/core/algorithms/intervalStats.ts
```

Do not quietly sort the event table during the rewrite; the existing implementation explicitly assumes chronological ordering.

---

# 17. Plot summaries and export

Current:

```text
plot_summary.py
```

supports plot summaries and TSV-style formatting.

This functionality belongs in the web rewrite.

Recommended public result model:

```ts
export interface PlotSummary {
  plotType: PlotType;
  title?: string;
  metadata: Record<string, unknown>;
  columns: string[];
  rows: Array<Record<string, NicePoolValue>>;
}
```

Formatting should be separate:

```ts
plotSummaryToTsv(summary)
plotSummaryToCsv(summary)
```

Browser actions may offer:

- copy summary;
- download summary;
- parent callback with structured summary.

Never make TSV strings the only canonical representation.

---

# 18. Plot validation and errors

Current:

```text
plot_errors.py
plot_preset_validation.py
```

contains important behavior:

- categorical-column validation;
- group-column requirements;
- histogram numeric validation;
- friendly configuration errors;
- empty Plotly figure containing an error message;
- stale config repair;
- range clamping;
- fallback columns.

Port this behavior instead of letting Plotly throw low-level errors.

Define errors such as:

```ts
NicePoolConfigurationError
NicePoolDataError
NicePoolStateError
```

UI should display a friendly inline plot error.

The parent should also receive an error event/callback.

---

# 19. Column classification

Current:

```text
plot_helpers.py
```

contains:

- numeric-column detection;
- categorical-candidate detection;
- categorical-column checks;
- Plotly lasso path parsing;
- point-in-polygon selection.

Move data classification into:

```text
src/core/schema.ts
```

Move geometry into:

```text
src/core/geometry.ts
```

Do not place UI CSS helpers in the core.

---

# 20. Selection model

This is a major feature and must be included in the first-class architecture.

Current:

```text
selection_handler.py
```

supports:

- scatter/swarm selection compatibility;
- rectangular selection;
- lasso selection;
- linked selection across plots;
- selection by stable row ID;
- selection by multiple IDs;
- selection clearing;
- Escape-to-clear;
- Meta/Control extend-selection modifier;
- update of selection feedback;
- source-row callback after point click.

The web rewrite should own selection in core state:

```ts
selectedRowIds: Set<string>
```

and expose it publicly as serializable arrays:

```ts
getSelectedRowIds(): string[]
setSelectedRowIds(ids: Iterable<string>): void
clearSelection(): void
```

All selection-compatible plots should render the same linked selection.

The table and plots should use the same selection model.

---

# 21. Selection geometry

The current code can calculate selection from Plotly relayout geometry rather than trusting only point indices.

Port:

- rectangle hit-testing;
- lasso SVG-path parsing if still required by Plotly.js event payloads;
- point-in-polygon ray casting.

Before porting the SVG parser exactly, inspect actual Plotly.js browser events because direct `plotly_selected` events may provide more convenient selected point data than the NiceGUI bridge currently exposes.

**Senior-dev recommendation:** prefer native Plotly.js `plotly_selected` event data in the browser when it provides stable source-point mapping. Retain geometry hit-testing only where required to reproduce behavior.

This is a good example of behavior to preserve while not mechanically copying a NiceGUI workaround.

---

# 22. Plot-click behavior

Current controller behavior:

- scatter/swarm click resolves a row ID from `customdata`;
- caller callback receives row ID and full row;
- linked selection changes to that row;
- grouped plot click is treated as an aggregate group, not an individual source row.

Web API should make that distinction explicit.

Recommended events:

```ts
interface NicePoolPointClickEvent {
  plotIndex: number;
  rowId: string;
  row: NicePoolRow;
  plotState: PlotState;
}

interface NicePoolGroupClickEvent {
  plotIndex: number;
  groupValue: string | number;
  value: number | null;
  plotState: PlotState;
}
```

---

# 23. Data table

Current:

```text
dataframe_table_view.py
dataframe_adapter.py
```

supports optional table rendering and row selection.

The web widget should retain an optional data table feature, but the table implementation should not contaminate the core.

Recommended:

```text
core rows + selection
      ↑
DataTable component
```

For first implementation, use a simple virtualizable table only if actual datasets require virtualization.

Do not prematurely introduce a large grid dependency.

The table should:

- display source rows;
- respect current filter state if that matches Python behavior;
- select a row;
- propagate row selection to linked plots;
- receive selection updates from plots.

---

# 24. Multi-plot layouts

Current validation supports:

```text
1x1
1x2
2x1
2x2
```

and therefore one to four plot slots.

Preserve this.

Create:

```ts
type PlotLayout = '1x1' | '1x2' | '2x1' | '2x2';
```

Widget state should contain:

```ts
interface NicePoolState {
  schemaVersion: number;
  layout: PlotLayout;
  plots: PlotState[];
  currentPlotIndex: number;
  selectedRowIds: string[];
  controlPanelSize?: number;
}
```

Only the number of plot states required by the layout should be active/rendered.

Changing layout should not unintentionally destroy reusable inactive plot states unless that is the current intended behavior.

---

# 25. Current plot / apply-to-others behavior

The Python controller tracks a current plot and includes controls for selecting a plot slot and applying the current configuration to other plots.

Preserve this UI concept.

Core operations should include:

```ts
setCurrentPlot(index)
setPlotState(index, state)
copyPlotState(sourceIndex, targetIndices)
```

The caller should be able to configure all plot slots at initialization without simulating UI interaction.

---

# 26. Saved plot configuration

Current:

```text
pool_plot_config.py
```

implements persisted widget/session configuration with schema versions and backward compatibility.

Browser persistence should use:

- JSON serialization;
- `localStorage` for local user configuration initially;
- optional caller-provided persistence adapter later.

Do not write persistence into the scientific core.

Define:

```ts
interface NicePoolPersistenceAdapter {
  load(key: string): Promise<NicePoolPersistedState | null>;
  save(key: string, value: NicePoolPersistedState): Promise<void>;
  delete?(key: string): Promise<void>;
}
```

Provide a built-in `localStorage` adapter.

A thin host may disable persistence entirely and own state itself.

---

# 27. Named plot presets

Current:

```text
plot_preset_config.py
plot_preset_validation.py
```

supports named presets and repairs stale values against the current dataframe schema.

Preserve this concept.

Browser preset schema should be versioned.

Preset validation should:

- reject/repair missing x/y columns;
- remove stale group/color columns;
- validate plot type;
- validate grouped statistic;
- clamp bins/jitter/line widths/etc.;
- repair pre-filter values that no longer exist;
- fill missing plot slots from defaults;
- validate layout.

Presets can initially use `localStorage`.

The public API should allow thin clients to supply presets without depending on persistence:

```ts
setPresets(...)
getPresets()
applyPreset(name)
```

---

# 28. UI architecture

Use Vue internally, but isolate it.

Recommended directory structure:

```text
nicepool-web/
  src/
    core/
      types.ts
      state.ts
      DataProcessor.ts
      statistics.ts
      schema.ts
      geometry.ts
      selection.ts
      validation.ts

      algorithms/
        scatter.ts
        swarm.ts
        grouped.ts
        histogram.ts
        intervalStats.ts

    plotting/
      FigureGenerator.ts
      scatterFigure.ts
      swarmFigure.ts
      boxFigure.ts
      violinFigure.ts
      groupedFigure.ts
      histogramFigure.ts
      plotlyTypes.ts

    io/
      csvLoader.ts
      persistence.ts
      presets.ts

    ui/
      NicePoolRoot.vue
      PoolControlPanel.vue
      PlotGrid.vue
      PlotPanel.vue
      DataTable.vue
      SelectionFeedback.vue
      PresetControls.vue

    api/
      NicePool.ts
      events.ts
      options.ts

    index.ts
```

The exact folder names may change, but preserve these dependency directions:

```text
core     → no Vue, no DOM, ideally no Plotly
plotting → depends on core + Plotly types
io       → depends on core types
ui       → depends on core + plotting + Vue
api      → coordinates public instance and UI
```

Core code should be testable under Node/Vitest without a browser.

---

# 29. Framework-neutral public API

Thin parent clients must not have to import Vue components or manipulate NicePool internals.

Recommended public construction:

```ts
const nicePool = new NicePool(container, options);
await nicePool.ready();
```

or:

```ts
const nicePool = await createNicePool(container, options);
```

Recommended `NicePoolOptions`:

```ts
interface NicePoolOptions {
  data: NicePoolDataSource;

  uniqueRowIdColumn: string;

  preFilterColumns?: string[];
  autoPreFilterColumns?: string[];

  initialState?: Partial<NicePoolState>;
  initialPlotConfig?: NicePoolPlotConfig;

  darkMode?: boolean;
  theme?: 'light' | 'dark' | string;

  features?: {
    table?: boolean;
    selectionFeedback?: boolean;
    presets?: boolean;
    configPersistence?: boolean;
    summaryActions?: boolean;
  };

  persistenceKey?: string;
  persistenceAdapter?: NicePoolPersistenceAdapter;

  callbacks?: NicePoolCallbacks;
}
```

---

# 30. Required lifecycle API

Recommended:

```ts
class NicePool {
  ready(): Promise<void>;

  destroy(): void;

  resize(): void;

  setRows(rows: NicePoolRow[]): void;
  loadCsvUrl(url: string, options?: CsvOptions): Promise<void>;
  loadCsvFile(file: File, options?: CsvOptions): Promise<void>;

  getState(): NicePoolState;
  setState(state: Partial<NicePoolState>): void;

  getPlotState(index: number): PlotState;
  setPlotState(index: number, state: Partial<PlotState>): void;

  getSelectedRowIds(): string[];
  setSelectedRowIds(ids: Iterable<string>): void;
  clearSelection(): void;

  setDarkMode(enabled: boolean): void;

  getPlotSummary(index: number): PlotSummary | null;

  refresh(): void;
}
```

`setRows()` should replace the source data and sanitize current plot states against the new schema, analogous to the Python `update_df()` behavior.

---

# 31. Required callbacks / events

Thin callers need upward communication.

Provide both:

1. initialization callbacks;
2. DOM `CustomEvent` dispatch from the NicePool root/container.

This gives simple direct use and framework interoperability.

Recommended callbacks:

```ts
interface NicePoolCallbacks {
  onReady?: (event: NicePoolReadyEvent) => void;

  onSelectionChange?: (
    event: NicePoolSelectionChangeEvent
  ) => void;

  onRowClick?: (
    event: NicePoolPointClickEvent
  ) => void;

  onGroupClick?: (
    event: NicePoolGroupClickEvent
  ) => void;

  onStateChange?: (
    event: NicePoolStateChangeEvent
  ) => void;

  onPlotStateChange?: (
    event: NicePoolPlotStateChangeEvent
  ) => void;

  onFilterChange?: (
    event: NicePoolFilterChangeEvent
  ) => void;

  onPlotTypeChange?: (
    event: NicePoolPlotTypeChangeEvent
  ) => void;

  onError?: (
    error: NicePoolErrorEvent
  ) => void;

  onHover?: (
    event: NicePoolHoverEvent
  ) => void;
}
```

Hover should be optional because high-frequency hover callbacks can create unnecessary cross-component traffic.

---

# 32. Selection callback contract

This is especially important for integration with other viewers.

Recommended:

```ts
interface NicePoolSelectionChangeEvent {
  source:
    | 'plot-click'
    | 'plot-rect'
    | 'plot-lasso'
    | 'table'
    | 'api'
    | 'clear';

  plotIndex: number | null;

  rowIds: string[];

  rows: NicePoolRow[];
}
```

A parent image viewer can then react to NicePool selection without knowing anything about Plotly or Vue.

Likewise a parent can call:

```ts
nicePool.setSelectedRowIds(ids);
```

to make NicePool reflect selection originating elsewhere.

This makes linked-selection integration symmetric.

---

# 33. State-change callback contract

Avoid emitting only vague `"changed"` events.

Provide structured events.

Recommended:

```ts
interface NicePoolStateChangeEvent {
  state: NicePoolState;
  reason:
    | 'filter'
    | 'plot-config'
    | 'layout'
    | 'selection'
    | 'data'
    | 'preset'
    | 'api';
}
```

For high-frequency interactions, state emission may be debounced.

Selection should have its own immediate event.

---

# 34. Parent/child ownership boundary

The parent owns:

- where NicePool is mounted;
- the source dataset URL/file/rows;
- stable row-ID meaning across the larger application;
- optional initial configuration;
- optional persistence outside NicePool;
- responses to NicePool events;
- cross-widget orchestration.

NicePool owns:

- parsing when given CSV;
- schema inspection;
- filtering;
- statistics;
- plot preparation;
- Plotly rendering;
- internal controls;
- multi-plot layout;
- linked selection inside NicePool;
- plot summaries;
- optional local presets and local persistence.

Do not make the parent reach into NicePool Vue refs or Plotly divs.

---

# 35. Packaging recommendation

First target:

```text
ES module package
```

with:

```ts
import { NicePool } from '@nicewidgets/nicepool-web';
```

and CSS import.

Example:

```ts
import { NicePool } from '@nicewidgets/nicepool-web';
import '@nicewidgets/nicepool-web/style.css';

const widget = new NicePool(
  document.getElementById('nicepool')!,
  options,
);
```

This is the preferred initial public API.

A Vue wrapper can also be exported:

```ts
import { NicePoolView } from '@nicewidgets/nicepool-web/vue';
```

but should not be required.

A Web Component/custom element can be added later:

```html
<nice-pool></nice-pool>
```

but should **not** be the first architectural constraint.

**Senior-dev recommendation:** start with the imperative ESM `NicePool` class/factory. It gives framework-neutral integration without adding the lifecycle/event/property complications of a custom element. Add a custom-element facade only when a real caller benefits from it.

---

# 36. Plotly.js integration

The Python `FigureGenerator` already produces Plotly-compatible conceptual structures.

Do not attempt to translate Plotly.py calls line-for-line.

Instead:

```text
Python:
DataFrame → go.Scatter / go.Box / go.Violin → figure

Web:
rows → typed prepared plot data → Plotly.Data[] + Layout
```

Centralize figure generation behind:

```ts
class FigureGenerator {
  makeFigure(
    rows: NicePoolRow[],
    state: PlotState,
    selection: ReadonlySet<string>
  ): FigureResult;
}
```

Where:

```ts
interface FigureResult {
  data: Plotly.Data[];
  layout: Partial<Plotly.Layout>;
  config?: Partial<Plotly.Config>;
  summary: PlotSummary | null;
}
```

---

# 37. Plot rendering lifecycle

Do not make Vue deeply reactive over Plotly figure objects.

Use Vue for state and controls, then explicitly call Plotly methods.

Recommended:

- initial render: `Plotly.newPlot`;
- ordinary updates: `Plotly.react`;
- resize: `Plotly.Plots.resize`;
- cleanup: `Plotly.purge`.

This should eliminate the NiceGUI-specific need to rebuild a plot when the Python bridge cannot reliably update structural Plotly changes.

Test structural transitions such as:

```text
scatter → swarm
swarm → histogram
histogram → box
```

using `Plotly.react()` before adding workaround rebuilds.

---

# 38. Theme and dark mode

Current public API supports dark mode and the Python `FigureGenerator` forwards theme behavior.

Retain:

```ts
setDarkMode(enabled: boolean)
```

Theme belongs above individual plot builders.

Do not hard-code colors in each algorithm.

Use a central NicePool theme abstraction whose result configures Plotly layout/traces.

---

# 39. Current control panel mapping

Current:

```text
pool_control_panel.py
```

should map to Vue controls.

Conceptual mapping:

```text
NiceGUI select       → Vue select/component
NiceGUI checkbox     → checkbox
NiceGUI number       → number input
NiceGUI radio        → radio/tab group
NiceGUI button       → button
NiceGUI aggrid       → simple table/grid component
NiceGUI dialog       → Vue modal/dialog
NiceGUI callbacks    → Vue emits / NicePool controller actions
```

The UI should bind to typed state, not directly alter Plotly.

Flow:

```text
control
  ↓
action/store
  ↓
PlotState
  ↓
FigureGenerator
  ↓
Plotly.react()
```

---

# 40. State-management recommendation

Do not add Pinia initially.

NicePool is a self-contained widget.

Use:

- Vue `reactive` / `ref`;
- a plain TypeScript `NicePoolModel` or controller;
- explicit actions.

Add a state library only if the actual component tree becomes difficult to manage.

Keeping state outside Vue-specific stores also makes the framework-neutral public API cleaner.

---

# 41. Current config defaults and schema sanitization

Current Python code already performs defensive state repair.

The web rewrite should explicitly support a schema lifecycle.

Recommended persisted root:

```ts
interface NicePoolPersistedConfig {
  schemaVersion: number;
  layout: PlotLayout;
  plotStates: SerializedPlotState[];
  controlPanelSize?: number;
}
```

State sanitation must occur when:

- loading persisted config;
- loading a named preset;
- changing datasets;
- calling `setState()` from a parent;
- changing columns.

Never assume saved column names still exist.

---

# 42. Data replacement / refresh

Current Python supports:

```text
update_df()
on_refresh_requested
```

The browser equivalent should distinguish:

```ts
setRows(...)
```

from optional refresh orchestration.

A thin caller may own refresh:

```ts
const newRows = await callerLoadData();
nicePool.setRows(newRows);
```

NicePool may also support:

```ts
reload()
```

when its data source is a known CSV URL.

Do not require an `onRefreshRequested` callback in the core API. A UI refresh button can call a configured async data loader.

Suggested option:

```ts
reloadData?: () => Promise<NicePoolRow[]>
```

---

# 43. Pandas-like library alternatives

The first implementation should remain pure TS, but the handoff should preserve alternatives.

## 43.1 Arquero

Best alternative if NicePool's transformations grow significantly.

Strengths for NicePool:

- JS-native table model;
- filtering;
- derive;
- groupby;
- aggregation;
- sorting;
- reshaping;
- good fit for plot-preparation pipelines.

Potential use:

```text
CSV
 ↓
Arquero Table
 ↓
filter / derive / groupby / rollup
 ↓
typed NicePool plot model
```

Why it is not the initial choice:

- current transformations are small enough to implement clearly in typed TS;
- NicePool still needs its own statistics semantics;
- swarm/selection logic is custom and not simplified much by a dataframe library;
- adding Arquero would create another core abstraction without yet solving a demonstrated problem.

### Reconsider Arquero when

- multiple algorithms duplicate complex group/filter logic;
- joins/pivots become requirements;
- pure TS grouping code becomes a maintenance burden;
- large tables expose material performance issues with row-oriented loops;
- benchmarks show Arquero provides a meaningful benefit.

---

## 43.2 Danfo.js

Danfo is attractive because its API is pandas-like.

Strength:

- easiest conceptual transition for a developer reading pandas code;
- familiar DataFrame/Series/groupby model.

Why it is not preferred:

- a direct pandas-style translation preserves implementation idioms rather than defining a clean web-native core;
- NicePool does not currently need enough general DataFrame behavior to justify pandas emulation;
- type-safe intermediate models are clearer for custom plotting algorithms;
- NicePool must still own statistical compatibility rather than trusting library defaults.

### Reconsider Danfo when

- maintaining syntax similarity with a large body of pandas code becomes a primary product requirement;
- substantially more pandas-based NicePool algorithms are added;
- translation cost becomes more important than browser-native architecture.

---

## 43.3 Data-Forge

Another TypeScript/JavaScript dataframe abstraction.

It is not recommended initially because it adds an abstraction without an obvious NicePool-specific advantage over:

- pure TypeScript; or
- Arquero.

Keep it off the initial dependency list.

---

## 43.4 DuckDB-Wasm

Not recommended for NicePool's current workload.

It would make sense for:

- very large tabular inputs;
- many joins;
- query-heavy exploration;
- Parquet/Arrow-oriented analytics;
- complex relational datasets.

Current NicePool uses CSV/dataframe-style transformations and does not use SQL.

Adding a browser database and WASM runtime would be unnecessary architecture at this stage.

---

## 43.5 browser Polars / WASM

Do not select browser Polars merely because some Python users use Polars.

NicePool's source reviewed for this handoff is pandas-driven and does not need Polars-specific semantics.

Revisit only if browser Polars becomes a mature, compelling dependency and there is a concrete performance or API benefit.

---

# 44. When to add any dataframe library

Do not decide by taste.

Use implementation evidence.

After the first pure-TS processor is written, evaluate:

- lines of repeated grouping/filter code;
- readability;
- correctness risk;
- bundle size;
- benchmark performance on representative CSVs;
- ease of maintaining statistical semantics.

If a dataframe layer is warranted, **Arquero is the current preferred first evaluation**.

The rest of the architecture must not depend on this decision.

This is why the data processor should sit behind a clean NicePool-owned interface.

---

# 45. Recommended processor interface

Example:

```ts
export interface NicePoolDataProcessor {
  readonly columns: readonly string[];

  getPreFilterValues(column: string): string[];

  filter(
    state: PlotState
  ): NicePoolRow[];

  numericColumn(
    rows: readonly NicePoolRow[],
    column: string,
    transform: NumericTransform
  ): Array<number | null>;

  classifyColumn(column: string): ColumnKind;

  groupValues<T>(
    rows: readonly NicePoolRow[],
    groupColumn: string,
    value: (row: NicePoolRow) => T
  ): Map<string, T[]>;

  rowById(id: string): NicePoolRow | null;
}
```

A future Arquero-backed processor could implement the same high-level contract without changing the public NicePool API.

---

# 46. Performance expectations

NicePool is tabular analysis/plotting, not raster data.

Start simple.

Recommended assumptions until real profiling disproves them:

- source CSVs fit in browser memory;
- raw rows are held in memory;
- a few copies during filtering are acceptable;
- Plotly point count may become a larger bottleneck than pure JS filtering.

Do not optimize prematurely with WASM.

Measure:

1. CSV parse time;
2. filter/aggregation time;
3. figure-construction time;
4. Plotly render/update time;
5. table rendering time.

If processing becomes expensive:

- cache normalized numeric columns;
- cache unique categorical values;
- cache filtered row indices;
- use row-index arrays instead of copying rows;
- move expensive pure processing into a Web Worker;
- then evaluate Arquero.

A Web Worker is a more targeted escalation than immediately introducing DuckDB-Wasm.

---

# 47. Recommended internal optimization path

Version 1:

```text
NicePoolRow[]
```

If profiling requires improvement:

```text
source rows
+
normalized column cache
+
filtered index arrays
```

Example:

```ts
Map<string, Float64Array>
```

for cached numeric columns is possible later without exposing columnar storage to callers.

Do not design Version 1 around typed arrays unless benchmarks require it.

---

# 48. Static-hosting constraint

The resulting widget must work when hosted from static infrastructure such as:

- GitHub Pages;
- Cloudflare Pages;
- an ordinary static web server;
- another application bundle.

Normal viewer operation must not require:

- Python;
- NiceGUI;
- FastAPI;
- Node server;
- login;
- database;
- background service.

Remote CSVs require normal browser CORS support.

Local CSV files should be loadable through browser File APIs.

---

# 49. Testing strategy

The Python implementation is the behavioral oracle during the port.

Do not validate only by looking at plots.

Create shared fixtures.

For each fixture:

```text
input CSV
+
Python expected intermediate/summary JSON
+
TypeScript test
```

Use Python to emit stable expected results during migration.

The web tests should not invoke Python at runtime.

---

# 50. Unit tests

At minimum test:

## Data processing

- pre-filter `(none)`;
- `roi_id` filtering;
- multiple simultaneous pre-filters;
- missing pre-filter columns;
- categorical string conversion;
- numeric coercion;
- missing values;
- absolute values;
- threshold removal;
- row-ID indexing;
- duplicate ID validation.

## Statistics

- count;
- min/max;
- mean;
- median;
- sum;
- sample std;
- SEM;
- CV;
- near-zero CV denominator;
- 0/1/N point groups.

## Histogram

- bin edges;
- counts;
- missing values;
- cumulative behavior.

## Interval stats

- chronological inputs;
- first event;
- zero IEI;
- `n_original`;
- invalid numeric timestamps;
- aggregate statistics.

## State

- serialization;
- sanitation;
- stale columns;
- stale filters;
- invalid layouts;
- invalid plot types;
- clamped numeric options;
- schema migration.

## Selection

- row click;
- API selection;
- rectangle;
- lasso;
- clear;
- extend selection;
- selection across filtered plots;
- rows absent from a plot's filter.

---

# 51. Figure-level tests

Do not snapshot entire Plotly objects blindly.

Test semantic properties:

- number/type of traces;
- x/y arrays;
- customdata row IDs;
- group labels;
- selected overlays;
- error bar values;
- histogram arrays;
- layout category ordering;
- legend flags.

Use selective snapshots only for stable normalized objects.

---

# 52. Browser interaction tests

Use Playwright for:

- widget initialization;
- CSV URL load;
- local fixture load where practical;
- plot type changes;
- changing x/y/group columns;
- pre-filter changes;
- 1x1 ↔ 2x2;
- point click;
- plot-to-table selection;
- API-driven external selection;
- lasso/rectangle if reliable to automate;
- preset save/load;
- config persistence;
- dark mode;
- destruction/remount.

---

# 53. Parent integration tests

Create a tiny host page that does not use Vue directly.

Example behavior:

```text
parent
  ↓ init options
NicePool

NicePool selection
  ↓ callback
parent label/list

parent button
  ↓ setSelectedRowIds()
NicePool
```

This is the acceptance test for the framework-neutral boundary.

The parent test should prove that no Vue internals are required.

---

# 54. Compatibility matrix

During implementation maintain a matrix:

| Python feature | Source | Web status | Acceptance |
|---|---|---:|---|
| scatter | `figure_generator.py` | pending | Python fixture + UI |
| swarm | `figure_generator.py`, `swarm_stats.py` | pending | Python fixture + UI |
| box | `figure_generator.py` | pending | UI + grouping tests |
| violin | `figure_generator.py` | pending | UI + grouping tests |
| histogram | `figure_generator.py` | pending | exact bins/counts |
| cumulative histogram | `figure_generator.py` | pending | exact cumulative output |
| grouped | `group_plot.py` | pending | all y-stat modes |
| pre-filters | `dataframe_processor.py` | pending | exact fixture |
| selection | `selection_handler.py` | pending | browser tests |
| table | `dataframe_table_view.py` | pending | linked selection |
| summaries | `plot_summary.py` | pending | expected TSV/data |
| layouts | controller | pending | 1/2/4 plots |
| presets | preset modules | pending | stale schema repair |
| persistence | `pool_plot_config.py` | pending | reload test |
| dark mode | generator/controller | pending | visual/layout test |
| external selection API | controller/NicePool | pending | host integration |

Update this table as the port proceeds.

---

# 55. Implementation phases

## Phase 0 — Freeze behavior and fixtures

Before writing UI:

1. select representative CSV fixture(s);
2. run existing Python NicePool algorithms;
3. export expected:
   - filtered row IDs;
   - transformed x/y;
   - grouped stats;
   - swarm stats;
   - histogram results;
   - plot summaries;
   - interval stats where applicable;
4. document any currently ambiguous behavior.

Deliverable:

```text
tests/fixtures/
```

with source CSV plus expected JSON.

---

## Phase 1 — Pure TypeScript core

Implement:

- types;
- state;
- pre-filter conventions;
- numeric normalization;
- schema inspection;
- DataProcessor;
- statistics;
- state sanitation;
- selection model excluding Plotly event plumbing.

No Vue.

No Plotly rendering.

Acceptance:

- TypeScript core reproduces Python fixture outputs.

This is the first critical milestone.

---

## Phase 2 — Plot-preparation algorithms

Implement typed prepared-data algorithms for:

- scatter;
- swarm;
- grouped;
- histogram/cumulative histogram;
- box/violin grouping;
- interval stats if it is part of the standalone NicePool deliverable.

Acceptance:

- intermediate outputs match Python;
- no DOM;
- no Vue.

---

## Phase 3 — Plotly.js figure generation

Implement:

```text
FigureGenerator.ts
```

and one builder per plot family.

Start with:

1. scatter;
2. grouped;
3. histogram;
4. swarm;
5. box;
6. violin;
7. cumulative histogram.

Acceptance:

- figures render from static fixtures;
- point customdata contains row IDs;
- summaries match Python.

---

## Phase 4 — Minimal standalone UI

Build:

- single plot;
- x/y controls;
- plot-type control;
- pre-filter controls;
- group/color controls as relevant;
- basic visual controls;
- static CSV loading.

Acceptance:

- a static Vite build runs without Python/server logic.

---

## Phase 5 — Selection and callbacks

Implement:

- Plotly click;
- selected row IDs;
- linked external selection;
- rectangle/lasso;
- selection feedback;
- optional table;
- parent callbacks;
- DOM events.

Acceptance:

- selection works both directions:
  - NicePool → parent;
  - parent → NicePool.

This is mandatory before calling the widget integration-ready.

---

## Phase 6 — Multi-plot layout

Implement:

```text
1x1
1x2
2x1
2x2
```

plus:

- current plot;
- per-plot state;
- linked selection;
- apply current plot settings to others.

Acceptance:

- each plot has independent filter/config state;
- selection remains shared.

---

## Phase 7 — State, persistence, presets

Implement:

- JSON state;
- schema versioning;
- sanitation;
- localStorage adapter;
- named presets;
- stale-preset repair;
- initial caller-supplied configuration.

Acceptance:

- saved state reloads;
- stale state does not crash against changed CSV columns.

---

## Phase 8 — Summary/export UX

Implement:

- structured `PlotSummary`;
- copy TSV;
- optional download CSV/TSV;
- summary callback/API.

---

## Phase 9 — Performance pass

Only after feature parity:

- profile representative real datasets;
- optimize bottlenecks;
- evaluate caches;
- consider worker;
- evaluate Arquero only if justified.

---

# 56. First implementation milestone

The first local project should **not** start by recreating the entire NiceGUI layout.

The first milestone is:

> Given one representative NicePool CSV, load it in TypeScript, apply the same pre-filter/numeric/statistical transformations as the Python `DataFrameProcessor`, and generate a Plotly.js scatter plot whose plotted row IDs and summary statistics match the Python implementation.

Required components:

```text
types.ts
PlotState.ts
DataProcessor.ts
statistics.ts
csvLoader.ts
scatter.ts
FigureGenerator.ts
minimal demo page
tests
```

Required demo:

```text
CSV URL
  ↓
load
  ↓
pre-filter
  ↓
x/y selection
  ↓
Plotly scatter
  ↓
point click emits row ID
```

This proves all core architectural seams before implementing the full control panel.

---

# 57. Recommended second milestone

Add swarm because it exercises the important custom behavior:

- categorical grouping;
- nested color groups;
- jitter;
- mean/error overlays;
- stable row mapping;
- group statistics.

If swarm can be ported cleanly in pure TS, the case for adding a dataframe library becomes much weaker.

After the swarm milestone, explicitly review whether Arquero would materially simplify the remaining implementation.

---

# 58. Decisions that should NOT be reopened without evidence

The next project should treat these as current decisions:

1. NicePool web is static/browser-native.
2. No Python server is required.
3. No Pyodide for NicePool.
4. No SQL-based architecture.
5. Plotly.js remains the plotting engine.
6. Core scientific/data logic is TypeScript.
7. Pure TS rows are the initial dataframe replacement.
8. Vue may implement the UI but is hidden behind a framework-neutral public API.
9. Stable row IDs are fundamental.
10. Selection works in both directions between parent and NicePool.
11. State is JSON-serializable and versioned.
12. Python NicePool is the behavioral oracle during migration.
13. Statistical semantics are explicitly owned by NicePool.
14. Arquero is the first dataframe library to evaluate only if pure TS becomes demonstrably cumbersome.
15. Danfo is not the default merely because Python NicePool uses pandas.

---

# 59. Decisions that require explicit review if encountered

The next implementation chat should not guess on these if the source behavior cannot determine them.

## 59.1 Swarm jitter reproducibility

Question:

- Is Python jitter currently deterministic?
- Should the web version intentionally change to deterministic row-ID-based jitter?

Recommendation:

- deterministic jitter is preferable for a stable scientific UI.

But verify current behavior before changing it.

---

## 59.2 Exact histogram edge semantics

If Python/Numpy bin behavior cannot be matched unambiguously from fixtures, stop and compare concrete expected data.

Do not accept a visually similar histogram with different bin membership.

---

## 59.3 Config JSON naming compatibility

Decide whether external/persisted JSON should retain Python snake_case fields or move to camelCase with versioned migration.

Recommendation:

- camelCase TypeScript internals;
- explicit versioned serialization layer;
- support importing existing Python-style state only if there is a real migration use case.

---

## 59.4 Whether interval stats are core NicePool UI

`algorithms/intv_stats.py` is inside NicePool and should be ported as an algorithm if it is part of NicePool behavior, but implementation should verify where it is currently invoked.

Do not invent new UI for it merely because the module exists.

---

## 59.5 Very large datasets

Do not introduce workers/Arquero/WASM until representative data demonstrates that main-thread pure TS is inadequate.

---

# 60. Non-goals for initial rewrite

Do not expand the project into:

- general pandas emulation;
- arbitrary Python execution;
- arbitrary dataframe notebook functionality;
- SQL querying;
- a backend service;
- image/raster viewing;
- general scientific statistics beyond existing NicePool behavior;
- collaboration/multi-user persistence;
- cloud database storage;
- custom Plotly replacement;
- generic spreadsheet editing.

The initial objective is faithful standalone NicePool.

---

# 61. Definition of feature-complete web NicePool

The rewrite can be called feature-complete when it can:

1. initialize from rows or CSV;
2. discover/configure columns;
3. filter by configured pre-filter columns such as `roi_id`;
4. select x/y/group/color fields;
5. render all current plot types;
6. preserve current transformations/statistics;
7. support 1–4 plot layouts;
8. show current/raw/summary plot controls;
9. support linked selection across plots;
10. optionally synchronize a table;
11. emit row selections to a parent;
12. accept parent-driven row selection;
13. expose get/set state;
14. accept caller-supplied initial config;
15. update/replace source data;
16. sanitize state against new schemas;
17. support dark/light theme;
18. support structured plot summaries;
19. support named presets;
20. optionally persist config locally;
21. run entirely from a static build;
22. expose no Vue implementation details to thin callers.

---

# 62. Example thin-client integration

The target usage should eventually be approximately this simple:

```ts
import {
  NicePool,
  type NicePoolSelectionChangeEvent,
} from '@nicewidgets/nicepool-web';

const nicePool = new NicePool(
  document.querySelector('#nicepool')!,
  {
    data: {
      type: 'csv-url',
      url: '/analysis/velocity.csv',
    },

    uniqueRowIdColumn: 'pool_row_id',

    preFilterColumns: [
      'accept',
      'channel',
      'roi_id',
    ],

    initialPlotConfig: {
      layout: '1x2',
      plotStates: [
        {
          plotType: 'scatter',
          xColumn: 'x',
          yColumn: 'velocity',
        },
        {
          plotType: 'swarm',
          groupColumn: 'condition',
          yColumn: 'velocity',
        },
      ],
    },

    callbacks: {
      onSelectionChange(event: NicePoolSelectionChangeEvent) {
        parentViewer.selectRows(event.rowIds);
      },

      onRowClick(event) {
        parentViewer.openRow(event.rowId);
      },

      onError(event) {
        console.error(event);
      },
    },
  },
);
```

And external selection should be:

```ts
nicePool.setSelectedRowIds([
  'row-123',
  'row-456',
]);
```

The parent should never need:

- the Vue app instance;
- component refs;
- Plotly trace indexes;
- NicePool internal arrays.

---

# 63. Event alternative for non-callback callers

The root element should also dispatch events such as:

```text
nicepool:ready
nicepool:selection-change
nicepool:row-click
nicepool:group-click
nicepool:state-change
nicepool:filter-change
nicepool:error
```

Example:

```ts
container.addEventListener(
  'nicepool:selection-change',
  event => {
    const detail = (event as CustomEvent).detail;
    console.log(detail.rowIds);
  },
);
```

Callbacks and DOM events should originate from the same internal event dispatcher so they cannot diverge in behavior.

---

# 64. API versioning

The public API should have its own version independent of persistence schema.

Recommended:

```ts
export const NICEPOOL_API_VERSION = 1;
export const NICEPOOL_STATE_SCHEMA_VERSION = 1;
```

Breaking event/state changes require explicit version changes/migrations.

This is important because NicePool is intended to be embedded by thin clients.

---

# 65. Error policy for embedded use

Never rely only on toast notifications.

For each recoverable error:

- show useful inline widget feedback;
- call/dispatch `onError`;
- preserve widget usability when possible.

For fatal initialization errors:

- reject `ready()`;
- render a clear error state;
- dispatch `nicepool:error`.

Example fatal errors:

- no data source;
- stable row ID column required but absent under strict configuration;
- malformed initial data contract.

Example recoverable errors:

- stale plot group column;
- nonnumeric histogram column;
- missing optional pre-filter;
- invalid preset field.

---

# 66. Accessibility and keyboard behavior

Preserve Escape-to-clear selection.

Document modifier behavior for extending selections.

Use actual `<button>`, `<select>`, labels, and keyboard-accessible dialogs.

Do not rely only on Plotly hover text to communicate essential state.

---

# 67. Build and deployment

Recommended development stack:

```text
TypeScript
Vite
Vue 3
Plotly.js
Papa Parse
Vitest
Playwright
```

Produce:

- ESM package build;
- standalone demo application;
- CSS artifact;
- type declarations.

A static demo page should be deployable without server code.

---

# 68. Bundle-size discipline

Plotly.js can be a large dependency.

During initial parity work, prioritize correctness.

After parity:

- evaluate partial Plotly bundles if practical;
- lazy-load optional table/preset UI only if useful;
- do not add a large dataframe library casually.

Bundle size is another reason to avoid Danfo/Arquero/DuckDB until they provide concrete value.

---

# 69. Migration workflow for the new local ChatGPT project

When starting the new local project linked to the repository:

1. provide this roadmap as the plan;
2. instruct the agent that `src/nicewidgets/nicepool/` is the current Python behavioral reference;
3. do not ask it to convert unrelated `nicewidgets`;
4. have it inspect the Python NicePool source before implementing each feature;
5. create parity fixtures first;
6. implement TypeScript core before recreating UI;
7. keep a Python→TypeScript feature parity table;
8. ask rather than invent behavior where the source and fixtures do not resolve a scientific/detail question;
9. make normal implementation decisions autonomously when they do not change scientific behavior or public contracts.

Suggested opening prompt for that project:

```text
We are rewriting src/nicewidgets/nicepool from Python/NiceGUI/pandas into
a standalone static TypeScript web widget. Read nicepool-dev-roadmap.md
first and treat it as the architecture/implementation plan. The existing
Python code in src/nicewidgets/nicepool is the behavioral source of truth.

Do not mechanically translate pandas or NiceGUI. Preserve behavior and
statistical semantics behind the boundaries defined in the roadmap.

Start with Phase 0/Phase 1: establish parity fixtures and implement the
pure TypeScript core before building the full Vue UI. Ask before making a
behavioral/scientific change that cannot be resolved from the source.
Make ordinary senior-development implementation decisions yourself.
```

---

# 70. Final architecture

The intended final dependency direction is:

```text
                    thin parent application
                           │
             init options / methods / events
                           │
                           ▼
                 ┌──────────────────┐
                 │ NicePool public  │
                 │ JS/TS API        │
                 └────────┬─────────┘
                          │
                ┌─────────┴─────────┐
                │                   │
                ▼                   ▼
       ┌─────────────────┐   ┌───────────────┐
       │ Vue UI shell    │   │ persistence / │
       │ controls/table  │   │ CSV adapters  │
       └────────┬────────┘   └───────┬───────┘
                │                    │
                └──────────┬─────────┘
                           ▼
                 ┌──────────────────┐
                 │ NicePool model / │
                 │ state / selection│
                 └────────┬─────────┘
                          │
                 ┌────────┴─────────┐
                 │ pure TypeScript  │
                 │ data/statistics  │
                 │ algorithms       │
                 └────────┬─────────┘
                          │
                 ┌────────▼─────────┐
                 │ FigureGenerator  │
                 │ Plotly.js specs  │
                 └────────┬─────────┘
                          │
                          ▼
                       Plotly.js
```

No Python is present in the runtime path.

No parent application is coupled to Vue.

No parent application is coupled to Plotly.

No dataframe-library API is exposed publicly.

That is the key architectural target.

---

# 71. Summary of the recommended rewrite

The current Python NicePool already has useful conceptual boundaries:

```text
PlotState
DataFrameProcessor
FigureGenerator
selection handler
plot summaries
controller/control panel
table adapter
preset/config validation
scientific algorithms
```

The rewrite should preserve those **responsibilities**, not necessarily those exact classes.

The central migration is:

```text
pandas
  ↓
small NicePool-owned TypeScript data/statistics layer
```

not:

```text
pandas
  ↓
browser clone of pandas
```

The library decision is deliberately reversible. If pure TypeScript becomes cumbersome, put Arquero behind the data-processing boundary. Do not redesign the widget around it.

The long-term value of the rewrite is not just that NicePool runs without Python. The new architecture makes NicePool a genuinely reusable web component with a stable integration contract:

```text
caller supplies data + configuration
NicePool renders/manages plots
NicePool emits stable row IDs and state
caller can drive selection/configuration back into NicePool
```

That public boundary should be treated as a core product requirement from the first implementation milestone.
