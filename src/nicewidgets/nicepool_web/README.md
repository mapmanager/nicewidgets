# NicePool Web

NicePool Web is a browser-native rewrite of the Python/NiceGUI NicePool widget.
The first usable slice supports authoritative dataset replacement, stable row
identity, filtering, scatter, swarm, box, violin, histogram, and cumulative
histogram preparation, plot-specific summaries,
linked selection, four persistent plot slots, saved single-plot presets, a Vue
component, and a framework-neutral Custom Element.

Scatter supports numeric or categorical X columns, numeric Y columns, and an
optional categorical **Color by** column. Swarm uses **Group** for categorical
X-axis positions and **Color by** to subdivide each group by color and offset.
Box and violin reuse the same grouped observations and add quartile summaries.
Histogram variants use a user-controlled bin count (default 50) and
NicePool-owned, globally aligned bin edges shared by rendering and summaries.

The widget defaults to a dark theme for both controls and Plotly. Its active
plot has a collapsible Display options panel for legend visibility/position,
Plotly toolbar visibility, combined axis chrome, and independent horizontal and
vertical grid lines. Theme is workspace presentation; display options are part
of each serializable plot state.

## Development

```bash
npm install
npm run dev
npm test
npm run build
```

The build emits three artifacts: the standalone SPA (`dist/`), an ESM library
with Vue and Plotly externalized (`dist-lib/`), and a self-contained registered
Custom Element module (`dist-element/`). `element-demo.html` is the plain-HTML
development host. The ESM library includes generated TypeScript declarations.

The standalone development host generates deterministic demonstration data.
Its velocity column intentionally contains negative values, extreme values, and
explicit missing values for exercising plot controls. `public/sample.csv` is a
small human-readable integration fixture; CSV adapters must convert blank cells
to JSON `null` before calling `setData`.

## NiceGUI integration demo

Build the Custom Element, then run the minimal Python host from the
`nicewidgets` repository root:

```bash
cd src/nicewidgets/nicepool_web
npm run build
cd ../../..
uv run python src/nicewidgets/nicepool_web/nicegui_demo.py
```

The demo converts a pandas DataFrame to the JSON data contract and exercises
dataset replacement, state read/write, Python-driven primary selection, and
browser-to-Python selection/state events. A NiceGUI switch changes both Quasar
and NicePool between the dark default and light theme. Its `NicePoolWebView`
class is a thin
bridge; all plotting and authoritative selection behavior remains in the web
engine.

## Package boundaries

- `src/core/`: framework- and DOM-independent data, state, statistics, and selection.
- `src/plots/`: prepared plot data, summaries, and the isolated Plotly adapter.
- `src/vue/`: thin Vue presentation components.
- `src/element/`: the `<nice-pool>` host boundary for plain HTML and NiceGUI.
- `src/app/`: standalone development application.

See `docs/architecture.md`, `docs/data-and-selection-api.md`,
`docs/plot-and-summary-semantics.md`, `docs/state-and-presets.md`, and
`docs/preset-persistence.md` for the behavioral contracts.
