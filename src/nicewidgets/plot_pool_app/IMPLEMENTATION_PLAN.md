# Plot Pool App Implementation Plan

Step-by-step implementation for upgrading `plot_pool_app.py` per the specification and Q&A.

---

## Feedback on "idk" Answers

### 1. Theme — Recommendation: KISS version of kymflow

Kymflow uses:
- `app.storage.user["kymflow_dark_mode"]` (persists across sessions)
- `ui.dark_mode()` per page
- `context.init_dark_mode_for_page()` to restore and sync
- `context.toggle_theme(dark_mode)` to toggle, persist, and sync AppState
- AppState.theme_mode for plotting components

For plot pool app (single page, no AppState):
- Use the same pattern but without AppContext/AppState
- `app.storage.user["plot_pool_dark_mode"]` (default `True` = dark)
- `ui.dark_mode()` at page start
- Simple `_toggle_theme(dark_mode)` that toggles `dark_mode.value` and writes to storage
- No ThemeMode enum or AppState needed

**Recommendation:** Implement a small `init_dark_mode()` and `toggle_theme(dark_mode)` in the header or a `theme.py` helper. Keep it in one place and call from the page.

### 2. Header — Recommendation: `ui.header()` (not Quasar QHeader)

| Option | Pros | Cons |
|--------|------|------|
| `ui.header()` | NiceGUI-native, simple, matches kymflow, standard toolbar layout | Fixed to top; no collapse/expand behavior |
| Quasar `QHeader` (nicegui.elements.header) | Layout-aware (view), can show/hide, scroll padding | More complex, tied to client layout, overkill for a single toolbar |

**Recommendation:** Use `ui.header()`:
- Same as kymflow
- Single toolbar, no layout tricks needed
- Simple `with ui.header().classes("items-center justify-between")`

---

## Implementation Steps

### Step 1: Add `schema.py` — CSV discovery and loading

**File:** `nicewidgets/plot_pool_app/schema.py`

**Functions and data:**
- `get_data_dir() -> Path` — Resolve `nicewidgets/data/` relative to package root  
  - `Path(__file__).resolve().parent.parent.parent.parent / "data"`
  - Or: `importlib.resources.files("nicewidgets").parent / "data"` if data is packaged
  - Fallback: `Path(__file__).resolve().parent.parent.parent.parent.parent / "nicewidgets" / "data"` when run from kymflow_outer
- `get_data_csv_files() -> list[str]` — List `.csv` filenames in data dir (sorted)
- `CSV_SCHEMA: dict[str, PlotPoolConfig]` — Map filename → config  
  - `radon_report_db.csv`: `pre_filter_columns=["roi_id"]`, `unique_row_id_col="unique_row_id"`, `db_type="radon_db"`, `app_name="nicewidgets"`
  - `kym_event_report.csv`: `pre_filter_columns=["roi_id"]`, `unique_row_id_col="kym_event_id"`, `db_type="kym_event_db"`, `app_name="nicewidgets"`
- `load_csv_for_file(filename: str) -> pd.DataFrame` — Load CSV, add `unique_row_id` for radon if needed (same logic as `load_csv_and_build_df`), return df
- `DEFAULT_CSV = "radon_report_db.csv"` — First in list, or explicit constant

**Notes:**
- For radon_report_db: no `row_id`, has `path` and `roi_id` → add `unique_row_id = path|roi_id`
- For kym_event_report: already has `kym_event_id`
- Handle missing data dir (empty list, graceful error)

---

### Step 2: Add `header.py` — Modular header component

**File:** `nicewidgets/plot_pool_app/header.py`

**Function:**
```python
THEME_STORAGE_KEY = "plot_pool_dark_mode"

def build_plot_pool_header(
    *,
    github_url: str = "https://github.com/mapmanager/nicewidgets",
) -> ui.dark_mode:
```

**Logic:**
1. `dark_mode = ui.dark_mode()`
2. Restore: `dark_mode.value = app.storage.user.get(THEME_STORAGE_KEY, True)`
3. `with ui.header().classes("items-center justify-between"):`
   - Left: `ui.label("Plot Pool").classes("!text-lg font-bold italic text-white")`
   - Right: `ui.row()` with:
     - Theme toggle button (icon `light_mode` when dark, `dark_mode` when light) — `_toggle_theme()`: flip `dark_mode.value`, `app.storage.user[THEME_STORAGE_KEY] = dark_mode.value`, update icon
     - GitHub link (image or icon) — `open_external(github_url)` (copy from kymflow `navigation.py`)
4. Return `dark_mode` (caller can use if needed)

**Helper:** `open_external(url)` — same as kymflow: `webbrowser.open` if native, else `ui.run_javascript('window.open(...)')`

---

### Step 3: Add `toolbar.py` — Top toolbar

**File:** `nicewidgets/plot_pool_app/toolbar.py`

**Function:**
```python
def build_toolbar(
    *,
    csv_options: list[str],
    default_csv: str,
    on_csv_selected: Callable[[str], None],
    on_open_click: Callable[[], None],
    on_upload_click: Callable[[], None],
) -> ui.select:
```

**Logic:**
1. `with ui.row().classes("w-full items-center gap-3 flex-wrap"):`
   - `ui.select("Files", options=csv_options, value=default_csv, on_change=...)` — label "Files", options=filenames, value=default, on_change calls `on_csv_selected(e.value)`
   - Open button: `ui.button(icon="folder_open", on_click=on_open_click).tooltip("Open CSV file")`
   - Upload button: `ui.button(icon="upload", on_click=on_upload_click).tooltip("Upload CSV file")`
2. Return the `ui.select` so caller can update value if needed

**Placeholder callbacks:** `on_open_click` and `on_upload_click` can be `lambda: ui.notify("Open: coming soon")` for now.

---

### Step 4: Refactor `plot_pool_app.py` — Wire everything

**Changes to `plot_pool_app.py`:**

1. Remove `CSV_PATH` and hardcoded path.
2. Import: `schema`, `header`, `toolbar`.
3. In `home()`:
   - `setUpGuiDefaults()`
   - `ui.page_title("Plot Pool")`
   - `dark_mode = build_plot_pool_header()`
   - `setUpGuiDefaults()` (if not called earlier; kymflow calls it in AppContext; we call once at page start)
   - Get CSV list: `csv_files = schema.get_data_csv_files()`
   - Default: `default_csv = schema.DEFAULT_CSV` (or first in list if DEFAULT_CSV not in list)
   - Create main content container: `main_container = ui.column().classes("w-full flex-1 min-h-0")`
   - Callback `_on_csv_selected(filename: str)`:
     - Load: `df = schema.load_csv_for_file(filename)`
     - Config: `cfg = schema.CSV_SCHEMA.get(filename)` or build from schema
     - Clear `main_container`, then `with main_container:` build new `PlotPoolController(df, config=cfg).build(container=main_container)`
   - Build toolbar: `build_toolbar(csv_options=csv_files, default_csv=default_csv, on_csv_selected=_on_csv_selected, on_open_click=..., on_upload_click=...)`
   - Bootstrap: call `_on_csv_selected(default_csv)` once to populate main content
4. Layout:
   ```
   with ui.column().classes("w-full h-screen flex flex-col p-4 gap-4"):
       # Header is built by build_plot_pool_header (already in DOM)
       build_plot_pool_header()
       build_toolbar(...)
       main_container = ui.column().classes("w-full flex-1 min-h-0 overflow-auto")
       # Bootstrap: _on_csv_selected(default_csv)
   ```

**Important:** Bootstrap must run *after* `main_container` exists. So: create container, build toolbar (with callback that uses `main_container`), then call `_on_csv_selected(default_csv)`.

---

### Step 5: Define CSV schema in `schema.py`

**Per-file config:**

```python
# radon_report_db.csv
# - unique_row_id: construct path|roi_id
# - pre_filter_columns: ["roi_id"]
# - db_type: "radon_db"

# kym_event_report.csv  
# - unique_row_id_col: "kym_event_id"
# - pre_filter_columns: ["roi_id"]
# - db_type: "kym_event_db"
```

Implement `load_csv_for_file()` to:
- Add `unique_row_id` for radon when needed
- Raise or return None if file missing; caller handles

---

### Step 6: Resolve data path robustly

When run as `uv run python -m nicewidgets.plot_pool_app.plot_pool_app` from `kymflow_outer/` or `nicewidgets/`:
- CWD = kymflow_outer
- `__file__` = `.../nicewidgets/src/nicewidgets/plot_pool_app/plot_pool_app.py`
- Package root = `.../nicewidgets/src/nicewidgets/`
- Project root (where `data/` lives) = `.../nicewidgets/`
- So: `Path(__file__).resolve().parent.parent.parent.parent` = `.../nicewidgets/` (plot_pool_app -> plot_pool_app -> nicewidgets -> src)
- Actually: `parent` = plot_pool_app dir, `parent.parent` = nicewidgets pkg, `parent.parent.parent` = src, `parent.parent.parent.parent` = nicewidgets project root. So `.../nicewidgets/` + `data` = `.../nicewidgets/data/`. Correct.

---

### Step 7: Optional global styles

If header looks off, add minimal CSS (e.g. from kymflow `inject_global_styles`):
- Header height
- Button/label alignment

Can be deferred until layout is in place.

---

## File Summary

| File | Purpose |
|------|---------|
| `schema.py` | Data path, CSV list, load logic, CSV_SCHEMA |
| `header.py` | build_plot_pool_header(), theme toggle, GitHub link |
| `toolbar.py` | build_toolbar(), Files select, Open, Upload |
| `plot_pool_app.py` | Page, layout, wiring, bootstrap |

---

## Bootstrap Order

1. `setUpGuiDefaults()`
2. `ui.page_title("Plot Pool")`
3. `build_plot_pool_header()` 
4. Create `main_container`
5. Build toolbar with `on_csv_selected` that clears and rebuilds `main_container`
6. Call `_on_csv_selected(default_csv)` to populate on first load

---

## Edge Cases

- **No data dir or no CSVs:** Show empty select and message in main content (e.g. "No CSV files in nicewidgets/data").
- **Load error:** Catch in `load_csv_for_file`, return None or raise; show `ui.notify` and keep previous content.
- **Schema missing for file:** Fall back to generic config or show error.
- **Default CSV not in list:** Use first file in list.

---

## Testing Checklist

- [ ] Header: "Plot Pool" left, theme + GitHub right
- [ ] Theme toggle persists across reload
- [ ] Files select: radon_report_db.csv, kym_event_report.csv
- [ ] Selecting different CSV rebuilds PlotPoolController
- [ ] Default selection on load: radon_report_db.csv
- [ ] Open/Upload buttons present; placeholder behavior
- [ ] No disclosure triangle (direct build, not build_lazy)
- [ ] Data path works when run from kymflow_outer and from nicewidgets root
