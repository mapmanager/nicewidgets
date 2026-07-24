# NiceWidgets Phase 1 — Standalone repository extraction

## 1. Source and destination

| Role | Path |
|------|------|
| Read-only source | `/Users/cudmore/Sites/cs_project/cloudscope/` |
| Read-only sibling precedent | `/Users/cudmore/Sites/cs_project/acqstore/` |
| Writable destination | `/Users/cudmore/Sites/cs_project/nicewidgets/` |

Execution specification: `tmp/handoff-nicewidgets-standalone-repository-final-v2.md`.

## 2. CloudScope was read-only

Confirmed. No tracked files were edited, staged, committed, or otherwise modified in CloudScope.

## 3. AcqStore was read-only

Confirmed. No tracked files were edited, staged, committed, or otherwise modified in AcqStore. AcqStore was inspected only for packaging, CI, `.gitignore`, source-ZIP, and documentation precedents.

## 4. Exact extraction inventory (classification)

### Runtime package (`src/nicewidgets/`)

All Python modules under the CloudScope `src/nicewidgets/` tree except relocated demos and the empty leftover tests directory, including:

- top-level: `__init__.py`, `compact_select_styles.py`, `gui_defaults.py`, `plotly_axis_layout.py`, `plotly_layout_margins.py`, `plotly_theme.py`
- packages: `aggrid_common/`, `aggrid_gold_standard/`, `contrast_widget/`, `echart_widget/`, `image_toolbar_widget/`, `nicepool/` (including `algorithms/`), `plotly_plot/`, `raster_viewer/` (backend + frontend), `smart_expansion_widget/`, `table_widget/`, `tree_widget/`, `upload_widget/`, `utils/`
- colocated package READMEs and `nicepool/algorithms/intv_stats.md` (kept as package documentation assets)

### Tests

- `tests/nicewidgets/*.py` (complete suite from CloudScope)
- added `tests/__init__.py`

### Examples (relocated from package)

- `examples/raster_viewer/` from `src/nicewidgets/raster_viewer/demo/`
- `examples/table_widget/demo_app.py` from `src/nicewidgets/table_widget/demo_app.py`

### Developer scripts

- `scripts/try_plotly_plot_event_overlays.py`
- `scripts/try_plotly_plot_widget.py`
- `scripts/make_source_zip.sh` (new)

### Public documentation

- New focused MkDocs site under `docs/`
- Adapted API notes from CloudScope `docs-dev/nicewidgets_api.md` → `docs/api/widget-api.md`

### Developer documentation

- `docs-dev/nicewidgets_api.md` (source copy retained)
- `docs-dev/migrations/nicewidgets-phase-1.md` (this report)

### Generated / local artifacts (excluded)

- `__pycache__/`, `.pytest_cache/`, `.DS_Store`
- empty leftover `src/nicewidgets/raster_viewer/tests/` (no `.py` tests after ticket 119 relocation)
- `site/`, `dist/`, `tmp/`, `*.zip` (local only / ignored)

### CloudScope-owned consumer code (not copied)

All of `src/cloudscope/`, CloudScope tests outside `tests/nicewidgets/`, and AcqStore sources.

## 5. Files copied

- `cloudscope/src/nicewidgets/` → `nicewidgets/src/nicewidgets/` (then demos/tests adjusted)
- `cloudscope/tests/nicewidgets/` → `nicewidgets/tests/nicewidgets/`
- `cloudscope/scripts/nicewidgets/*.py` → `nicewidgets/scripts/` (flattened)
- `cloudscope/LICENSE` → `nicewidgets/LICENSE`
- `cloudscope/.python-version` → `nicewidgets/.python-version`
- `cloudscope/docs-dev/nicewidgets_api.md` → `docs-dev/nicewidgets_api.md` and adapted into `docs/api/widget-api.md`

## 6. Files relocated

| From | To | Reason |
|------|----|--------|
| `src/nicewidgets/raster_viewer/demo/` | `examples/raster_viewer/` | Repository examples; not runtime API; must not ship in wheel |
| `src/nicewidgets/table_widget/demo_app.py` | `examples/table_widget/demo_app.py` | Same; demo-only `-m` path intentionally dropped |

## 7. Files intentionally excluded

- Empty `src/nicewidgets/raster_viewer/tests/` leftover directories
- CloudScope application / AcqStore / server code
- Ticket reports and CloudScope docs that only mention NiceWidgets incidentally
- Generated caches and editor state

## 8. Files modified after copying

| File | Why |
|------|-----|
| `src/nicewidgets/raster_viewer/backend/raster_service.py` | Replace `from cloudscope.utils.logging import get_logger` with `from nicewidgets.utils.logging import get_logger` |
| `src/nicewidgets/table_widget/README.md` | Demo command → `uv run python examples/table_widget/demo_app.py` |
| `src/nicewidgets/raster_viewer/README.md` | Demo command → `uv run python examples/raster_viewer/nicegui_raster_demo.py` |
| `examples/table_widget/demo_app.py` | Update entry-point docstring to example path |
| `docs/api/widget-api.md` | Standalone framing; preserve widget contract content |

## 9. Public API preservation

- No public runtime modules, classes, or functions were renamed.
- Package root still does not re-export symbols.
- Accepted non-API change: removal of demo-only `python -m nicewidgets.table_widget.demo_app` (not a supported runtime API per handoff v2).
- No compatibility shims added.

## 10. CloudScope runtime coupling removed

Only coupling found under NiceWidgets runtime:

```python
# before (CloudScope tree)
from cloudscope.utils.logging import get_logger

# after (standalone)
from nicewidgets.utils.logging import get_logger
```

in `raster_viewer/backend/raster_service.py`. Existing `nicewidgets.utils.logging` already provided `get_logger`.

Zero remaining runtime imports from `cloudscope`, `acqstore`, or `acqstore_server`.

## 11. Dependency classifications

| Dependency | Classification | Notes |
|------------|----------------|-------|
| `nicegui` | required runtime | Floor `>=3.14.0` (see Finalization Action 2) |
| `numpy` | required runtime | Floor `>=1.26.0` |
| `pandas` | required runtime | Floor `>=2.0.0` |
| `plotly` | required runtime | Floor `>=5.18.0` |
| `pillow` | required runtime | Floor `>=10.0.0`; unconditional use in `raster_service` |
| `platformdirs` | required runtime | Floor `>=4.0.0`; logging file paths |
| `pyperclip` | optional runtime (`desktop` extra) | `try/except ImportError` in tree/clipboard |
| `pyperclipimg` | optional runtime (`desktop` extra) | Lazy import for native PNG clipboard |
| `pywebview` | optional runtime (`desktop` extra) | Lazy import in desktop detection |
| `pytest`, `pytest-cov`, `ruff` | development / test (`dev` group) | |
| MkDocs stack | documentation (`docs` group) | |

Exact resolved versions live in `uv.lock`. Lower-bound rationale is recorded in the Finalization section.

## 12. Demo and test relocation decisions

Per handoff v2 (decided):

- Demos → `examples/`; documentation updated; no shims; not in wheel.
- Empty `raster_viewer/tests/` excluded (no Python tests).
- Primary suite remains `tests/nicewidgets/`.

## 13. Documentation ownership decisions

Copied / adapted only NiceWidgets-owned material:

- `docs-dev/nicewidgets_api.md` (developer record + public API notes adaptation)
- New concise public docs (`index`, guide, developer pages)

Not copied: CloudScope architecture pages, ticket reports, packaging docs, AcqStore docs, or incidental mentions.

## 14. Baseline CloudScope NiceWidgets test results

- Command: `cd cloudscope && uv run python -m pytest tests/nicewidgets -ra --tb=no`
- Environment: CloudScope `.venv`, Python 3.12.12, pytest 9.1.1
- **578 passed**, **0 skipped**, **0 failed**
- Warnings: 2× `RuntimeWarning: All-NaN slice encountered` in `test_array_to_png_data_uri_handles_all_nan`
- GUI/display: suite runs headless; no display required for this baseline
- Path injection: uses CloudScope `pyproject.toml` `pythonpath = ["src"]` (includes in-tree nicewidgets)

## 15. Standalone test results

- Command: `cd nicewidgets && uv run --no-sync pytest -ra --tb=no`
- **578 passed**, **0 skipped**, **0 failed**
- Same 2 All-NaN warnings
- Matches baseline exactly

## 16. Exact validation commands

```bash
uv lock --check
uv sync --frozen
uv sync --frozen --group dev --group docs
uv run --no-sync pytest -ra
DISABLE_MKDOCS_2_WARNING=true uv run --no-sync mkdocs build --strict
uv build
./scripts/make_source_zip.sh nicewidgets_20260724_v1.zip
```

## 17. Lockfile result

`uv.lock` generated and `uv lock --check` succeeded. Frozen syncs for default, `dev`, and `docs` groups succeeded.

## 18. MkDocs strict result

`DISABLE_MKDOCS_2_WARNING=true uv run --no-sync mkdocs build --strict` succeeded. Site written to local `site/` (gitignored; not committed).

## 19. Wheel inspection result

- File: `dist/nicewidgets-0.1.0-py3-none-any.whl` (~227 KiB)
- Contains only `nicewidgets/` package modules + `nicewidgets-0.1.0.dist-info/`
- **No** `tests/`, `docs/`, `docs-dev/`, `examples/`, `scripts/`, `tmp/`, demos, caches

## 20. Sdist inspection result

- File: `dist/nicewidgets-0.1.0.tar.gz`
- Contains packaging metadata (`pyproject.toml`, `PKG-INFO` with `License-Expression: GPL-3.0-only`), `README.md`, `LICENSE`, and `src/nicewidgets/`
- No demos, tests, docs trees, `tmp/`, caches
- **Superseded by Phase 1 finalization:** full `LICENSE` is now included via `license-files = ["LICENSE"]` (see Finalization section below).

## 21. Source-ZIP inspection result

- Initial extraction ZIP: `nicewidgets_20260724_v1.zip`
- **Finalization ZIP:** `nicewidgets_20260724_v2.zip` (see Finalization section)

## 22. CloudScope consumer-compatibility audit

Audited all non-package CloudScope / colocated consumer imports of `nicewidgets` (244 import statements across application, tests, scripts, and the AcqStore demo app living inside CloudScope).

**Result:** every imported module/symbol path is preserved unchanged in the standalone package.

Future CloudScope change (deferred; not done in this phase): replace the in-tree `src/nicewidgets/` with a dependency on the standalone `nicewidgets` package. No consumer source edits are required for import paths once the package is on `PYTHONPATH` / installed.

Representative preserved imports include:

- `nicewidgets.raster_viewer.frontend.plotly_viewer.PlotlyRasterViewer`
- `nicewidgets.plotly_plot.widget.PlotlyPlotWidget`
- `nicewidgets.table_widget.table_widget.TableWidget`
- `nicewidgets.tree_widget.tree_widget.TreeWidget`
- `nicewidgets.nicepool.NicePool` / `NicePoolConfig`
- `nicewidgets.utils.logging.setup_logging`
- `nicewidgets.gui_defaults.setUpGuiDefaults`

## 23. Pre- and post-work sibling-repository status

### Preflight

| Repo | Branch | HEAD | `git status --short` |
|------|--------|------|----------------------|
| CloudScope | `main` | `7233f6155c2d053944962ea914259b1fc0841d6a` | clean |
| AcqStore | `main` | `5133b6839a225d1ba85b7cd07a852d67e0c82475` | clean |

### Post-work

| Repo | HEAD | `git status --short` |
|------|------|----------------------|
| CloudScope | `7233f6155c2d053944962ea914259b1fc0841d6a` | clean |
| AcqStore | `5133b6839a225d1ba85b7cd07a852d67e0c82475` | clean |

No Cursor-introduced tracked changes in either sibling.

## 24. Deferred issues

1. Wire CloudScope / in-monorepo AcqStore demo app to consume standalone NiceWidgets (dependency + remove in-tree package) — next migration phase.
2. Optional: publish to PyPI / configure package publishing automation — explicitly out of scope.
3. Initialize Git / create GitHub remote — user action after review.

~~Previously deferred: include full LICENSE in wheel/sdist~~ — resolved in Phase 1 finalization.

## 25. Confirmation that Git was not initialized

Confirmed: `cs_project/nicewidgets/` has no `.git` directory. No GitHub repository was created. No push was performed.

---

# Phase 1 finalization (follow-up)

Specification: `tmp/handoff-nicewidgets-phase-1-finalization.md`.

Writable destination only: `cs_project/nicewidgets/`. CloudScope and AcqStore remained read-only.

## Actions completed

### Action 1 — License in distributions

- Added `license-files = ["LICENSE"]` under `[project]` (PEP 639 / `uv_build`).
- Preserved `license = "GPL-3.0-only"`.
- Wheel contains `nicewidgets-0.1.0.dist-info/licenses/LICENSE`.
- Sdist contains `nicewidgets-0.1.0/LICENSE`.
- PKG-INFO includes `License-File: LICENSE`.

### Action 2 — Dependency lower bounds

| Package | Final floor | Rationale |
|---------|-------------|-----------|
| `nicegui` | `>=3.14.0` | Preserved intentional CloudScope bump from `3.10.0` (Jul 2026 packaging commit); current widget stack targets that floor |
| `numpy` | `>=1.26.0` | Conservative Py3.12-era floor; code uses longstanding ndarray/`nanmin`/`nanmax` APIs, not tip lockfile versions |
| `pandas` | `>=2.0.0` | DataFrame APIs used are pandas-2 capable; no pandas-3-only requirement |
| `plotly` | `>=5.18.0` | Modern plotly.py figure/dict APIs; no code dependence on 6.7 tip |
| `pillow` | `>=10.0.0` | `Image.fromarray` / `Image.open` long available; not tip 12.x |
| `platformdirs` | `>=4.0.0` | `user_config_dir` on platformdirs 4.x; not tip 4.9.x |

Exact resolved versions remain in `uv.lock`. Optional desktop extras unchanged and still optional.

### Action 3 — Desktop optional extra in CI

`.github/workflows/tests.yml` matrix profiles:

- **core:** `uv sync --frozen --group dev` then `uv run --no-sync pytest -ra` (with coverage)
- **desktop:** `uv sync --frozen --group dev --extra desktop` then `uv run --no-sync pytest -ra`

Coverage runs only on core (`--cov=nicewidgets`).

### Action 4 — Desktop import validation

- Added `tests/nicewidgets/test_desktop_extra_imports.py` with `desktop_extra` marker.
- Skips cleanly without the extra; must pass under desktop profile.
- Local desktop profile: import check `import pyperclip, pyperclipimg, webview` succeeded; test passed (no skip).

### Action 5 — Coverage targeting

CI coverage uses `--cov=nicewidgets` (package-based), not `--cov=src/nicewidgets`.

### Action 6 — Ruff baseline

- `uv run --no-sync ruff check .` passes.
- Fixed UP037/B009/F401 auto-fixes, E402 import order in echart tests, unused `state` in selection-handler helper.
- Adjusted tree-widget browser clipboard test to patch `is_pywebview_desktop` after removing unused `nicegui.app` import (behavior-preserving).
- CI now runs Ruff; intentional-non-enforcement comment removed.

### Action 7 — Example validation

- `python -m compileall -q examples` succeeds.
- Example `nicewidgets.*` imports resolve.
- README / docs paths use `examples/...`; no stale `-m nicewidgets.table_widget.demo_app`.
- Limitation documented: demos still call blocking `ui.run()` at `__main__`; not redesigned; CI compiles only.
- Example compilation added to CI.

### Action 8 — CloudScope-specific runtime wording

Removed stale Option C / CloudScope-as-API wording from runtime docstrings/comments. Post-edit search:

```bash
rg -n "CloudScope|Option C" src/nicewidgets
```

returns no matches. Migration docs under `docs-dev/migrations/` retain historical references intentionally.

### Action 9 — Rebuild and artifacts

Validation commands run (with notes):

```bash
uv lock && uv lock --check
uv sync --frozen
uv sync --frozen --group dev --group docs
uv sync --frozen --group dev --extra desktop
uv sync --frozen --group dev --group docs --extra desktop   # for local docs+desktop together
uv run --no-sync ruff check .
uv run --no-sync pytest -ra   # core and desktop profiles
uv run --no-sync python -m compileall -q examples
DISABLE_MKDOCS_2_WARNING=true uv run --no-sync mkdocs build --strict
uv build
./scripts/make_source_zip.sh nicewidgets_20260724_v2.zip
```

## Finalization files changed

| Path | Why |
|------|-----|
| `pyproject.toml` | `license-files`; conservative floors; pytest `desktop_extra` marker |
| `uv.lock` | Regenerated for new floors |
| `.github/workflows/tests.yml` | core/desktop matrix; Ruff; example compile; `--cov=nicewidgets` |
| `tests/nicewidgets/test_desktop_extra_imports.py` | Desktop import check |
| `tests/nicewidgets/test_echart_widget.py` | Ruff E402 import order |
| `tests/nicewidgets/test_selection_handler.py` | Remove unused local |
| `tests/nicewidgets/test_tree_widget_smoke.py` | Patch `is_pywebview_desktop` instead of removed `app` |
| Multiple `src/nicewidgets/**` modules | Ruff auto-fixes + standalone wording |
| `docs-dev/migrations/nicewidgets-phase-1.md` | This finalization record |

## Finalization results

| Check | Result |
|-------|--------|
| Core tests | **578 passed**, **1 skipped** (desktop_extra), 0 failed |
| Desktop tests | **579 passed**, 0 skipped, 0 failed |
| Ruff | All checks passed |
| Desktop imports | `pyperclip`, `pyperclipimg`, `webview` OK |
| Examples | compileall + import smoke OK |
| MkDocs `--strict` | Passed |
| Wheel | `dist/nicewidgets-0.1.0-py3-none-any.whl` — license present; no tests/docs/examples/scripts/tmp |
| Sdist | `dist/nicewidgets-0.1.0.tar.gz` — `LICENSE` present |
| Source ZIP | `nicewidgets_20260724_v2.zip` — includes docs-dev/migrations; excludes tmp/site/dist/.venv |
| CloudScope | Unchanged (clean) |
| AcqStore | Unchanged (clean) |
| Git | Not initialized |

## Sibling status (finalization)

### Preflight

Both clean (`git status --short` empty).

### Post-work

Both clean. No Cursor-introduced tracked changes.
