# NiceWidgets — Agent Instructions

## Repository role

`nicewidgets` is an independent Git repository containing reusable NiceGUI
widgets for scientific and desktop applications.

- Distribution name: `nicewidgets`
- Python import package: `nicewidgets`
- Source: `src/nicewidgets/`
- Tests: `tests/`
- Browser tests: `browser_tests/`
- User documentation: `docs/`
- Development notes: `docs-dev/`
- Examples: `examples/`

NiceWidgets provides general-purpose widgets (Plotly plots, AG Grid tree/table,
raster viewer, contrast, upload, nicepool, etc.). Widgets MUST be reusable and
must **not** know about CloudScope, AcqStore, or the server layer.

> Coexistence note: This file is the primary instruction file when this repo is
> the working root (e.g. a Codex project with `nicewidgets/` primary). When this
> repo is opened as part of the outer `cs_project/` workspace (e.g. in Cursor),
> the outer `cs_project/AGENTS.md` plus `cs_project/.cursor/rules/` provide
> workspace-wide guidance and take precedence for cross-repo scope; this file
> stays repo-local and must not contradict it.

## Attached sibling repositories

`nicewidgets` has **no** source dependencies on sibling repositories. Consumers
depend on it, not the other way around.

| Repository | Local path | Relationship |
|---|---|---|
| CloudScope App | `../cloudscope-app/` | Depends on `nicewidgets` (editable path dep) |

`nicewidgets` MUST NOT import from `cloudscope-app`, `acqstore`, or
`acqstore-server`. If a widget seems to need CloudScope- or acquisition-specific
behavior, stop and report — that logic belongs in the consumer, passed in via
the widget's public API.

## Package boundaries

Put code in `nicewidgets` when it implements:

- reusable NiceGUI widgets and their public APIs;
- generic Plotly / AG Grid / ECharts styling and helpers;
- widget-level state and rendering that works without any specific app.

Do not place these in `nicewidgets`:

- CloudScope- or acquisition-specific models, loaders, or analysis;
- application orchestration, controllers, or app-specific views;
- server endpoints or transport behavior.

## Default task scope

Work only in `nicewidgets` unless the task explicitly includes another
repository.

- Start with files named by the user and their direct dependencies.
- Do not make unrelated cleanup or consistency edits.
- Do not add abstractions for hypothetical future requirements.
- Do not add or change production dependencies without asking first.
- Do not modify GUI behavior unless GUI work is part of the request.
- Ask a focused question, with a recommended answer, when a material decision
  remains ambiguous after inspecting the relevant code.

## Curated public API (frozen)

New or updated `__init__.py` files are **empty by default**. Do not add imports,
`__all__`, docstrings, or side effects unless the task explicitly extends a
curated public API surface.

Frozen curated allowlist (do not modify without an explicit request that names
the file and symbols):

- `src/nicewidgets/nicepool/__init__.py`
- `src/nicewidgets/upload_widget/__init__.py`

Elsewhere, import via full module paths.

## Environment and commands

Run commands from the `nicewidgets/` repository root. Use `uv run`.

```bash
uv sync
uv run pytest path/to/test_file.py   # focused first
uv run pytest                        # full suite
uv run ruff check src tests
```

`raster_viewer_widget` is type-checked under strict mypy:

```bash
uv run mypy
```

## Verification

Verification must match the change.

- Source or API changes: run focused tests, then the full suite when practical.
- Formatting or lint-sensitive changes: run the relevant Ruff check.
- `raster_viewer_widget` changes: run mypy (strict).
- GUI behavior changes: run the relevant example/browser test and inspect actual
  browser behavior when the environment allows.

Do not claim a GUI, AG Grid, Plotly, or raster-viewer problem is fixed solely
because unit tests pass. If live verification is blocked, describe the candidate
change as unverified and state exactly what remains to be tested. Do not weaken a
meaningful test merely to make it pass.

## GUI and callback caution

Treat browser event behavior as an end-to-end interaction, not only as a Python
callback. Before changing NiceGUI, Plotly, or AG Grid callbacks:

1. inspect the existing event path;
2. distinguish user-initiated events from programmatic state synchronization;
3. verify API names and call conventions against current authoritative
   documentation (NiceGUI / AG Grid / Plotly) or verified repository usage;
4. avoid combining callback changes with unrelated refactoring.

Do not invent AG Grid / Plotly option keys, method names, event names, or return
shapes. If unsure, cite the source of truth and confirm before implementing.

## Coding conventions

- Keep changes small, direct, and maintainable.
- Prefer KISS and DRY without speculative shared modules.
- Use type annotations and Google-style docstrings (`Args`, `Returns`,
  `Raises`) for public APIs.
- Fail clearly on invalid input rather than silently guessing.
- Preserve existing architecture and naming unless the task is a deliberate
  refactor.

## Documentation and ticket reports

Do not update the repository-root `README.md` unless the task explicitly
requests a README change.

Do not create a `docs-dev/cursor_tickets/` report by default. Create one only
when the user explicitly identifies the work as a tracked implementation ticket
or requests a report. Use the next unused three-digit prefix. The
`cursor_tickets/` name is a project convention regardless of which agent writes
the report. Record: requested scope; repositories and files changed; important
decisions; verification performed; unresolved or unverified behavior.

## Search exclusions

Unless the task explicitly requires them, do not inspect or search:

- `.venv/`, `venv/`, `__pycache__/`, and tool caches;
- `build/`, `dist/`, `site/`, `zips/`, and generated output;
- `*.zip`, `*.tar`, `*.tar.gz`, and `*.whl`;
- `.git/`;
- large generated data or binary assets under `docs/assets/`.

Do not treat old monorepo paths or archived development notes as current
architecture when they conflict with the present repository.

## Git discipline

This directory is an independent Git repository.

- Check `git status` before and after material work.
- Preserve unrelated user changes.
- Do not commit, push, create branches, or open pull requests unless explicitly
  requested.
- For cross-repository work, report and verify changes separately in each
  affected repository.
