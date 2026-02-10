# nicewidgets UploadWidget — development roadmap

Date: 2026-02-08

This document is the handoff + roadmap for the **nicewidgets UploadWidget** module (NiceGUI upload helper/widget).

## Goal

Provide a **portable, self-contained NiceGUI widget** that lets app developers accept user file uploads and receive
a **ready-to-use list of filesystem `Path` objects** on the server side — with:

- consistent handling of both NiceGUI upload modes (**small in-memory** vs **large temp-file** uploads)
- support for **multi-file “drop batches”**
- optional **progress callbacks**
- optional **cancel support** (best-effort, primarily for post-upload processing)
- clean logging (no huge byte dumps in logs)

The widget is intended to live in:

`nicewidgets/src/nicewidgets/upload_widget/`

and be reusable by:
- kymflow (your app)
- other teams/projects “in the wild”

---

## What we have built so far

### 1) Module structure (self-contained)
Files currently in the module:

- `upload_widget.py` — main widget + batching logic + progress/cancel API
- `normalize.py` — `normalize_uploaded_file(...)` helper that turns NiceGUI’s upload objects into `Path`
- `pairing.py` — helper(s) for grouping/pairing files (kept small and optional)
- `__init__.py` — clean exports
- `readme_upload_widget.md` — module README
- `tutorials/demo_upload_multi.py` — multi-file batch demo
- `tutorials/demo_upload_tif.py` — TIFF demo (optional dependency: `tifffile`, `numpy`)

### 2) Normalization across NiceGUI upload variants
NiceGUI’s `UploadEventArguments.file` can be:

- `LargeFileUpload` → already written to disk; exposes `._path` (a `Path`)
- `SmallFileUpload` → in-memory; typically exposes bytes via `._data` (and may not have a public `.read()`)

We implemented a normalization helper that:
- prefers `._path` when available and existing
- otherwise, safely copies small uploads to a temp file and returns that `Path`
- raises a clear error if neither disk-path nor bytes are accessible

### 3) Multi-file support with “once per drop” semantics
NiceGUI fires **one upload event per file**.

We added “batching” so the widget can call the user callback **once per drop** rather than once per file.
The idea is:

- accumulate incoming events into a staging list
- finalize a “batch” when no new events arrive for a short debounce window
- call `on_paths_ready(List[Path], cancel_token)` once with the finalized list

This avoids partial UI updates and keeps the caller logic simpler.

### 4) Progress + cancellation API
We support:

- `on_progress(progress: float, message: str|None)` for UI updates
- `CancelToken` passed to `on_paths_ready` so caller can abandon work if the user clicks cancel

Notes:
- cancellation can’t stop a file already uploaded by the browser; it is intended to stop *post-upload processing*
  (parsing, validation, reading TIFF, etc.)

### 5) Safer logging
We avoid dumping upload file bytes into logs.
Instead we log:
- upload type (small/large)
- filename
- size/metadata if available
- resolved path

---

## What to expect from the demos (current behavior)

### `demo_upload_multi.py`
- user drops multiple files → widget groups them into **one batch**
- demo prints the received file count and paths once batch finalizes

### `demo_upload_tif.py`
- user uploads one or more TIFFs
- demo reads them with `tifffile` using `run.io_bound(...)`
- demo updates a small GUI table with shape/dtype (and prints to console)

---

## Known limitations / edge cases (current)

1. **“Folder uploads” are not a first-class browser feature.**
   Browsers do not reliably allow dragging an actual folder unless the input uses `webkitdirectory` / directory picker
   (not universally supported). In practice:
   - users can still drag multiple files from a folder (works well)
   - true folder upload support will need explicit UI and browser-specific features

2. **Temp-file location and lifecycle**
   - Large uploads: temp file location is managed by NiceGUI
   - Small uploads: we copy bytes to a temp file (owned by our widget)
   Next step: add a cleanup strategy (delete on shutdown, or after processing completes).

3. **“Once per drop” batching is heuristic**
   The debounce approach is pragmatic and works well, but it’s still a heuristic.
   We can improve it by:
   - tracking `client` + timing + file counts, or
   - supporting an explicit “Done” button mode if a workflow needs strict confirmation

---

## Next steps (planned)

### Phase A — Hardening & correctness
- Add unit tests around:
  - normalization (large vs small)
  - batch finalize behavior (debounce)
  - cancellation token behavior
- Improve “debug event” logging formatting (always avoid byte dumps).
- Decide temp-file cleanup policy and implement it (opt-in first).

### Phase B — Better multi-file workflows
- Add optional constraints:
  - allowed extensions
  - required “pairing” rules (e.g. expect .tif + .txt per dataset)
  - max total files / max total size (guardrails)
- Improve `pairing.py` (keep it optional; don’t force opinions into the base widget).

### Phase C — UX polish
- “Drop zone overlay”: a visual hint area that highlights when user drags files over it.
  (This is mostly CSS + NiceGUI/Quasar slot/layout tricks.)
- Add a visible queue list (file names) before processing begins.

### Phase D — Cloud-ready example
- Provide a demo that stores uploads into an app-specific directory and exposes a “download processed results” link.

### Phase E — Documentation
- Expand `readme_upload_widget.md`:
  - integration recipes
  - common pitfalls (small vs large uploads)
  - recommended patterns (io-bound parsing, avoid blocking UI thread)

---

## Integration guidance (kymflow)

When wiring into kymflow:

- keep the widget as a pure UI component that yields `List[Path]`
- do parsing/processing in kymflow code, ideally using `run.io_bound(...)` for file IO and decoding
- for long tasks, tie `on_progress` to a UI progress bar/spinner
- keep cancellation best-effort: stop future steps, don’t attempt to “cancel” an in-flight HTTP upload

---

## Where we are now

We are **midway** through the “multi-file + batching + normalize” milestone:
- the core architecture is in place
- the next most valuable work is **hardening** (tests, cleanup, edge cases) and **pairing/workflow helpers**

