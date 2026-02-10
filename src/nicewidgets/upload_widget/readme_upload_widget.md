# nicewidgets.upload_widget

> **Path:** `nicewidgets/src/nicewidgets/upload_widget/readme_upload_widget.md`

A self-contained **NiceGUI** upload widget that normalizes uploads into `pathlib.Path` objects.

## Current behavior (v2)

- **Batch semantics (recommended):** one callback per *drop/selection*
  - `ui.upload(on_upload=...)` is used to collect individual files.
  - `ui.upload(on_multi_upload=...)` triggers a single `on_paths_ready(paths)` call when the batch completes.

## API

```python
from nicewidgets.upload_widget import UploadWidget, CancelToken
```

### UploadWidget
- `on_paths_ready(paths: list[Path], cancel: CancelToken)` called **once per batch**
- Optional `on_progress(progress: float, message: str|None)`

### CancelToken
- Cooperative cancellation for *post-upload processing* (does not stop the browser upload)

## Normalization

`normalize_uploaded_file()` supports multiple NiceGUI 3.x file shapes:
- `LargeFileUpload._path` (already written to disk)
- `SmallFileUpload.save(path)` (async)
- `SmallFileUpload.read()` (async)
- `SmallFileUpload._data` (bytes)

## Tutorials

- `tutorials/demo_upload_multi.py` (multi-file batch callback)
- `tutorials/demo_upload_tif.py` (optional; requires `tifffile`)


## Temp files and cleanup

When an upload is kept in memory by NiceGUI (e.g. `SmallFileUpload`), `normalize_uploaded_file()`
will write the bytes to a temporary file and return a `Path` to that file.

**Cleanup policy:** the `UploadWidget` does **not** delete temporary files automatically.
The **caller owns cleanup** of any returned temp paths.

## Fallback batching (optional)

`UploadWidget` uses `on_multi_upload` as the authoritative "end-of-batch" signal when it is
available (recommended).

If you run in an environment where only per-file upload events are observed, you can enable an
optional debounce fallback via:

- `fallback_batch_debounce_sec: float | None`

When set (e.g. `0.5`), the widget will treat "no new files received for N seconds" as the end
of the batch and will call `on_paths_ready(...)` once.
