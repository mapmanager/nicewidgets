# `image_toolbar_widget`

NiceGUI toolbar: file label (no file picker), `channel` and `roi` `ui.select` controls, and ROI actions (add, delete, edit, full width/height, OK, cancel).

Import from submodules, for example:

```python
from nicewidgets.image_toolbar_widget.image_toolbar_widget import ImageToolbarWidget
from nicewidgets.image_toolbar_widget.intent import ImageToolbarIntent
```

## Intents

User actions invoke a single callback:

- `on_intent: Callable[[ImageToolbarIntent], None] | None`

`ImageToolbarIntent` is a union of frozen dataclasses in `nicewidgets.image_toolbar_widget.intent` (selection: channel / ROI; ROI CRUD / edit lifecycle; no file intents). **ROI ids are `int`**, matching backend ROI identifiers.

If `on_intent` is `None`, ROI action buttons are disabled; selection intents are still emitted when the toolbar is enabled.

## Programmatic API (`*_ext`)

These methods update the UI and internal state **without** calling `on_intent`.

### `set_file_ext(file_id, channel, roi_id, *, channel_options, roi_options)`

- Always requires **both** `channel_options` and `roi_options`.
- **Channel:** empty `channel_options` ⇒ `channel` must be `None`; non-empty ⇒ `channel` must be non-`None` and `str(channel)` must appear in `channel_options`.
- **ROI:** empty `roi_options` ⇒ `roi_id` must be `None`; non-empty ⇒ **`roi_id` must be non-`None`** and must appear in `roi_options` (as `int`).
- Resets ROI edit mode to idle.

### `set_roi_options_and_selection_ext(roi_options, roi_id)`

Same empty / non-empty ROI rules as above, applied atomically.

### `set_channel_options_ext(options)` / `set_roi_options_ext(options)`

Replace options for one axis. Raises `ValueError` if the **current** selection is set and is **not** in the new list. If the current selection is `None`, the update is allowed (follow with `set_channel_ext` / `set_roi_ext` if needed).

### `set_channel_ext(channel)` / `set_roi_ext(roi_id)`

Raises if the value is inconsistent with the **current** option list (empty list ⇒ value must be `None`; non-empty ⇒ value required and must be in the list).

### `set_enabled_ext(enabled: bool)`

Disables interaction (no intents); programmatic `*_ext` methods still run.

## Channel vs ROI options

- **Channel** options are `list[str]` in the select (e.g. `"0"`, `"1"`); programmatic `set_channel_ext` uses `int | None` and matches via `str(channel)`.
- **ROI** options are **`list[int]`** in the API and in the NiceGUI select.

See also: `docs-dev/packages/README.md` for other packages and doc entry points.
